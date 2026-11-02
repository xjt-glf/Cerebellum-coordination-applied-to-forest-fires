# -*- coding: utf-8 -*-
"""任务条件低层环境包装器（集中式规划 + 多火点搜寻场景）。

对外接口（gym 风格，多智能体同步）：
    obs = env.reset(plan=None)                   -> np.ndarray [N, OBS_DIM]
    obs, rew, done, info = env.step(actions)     -> actions: [N] in {0..8}
    env.set_plan(TeamPlan | List[UavOrder])      -> 中途整队换指令（高层重规划）

策略输出的就是每架无人机的 9 个离散动作，指令只是输入条件。
接入大模型后只需把 set_plan() 的调用方从课程采样器换成
"LLM Task Planner -> ReAd critic 打分通过后下发"。
"""
import importlib
import math
import os
import random
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if os.path.join(_ROOT, "Env") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "Env"))

_env_mod = importlib.import_module("低层环境")
GridEnv = _env_mod.GridEnv

from .task_spec import (
    N_TASKS, UavOrder, TeamPlan,
    TASK_FIRE_SENSING, TASK_COMM_RELAY, TASK_RECHARGE,
    SENSE_EXPLORE, SENSE_PATROL,
)
from .obs_builder import (
    build_obs, shared_cache, assigned_station, OBS_DIM, OBS_SLICES, team_map,
)
from .reward import LowLevelReward, RewardConfig

N_ACTIONS = 9


class TaskConditionedEnv:
    GLOBAL_DIM = 8

    def __init__(self, num_uavs=4, max_steps=600, spread_steps=40, grid_size=(100, 100),
                 n_fire_seeds=3, n_fire_seeds_range=(2, 4), randomize_wind=True,
                 spread_gain=1.0, num_rescuers=8,
                 reward_cfg=None, seed=None, task_sampler="planner",
                 replan_every=25, terminate_on_all_depleted=True,
                 cwd_for_terrain=None, env_kwargs=None):
        self.num_uavs = int(num_uavs)
        self.max_steps = int(max_steps)
        self.spread_steps = int(spread_steps)
        self.grid_size = tuple(grid_size)
        self.n_fire_seeds = int(n_fire_seeds)
        self.n_fire_seeds_range = n_fire_seeds_range
        self.randomize_wind = bool(randomize_wind)
        self.spread_gain = float(spread_gain)
        self.num_rescuers = int(num_rescuers)
        self.task_sampler = task_sampler
        self.replan_every = int(replan_every)
        self.terminate_on_all_depleted = terminate_on_all_depleted
        self._cwd = cwd_for_terrain or _ROOT
        self._env_kwargs = dict(env_kwargs or {})

        self.obs_dim = OBS_DIM
        self.n_actions = N_ACTIONS
        self.rew = LowLevelReward(reward_cfg or RewardConfig())

        self._rng = random.Random(seed)
        self._nprng = np.random.RandomState(seed if seed is not None else 0)

        self.env = None
        self.orders = [UavOrder() for _ in range(self.num_uavs)]
        self.t = 0

    # ------------------------------------------------------------------ 内部
    def _make_env(self):
        cwd0 = os.getcwd()
        # GridEnv 内部的随机性（可燃度图、火点位置、风向、无人机初始位置）全部走
        # **全局** np.random。多进程采样是 fork 出来的，各 worker 的全局随机状态完全相同，
        # 而 TaskConditionedEnv 的 seed 只喂给了指令采样器——结果是 96 个并行环境
        # 在跑**同一个**回合序列（实测 4 个 worker 的风向/火点数/火点坐标一模一样）。
        # 这里用本环境自己的 RNG 给全局流播种，既拿回多样性，又保持按 seed 可复现。
        np.random.seed(int(self._nprng.randint(0, 2 ** 31 - 1)))
        try:
            os.chdir(self._cwd)          # terrain_height.csv 是相对路径读取的
            env = GridEnv(grid_size=self.grid_size, n_fire_seeds=self.n_fire_seeds,
                          n_fire_seeds_range=self.n_fire_seeds_range,
                          randomize_wind=self.randomize_wind,
                          spread_gain=self.spread_gain, spread_steps=self.spread_steps,
                          num_uavs=self.num_uavs, num_rescuers=self.num_rescuers,
                          **self._env_kwargs)
            env.reset()
        finally:
            os.chdir(cwd0)
        return env

    # ------------------------------------------------------- 课程：模拟高层规划
    def _sample_plan(self):
        """训练期的指令采样。真实部署时由 LLM 一次性给出全队指令，这里只是课程。

        'mix'       : 每机独立均匀采样任务，目标一律留空（策略自己就近决定）
        'mix+single': 30% 全队同任务，其余同 mix
        'planner'   : **按当前态势模拟一个讲道理的规划器**——未探索完就多派搜寻，
                      探索完就转巡逻，始终留人做中继，返航余量不足的换去补给，
                      并给出显式目标（未知区域中心 / 火点质心 + 环半径 / 指定人员）。
                      这一档最接近部署形态，也是正式训练用的。
        """
        n = self.num_uavs
        if self.task_sampler == "planner":
            return self._planner_orders()
        if self.task_sampler == "mix+single" and self._rng.random() < 0.3:
            t = self._rng.randrange(N_TASKS)
            tasks = [t] * n
        else:
            tasks = [self._rng.randrange(N_TASKS) for _ in range(n)]
        orders = []
        for t in tasks:
            o = UavOrder(task=t)
            if t == TASK_FIRE_SENSING:
                o.mode = self._rng.choice([SENSE_EXPLORE, SENSE_PATROL])
                o.ring_radius = None      # 自动按火点规模取
            o.proactive_handover = bool(self._rng.random() < 0.5)
            orders.append(o)
        return orders

    def _unknown_region_center(self):
        """随机挑一片未探索区域的中心，作为搜寻目标。

        在 10x10 的粗网格上按"未探索比例"加权采样，避免总是指向同一个角落。
        """
        M = team_map(self.env)
        H, W = M.shape
        C = 10
        bh, bw = H // C, W // C
        unknown = (M < 0.0)[:bh * C, :bw * C].reshape(C, bh, C, bw).mean(axis=(1, 3))
        w = unknown.ravel()
        if w.sum() <= 1e-6:
            return None
        k = int(self._nprng.choice(len(w), p=w / w.sum()))
        bx, by = divmod(k, C)
        return (float(bx * bh + bh / 2.0), float(by * bw + bw / 2.0))

    def _planner_orders(self):
        env = self.env
        c = shared_cache(env)
        n = self.num_uavs
        explored = c["explored_rate"]
        clusters = c["clusters"]
        rescuers = getattr(env, "rescuers", [])
        cost = env.energy_cost_per_step(0) if hasattr(env, "energy_cost_per_step") else 2.5

        # 1) 返航余量不足的先派去补给（余量留 20% 冗余）
        need_charge = []
        for i, u in enumerate(env.uavs):
            _st, d = assigned_station(env, i)
            e_max = float(getattr(u, "e_max", getattr(env, "uav_energy", 600.0))) or 600.0
            if float(u.e) - cost * d < 0.20 * e_max:
                need_charge.append(i)

        # 2) 至少留 1 台中继（人员离开基站覆盖后就必须有人接力）
        n_relay = 1 if n <= 2 else min(2, max(1, n - len(need_charge) - 1))
        rest = [i for i in range(n) if i not in need_charge]
        relay = rest[:n_relay]
        sensing = rest[n_relay:]

        orders = [None] * n
        for i in need_charge:
            orders[i] = UavOrder(task=TASK_RECHARGE)
        uncovered = [r for r in rescuers if not r.is_covered]
        for k, i in enumerate(relay):
            o = UavOrder(task=TASK_COMM_RELAY, proactive_handover=True)
            pool = uncovered or rescuers
            if pool:
                o.rescuer_id = int(getattr(pool[k % len(pool)], "rescuer_id", 1))
            orders[i] = o
        for k, i in enumerate(sensing):
            # 探索率不到 90% 就继续铺开未知区域；铺开之后转为绕火点巡逻侦测
            if explored < 0.90 or not clusters:
                tgt = self._unknown_region_center()
                orders[i] = UavOrder(task=TASK_FIRE_SENSING, mode=SENSE_EXPLORE, target=tgt)
            else:
                cl = clusters[k % len(clusters)]
                orders[i] = UavOrder(task=TASK_FIRE_SENSING, mode=SENSE_PATROL,
                                     target=cl["centroid"], ring_radius=None)
        return [o if o is not None else UavOrder() for o in orders]

    # ------------------------------------------------------------------ 对外
    def _sync_options(self):
        for i, u in enumerate(self.env.uavs):
            u.option = int(self.orders[i].task)

    def _obs(self):
        c = shared_cache(self.env)
        return np.stack([
            build_obs(self.env, i, self.orders[i], step_t=self.t,
                      max_steps=self.max_steps, cache=c)
            for i in range(self.num_uavs)
        ]).astype(np.float32)

    def set_plan(self, plan):
        """整队换指令。plan 可以是 TeamPlan、List[UavOrder]，或 LLM 的原始 JSON。"""
        if isinstance(plan, TeamPlan):
            plan.validate(self.num_uavs)
            orders = plan.orders
        elif isinstance(plan, (list, tuple)) and all(isinstance(o, UavOrder) for o in plan):
            tp = TeamPlan(list(plan)); tp.validate(self.num_uavs)
            orders = tp.orders
        else:
            orders = TeamPlan.from_llm_json(plan, self.num_uavs).orders
        changed = [i for i in range(self.num_uavs)
                   if int(orders[i].task) != int(self.orders[i].task)
                   or orders[i].target != self.orders[i].target
                   or int(orders[i].mode) != int(self.orders[i].mode)]
        self.orders = orders
        self._sync_options()
        for i in changed:
            self.rew.clear_shape(i)   # 换目标后重置势函数，避免虚假 shaping
        return self._obs()

    # 兼容旧接口名
    def set_task(self, plan):
        return self.set_plan(plan)

    def reset(self, plan=None):
        self.env = self._make_env()
        self.t = 0
        self.env._obs_cache = None
        self.env.fuse_team_maps_partial()
        self.env.update_perception_metrics()
        self.orders = [UavOrder() for _ in range(self.num_uavs)]
        self.orders = list(plan) if plan is not None else self._sample_plan()
        if not all(isinstance(o, UavOrder) for o in self.orders):
            self.orders = TeamPlan.from_llm_json(self.orders, self.num_uavs).orders
        self._sync_options()
        self.rew.reset(self.env)
        return self._obs()

    def step(self, actions):
        acts = [int(a) % N_ACTIONS for a in actions]
        # 电量耗尽的无人机被迫悬停：让"能量管理"具备真实后果，否则该技能不可学
        for i, u in enumerate(self.env.uavs):
            if float(u.e) <= 0.0:
                acts[i] = 8
        self.rew.snapshot(self.env)
        cwd0 = os.getcwd()
        try:
            os.chdir(self._cwd)
            self.env.step(acts)          # 丢弃环境的粗奖励，只借用其物理演化
        finally:
            os.chdir(cwd0)
        self.t += 1
        self.env._obs_cache = None       # 状态已变，重建按步缓存

        rewards, info = self.rew.compute(self.env, self.orders, max_steps=self.max_steps)
        # 高层按 replan_every 重规划（部署时对应 LLM 的一次重新下发）
        if self.replan_every > 0 and self.t % self.replan_every == 0:
            self.orders = self._sample_plan()
            self._sync_options()

        done = self.t >= self.max_steps
        if self.terminate_on_all_depleted and info["n_depleted"] >= self.num_uavs:
            done = True
        info["t"] = self.t
        info["tasks"] = [int(o.task) for o in self.orders]
        info["modes"] = [int(o.mode) for o in self.orders]
        return self._obs(), np.asarray(rewards, dtype=np.float32), bool(done), info

    def global_features(self):
        """给中心化 critic 的全局附加特征（与智能体数量无关）。"""
        env = self.env
        e_max = float(getattr(env, "uav_energy", 600.0)) or 600.0
        return np.array([
            float(getattr(env, "explored_rate", 0.0)),
            float(getattr(env, "fire_discovery_rate", 0.0)),
            float(getattr(env, "burned_area_rate", 0.0)),
            float(getattr(env, "rescuer_known_fire_rate", 0.0)),
            float(getattr(env, "coverage_rate", 0.0)),
            min(float(getattr(env, "cur_max_AoI", 0)) / 50.0, 1.0),
            float(min(float(u.e) for u in env.uavs)) / e_max,
            float(self.t) / max(1, self.max_steps),
        ], dtype=np.float32)

    @property
    def taus(self):
        return [int(o.task) for o in self.orders]

    def close(self):
        self.env = None

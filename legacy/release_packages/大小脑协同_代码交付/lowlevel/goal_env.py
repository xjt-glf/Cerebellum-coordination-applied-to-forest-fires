# -*- coding: utf-8 -*-
"""目标条件训练环境（H200 计划 §3.3：训练包装器层，不改动物理环境）。

在 TaskConditionedEnv 之上替换两件事：
1. 指令采样：TaskSampler（角色随机化 + 合法目标 + 距离课程 + 目标保持）；
2. 奖励：RewardV2（目标条件奖励）或原版 LowLevelReward（对照组）。

课程阶段由累计环境步（steps_offset + 本进程步数）自动推进，不需要主进程 IPC。
观测维度、动作空间、物理演化、global_features 全部与原环境一致。
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lowlevel.env_wrapper import TaskConditionedEnv
from lowlevel.reward import LowLevelReward, RewardConfig
from lowlevel.task_spec import UavOrder, TeamPlan
from .goal_manager import GoalManager
from .reward_v2 import RewardV2, RewardV2Config
from .sampler import TaskSampler


class GoalEnv(TaskConditionedEnv):
    """训练用环境。reward_version: 'P1' | 'P2' | 'original'。"""

    def __init__(self, *, reward_version="P1", role_mode="randomized",
                 stage="S1", stage_steps=None, steps_offset=0,
                 goal_cfg=None, sampler_cfg=None,
                 num_uavs=4, max_steps=600, spread_steps=40, grid_size=(100, 100),
                 n_fire_seeds=3, n_fire_seeds_range=(2, 4), randomize_wind=True,
                 spread_gain=1.0, num_rescuers=10, seed=None, replan_every=20,
                 terminate_on_all_depleted=True, env_kwargs=None):
        super().__init__(
            num_uavs=num_uavs, max_steps=max_steps, spread_steps=spread_steps,
            grid_size=grid_size, n_fire_seeds=n_fire_seeds,
            n_fire_seeds_range=n_fire_seeds_range, randomize_wind=randomize_wind,
            spread_gain=spread_gain, num_rescuers=num_rescuers, seed=seed,
            task_sampler="goal", replan_every=replan_every,
            terminate_on_all_depleted=terminate_on_all_depleted, env_kwargs=env_kwargs)
        gc = dict(goal_cfg or {})
        self.goal_mgr = GoalManager(grid_size=grid_size,
                                    explore_half_width=gc.get("explore_half_width", 5),
                                    completion_frac=gc.get("completion_fraction", 0.8))
        sc = dict(sampler_cfg or {})
        sc.setdefault("role_mode", role_mode)
        self.sampler = TaskSampler(**sc)
        self.stage = stage
        self.stage_steps = dict(stage_steps or {})
        self.steps_offset = int(steps_offset)
        self._total_steps = 0
        self.reward_version = reward_version
        if reward_version == "original":
            self.rew = LowLevelReward(RewardConfig())
        else:
            cfgv2 = RewardV2Config(base=reward_version, gamma=RewardConfig().gamma)
            self.rew = RewardV2(cfgv2, goal_manager=self.goal_mgr)

    # ------------------------------------------------------------ 课程阶段
    def cur_stage(self):
        if not self.stage_steps:
            return self.stage
        t = self.steps_offset + self._total_steps
        if t < float(self.stage_steps.get("S1", 2e6)):
            return "S1"
        if t < float(self.stage_steps.get("S2", 5e6)):
            return "S2"
        return "S3"

    # ------------------------------------------------------------ 采样与目标
    def _sample_plan(self):
        return self.sampler.sample_plan(self, self.orders, self.goal_mgr, self.t,
                                        self.cur_stage())

    def reset(self, plan=None):
        self.env = self._make_env()
        self.t = 0
        self.env._obs_cache = None
        self.env.fuse_team_maps_partial()
        self.env.update_perception_metrics()
        if plan is not None:
            self.orders = list(plan)
        else:
            self.orders = self.sampler.sample_plan(self, None, self.goal_mgr, 0,
                                                   self.cur_stage())
        if not all(isinstance(o, UavOrder) for o in self.orders):
            self.orders = TeamPlan.from_llm_json(self.orders, self.num_uavs).orders
        self._sync_options()
        self.goal_mgr.reset()
        self.env._goal_t = 0
        self.goal_mgr.sync(self.env, self.orders, 0)
        self.rew.reset(self.env)
        return self._obs()

    def set_plan(self, plan):
        obs = super().set_plan(plan)
        self.env._goal_t = self.t
        self.goal_mgr.sync(self.env, self.orders, self.t)
        return obs

    # ------------------------------------------------------------ 步进
    def step(self, actions):
        acts = [int(a) % 9 for a in actions]
        for i, u in enumerate(self.env.uavs):
            if float(u.e) <= 0.0:
                acts[i] = 8
        self.rew.snapshot(self.env)
        cwd0 = os.getcwd()
        try:
            os.chdir(self._cwd)
            self.env.step(acts)
        finally:
            os.chdir(cwd0)
        self.t += 1
        self._total_steps += 1
        self.env._obs_cache = None
        self.env._goal_t = self.t
        # 目标事件更新（训练与评估共用同一判定）
        self.goal_mgr.update(self.env, self.orders, self.t)
        rewards, info = self.rew.compute(self.env, self.orders, max_steps=self.max_steps)
        # 20 步检查与必要重规划（目标保持由 sampler 内部处理）
        if self.replan_every > 0 and self.t % self.replan_every == 0:
            self.orders = self._sample_plan()
            self._sync_options()
            self.goal_mgr.sync(self.env, self.orders, self.t)

        done = self.t >= self.max_steps
        if self.terminate_on_all_depleted and info["n_depleted"] >= self.num_uavs:
            done = True
        info["t"] = self.t
        info["tasks"] = [int(o.task) for o in self.orders]
        info["modes"] = [int(o.mode) for o in self.orders]
        info["goal_summary"] = self.goal_mgr.goal_info()
        return self._obs(), np.asarray(rewards, dtype=np.float32), bool(done), info

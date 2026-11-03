# -*- coding: utf-8 -*-
"""部署接口：给定 (环境, 全队指令) -> 每架无人机 9 个离散动作之一。

这是"底层模型"对外的唯一入口，也是接入 LLM Task Planner + ReAd critic 的契合点。
接大模型时只需把 `set_plan()` 的入参来源换成
    LLM 生成 JSON -> ReAd critic 打分 S_ReAd -> S > alpha 才下发
其余完全不变。

用法：
    from lowlevel.policy_api import LowLevelController
    ctl = LowLevelController("lowlevel/runs/w1_ent02/actor_best.pt")
    ctl.set_plan({"uavs": [{"task": "patrol", "target": [36, 48], "ring_radius": "auto"},
                           {"task": "relay", "rescuer_id": 3},
                           {"task": "charge"},
                           {"task": "explore", "target": [80, 20]}]})
    for t in range(300):
        actions = ctl.act(env)          # env 是原始 GridEnv
        env.step(actions)
"""
import numpy as np
import torch

from .models import Actor
from .obs_builder import build_obs, shared_cache, OBS_DIM
from .task_spec import UavOrder, TeamPlan, N_TASKS, TASK_FIRE_SENSING

N_ACTIONS = 9


class LowLevelController:
    def __init__(self, actor_path=None, device="cpu", obs_dim=OBS_DIM,
                 n_actions=N_ACTIONS, hidden=None, n_layers=2, max_steps=600):
        self.device = torch.device(device)
        sd = None
        if actor_path:
            sd = torch.load(actor_path, map_location=self.device)
            if isinstance(sd, dict) and "actor" in sd:
                sd = sd["actor"]
            if hidden is None:
                hidden = int(sd["head.weight"].shape[1])
        self.actor = Actor(obs_dim, n_actions, hidden or 256, n_layers).to(self.device)
        if sd is not None:
            self.actor.load_state_dict(sd)
        self.actor.eval()
        self.max_steps = max_steps
        self.orders = None
        self.t = 0
        self.violations = 0

    # -------------------------------------------------------------- 指令下发
    def set_plan(self, plan, num_uavs=None):
        """由高层调用。plan 可以是 TeamPlan、List[UavOrder]，或 LLM 的原始 JSON。

        非法/缺失字段一律回落默认值并累加 `violations`，不抛异常——与
        `TeamPlan.from_llm_json` 的 AugParse 容错策略一致。
        """
        if isinstance(plan, TeamPlan):
            tp = plan
        elif isinstance(plan, (list, tuple)) and all(isinstance(o, UavOrder) for o in plan):
            tp = TeamPlan(list(plan))
        else:
            tp = TeamPlan.from_llm_json(plan)
        if num_uavs is not None:
            self.violations += tp.validate(num_uavs)
        self.orders = tp.orders
        return self.orders

    # 兼容旧接口名
    def set_task(self, plan, num_uavs=None):
        return self.set_plan(plan, num_uavs)

    def reset(self):
        self.t = 0

    def _ensure(self, n):
        if self.orders is None:
            self.orders = [UavOrder() for _ in range(n)]
        while len(self.orders) < n:
            self.orders.append(UavOrder())
        return self.orders[:n]

    def _obs(self, env):
        n = len(env.uavs)
        orders = self._ensure(n)
        env._obs_cache = None                # 环境状态已推进，重建按步缓存
        c = shared_cache(env)
        return np.stack([
            build_obs(env, i, orders[i], step_t=self.t, max_steps=self.max_steps, cache=c)
            for i in range(n)
        ]).astype(np.float32)

    # -------------------------------------------------------------- 动作输出
    @torch.no_grad()
    def act(self, env, deterministic=True):
        """返回 List[int]，长度 = 无人机数，每个元素 ∈ {0..8}。"""
        n = len(env.uavs)
        orders = self._ensure(n)
        o = torch.as_tensor(self._obs(env), device=self.device)
        a, _, _ = self.actor.act(o, deterministic=deterministic)
        self.t += 1

        acts = a.cpu().numpy().tolist()
        for i, u in enumerate(env.uavs):
            # 电量耗尽者强制悬停，与训练时的环境约束保持一致
            if float(u.e) <= 0.0:
                acts[i] = 8
            u.option = int(orders[i].task)
        return acts

    @torch.no_grad()
    def action_probs(self, env):
        """返回 [N, 9] 的动作概率，供 ReAd critic 或可视化使用。"""
        o = torch.as_tensor(self._obs(env), device=self.device)
        return torch.softmax(self.actor(o), dim=-1).cpu().numpy()

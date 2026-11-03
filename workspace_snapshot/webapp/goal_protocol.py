"""网页演示的目标完成/失效更新协议；只更新指令，不接管 Actor 动作。"""
import copy
import random

from experiments.lowlevel_training.goal_manager import GoalManager
from experiments.lowlevel_training.sampler import TaskSampler
from lowlevel.obs_builder import shared_cache
from lowlevel.task_spec import TeamPlan, UavOrder


class GoalProtocol:
    def __init__(self, env, seed):
        self.env = env
        self.manager = GoalManager(grid_size=env.env.grid_size)
        self.sampler = TaskSampler(role_mode="randomized")
        self.rng = random.Random(seed)
        self.current = copy.deepcopy(env.orders)
        self.requested = None
        self.event = None

    def apply(self, plan, source="heuristic", event=None):
        env, e = self.env, self.env.env
        requested = TeamPlan.from_llm_json(plan, env.num_uavs).orders
        e._goal_t = env.t
        self.manager.sync(e, self.current, env.t)
        orders = copy.deepcopy(requested)
        notes = []
        for i, order in enumerate(requested):
            goal = self.manager.goals.get(i)
            kind = self.manager.kind_of(order)
            # 大模型明确改变任务/目标或用户注入新事件时，立即接受新指令。
            explicit_change = source in {"llm", "codex_joint"} and (
                self.requested is None or event != self.event
                or order.to_dict() != self.requested[i].to_dict())
            if explicit_change:
                continue
            keep = goal is not None and goal.kind == kind and not goal.completed \
                and env.t <= goal.deadline_t
            if keep and kind == "patrol":
                keep = bool(shared_cache(e)["clusters"])
            if keep and kind == "explore" and goal.initial_unknown is not None:
                left = (goal.initial_unknown & (shared_cache(e)["M"] < 0)).sum()
                keep = not goal.arrived or left > .2 * max(1, goal.initial_unknown.sum())
            if keep:
                orders[i] = copy.deepcopy(self.current[i])
            elif kind == "explore":
                target, _ = self.sampler.sample_explore_target(
                    e, self.rng, float(e.uavs[i].x), float(e.uavs[i].y),
                    "S3", float(e.uavs[i].e))
                if target is not None:
                    orders[i] = UavOrder(task=0, mode=0, target=target)
                    notes.append(f"[目标管理] UAV{i} 更新探索目标为 {target}")
        self.requested = copy.deepcopy(requested)
        self.current = orders
        self.event = event
        applied = {"uavs": [order.to_dict() for order in orders]}
        obs = env.set_plan(orders)
        self.manager.sync(e, orders, env.t)
        return obs, applied, notes

    def update(self):
        e = self.env.env
        e._goal_t = self.env.t
        self.manager.update(e, self.current, self.env.t)

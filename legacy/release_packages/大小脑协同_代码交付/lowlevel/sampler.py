# -*- coding: utf-8 -*-
"""训练分布采样器（H200 计划 §4.3/§4.4）。

与旧 planner 采样器的核心差异（根因 2 的修复）：
- 角色**随机化**：焦点任务与支持角色都在能量可行的机队中随机挑人，
  不再按索引顺序把 relay 焊在 UAV0/1、sensing 焊在 UAV2/3；
- 目标合法：explore 只给未知比例 >= 0.6、能量可往返的区域中心；
  patrol 只给实际认知地图中已发现、未合围的火点，显式半径排除火场内部的小环；
- 20 步检查 + 目标保持：合法未完成目标保持到 完成/失效/能量紧急/超时；
- 距离课程：S1 以 8–25 为主、部分 25–50；S2/S3 8–90 混合/真实态势。

legacy_fixed 模式复刻旧 planner 的按索引固化角色（对照组 A_reward_s1501 用）。
"""
import math

from lowlevel.task_spec import (
    UavOrder, TASK_FIRE_SENSING, TASK_COMM_RELAY, TASK_RECHARGE,
    SENSE_EXPLORE, SENSE_PATROL,
)
from lowlevel.obs_builder import shared_cache, assigned_station, cluster_ring_radius
from .goal_manager import chebyshev_steps

# 焦点任务比例（explore/patrol/relay/charge）
FOCUS_PROBS = {
    "S1": (0.35, 0.35, 0.15, 0.15),
    "S2": (0.30, 0.30, 0.20, 0.20),
    "S3": (0.30, 0.30, 0.20, 0.20),
}

# explore 目标距离课程：(主区间, 次区间比例, 次区间)
DIST_CURRICULUM = {
    "S1": ((8.0, 25.0), 0.30, (25.0, 50.0)),
    "S2": ((8.0, 90.0), 0.0, (8.0, 90.0)),
    "S3": ((8.0, 90.0), 0.0, (8.0, 90.0)),
}

# 课程阶段切换的环境步门槛（由 train.py 传入，可配置）
STAGE_STEPS_DEFAULT = {"S1": 2_000_000, "S2": 5_000_000}


class TaskSampler:
    def __init__(self, role_mode="randomized", min_unknown_frac=0.6,
                 explicit_ring_prob=0.6, relay_point_prob=0.25,
                 relay_mixed_field_prob=0.10, energy_margin=0.10,
                 goal_keep_prob=0.95):
        self.role_mode = role_mode          # randomized | legacy_fixed
        self.min_unknown_frac = float(min_unknown_frac)
        self.explicit_ring_prob = float(explicit_ring_prob)
        self.relay_point_prob = float(relay_point_prob)
        self.relay_mixed_prob = float(relay_mixed_field_prob)
        self.energy_margin = float(energy_margin)
        self.goal_keep_prob = float(goal_keep_prob)

    # ------------------------------------------------------------ 目标生成
    @staticmethod
    def _region_unknown_frac(M, cx, cy, hw=5):
        H, W = M.shape
        x0, x1 = max(0, int(cx) - hw), min(H, int(cx) + hw + 1)
        y0, y1 = max(0, int(cy) - hw), min(W, int(cy) + hw + 1)
        reg = M[x0:x1, y0:y1]
        return float((reg < 0.0).mean()) if reg.size else 0.0

    def sample_explore_target(self, env, rng, ux, uy, stage, energy):
        """在距离课程内找合法未知区域中心。返回 ((x,y), unknown_frac) 或 (None, 0)。"""
        M = shared_cache(env)["M"]
        H, W = M.shape
        main, p_second, second = DIST_CURRICULUM[stage]
        lo, hi = main
        if rng.random() < p_second:
            lo, hi = second
        cost = env.energy_cost_per_step(0) if hasattr(env, "energy_cost_per_step") else 2.5
        best = None
        for _ in range(24):
            ang = rng.uniform(0, 2 * math.pi)
            d = rng.uniform(lo, hi)
            cx = int(min(max(6, ux + d * math.cos(ang)), H - 7))
            cy = int(min(max(6, uy + d * math.sin(ang)), W - 7))
            if math.hypot(cx - ux, cy - uy) < 6:
                continue
            # 能量可行：去 + 返
            if energy < cost * (2 * chebyshev_steps(ux, uy, cx, cy)) * (1 + self.energy_margin):
                continue
            frac = self._region_unknown_frac(M, cx, cy)
            if frac >= self.min_unknown_frac:
                return (float(cx), float(cy)), frac
            if best is None or frac > best[1]:
                best = ((float(cx), float(cy)), frac)
        # 放宽：地图快铺开时允许较低未知比例（S3 后期）
        if best is not None and best[1] >= 0.35:
            return best[0], best[1]
        return None, 0.0

    def sample_patrol_order(self, env, rng, cluster):
        """patrol 指令：指定火点 + 合法环半径（auto 或显式）。"""
        cl_r = cluster_ring_radius(cluster)
        if rng.random() < self.explicit_ring_prob:
            ring = rng.uniform(cl_r + 1.0, min(30.0, cl_r + 8.0))
            return UavOrder(task=TASK_FIRE_SENSING, mode=SENSE_PATROL,
                            target=(float(cluster["centroid"][0]),
                                    float(cluster["centroid"][1])),
                            ring_radius=float(round(ring, 1)))
        return UavOrder(task=TASK_FIRE_SENSING, mode=SENSE_PATROL,
                        target=(float(cluster["centroid"][0]),
                                float(cluster["centroid"][1])),
                        ring_radius=None)

    def sample_relay_order(self, env, rng):
        rescuers = list(getattr(env, "rescuers", []))
        if not rescuers:
            return UavOrder(task=TASK_COMM_RELAY, proactive_handover=True)
        r = rng.choice(rescuers)
        o = UavOrder(task=TASK_COMM_RELAY, proactive_handover=bool(rng.random() < 0.5))
        u = rng.random()
        if u < self.relay_point_prob:
            o.target = (float(r.x), float(r.y))          # 固定点接口
        elif u < self.relay_point_prob + self.relay_mixed_prob:
            # 混合字段：rescuer_id 优先（契约 v2），target 是干扰项
            o.rescuer_id = int(r.rescuer_id)
            other = rng.choice(rescuers)
            o.target = (float(other.x), float(other.y))
        else:
            o.rescuer_id = int(r.rescuer_id)
        return o

    # ------------------------------------------------------------ 主入口
    def sample_plan(self, wrapper, prev_orders, goal_mgr, t, stage):
        """生成全队指令。wrapper 是 TaskConditionedEnv/GoalEnv。"""
        rng = wrapper._rng
        env = wrapper.env
        cache = shared_cache(env)
        n = len(env.uavs)
        orders = [None] * n

        # 1) 能量紧急的强制补给（余量 < 20% e_max）
        forced = set()
        for i, u in enumerate(env.uavs):
            _st, d = assigned_station(env, i)
            e_max = float(getattr(u, "e_max", getattr(env, "uav_energy", 600.0))) or 600.0
            cost = env.energy_cost_per_step(i) if hasattr(env, "energy_cost_per_step") else 2.5
            if float(u.e) - cost * d < 0.20 * e_max:
                forced.add(i)
                orders[i] = UavOrder(task=TASK_RECHARGE)

        # 2) 目标保持：合法未完成目标保留到 完成/失效/能量紧急/超时
        assignable = []
        for i in range(n):
            if orders[i] is not None:
                continue
            g = goal_mgr.goals.get(i) if goal_mgr else None
            prev = prev_orders[i] if prev_orders is not None and i < len(prev_orders) else None
            keep = False
            if (g is not None and prev is not None and i not in forced
                    and not g.completed and t <= g.deadline_t
                    and rng.random() < self.goal_keep_prob):
                if g.kind == "patrol":
                    keep = True          # 连续任务；火点漂移由稳定 key 处理
                elif g.kind == "explore" and g.initial_unknown is not None:
                    left = float((g.initial_unknown & (cache["M"] < 0.0)).sum())
                    total = max(1, int(g.initial_unknown.sum()))
                    keep = (not g.arrived) or left > 0.2 * total
                elif g.kind == "relay":
                    keep = True
                elif g.kind == "charge":
                    keep = (not g.arrived) or float(env.uavs[i].e) < 0.95 * float(
                        getattr(env.uavs[i], "e_max", 600.0))
            if keep:
                orders[i] = prev
            else:
                assignable.append(i)

        if not assignable:
            return [o if o is not None else UavOrder() for o in orders]

        # 3) 角色分配（randomized：随机洗牌；legacy_fixed：保持索引序）
        pool = list(assignable)
        if self.role_mode == "randomized":
            rng.shuffle(pool)

        # 4) 焦点任务
        probs = FOCUS_PROBS[stage]
        kinds = ["explore", "patrol", "relay", "charge"]
        feasible = set(kinds)
        if not cache["clusters"]:
            feasible.discard("patrol")
        if not getattr(env, "rescuers", []):
            feasible.discard("relay")
        if pool:
            cand = [k for k in kinds if k in feasible]
            w = [probs[kinds.index(k)] for k in cand]
            focus = rng.choices(cand, weights=w, k=1)[0] if cand else "explore"

            if focus == "charge":
                a = min(pool, key=lambda i: float(env.uavs[i].e))
                orders[a] = UavOrder(task=TASK_RECHARGE)
            elif focus == "explore":
                a = pool[0]
                tgt, frac = self.sample_explore_target(
                    env, rng, float(env.uavs[a].x), float(env.uavs[a].y), stage,
                    float(env.uavs[a].e))
                if tgt is not None:
                    orders[a] = UavOrder(task=TASK_FIRE_SENSING, mode=SENSE_EXPLORE,
                                         target=tgt)
                elif cache["clusters"]:
                    orders[a] = self.sample_patrol_order(env, rng,
                                                         rng.choice(cache["clusters"]))
                else:
                    orders[a] = UavOrder(task=TASK_FIRE_SENSING, mode=SENSE_EXPLORE)
            elif focus == "patrol":
                a = pool[0]
                orders[a] = self.sample_patrol_order(env, rng, rng.choice(cache["clusters"]))
            elif focus == "relay":
                orders[pool[0]] = self.sample_relay_order(env, rng)

        # 5) 支持角色：1 台 relay（有人未覆盖时优先）、其余 explore/patrol
        rest = [i for i in pool if orders[i] is None]
        uncovered = [r for r in getattr(env, "rescuers", []) if not r.is_covered]
        explored = cache["explored_rate"]
        if rest and (uncovered or getattr(env, "rescuers", [])) and len(rest) >= 1:
            a = rest.pop(0)
            orders[a] = self.sample_relay_order(env, rng)
        for a in rest:
            if explored < 0.90 or not cache["clusters"]:
                tgt, frac = self.sample_explore_target(
                    env, rng, float(env.uavs[a].x), float(env.uavs[a].y), stage,
                    float(env.uavs[a].e))
                if tgt is not None:
                    orders[a] = UavOrder(task=TASK_FIRE_SENSING, mode=SENSE_EXPLORE,
                                         target=tgt)
                elif cache["clusters"]:
                    orders[a] = self.sample_patrol_order(env, rng,
                                                         rng.choice(cache["clusters"]))
                else:
                    orders[a] = UavOrder(task=TASK_FIRE_SENSING, mode=SENSE_EXPLORE)
            else:
                orders[a] = self.sample_patrol_order(env, rng, rng.choice(cache["clusters"]))
        return [o if o is not None else UavOrder() for o in orders]

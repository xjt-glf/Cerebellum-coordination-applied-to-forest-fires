# -*- coding: utf-8 -*-
"""目标生命周期管理器（H200 训练计划 §4）。

职责：
- 每架无人机的当前目标：goal_key、解析目标、到访/完成事件、有效期、首步记录；
- 事件按 goal_key 去重：同一计划重发不重置一次性奖励，目标失效/完成/明确新指令才换目标；
- patrol 的火点关联用稳定 key（质心漂移/半径更新不创建新目标）；
- explore 目标保存指令时刻的初始未知掩码，完成 = 本机有效观测覆盖初始未知的 80%。

与奖励（reward_v2）和评估（evaluate）共用同一份事件判定，保证训练口径 = 验收口径。
"""
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import numpy as np

from lowlevel.task_spec import (
    UavOrder, TASK_FIRE_SENSING, TASK_COMM_RELAY, TASK_RECHARGE,
    SENSE_EXPLORE, SENSE_PATROL,
)
from lowlevel.obs_builder import (
    resolve_order_target, shared_cache, assigned_station, cluster_ring_radius,
)

EXPLORE_HALF_WIDTH = 5          # 目标区域 11x11
EXPLORE_ARRIVAL_RADIUS = 5.0    # 到访半径（格）
PATROL_BAND_TOLERANCE = 3.0     # 环带容差
PATROL_SECTORS = 12
MIN_GOAL_TIMEOUT = 40
MAX_GOAL_TIMEOUT = 160


def chebyshev_steps(x0, y0, x1, y1):
    """8 方向 1 格/步的格点最短步数（无障碍时的真实下界）。"""
    return max(abs(float(x1) - float(x0)), abs(float(y1) - float(y0)))


def goal_timeout(d_steps, lo=MIN_GOAL_TIMEOUT, hi=MAX_GOAL_TIMEOUT):
    return int(min(hi, max(lo, 2 * int(d_steps) + 20)))


@dataclass
class Goal:
    agent: int
    kind: str                      # explore / patrol / relay / charge / free
    goal_key: tuple
    target: tuple                  # 解析目标（创建时刻；relay 人员会移动，按需重解析）
    ring: Optional[float] = None   # patrol 显式半径；None = auto
    created_t: int = 0
    deadline_t: int = 10 ** 9
    d_init: float = 0.0            # 指令下发生（动作前）的初始距离
    e_init: float = 0.0            # 指令下发生（动作前）的初始电量
    arrived: bool = False
    arrival_t: Optional[int] = None
    completed: bool = False
    completed_t: Optional[int] = None
    # explore
    region_mask: Optional[np.ndarray] = None    # [H,W] 目标区域（裁剪到地图）
    initial_unknown: Optional[np.ndarray] = None
    own_seen: Optional[np.ndarray] = None       # 本机足迹覆盖过的区域格
    # patrol
    sectors: Dict[int, int] = field(default_factory=dict)   # 扇区 -> 最近记功步
    # 逐步状态（供进度奖励）
    d_prev: float = 0.0
    err_prev: float = 0.0
    # 统计
    own_roi_new_cells: float = 0.0

    def resolved_target(self, env, order, cache):
        """每步重解析目标（relay 人员移动 / patrol auto 半径变化）。"""
        tgt, is_fire, ring = resolve_order_target(env, self.agent, order, cache)
        return tgt, is_fire, ring


class GoalManager:
    """每个环境实例一个；由 GoalEnv（训练）与 evaluate（验收）共同使用。"""

    def __init__(self, grid_size=(100, 100), explore_half_width=EXPLORE_HALF_WIDTH,
                 completion_frac=0.8, sector_recredit_steps=20,
                 patrol_cluster_match_dist=15.0):
        self.H, self.W = int(grid_size[0]), int(grid_size[1])
        self.hw = int(explore_half_width)
        self.completion_frac = float(completion_frac)
        self.sector_recredit = int(sector_recredit_steps)
        self.cluster_match_dist = float(patrol_cluster_match_dist)
        self.goals: Dict[int, Optional[Goal]] = {}
        self.last_events = {}

    # ------------------------------------------------------------ 生命周期
    def reset(self):
        self.goals = {}
        self.last_events = {}

    def kind_of(self, order):
        task, mode = int(order.task), int(order.mode)
        if task == TASK_FIRE_SENSING:
            return "explore" if mode == SENSE_EXPLORE else "patrol"
        if task == TASK_COMM_RELAY:
            return "relay"
        if task == TASK_RECHARGE:
            return "charge"
        return "free"

    def _region_mask(self, cx, cy):
        m = np.zeros((self.H, self.W), dtype=bool)
        x0 = max(0, int(round(cx)) - self.hw)
        x1 = min(self.H, int(round(cx)) + self.hw + 1)
        y0 = max(0, int(round(cy)) - self.hw)
        y1 = min(self.W, int(round(cy)) + self.hw + 1)
        m[x0:x1, y0:y1] = True
        return m

    def sync(self, env, orders, t):
        """指令变更（reset/set_plan/replan 检查）时同步目标。已有有效目标保持不变。"""
        cache = shared_cache(env)
        for i, order in enumerate(orders):
            tgt, is_fire, ring = resolve_order_target(env, i, order, cache)
            kind = self.kind_of(order)
            g = self.goals.get(i)
            keep = False
            if g is not None and g.kind == kind:
                if kind == "patrol":
                    # 火点关联稳定：目标火点仍在附近（质心漂移）且未合围 -> 同一目标
                    if tgt is not None and math.hypot(tgt[0] - g.target[0],
                                                     tgt[1] - g.target[1]) <= self.cluster_match_dist:
                        keep = True
                elif kind == "relay":
                    if order.rescuer_id is not None:
                        keep = (g.goal_key[1] == order.rescuer_id)
                    elif order.target is not None:
                        keep = (g.goal_key[0] == "relay_point"
                                and math.hypot(order.target[0] - g.target[0],
                                               order.target[1] - g.target[1]) < 1e-6)
                elif kind == "explore":
                    if order.target is not None and tgt is not None:
                        keep = (math.hypot(tgt[0] - g.goal_key[1],
                                           tgt[1] - g.goal_key[2]) < 1e-6)
                elif kind == "charge":
                    keep = True     # 站点分配基本稳定；具体站点变了再换
                    if tgt is not None and math.hypot(tgt[0] - g.target[0],
                                                      tgt[1] - g.target[1]) > 2.5:
                        keep = False
            if keep:
                # 显式半径变更（auto <-> 显式）更新 ring，但不重置事件
                if kind == "patrol":
                    g.ring = order.ring_radius
                g.d_prev = self._dist(env, i, g, order, cache)
                continue
            self.goals[i] = self._new_goal(env, i, order, kind, tgt, ring, t, cache)

    def _new_goal(self, env, i, order, kind, tgt, ring, t, cache):
        u = env.uavs[i]
        ux, uy = float(u.x), float(u.y)
        if tgt is None:
            key = (kind, round(ux, 1), round(uy, 1), t)
        elif kind == "explore":
            key = (kind, float(tgt[0]), float(tgt[1]))
        elif kind == "patrol":
            key = (kind, round(tgt[0] / 4.0), round(tgt[1] / 4.0))
        elif kind == "relay":
            key = (("relay_id", order.rescuer_id) if order.rescuer_id is not None
                   else ("relay_point", round(tgt[0], 1), round(tgt[1], 1)))
        else:
            key = (kind, round(tgt[0], 1), round(tgt[1], 1))
        d0 = math.hypot(tgt[0] - ux, tgt[1] - uy) if tgt is not None else 0.0
        g = Goal(agent=i, kind=kind, goal_key=key, target=tuple(tgt) if tgt else (ux, uy),
                 ring=(float(ring) if order.ring_radius is not None else None),
                 created_t=t, d_init=d0, e_init=float(u.e),
                 deadline_t=t + goal_timeout(chebyshev_steps(ux, uy, *tgt) if tgt else 0))
        g.d_prev = d0
        if kind == "explore" and tgt is not None:
            rm = self._region_mask(tgt[0], tgt[1])
            M = cache["M"]
            g.region_mask = rm
            g.initial_unknown = (rm & (M < 0.0))
            g.own_seen = np.zeros_like(rm)
        if kind == "patrol":
            g.err_prev = abs(d0 - float(ring if ring is not None else 6.0))
        return g

    def _dist(self, env, i, g, order, cache):
        tgt, _, _ = g.resolved_target(env, order, cache)
        if tgt is None:
            return 0.0
        u = env.uavs[i]
        return math.hypot(tgt[0] - float(u.x), tgt[1] - float(u.y))

    # ------------------------------------------------------------ 逐步更新
    def update(self, env, orders, t, newly_known=None, foot=None):
        """env.step 之后调用：更新事件（到访/完成/扇区）与逐步距离。返回每机事件字典。"""
        cache = shared_cache(env)
        events = {}
        for i, order in enumerate(orders):
            g = self.goals.get(i)
            if g is None:
                self.sync(env, orders, t)
                g = self.goals.get(i)
                if g is None:
                    continue
            ev = dict(arrived=False, completed=False, new_sector=False, in_band=False)
            tgt, is_fire, ring = g.resolved_target(env, order, cache)
            u = env.uavs[i]
            ux, uy = float(u.x), float(u.y)
            d_now = math.hypot(tgt[0] - ux, tgt[1] - uy) if tgt is not None else 0.0
            ev["d_now"] = d_now
            ev["d_prev"] = g.d_prev
            ev["progress"] = g.d_prev - d_now
            g.d_prev = d_now

            if g.kind == "explore":
                # 本机足迹对目标区域的有效观测（本机传感器覆盖即有效，不做队伍归因）
                if g.own_seen is not None:
                    xi, yi = int(round(ux)), int(round(uy))
                    r = int(getattr(env, "sense_radius", 4))
                    x0, x1 = max(0, xi - r), min(self.H, xi + r + 1)
                    y0, y1 = max(0, yi - r), min(self.W, yi + r + 1)
                    g.own_seen[x0:x1, y0:y1] |= g.region_mask[x0:x1, y0:y1]
                    n_unknown = max(1, int(g.initial_unknown.sum()))
                    ev["own_region_frac"] = float(
                        (g.own_seen & g.initial_unknown).sum()) / n_unknown
                if (not g.arrived and d_now <= EXPLORE_ARRIVAL_RADIUS):
                    g.arrived = True
                    g.arrival_t = t
                    ev["arrived"] = True
                if (not g.completed and ev.get("own_region_frac", 0.0) >= self.completion_frac
                        and g.arrived):
                    g.completed = True
                    g.completed_t = t
                    ev["completed"] = True
                # 区域已全部已知（被任何人看光）且到访过 -> 也算完成（团队口径，仅记录）
                if g.initial_unknown is not None:
                    M = cache["M"]
                    ev["team_region_frac"] = float(
                        (g.initial_unknown & (M >= 0.0)).sum()) / max(
                        1, int(g.initial_unknown.sum()))

            elif g.kind == "patrol":
                ring_eff = float(ring if ring is not None else 6.0)
                err = abs(d_now - ring_eff)
                ev["ring_err"] = err
                ev["ring"] = ring_eff
                ev["ring_progress"] = g.err_prev - err
                g.err_prev = err
                ev["in_band"] = bool(err <= PATROL_BAND_TOLERANCE)
                if ev["in_band"] and not g.arrived:
                    g.arrived = True
                    g.arrival_t = t
                    ev["arrived"] = True
                if ev["in_band"] and tgt is not None:
                    ang = math.atan2(uy - tgt[1], ux - tgt[0])
                    sec = int((ang + math.pi) / (2 * math.pi) * PATROL_SECTORS) % PATROL_SECTORS
                    last = g.sectors.get(sec, -10 ** 9)
                    if t - last >= self.sector_recredit:
                        g.sectors[sec] = t
                        ev["new_sector"] = True
                        ev["sector"] = sec
                ev["n_sectors"] = len(g.sectors)

            elif g.kind == "relay":
                r_comm = float(getattr(env, "uav_comm_radius", 12))
                if order.rescuer_id is not None:
                    # 人员移动：目标重解析已在 resolved_target 完成
                    ev["own_link"] = bool(d_now <= r_comm)
                else:
                    ev["own_link"] = bool(d_now <= r_comm)
                if ev["own_link"] and not g.arrived:
                    g.arrived = True
                    g.arrival_t = t
                    ev["arrived"] = True

            elif g.kind == "charge":
                on_st = bool(env.at_station(ux, uy)) if hasattr(env, "at_station") else False
                ev["on_station"] = on_st
                if on_st and not g.arrived:
                    g.arrived = True
                    g.arrival_t = t
                    ev["arrived"] = True

            ev["expired"] = bool(t > g.deadline_t)
            ev["g_arrived"] = g.arrived
            ev["g_completed"] = g.completed
            ev["goal"] = g
            events[i] = ev
        self.last_events = events
        return events

    def goal_info(self):
        """日志用摘要。"""
        out = {}
        for i, g in self.goals.items():
            if g is None:
                continue
            out[i] = dict(kind=g.kind, key=str(g.goal_key), arrived=g.arrived,
                          completed=g.completed, age_arrival=(None if g.arrival_t is None
                                                              else g.arrival_t - g.created_t))
        return out

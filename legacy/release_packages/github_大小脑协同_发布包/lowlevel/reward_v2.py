# -*- coding: utf-8 -*-
"""RewardV2：目标条件奖励（H200 计划 §5）。

与 V1（lowlevel/reward.py）的差异：
- explore 有显式目标时：区域**内**新格/新火格按本机归因给主体收入，区域外仅小乘数
  （P2 为 0）；目标距离进度项 + 首次到访/区域完成事件奖励（按 goal_key 去重）。
- patrol：环误差进度 + 合法环带保持 + 新角度扇区 + 目标火点邻域的有效刷新；
  去掉"自由探索新格"这个可替代目标的主要收入；环半径用每步实际解析值。
- relay/charge 初始沿用 V1 奖励（验收口径改为本机链路，见 evaluate.py）。
- 保留 V1 的全部通用安全项（时间/越界/阻挡/能量/耗尽/过火增长）。

reward.parts 完整记录各项真实贡献，供监控与奖励归因审计。
"""
import math

import numpy as np

from lowlevel.reward import LowLevelReward, RewardConfig
from lowlevel.task_spec import (
    TASK_FIRE_SENSING, TASK_COMM_RELAY, TASK_RECHARGE,
    SENSE_EXPLORE, SENSE_PATROL,
)
from lowlevel.obs_builder import assigned_station, resolve_order_target, shared_cache

SQRT2 = math.sqrt(2.0)


class RewardV2Config:
    """P1（平衡目标奖励）。P2 通过 base="P2" 覆盖强目标系数。"""

    def __init__(self, base="P1", gamma=0.99):
        self.gamma = gamma
        # ---- explore（有显式目标）----
        self.explore_inside_new_cell = 0.04
        self.explore_inside_new_fire = 0.40
        self.explore_outside_mult = 0.05          # P2: 0
        self.grid_progress_w = 0.05               # P2: 0.10
        self.arrival_bonus = 2.0                  # P2: 3
        self.completion_bonus = 3.0               # P2: 5
        # ---- patrol ----
        self.patrol_progress_w = 0.05             # P2: 0.10
        self.patrol_in_band = 0.02                # P2: 0.04
        self.patrol_new_sector = 0.15             # P2: 0.25
        self.patrol_refresh_w = 0.05
        self.patrol_refresh_margin = 6.0          # 火点邻域 = ring + margin
        # ---- 兼容 V1 通用项（从 RewardConfig 拷贝默认）----
        v1 = RewardConfig()
        for k in ("w_time", "w_oob", "w_blocked", "w_unsafe", "w_depleted",
                  "w_stranded", "w_burn_growth"):
            setattr(self, k, getattr(v1, k))
        if base == "P2":
            self.explore_outside_mult = 0.0
            self.grid_progress_w = 0.10
            self.arrival_bonus = 3.0
            self.completion_bonus = 5.0
            self.patrol_progress_w = 0.10
            self.patrol_in_band = 0.04
            self.patrol_new_sector = 0.25
        if base == "P1R":
            # P1 + 计划 §5.2 授权的两项定向修正（证据：P3@60min 验收缺口）：
            # 1) relay 指定人员本机链路奖励（relay_own_after 69.7% < 90% 门槛，V1 覆盖
            #    奖励只鼓励"覆盖多个人"，不鼓励"钉住指定的人"）；
            # 2) patrol 环带保持/扇区小幅增强（in-band 62% < 70%，sectors 2.5 < 6/12）。
            self.relay_own_link = 0.10
            self.patrol_in_band = 0.03
            self.patrol_new_sector = 0.20
        else:
            self.relay_own_link = 0.0
        self.base = base


class RewardV2(LowLevelReward):
    """接口与 LowLevelReward 相同（reset/snapshot/compute/clear_shape）。"""

    def __init__(self, cfg: RewardV2Config = None, goal_manager=None):
        super().__init__(RewardConfig())
        self.cfg = cfg or RewardV2Config()
        self.gm = goal_manager
        self.last_events = {}

    # ------------------------------------------------------------ 主入口
    def compute(self, env, orders, max_steps=600):
        cfg = self.cfg
        n = len(env.uavs)
        H, W = env.grid_size
        d_max = math.hypot(float(H), float(W))
        bs = env.grid["burning_state"]
        cache = shared_cache(env)
        M = cache["M"]
        t = int(getattr(env, "_goal_t", 0))

        # ---- 目标事件由 GoalEnv.step 先调用 gm.update 更新（训练与评估同口径）----
        events = (self.gm.last_events if self.gm is not None else {}) or {}

        # ---- 全队共享项：过火面积增量 ----
        burned_now = int((bs > 0).sum())
        burn_growth = max(0, burned_now - int(self._prev_burned))
        team_term = cfg.w_burn_growth * burn_growth

        # ---- 新探索格 / 新火格：按足迹归属，重叠均分（与 V1 相同的本机归因）----
        prev_team = self._prev_team if self._prev_team is not None else M
        newly_known = (prev_team < 0.0) & (M >= 0.0)
        newly_fire = newly_known & (M > 0) & (M < 2)
        foot = np.zeros((n, H, W), dtype=bool)
        for i, u in enumerate(env.uavs):
            xi, yi = int(round(float(u.x))), int(round(float(u.y)))
            r = int(getattr(env, "sense_radius", 4))
            foot[i, max(0, xi - r):min(H, xi + r + 1),
                 max(0, yi - r):min(W, yi + r + 1)] = True
        n_cover = foot.sum(axis=0).astype(np.float32)
        share = np.where(n_cover > 0, 1.0 / np.maximum(n_cover, 1.0), 0.0)

        # ---- 认知纠偏（patrol 用）----
        prev_known = (prev_team >= 0.0)
        stale_corrected = prev_known & (np.abs(prev_team - bs) > 0.05)
        is_fire_now = (bs > 0) & (bs < 2)

        # ---- 中继/投递公共量（V1 口径）----
        r_known_now = self._rescuer_known_mask(env)
        if r_known_now is not None and self._prev_rescuer_known is not None:
            delivered = int((r_known_now & ~self._prev_rescuer_known).sum())
        else:
            delivered = 0
        comm_idx = [i for i, o in enumerate(orders) if int(o.task) == TASK_COMM_RELAY]
        excl, shar = self.coverage_counts(env, comm_idx)
        cov_now = float(getattr(env, "coverage_rate", 0.0))
        comm_now = float(getattr(env, "comm_coverage_rate", cov_now))
        cov_drop = max(0.0, self._prev_cov - comm_now)
        blackout = comm_now <= 1e-6
        deliver_each = (min(RewardConfig().deliver_cap, RewardConfig().w_deliver * delivered)
                        / len(comm_idx) if comm_idx and delivered > 0 else 0.0)

        rewards = [0.0] * n
        parts = []
        for i, u in enumerate(env.uavs):
            order = orders[i]
            task, mode = int(order.task), int(order.mode)
            ux, uy = float(u.x), float(u.y)
            e_max = float(getattr(u, "e_max", getattr(env, "uav_energy", 600.0))) or 600.0
            cost = env.energy_cost_per_step(i) if hasattr(env, "energy_cost_per_step") else 2.5
            r = cfg.w_time + team_term
            p = {"time": cfg.w_time, "team": team_term}

            # ---------------- 通用：撞墙 / 被挡 / 能量 ----------------
            if getattr(u, "oob", False):
                r += cfg.w_oob; p["oob"] = cfg.w_oob
            if getattr(u, "blocked", False):
                r += cfg.w_blocked; p["blocked"] = cfg.w_blocked
            st, st_d = assigned_station(env, i)
            if float(u.e) - cost * st_d < 0.0:
                r += cfg.w_unsafe; p["unsafe"] = cfg.w_unsafe
            if float(u.e) <= 0.0:
                if i not in self._charged_depleted:
                    self._charged_depleted.add(i)
                    r += cfg.w_depleted; p["depleted"] = cfg.w_depleted
                else:
                    r += cfg.w_stranded; p["stranded"] = cfg.w_stranded
            elif i in self._charged_depleted:
                self._charged_depleted.discard(i)

            ev = events.get(i, {})

            # ---------------- tau = 0 火情感知 ----------------
            if task == TASK_FIRE_SENSING:
                g = (self.gm.goals.get(i) if self.gm is not None else None)
                if mode == SENSE_PATROL and g is not None and g.kind == "patrol":
                    # ---- patrol：环带/扇区/目标火点邻域刷新 ----
                    ring_eff = float(ev.get("ring", 6.0))
                    tgt = g.target
                    prog = float(ev.get("ring_progress", 0.0))
                    v = cfg.patrol_progress_w * float(np.clip(prog, -SQRT2, SQRT2))
                    r += v; p["ring_prog"] = v
                    if ev.get("in_band", False):
                        r += cfg.patrol_in_band; p["band"] = cfg.patrol_in_band
                    if ev.get("new_sector", False):
                        r += cfg.patrol_new_sector; p["sector"] = cfg.patrol_new_sector
                    # 目标火点邻域内的有效刷新（本机足迹 & 过时纠正 & 邻域）
                    if tgt is not None:
                        yy, xx = np.mgrid[0:H, 0:W]
                        near = ((xx - tgt[0]) ** 2 + (yy - tgt[1]) ** 2) <= (
                            ring_eff + cfg.patrol_refresh_margin) ** 2
                        ref = float((((foot[i] & stale_corrected & near) * share)).sum())
                        v2 = cfg.patrol_refresh_w * ref
                        r += v2; p["refresh"] = v2
                    xi, yi = int(round(ux)), int(round(uy))
                    if 0 <= xi < H and 0 <= yi < W and is_fire_now[xi, yi]:
                        r += RewardConfig().w_in_fire; p["in_fire"] = RewardConfig().w_in_fire
                    if ev.get("arrived", False):
                        r += cfg.arrival_bonus; p["arrival"] = cfg.arrival_bonus
                elif g is not None and g.kind == "explore" and order.target is not None:
                    # ---- explore（显式目标区域）----
                    rm = g.region_mask
                    inside_new = float(((foot[i] & newly_known & rm) * share).sum())
                    inside_fire = float(((foot[i] & newly_fire & rm) * share).sum())
                    outside_new = float(((foot[i] & newly_known & ~rm) * share).sum())
                    v = (cfg.explore_inside_new_cell * inside_new
                         + cfg.explore_inside_new_fire * inside_fire
                         + cfg.explore_inside_new_cell * cfg.explore_outside_mult * outside_new)
                    r += v
                    p["roi_new"] = cfg.explore_inside_new_cell * inside_new
                    p["roi_fire"] = cfg.explore_inside_new_fire * inside_fire
                    p["outside_new"] = (cfg.explore_inside_new_cell
                                        * cfg.explore_outside_mult * outside_new)
                    g.own_roi_new_cells += inside_new
                    prog = float(ev.get("progress", 0.0))
                    v2 = cfg.grid_progress_w * float(np.clip(prog, -SQRT2, SQRT2))
                    r += v2; p["goal_prog"] = v2
                    if ev.get("arrived", False):
                        r += cfg.arrival_bonus; p["arrival"] = cfg.arrival_bonus
                    if ev.get("completed", False):
                        r += cfg.completion_bonus; p["complete"] = cfg.completion_bonus
                else:
                    # ---- 自由探索（无显式目标）：保留 V1 能力 ----
                    v = (RewardConfig().w_new_cell * float((newly_known & foot[i]).sum())
                         + RewardConfig().w_new_fire_cell * float((newly_fire & foot[i]).sum()))
                    r += v; p["sense"] = v
                    tgt2, _tf, _rr = resolve_order_target(env, i, order, cache)
                    d_tgt = (math.hypot(tgt2[0] - ux, tgt2[1] - uy)
                             if tgt2 is not None else d_max)
                    sh = RewardConfig().w_shape_target * self._shape(i, "target", d_tgt, d_max)
                    r += sh; p["shape"] = sh

            # ---------------- tau = 1 通信中继（V1）----------------
            elif task == TASK_COMM_RELAY:
                n_r = max(1, len(getattr(env, "rescuers", []) or [1]))
                v = (RewardConfig().w_exclusive * excl.get(i, 0) / n_r
                     + RewardConfig().w_shared * shar.get(i, 0) / n_r)
                aoi = RewardConfig().w_aoi * min(1.0, float(getattr(env, "cur_max_AoI", 0))
                                                 / RewardConfig().aoi_cap)
                r += v + aoi; p["cover"] = v; p["aoi"] = aoi
                if deliver_each:
                    r += deliver_each; p["deliver"] = deliver_each
                if blackout:
                    r += RewardConfig().w_blackout; p["blackout"] = RewardConfig().w_blackout
                if cov_drop > 0:
                    cd = RewardConfig().w_cov_drop * cov_drop
                    r += cd; p["cov_drop"] = cd
                for j in comm_idx:
                    if j != i and math.hypot(env.uavs[j].x - ux, env.uavs[j].y - uy) < 4.0:
                        r += RewardConfig().w_cluster; p["cluster"] = RewardConfig().w_cluster
                        break
                tgt2, _tf, _rr = resolve_order_target(env, i, order, cache)
                d_tgt = (math.hypot(tgt2[0] - ux, tgt2[1] - uy)
                         if tgt2 is not None else d_max)
                sh = RewardConfig().w_shape_target * self._shape(i, "target", d_tgt, d_max)
                r += sh; p["shape"] = sh
                # P1R：指定人员本机链路奖励（在覆盖范围内持续钉住指定人员）
                if cfg.relay_own_link > 0 and ev.get("own_link", False):
                    r += cfg.relay_own_link; p["own_link"] = cfg.relay_own_link
                if order.proactive_handover:
                    weak = None
                    for j in comm_idx:
                        uj = env.uavs[j]
                        sj, sjd = assigned_station(env, j)
                        if j != i and float(uj.e) - cost * sjd < 0.0:
                            weak = (float(uj.x), float(uj.y))
                            break
                    hd = (math.hypot(weak[0] - ux, weak[1] - uy) if weak else d_max)
                    sh2 = RewardConfig().w_handover * self._shape(i, "handover", hd, d_max)
                    r += sh2; p["handover"] = sh2

            # ---------------- tau = 2 返航补给（V1）----------------
            elif task == TASK_RECHARGE:
                sh = RewardConfig().w_shape_station * self._shape(i, "station", st_d, d_max)
                r += sh; p["shape"] = sh
                on_station = env.at_station(ux, uy) if hasattr(env, "at_station") else \
                    (int(round(ux)), int(round(uy))) in \
                    [(int(s[0]), int(s[1])) for s in env.charging_stations]
                prev_e = self._prev_e[i] if self._prev_e is not None else float(u.e)
                needed = prev_e < RewardConfig().arrive_need_ratio * e_max
                if on_station and i not in self._on_station and needed:
                    r += RewardConfig().w_arrive; p["arrive"] = RewardConfig().w_arrive
                elif on_station:
                    r += RewardConfig().w_camp; p["camp"] = RewardConfig().w_camp

            rewards[i] = float(r)
            parts.append(p)

        st_set = [(int(s[0]), int(s[1])) for s in getattr(env, "charging_stations", [])]
        self._on_station = {j for j, uu in enumerate(env.uavs)
                            if (int(round(uu.x)), int(round(uu.y))) in st_set}
        self.last_events = events

        info = {
            "coverage_rate": cov_now,
            "comm_coverage_rate": comm_now,
            "explored_rate": float(getattr(env, "explored_rate", cache["explored_rate"])),
            "fire_discovery_rate": float(getattr(env, "active_discovery_rate", 0.0)),
            "fire_discovery_rate_all": float(getattr(env, "fire_discovery_rate", 0.0)),
            "burned_area_rate": float(getattr(env, "burned_area_rate", 0.0)),
            "rescuer_known_fire_rate": float(getattr(env, "active_rescuer_known_rate", 0.0)),
            "rescuer_known_fire_rate_all": float(getattr(env, "rescuer_known_fire_rate", 0.0)),
            "delivered_cells": int(delivered),
            "n_extinguished": int((bs == 3).sum()),
            "burn_growth": int(burn_growth),
            "max_AoI": float(getattr(env, "cur_max_AoI", 0)),
            "min_energy": float(min(float(u.e) for u in env.uavs)),
            "n_depleted": int(sum(1 for u in env.uavs if float(u.e) <= 0.0)),
            "blackout": bool(blackout),
            "n_clusters": len(cache["clusters"]),
            "n_contained": int(cache["n_contained"]),
            "energies": [float(u.e) for u in env.uavs],
            "parts": parts,
            "goal_events": {i: {k: v for k, v in ev.items() if k != "goal"}
                            for i, ev in events.items()},
        }
        return rewards, info

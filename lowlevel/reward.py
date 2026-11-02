# -*- coding: utf-8 -*-
"""按任务分解的逐步稠密奖励（多火点搜寻 + 信息回传场景）。

所有引导项都用 potential-based reward shaping：F = γ·Φ(s′) − Φ(s)，Φ = −d/D_max
（Ng, Harada & Russell, ICML 1999：这种形式不改变最优策略，因此可以放心加大权重）。

与第一版的主要区别（对应场景纠正）：
1. 探索奖励按**全地图新格**计数，不再只数火线边界格。地图上初始有多个分散火点，
   还没找到任何火点时"火线前沿"是空集，第一版在这种状态下没有梯度，
   策略退化成绕地图边界打转。
2. 感知任务拆成两种模式：`explore` 铺开未知区域，`patrol` 绕指定火点边界巡逻
   （环半径势函数），对应高层给的两类任务建议。
3. 中继任务新增**信息投递**奖励：无人机把人员还不知道的火情上传给人员才给分。
   这是"人员的火场感知来自无人机"这条因果链上的唯一一环——人员拿不到地图就不会去灭火。
4. 全队共享"过火面积增量"惩罚：早发现 → 早传递 → 早扑灭 → 过火面积小。
"""
import math

import numpy as np

from .task_spec import (
    TASK_FIRE_SENSING, TASK_COMM_RELAY, TASK_RECHARGE,
    SENSE_EXPLORE, SENSE_PATROL,
)
from .obs_builder import (
    assigned_station, resolve_order_target, shared_cache, team_map, PATCH_R,
)


class RewardConfig:
    gamma = 0.99

    # ---------------- 通用项 ----------------
    w_time = -0.005         # 每步时间成本
    w_oob = -0.30           # 动作越界（撞墙）
    w_blocked = -0.15       # 被队友阻挡
    w_unsafe = -0.02        # 返航余量为负的每步惩罚
    w_depleted = -2.00      # 电量耗尽（仅第一次触发）
    w_stranded = -0.03      # 耗尽后每步（已被迫悬停）
    # 过火面积增量：长周期目标，但它高度依赖火场本身的推进、单个无人机能影响的很有限，
    # 而且这个量噪声很大（配对测试里 random/greedy 的差值均值只有 +0.0004）。
    # 原来给 -0.05/格，一回合能罚到每机 -20，量级和整个探索收益相当，压过了有效信号。
    w_burn_growth = -0.02

    # ---------------- tau=0 感知：explore 模式 ----------------
    w_new_cell = 0.04       # 本机足迹内每个新探索格（全地图，不限火场）
    w_new_fire_cell = 0.40  # 本机足迹内每个新发现的燃烧格 —— 首要目标
    w_shape_target = 1.0    # 朝指令目标（未知区域中心/最近前沿）的 shaping

    # ---------------- tau=0 感知：patrol 模式 ----------------
    w_fire_refresh = 0.05   # 足迹内每纠正 1 个"认知已过时"的格子（巡逻的主要收益）
    w_shape_ring = 1.5      # 绕火点的环半径 shaping，Phi = -|d - ring|/D_max
    w_in_fire = -0.05       # 停在燃烧格上（无人机不该往火里钻）

    # ---------------- tau=1 通信中继 ----------------
    # 覆盖奖励按**人数比例**给，不按人数给：按人数时中继机每步最高能拿
    # 0.20 x 8 = 1.6，600 步一回合就是 960；而探索全图铺开总共才 200 左右，
    # 覆盖把探索完全压过去了——实测策略在 explore x4 指令下只铺开 0.44
    # （贪心基线 0.776），而 eval_return 因为覆盖高还在涨，最优权重会被选歪。
    w_exclusive = 0.50      # 仅由本机覆盖的人员占比（每步上限 0.50）
    w_shared = 0.15         # 本机覆盖的人员占比（含与他机重叠）
    # AoI 惩罚必须**有界**：原来是 -0.02 x (max_AoI / 10)，max_AoI 无上限且是历史最大值，
    # 600 步一回合能累到 -276，中继机不管怎么飞都摆脱不掉，直接把该任务的回报压成负数。
    w_aoi = -0.10           # 乘以 min(cur_max_AoI, aoi_cap) / aoi_cap，每步上限 0.10
    aoi_cap = 50.0
    w_blackout = -0.30      # 通信全断（覆盖率=0）时每台中继机每步
    w_cov_drop = -0.50      # 覆盖率相对上一步的下降量（乘下降幅度）
    w_cluster = -0.10       # 两台中继机距离过近（扎堆）
    w_deliver = 0.02        # 每向人员投递 1 个"人员原本不知道的燃烧格"
    deliver_cap = 2.0       # 单步投递奖励上限（首次上传会一次性同步上千格）
    w_handover = 0.30       # 接手低电量中继机（switching-aware）

    # ---------------- tau=2 返航补给 ----------------
    w_shape_station = 1.5
    w_arrive = 2.00         # 抵达补给点的事件奖励
    arrive_need_ratio = 0.6  # 且入站前电量必须低于 e_max 的这个比例才算真的需要
    w_camp = 0.00           # 满电还占着站点：不给奖励，靠 w_time 施加机会成本


class LowLevelReward:
    def __init__(self, cfg: RewardConfig = None):
        self.cfg = cfg or RewardConfig()
        self._prev_phi = {}
        self._charged_depleted = set()
        self._on_station = set()
        self._prev_e = None
        self._prev_team = None
        self._prev_burning = None
        self._prev_rescuer_known = None
        self._prev_cov = 0.0
        self._prev_burned = 0

    def reset(self, env):
        self._prev_phi.clear()
        self._charged_depleted.clear()
        self._on_station.clear()
        self._prev_e = None
        self._prev_team = team_map(env).copy()
        self._prev_burning = env.grid["burning_state"].copy()
        self._prev_rescuer_known = self._rescuer_known_mask(env)
        self._prev_cov = float(getattr(env, "comm_coverage_rate", 0.0))
        self._prev_burned = int((env.grid["burning_state"] > 0).sum())

    def snapshot(self, env):
        """在 env.step() 之前调用，保存差分所需的上一步量。"""
        self._prev_team = team_map(env).copy()
        self._prev_burning = env.grid["burning_state"].copy()
        self._prev_rescuer_known = self._rescuer_known_mask(env)
        self._prev_e = [float(u.e) for u in env.uavs]
        self._prev_cov = float(getattr(env, "comm_coverage_rate", 0.0))
        self._prev_burned = int((env.grid["burning_state"] > 0).sum())

    @staticmethod
    def _rescuer_known_mask(env):
        M = getattr(env, "rescuers_map_info", None)
        if M is None:
            return None
        return ((M > 0) & (M < 2)).copy()

    # ------------------------------------------------------------ shaping
    def _shape(self, idx, key, d_now, d_max):
        """F = gamma*Phi' - Phi, Phi = -d/D_max。"""
        phi_now = -min(float(d_now) / d_max, 1.0)
        k = (idx, key)
        phi_prev = self._prev_phi.get(k)
        self._prev_phi[k] = phi_now
        if phi_prev is None:
            return 0.0
        return self.cfg.gamma * phi_now - phi_prev

    def clear_shape(self, idx):
        for k in [k for k in self._prev_phi if k[0] == idx]:
            self._prev_phi.pop(k, None)

    # ------------------------------------------------------- 覆盖统计
    @staticmethod
    def coverage_counts(env, comm_indices):
        """统计每架中继机的 独占/共享 覆盖人数。

        独占 = 该救援人员只被这一架无人机覆盖（且不在基站覆盖内）。
        用独占而不是"总覆盖/机数"是为了解耦信用分配，否则多机扎堆覆盖同一批人
        却都拿满分。
        """
        R2 = float(getattr(env, "uav_comm_radius", 12)) ** 2
        base_x, base_y = getattr(env, "base_pos", (0.0, 50.0))
        base_r2 = float(getattr(env, "base_comm_radius", 15.0)) ** 2
        excl = {i: 0 for i in comm_indices}
        shar = {i: 0 for i in comm_indices}
        for r in getattr(env, "rescuers", []):
            if (r.x - base_x) ** 2 + (r.y - base_y) ** 2 <= base_r2:
                continue                      # 基站已覆盖，不计入无人机功劳
            hits = [i for i in comm_indices
                    if (r.x - env.uavs[i].x) ** 2 + (r.y - env.uavs[i].y) ** 2 <= R2]
            for i in hits:
                shar[i] += 1
            if len(hits) == 1:
                excl[hits[0]] += 1
        return excl, shar

    # ------------------------------------------------------------ 主入口
    def compute(self, env, orders, max_steps=600):
        """在 env.step() 之后调用，返回 (rewards, info)。orders: List[UavOrder]。"""
        cfg = self.cfg
        n = len(env.uavs)
        H, W = env.grid_size
        d_max = math.hypot(float(H), float(W))
        bs = env.grid["burning_state"]
        cache = shared_cache(env)
        M = cache["M"]

        # ---- 全队共享项：过火面积增量 ----
        burned_now = int((bs > 0).sum())
        burn_growth = max(0, burned_now - int(self._prev_burned))
        team_term = cfg.w_burn_growth * burn_growth

        # ---- 新探索格 / 新发现火格：按足迹归属，重叠则均分 ----
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
        new_cell_credit = [(foot[i] & newly_known) * share for i in range(n)]
        new_fire_credit = [(foot[i] & newly_fire) * share for i in range(n)]

        # ---- 认知纠偏（巡逻用）：足迹内"原本已知但已经过时"的格子被刷新 ----
        # 用"过时被纠正"而不是"真值发生变化"：半速扩散下全图每步只有 ~0.4 格
        # 状态变化，按后者给分几乎等于没有信号，巡逻拿不到正收益（实测贪心不如随机）。
        # 而火场向外长、认知留在旧快照上，绕着火线飞就会持续纠正大量过时格子。
        prev_known = (prev_team >= 0.0)
        stale_corrected = prev_known & (np.abs(prev_team - bs) > 0.05)
        is_fire_now = (bs > 0) & (bs < 2)

        # ---- 信息投递（中继用）：人员新知道的燃烧格 ----
        r_known_now = self._rescuer_known_mask(env)
        if r_known_now is not None and self._prev_rescuer_known is not None:
            delivered = int((r_known_now & ~self._prev_rescuer_known).sum())
        else:
            delivered = 0
        comm_idx = [i for i, o in enumerate(orders) if int(o.task) == TASK_COMM_RELAY]
        excl, shar = self.coverage_counts(env, comm_idx)
        cov_now = float(getattr(env, "coverage_rate", 0.0))          # 仅无人机贡献
        comm_now = float(getattr(env, "comm_coverage_rate", cov_now))  # 含基站
        cov_drop = max(0.0, self._prev_cov - comm_now)
        # blackout 要看"人员到底有没有通信"，不能用只算无人机的那个口径——
        # 人员在基站圈里时无人机覆盖为 0，但通信是好的，不该罚中继机。
        blackout = comm_now <= 1e-6
        # 投递功劳平摊给所有中继机（谁在覆盖圈里由环境决定，这里不再细分）
        deliver_each = (min(cfg.deliver_cap, cfg.w_deliver * delivered) / len(comm_idx)
                        if comm_idx and delivered > 0 else 0.0)

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
                self._charged_depleted.discard(i)      # 被补给点救回来了

            tgt, tgt_is_fire, ring = resolve_order_target(env, i, order, cache)
            d_tgt = (math.hypot(tgt[0] - ux, tgt[1] - uy) if tgt is not None else d_max)

            # ---------------- tau = 0 火情感知 ----------------
            if task == TASK_FIRE_SENSING:
                v = (cfg.w_new_cell * float(new_cell_credit[i].sum())
                     + cfg.w_new_fire_cell * float(new_fire_credit[i].sum()))
                r += v; p["sense"] = v
                # patrol 只有在"目标确实是火点"时才用环半径势函数；
                # 还没发现火点时它退化成搜寻，用普通的趋近势函数。
                if mode == SENSE_PATROL and tgt_is_fire:
                    ref = float((((foot[i] & stale_corrected) * share)).sum())
                    v2 = cfg.w_fire_refresh * ref
                    r += v2; p["refresh"] = v2
                    # 环半径势函数：Phi = -|d - ring| / D_max，让它绕着火点转而不是扎进去
                    err = abs(d_tgt - float(ring))
                    sh = cfg.w_shape_ring * self._shape(i, "ring", err, d_max)
                    r += sh; p["shape"] = sh
                    xi, yi = int(round(ux)), int(round(uy))
                    if 0 <= xi < H and 0 <= yi < W and is_fire_now[xi, yi]:
                        r += cfg.w_in_fire; p["in_fire"] = cfg.w_in_fire
                else:
                    sh = cfg.w_shape_target * self._shape(i, "target", d_tgt, d_max)
                    r += sh; p["shape"] = sh

            # ---------------- tau = 1 通信中继 ----------------
            elif task == TASK_COMM_RELAY:
                n_r = max(1, len(getattr(env, "rescuers", []) or [1]))
                v = (cfg.w_exclusive * excl.get(i, 0) / n_r
                     + cfg.w_shared * shar.get(i, 0) / n_r)
                aoi = cfg.w_aoi * min(1.0, float(getattr(env, "cur_max_AoI", 0))
                                      / cfg.aoi_cap)
                r += v + aoi; p["cover"] = v; p["aoi"] = aoi
                if deliver_each:
                    r += deliver_each; p["deliver"] = deliver_each
                if blackout:
                    r += cfg.w_blackout; p["blackout"] = cfg.w_blackout
                if cov_drop > 0:
                    cd = cfg.w_cov_drop * cov_drop
                    r += cd; p["cov_drop"] = cd
                for j in comm_idx:
                    if j != i and math.hypot(env.uavs[j].x - ux, env.uavs[j].y - uy) < 4.0:
                        r += cfg.w_cluster; p["cluster"] = cfg.w_cluster
                        break
                sh = cfg.w_shape_target * self._shape(i, "target", d_tgt, d_max)
                r += sh; p["shape"] = sh
                if order.proactive_handover:
                    # 朝"返航余量已为负的中继机"靠拢，提前接手它的覆盖
                    weak = None
                    for j in comm_idx:
                        uj = env.uavs[j]
                        sj, sjd = assigned_station(env, j)
                        if j != i and float(uj.e) - cost * sjd < 0.0:
                            weak = (float(uj.x), float(uj.y))
                            break
                    hd = (math.hypot(weak[0] - ux, weak[1] - uy) if weak else d_max)
                    sh2 = cfg.w_handover * self._shape(i, "handover", hd, d_max)
                    r += sh2; p["handover"] = sh2

            # ---------------- tau = 2 返航补给 ----------------
            elif task == TASK_RECHARGE:
                sh = cfg.w_shape_station * self._shape(i, "station", st_d, d_max)
                r += sh; p["shape"] = sh
                on_station = env.at_station(ux, uy) if hasattr(env, "at_station") else \
                    (int(round(ux)), int(round(uy))) in \
                    [(int(s[0]), int(s[1])) for s in env.charging_stations]
                # 只判"离站->在站"跳变不够：满电的机可以在站点和邻格之间来回跳，
                # 每两步白拿一次 w_arrive。必须再要求"入站前确实缺电"。
                prev_e = self._prev_e[i] if self._prev_e is not None else float(u.e)
                needed = prev_e < cfg.arrive_need_ratio * e_max
                if on_station and i not in self._on_station and needed:
                    r += cfg.w_arrive; p["arrive"] = cfg.w_arrive
                elif on_station:
                    r += cfg.w_camp; p["camp"] = cfg.w_camp

            rewards[i] = float(r)
            parts.append(p)

        st_set = [(int(s[0]), int(s[1])) for s in getattr(env, "charging_stations", [])]
        self._on_station = {j for j, uu in enumerate(env.uavs)
                            if (int(round(uu.x)), int(round(uu.y))) in st_set}

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
        }
        return rewards, info

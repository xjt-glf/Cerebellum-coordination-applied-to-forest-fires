# -*- coding: utf-8 -*-
"""任务条件观测构造器（多火点搜寻 + 信息回传场景）。

低层策略的输入 o_i = f(shared_state, i, order_i)。

两个关键前提（与第一版不同）：
1. **全队共享观测**。上层是集中式规划，所有无人机看同一份融合后的火势认知图
   `env.team_map_info`（未知 = −1）。个体差异来自自身位姿、电量、被指派的指令。
2. **任务区域 100x100、初始 3 个分散火点**。所以火图不能再整张塞进观测：
   近处给 9x9 细节 patch，全局给 10x10 的两通道粗视图（已探索比例 / 已知火比例）。
   这也让"还有哪片区域没探索"在观测里直接可读——第一版只给了"火线前沿"，
   在还没找到任何火点时该项恒为空，策略于是退化成绕地图边界打转。

分区布局见 OBS_SLICES。
"""
import math

import numpy as np

from .task_spec import (
    N_TASKS, N_SENSE_MODES, UavOrder,
    TASK_FIRE_SENSING, TASK_COMM_RELAY, TASK_RECHARGE,
    SENSE_EXPLORE, SENSE_PATROL,
)

PATCH_R = 4                      # 局部 patch 半径 → 9x9
PATCH = 2 * PATCH_R + 1
COARSE = 10                      # 全局粗视图 10x10
MAX_AGENTS = 8
D_MAX_DEFAULT = math.hypot(100.0, 100.0)

_DIMS = [
    ("self", 7),
    ("agent_id", MAX_AGENTS),
    ("task", N_TASKS),
    ("order", UavOrder.vector_dim()),
    ("local", PATCH * PATCH),
    ("coarse", 2 * COARSE * COARSE),
    ("frontier", 4),
    ("fire", 9),
    ("comm", 8),
    ("energy", 4),
    ("team", 7),
    ("coord", 6),
    ("global", 8),
]

OBS_SLICES = {}
_off = 0
for _name, _d in _DIMS:
    OBS_SLICES[_name] = slice(_off, _off + _d)
    _off += _d
OBS_DIM = _off


def d_max_of(env):
    H, W = env.grid_size
    return math.hypot(float(H), float(W))


def _rel(ux, uy, tx, ty, dmax):
    """到目标点的 (单位方向 x, 单位方向 y, 归一化距离)。目标为 None 时返回 (0,0,1)。"""
    if tx is None or ty is None:
        return 0.0, 0.0, 1.0
    dx, dy = float(tx) - float(ux), float(ty) - float(uy)
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return 0.0, 0.0, 0.0
    return dx / d, dy / d, min(d / dmax, 1.0)


# ====================================================================== 缓存
def team_map(env):
    """全队融合后的火势认知图（未知 = −1）。"""
    m = getattr(env, "team_map_info", None)
    if m is None:
        H, W = env.grid_size
        m = (env.uavs[0].map_info.copy() if env.uavs
             else np.full((H, W), -1.0, dtype=np.float32))
        for uav in env.uavs[1:]:
            m = np.maximum(m, uav.map_info)
        env.team_map_info = m
    return m


def shared_cache(env):
    """按步缓存全队公共量：4 架无人机的观测都从这里取，只算一次。"""
    key = int(getattr(env, "fire_update_counter", 0))
    c = getattr(env, "_obs_cache", None)
    if c is not None and c[0] == key:
        return c[1]

    M = team_map(env)
    H, W = M.shape
    known = M >= 0.0
    unknown = ~known
    burning = (M > 0) & (M < 2)

    # 地图前沿：已知格且八邻域里有未知格 —— 这是"该往哪继续探索"的直接信号
    pad = np.pad(unknown, 1, mode="constant", constant_values=False)
    has_unknown_nb = np.zeros_like(unknown)
    for dx in (0, 1, 2):
        for dy in (0, 1, 2):
            if dx == 1 and dy == 1:
                continue
            has_unknown_nb |= pad[dx:dx + H, dy:dy + W]
    frontier = known & has_unknown_nb
    fx, fy = np.nonzero(frontier)

    # 已知火线边界格（已知燃烧且邻接已知健康格）
    healthy = np.isclose(M, 0.0)
    padh = np.pad(healthy, 1, mode="constant", constant_values=False)
    has_healthy_nb = np.zeros_like(healthy)
    for dx in (0, 1, 2):
        for dy in (0, 1, 2):
            if dx == 1 and dy == 1:
                continue
            has_healthy_nb |= padh[dx:dx + H, dy:dy + W]
    fire_edge = burning & has_healthy_nb
    ex, ey = np.nonzero(fire_edge)

    # 全局粗视图：块内"已探索比例"和"已知火比例"
    bh, bw = H // COARSE, W // COARSE
    kb = known[:bh * COARSE, :bw * COARSE].reshape(COARSE, bh, COARSE, bw).mean(axis=(1, 3))
    fb = burning[:bh * COARSE, :bw * COARSE].reshape(COARSE, bh, COARSE, bw).mean(axis=(1, 3))

    # 已知火点簇（基于共享认知图，不是人员图）。
    # 只把**未合围**的火点当作待处理目标：合围后的火不会再扩散，
    # 按设计既不派人员去灭，也不派无人机去侦查。
    all_clusters = env.known_fire_clusters(map_info=M, min_cells=3)
    clusters = [c for c in all_clusters if not c.get("contained", False)]
    n_contained = len(all_clusters) - len(clusters)

    # 未被覆盖的人员
    uncovered = [(r.x, r.y) for r in getattr(env, "rescuers", []) if not r.is_covered]
    all_r = [(r.x, r.y) for r in getattr(env, "rescuers", [])]

    cache = dict(
        M=M, known=known, burning=burning,
        frontier_pts=np.stack([fx, fy], axis=1) if fx.size else np.zeros((0, 2)),
        fire_edge_pts=np.stack([ex, ey], axis=1) if ex.size else np.zeros((0, 2)),
        coarse=np.concatenate([kb.ravel(), fb.ravel()]).astype(np.float32),
        explored_rate=float(known.mean()),
        clusters=clusters, n_contained=n_contained,
        uncovered=uncovered, all_rescuers=all_r,
    )
    env._obs_cache = (key, cache)
    return cache


def nearest_unknown_block_center(env, ux, uy, cache=None, thresh=0.5):
    """最近的"大片未探索"粗块中心。

    直接用"最近的地图前沿格"当搜寻目标是不够的：前沿总是紧贴自己足迹的边缘，
    距离只有几格，势函数梯度极小，策略可以原地悬停也不吃亏（实测就是这样）。
    改成指向一个**远处、稳定**的未探索块中心。
    """
    c = cache if cache is not None else shared_cache(env)
    H, W = env.grid_size
    bh, bw = H // COARSE, W // COARSE
    known_frac = c["coarse"][:COARSE * COARSE].reshape(COARSE, COARSE)
    cand = np.argwhere(known_frac < (1.0 - thresh))
    if cand.size == 0:
        cand = np.argwhere(known_frac < 0.999)
    if cand.size == 0:
        return None
    cx = cand[:, 0] * bh + bh / 2.0
    cy = cand[:, 1] * bw + bw / 2.0
    d2 = (cx - ux) ** 2 + (cy - uy) ** 2
    i = int(np.argmin(d2))
    return (float(cx[i]), float(cy[i]))


def _nearest_pt(pts, ux, uy):
    """pts: [N,2] ndarray。返回 ((x,y), dist)；空则 (None, inf)。"""
    if pts is None or len(pts) == 0:
        return None, float("inf")
    d2 = (pts[:, 0] - ux) ** 2 + (pts[:, 1] - uy) ** 2
    i = int(np.argmin(d2))
    return (float(pts[i, 0]), float(pts[i, 1])), float(math.sqrt(d2[i]))


# ============================================================ 补给点全局匹配
def station_assignment(env):
    """把补给点全局去冲突地分配给各架无人机（贪心最小距离匹配）。

    必须是**全局**匹配：只按"索引更小的队友"局部去冲突时，各机看到的空闲集合
    互相矛盾，且队友一移动分配就翻转，势函数的目标每步抖动——实测表现是有无人机
    停在已被占用的补给点旁边来回游走直到耗尽。
    """
    stations = [(int(s[0]), int(s[1])) for s in getattr(env, "charging_stations", [])]
    if not stations:
        return {}
    key = tuple((round(float(u.x), 3), round(float(u.y), 3)) for u in env.uavs)
    cached = getattr(env, "_station_assign_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    pairs = []
    for i, u in enumerate(env.uavs):
        for s in stations:
            pairs.append((math.hypot(s[0] - float(u.x), s[1] - float(u.y)), i, s))
    pairs.sort(key=lambda p: (p[0], p[1]))
    assign, used_u, used_s = {}, set(), set()
    for _d, i, s in pairs:
        if i in used_u or s in used_s:
            continue
        assign[i] = (float(s[0]), float(s[1]))
        used_u.add(i)
        used_s.add(s)
    env._station_assign_cache = (key, assign)
    return assign


def assigned_station(env, idx):
    """第 idx 架无人机被分配到的补给点，返回 ((x,y), 距离)。"""
    stations = list(getattr(env, "charging_stations", []))
    if not stations:
        return None, float("inf")
    u = env.uavs[idx]
    # 网页部署的补给协议可以指定实际站点；训练默认仍使用原全局匹配。
    selected = getattr(env, "_deployment_station_targets", {}).get(idx)
    if selected is not None and tuple(selected) in [tuple(st) for st in stations]:
        s = tuple(map(float, selected))
        return s, math.hypot(s[0] - float(u.x), s[1] - float(u.y))
    if getattr(env, "_deployment_nearest_charger", False):
        radius = float(getattr(env, "charge_radius", 1))
        s = min(stations, key=lambda st: (
            max(0, max(abs(float(st[0]) - u.x), abs(float(st[1]) - u.y)) - radius),
            math.hypot(float(st[0]) - u.x, float(st[1]) - u.y)))
        s = tuple(map(float, s))
        return s, math.hypot(s[0] - float(u.x), s[1] - float(u.y))
    s = station_assignment(env).get(idx)
    if s is None:
        best, bd = None, float("inf")
        for st in stations:
            d = math.hypot(float(st[0]) - u.x, float(st[1]) - u.y)
            if d < bd:
                best, bd = (float(st[0]), float(st[1])), d
        return best, bd
    return s, math.hypot(s[0] - float(u.x), s[1] - float(u.y))


def cluster_ring_radius(cl):
    """火点的等效半径 + 2 格余量：绕圈半径要随火场大小变，写死 6 格会让无人机
    在比 6 格更大的火场里面绕，白吃"停在燃烧格上"的惩罚。"""
    return math.sqrt(max(1.0, float(cl["size"])) / math.pi) + 2.0


def _ring(order, auto):
    if order.ring_radius is not None:
        return float(order.ring_radius)
    return float(auto) if auto is not None else 6.0


def resolve_order_target(env, idx, order, cache=None):
    """把 order 里可能为 None 的目标解析成一个具体坐标。

    返回 (target, target_is_fire, ring)。target_is_fire 表示"这个目标是火点，
    应该绕着它转"；ring 是该绕多大半径（order 没指定时按火点规模自动取）。

    高层没指定时策略需要一个兜底目标，否则势函数没有梯度（这正是第一版
    "没找到火点就绕地图边界打转"的根源）：
        感知-搜寻 → 最近的地图前沿格
        感知-巡逻 → 最近的已知火点簇质心
        中继      → 指定人员，否则最近的未覆盖人员，否则人员质心
        补给      → 全局匹配给自己的补给点
    """
    c = cache if cache is not None else shared_cache(env)
    u = env.uavs[idx]
    ux, uy = float(u.x), float(u.y)
    task, mode = int(order.task), int(order.mode)

    if task == TASK_RECHARGE:
        st, _ = assigned_station(env, idx)
        # 基站/任意坐标不能充电；只接受真实停机坪内的显式目标。
        target = order.target
        if target is not None and env.at_station(*target):
            return target, False, _ring(order, None)
        return st, False, _ring(order, None)

    if task == TASK_COMM_RELAY:
        if order.rescuer_id is not None:
            for r in getattr(env, "rescuers", []):
                if int(getattr(r, "rescuer_id", -1)) == int(order.rescuer_id):
                    return (float(r.x), float(r.y)), False, _ring(order, None)
        if order.target is not None:
            return order.target, False, _ring(order, None)
        pts = np.array(c["uncovered"], dtype=np.float64) if c["uncovered"] else None
        p, _ = _nearest_pt(pts, ux, uy)
        if p is not None:
            return p, False, _ring(order, None)
        if c["all_rescuers"]:
            arr = np.array(c["all_rescuers"], dtype=np.float64)
            return ((float(arr[:, 0].mean()), float(arr[:, 1].mean())), False,
                    _ring(order, None))
        return None, False, _ring(order, None)

    # 感知任务
    if order.target is not None:
        auto = None
        if mode == SENSE_PATROL and c["clusters"]:
            cl = min(c["clusters"], key=lambda z: (z["centroid"][0] - order.target[0]) ** 2
                     + (z["centroid"][1] - order.target[1]) ** 2)
            auto = cluster_ring_radius(cl)
        return order.target, (mode == SENSE_PATROL), _ring(order, auto)
    if mode == SENSE_PATROL and c["clusters"]:
        # 高层没指定火点时按机号轮转，而不是各自挑"最近的"——后者会让整队
        # 全部叠在同一个火点上，另外几处火情无人侦测。
        cl = c["clusters"][idx % len(c["clusters"])]
        return ((float(cl["centroid"][0]), float(cl["centroid"][1])), True,
                _ring(order, cluster_ring_radius(cl)))
    # patrol 但还没发现任何火点时退化成搜寻——否则环半径势函数会围着
    # "紧贴自己足迹的前沿格"打转，无人机原地不动（实测 explored 只有 0.07）
    p = nearest_unknown_block_center(env, ux, uy, c)
    if p is not None:
        return p, False, _ring(order, None)
    p, _ = _nearest_pt(c["frontier_pts"], ux, uy)
    if p is not None:
        return p, False, _ring(order, None)
    cents = (np.array([cl["centroid"] for cl in c["clusters"]], dtype=np.float64)
             if c["clusters"] else None)
    p, _ = _nearest_pt(cents, ux, uy)
    return p, (p is not None), _ring(order, None)


# ================================================================ 主构造函数
def build_obs(env, idx, order, step_t=0, max_steps=600, cache=None):
    """构造第 idx 架无人机在指令 order 下的观测向量。"""
    c = cache if cache is not None else shared_cache(env)
    uav = env.uavs[idx]
    H, W = env.grid_size
    dmax = math.hypot(float(H), float(W))
    ux, uy = float(uav.x), float(uav.y)
    e_max = float(getattr(uav, "e_max", getattr(env, "uav_energy", 600.0))) or 600.0
    cost = env.energy_cost_per_step(idx) if hasattr(env, "energy_cost_per_step") else 2.5
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    task, mode = int(order.task), int(order.mode)

    st, st_d = assigned_station(env, idx)
    return_margin = (float(uav.e) - cost * st_d) / e_max

    # ---------------- self (7) ----------------
    obs[OBS_SLICES["self"]] = [
        ux / H, uy / W, float(uav.e) / e_max, float(uav.d) / 5.0,
        float(np.clip(return_margin, -1.0, 1.0)),
        float(step_t) / max(1, max_steps),
        1.0 if float(uav.e) <= 0.0 else 0.0,
    ]

    # ---------------- agent id one-hot (8) ----------------
    # 同构机队 + 参数共享策略，不给身份信息则各机策略完全对称，会同时耗尽、
    # 同时返航。给身份是让分工可学的标准做法（agent indication）。
    aid = np.zeros(MAX_AGENTS, dtype=np.float32)
    if 0 <= idx < MAX_AGENTS:
        aid[idx] = 1.0
    obs[OBS_SLICES["agent_id"]] = aid

    # ---------------- task one-hot (3) ----------------
    tv = np.zeros(N_TASKS, dtype=np.float32)
    if 0 <= task < N_TASKS:
        tv[task] = 1.0
    obs[OBS_SLICES["task"]] = tv

    # ---------------- order (9) ----------------
    resolved, tgt_is_fire, ring = resolve_order_target(env, idx, order, c)
    obs[OBS_SLICES["order"]] = order.to_vector(ux, uy, dmax, resolved_target=resolved,
                                              target_is_fire=tgt_is_fire, ring=ring)

    # ---------------- local 9x9 patch of shared map (81) ----------------
    # 取共享认知图而不是真值图：未知仍是未知，策略必须自己去看
    M = c["M"]
    xi, yi = int(round(ux)), int(round(uy))
    patch = np.full((PATCH, PATCH), -1.0, dtype=np.float32)
    x0, x1 = max(0, xi - PATCH_R), min(H, xi + PATCH_R + 1)
    y0, y1 = max(0, yi - PATCH_R), min(W, yi + PATCH_R + 1)
    patch[x0 - (xi - PATCH_R):x1 - (xi - PATCH_R),
          y0 - (yi - PATCH_R):y1 - (yi - PATCH_R)] = M[x0:x1, y0:y1]
    obs[OBS_SLICES["local"]] = (patch / 3.0).ravel()

    # ---------------- coarse global view (2x10x10) ----------------
    obs[OBS_SLICES["coarse"]] = c["coarse"]

    # ---------------- frontier (4) ----------------
    fp, fd = _nearest_pt(c["frontier_pts"], ux, uy)
    obs[OBS_SLICES["frontier"]] = [*_rel(ux, uy, fp[0] if fp else None,
                                         fp[1] if fp else None, dmax),
                                   1.0 - c["explored_rate"]]

    # ---------------- fire (8) ----------------
    clusters = c["clusters"]
    if clusters:
        cents = np.array([cl["centroid"] for cl in clusters], dtype=np.float64)
        cp, cd = _nearest_pt(cents, ux, uy)
        ci = int(np.argmin((cents[:, 0] - ux) ** 2 + (cents[:, 1] - uy) ** 2))
        csize = min(1.0, clusters[ci]["size"] / 200.0)
        n_cl = min(1.0, len(clusters) / 6.0)
    else:
        cp, csize, n_cl = None, 0.0, 0.0
    ep, ed = _nearest_pt(c["fire_edge_pts"], ux, uy)
    obs[OBS_SLICES["fire"]] = [
        *_rel(ux, uy, cp[0] if cp else None, cp[1] if cp else None, dmax),
        csize, n_cl,
        *_rel(ux, uy, ep[0] if ep else None, ep[1] if ep else None, dmax),
        min(1.0, c["n_contained"] / 6.0),      # 已合围（不必再管）的火点数
    ]

    # ---------------- comm (8) ----------------
    unc = np.array(c["uncovered"], dtype=np.float64) if c["uncovered"] else None
    rp, rd = _nearest_pt(unc, ux, uy)
    if c["all_rescuers"]:
        arr = np.array(c["all_rescuers"], dtype=np.float64)
        rc = (float(arr[:, 0].mean()), float(arr[:, 1].mean()))
    else:
        rc = None
    obs[OBS_SLICES["comm"]] = [
        *_rel(ux, uy, rp[0] if rp else None, rp[1] if rp else None, dmax),
        *_rel(ux, uy, rc[0] if rc else None, rc[1] if rc else None, dmax),
        float(getattr(env, "coverage_rate", 0.0)),
        min(float(getattr(env, "cur_max_AoI", 0)) / 50.0, 1.0),
    ]

    # ---------------- energy (4) ----------------
    obs[OBS_SLICES["energy"]] = [
        *_rel(ux, uy, st[0] if st else None, st[1] if st else None, dmax),
        float(np.clip(return_margin, -1.0, 1.0)),
    ]

    # ---------------- team (7) ----------------
    mp, mp_d, mp_j = None, float("inf"), None
    for j, u2 in enumerate(env.uavs):
        if j == idx:
            continue
        d = math.hypot(float(u2.x) - ux, float(u2.y) - uy)
        if d < mp_d:
            mp_d, mp, mp_j = d, (float(u2.x), float(u2.y)), j
    mate_task = np.zeros(N_TASKS, dtype=np.float32)
    if mp_j is not None:
        mt = int(getattr(env.uavs[mp_j], "option", 0))
        if 0 <= mt < N_TASKS:
            mate_task[mt] = 1.0
    obs[OBS_SLICES["team"]] = [
        *_rel(ux, uy, mp[0] if mp else None, mp[1] if mp else None, dmax),
        *mate_task,
        1.0 if mp_d < 4.0 else 0.0,
    ]

    # ---------------- coord (6) ----------------
    opts = [int(getattr(u2, "option", 0)) for u2 in env.uavs]
    n = max(1, len(opts))
    energies = [float(u2.e) for u2 in env.uavs]
    rank = sum(1 for e2 in energies if e2 < float(uav.e)) / float(n)
    relay_idx = [j for j, o in enumerate(opts) if o == TASK_COMM_RELAY]
    is_weakest_relay = (1.0 if (task == TASK_COMM_RELAY and relay_idx
                                and idx == min(relay_idx, key=lambda j: energies[j]))
                        else 0.0)
    obs[OBS_SLICES["coord"]] = [
        opts.count(TASK_FIRE_SENSING) / n, opts.count(TASK_COMM_RELAY) / n,
        opts.count(TASK_RECHARGE) / n,
        rank, min(energies) / e_max, is_weakest_relay,
    ]

    # ---------------- global (6) ----------------
    # 风向必须进观测：每回合随机风向后，火场往哪边长是策略必须知道的，
    # 否则这就是不可观测的噪声
    wd = float(getattr(env, "wind_direction", 0.0))
    obs[OBS_SLICES["global"]] = [
        c["explored_rate"],
        float(getattr(env, "fire_discovery_rate", 0.0)),
        float(getattr(env, "burned_area_rate", 0.0)),
        float(getattr(env, "rescuer_known_fire_rate", 0.0)),
        float(np.clip(float(getattr(env, "fire_diff", 0.0)) / 20.0, -1.0, 1.0)),
        float(step_t) / max(1, max_steps),
        math.cos(wd), math.sin(wd),
    ]
    return obs

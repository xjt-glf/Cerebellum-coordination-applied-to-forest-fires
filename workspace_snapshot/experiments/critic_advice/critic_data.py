# -*- coding: utf-8 -*-
"""EACC 数据层：决策上下文 v2、机制指标、参数化编辑空间、模型特征编码。

一轮（screen40）的经验融入设计：
- 编辑上下文必须取在 H 计划 set_plan **之后**（否则编辑对象不是 H 计划）。
- 机制标签用配对差（分支 B − 分支 A），消除态势基线，聚焦计划效应。
- ctx 里补上 10x10 认知粗视图（一轮缺失的关键特征；全部可观测）。
- 编辑空间参数化且确定性（exec/episode/闭环共用同一候选生成器 → 数据与部署一致）。
"""
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lowlevel.obs_builder import shared_cache, resolve_order_target, assigned_station
from lowlevel.task_spec import (
    UavOrder, TASK_FIRE_SENSING, TASK_COMM_RELAY, TASK_RECHARGE,
    SENSE_EXPLORE, SENSE_PATROL,
)
from harness import task_metrics

COARSE = 10
MAX_SLOTS = 8


# ============================================================ 决策上下文 v2
def ctx_v2(env):
    """重规划边界的完整可观测上下文（含认知粗视图）。全部部署期可知。"""
    from lowlevel.mission_demo import sense_state
    e = env.env
    c = shared_cache(e)
    s = sense_state(env)
    H, W = e.grid_size
    orders = []
    for i, o in enumerate(env.orders):
        tgt, is_fire, ring = resolve_order_target(e, i, o, c)
        u = e.uavs[i]
        orders.append(dict(
            i=i, task=int(o.task), mode=int(o.mode),
            target=(None if o.target is None else [float(o.target[0]), float(o.target[1])]),
            rescuer_id=o.rescuer_id, ring_radius=o.ring_radius,
            handover=bool(o.proactive_handover),
            resolved_target=(None if tgt is None else [float(tgt[0]), float(tgt[1])]),
            target_is_fire=bool(is_fire),
            ring=(float(ring) if ring is not None else None),
            pos=[float(u.x), float(u.y)], e=float(u.e),
            margin=float(u.e) - 2.5 * assigned_station(e, i)[1],
        ))
    return dict(
        t=s["t"], H=H, W=W,
        explored=s["explored"], discovery=s["discovery"], coverage=s["coverage"],
        rescuer_known=s["rescuer_known"], burned=s["burned"], max_aoi=s["max_aoi"],
        n_uavs=len(e.uavs), n_rescuers=len(e.rescuers),
        coarse=[float(v) for v in np.asarray(c["coarse"], dtype=np.float32)],
        clusters=[[float(cl["centroid"][0]), float(cl["centroid"][1]), float(cl["size"])]
                  for cl in c["clusters"]],
        rescuers=[[int(r.rescuer_id), float(r.x), float(r.y),
                   int(bool(r.is_covered)), int(r.AoI)] for r in e.rescuers],
        uavs=[[int(i), float(u.x), float(u.y), float(u.e), orders[i]["margin"]]
              for i, u in enumerate(e.uavs)],
        orders=orders,
        base_pos=[float(e.base_pos[0]), float(e.base_pos[1])],
    )


# ============================================================ 机制指标
def mech_metrics(env):
    """段末机制向量（标签用）。per-UAV 槽位固定 MAX_SLOTS。"""
    e = env.env
    m = task_metrics(env)
    n_r = len(e.rescuers)
    m["covered_count"] = float(getattr(e, "coverage_rate", 0.0)) * n_r
    m["high_aoi_count"] = float(sum(1 for r in e.rescuers if r.AoI > 30))
    flat = []
    for i in range(MAX_SLOTS):
        if i < len(e.uavs):
            u = e.uavs[i]
            flat += [float(u.x), float(u.y), float(u.e)]
        else:
            flat += [0.0, 0.0, 0.0]
    m["uav_flat"] = flat
    return m


MECH_KEYS = ("covered_count", "high_aoi_count", "mean_aoi", "max_aoi",
              "rknown", "active_rknown", "discovery", "active_discovery",
              "explored", "burned_cells", "extinguished_cells", "contained",
              "burning_cells", "active_clusters", "depleted")
MECH_DIM = len(MECH_KEYS) + 3 * MAX_SLOTS     # + per-UAV (x, y, e)


def mech_vec(m):
    v = [float(m[k]) for k in MECH_KEYS]
    v += [float(x) for x in m["uav_flat"]]
    return v


EP_KEYS = ("extinguished_cells", "burned_cells", "contained", "rknown_mean",
           "cov_mean", "explored", "depleted")


# ============================================================ 参数化编辑空间
def _uav_feat(o):
    return UavOrder(task=int(o["task"]), mode=int(o["mode"]),
                    target=(None if o["target"] is None else tuple(o["target"])),
                    ring_radius=o["ring_radius"], rescuer_id=o["rescuer_id"],
                    proactive_handover=bool(o["handover"]))


def orders_from_ctx(ctx):
    return [_uav_feat(o) for o in ctx["orders"]]


def orders_dicts(env, orders):
    """List[UavOrder] -> ctx["orders"] 同构 dict（含解析目标），供特征编码与存档。"""
    e = env.env
    c = shared_cache(e)
    out = []
    for i, o in enumerate(orders):
        tgt, is_fire, ring = resolve_order_target(e, i, o, c)
        out.append(dict(
            i=i, task=int(o.task), mode=int(o.mode),
            target=(None if o.target is None else [float(o.target[0]), float(o.target[1])]),
            rescuer_id=o.rescuer_id, ring_radius=o.ring_radius,
            handover=bool(o.proactive_handover),
            resolved_target=(None if tgt is None else [float(tgt[0]), float(tgt[1])]),
            target_is_fire=bool(is_fire),
            ring=(float(ring) if ring is not None else None),
        ))
    return out


def candidate_edits(ctx, max_edits=9, coarse_blocks=None):
    """从 H 已下发计划出发的确定性参数化编辑候选。

    返回 [(name, params, new_orders(List[UavOrder]), evidence_str)]。
    候选按优先级排序截断到 max_edits-1，最后附 noop（原计划）。
    """
    s = ctx
    orders = ctx["orders"]
    n = len(orders)
    out = []
    ev = []

    def mk(changed):
        """changed: {i: UavOrder} -> (new_orders, changed_idx)"""
        ns = list(orders_from_ctx(ctx))
        for i, o in changed.items():
            ns[i] = o
        return ns

    ups = {o["i"]: o for o in orders}    # ctx orders 自带 pos/i，别用 ctx["uavs"]（list 格式）

    # ---- 1. relay_assign：未覆盖且无人认领的人员（AoI 降序），最近非中继机接手
    claimed = {o["rescuer_id"] for o in orders if o["task"] == 1 and o["rescuer_id"] is not None}
    unc = [r for r in s["rescuers"] if not r[3] and r[0] not in claimed]
    unc.sort(key=lambda r: (-r[4], r[0]))
    for r in unc:
        free = [u for u in orders if u["task"] != 1]
        if not free:
            break
        sub = min(free, key=lambda u: abs(ups[u["i"]]["pos"][0] - r[1])
                  + abs(ups[u["i"]]["pos"][1] - r[2]))
        no = UavOrder(task=TASK_COMM_RELAY, rescuer_id=int(r[0]), proactive_handover=True)
        out.append(("relay_assign", dict(uav=sub["i"], rescuer=int(r[0])),
                    mk({sub["i"]: no}),
                    f"人员#{r[0]}未覆盖且无人认领 AoI={r[4]}，UAV{sub['i']}就近中继"))

    # ---- 2. charge_now：余量<150 的非充电机返航（充电时机提前）
    low = sorted([u for u in orders if u["task"] != 2],
                 key=lambda u: ups[u["i"]]["margin"])
    for u in low:
        if ups[u["i"]]["margin"] >= 150:
            break
        no = UavOrder(task=TASK_RECHARGE)
        out.append(("charge_now", dict(uav=u["i"]),
                    mk({u["i"]: no}),
                    f"UAV{u['i']} margin={ups[u['i']]['margin']:.0f}<150 未充电，提前返航"))

    # ---- 3. explore_block：感知机改搜未知比例最高的粗块（≠当前目标块）
    if coarse_blocks is None:
        coarse_blocks = _unknown_blocks(ctx)
    sensing = [u for u in orders if u["task"] == 0]
    for u in sensing[:2]:
        cur = u["resolved_target"]
        cands = [b for b in coarse_blocks[:4]
                 if cur is None
                 or abs(b[0] - cur[0]) + abs(b[1] - cur[1]) > 12]
        if not cands:
            continue
        b = cands[0]
        no = UavOrder(task=TASK_FIRE_SENSING, mode=SENSE_EXPLORE, target=(b[0], b[1]))
        out.append(("explore_block", dict(uav=u["i"], block=[b[0], b[1]]),
                    mk({u["i"]: no}),
                    f"UAV{u['i']} 搜寻改指向未知块({b[0]:.0f},{b[1]:.0f})"))

    # ---- 4. patrol_bind：感知机改绑巡逻数最少的已知火点
    cl = s["clusters"]
    if cl:
        cnt = [0] * len(cl)
        for u in orders:
            if u["task"] == 0 and u["mode"] == 1 and u["resolved_target"] is not None:
                for ci, c in enumerate(cl):
                    if abs(c[0] - u["resolved_target"][0]) < 4 and abs(c[1] - u["resolved_target"][1]) < 4:
                        cnt[ci] += 1
        ci = int(np.argmin(cnt))
        u = sensing[0] if sensing else None
        if u is not None:
            no = UavOrder(task=TASK_FIRE_SENSING, mode=SENSE_PATROL,
                          target=(cl[ci][0], cl[ci][1]), ring_radius=None)
            out.append(("patrol_bind", dict(uav=u["i"], cluster=ci),
                        mk({u["i"]: no}),
                        f"UAV{u['i']} 改绑火点{ci}({cl[ci][0]:.0f},{cl[ci][1]:.0f})，当前无人巡逻"))

    # ---- 5. release_relay：无覆盖缺口时释放中继转搜寻（占用-覆盖权衡）
    if not unc:
        for u in orders:
            if u["task"] == 1:
                no = UavOrder(task=TASK_FIRE_SENSING, mode=SENSE_EXPLORE)
                out.append(("release_relay", dict(uav=u["i"]),
                            mk({u["i"]: no}),
                            f"无未认领覆盖缺口，释放 UAV{u['i']} 中继转搜寻"))
                break

    # ---- 6. swap_roles：中继机与最近感知机交换角色
    relays = [u for u in orders if u["task"] == 1]
    sensing2 = [u for u in orders if u["task"] == 0]
    if relays and sensing2:
        ru = relays[0]
        su = min(sensing2, key=lambda u: abs(ups[u["i"]]["pos"][0] - ups[ru["i"]]["pos"][0])
                 + abs(ups[u["i"]]["pos"][1] - ups[ru["i"]]["pos"][1]))
        ns = list(orders_from_ctx(ctx))
        ns[ru["i"]], ns[su["i"]] = ns[su["i"]], ns[ru["i"]]
        out.append(("swap_roles", dict(uav_a=ru["i"], uav_b=su["i"]), ns,
                    f"UAV{ru['i']}(中继) 与 UAV{su['i']}(感知) 交换角色"))

    # ---- 7. handover_off：唯一中继机关闭交接
    if len(relays) == 1 and relays[0]["handover"]:
        no = UavOrder(task=TASK_COMM_RELAY,
                      rescuer_id=relays[0]["rescuer_id"],
                      proactive_handover=False)
        out.append(("handover_off", dict(uav=relays[0]["i"]),
                    mk({relays[0]["i"]: no}),
                    f"UAV{relays[0]['i']} 是唯一中继，关闭提前交接"))

    out = out[:max_edits - 1]
    out.append(("noop", {}, list(orders_from_ctx(ctx)), "原计划（不修改）"))
    return out


def _unknown_blocks(ctx):
    """认知粗视图里未知比例最高的块中心（降序）。"""
    c = np.asarray(ctx["coarse"], dtype=np.float32)
    known = c[:COARSE * COARSE].reshape(COARSE, COARSE)
    fire = c[COARSE * COARSE:].reshape(COARSE, COARSE)
    H, W = ctx["H"], ctx["W"]
    bh, bw = H // COARSE, W // COARSE
    score = (1.0 - known) + 0.5 * fire          # 未知为主，带火优先
    idx = np.argsort(score.ravel())[::-1]
    return [(float(i // COARSE * bh + bh / 2), float((i % COARSE) * bw + bw / 2))
            for i in idx[:6]]


# ============================================================ 模型特征
def build_features(ctx, base_orders, new_orders):
    """(ctx, 计划A, 计划B) -> 固定维特征向量。全部部署期可观测。

    计划 A/B 都是 per-UAV dict（ctx["orders"] 格式）或 UavOrder dict 化格式。
    """
    H, W = ctx["H"], ctx["W"]
    dmax = math.hypot(H, W)
    f = [ctx["t"] / 600.0, ctx["explored"], ctx["discovery"], ctx["coverage"],
         ctx["rescuer_known"], ctx["burned"] * 50.0, min(ctx["max_aoi"], 150) / 150.0,
         len(ctx["clusters"]) / 6.0,
         ctx["n_uavs"] / 8.0, ctx["n_rescuers"] / 20.0]
    ta = [0.0] * 3
    tb = [0.0] * 3
    for o in base_orders:
        ta[int(o["task"])] += 1
    for o in new_orders:
        tb[int(o["task"])] += 1
    f += [x / max(1, len(base_orders)) for x in ta]
    f += [x / max(1, len(new_orders)) for x in tb]
    f += [float(x) for x in np.asarray(ctx["coarse"], dtype=np.float32)]

    def uav_block(o, pos, e, margin):
        b = [pos[0] / H, pos[1] / W, e / 600.0, max(-1.0, min(1.0, margin / 200.0))]
        t = [0.0, 0.0, 0.0]
        t[int(o["task"])] = 1.0
        b += t + [float(o["mode"])]
        tgt = o.get("resolved_target") or o.get("target")
        if tgt is not None:
            dx, dy = tgt[0] - pos[0], tgt[1] - pos[1]
            d = math.hypot(dx, dy)
            b += [dx / dmax, dy / dmax, min(d / dmax, 1.0)]
        else:
            b += [0.0, 0.0, 1.0]
        b += [1.0 if o.get("target") is not None or o.get("rescuer_id") is not None else 0.0,
              1.0 if o.get("target_is_fire") else 0.0,
              (float(o["ring"]) if o.get("ring") is not None else 6.0) / 20.0,
              1.0 if o.get("handover") else 0.0]
        return b

    ups = {u[0]: (u[1], u[2], u[3], u[4]) for u in ctx["uavs"]}
    pa, pb = [], []
    UAV_BLK = 15
    for i in range(MAX_SLOTS):
        if i < len(base_orders):
            x, y, e, mg = ups[i]
            pa += uav_block(base_orders[i], [x, y], e, mg)
            pb += uav_block(new_orders[i], [x, y], e, mg)
        else:
            pa += [0.0] * UAV_BLK
            pb += [0.0] * UAV_BLK
    f += pa + pb

    # 人员集合编码（mean/max 池化，per-rescuer MLP 在模型侧做）
    # 这里只输出原始 per-rescuer 特征矩阵，定长截断/补齐到 20
    rr = []
    claimed = {o["rescuer_id"] for o in new_orders
               if int(o["task"]) == 1 and o.get("rescuer_id") is not None}
    for r in ctx["rescuers"][:20]:
        rr += [r[1] / H, r[2] / W, 1.0 if r[3] else 0.0, min(r[4], 150) / 150.0,
               1.0 if r[0] in claimed else 0.0]
    rr += [0.0] * 5 * (20 - len(rr) // 5)
    f += rr
    # 火点簇集合（截断/补齐到 8）
    cc = []
    for c in ctx["clusters"][:8]:
        cc += [c[0] / H, c[1] / W, min(c[2], 400) / 400.0]
    cc += [0.0] * 3 * (8 - len(cc) // 3)
    f += cc
    return np.asarray(f, dtype=np.float32)


FEATURE_DIM = 10 + 6 + 200 + 2 * MAX_SLOTS * 15 + 20 * 5 + 8 * 3   # = 580

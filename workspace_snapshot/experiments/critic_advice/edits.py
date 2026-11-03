# -*- coding: utf-8 -*-
"""指令级局部修改生成器（阶段 B：不接 LLM，检验修改空间里是否存在可检出的真实收益）。

每个生成器输入"可观测证据"（decision_context 的输出 + 已解析目标），输出
(编辑名, 受影响 uav_ids, 新 orders, 依据描述)。全部只用策略可知信息。

编辑目录分三类：
  rule_*  : 有明确证据谓词的修复编辑（后续学习型 Critic 要超越的"规则基线"）
  ctrl_*  : 同格式但内容随机的对照编辑（"随机修改"基线）
  noop    : 原计划重发（哈尼斯确定性检查用）
"""
import math
import random as _random

from harness import orders_to_json
from lowlevel.task_spec import (
    UavOrder, TASK_FIRE_SENSING, TASK_COMM_RELAY, TASK_RECHARGE,
    SENSE_EXPLORE, SENSE_PATROL,
)


def _nearest_rescuer(uav, rescuers, prefer_uncovered=True):
    pool = [r for r in rescuers if not r["covered"]] if prefer_uncovered else []
    pool = pool or list(rescuers)
    if not pool:
        return None
    return min(pool, key=lambda r: abs(r["pos"][0] - uav["pos"][0])
               + abs(r["pos"][1] - uav["pos"][1]))


def _unknown_block_center(sense, exclude, rng=None):
    """从 10x10 粗视图语义近似：在态势里没有粗视图，用 clusters/rescuers 之外的
    空间划分。这里返回一个远离 exclude 点的候选探索目标（象限中心采样）。"""
    H, W = sense["H"], sense["W"]
    quads = [(0.25, 0.25), (0.25, 0.75), (0.75, 0.75), (0.75, 0.25)]
    best, bd = None, -1.0
    for qx, qy in quads:
        cx, cy = qx * H, qy * W
        d = min((math.hypot(cx - ex, cy - ey) for ex, ey in exclude), default=999)
        if d > bd:
            best, bd = (cx, cy), d
    return best


def applicable_edits(ctx, rng):
    """返回当前决策点上适用的 (edit_name, uav_ids, new_orders, evidence_str) 列表。

    ctx: decision_context() 的输出（sense + orders 已解析信息）。
    只依赖可观测信息；每类编辑至多给一个实例，保持每次决策的分支数可控。
    """
    sense = ctx["sense"]
    orders = [o for o in ctx["orders"]]
    out = []

    def replace(i, new_order, ev):
        ns = [UavOrder(**_kw(o)) for o in orders]
        ns[i] = new_order
        return ns, ev

    def _kw(o):
        return dict(task=int(o["task"]), mode=int(o["mode"]),
                    target=(None if o["target"] is None else tuple(o["target"])),
                    ring_radius=o["ring_radius"], rescuer_id=o["rescuer_id"],
                    proactive_handover=bool(o["handover"]))

    n = len(orders)
    uav_by_id = {u["i"]: u for u in orders}

    # ---- rule_charge_lowest：返航余量最低且未在补给的非充电机 → charge
    cand = [u for u in orders if u["task"] != 2]
    if cand:
        u = min(cand, key=lambda z: sense["uavs"][z["i"]]["margin"])
        if sense["uavs"][u["i"]]["margin"] < 120.0:   # 证据谓词：余量偏低
            ns, ev = replace(u["i"], UavOrder(task=TASK_RECHARGE),
                             f"UAV{u['i']} margin={sense['uavs'][u['i']]['margin']:.0f} 未充电")
            out.append(("rule_charge_lowest", [u["i"]], ns, ev))

    # ---- rule_relay_return_fix：中继机返航余量为负 → 派去补给，且由最近空闲机接替
    relay_ids = [u["i"] for u in orders if u["task"] == 1]
    for ri in relay_ids:
        if sense["uavs"][ri]["margin"] < 0:
            r_target = None
            if orders[ri]["rescuer_id"] is not None:
                rr = [x for x in sense["rescuers"] if x["id"] == orders[ri]["rescuer_id"]]
                r_target = rr[0] if rr else None
            others = [u for u in orders if u["task"] not in (1, 2)]
            if others:
                sub = min(others, key=lambda z: math.hypot(
                    sense["uavs"][z["i"]]["pos"][0] - (r_target["pos"][0] if r_target else 0),
                    sense["uavs"][z["i"]]["pos"][1] - (r_target["pos"][1] if r_target else 0)))
                ns = [UavOrder(**_kw(o)) for o in orders]
                ns[ri] = UavOrder(task=TASK_RECHARGE)
                ns[sub["i"]] = UavOrder(task=TASK_COMM_RELAY,
                                        rescuer_id=(r_target["id"] if r_target else None),
                                        proactive_handover=True)
                out.append(("rule_relay_return_fix", [ri, sub["i"]], ns,
                            f"UAV{ri} 中继但 margin={sense['uavs'][ri]['margin']:.0f}<0；"
                            f"UAV{sub['i']} 接替"))
            break

    # ---- rule_relay_uncovered：存在"未覆盖且无人在管"的人员且中继名额未满 → 最近机去中继
    # 注意谓词：is_covered=False 不代表没人管——在途的认领（rescuer_id 已被某条
    # relay 指令认领）不算覆盖缺口，否则会对同一人员重复派单。
    claimed = {u["rescuer_id"] for u in orders
               if u["task"] == 1 and u["rescuer_id"] is not None}
    uncovered = [r for r in sense["rescuers"]
                 if not r["covered"] and r["id"] not in claimed]
    if uncovered and len(relay_ids) < 2:
        free = [u for u in orders if u["task"] == 0]
        if free:
            tr = max(uncovered, key=lambda r: r["aoi"])
            sub = min(free, key=lambda z: math.hypot(
                sense["uavs"][z["i"]]["pos"][0] - tr["pos"][0],
                sense["uavs"][z["i"]]["pos"][1] - tr["pos"][1]))
            ns, ev = replace(sub["i"], UavOrder(task=TASK_COMM_RELAY,
                                                rescuer_id=tr["id"],
                                                proactive_handover=True),
                             f"人员#{tr['id']} 未覆盖 AoI={tr['aoi']}，UAV{sub['i']} 就近中继")
            out.append(("rule_relay_uncovered", [sub["i"]], ns, ev))

    # ---- rule_redundant_relay：无覆盖缺口且中继≥2 时，释放 AoI 最低的中继机转探索
    if not uncovered and len(relay_ids) >= 2 and sense["coverage"] > 0.9:
        relays = [u for u in orders if u["task"] == 1]
        aoi_of = {}
        for u in relays:
            aoi_of[u["i"]] = 0
            if u["rescuer_id"] is not None:
                for rr in sense["rescuers"]:
                    if rr["id"] == u["rescuer_id"]:
                        aoi_of[u["i"]] = rr["aoi"]
        ri = min(relays, key=lambda u: aoi_of[u["i"]])["i"]
        ns = [UavOrder(**_kw(o)) for o in orders]
        ns[ri] = UavOrder(task=TASK_FIRE_SENSING, mode=SENSE_EXPLORE)
        out.append(("rule_redundant_relay", [ri], ns,
                    f"无覆盖缺口 coverage={sense['coverage']:.2f} 中继×{len(relays)}，"
                    f"UAV{ri}(AoI={aoi_of[ri]}) 转探索"))

    # ---- rule_patrol_dedupe：两机巡逻同一火点 → 其一改绑最少巡逻的火点
    patrol_targets = {}
    for u in orders:
        if u["task"] == 0 and u["mode"] == 1 and u["resolved_target"] is not None:
            key = (round(u["resolved_target"][0]), round(u["resolved_target"][1]))
            patrol_targets.setdefault(key, []).append(u)
    dup = [(k, v) for k, v in patrol_targets.items() if len(v) >= 2]
    if dup and len(sense["clusters"]) >= 2:
        key, group = dup[0]
        counts = {i: 0 for i in range(len(sense["clusters"]))}
        for u in orders:
            if u["task"] == 0 and u["mode"] == 1 and u["resolved_target"] is not None:
                for ci, cl in enumerate(sense["clusters"]):
                    if (abs(cl["centroid"][0] - u["resolved_target"][0]) < 3
                            and abs(cl["centroid"][1] - u["resolved_target"][1]) < 3):
                        counts[ci] += 1
        tgt_cl = min(sense["clusters"], key=lambda cl: counts[sense["clusters"].index(cl)])
        mover = group[-1]
        ns = [UavOrder(**_kw(o)) for o in orders]
        ns[mover["i"]] = UavOrder(task=TASK_FIRE_SENSING, mode=SENSE_PATROL,
                                  target=tuple(tgt_cl["centroid"]), ring_radius=None)
        out.append(("rule_patrol_dedupe", [mover["i"]], ns,
                    f"UAV{mover['i']} 与他机同巡逻 {key}，改绑 {tgt_cl['centroid']}"))

    # ---- rule_explore_redirect：探索机的目标已在已探索区 → 指向远离各机目标的象限
    explorers = [u for u in orders if u["task"] == 0 and u["mode"] == 0]
    if sense["explored"] < 0.95 and explorers:
        exclude = [tuple(u["resolved_target"]) for u in orders
                   if u["resolved_target"] is not None]
        tgt = _unknown_block_center(sense, exclude)
        u = explorers[0]
        ns, ev = replace(u["i"], UavOrder(task=TASK_FIRE_SENSING, mode=SENSE_EXPLORE,
                                          target=tgt),
                         f"UAV{u['i']} 搜寻目标重定向到 {tgt}")
        out.append(("rule_explore_redirect", [u["i"]], ns, ev))

    # ---- rule_ring_auto：巡逻机显式环半径偏离自动值 → 改 auto
    for u in orders:
        if u["task"] == 0 and u["mode"] == 1 and u["ring_radius"] is not None:
            for cl in sense["clusters"]:
                if (abs(cl["centroid"][0] - (u["resolved_target"] or (9e9, 9e9))[0]) < 3
                        and abs(cl["centroid"][1] - (u["resolved_target"] or (9e9, 9e9))[1]) < 3):
                    auto = math.sqrt(max(1.0, cl["size"]) / math.pi) + 2.0
                    if abs(u["ring_radius"] - auto) > 2.0:
                        ns = [UavOrder(**_kw(o)) for o in orders]
                        ns[u["i"]] = UavOrder(task=TASK_FIRE_SENSING, mode=SENSE_PATROL,
                                              target=tuple(u["target"] or cl["centroid"]),
                                              ring_radius=None)
                        out.append(("rule_ring_auto", [u["i"]], ns,
                                    f"UAV{u['i']} ring={u['ring_radius']:.0f} 偏离 auto={auto:.1f}"))
                    break

    # ---- ctrl_random_edit：同格式随机编辑（对照）
    for _ in range(2):
        i = rng.randrange(n)
        t = rng.choice([TASK_FIRE_SENSING, TASK_COMM_RELAY, TASK_RECHARGE])
        o = UavOrder(task=t)
        ev = f"UAV{i} 随机改为 {['sense', 'relay', 'charge'][t]}"
        if t == TASK_FIRE_SENSING:
            o.mode = rng.choice([SENSE_EXPLORE, SENSE_PATROL])
            if rng.random() < 0.7:
                if o.mode == SENSE_PATROL and sense["clusters"]:
                    cl = rng.choice(sense["clusters"])
                    o.target = tuple(cl["centroid"])
                else:
                    o.target = (float(rng.randrange(sense["H"])),
                                float(rng.randrange(sense["W"])))
                ev += f" target={o.target}"
        elif t == TASK_COMM_RELAY and sense["rescuers"]:
            r = rng.choice(sense["rescuers"])
            o.rescuer_id = int(r["id"])
            ev += f" rescuer={r['id']}"
        ns, ev2 = replace(i, o, ev)
        out.append((f"ctrl_random_edit", [i], ns, ev))

    return out

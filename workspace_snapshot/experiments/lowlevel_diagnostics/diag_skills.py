# -*- coding: utf-8 -*-
"""底层 Actor 指令服从诊断（受控实验）。

设计（对应诊断任务书 §五/§六/§七）：
- 起点：H 协议跑到 t0 ∈ {0, 200} 的快照（真实态势），测试窗口 W=60 步，
  **计划固定不重规划**（隔离"服从这一条指令"）。
- 每个测试只改一架 UAV 的指令，其余保持 H 计划（其余机的反应 = 团队耦合测量）。
- 分层：regime（normal/scarce）× seed × t0 × 受测 UAV 编号。
- 对照：贪心走位控制器给出 W 步内的物理可达进度上界（参考，非 Actor 效果）。

指标（按任务）：
- explore: 距目标进度、最小距离、向目标移动步占比、目标区（5格内）到访步数、
  新探索格数（区分"探索了"与"探索了指定区域"）
- patrol: 后 40 步 |d−ring| 均值 / 最小值、刷新格数
- relay: 指定人员被覆盖步数、其 AoI 终值、任意人员覆盖（区分"指定 vs 任何人"）；
  rescuer_id 与 target 两种接口
- charge: 到站时间、是否充电、余量
- 字段受控比较：同一快照只改一个字段 → 首步概率 L1 / argmax 翻转 / 轨迹分叉 /
  其他机首步动作变化（耦合通道）
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "critic_advice"))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from harness import BENCH, Snapshot, make_env, load_actor, make_h_brain, \
    run_protocol, task_metrics
from lowlevel.obs_builder import shared_cache, resolve_order_target
from lowlevel.task_spec import (
    UavOrder, TASK_FIRE_SENSING, TASK_COMM_RELAY, TASK_RECHARGE,
    SENSE_EXPLORE, SENSE_PATROL,
)
from lowlevel.mission_demo import brain as h_brain

W = 60
T0S = (0, 200)
REGIMES = {"normal": dict(), "scarce": dict(uavs=3, rescuers=14)}


def regime_cfg(regime):
    c = dict(BENCH)
    c.update(REGIMES[regime])
    return c


def greedy_bound():
    """W 步贪心直行的物理进度上界（斜向 1.41/步）。"""
    return W * 1.41


def fixed_plan_rollout(snapshot, orders, actor, regime_note=None):
    """从快照出发：set_plan 一次，W 步不重规划。返回逐步记录。"""
    import torch
    env = snapshot.restore()
    obs = env.set_plan(list(orders))
    e = env.env
    recs = []
    for k in range(W):
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32)
            probs = torch.softmax(actor(o), dim=-1).numpy()
            a = probs.argmax(axis=-1)
        obs, _r, done, info = env.step(a.tolist())
        recs.append(dict(t=env.t, explored=float(getattr(e, "explored_rate", 0.0)),
                         uav_xy=[(float(u.x), float(u.y)) for u in e.uavs],
                         e=[float(u.e) for u in e.uavs],
                         cover=[bool(r.is_covered) for r in e.rescuers],
                         aoi=[int(r.AoI) for r in e.rescuers],
                         recharged=[bool(getattr(u, "recharged", False)) for u in e.uavs]))
        if done:
            break
    return recs, env


def analyze_explore(recs, i, target, m0):
    ds = [math.hypot(x - target[0], y - target[1])
          for r in recs for x, y in [r["uav_xy"][i]]]
    toward = 0
    prev = math.hypot(recs[0]["uav_xy"][i][0] - target[0],
                      recs[0]["uav_xy"][i][1] - target[1]) if recs else None
    for r in recs[1:]:
        d = math.hypot(r["uav_xy"][i][0] - target[0], r["uav_xy"][i][1] - target[1])
        if d < prev - 0.1:
            toward += 1
        prev = d
    near = sum(1 for d in ds if d < 5.0)
    return dict(d_init=ds[0] if ds else None, d_final=ds[-1] if ds else None,
                d_min=min(ds) if ds else None, progress=(ds[0] - ds[-1]) if ds else None,
                toward_frac=toward / max(1, len(recs) - 1), near_steps=near)


def analyze_patrol(recs, i, centroid, ring):
    tail = recs[-40:]
    errs = [abs(math.hypot(r["uav_xy"][i][0] - centroid[0],
                           r["uav_xy"][i][1] - centroid[1]) - ring) for r in tail]
    return dict(ring_err_mean=float(np.mean(errs)), ring_err_min=float(min(errs)),
                ring_within2=sum(1 for e in errs if e < 2.0) / len(errs))


def analyze_relay(recs, i, rid, n_resc):
    idx = rid - 1
    cov = sum(1 for r in recs if r["cover"][idx])
    any_cov = sum(1 for r in recs if any(r["cover"]))
    return dict(spec_cov_frac=cov / len(recs), any_cov_frac=any_cov / len(recs),
                spec_aoi_final=recs[-1]["aoi"][idx])


def analyze_charge(recs, i):
    reach_t = None
    for r in recs:
        if r["recharged"][i]:
            reach_t = r["t"]
            break
    return dict(recharged=bool(reach_t), reach_t=reach_t,
                e_init=recs[0]["e"][i], e_final=recs[-1]["e"][i])


def first_step_probs(snapshot, orders, actor):
    """快照状态下给定计划的首步动作概率（不执行）。"""
    import torch
    env = snapshot.restore()
    obs = env.set_plan(list(orders))
    with torch.no_grad():
        p = torch.softmax(actor(torch.as_tensor(obs, dtype=torch.float32)), -1).numpy()
    return p


def collect_seed(seed, regime):
    """一个种子的全部诊断测试。"""
    import torch
    torch.set_num_threads(1)
    cfg = regime_cfg(regime)
    env = make_env(seed, cfg)
    env.reset()
    actor = load_actor(BENCH["actor"])
    brain = make_h_brain(cfg)
    replan = cfg["replan_every"]
    e = env.env
    out = []
    for t0 in T0S:
        # ---- H 协议推进到 t0 ----
        obs, done = None, False
        while env.t < t0 and not done:
            if env.t % replan == 0:
                from lowlevel.mission_demo import sense_state
                s = sense_state(env)
                plan, _, _ = brain.plan(s)
                obs = env.set_plan(plan)
            with torch.no_grad():
                a = actor.act(torch.as_tensor(obs, dtype=torch.float32),
                              deterministic=True)[0].numpy()
            obs, _r, done, _i = env.step(a.tolist())
        if done:
            break
        snap = Snapshot(env)
        c = shared_cache(e)
        m0 = task_metrics(env)
        clusters = c["clusters"]
        rescuers = e.rescuers
        n_uav = len(e.uavs)

        # 基线计划 = H 在 t0 边界会下发的计划（set_plan 解析后的 orders）
        from lowlevel.mission_demo import sense_state
        s = sense_state(env)
        plan, _, _ = brain.plan(s)
        env.set_plan(plan)
        base_orders = list(env.orders)

        # 每架受测：改第 i 架
        for i in range(n_uav):
            ups = e.uavs[i]

            # ---- explore：远/近两个目标 ----
            quad_far = max([(25.0, 25.0), (25.0, 75.0), (75.0, 75.0), (75.0, 25.0)],
                           key=lambda q: math.hypot(q[0] - ups.x, q[1] - ups.y))
            quad_near = min([(25.0, 25.0), (25.0, 75.0), (75.0, 75.0), (75.0, 25.0)],
                            key=lambda q: math.hypot(q[0] - ups.x, q[1] - ups.y))
            for tag, tgt in (("explore_far", quad_far), ("explore_near", quad_near)):
                if math.hypot(tgt[0] - ups.x, tgt[1] - ups.y) < 8:
                    continue
                orders = list(base_orders)
                orders[i] = UavOrder(task=TASK_FIRE_SENSING, mode=SENSE_EXPLORE,
                                     target=tgt)
                recs, _ = fixed_plan_rollout(snap, orders, actor)
                rec_an = analyze_explore(recs, i, tgt, m0)
                rec_an["explored_gain"] = (recs[-1]["explored"] - m0["explored"]
                                           if recs else 0.0)
                out.append(dict(regime=regime, seed=seed, t0=t0, uav=i, task=tag,
                                target=list(tgt), **rec_an))

            # ---- patrol：绑一个已知火点，auto 环半径 ----
            if clusters:
                cl = clusters[(i + 1) % len(clusters)]
                ring = math.sqrt(max(1.0, cl["size"]) / math.pi) + 2.0
                orders = list(base_orders)
                orders[i] = UavOrder(task=TASK_FIRE_SENSING, mode=SENSE_PATROL,
                                     target=tuple(cl["centroid"]), ring_radius=None)
                recs, _ = fixed_plan_rollout(snap, orders, actor)
                an = analyze_patrol(recs, i, cl["centroid"], ring)
                out.append(dict(regime=regime, seed=seed, t0=t0, uav=i, task="patrol",
                                target=[float(v) for v in cl["centroid"]],
                                ring=ring, **an))
                # 显式小环半径对照（ring=6）
                orders[i] = UavOrder(task=TASK_FIRE_SENSING, mode=SENSE_PATROL,
                                     target=tuple(cl["centroid"]), ring_radius=6.0)
                recs, _ = fixed_plan_rollout(snap, orders, actor)
                an = analyze_patrol(recs, i, cl["centroid"], 6.0)
                out.append(dict(regime=regime, seed=seed, t0=t0, uav=i,
                                task="patrol_ring6", target=[float(v) for v in cl["centroid"]],
                                ring=6.0, **an))

            # ---- relay：rescuer_id 接口（最近/最远人员） ----
            if rescuers:
                by_d = sorted(rescuers, key=lambda r: math.hypot(r.x - ups.x, r.y - ups.y))
                for tag, rr in (("relay_near", by_d[0]),
                                ("relay_far", by_d[-1] if len(by_d) > 1 else by_d[0])):
                    orders = list(base_orders)
                    orders[i] = UavOrder(task=TASK_COMM_RELAY, rescuer_id=int(rr.rescuer_id))
                    recs, _ = fixed_plan_rollout(snap, orders, actor)
                    an = analyze_relay(recs, i, int(rr.rescuer_id), len(rescuers))
                    an["d_init"] = math.hypot(rr.x - ups.x, rr.y - ups.y)
                    out.append(dict(regime=regime, seed=seed, t0=t0, uav=i, task=tag,
                                    rescuer=int(rr.rescuer_id), **an))
                # ---- relay：target 坐标接口（最远人员的坐标，不带 rescuer_id） ----
                rr = by_d[-1]
                orders = list(base_orders)
                orders[i] = UavOrder(task=TASK_COMM_RELAY, target=(float(rr.x), float(rr.y)))
                recs, _ = fixed_plan_rollout(snap, orders, actor)
                an = analyze_relay(recs, i, int(rr.rescuer_id), len(rescuers))
                an["d_init"] = math.hypot(rr.x - ups.x, rr.y - ups.y)
                out.append(dict(regime=regime, seed=seed, t0=t0, uav=i, task="relay_point",
                                rescuer=int(rr.rescuer_id), **an))

            # ---- charge ----
            orders = list(base_orders)
            orders[i] = UavOrder(task=TASK_RECHARGE)
            recs, _ = fixed_plan_rollout(snap, orders, actor)
            an = analyze_charge(recs, i)
            from lowlevel.obs_builder import assigned_station
            st, st_d = assigned_station(e, i)
            an["d_station"] = st_d
            out.append(dict(regime=regime, seed=seed, t0=t0, uav=i, task="charge", **an))

        # ---- 字段受控比较（§六）：只改 explore 目标 A vs B ----
        i = 0
        quads = [(25.0, 25.0), (75.0, 75.0)]
        pa, pb = list(base_orders), list(base_orders)
        pa[i] = UavOrder(task=TASK_FIRE_SENSING, mode=SENSE_EXPLORE, target=quads[0])
        pb[i] = UavOrder(task=TASK_FIRE_SENSING, mode=SENSE_EXPLORE, target=quads[1])
        pA = first_step_probs(snap, pa, actor)
        pB = first_step_probs(snap, pb, actor)
        l1 = float(np.abs(pA - pB).sum(axis=1).mean())
        flip = int((pA.argmax(1) != pB.argmax(1)).sum())
        recsA, _ = fixed_plan_rollout(snap, pa, actor)
        recsB, _ = fixed_plan_rollout(snap, pb, actor)
        traj_diff = sum(1 for ra, rb in zip(recsA, recsB) if ra["uav_xy"] != rb["uav_xy"])
        other_flip = int((pA.argmax(1) != pB.argmax(1))[1:].sum())
        out.append(dict(regime=regime, seed=seed, t0=t0, uav=i, task="probe_target_AB",
                        prob_l1=l1, argmax_flips=flip, other_uav_flips=other_flip,
                        traj_diff_steps=traj_diff))
        env = snap.restore()   # 回到快照态继续外层 t0 循环（重新推进）
        e = env.env
        obs = None
    return out


def _worker(a):
    seed, regime = a
    try:
        return collect_seed(seed, regime)
    except Exception as ex:
        import traceback
        return [dict(error=f"{type(ex).__name__}: {ex}", seed=seed, regime=regime,
                     trace=traceback.format_exc()[-1000:])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=60000)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    jobs = [(args.seed0 + k, rg) for rg in REGIMES for k in range(args.seeds)]
    with mp.get_context("fork").Pool(min(args.workers, len(jobs))) as pool:
        results = pool.map(_worker, jobs)
    rows = [r for chunk in results for r in chunk if "error" not in r]
    errs = [r for chunk in results for r in chunk if "error" in r]
    for r in errs[:3]:
        print(f"FAIL seed={r['seed']}: {r['error']}\n{r['trace'][:400]}")
    out_path = os.path.join(_HERE, "runs", "skills.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[saved] {out_path}  n={len(rows)} 条（贪心物理上界 {greedy_bound():.0f} 格/{W}步）")

    # ---- 汇总：任务 × regime × 编号 ----
    from collections import defaultdict
    def agg(key_fn, rows, metrics):
        g = defaultdict(list)
        for r in rows:
            g[key_fn(r)].append(r)
        return g

    print("\n=== explore 服从（progress=距离减少量；toward=向目标移动步占比；near=目标5格内步数）===")
    for rg in REGIMES:
        sub = [r for r in rows if r["task"].startswith("explore") and r["regime"] == rg]
        for i in sorted({r["uav"] for r in sub}):
            s = [r for r in sub if r["uav"] == i]
            if not s:
                continue
            print(f"  {rg} UAV{i}: progress {np.mean([r['progress'] for r in s]):6.1f}"
                  f"/{greedy_bound():.0f}  d_min {np.mean([r['d_min'] for r in s]):6.1f}"
                  f"  toward {np.mean([r['toward_frac'] for r in s]):.2f}"
                  f"  near {np.mean([r['near_steps'] for r in s]):5.1f}/{W}"
                  f"  (n={len(s)})")

    print("\n=== patrol 环距（后40步 |d−ring|，越小越服从；within2=±2格内占比）===")
    for rg in REGIMES:
        for task in ("patrol", "patrol_ring6"):
            sub = [r for r in rows if r["task"] == task and r["regime"] == rg]
            if not sub:
                continue
            by_uav = defaultdict(list)
            for r in sub:
                by_uav[r["uav"]].append(r)
            for i, s in sorted(by_uav.items()):
                print(f"  {rg} {task} UAV{i}: |d−ring| {np.mean([r['ring_err_mean'] for r in s]):5.1f}"
                      f"  min {np.mean([r['ring_err_min'] for r in s]):5.1f}"
                      f"  within2 {np.mean([r['ring_within2'] for r in s]):.2f} (n={len(s)})")

    print("\n=== relay 服从（spec=指定人员被覆盖步占比；any=任意人员；越小距离越应覆盖）===")
    for rg in REGIMES:
        for task in ("relay_near", "relay_far", "relay_point"):
            sub = [r for r in rows if r["task"] == task and r["regime"] == rg]
            if not sub:
                continue
            by_uav = defaultdict(list)
            for r in sub:
                by_uav[r["uav"]].append(r)
            for i, s in sorted(by_uav.items()):
                print(f"  {rg} {task} UAV{i}: spec_cov {np.mean([r['spec_cov_frac'] for r in s]):.2f}"
                      f"  any_cov {np.mean([r['any_cov_frac'] for r in s]):.2f}"
                      f"  aoifin {np.mean([r['spec_aoi_final'] for r in s]):5.1f}"
                      f"  d_init {np.mean([r['d_init'] for r in s]):5.1f} (n={len(s)})")

    print("\n=== charge 服从 ===")
    for rg in REGIMES:
        sub = [r for r in rows if r["task"] == "charge" and r["regime"] == rg]
        by_uav = defaultdict(list)
        for r in sub:
            by_uav[r["uav"]].append(r)
        for i, s in sorted(by_uav.items()):
            ok = [r for r in s if r["recharged"]]
            print(f"  {rg} UAV{i}: 到站充电 {len(ok)}/{len(s)}"
                  f"  平均到站步 {np.mean([r['reach_t'] - r['t0'] for r in ok]) if ok else float('nan'):.1f}"
                  f"  距站 {np.mean([r['d_station'] for r in s]):.1f}"
                  f"  e_init {np.mean([r['e_init'] for r in s]):.0f}")

    print("\n=== 字段受控探针（explore 目标 A vs B，只改一个字段）===")
    for rg in REGIMES:
        sub = [r for r in rows if r["task"] == "probe_target_AB" and r["regime"] == rg]
        if sub:
            print(f"  {rg}: 首步概率L1 {np.mean([r['prob_l1'] for r in sub]):.3f}"
                  f"  argmax翻转 {np.mean([r['argmax_flips'] for r in sub]):.2f}机"
                  f"  其他机翻转 {np.mean([r['other_uav_flips'] for r in sub]):.2f}"
                  f"  轨迹分叉步 {np.mean([r['traj_diff_steps'] for r in sub]):.1f}/{W}")


if __name__ == "__main__":
    main()

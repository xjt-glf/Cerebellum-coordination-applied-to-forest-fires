# -*- coding: utf-8 -*-
"""复合策略实验：把"每次决策点都应用某类建议"固化为策略变体，整局配对比较。

单次编辑的价值结构（screen40/screen40_ep）已证明：一次局部修改的任务收益
（±40 格@60 / ±14 格@episode）是瞬态噪声主导、无法从编辑前状态预测。
部署形态的真实问题是**复合**：反馈机制每个规划周期都起作用。所以这里把
候选建议固化为持续策略，与 H 基线同种子整局配对（确定性 actor → 精确配对差）。

P0 = H 基线（charge_at=260, explore_until=0.70, max_chargers=2, 早期 1 中继）
P1 = 始终 2 中继（把 rule_relay_uncovered 的建议应用到每个决策点）
P2 = explore_until=0.85（更久搜寻）
P3 = max_chargers=1 / 3（充电名额敏感性）
P4 = charge_at=200 / 320（充电时机敏感性）

统计：按种子配对差，均值 ± 按种子的 95% CI，n=40。
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

from harness import BENCH, make_env, load_actor, task_metrics
from lowlevel.mission_demo import sense_state

QUADRANTS = [(0.25, 0.25), (0.25, 0.75), (0.75, 0.75), (0.75, 0.25)]


def brain_variant(s, explore_until=0.70, charge_at=260.0, max_chargers=2,
                  n_relay_min=None):
    """lowlevel/mission_demo.py brain() 的参数化副本（加 n_relay_min 覆盖）。

    与原版逐行同构，只把 n_relay 一处做成可覆盖；不改任何现有源码。
    """
    n = len(s["uavs"])
    orders = [None] * n
    charging = []
    for u in sorted(s["uavs"], key=lambda z: z["e"]):
        if u["margin"] < 0:
            orders[u["id"]] = {"task": "charge"}
            charging.append((u["id"], f"返航余量 {u['margin']} 已为负"))
        elif len(charging) < max_chargers and u["e"] < charge_at:
            orders[u["id"]] = {"task": "charge"}
            charging.append((u["id"], f"电量 {u['e']} < {charge_at:.0f}"))
    rest = [u["id"] for u in s["uavs"] if orders[u["id"]] is None]
    if n_relay_min is not None:
        n_relay = max(n_relay_min, 2) if (s["explored"] >= explore_until and len(rest) >= 3) \
            else n_relay_min
    else:
        n_relay = 2 if (s["explored"] >= explore_until and len(rest) >= 3) else 1
    picked = []
    pool = [r for r in s["rescuers"] if not r["covered"]] or list(s["rescuers"])
    for _ in range(min(n_relay, len(rest))):
        cand = [r for r in pool
                if all(abs(r["pos"][0] - q["pos"][0]) + abs(r["pos"][1] - q["pos"][1]) > 20
                       for q in picked)]
        if not cand:
            break
        tr = max(cand, key=lambda r: r["aoi"])
        relay = min(rest, key=lambda i: abs(s["uavs"][i]["pos"][0] - tr["pos"][0])
                    + abs(s["uavs"][i]["pos"][1] - tr["pos"][1]))
        orders[relay] = {"task": "relay", "rescuer_id": tr["id"], "handover": True}
        rest.remove(relay)
        picked.append(tr)
    keep_explorer = s["explored"] < 0.95
    for k, i in enumerate(rest):
        want_explore = (s["explored"] < explore_until or not s["clusters"]
                        or (keep_explorer and k == 0))
        if want_explore:
            qx, qy = QUADRANTS[i % 4]
            orders[i] = {"task": "explore",
                         "target": [round(qx * s["H"], 1), round(qy * s["W"], 1)]}
        else:
            cl = s["clusters"][k % len(s["clusters"])]
            orders[i] = {"task": "patrol", "target": cl["centroid"], "ring_radius": "auto"}
    return {"uavs": [o if o is not None else {"task": "explore"} for o in orders]}


POLICIES = {
    "P0_base": dict(),
    "P1_relay2": dict(n_relay_min=2),
    "P2_explore85": dict(explore_until=0.85),
    "P3a_charge1": dict(max_chargers=1),
    "P3b_charge3": dict(max_chargers=3),
    "P4a_chargeat200": dict(charge_at=200.0),
    "P4b_chargeat320": dict(charge_at=320.0),
}


def run_policy(seed, policy_kw, cfg=None):
    import torch
    torch.set_num_threads(1)
    env = make_env(seed, cfg)
    env.reset()
    actor = load_actor(BENCH["actor"])
    replan = BENCH["replan_every"]
    max_steps = env.max_steps
    obs, cov, rkn, done = None, [], [], False
    while env.t < max_steps and not done:
        if env.t % replan == 0:
            s = sense_state(env)
            plan = brain_variant(s, **policy_kw)
            obs = env.set_plan(plan)
        with torch.no_grad():
            a = actor.act(torch.as_tensor(obs, dtype=torch.float32),
                          deterministic=True)[0].numpy()
        obs, _r, done, info = env.step(a)
        cov.append(float(info.get("coverage_rate", 0.0)))
        m = task_metrics(env)
        rkn.append(m["rknown"])
    m = task_metrics(env)
    m["cov_mean"] = float(np.mean(cov)) if cov else 0.0
    m["rknown_mean"] = float(np.mean(rkn)) if rkn else 0.0
    m["steps"] = env.t
    return m


def _worker(args):
    seed, name, kw, cfg = args
    try:
        return (seed, name, run_policy(seed, kw, cfg))
    except Exception as e:
        import traceback
        return (seed, name, dict(error=f"{type(e).__name__}: {e}",
                                 trace=traceback.format_exc()[-800:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=31000)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--policies", default="", help="逗号分隔的策略子集（默认全部）")
    ap.add_argument("--rescuers", type=int, default=None, help="覆盖基准人员数（压力场景）")
    ap.add_argument("--fires", type=int, default=None, help="覆盖基准火点数（压力场景）")
    ap.add_argument("--uavs", type=int, default=None, help="覆盖基准无人机数")
    ap.add_argument("--out", default=os.path.join(_HERE, "runs", "policy_ep",
                                                  "results.json"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cfg = dict(BENCH)
    if args.rescuers:
        cfg["rescuers"] = args.rescuers
    if args.fires:
        cfg["fires"] = args.fires
    if args.uavs:
        cfg["uavs"] = args.uavs
    seeds = [args.seed0 + i for i in range(args.seeds)]
    policies = POLICIES
    if args.policies:
        keep = set(args.policies.split(",")) | {"P0_base"}
        policies = {k: v for k, v in POLICIES.items() if k in keep}
    import harness as _h
    global BENCH_NOTE
    with open(os.path.join(os.path.dirname(args.out), "config_used.json"),
              "w", encoding="utf-8") as f:
        json.dump(dict(cfg=cfg, seeds=[args.seed0, args.seed0 + args.seeds - 1],
                        policies=list(policies)), f, ensure_ascii=False, indent=2)

    jobs = [(s, name, kw, cfg) for s in seeds for name, kw in policies.items()]
    ctx = mp.get_context("fork")
    with ctx.Pool(min(args.workers, len(jobs))) as pool:
        results = pool.map(_worker, jobs)

    by = {}
    for seed, name, m in results:
        if "error" in m:
            print(f"[FAIL] {name} seed={seed}: {m['error']}\n{m.get('trace','')[:300]}")
            continue
        by.setdefault(name, {})[seed] = m

    KEYS = ("extinguished_cells", "burned_cells", "contained", "cov_mean",
            "rknown_mean", "explored", "depleted", "min_energy", "steps")
    print(f"\n[策略整局配对] n={args.seeds} 种子（基线 P0_base，配对差 = 策略 − P0）")
    print(f"{'策略':<14s} {'Δ扑灭':>9s} {'Δ过火':>8s} {'Δ合围':>7s} {'Δ覆盖均值':>9s} "
          f"{'Δ已知均值':>9s} {'Δ探索':>8s} {'Δ耗尽':>6s} {'Δ最低电':>8s}")
    base = by.get("P0_base", {})
    try:
        from scipy import stats as st
        HAVE_SCIPY = True
    except Exception:
        HAVE_SCIPY = False

    def t_ci(ds):
        if len(ds) <= 1:
            return float("nan")
        if HAVE_SCIPY:
            return float(st.t.ppf(0.975, len(ds) - 1) * ds.std(ddof=1) / np.sqrt(len(ds)))
        return float(2.0227 * ds.std(ddof=1) / np.sqrt(len(ds)))   # t_{0.975, df>=40}

    def t_p(ds):
        if len(ds) <= 1:
            return float("nan")
        if HAVE_SCIPY:
            return float(st.ttest_rel(ds, np.zeros_like(ds)).pvalue)
        return float("nan")
    out_rows = []
    for name in POLICIES:
        if name not in by:
            continue
        cells = [f"{name:<14s}"]
        row = dict(policy=name, n=len(by[name]))
        for k in KEYS:
            ds = np.array([by[name][s][k] - base[s][k] for s in by[name] if s in base],
                          dtype=float)
            if name == "P0_base":
                cells.append(f"{ds.mean():8.1f}" if k in ("extinguished_cells", "burned_cells")
                             else f"{ds.mean():8.3f}")
                row[f"base_{k}"] = float(np.mean([by[name][s][k] for s in by[name]]))
                continue
            if len(ds) > 1:
                ci = t_ci(ds)
            else:
                ci = float("nan")
            cells.append(f"{ds.mean():+7.2f}±{ci:4.2f}")
            row[f"d_{k}"] = float(ds.mean())
            row[f"ci_{k}"] = ci
            row[f"p_{k}"] = t_p(ds)
        print("  ".join(cells))
        out_rows.append(row)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(dict(policies=out_rows, seeds=seeds,
                       base={k: float(np.mean([base[s][k] for s in base])) for k in KEYS}),
                  f, ensure_ascii=False, indent=2)
    print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()

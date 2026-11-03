# -*- coding: utf-8 -*-
"""配对分支分析（阶段 B 筛选）。

回答三个问题：
  1. 修改空间里是否存在可检出的真实收益（区分编辑类型）？
  2. 收益是否持久（+20 vs +60）？
  3. 失败类型分布（建议无效/Actor 无响应/瞬时收益回落/有害）？

统计口径：
  - 配对差 = 分支B − 基线A（同一快照、同一后续协议）。
  - 置信区间按种子聚类（每种子先平均，再对种子数做 t 分布）。
  - 提前结束（censored）与 order_changed=False（建议未生效）单列，不入均值。
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

PRIMARY = ("burned_cells", "extinguished_cells")
GUARDS = ("cov_mean", "depleted", "explored", "rknown", "min_energy", "mean_aoi")
ALLM = PRIMARY + GUARDS


def load_pairs(run_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(run_dir, "pairs_s*.jsonl"))):
        with open(p, encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
    return rows


def delta(rec, w, key):
    return rec[f"w{w}_br"][key] - rec[f"w{w}_base"][key]


def clustered_ci(vals_by_seed):
    """按种子聚类的均值 95% CI。vals_by_seed: {seed: [ diffs ]}"""
    seeds = sorted(vals_by_seed)
    if len(seeds) < 2:
        m = float(np.mean(vals_by_seed[seeds[0]])) if seeds and vals_by_seed[seeds[0]] else 0.0
        return m, float("nan")
    sm = np.array([np.mean(vals_by_seed[s]) for s in seeds])
    m = float(sm.mean())
    se = float(sm.std(ddof=1) / np.sqrt(len(sm)))
    from scipy import stats as _st
    ci = float(_st.t.ppf(0.975, len(sm) - 1) * se)
    return m, ci


def analyze(run_dir, out_md=None):
    rows = load_pairs(run_dir)
    print(f"[load] {len(rows)} 配对记录")

    # ---------- 1. noop 确定性检查 ----------
    noops = [r for r in rows if r["edit"] == "__noop__"]
    bad = 0
    for r in noops:
        for w in (20, 60):
            for k in ALLM:
                if delta(r, w, k) != 0:
                    bad += 1
                    print(f"  [WARN] noop 不为零: seed={r['seed']} t={r['t']} "
                          f"w={w} {k} d={delta(r, w, k)}")
    print(f"[noop] {len(noops)} 条检查，{bad} 条非零差 "
          f"{'（确定性成立）' if bad == 0 else '（确定性被破坏！）'}")

    # ---------- 2. 编辑类型统计 ----------
    edits = {}
    for r in rows:
        if r["edit"] == "__noop__":
            continue
        edits.setdefault(r["edit"], []).append(r)

    lines = []
    lines.append("| 编辑类型 | n | 生效 | 删失 | Δ过火@20 | Δ扑灭@20 | Δ过火@60 | Δ扑灭@60 | 零响应@20 | 正/负@60 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    detail = {}
    for name in sorted(edits):
        rs = edits[name]
        eff = [r for r in rs if r["order_changed"]]
        cens = [r for r in eff if r["censored"]]
        ok = [r for r in eff if not r["censored"]]

        def stat(w, key, pool):
            vbs = {}
            for r in pool:
                vbs.setdefault(r["seed"], []).append(delta(r, w, key))
            return clustered_ci(vbs)

        d_burn20 = stat(20, "burned_cells", ok)
        d_ext20 = stat(20, "extinguished_cells", ok)
        d_burn60 = stat(60, "burned_cells", ok)
        d_ext60 = stat(60, "extinguished_cells", ok)
        zero20 = sum(1 for r in ok if delta(r, 20, "burned_cells") == 0
                     and delta(r, 20, "extinguished_cells") == 0
                     and delta(r, 20, "cov_mean") == 0)
        pos60 = sum(1 for r in ok if delta(r, 60, "extinguished_cells") > 0
                    or delta(r, 60, "burned_cells") < 0)
        neg60 = sum(1 for r in ok if delta(r, 60, "extinguished_cells") < 0
                    or delta(r, 60, "burned_cells") > 0)
        lines.append(
            f"| {name} | {len(rs)} | {len(eff)} | {len(cens)} "
            f"| {d_burn20[0]:+.1f}±{d_burn20[1]:.1f} | {d_ext20[0]:+.2f}±{d_ext20[1]:.2f} "
            f"| {d_burn60[0]:+.1f}±{d_burn60[1]:.1f} | {d_ext60[0]:+.2f}±{d_ext60[1]:.2f} "
            f"| {zero20}/{len(ok)} | {pos60}/{neg60} |")
        detail[name] = dict(n=len(rs), effective=len(eff), censored=len(cens),
                            d_burn20=d_burn20, d_ext20=d_ext20,
                            d_burn60=d_burn60, d_ext60=d_ext60,
                            zero20=zero20, pos60=pos60, neg60=neg60, n_ok=len(ok))
    table = "\n".join(lines)
    print(table)

    # ---------- 3. 护栏指标（全部编辑合并，按类型） ----------
    guard_lines = ["| 编辑类型 | Δ覆盖率@60 | Δ耗尽@60 | Δ探索@60 | Δ已知@60 | Δ最低电@60 |",
                   "|---|---|---|---|---|---|"]
    for name in sorted(edits):
        ok = [r for r in edits[name] if r["order_changed"] and not r["censored"]]
        if not ok:
            continue
        cells = [name]
        for k in ("cov_mean", "depleted", "explored", "rknown", "min_energy"):
            vbs = {}
            for r in ok:
                vbs.setdefault(r["seed"], []).append(delta(r, 60, k))
            m, ci = clustered_ci(vbs)
            cells.append(f"{m:+.3f}±{ci:.3f}" if ci == ci else f"{m:+.3f}")
        guard_lines.append("| " + " | ".join(cells) + " |")
    guard_table = "\n".join(guard_lines)
    print(guard_table)

    # ---------- 4. 最大单点效应（找到可重复执行瓶颈的证据） ----------
    print("\n[最大单点正效应 top10]（Δ扑灭@60 或 Δ过火@60）")
    scored = []
    for r in rows:
        if r["edit"] == "__noop__" or not r["order_changed"] or r["censored"]:
            continue
        gain = (delta(r, 60, "extinguished_cells") * 5.0
                - delta(r, 60, "burned_cells"))
        scored.append((gain, r))
    scored.sort(key=lambda x: -x[0])
    for g, r in scored[:10]:
        print(f"  +{g:6.1f}  seed={r['seed']} t={r['t']:3d} {r['edit']:<22s} "
              f"Δext60={delta(r, 60, 'extinguished_cells'):+4.0f} "
              f"Δburn60={delta(r, 60, 'burned_cells'):+4.0f} "
              f"Δext20={delta(r, 20, 'extinguished_cells'):+4.0f}  {r['evidence'][:60]}")
    print("\n[最大单点负效应 top5]")
    for g, r in scored[-5:]:
        print(f"  {g:6.1f}  seed={r['seed']} t={r['t']:3d} {r['edit']:<22s} "
              f"Δext60={delta(r, 60, 'extinguished_cells'):+4.0f} "
              f"Δburn60={delta(r, 60, 'burned_cells'):+4.0f}  {r['evidence'][:60]}")

    if out_md:
        with open(out_md, "w", encoding="utf-8") as f:
            f.write("# 配对分支筛选结果\n\n```\n" + table + "\n```\n\n"
                    + guard_table + "\n")
        print(f"\n[save] {out_md}")
    return detail


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="screen1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_dir = os.path.join(_HERE, "runs", args.run)
    out = args.out or os.path.join(run_dir, "analysis.md")
    analyze(run_dir, out)

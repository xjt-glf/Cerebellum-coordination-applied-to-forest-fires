# -*- coding: utf-8 -*-
"""整局窗口（screen40_ep）分析：级联噪声是否在整局尺度摊平。

screen40_ep 用 --window-long 600 采集：记录里 w60_* 字段实际是**整局末**指标。
要回答：
  1. Δext@episode 的分布 vs Δext@60（±40 双尾是窗口噪声还是持续效应）
  2. 编辑前特征 / 已实现 @20 / @60 各层对整局收益的预测力
  3. 机制收益（Δcov/rknown）与整局任务收益的相关
  4. noop 确定性在整局尺度是否仍成立（分叉后 500+ 步的重放一致性）
"""
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(_HERE, "runs", "screen40_ep")


def d(r, w, k):
    return r[f"w{w}_br"][k] - r[f"w{w}_base"][k]


def load():
    rows = []
    for p in sorted(glob.glob(os.path.join(RUN, "pairs_s*.jsonl"))):
        for line in open(p, encoding="utf-8"):
            rows.append(json.loads(line))
    return rows


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    rows = load()
    noop = [r for r in rows if r["edit"] == "__noop__"]
    bad = sum(1 for r in noop for k in ("burned_cells", "extinguished_cells",
                                        "contained", "depleted", "explored")
              if d(r, 60, k) != 0 or d(r, 20, k) != 0)
    eff = [r for r in rows if r["edit"] != "__noop__" and r["order_changed"]
           and not r["censored"]]
    print(f"[load] {len(rows)} 条（noop {len(noop)}，有效 {len(eff)}）")
    print(f"[noop 整局确定性] 非零差 {bad} 条 {'（成立）' if bad == 0 else '（被破坏！）'}")

    y_ep = np.array([d(r, 60, "extinguished_cells") for r in eff], dtype=float)
    y_60 = None
    y_burn_ep = np.array([d(r, 60, "burned_cells") for r in eff], dtype=float)
    print(f"\n[分布] Δext@episode: p10={np.percentile(y_ep,10):+.1f} p50={np.percentile(y_ep,50):+.1f} "
          f"p90={np.percentile(y_ep,90):+.1f} p99={np.percentile(y_ep,99):+.1f} "
          f"max={y_ep.max():+.0f} min={y_ep.min():+.0f} std={y_ep.std():.1f}")
    print(f"[分布] Δburn@episode: p50={np.percentile(y_burn_ep,50):+.1f} "
          f"p90={np.percentile(y_burn_ep,90):+.1f} p10={np.percentile(y_burn_ep,10):+.1f} "
          f"std={y_burn_ep.std():.1f}")

    print("\n[相关性 vs Δext@episode]")
    probes = [("Δext20", lambda r: d(r, 20, "extinguished_cells")),
              ("Δcov20", lambda r: d(r, 20, "cov_mean")),
              ("Δrknown20", lambda r: d(r, 20, "rknown")),
              ("Δcov@ep", lambda r: d(r, 60, "cov_mean")),
              ("Δrknown@ep", lambda r: d(r, 60, "rknown")),
              ("Δexplored@ep", lambda r: d(r, 60, "explored")),
              ("Δburn@ep", lambda r: d(r, 60, "burned_cells")),
              ("Δdepleted@ep", lambda r: d(r, 60, "depleted"))]
    for name, fn in probes:
        x = np.array([fn(r) for r in eff], dtype=float)
        pe = np.corrcoef(x, y_ep)[0, 1]
        print(f"  {name:<12s} pearson {pe:+.3f}  spearman {spearman(x, y_ep):+.3f}")

    # 分编辑类型
    print("\n[分编辑类型] Δext@episode（按种子聚类 CI）")
    from collections import defaultdict
    by_edit = defaultdict(list)
    for r in eff:
        by_edit[r["edit"]].append(r)
    for name in sorted(by_edit):
        rs = by_edit[name]
        vbs = defaultdict(list)
        for r in rs:
            vbs[r["seed"]].append(d(r, 60, "extinguished_cells"))
        sm = np.array([np.mean(v) for v in vbs.values()])
        ci = float(sm.std(ddof=1) / np.sqrt(len(sm)) * 1.96) if len(sm) > 1 else float("nan")
        pos = sum(1 for r in rs if d(r, 60, "extinguished_cells") > 3)
        neg = sum(1 for r in rs if d(r, 60, "extinguished_cells") < -3)
        print(f"  {name:<22s} n={len(rs):4d} 均值 {sm.mean():+6.2f}±{ci:5.2f} "
              f"正/负(>3格) {pos}/{neg}")

    # @60 与 @episode 的一致性（需要对照 run）
    print("\n[跨窗口对照] Δext@60（screen40）与 Δext@episode（screen40_ep）同键配对")
    ref = {}
    for p in sorted(glob.glob(os.path.join(_HERE, "runs", "screen40", "pairs_s*.jsonl"))):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            ref[(r["seed"], r["t"], r["edit"], json.dumps(r["new_orders"], sort_keys=True))] = r
    xs, ys = [], []
    for r in eff:
        key = (r["seed"], r["t"], r["edit"], json.dumps(r["new_orders"], sort_keys=True))
        if key in ref:
            xs.append(d(ref[key], 60, "extinguished_cells"))
            ys.append(d(r, 60, "extinguished_cells"))
    xs, ys = np.array(xs, float), np.array(ys, float)
    if len(xs) > 10:
        print(f"  n={len(xs)} corr(Δext@60, Δext@ep) = {np.corrcoef(xs, ys)[0,1]:+.3f} "
              f"spearman {spearman(xs, ys):+.3f}")
        same_sign = float(np.mean(np.sign(xs) == np.sign(ys)))
        big = np.abs(xs) > 10
        if big.any():
            big_sign = float(np.mean(np.sign(xs[big]) == np.sign(ys[big])))
            print(f"  |Δext@60|>10 的 {int(big.sum())} 条里符号一致率 {big_sign:.1%}")
            print(f"  全体符号一致率 {same_sign:.1%}（随机 50%）")
    else:
        print("  匹配不足")


if __name__ == "__main__":
    main()

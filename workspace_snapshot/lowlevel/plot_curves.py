# -*- coding: utf-8 -*-
"""绘制低层 MAPPO 训练曲线。

    python3 lowlevel/plot_curves.py                            # 画 w3_* 各变体
    python3 lowlevel/plot_curves.py --pattern "w3_*" --best w3_g995_ent01
    python3 lowlevel/plot_curves.py --runs w3_g995_ent01       # 只画一个

产物（默认写到 lowlevel/runs/pic/）：
    reward_curve_all.png/pdf   同阶段各变体的训练回报对比（原始 + 滑动平均）
    reward_curve_best.png/pdf  成果版本：训练回报 + 确定性评估回报
    metrics_best.png/pdf       成果版本的 6 个评估指标
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
          "#9467bd", "#8c564b", "#e377c2", "#7f7f7f")

METRIC_COLS = ("eval_explored", "eval_discovery", "eval_rknown",
               "eval_cov", "eval_burned", "eval_depl")
METRIC_NAMES = ("Whole-map explored", "Fire discovery", "Rescuer known fire",
                "UAV coverage", "Burned area", "# depleted UAVs")


def smooth(s, window=20):
    return s.rolling(window, min_periods=1).mean()


def load(run):
    return pd.read_csv(os.path.join(RUNS, run, "train_log.csv"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--pattern", default="w3_*", help="没给 --runs 时按此通配符收集 run")
    ap.add_argument("--out", default=os.path.join(RUNS, "pic"))
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--best", default=None,
                    help="指定成果版本。默认按 eval_return 最高挑，但正式选型是按"
                         "evaluate.py 的多指标结果定的，需要时用这个覆盖")
    args = ap.parse_args()

    runs = args.runs or sorted(
        os.path.basename(os.path.dirname(p))
        for p in glob.glob(os.path.join(RUNS, args.pattern, "train_log.csv")))
    if not runs:
        raise SystemExit("no train_log.csv found under %s" % RUNS)
    os.makedirs(args.out, exist_ok=True)
    dfs = {r: load(r) for r in runs}

    # ---------------- 图 1：各变体训练回报对比 ----------------
    plt.figure(figsize=(9, 5))
    for i, (r, df) in enumerate(dfs.items()):
        c = COLORS[i % len(COLORS)]
        x = df["env_steps"] / 1e6
        plt.plot(x, df["train_return"], color=c, alpha=0.13, lw=0.7)
        plt.plot(x, smooth(df["train_return"], args.window), color=c, lw=1.6, label=r)
    plt.xlabel("Environment steps (millions)")
    plt.ylabel("Team return per update (96 envs x 64 steps)")
    plt.title("Low-level MAPPO training return (variants)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(args.out, "reward_curve_all." + ext),
                    dpi=150, bbox_inches="tight")
    plt.close()

    # ---------------- 选出成果版本 ----------------
    if args.best and args.best in dfs:
        best_run = args.best
    else:
        best_run = max(dfs, key=lambda r: pd.to_numeric(
            dfs[r]["eval_return"], errors="coerce").max())
    df = dfs[best_run]
    ev = df.dropna(subset=["eval_return"])
    print(f"[best] {best_run}")

    # ---------------- 图 2：成果版本 reward 曲线 ----------------
    fig, ax = plt.subplots(figsize=(9, 5))
    x = df["env_steps"] / 1e6
    ax.plot(x, df["train_return"], color=COLORS[0], alpha=0.16, lw=0.7,
            label="train return (raw)")
    ax.plot(x, smooth(df["train_return"], args.window), color=COLORS[0], lw=1.8,
            label=f"train return (MA{args.window})")
    ax2 = ax.twinx()
    ax2.plot(ev["env_steps"] / 1e6, ev["eval_return"], color=COLORS[3], lw=1.5,
             marker="o", ms=3, label="eval return (deterministic)")
    ax.set_xlabel("Environment steps (millions)")
    ax.set_ylabel("Team return per update (96 envs x 64 steps)")
    ax2.set_ylabel("Deterministic eval return per episode")
    ax.set_title(f"Low-level MAPPO reward curve — {best_run}")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out, "reward_curve_best." + ext),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------------- 图 3：成果版本的评估指标 ----------------
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, col, name in zip(axes.ravel(), METRIC_COLS, METRIC_NAMES):
        if col not in ev.columns:
            continue
        ax.plot(ev["env_steps"] / 1e6, ev[col], color=COLORS[2], lw=1.5,
                marker="o", ms=3)
        ax.set_xlabel("Env steps (M)")
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Evaluation metrics — {best_run}")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out, "metrics_best." + ext),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[out]", args.out)


if __name__ == "__main__":
    main()

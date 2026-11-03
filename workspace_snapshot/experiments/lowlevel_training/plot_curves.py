# -*- coding: utf-8 -*-
"""训练曲线绘制：每个 trial 的 return / 熵 / KL / 目标到访率。"""
import csv, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/ssd3/xiangjutao/hrl_dqn_improve/experiments/lowlevel_training/runs"


def load(t):
    p = os.path.join(ROOT, t, "train_log.csv")
    if not os.path.exists(p):
        return None
    return list(csv.DictReader(open(p)))


def main(out_dir, trials):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    metrics = [("ret", "team return / rollout", axes[0, 0]),
               ("ent", "policy entropy", axes[0, 1]),
               ("kl", "approx KL", axes[1, 0]),
               ("clipfrac", "clip fraction", axes[1, 1]),
               ("goal_exp_arr", "train goal: explore arrived frac", axes[2, 0]),
               ("goal_pat_arr", "train goal: patrol arrived frac", axes[2, 1])]
    for t in trials:
        rows = load(t)
        if not rows:
            continue
        steps = [int(float(r["env_steps"])) / 1e6 for r in rows]
        for k, label, ax in metrics:
            ax.plot(steps, [float(r[k]) for r in rows], label=t, lw=1.2)
    for k, label, ax in metrics:
        ax.set_xlabel("env steps (M)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("goal-conditioned overnight trials")
    fig.tight_layout()
    out = os.path.join(out_dir, "training_curves.png")
    fig.savefig(out, dpi=110)
    print("saved", out)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "curves")
    trials = sys.argv[2:] or ["P1_s1101", "P1_s1102", "P2_s1201", "P2_s1202",
                               "P3_s1301", "P3_s1302", "REL_s1303", "REL_s1304",
                               "P2_s1203", "P2BC_s1204"]
    main(out, trials)

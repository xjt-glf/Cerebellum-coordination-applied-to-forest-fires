"""训练健康检查 + 指标趋势。RUN_GLOB 可指定 run 通配。"""
import glob
import os

import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pat = os.environ.get("RUN_GLOB", "lowlevel/runs/w2_*/train_log.csv")
print("%-16s %5s %5s %7s %8s %8s %8s %9s" % (
    "run", "Msteps", "nan", "kl_max", "clip_max", "ent_last", "v_last", "ret_slope"))
for p in sorted(glob.glob(pat)):
    r = os.path.basename(os.path.dirname(p))
    d = pd.read_csv(p)
    cols = ["train_return", "pg", "v", "ent", "kl"]
    bad = int((~np.isfinite(d[cols].to_numpy())).sum())
    y = d["train_return"].rolling(20, min_periods=1).mean().to_numpy()
    slope = float(np.polyfit(np.arange(len(d)), y, 1)) if False else float(
        np.polyfit(np.arange(len(d)), y, 1)[0])
    print("%-16s %6.2f %5d %7.4f %8.3f %8.3f %8.4f %9.2f" % (
        r, d["env_steps"].iloc[-1] / 1e6, bad, d["kl"].max(), d["clipfrac"].max(),
        d["ent"].iloc[-1], d["v"].iloc[-1], slope))

print("\n%-16s %s" % ("run", "eval 指标：前 3 次 -> 后 3 次均值"))
for p in sorted(glob.glob(pat)):
    r = os.path.basename(os.path.dirname(p))
    d = pd.read_csv(p).dropna(subset=["eval_return"])
    if len(d) < 6:
        continue
    a, b = d.head(3).mean(numeric_only=True), d.tail(3).mean(numeric_only=True)
    print("%-16s ret %5.0f->%5.0f | exp %.3f->%.3f | disc %.3f->%.3f | "
          "cov %.3f->%.3f | rk %.3f->%.3f | ext %3.0f->%3.0f | depl %.1f->%.1f" % (
              r, a["eval_return"], b["eval_return"], a["eval_explored"], b["eval_explored"],
              a["eval_discovery"], b["eval_discovery"], a["eval_cov"], b["eval_cov"],
              a["eval_rknown"], b["eval_rknown"], a["eval_ext"], b["eval_ext"],
              a["eval_depl"], b["eval_depl"]))

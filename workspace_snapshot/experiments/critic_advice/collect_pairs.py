# -*- coding: utf-8 -*-
"""采集配对分支数据（阶段 B 入口实验）。

对每个种子：跑一条 H 基线整局轨迹（确定协议），在每个重规划边界快照，
对该边界适用的每条编辑跑一个 60 步分支（前 20 步为修改段，之后恢复同一 H 协议），
记录 +20 / +60 两个检查点的真实任务指标，与基线轨迹同检查点做配对差。

产出（runs/<name>/）：
  config.json     显式配置 + 权重/源码哈希（版本清单）
  pairs_s*.jsonl  每行一个 (边界, 编辑) 配对记录
  episodes.jsonl  每种子基线整局摘要

已登记的主要配对指标（跑之前固定）：
  主：Δ过火格数(burned_cells)、Δ扑灭格数(extinguished_cells) @ +20 / +60
  护栏：Δ窗口覆盖率均值、Δ耗尽数、Δ探索率、Δ人员已知、Δ最低电量
提前结束口径：任一分支提前 done（全机耗尽）时双方 steps_run 不一致，
该窗口配对差按删失处理（steps_run 不等的记录在分析中单独归类，不混入均值）。
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import harness
from harness import (BENCH, Snapshot, decision_context, make_env, load_actor,
                     make_h_brain, run_protocol, task_metrics, orders_to_json)
from edits import applicable_edits

W_KEYS = ("t", "explored", "burned_cells", "extinguished_cells", "contained",
          "discovery", "rknown", "active_discovery", "active_rknown",
          "coverage", "mean_aoi", "max_aoi", "depleted", "min_energy",
          "burning_cells", "cov_mean", "steps_run")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def source_fingerprint():
    out = {}
    root = os.path.abspath(os.path.join(_HERE, "..", ".."))
    for f in ("harness.py", "edits.py", "collect_pairs.py"):
        p = os.path.join(_HERE, f)
        if os.path.exists(p):
            out[f] = sha256(p)[:16]
    for rel in ("lowlevel/env_wrapper.py", "lowlevel/obs_builder.py",
                "lowlevel/task_spec.py", "lowlevel/mission_demo.py",
                "webapp/brains.py", "Env/低层环境.py"):
        p = os.path.join(root, rel)
        if os.path.exists(p):
            out[rel] = sha256(p)[:16]
    return out


def _window_metrics(m, cov_hist_slice):
    d = {k: m.get(k) for k in W_KEYS}
    d["cov_mean"] = float(np.mean(cov_hist_slice)) if len(cov_hist_slice) else 0.0
    return d


def run_episode_collect(seed, out_dir, w_short=20, w_long=60, max_edits=8,
                        noop_every=5, cfg=None):
    """单种子采集。写 pairs_s<seed>.jsonl，返回摘要 dict。"""
    import torch
    torch.set_num_threads(1)
    rng = random.Random(seed * 7919 + 13)

    env = make_env(seed, cfg)
    env.reset()
    actor = load_actor(BENCH["actor"])
    brain = make_h_brain(cfg)
    replan = BENCH["replan_every"]
    max_steps = env.max_steps

    # ---------- 基线轨迹：逐步检查点 + 每边界快照 ----------
    ckpt = {}                      # env.t -> metrics（逐步）
    cov_hist = []                  # 第 k 项 = env.t=k+1 时的 coverage
    snapshots, ctxs, plans = {}, {}, {}
    done_latch = False
    ckpt[0] = task_metrics(env)
    obs = None
    t_start = time.time()
    while env.t < max_steps and not done_latch:
        if env.t % replan == 0:
            snapshots[env.t] = Snapshot(env)      # 快照在 set_plan 之前（分支自行下发计划）
            from lowlevel.mission_demo import sense_state
            s = sense_state(env)
            plan, reason, meta = brain.plan(s)
            obs = env.set_plan(plan)
            plans[env.t] = orders_to_json(env.orders)
            # 编辑上下文必须在 set_plan 之后：edits 修改的是**H 已下发的计划**，
            # 不是环境内部 sampler 的计划（第一版就在这里出错，整批数据作废）
            ctxs[env.t] = decision_context(env)
        with torch.no_grad():
            a = actor.act(torch.as_tensor(obs, dtype=torch.float32),
                          deterministic=True)[0].numpy()
        obs, _r, done, info = env.step(a)
        cov_hist.append(float(info.get("coverage_rate", 0.0)))
        ckpt[env.t] = task_metrics(env)
        if done:
            done_latch = True
    T_end = env.t
    base_final = dict(task_metrics(env))
    base_final["cov_mean"] = float(np.mean(cov_hist))
    base_final["steps"] = T_end

    def base_ckpt(target):
        """目标检查点；回合提前结束则取最后状态（删失口径）。"""
        return ckpt.get(target) or ckpt[T_end]

    # ---------- 配对分支 ----------
    pairs_path = os.path.join(out_dir, f"pairs_s{seed}.jsonl")
    n_pair = n_cens = 0
    with open(pairs_path, "w", encoding="utf-8") as fo:
        for t in sorted(snapshots):
            if t + w_short > max_steps:
                continue
            ctx = ctxs[t]
            base_orders = plans[t]
            edits = applicable_edits(ctx, rng)[:max_edits]
            if (t // replan) % noop_every == 0:
                from lowlevel.task_spec import TeamPlan
                same = TeamPlan.from_llm_json({"uavs": base_orders},
                                              len(base_orders)).orders
                edits = edits + [("__noop__", list(range(len(base_orders))), same,
                                 "determinism check")]
            for name, uids, new_orders, ev in edits:
                t20, t60 = t + w_short, min(t + w_long, max_steps)
                b_env = snapshots[t].restore()
                m20, cov20, _ = run_protocol(b_env, actor, brain, t20,
                                             override_at=t, override_orders=new_orders,
                                             replan_every=replan)
                m60, cov60, _ = run_protocol(b_env, actor, brain, t60,
                                             replan_every=replan)
                cov_all = cov20 + cov60
                # 基线同窗口
                b_m20, b_m60 = base_ckpt(t20), base_ckpt(t60)
                b_cov20 = cov_hist[t:t20] if t20 <= T_end else cov_hist[t:T_end]
                b_cov60 = cov_hist[t:t60] if t60 <= T_end else cov_hist[t:T_end]
                censored = (m20.get("steps_run", 0) < t20 - t
                            or m60.get("steps_run", 0) + m20.get("steps_run", 0) < t60 - t
                            or t20 > T_end or t60 > T_end)
                if censored:
                    n_cens += 1
                rec = dict(
                    seed=seed, t=t, edit=name, uav_ids=uids, evidence=ev,
                    order_changed=(orders_to_json(new_orders) != base_orders),
                    censored=bool(censored),
                    base_orders=base_orders, new_orders=orders_to_json(new_orders),
                    m_t={k: ckpt[t].get(k) for k in W_KEYS},
                    w20_base=_window_metrics(b_m20, b_cov20),
                    w20_br=_window_metrics(m20, cov20),
                    w60_base=_window_metrics(b_m60, b_cov60),
                    w60_br=_window_metrics(m60, cov_all),
                    ctx=snap_ctx(ctx),
                )
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_pair += 1
    return dict(seed=seed, pairs=n_pair, censored=n_cens, final=base_final,
                T_end=T_end, secs=round(time.time() - t_start, 1))


def snap_ctx(ctx):
    """决策上下文瘦身：只留证据字段（orders 已含每机解析目标）。"""
    s = ctx["sense"]
    return dict(t=s["t"], explored=s["explored"], discovery=s["discovery"],
                coverage=s["coverage"], rescuer_known=s["rescuer_known"],
                burned=s["burned"], max_aoi=s["max_aoi"],
                n_clusters=len(s["clusters"]),
                clusters=[[c["centroid"][0], c["centroid"][1], c["size"]]
                          for c in s["clusters"]],
                rescuers=[[r["id"], r["pos"][0], r["pos"][1],
                           int(r["covered"]), r["aoi"]] for r in s["rescuers"]],
                uavs=[[u["id"], u["pos"][0], u["pos"][1], u["e"], u["margin"]]
                      for u in s["uavs"]],
                orders=ctx["orders"], uncovered_ids=ctx["uncovered_ids"])


def _worker(args):
    seed, out_dir, kwargs = args
    try:
        return run_episode_collect(seed, out_dir, **kwargs)
    except Exception as e:
        import traceback
        return dict(seed=seed, error=f"{type(e).__name__}: {e}",
                    trace=traceback.format_exc()[-1500:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="screen1")
    ap.add_argument("--seeds", type=int, default=10, help="种子数（从 seed0 起）")
    ap.add_argument("--seed0", type=int, default=30000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--window-short", type=int, default=20)
    ap.add_argument("--window-long", type=int, default=60)
    ap.add_argument("--max-edits", type=int, default=8)
    args = ap.parse_args()

    out_dir = os.path.join(_HERE, "runs", args.run)
    os.makedirs(out_dir, exist_ok=True)
    cfg = dict(BENCH)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(dict(config=cfg, seeds=[args.seed0 + i for i in range(args.seeds)],
                       windows=(args.window_short, args.window_long),
                       max_edits=args.max_edits,
                       actor_sha256=sha256(BENCH["actor"]),
                       sources=source_fingerprint(),
                       cmd=" ".join(sys.argv)), f, ensure_ascii=False, indent=2)

    seeds = [args.seed0 + i for i in range(args.seeds)]
    kwargs = dict(w_short=args.window_short, w_long=args.window_long,
                  max_edits=args.max_edits)
    if args.workers <= 1:
        results = [_worker((s, out_dir, kwargs)) for s in seeds]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(min(args.workers, len(seeds))) as pool:
            results = pool.map(_worker, [(s, out_dir, kwargs) for s in seeds])

    with open(os.path.join(out_dir, "episodes.jsonl"), "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = [r for r in results if "error" not in r]
    print(f"[collect] {len(ok)}/{len(results)} 种子完成 -> {out_dir}")
    for r in results:
        if "error" in r:
            print(f"  seed {r['seed']} FAILED: {r['error']}\n{r.get('trace', '')[:600]}")
        else:
            fin = r["final"]
            print(f"  seed {r['seed']}: pairs={r['pairs']} censored={r['censored']} "
                  f"burned={fin['burned_cells']} ext={fin['extinguished_cells']} "
                  f"contained={fin['contained']} depleted={fin['depleted']} "
                  f"({r['secs']}s)")


if __name__ == "__main__":
    main()

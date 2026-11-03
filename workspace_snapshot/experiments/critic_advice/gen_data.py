# -*- coding: utf-8 -*-
"""EACC 数据采集：参数化编辑候选的配对分支标签。

两种模式：
  exec    : 分支跑 20 步 -> 段末机制配对差（训练执行模型；便宜，~100 种子/regime）
  episode : 分支先 20 步再跑到整局 -> 整局任务配对差（训练价值头；~40 种子/regime）

双 regime（预注册）：
  normal : 4 机 / 10 人 / 3 火点（指南基准）
  scarce : 3 机 / 14 人 / 3 火点（建议方差存在的域）

配对协议与一轮相同：快照在 H 计划下发前，编辑上下文在下发后；分支 B 只在本段
替换计划，之后恢复 H 协议；noop 候选做确定性锚点（每 5 个边界必含 1 个 noop）。
"""
import argparse
import gzip
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import harness
from harness import BENCH, Snapshot, make_env, load_actor, make_h_brain, \
    run_protocol, task_metrics
from critic_data import (ctx_v2, mech_metrics, mech_vec, candidate_edits,
                        orders_dicts, MECH_KEYS, EP_KEYS)

REGIMES = {
    "normal": dict(),
    "scarce": dict(uavs=3, rescuers=14, fires=3),
}


def regime_cfg(regime):
    c = dict(BENCH)
    c.update(REGIMES[regime])
    return c


def _ep_metrics(env, cov_hist):
    """整局标签：终值指标 + 全程覆盖均值（两侧同口径）。"""
    m = task_metrics(env)
    return dict(
        extinguished_cells=float(m["extinguished_cells"]),
        burned_cells=float(m["burned_cells"]),
        contained=float(m["contained"]),
        rknown=float(m["rknown"]),
        explored=float(m["explored"]),
        depleted=float(m["depleted"]),
        cov_mean=float(np.mean(cov_hist)) if cov_hist else 0.0,
    )


def collect_seed(seed, regime, mode, out_dir, max_edits=9, w_short=20,
                 noop_every=1, w_long=600):
    """noop_every=1：noop 恒在候选尾部（candidate_edits 保证）。

    每 noop_every*n 个边界额外插入一个 noop 检查记录（与候选尾部 noop 重复时去重）。
    """
    import torch
    torch.set_num_threads(1)
    cfg = regime_cfg(regime)
    env = make_env(seed, cfg)
    env.reset()
    actor = load_actor(BENCH["actor"])
    brain = make_h_brain(cfg)
    replan = cfg["replan_every"]
    max_steps = cfg["steps"]

    # ---------- 基线轨迹 ----------
    mech_at = {}                       # t -> mech vec
    cov_hist = []
    snapshots, ctxs, plans = {}, {}, {}
    obs = None
    done = False
    mech_at[0] = mech_vec(mech_metrics(env))
    t0 = time.time()
    while env.t < max_steps and not done:
        if env.t % replan == 0:
            snapshots[env.t] = Snapshot(env)
            from lowlevel.mission_demo import sense_state
            s = sense_state(env)
            plan, _reason, _meta = brain.plan(s)
            obs = env.set_plan(plan)
            ctxs[env.t] = ctx_v2(env)
            plans[env.t] = [dict(o) for o in ctxs[env.t]["orders"]]
        with torch.no_grad():
            a = actor.act(torch.as_tensor(obs, dtype=torch.float32),
                          deterministic=True)[0].numpy()
        obs, _r, done, info = env.step(a)
        cov_hist.append(float(info.get("coverage_rate", 0.0)))
        if env.t % replan == 0 or done or env.t >= max_steps - 1:
            mech_at[env.t] = mech_vec(mech_metrics(env))
    T_end = env.t
    ep_base = _ep_metrics(env, cov_hist)
    ep_base["steps"] = T_end

    # ---------- 配对分支 ----------
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{regime}_{mode}_s{seed}.jsonl.gz")
    n_rec = n_bad = 0
    with gzip.open(path, "wt", encoding="utf-8") as fo:
        for t in sorted(snapshots):
            if t + w_short > max_steps:
                continue
            ctx = ctxs[t]
            base_orders = plans[t]
            m_t = mech_at[t]
            m20_base = mech_at.get(t + w_short) or mech_at[T_end]
            b_cov20 = cov_hist[t:t + w_short] if t + w_short <= T_end else cov_hist[t:T_end]
            cov20_base = float(np.mean(b_cov20)) if b_cov20 else 0.0
            cands = candidate_edits(ctx, max_edits=max_edits)
            for name, params, new_orders, evidence in cands:
                is_noop = (name == "noop")
                b_env = snapshots[t].restore()
                _m20, cov20_hist, _ = run_protocol(b_env, actor, brain, t + w_short,
                                                  override_at=t, override_orders=new_orders,
                                                  replan_every=replan)
                m20_br = mech_vec(mech_metrics(b_env))
                cov20_br = float(np.mean(cov20_hist)) if cov20_hist else 0.0
                ep_br = None
                if mode == "episode":
                    _mep, cov_ep, _ = run_protocol(b_env, actor, brain,
                                                    min(t + w_long, max_steps),
                                                    replan_every=replan)
                    ep_br = _ep_metrics(b_env, cov20_hist + cov_ep)
                rec = dict(
                    seed=seed, regime=regime, mode=mode, t=t,
                    edit=name, params=params, evidence=evidence,
                    base_orders=base_orders,
                    new_orders=orders_dicts(b_env, new_orders),
                    ctx=ctx,
                    m_t=m_t, m20_base=m20_base, m20_br=m20_br,
                    cov20_base=cov20_base, cov20_br=cov20_br,
                    ep_base=ep_base, ep_br=ep_br,
                )
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_rec += 1
                if is_noop:
                    if m20_br != m20_base:
                        n_bad += 1
    return dict(seed=seed, regime=regime, mode=mode, records=n_rec, noop_bad=n_bad,
                T_end=T_end, ep_base=ep_base,
                secs=round(time.time() - t0, 1))


def _worker(a):
    seed, regime, mode, out_dir, kw = a
    try:
        return collect_seed(seed, regime, mode, out_dir, **kw)
    except Exception as e:
        import traceback
        return dict(seed=seed, regime=regime, mode=mode,
                    error=f"{type(e).__name__}: {e}",
                    trace=traceback.format_exc()[-1200:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["exec", "episode"], required=True)
    ap.add_argument("--regimes", default="normal,scarce")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--seed-base", default="40000:41000",
                    help="冒号分隔：normal 起点:scarce 起点")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--max-edits", type=int, default=9)
    args = ap.parse_args()

    nb, sb = (int(x) for x in args.seed_base.split(":"))
    jobs = []
    out_dir = os.path.join(_HERE, "runs", f"critic_data_{args.mode}")
    os.makedirs(out_dir, exist_ok=True)
    for regime in args.regimes.split(","):
        s0 = nb if regime == "normal" else sb
        for i in range(args.seeds):
            jobs.append((s0 + i, regime, args.mode, out_dir,
                         dict(max_edits=args.max_edits)))
    ctx = mp.get_context("fork")
    with ctx.Pool(min(args.workers, len(jobs))) as pool:
        results = pool.map(_worker, jobs)
    ok = [r for r in results if "error" not in r]
    bad = [r for r in results if "error" in r]
    n_bad_noop = sum(r.get("noop_bad", 0) for r in ok)
    print(f"[collect {args.mode}] {len(ok)}/{len(results)} 种子完成 -> {out_dir} "
          f"（noop 非零差 {n_bad_noop}）")
    with open(os.path.join(out_dir, "summary.jsonl"), "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for r in bad[:3]:
        print(f"  FAIL seed={r['seed']} {r['regime']}: {r['error']}\n{r.get('trace','')[:400]}")
    for r in ok[:2] + ok[-2:] if len(ok) > 4 else ok:
        print(f"  {r['regime']} seed={r['seed']}: {r['records']} 条 "
              f"ep_ext={r['ep_base']['extinguished_cells']:.0f} "
              f"cov={r['ep_base']['cov_mean']:.3f} ({r['secs']}s)")


if __name__ == "__main__":
    main()

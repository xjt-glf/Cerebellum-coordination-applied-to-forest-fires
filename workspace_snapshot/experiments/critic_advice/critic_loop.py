# -*- coding: utf-8 -*-
"""EACC 闭环评估：Critic 在每个重规划边界给出 0/1 个局部编辑，整局任务指标配对比较。

闭环机制（与训练数据语义一致）：
  每 20 步边界：H 生成计划并下发 -> ctx（此时 orders=H 计划）-> 参数化候选编辑
  （含 noop）-> Critic 打分（执行模型 + 价值头）-> 应用 argmax（或阈值接受）。

臂：
  base     H 原计划（不修改）
  random   每边界随机选一个非 noop 候选（同候选预算）
  rule     每边界应用优先序第一个非 noop 候选
  scalar   标量对照头排序（无执行模型，ReAd++ 式），argmax 含 noop
  eacc     执行模型+价值头排序，argmax 含 noop（noop=拒答）
  eacc_pos 同 eacc 但显式阈值：非 noop 最优预测>0 才应用
消融在臂间完成：eacc vs scalar（去执行模型）、eacc vs rule（学习 vs 规则）、
eacc vs random（学习 vs 同预算随机）、noop 拒答 = noop 被选中频率。
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import harness
from harness import BENCH, make_env, load_actor, make_h_brain, task_metrics
from gen_data import regime_cfg
from critic_data import (ctx_v2, candidate_edits, orders_dicts, build_features)

MODEL_DIR = os.path.join(_HERE, "runs", "critic_models")


def load_heads(policies):
    from exec_model import load_bundle
    heads = {}
    if any(p.startswith("eacc") for p in policies):
        heads["exec_nets"], _ = load_bundle(os.path.join(MODEL_DIR, "exec_model.pt"))
        heads["value_nets"], _ = load_bundle(os.path.join(MODEL_DIR, "value_head.pt"))
    if "scalar" in policies:
        heads["scalar_nets"], _ = load_bundle(os.path.join(MODEL_DIR, "scalar_head.pt"))
    return heads


def _pred_mean(nets, X):
    return np.mean([m.predict(X) for m in nets], 0)


class Critic:
    """闭环里的建议器。policy 决定选法。"""

    def __init__(self, policy, exec_nets=None, value_nets=None, scalar_nets=None):
        self.policy = policy
        self.exec_nets = exec_nets
        self.value_nets = value_nets
        self.scalar_nets = scalar_nets
        self.n_edit = self.n_noop = self.n_bound = 0
        self.edit_log = []

    def choose(self, env, ctx, cands, allow_edit=True):
        """返回 (cand_name, orders)。cands: [(name, params, orders, evidence)]"""
        self.n_bound += 1
        if not allow_edit:
            self.n_noop += 1
            return "noop", cands[-1][2]
        if self.policy == "base":
            self.n_noop += 1
            return "noop", cands[-1][2]                # noop 恒在尾部
        non_noop = [(i, c) for i, c in enumerate(cands) if c[0] != "noop"]
        if not non_noop:
            self.n_noop += 1
            return "noop", cands[-1][2]
        if self.policy in ("random", "random_rare"):
            i, c = non_noop[int(np.random.RandomState(
                (ctx["t"] + 1) * 31).randint(len(non_noop)))]
            self.n_edit += 1
            return c[0], c[2]
        if self.policy == "rule":
            i, c = non_noop[0]
            self.n_edit += 1
            return c[0], c[2]
        # 学习型：全部候选（含 noop）打分
        feats, cand_idx = [], []
        for i, (name, _params, orders, _ev) in enumerate(cands):
            nd = orders_dicts(env, orders)
            feats.append(build_features(ctx, base_dicts := ctx["orders"], nd))
            cand_idx.append(i)
        X = np.stack(feats)
        if self.policy == "scalar":
            score = _pred_mean(self.scalar_nets, X).ravel()
        else:
            dmech = _pred_mean(self.exec_nets, X)
            Xv = np.concatenate([X, dmech], 1)
            score = _pred_mean(self.value_nets, Xv).ravel()
        if self.policy in ("eacc", "eacc_rare", "scalar"):
            j = int(np.argmax(score))
        elif self.policy in ("eacc_pos", "eacc_t3"):
            thr = 0.0 if self.policy == "eacc_pos" else 3.0
            jn = [k for k, i in enumerate(cand_idx) if cands[i][0] != "noop"]
            j = jn[int(np.argmax([score[k] for k in jn]))]
            if score[j] <= thr:
                j = len(cands) - 1
        else:
            raise ValueError(self.policy)
        name = cands[cand_idx[j]][0]
        if name == "noop":
            self.n_noop += 1
        else:
            self.n_edit += 1
            self.edit_log.append(dict(t=ctx["t"], edit=name,
                                      score=float(score[j]),
                                      evidence=cands[cand_idx[j]][3]))
        return name, cands[cand_idx[j]][2]


def run_episode(seed, regime, policy, heads=None):
    import torch
    torch.set_num_threads(1)
    cfg = regime_cfg(regime)
    env = make_env(seed, cfg)
    env.reset()
    actor = load_actor(BENCH["actor"])
    brain = make_h_brain(cfg)
    replan = cfg["replan_every"]
    max_steps = cfg["steps"]
    critic = Critic(policy, **(heads or {}))
    obs, cov = None, []
    done = False
    while env.t < max_steps and not done:
        if env.t % replan == 0:
            from lowlevel.mission_demo import sense_state
            s = sense_state(env)
            plan, _r, _m = brain.plan(s)
            obs = env.set_plan(plan)
            ctx = ctx_v2(env)
            cands = candidate_edits(ctx)
            allow = (env.t // replan) % 3 == 0 if policy.endswith("_rare") else True
            name, orders = critic.choose(env, ctx, cands, allow_edit=allow)
            if name != "noop":
                obs = env.set_plan(list(orders))       # 与 H 计划有差异才重新下发
        with torch.no_grad():
            a = actor.act(torch.as_tensor(obs, dtype=torch.float32),
                          deterministic=True)[0].numpy()
        obs, _rew, done, info = env.step(a)
        cov.append(float(info.get("coverage_rate", 0.0)))
    m = task_metrics(env)
    return dict(seed=seed, regime=regime, policy=policy,
                extinguished_cells=float(m["extinguished_cells"]),
                burned_cells=float(m["burned_cells"]),
                contained=float(m["contained"]),
                rknown=float(m["rknown"]),
                cov_mean=float(np.mean(cov)) if cov else 0.0,
                explored=float(m["explored"]),
                depleted=float(m["depleted"]),
                n_edit=critic.n_edit, n_noop=critic.n_noop, n_bound=critic.n_bound)


def _worker(a):
    seed, regime, policy, heads = a
    try:
        return run_episode(seed, regime, policy, heads)
    except Exception as e:
        import traceback
        return dict(seed=seed, regime=regime, policy=policy,
                    error=f"{type(e).__name__}: {e}",
                    trace=traceback.format_exc()[-800:])


def evaluate(policies, regimes, seeds0, n_seeds, workers, out_path, note=""):
    jobs = []
    heads = None
    need = any(p in ("eacc", "eacc_pos", "scalar") for p in policies)
    if need:
        heads = load_heads(policies)
    for regime in regimes:
        s0 = seeds0[regime]
        for p in policies:
            for i in range(n_seeds):
                jobs.append((s0 + i, regime, p, heads))
    ctx = mp.get_context("fork")
    with ctx.Pool(min(workers, len(jobs))) as pool:
        results = pool.map(_worker, jobs)
    ok = [r for r in results if "error" not in r]
    bad = [r for r in results if "error" in r]
    for r in bad[:3]:
        print(f"[FAIL] {r['policy']} {r['regime']} s{r['seed']}: {r['error']}"
              f"\n{r.get('trace','')[:300]}")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report(ok, policies, regimes, out_path, note)
    return results


def report(ok, policies, regimes, out_path, note):
    from collections import defaultdict
    by = defaultdict(dict)
    for r in ok:
        by[(r["regime"], r["policy"])][r["seed"]] = r
    KEYS = ("extinguished_cells", "burned_cells", "contained", "rknown",
            "cov_mean", "explored", "depleted")
    print(f"\n===== 闭环配对结果（{note}，配对差 = 臂 − base）=====")
    print(f"{'regime':<8s}{'臂':<10s}{'Δ扑灭':>14s}{'Δ过火':>13s}{'Δ合围':>12s}"
          f"{'Δ已知':>13s}{'Δ覆盖':>13s}{'Δ耗尽':>8s}  编辑率")
    summary = []
    for regime in regimes:
        base = by.get((regime, "base"), {})
        for p in policies:
            if p == "base":
                if base:
                    print(f"{regime:<8s}{p:<10s}"
                          f"{np.mean([base[s]['extinguished_cells'] for s in base]):13.1f}"
                          f"{np.mean([base[s]['burned_cells'] for s in base]):13.1f}"
                          f"{np.mean([base[s]['contained'] for s in base]):12.2f}"
                          f"{np.mean([base[s]['rknown'] for s in base]):13.3f}"
                          f"{np.mean([base[s]['cov_mean'] for s in base]):13.3f}"
                          f"{np.mean([base[s]['depleted'] for s in base]):8.2f}  -")
                continue
            arm = by.get((regime, p), {})
            common = [s for s in arm if s in base]
            if not common:
                continue
            cells = [f"{regime:<8s}{p:<10s}"]
            row = dict(regime=regime, policy=p, n=len(common))
            try:
                from scipy import stats as st
                have_scipy = True
            except Exception:
                have_scipy = False
            for k in KEYS:
                d = np.array([arm[s][k] - base[s][k] for s in common], float)
                if len(d) > 1:
                    ci = (st.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
                          if have_scipy else 1.96 * d.std(ddof=1) / np.sqrt(len(d)))
                    pv = (float(st.ttest_rel(d, np.zeros_like(d)).pvalue)
                          if have_scipy else float("nan"))
                else:
                    ci, pv = float("nan"), float("nan")
                cells.append(f"{d.mean():+8.2f}±{ci:4.2f}" if k in ("extinguished_cells",)
                             else (f"{d.mean():+7.1f}±{ci:4.1f}" if k == "burned_cells"
                                   else f"{d.mean():+7.3f}±{ci:5.3f}"))
                row[f"d_{k}"] = float(d.mean())
                row[f"p_{k}"] = pv
            er = np.mean([arm[s]["n_edit"] / max(1, arm[s]["n_bound"]) for s in common])
            cells.append(f"  {er:.2f}")
            row["edit_rate"] = float(er)
            print("".join(cells))
            summary.append(row)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["expl", "confirm"], default="expl")
    ap.add_argument("--policies", default="base,random,rule,scalar,eacc,eacc_pos")
    ap.add_argument("--regimes", default="normal,scarce")
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    if args.stage == "expl":
        seeds0 = {"normal": 44000, "scarce": 45000}
        out = os.path.join(_HERE, "runs", "critic_loop_expl.jsonl")
    else:
        seeds0 = {"normal": 46000, "scarce": 47000}
        out = os.path.join(_HERE, "runs", "critic_loop_confirm.jsonl")
    evaluate(args.policies.split(","), args.regimes.split(","), seeds0,
             args.seeds, args.workers, out, note=f"{args.stage}/{args.seeds}种子")


if __name__ == "__main__":
    main()

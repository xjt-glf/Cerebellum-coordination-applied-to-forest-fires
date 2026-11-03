# -*- coding: utf-8 -*-
"""阶段 B 核心：从可观测证据预测编辑收益（学习型建议 vs 规则 vs 随机）。

问题设定（对应指南 §5.1/§6）：
  输入  : 决策边界的可观测证据 + 候选编辑描述（全部部署期可知）
  输出  : 预测 Δ(任务指标)@60 —— 修改的真实收益
  评估  : (a) 回归质量（按种子聚类的测试集）
          (b) 决策规则：预测 > 阈值才应用；对比 always/never/random-rule/oracle
          (c) 标定：预测分箱 vs 真实均值
  分层  : 全部按**种子（回合）**切分，同一回合的配对永不跨集合。

两级分解（screen1 的发现：+20 机制指标更机械、+60 任务收益受级联放大）：
  M1: 预测 Δcov_mean@20 / Δrknown@20（机制层，供"依据"）
  M2: 预测 Δext@60 / Δburn@60（任务层，供"预测收益"）
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

EDIT_TYPES = ["rule_charge_lowest", "rule_relay_return_fix", "rule_relay_uncovered",
              "rule_redundant_relay", "rule_patrol_dedupe", "rule_explore_redirect",
              "rule_ring_auto", "ctrl_random_edit"]


def load_pairs(run_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(run_dir, "pairs_s*.jsonl"))):
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["edit"] == "__noop__" or not r["order_changed"] or r["censored"]:
                    continue
                rows.append(r)
    return rows


# ------------------------------------------------------------ 特征
def features(rec):
    ctx = rec["ctx"]
    s = ctx
    f = []
    # ---- 全局态势
    f += [s["t"] / 600.0, s["explored"], s["discovery"], s["coverage"],
          s["rescuer_known"], s["burned"] * 100.0, min(s["max_aoi"], 100) / 100.0,
          s["n_clusters"] / 6.0]
    sizes = [c[2] for c in s["clusters"]]
    f += [len(sizes) / 6.0, (sum(sizes) or 0) / 400.0, (max(sizes) if sizes else 0) / 200.0]
    # ---- 人员
    rs = s["rescuers"]
    unc = [r for r in rs if not r[3]]
    claimed = {o["rescuer_id"] for o in s["orders"]
               if o["task"] == 1 and o["rescuer_id"] is not None}
    unc_unclaimed = [r for r in unc if r[0] not in claimed]
    f += [len(unc) / len(rs), len(unc_unclaimed) / len(rs),
          (max((r[4] for r in unc), default=0)) / 50.0,
          float(np.mean([r[4] for r in rs])) / 50.0]
    # ---- 编辑机 & 新指令语义
    et = rec["edit"]
    onehot = [1.0 if et == e else 0.0 for e in EDIT_TYPES]
    n_new = [o for o in rec["new_orders"]]
    changed_idx = [i for i, (a, b) in enumerate(zip(rec["base_orders"], n_new))
                   if a != b]
    ci = changed_idx[0] if changed_idx else 0
    new_o, old_o = n_new[ci], rec["base_orders"][ci]
    uav = s["uavs"][ci]
    f += onehot
    f += [len(changed_idx) / 4.0]
    f += [uav[3] / 600.0, uav[4] / 200.0]                       # e, margin
    task_onehot = [0.0, 0.0, 0.0]
    tn = {"fire_sensing": 0, "comm_relay": 1, "recharge": 2}.get(new_o["task"], 0)
    task_onehot[tn] = 1.0
    f += task_onehot
    # 新目标的机制量：relay→新人员 AoI/距离；explore→新旧目标距离；charge→到站距离
    if tn == 1 and new_o.get("rescuer_id"):
        tgt_r = [r for r in rs if r[0] == new_o["rescuer_id"]]
        if tgt_r:
            d = float(np.hypot(tgt_r[0][1] - uav[1], tgt_r[0][2] - uav[2]))
            f += [tgt_r[0][4] / 50.0, d / 140.0, 1.0 if tgt_r[0][3] else 0.0]
        else:
            f += [0.0, 0.0, 0.0]
    else:
        f += [0.0, 0.0, 0.0]
    # ---- 机制量：人员/无人机 ↔ 未合围火点的几何（relay→知情→灭火的因果链）
    cl = s["clusters"]                      # [cx, cy, size]
    if tn == 1 and new_o.get("rescuer_id"):
        tgt_r = [r for r in rs if r[0] == new_o["rescuer_id"]]
        if tgt_r and cl:
            rx, ry = tgt_r[0][1], tgt_r[0][2]
            ds = [float(np.hypot(rx - c[0], ry - c[1])) for c in cl]
            j = int(np.argmin(ds))
            others = sum(1 for r in rs
                         if np.hypot(r[1] - cl[j][0], r[2] - cl[j][1]) < 15.0)
            f += [ds[j] / 140.0, cl[j][2] / 200.0, min(others, 4) / 4.0,
                  ds[j] / 140.0 * cl[j][2] / 200.0]
        else:
            f += [0.0, 0.0, 0.0, 0.0]
    else:
        f += [0.0, 0.0, 0.0, 0.0]
    if cl:
        du = [float(np.hypot(uav[1] - c[0], uav[2] - c[1])) for c in cl]
        f += [min(du) / 140.0]
    else:
        f += [1.0]
    if tn == 0 and new_o.get("target") and old_o.get("target"):
        d = float(np.hypot(new_o["target"][0] - old_o["target"][0],
                           new_o["target"][1] - old_o["target"][1]))
        f += [d / 140.0]
    else:
        f += [0.0]
    # 被改机的当前任务
    old_task = [0.0, 0.0, 0.0]
    otn = {"fire_sensing": 0, "comm_relay": 1, "recharge": 2}.get(old_o["task"], 0)
    old_task[otn] = 1.0
    f += old_task
    # 队伍构成
    tasks_now = [o["task"] for o in rec["base_orders"]]
    f += [tasks_now.count("fire_sensing") / 4.0, tasks_now.count("comm_relay") / 4.0,
          tasks_now.count("recharge") / 4.0]
    # 中继机里 margin 最低者
    relay_ms = [s["uavs"][i][4] for i, o in enumerate(rec["base_orders"])
                if o["task"] == 1]
    f += [min(relay_ms) / 200.0 if relay_ms else 1.0]
    return np.asarray(f, dtype=np.float32)


TARGETS = dict(
    ext60=lambda r: r["w60_br"]["extinguished_cells"] - r["w60_base"]["extinguished_cells"],
    burn60=lambda r: r["w60_br"]["burned_cells"] - r["w60_base"]["burned_cells"],
    cov20=lambda r: r["w20_br"]["cov_mean"] - r["w20_base"]["cov_mean"],
    rknown20=lambda r: r["w20_br"]["rknown"] - r["w20_base"]["rknown"],
    cov60=lambda r: r["w60_br"]["cov_mean"] - r["w60_base"]["cov_mean"],
)


def build_xy(rows):
    X = np.stack([features(r) for r in rows])
    Y = {k: np.array([f(r) for r in rows], dtype=np.float32) for k, f in TARGETS.items()}
    seeds = np.array([r["seed"] for r in rows])
    return X, Y, seeds


# ------------------------------------------------------------ 模型
def train_mlp(Xtr, ytr, Xva, yva, hidden=64, epochs=300, lr=1e-3, wd=1e-4,
              seed=0, classify=False):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    def nz(x):
        return (x - mu) / sd
    net = nn.Sequential(nn.Linear(Xtr.shape[1], hidden), nn.Tanh(),
                        nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.as_tensor(nz(Xtr), dtype=torch.float32)
    yt = torch.as_tensor(ytr, dtype=torch.float32).unsqueeze(1)
    Xv = torch.as_tensor(nz(Xva), dtype=torch.float32)
    best, best_sd = float("inf"), None
    for ep in range(epochs):
        net.train()
        opt.zero_grad()
        if classify:
            logits = net(Xt).squeeze(1)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, (yt.squeeze(1) > 3.0).float())
        else:
            out = net(Xt)
            loss = nn.functional.huber_loss(out, yt, delta=5.0)
        loss.backward()
        opt.step()
        with torch.no_grad():
            net.eval()
            if classify:
                lv = nn.functional.binary_cross_entropy_with_logits(
                    net(Xv).squeeze(1),
                    torch.as_tensor(yva > 3.0, dtype=torch.float32))
            else:
                lv = float(torch.sqrt(torch.mean((net(Xv).squeeze(1)
                    - torch.as_tensor(yva, dtype=torch.float32)) ** 2)))
            if lv < best:
                best, best_sd = lv, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_sd)
    return (lambda X: net(torch.as_tensor(nz(X), dtype=torch.float32))
            .squeeze(1).detach().numpy()), best


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


# ------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="screen40")
    ap.add_argument("--target", default="ext60")
    ap.add_argument("--train-seeds", type=int, default=28)
    ap.add_argument("--val-seeds", type=int, default=4)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(4)
    rows = load_pairs(os.path.join(_HERE, "runs", args.run))
    X, Y, seeds = build_xy(rows)
    useeds = sorted(set(seeds.tolist()))
    print(f"[data] {len(rows)} pairs, {len(useeds)} seeds, dim={X.shape[1]}")
    rng = np.random.RandomState(0)
    rng.shuffle(useeds)
    tr_s = set(useeds[:args.train_seeds])
    va_s = set(useeds[args.train_seeds:args.train_seeds + args.val_seeds])
    te_s = set(useeds[args.train_seeds + args.val_seeds:])
    m_tr = np.isin(seeds, list(tr_s))
    m_va = np.isin(seeds, list(va_s))
    m_te = np.isin(seeds, list(te_s))
    print(f"[split] train {m_tr.sum()} (/{len(tr_s)} seeds), val {m_va.sum()}, test {m_te.sum()}")

    tgt = args.target
    y = Y[tgt]

    # ---- 回归
    pred_fn, val_rmse = train_mlp(X[m_tr], y[m_tr], X[m_va], y[m_va])
    p_te = pred_fn(X[m_te])
    base_rmse = float(np.sqrt(np.mean((y[m_te].mean() - y[m_te]) ** 2)))
    print(f"\n[regress:{tgt}] test RMSE {np.sqrt(np.mean((p_te - y[m_te])**2)):.3f} "
          f"(常数基线 {base_rmse:.3f}) | pearson {np.corrcoef(p_te, y[m_te])[0,1]:.3f} "
          f"| spearman {spearman(p_te, y[m_te]):.3f}")

    # ---- 分类：收益 > +3 格
    clf_fn, _ = train_mlp(X[m_tr], y[m_tr], X[m_va], y[m_va], classify=True)
    pcls = clf_fn(X[m_te])
    truth = (y[m_te] > 3.0)
    for thr in (0.3, 0.5, 0.7):
        sel = pcls > thr
        if sel.sum() == 0:
            continue
        prec = float(truth[sel].mean())
        rec = float(truth[sel].sum() / max(1, truth.sum()))
        gain = float(y[m_te][sel].mean())
        print(f"[classify gain>3] thr={thr}: 选中 {int(sel.sum())}/{len(sel)} "
              f"precision {prec:.2f} recall {rec:.2f} 选中集真实均值 {gain:+.2f}")

    # ---- 决策规则（按测试回合聚合）：候选集中选预测最优，低于阈值不选
    print("\n[决策规则] 每个决策点从候选编辑中选 1 个或全拒（真实值 = Δext@60）")
    te_rows = [r for r, s in zip(rows, seeds) if s in te_s]
    from collections import defaultdict
    by_bt = defaultdict(list)
    for r in te_rows:
        by_bt[(r["seed"], r["t"])].append(r)

    def evaluate(policy, name):
        gains, applied, negs = [], 0, 0
        cov_effects, burn_effects = [], []
        for key, cands in by_bt.items():
            pick = policy(cands)
            if pick is None:
                gains.append(0.0)
                continue
            applied += 1
            d_ext = TARGETS["ext60"](pick)
            gains.append(d_ext)
            if d_ext < -5:
                negs += 1
            cov_effects.append(TARGETS["cov60"](pick))
            burn_effects.append(TARGETS["burn60"](pick))
        gains = np.array(gains)
        print(f"  {name:<28s} 应用率 {applied/len(by_bt):.2f} "
              f"总收益 {gains.sum():+7.1f} 均值 {gains.mean():+6.2f} "
              f"负收益编辑 {negs}/{applied}  Δcov {np.mean(cov_effects) if cov_effects else 0:+.3f} "
              f"Δburn {np.mean(burn_effects) if burn_effects else 0:+.2f}")
        return gains

    feat_cache = {id(r): features(r) for r in te_rows}

    def policy_pred(cands):
        ps = [float(pred_fn(feat_cache[id(r)][None])[0]) for r in cands]
        i = int(np.argmax(ps))
        return cands[i] if ps[i] > 0 else None

    def policy_pred_cls(cands):
        ps = [float(clf_fn(feat_cache[id(r)][None])[0]) for r in cands]
        i = int(np.argmax(ps))
        return cands[i] if ps[i] > 0.5 else None

    def policy_rule_first(cands):
        rules = [r for r in cands if r["edit"].startswith("rule_")]
        return rules[0] if rules else None

    def policy_random(cands):
        return cands[np.random.RandomState(len(cands) * 7 + 3).randint(len(cands))]

    def policy_oracle(cands):
        return max(cands, key=lambda r: TARGETS["ext60"](r))

    evaluate(lambda c: None, "never-apply")
    evaluate(policy_rule_first, "rule-first（规则优先）")
    evaluate(policy_random, "random-choice")
    evaluate(policy_pred, "MLP-regress(阈值0)")
    evaluate(policy_pred_cls, "MLP-classify(0.5)")
    evaluate(policy_oracle, "oracle（上界）")

    # ---- 标定
    print("\n[标定] 预测分箱 → 真实均值")
    qs = np.percentile(p_te, [0, 20, 40, 60, 80, 100])
    for i in range(5):
        m = (p_te >= qs[i]) & (p_te <= qs[i + 1])
        if m.any():
            print(f"  pred∈[{qs[i]:+.2f},{qs[i+1]:+.2f}] n={int(m.sum()):4d} "
                  f"真实均值 {y[m_te][m].mean():+7.2f}")


if __name__ == "__main__":
    main()

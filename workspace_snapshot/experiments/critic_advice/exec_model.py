# -*- coding: utf-8 -*-
"""EACC 模型训练与评估。

三个模型（同一套特征/数据管道，按种子切分）：
  exec    : 执行模型  f(特征) -> 段末机制配对差 Δmech@20（40 维）
  value   : 任务价值头 g(特征, 预测Δmech) -> Δext@episode（吃执行模型的预测，部署一致）
  scalar  : 标量对照头 h(特征) -> Δext@episode（ReAd++ 式：不看执行响应直接回归）

特征全部部署期可观测（critic_data.build_features）；标签全部配对真值（分支B−基线A）。
按**种子**切分 train/val/test（同回合样本永不跨集合）。
"""
import argparse
import glob
import gzip
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from critic_data import FEATURE_DIM, build_features, MECH_KEYS

MECH_DIM = len(MECH_KEYS) + 3 * 8       # 39
COV_IDX = MECH_DIM                       # Δcov_mean@20 拼在最后
LABEL_DIM = MECH_DIM + 1                 # 40

DATA_DIR = os.path.join(_HERE, "runs")
MODEL_DIR = os.path.join(DATA_DIR, "critic_models")


# ============================================================ 数据
def load_records(mode, skip_bad=True):
    rows = []
    for p in sorted(glob.glob(os.path.join(DATA_DIR, f"critic_data_{mode}",
                                            "*_s*.jsonl.gz"))):
        try:
            with gzip.open(p, "rt", encoding="utf-8") as f:
                rows.append([json.loads(line) for line in f])
        except (EOFError, json.JSONDecodeError, OSError) as e:
            if skip_bad:
                print(f"  [skip 未完成文件] {os.path.basename(p)}: {type(e).__name__}")
                continue
            raise
    return [r for chunk in rows for r in chunk]


def xy_exec(rows):
    X, Y, S, R = [], [], [], []
    for r in rows:
        X.append(build_features(r["ctx"], r["base_orders"], r["new_orders"]))
        y = [a - b for a, b in zip(r["m20_br"], r["m20_base"])]
        y.append(r["cov20_br"] - r["cov20_base"])
        Y.append(y)
        S.append(r["seed"])
        R.append(r["regime"])
    return (np.asarray(X, np.float32), np.asarray(Y, np.float32),
            np.asarray(S), np.asarray(R))


def xy_episode(rows, exec_pred_fn=None):
    """episode 模式记录：标签 = 整局配对差。"""
    X, Y, S, R = [], [], [], []
    for r in rows:
        X.append(build_features(r["ctx"], r["base_orders"], r["new_orders"]))
        dmech = [a - b for a, b in zip(r["m20_br"], r["m20_base"])]
        dmech.append(r["cov20_br"] - r["cov20_base"])
        if exec_pred_fn is not None:
            dmech = exec_pred_fn(X[-1])
        Y.append(r["ep_br"]["extinguished_cells"] - r["ep_base"]["extinguished_cells"])
        S.append(r["seed"])
        R.append(r["regime"])
    return (np.asarray(X, np.float32), np.asarray(Y, np.float32),
            np.asarray(S), np.asarray(R))


def split_by_seed(X, Y, S, R, frac=(0.7, 0.15, 0.15), seed=0):
    rng = np.random.RandomState(seed)
    masks = {}
    for regime in ("normal", "scarce"):
        useeds = sorted(set(S[(R == regime)].tolist()))
        rng.shuffle(useeds)
        n = len(useeds)
        n_tr = int(frac[0] * n)
        n_va = int(frac[1] * n)
        masks[regime] = dict(
            tr=set(useeds[:n_tr]), va=set(useeds[n_tr:n_tr + n_va]),
            te=set(useeds[n_tr + n_va:]))
    out = {}
    for part in ("tr", "va", "te"):
        m = np.zeros(len(S), bool)
        for regime, d in masks.items():
            m |= np.isin(S, list(d[part])) & (R == regime)
        out[part] = m
    return out


# ============================================================ 模型
def make_mlp(in_dim, out_dim, hidden=256, layers=2):
    import torch
    import torch.nn as nn
    mods = [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.Tanh()]
    for _ in range(layers - 1):
        mods += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.Tanh()]
    mods.append(nn.Linear(hidden, out_dim))
    net = nn.Sequential(*mods)
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.zeros_(m.bias)
    return net


def train_mlp(Xtr, ytr, Xva, yva, out_dim, hidden=256, epochs=400, lr=1e-3,
              batch=1024, seed=0, huber=1.0, l2=1e-5):
    import torch
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    ymu, ysd = ytr.mean(0), ytr.std(0) + 1e-6
    nz = lambda x: (x - mu) / sd
    nzy = lambda y: (y - ymu) / ysd
    net = make_mlp(Xtr.shape[1], out_dim, hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=l2)
    Xt = torch.as_tensor(nz(Xtr), dtype=torch.float32)
    Yt = torch.as_tensor(nzy(ytr), dtype=torch.float32)
    Xv = torch.as_tensor(nz(Xva), dtype=torch.float32)
    Yv = torch.as_tensor(nzy(yva), dtype=torch.float32)
    best, best_sd = float("inf"), None
    n = len(Xt)
    for ep in range(epochs):
        net.train()
        idx = torch.randperm(n)
        for b in range(0, n, batch):
            j = idx[b:b + batch]
            out = net(Xt[j])
            loss = torch.nn.functional.huber_loss(out, Yt[j], delta=huber)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        with torch.no_grad():
            net.eval()
            lv = float(torch.sqrt(torch.mean((net(Xv) - Yv) ** 2)))
            if lv < best:
                best, best_sd = lv, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_sd)
    return net, dict(mu=mu, sd=sd, ymu=ymu, ysd=ysd, val_rmse_norm=best)


class StandardModel:
    """训练产物包（net + 标准化参数 + 预测函数）。不持有模块引用，可 pickle。"""
    def __init__(self, net, stats):
        self.net = net
        self.s = stats

    def predict(self, X):
        import torch
        X = np.atleast_2d(np.asarray(X, np.float32))
        z = (X - self.s["mu"]) / self.s["sd"]
        with torch.no_grad():
            out = self.net(torch.as_tensor(z, dtype=torch.float32)).numpy()
        return out * self.s["ysd"] + self.s["ymu"]


def save_bundle(path, nets, meta):
    import torch
    objs = []
    for m in nets:
        objs.append(dict(net=m.net.state_dict(), **{k: v for k, v in m.s.items()}))
    torch.save(dict(models=objs, meta=meta, feature_dim=FEATURE_DIM,
                    label_dim=LABEL_DIM), path)


def load_bundle(path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    nets = []
    extra = ck["label_dim"] if ck["meta"].get("uses_exec_pred") else 0
    for o in ck["models"]:
        net = make_mlp(ck["feature_dim"] + extra, ck["meta"]["out_dim"],
                       ck["meta"].get("hidden", 256))
        net.load_state_dict(o["net"])
        net.eval()
        nets.append(StandardModel(net, dict(mu=o["mu"], sd=o["sd"],
                                            ymu=o["ymu"], ysd=o["ysd"])))
    return nets, ck["meta"]


# ============================================================ 评估
def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def report_exec(pred, truth, tag):
    keys = list(MECH_KEYS) + [f"uav{i}_{c}" for i in range(8) for c in ("x", "y", "e")] + ["cov20"]
    rows = []
    base_rmse = np.sqrt(np.mean((truth.mean(0) - truth) ** 2, 0))
    pr = np.sqrt(np.mean((pred - truth) ** 2, 0))
    for k in range(LABEL_DIM):
        pe = pearson(pred[:, k], truth[:, k])
        rows.append((keys[k], pe, pr[k], base_rmse[k]))
    print(f"\n[{tag}] 逐维 pearson / RMSE(模型 vs 常数):")
    for name, pe, r1, r2 in rows:
        flag = "  *" if pe > 0.5 else ""
        print(f"  {name:<14s} pearson {pe:+.3f}  rmse {r1:8.4f} (常数 {r2:8.4f}){flag}")
    macro = np.nanmean([r[1] for r in rows])
    print(f"  宏平均 pearson {macro:+.3f}")
    return rows


# ============================================================ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["exec", "value", "scalar"], required=True)
    ap.add_argument("--seeds-train-frac", type=float, default=0.7)
    ap.add_argument("--ensemble", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=400)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(16)
    os.makedirs(MODEL_DIR, exist_ok=True)

    if args.model == "exec":
        rows = load_records("exec")
        X, Y, S, R = xy_exec(rows)
        print(f"[exec data] {len(rows)} 条 / {len(set(S))} 种子 / dim {X.shape[1]}")
        sp = split_by_seed(X, Y, S, R)
        nets, stats_rows = [], []
        for k in range(args.ensemble):
            net, st = train_mlp(X[sp["tr"]], Y[sp["tr"]], X[sp["va"]], Y[sp["va"]],
                                LABEL_DIM, hidden=args.hidden, epochs=args.epochs,
                                seed=k)
            nets.append(StandardModel(net, st))
        pred = np.mean([m.predict(X[sp["te"]]) for m in nets], 0)
        truth = Y[sp["te"]]
        rows_rep = report_exec(pred, truth, f"exec 集成{args.ensemble} (test {sp['te'].sum()})")
        save_bundle(os.path.join(MODEL_DIR, "exec_model.pt"), nets,
                    dict(out_dim=LABEL_DIM, hidden=args.hidden, kind="exec",
                         test_seeds=sorted(set(S[sp["te"]].tolist()))))
        # 特征消融：只给全局标量+计划计数（类 45 维平面特征）
        flat = np.zeros_like(X); flat[:, :16] = X[:, :16]
        netf, stf = train_mlp(flat[sp["tr"]], Y[sp["tr"]], flat[sp["va"]], Y[sp["va"]],
                              LABEL_DIM, hidden=128, epochs=args.epochs, seed=0)
        predf = StandardModel(netf, stf).predict(flat[sp["te"]])
        print("\n[消融：平面特征(仅全局+计划计数)] 宏平均 pearson "
              f"{np.nanmean([pearson(predf[:, k], truth[:, k]) for k in range(LABEL_DIM)]):+.3f}")
        with open(os.path.join(MODEL_DIR, "exec_stats.json"), "w", encoding="utf-8") as f:
            json.dump(dict(rows=[(r[0], float(r[1]), float(r[2]), float(r[3])) for r in rows_rep],
                           n=len(rows), test=int(sp["te"].sum())), f, ensure_ascii=False, indent=1)

    else:  # value / scalar
        rows = load_records("episode")
        if args.model == "value":
            exec_nets, meta = load_bundle(os.path.join(MODEL_DIR, "exec_model.pt"))
            exec_pred = lambda x: np.mean([m.predict(x) for m in exec_nets], 0)
        else:
            exec_pred = None
        X, Y, S, R = xy_episode(rows, exec_pred_fn=exec_pred)
        print(f"[{args.model} data] {len(rows)} 条 / {len(set(S))} 种子")
        sp = split_by_seed(X, Y, S, R)
        in_dim = X.shape[1] + (LABEL_DIM if exec_pred is not None else 0)
        nets = []
        for k in range(args.ensemble):
            if exec_pred is not None:
                Xe = np.concatenate([X, exec_pred(X)], 1)
            else:
                Xe = X
            net, st = train_mlp(Xe[sp["tr"]], Y[sp["tr"]][:, None],
                                Xe[sp["va"]], Y[sp["va"]][:, None],
                                1, hidden=args.hidden, epochs=args.epochs, seed=k)
            nets.append(StandardModel(net, st))
        if exec_pred is not None:
            Xte = np.concatenate([X[sp["te"]], exec_pred(X[sp["te"]])], 1)
        else:
            Xte = X[sp["te"]]
        pred = np.mean([m.predict(Xte) for m in nets], 0).ravel()
        truth = Y[sp["te"]]
        base = np.sqrt(np.mean((truth.mean() - truth) ** 2))
        rm = np.sqrt(np.mean((pred - truth) ** 2))
        print(f"\n[{args.model}] test n={len(truth)}  pearson {pearson(pred, truth):+.3f}  "
              f"rmse {rm:.3f} (常数 {base:.3f})  truth mean {truth.mean():+.2f} std {truth.std():.2f}")
        # 标定
        qs = np.percentile(pred, [0, 20, 40, 60, 80, 100])
        for i in range(5):
            m = (pred >= qs[i]) & (pred <= qs[i + 1])
            if m.any():
                print(f"  pred∈[{qs[i]:+.2f},{qs[i+1]:+.2f}] n={int(m.sum()):4d} "
                      f"真实均值 {truth[m].mean():+7.2f}")
        # 分类视角：预测 top 分位 vs 底分位的真实均值
        top = pred > np.percentile(pred, 75)
        bot = pred < np.percentile(pred, 25)
        print(f"  顶 25% 预测的真实均值 {truth[top].mean():+.2f} / 底 25% {truth[bot].mean():+.2f}")
        save_bundle(os.path.join(MODEL_DIR, f"{args.model}_head.pt"), nets,
                    dict(out_dim=1, hidden=args.hidden, kind=args.model,
                         uses_exec_pred=exec_pred is not None,
                         test_seeds=sorted(set(S[sp["te"]].tolist()))))


if __name__ == "__main__":
    main()

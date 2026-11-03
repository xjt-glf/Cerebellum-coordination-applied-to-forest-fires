# -*- coding: utf-8 -*-
"""ReAd critic：给"上层任务分配"打分，用于 LLM 重规划门限。

参考 ReAd（Towards Efficient LLM Grounding for Embodied Multi-Agent Collaboration）：
在**冻结的低层执行器**下，用离线数据集 MC 回归一个 Q_θ(s, tau)，
然后用增量优势 `S_ReAd(tau_i) = Q^{1:i} - Q^{1:i-1}` 作为反馈，
低于阈值 alpha 就让大模型重新规划。

与原论文的两处偏离（容器内无法下载 BERT 权重，且我们的指令是结构化的）：
  1. 原文用**冻结 BERT** 编码自然语言子任务描述；这里改用 `UavOrder.to_vector()`
     的**结构化编码**（每机 10 维 + 存在位）。我们的指令本来就是结构化 JSON，
     不需要先转成句子再用语言模型编码，信息更全且没有分词歧义。
  2. 原文的 Q 只吃"状态 + 子任务序列"；这里的状态额外带了全图 10x10 双通道粗视图，
     因为本任务的核心是"还有哪片没探索 / 哪里有火"，缺了它无法判断一个搜寻目标好不好。

三个子命令：
    python3 lowlevel/read_critic.py collect --episodes 800 --workers 8
    python3 lowlevel/read_critic.py train   --epochs 60
    python3 lowlevel/read_critic.py eval
"""
import argparse
import math
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lowlevel.task_spec import (
    N_TASKS, N_SENSE_MODES, UavOrder, TeamPlan,
    TASK_FIRE_SENSING, TASK_COMM_RELAY, TASK_RECHARGE,
    SENSE_EXPLORE, SENSE_PATROL,
)

MAX_SLOTS = 8                      # 与观测里的 agent-id one-hot 上限一致
COARSE = 10
ORDER_DIM = UavOrder.VEC_DIM       # 10
# 高层状态：标量(9) + 粗视图双通道(200) + 每机 5 维(40) + 人员聚合(5)
STATE_DIM = 9 + 2 * COARSE * COARSE + 5 * MAX_SLOTS + 5
# 计划：每机 (order 10 + 存在位 1) + 任务/模式计数(5)
PLAN_DIM = MAX_SLOTS * (ORDER_DIM + 1) + 5

GAMMA_H = 0.95                     # 高层折扣（一个 segment = replan_every 步）
KINDS = ("planner", "perturb", "random", "degenerate")


# ============================================================ 编码
def encode_state(env):
    """把 TaskConditionedEnv 的当前态势编码成固定长度的高层状态向量。"""
    from lowlevel.obs_builder import shared_cache, assigned_station
    e = env.env
    c = shared_cache(e)
    H, W = e.grid_size
    e_max = float(getattr(e, "uav_energy", 600.0)) or 600.0
    cost = e.energy_cost_per_step(0) if hasattr(e, "energy_cost_per_step") else 2.5

    v = [float(env.t) / max(1, env.max_steps),
         c["explored_rate"],
         float(getattr(e, "active_discovery_rate", 0.0)),
         float(getattr(e, "burned_area_rate", 0.0)),
         float(getattr(e, "coverage_rate", 0.0)),
         float(getattr(e, "active_rescuer_known_rate", 0.0)),
         min(1.0, float(getattr(e, "cur_max_AoI", 0)) / 50.0),
         min(1.0, len(c["clusters"]) / 6.0),
         min(1.0, c["n_contained"] / 6.0)]
    v += list(np.asarray(c["coarse"], dtype=np.float32))

    for i in range(MAX_SLOTS):
        if i < len(e.uavs):
            u = e.uavs[i]
            _st, d = assigned_station(e, i)
            v += [float(u.x) / H, float(u.y) / W, float(u.e) / e_max,
                  float(np.clip((float(u.e) - cost * d) / e_max, -1.0, 1.0)), 1.0]
        else:
            v += [0.0, 0.0, 0.0, 0.0, 0.0]

    rs = getattr(e, "rescuers", [])
    if rs:
        xs = np.array([r.x for r in rs], dtype=np.float64)
        ys = np.array([r.y for r in rs], dtype=np.float64)
        v += [float(xs.mean()) / H, float(ys.mean()) / W,
              min(1.0, float(np.hypot(xs.std(), ys.std())) / 30.0),
              float(np.mean([1.0 if r.is_covered else 0.0 for r in rs])),
              min(1.0, float(np.mean([r.AoI for r in rs])) / 50.0)]
    else:
        v += [0.0, 0.0, 0.0, 0.0, 0.0]
    return np.asarray(v, dtype=np.float32)


def encode_plan(env, orders, n_prefix=None):
    """把全队指令编码成定长向量。n_prefix 指定只保留前 n 个子任务（ReAd-S 的增量打分）。"""
    from lowlevel.obs_builder import resolve_order_target
    e = env.env
    H, W = e.grid_size
    dmax = math.hypot(float(H), float(W))
    n = len(orders)
    k = n if n_prefix is None else max(0, min(n, int(n_prefix)))
    v, cnt = [], [0.0] * 5
    for i in range(MAX_SLOTS):
        if i < k:
            o = orders[i]
            u = e.uavs[i]
            tgt, is_fire, ring = resolve_order_target(e, i, o)
            v += list(o.to_vector(float(u.x), float(u.y), dmax,
                                 resolved_target=tgt, target_is_fire=is_fire, ring=ring))
            v.append(1.0)
            t = int(o.task)
            cnt[t] += 1.0
            if t == TASK_FIRE_SENSING:
                cnt[3 + int(o.mode)] += 1.0
        else:
            v += [0.0] * ORDER_DIM + [0.0]
    denom = float(max(1, n))
    v += [x / denom for x in cnt]
    return np.asarray(v, dtype=np.float32)


# ============================================================ 高层段奖励
class SegReward:
    """一个 segment（replan_every 步）的高层回报：只看任务指标的增量。

    低层 reward.py 里那套 shaping 是给"怎么飞"用的，不能当高层信号——
    高层要评的是"这套分工在这个态势下值不值"，所以只用可交付指标的差分。
    """
    W_EXPLORED = 6.0
    W_DISCOVERY = 4.0
    W_RKNOWN = 3.0
    W_EXT = 1.0 / 100.0        # 每 100 个格子被扑灭 = +1
    W_CONTAIN = 0.5
    W_COVER = 0.5              # segment 内平均覆盖率
    W_BURNED = -4.0
    W_DEPLETE = -0.5

    KEYS = ("explored_rate", "active_discovery_rate", "active_rescuer_known_rate",
            "burned_area_rate")

    def snapshot(self, env, info=None):
        e = env.env
        s = {k: float(getattr(e, k, 0.0)) for k in self.KEYS}
        s["ext"] = float((e.grid["burning_state"] == 3).sum())
        s["contained"] = float(getattr(e, "n_contained_clusters", 0))
        s["depleted"] = float(sum(1 for u in e.uavs if float(u.e) <= 0.0))
        return s

    def value(self, s0, s1, cover_mean):
        return float(
            self.W_EXPLORED * (s1["explored_rate"] - s0["explored_rate"])
            + self.W_DISCOVERY * (s1["active_discovery_rate"] - s0["active_discovery_rate"])
            + self.W_RKNOWN * (s1["active_rescuer_known_rate"] - s0["active_rescuer_known_rate"])
            + self.W_EXT * (s1["ext"] - s0["ext"])
            + self.W_CONTAIN * (s1["contained"] - s0["contained"])
            + self.W_COVER * cover_mean
            + self.W_BURNED * (s1["burned_area_rate"] - s0["burned_area_rate"])
            + self.W_DEPLETE * max(0.0, s1["depleted"] - s0["depleted"])
        )


# ============================================================ 计划采样（多样化）
def sample_plan(env, rng, kind=None):
    """故意做成混合分布：好计划、被扰动的计划、纯随机、退化计划都要有。

    只喂规划器的好计划，Q 会学成常数（所有 plan 都好），增量优势就没有区分度。
    """
    n = env.num_uavs
    if kind is None:
        kind = rng.choices(list(KINDS),
                           weights=[0.40, 0.25, 0.20, 0.15])[0]
    if kind == "planner":
        return env._planner_orders(), kind
    if kind == "perturb":
        orders = env._planner_orders()
        i = rng.randrange(n)
        orders[i] = _random_order(env, rng)
        return orders, kind
    if kind == "degenerate":
        t = rng.choice([TASK_FIRE_SENSING, TASK_COMM_RELAY, TASK_RECHARGE])
        orders = []
        for _ in range(n):
            o = UavOrder(task=t)
            if t == TASK_FIRE_SENSING:
                o.mode = rng.choice([SENSE_EXPLORE, SENSE_PATROL])
            orders.append(o)
        return orders, kind
    return [_random_order(env, rng) for _ in range(n)], kind


def _random_order(env, rng):
    from lowlevel.obs_builder import shared_cache
    e = env.env
    H, W = e.grid_size
    o = UavOrder(task=rng.randrange(N_TASKS))
    o.proactive_handover = bool(rng.random() < 0.5)
    if o.task == TASK_FIRE_SENSING:
        o.mode = rng.choice([SENSE_EXPLORE, SENSE_PATROL])
        if rng.random() < 0.7:
            if o.mode == SENSE_PATROL and shared_cache(e)["clusters"] and rng.random() < 0.6:
                cl = rng.choice(shared_cache(e)["clusters"])
                o.target = tuple(cl["centroid"])
            else:
                o.target = (float(rng.randrange(H)), float(rng.randrange(W)))
    elif o.task == TASK_COMM_RELAY:
        rs = getattr(e, "rescuers", [])
        if rs and rng.random() < 0.8:
            o.rescuer_id = int(getattr(rng.choice(rs), "rescuer_id", 1))
    return o


# ============================================================ 数据采集
def _make_env(args, seed):
    from lowlevel.env_wrapper import TaskConditionedEnv
    return TaskConditionedEnv(num_uavs=args.num_uavs, max_steps=args.max_steps,
                              grid_size=(args.grid, args.grid),
                              n_fire_seeds=args.fire_seeds, spread_gain=args.spread_gain,
                              num_rescuers=args.num_rescuers, seed=seed,
                              task_sampler="planner",
                              replan_every=0)          # 重规划由本脚本掌握


def _load_actor(path, device="cpu"):
    import torch
    from lowlevel.models import Actor
    from lowlevel.obs_builder import OBS_DIM
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and "actor" in sd:
        sd = sd["actor"]
    actor = Actor(OBS_DIM, 9, int(sd["head.weight"].shape[1]), 2).to(device)
    actor.load_state_dict(sd)
    actor.eval()
    return actor


def collect_episodes(args, seed0, n_ep, out_q=None):
    """跑 n_ep 个回合，每个 segment 采一条 (state, plan 前缀编码, 段奖励)。"""
    import random
    import torch
    torch.set_num_threads(1)
    rng = random.Random(seed0)
    actor = _load_actor(args.actor)
    env = _make_env(args, seed0)
    segr = SegReward()
    rows = []
    for ep in range(n_ep):
        obs = env.reset()
        ep_rows = []
        while True:
            orders, kind = sample_plan(env, rng)
            obs = env.set_plan(orders)
            s_vec = encode_state(env)
            plans = np.stack([encode_plan(env, orders, n_prefix=k)
                              for k in range(len(orders) + 1)])
            s0 = segr.snapshot(env)
            cov, done = [], False
            for _ in range(args.horizon):
                with torch.no_grad():
                    a, _, _ = actor.act(torch.as_tensor(obs), deterministic=False)
                obs, _r, done, info = env.step(a.numpy().tolist())
                cov.append(float(info.get("coverage_rate", 0.0)))
                if done:
                    break
            r = segr.value(s0, segr.snapshot(env), float(np.mean(cov)) if cov else 0.0)
            ep_rows.append((s_vec, plans, r, kind))
            if done:
                break
        # 段内 MC 回报
        g = 0.0
        rets = []
        for _s, _p, r, _k in reversed(ep_rows):
            g = r + GAMMA_H * g
            rets.append(g)
        rets.reverse()
        for (s_vec, plans, r, kind), g in zip(ep_rows, rets):
            rows.append(dict(state=s_vec, plans=plans, r=np.float32(r),
                             ret=np.float32(g), kind=kind, ep=f"{seed0}_{ep}"))
        if out_q is None:
            print(f"[collect seed={seed0}] ep {ep + 1}/{n_ep} segs={len(ep_rows)} "
                  f"ret0={rets[0]:+.2f}", flush=True)
    if out_q is not None:
        out_q.put(rows)
        return None
    return rows


# ============================================================ 网络
def _mlp(in_dim, hidden, layers, dropout):
    import torch.nn as nn
    dims = [in_dim] + [hidden] * layers
    mods = []
    for a, b in zip(dims[:-1], dims[1:]):
        mods += [nn.Linear(a, b), nn.LayerNorm(b), nn.Tanh()]
        if dropout > 0:
            mods.append(nn.Dropout(dropout))
    mods.append(nn.Linear(dims[-1], 1))
    net = nn.Sequential(*mods)
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.zeros_(m.bias)
    return net


def build_qnet(hidden=256, layers=2, a_hidden=64, dropout=0.1):
    """Q(s, tau) = V(s) + A(s, tau)：值分支只吃态势，优势分支才吃指令。

    单塔训练时回报方差几乎全来自态势（火场布局、时间余量、剩余电量），
    网络会拿大容量去记态势，指令那 93 维被淹没；实测按回合切分的验证集上
    单塔 R2=0.74 但"规划器计划 vs 退化计划"的排序只有 61%。
    拆成两支后可以只对优势分支做强正则（小宽度 + dropout + 输出收缩），
    而 ReAd 门限用的恰恰只是同一态势下的 Q 之差。
    """
    import torch
    import torch.nn as nn

    class QNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.v = _mlp(STATE_DIM, hidden, layers, dropout)
            self.a = _mlp(STATE_DIM + PLAN_DIM, a_hidden, 2, dropout)

        def forward(self, x, ret_adv=False):
            adv = self.a(x).squeeze(-1)
            q = self.v(x[:, :STATE_DIM]).squeeze(-1) + adv
            return (q, adv) if ret_adv else q

    return QNet().to(torch.device("cpu"))


class ReAdCritic:
    """Q_theta(s, tau) + ReAd-S 增量打分。"""

    ARCH = ("hidden", "layers", "a_hidden", "dropout")

    def __init__(self, path=None, hidden=256, layers=2, a_hidden=64, dropout=0.1,
                 device="cpu"):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        self.arch = dict(hidden=hidden, layers=layers, a_hidden=a_hidden, dropout=dropout)
        self.mu = np.zeros(STATE_DIM + PLAN_DIM, dtype=np.float32)
        self.sd = np.ones(STATE_DIM + PLAN_DIM, dtype=np.float32)
        if path:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            self.arch = {k: ck[k] for k in self.ARCH}
            self.net = build_qnet(**self.arch).to(self.device)
            self.net.load_state_dict(ck["net"])
            self.mu, self.sd = ck["mu"], ck["sd"]
        else:
            self.net = build_qnet(**self.arch).to(self.device)
        self.net.eval()

    def q(self, x):
        """x: [B, STATE_DIM + PLAN_DIM] 原始特征 -> [B] 的 Q。"""
        t = self.torch
        z = (np.asarray(x, dtype=np.float32) - self.mu) / self.sd
        with t.no_grad():
            return self.net(t.as_tensor(z, device=self.device)).cpu().numpy()

    def value(self, s):
        """V(s)：只看态势的基线值，反事实测试里用它补上"未来那一截"。"""
        t = self.torch
        z = (np.asarray(s, dtype=np.float32).reshape(-1, STATE_DIM)
             - self.mu[:STATE_DIM]) / self.sd[:STATE_DIM]
        with t.no_grad():
            return self.net.v(t.as_tensor(z, device=self.device)).squeeze(-1).cpu().numpy()

    def score(self, env, orders):
        """ReAd-S：返回 (每个子任务的增量优势, 整队 Q)。"""
        s = encode_state(env)
        P = np.stack([encode_plan(env, orders, n_prefix=k) for k in range(len(orders) + 1)])
        qs = self.q(np.concatenate([np.repeat(s[None], len(P), 0), P], axis=1))
        return np.diff(qs), float(qs[-1])


# ============================================================ 训练
def _flatten(rows, prefix_aug=True):
    """一条 segment 展开成 (n+1) 条前缀样本，同时给出 TD 目标所需的 (r, s')。

    前缀展开正是 ReAd-S 里 Q^{1:i} 的定义：对剩余子任务取期望。因为采集时
    其余槽位的指令是随机的，这个条件期望天然随 i 变化，增量才有信息量。

    返回 X(前缀特征) / Y(MC 回报) / R(本段奖励) / SN(下一段状态) / DONE / K(计划类别) / SRC(所属 segment)
    """
    S, R, SN, DONE, YS = [], [], [], [], []
    n_rows = len(rows)
    for j, r in enumerate(rows):
        S.append(r["state"])
        R.append(r["r"])
        YS.append(r["ret"])
        nxt = rows[j + 1] if j + 1 < n_rows else None
        same_ep = nxt is not None and nxt.get("ep") == r.get("ep")
        SN.append(nxt["state"] if same_ep else np.zeros_like(r["state"]))
        DONE.append(0.0 if same_ep else 1.0)
    X, Y, K, SRC = [], [], [], []
    for j, r in enumerate(rows):
        P = r["plans"]
        idx = range(len(P)) if prefix_aug else [len(P) - 1]
        for k in idx:
            X.append(np.concatenate([r["state"], P[k]]))
            Y.append(r["ret"])
            K.append(r["kind"])
            SRC.append(j)
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    return dict(X=f32(X), Y=f32(Y), K=np.asarray(K), SRC=np.asarray(SRC),
                S=f32(S), R=f32(R), SN=f32(SN), DONE=f32(DONE), YS=f32(YS))


def train(args):
    """两阶段拟合。

    直接把 Q(s,tau) 回归到 MC 回报是行不通的：一段回报的方差（std 4.3）几乎全部
    来自"这局火场长什么样、还剩多少时间"，而单次指令的效应只有 0.2 量级，
    实测选点只能停在第 1 个 epoch，此时 Q 对四类计划给的值几乎一样，门限失效。

    所以拆成：
      A. V(s) 用 MC 回报拟合（只吃态势）；
      B. 冻结 V，用 TD 目标 y = r + gamma * V(s') 训练优势分支，
         等价于让 A(s,tau) 去拟合这一段的优势 r + gamma V(s') - V(s)。
         未来回报被替换成它的期望，噪声大幅下降，指令效应才浮出来——
         这也正是 ReAd 里"advantage feedback"的语义。
    """
    import torch
    with open(args.data, "rb") as f:
        rows = pickle.load(f)
    rng = np.random.RandomState(0)
    # 按**回合**切分：同一回合的相邻 segment 回报强相关，按 segment 随机切会泄漏
    eps = sorted({r.get("ep", i) for i, r in enumerate(rows)})
    rng.shuffle(eps)
    val_eps = set(eps[:max(1, int(0.15 * len(eps)))])
    val_rows = [r for i, r in enumerate(rows) if r.get("ep", i) in val_eps]
    tr_rows = [r for i, r in enumerate(rows) if r.get("ep", i) not in val_eps]
    tr, va = _flatten(tr_rows), _flatten(val_rows)
    mu, sd = tr["X"].mean(0), tr["X"].std(0) + 1e-3
    print(f"train {tr['X'].shape} val {va['X'].shape} | "
          f"MC mean {tr['Y'].mean():+.3f} std {tr['Y'].std():.3f} | "
          f"seg r mean {tr['R'].mean():+.3f} std {tr['R'].std():.3f}")

    critic = ReAdCritic(hidden=args.hidden, layers=args.layers, a_hidden=args.a_hidden,
                        dropout=args.dropout, device=args.device)
    critic.mu, critic.sd = mu.astype(np.float32), sd.astype(np.float32)
    net, dev = critic.net, critic.device
    smu, ssd = mu[:STATE_DIM], sd[:STATE_DIM]
    T = lambda a: torch.as_tensor(a, device=dev)
    d = {k: T((tr[k] - (mu if k == "X" else smu)) / (sd if k == "X" else ssd))
         if k in ("X", "S", "SN") else T(tr[k])
         for k in ("X", "S", "SN", "Y", "R", "DONE", "YS")}
    dv = {k: T((va[k] - (mu if k == "X" else smu)) / (sd if k == "X" else ssd))
          if k in ("X", "S", "SN") else T(va[k])
          for k in ("X", "S", "SN", "Y", "R", "DONE", "YS")}

    # ---------------- 阶段 A：V(s) <- MC 回报
    optv = torch.optim.Adam(net.v.parameters(), lr=args.lr, weight_decay=args.wd)
    nS = d["S"].shape[0]
    best_v, best_vsd = float("inf"), None
    for ep in range(args.v_epochs):
        net.train()
        idx = torch.randperm(nS, device=dev)
        for b in range(0, nS, args.batch):
            j = idx[b:b + args.batch]
            loss = torch.nn.functional.huber_loss(net.v(d["S"][j]).squeeze(-1), d["YS"][j])
            optv.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.v.parameters(), 1.0)
            optv.step()
        net.eval()
        with torch.no_grad():
            rmse = float(torch.sqrt(torch.mean(
                (net.v(dv["S"]).squeeze(-1) - dv["YS"]) ** 2)))
        if rmse < best_v:
            best_v = rmse
            best_vsd = {k: v.detach().clone() for k, v in net.v.state_dict().items()}
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"[V] ep {ep + 1:3d} val_rmse {rmse:.4f} best {best_v:.4f}", flush=True)
    net.v.load_state_dict(best_vsd)
    for p in net.v.parameters():
        p.requires_grad_(False)

    # ---------------- 阶段 B：A(s,tau) <- r + gamma V(s') - V(s)
    net.eval()
    with torch.no_grad():
        adv_tr = (d["R"] + GAMMA_H * (1.0 - d["DONE"]) * net.v(d["SN"]).squeeze(-1)
                  - net.v(d["S"]).squeeze(-1))
        adv_va = (dv["R"] + GAMMA_H * (1.0 - dv["DONE"]) * net.v(dv["SN"]).squeeze(-1)
                  - net.v(dv["S"]).squeeze(-1))
    ytr = adv_tr[T(tr["SRC"]).long()]          # 前缀样本共享所属 segment 的优势目标
    yva = adv_va[T(va["SRC"]).long()]
    print(f"[adv] target mean {float(ytr.mean()):+.4f} std {float(ytr.std()):.4f}")
    opta = torch.optim.Adam(net.a.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opta, T_max=args.epochs)
    Kva, Yva_np = va["K"], va["Y"]
    yva_np = yva.cpu().numpy()
    best, best_sd = float("inf"), None
    best_rmse = best_disc = float("nan")
    n = d["X"].shape[0]
    for ep in range(args.epochs):
        net.train()
        idx = torch.randperm(n, device=dev)
        tot = 0.0
        for b in range(0, n, args.batch):
            j = idx[b:b + args.batch]
            a = net.a(d["X"][j]).squeeze(-1)
            loss = (torch.nn.functional.huber_loss(a, ytr[j], delta=1.0)
                    + args.a_l2 * torch.mean(a ** 2))
            opta.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.a.parameters(), 1.0)
            opta.step()
            tot += float(loss.detach()) * len(j)
        sched.step()
        net.eval()
        with torch.no_grad():
            pa = net.a(dv["X"]).squeeze(-1).cpu().numpy()
        rmse = float(np.sqrt(np.mean((pa - yva_np) ** 2)))
        # 计划效应的标定误差：类别是与态势独立采样的，所以各类别的目标均值就是
        # 该类计划效应的无偏估计，A 的类别均值应当复现它（去掉整体偏置后比较）
        eff_p = {k: pa[Kva == k].mean() - pa.mean() for k in KINDS if (Kva == k).any()}
        eff_y = {k: yva_np[Kva == k].mean() - yva_np.mean() for k in eff_p}
        disc_err = float(np.mean([abs(eff_p[k] - eff_y[k]) for k in eff_p]))
        crit = rmse + args.w_disc * disc_err
        if crit < best:
            best, best_rmse, best_disc = crit, rmse, disc_err
            best_sd = {k: v.detach().clone() for k, v in net.a.state_dict().items()}
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"[A] ep {ep + 1:3d} train {tot / n:.4f} val_rmse {rmse:.4f} "
                  f"disc_err {disc_err:.4f} crit {crit:.4f} best {best:.4f}", flush=True)
    net.a.load_state_dict(best_sd)
    torch.save(dict(net=net.state_dict(), mu=critic.mu, sd=critic.sd, **critic.arch), args.out)
    print(f"saved -> {args.out} (V rmse {best_v:.4f} | adv rmse {best_rmse:.4f} "
          f"disc_err {best_disc:.4f})")

    # 各类计划的优势：规划器/扰动应当高于随机/退化，且与目标同序
    with torch.no_grad():
        pa = net.a(dv["X"]).squeeze(-1).cpu().numpy()
        pq = net(dv["X"]).cpu().numpy()
    ss = float(np.var(Yva_np))
    print(f"[val] Q vs MC: rmse {np.sqrt(np.mean((pq - Yva_np) ** 2)):.3f} "
          f"R2 {1.0 - np.mean((pq - Yva_np) ** 2) / max(1e-9, ss):.3f}")
    for k in KINDS:
        m = Kva == k
        if m.any():
            print(f"  kind={k:<11s} n={int(m.sum()):6d} A {pa[m].mean():+.4f}  "
                  f"目标优势 {yva_np[m].mean():+.4f}  MC {Yva_np[m].mean():+.3f}")


# ============================================================ 评估
# ============================================================ 反事实配对测试
def _rollout_branch(env0, orders, actor, horizon, rng_state, critic, segr):
    """从同一个状态分叉：深拷贝环境 + 复位全局随机流（公共随机数），只换指令。

    返回这一段的**真实**优势 r + gamma * V(s') - V(s)，即 critic 该学的那个目标。
    """
    import copy
    import torch
    env = copy.deepcopy(env0)
    np.random.set_state(rng_state)          # 火扩散/风向走同一条随机路径
    obs = env.set_plan(orders)
    s0 = segr.snapshot(env)
    v0 = float(critic.value(encode_state(env))[0])
    cov, done = [], False
    for _ in range(horizon):
        with torch.no_grad():
            a, _, _ = actor.act(torch.as_tensor(obs), deterministic=True)
        obs, _r, done, info = env.step(a.numpy().tolist())
        cov.append(float(info.get("coverage_rate", 0.0)))
        if done:
            break
    r = segr.value(s0, segr.snapshot(env), float(np.mean(cov)) if cov else 0.0)
    v1 = 0.0 if done else float(critic.value(encode_state(env))[0])
    return r + GAMMA_H * v1 - v0


def probe_worker(args, seed0, n_ep, out_q=None):
    """驱动一条规划器轨迹，在每个 segment 起点做一次 A/B 反事实对比。"""
    import random
    import torch
    torch.set_num_threads(1)
    rng = random.Random(seed0)
    actor = _load_actor(args.actor)
    critic = ReAdCritic(path=args.ckpt)
    env = _make_env(args, seed0)
    segr = SegReward()
    pairs = []
    for ep in range(n_ep):
        obs = env.reset()
        while True:
            ka, kb = rng.sample(list(KINDS), 2)
            pa, _ = sample_plan(env, rng, kind=ka)
            pb, _ = sample_plan(env, rng, kind=kb)
            _, qa = critic.score(env, pa)
            _, qb = critic.score(env, pb)
            st = np.random.get_state()
            ta = _rollout_branch(env, pa, actor, args.horizon, st, critic, segr)
            tb = _rollout_branch(env, pb, actor, args.horizon, st, critic, segr)
            pairs.append((qa - qb, ta - tb, ka, kb))
            # 主干继续按规划器走，保证访问到的态势是 on-distribution 的
            np.random.set_state(st)
            plan, _ = sample_plan(env, rng, kind="planner")
            obs = env.set_plan(plan)
            done = False
            for _ in range(args.horizon):
                with torch.no_grad():
                    a, _, _ = actor.act(torch.as_tensor(obs), deterministic=True)
                obs, _r, done, _i = env.step(a.numpy().tolist())
                if done:
                    break
            if done:
                break
    if out_q is not None:
        out_q.put(pairs)
        return None
    return pairs


def report_pairs(pairs):
    dq = np.array([p[0] for p in pairs], dtype=np.float64)
    dt = np.array([p[1] for p in pairs], dtype=np.float64)
    n = len(dq)
    agree = (np.sign(dq) == np.sign(dt))
    print(f"[反事实配对] {n} 对（同一状态、公共随机数、只换指令）")
    print(f"  真实优势差 |dt| 均值 {np.abs(dt).mean():.4f}  critic 预测差 |dq| 均值 "
          f"{np.abs(dq).mean():.4f}")
    print(f"  符号一致率 {agree.mean():.1%}  "
          f"（随机猜 50%，二项 z = {(agree.mean() - 0.5) / (0.5 / math.sqrt(n)):.1f}）")
    thr = np.percentile(np.abs(dt), 50)
    m = np.abs(dt) >= thr
    print(f"  只看差距明显的一半（|dt| >= {thr:.3f}）：一致率 {agree[m].mean():.1%}")
    print(f"  corr(dq, dt) = {np.corrcoef(dq, dt)[0, 1]:.3f}")
    # 标定：按预测差分箱，看真实差是否单调跟随
    qs = np.percentile(dq, [0, 20, 40, 60, 80, 100])
    print("  标定（按预测差分箱 -> 真实差均值）：")
    for i in range(5):
        b = (dq >= qs[i]) & (dq <= qs[i + 1])
        if b.any():
            print(f"    dq∈[{qs[i]:+.3f},{qs[i + 1]:+.3f}] n={int(b.sum()):4d}  "
                  f"真实 dt 均值 {dt[b].mean():+.4f}")
    # 分类别：哪一类计划的判别最难
    print("  按类别对（仅列出现 >= 20 次的）：")
    seen = {}
    for q, t, ka, kb in pairs:
        key = tuple(sorted((ka, kb)))
        seen.setdefault(key, []).append(np.sign(q) == np.sign(t))
    for key, v in sorted(seen.items(), key=lambda kv: -len(kv[1])):
        if len(v) >= 20:
            print(f"    {key[0]:<11s} vs {key[1]:<11s} n={len(v):4d} 一致率 {np.mean(v):.1%}")


def probe_semantics(args, critic):
    """语义单元测试：在真实态势上，手工构造几组"明显该被否掉"的指令，看 Q 是否更低。"""
    import random
    import torch
    torch.set_num_threads(1)
    rng = random.Random(args.seed + 99)
    actor = _load_actor(args.actor)
    env = _make_env(args, args.seed + 99)
    from lowlevel.obs_builder import shared_cache

    def all_task(task, **kw):
        return [UavOrder(task=task, **kw) for _ in range(env.num_uavs)]

    cases = {
        "全队补给（电量还够）": lambda e: all_task(TASK_RECHARGE),
        "全队中继（没人搜寻）": lambda e: all_task(TASK_COMM_RELAY),
        "全队搜寻（没人中继）": lambda e: all_task(TASK_FIRE_SENSING, mode=SENSE_EXPLORE),
        "搜寻目标指向已探索区": lambda e: [
            UavOrder(task=TASK_FIRE_SENSING, mode=SENSE_EXPLORE,
                     target=(float(u.x), float(u.y))) for u in e.env.uavs],
        "巡逻环半径给 1（过小）": lambda e: [
            UavOrder(task=TASK_FIRE_SENSING, mode=SENSE_PATROL, ring_radius=1.0,
                     target=(shared_cache(e.env)["clusters"] or [{"centroid": (50, 50)}])[0]["centroid"])
            for _ in range(e.num_uavs)],
    }
    stat = {k: [0, 0, []] for k in cases}
    for ep in range(args.episodes):
        obs = env.reset()
        while True:
            good, _ = sample_plan(env, rng, kind="planner")
            _, q_good = critic.score(env, good)
            for name, fn in cases.items():
                try:
                    _, q_bad = critic.score(env, fn(env))
                except Exception:
                    continue
                stat[name][1] += 1
                stat[name][0] += int(q_good > q_bad)
                stat[name][2].append(q_good - q_bad)
            obs = env.set_plan(good)
            done = False
            for _ in range(args.horizon):
                with torch.no_grad():
                    a, _, _ = actor.act(torch.as_tensor(obs), deterministic=True)
                obs, _r, done, _i = env.step(a.numpy().tolist())
                if done:
                    break
            if done:
                break
    print("[语义单元测试] Q(规划器) 是否高于这些『该被否掉』的指令：")
    for name, (w, t, g) in stat.items():
        if t:
            print(f"  {name:<22s} {w}/{t} = {w / t:.1%}   mean gap {np.mean(g):+.4f}")
def eval_ranking(args, critic):
    """同一态势下配对比较：规划器计划 vs 退化/随机计划，看 Q 排序对不对。"""
    import random
    import torch
    torch.set_num_threads(1)
    rng = random.Random(args.seed + 777)
    actor = _load_actor(args.actor)
    env = _make_env(args, args.seed + 777)
    win = {"degenerate": [0, 0], "random": [0, 0]}
    gaps = {"degenerate": [], "random": []}
    incs = {k: [] for k in ("planner", "degenerate", "random")}
    for ep in range(args.episodes):
        obs = env.reset()
        while True:
            good, _ = sample_plan(env, rng, kind="planner")
            inc_g, q_good = critic.score(env, good)
            incs["planner"] += list(inc_g)
            for kind in ("degenerate", "random"):
                bad, _ = sample_plan(env, rng, kind=kind)
                inc_b, q_bad = critic.score(env, bad)
                incs[kind] += list(inc_b)
                win[kind][1] += 1
                win[kind][0] += int(q_good > q_bad)
                gaps[kind].append(q_good - q_bad)
            obs = env.set_plan(good)
            done = False
            for _ in range(args.horizon):
                with torch.no_grad():
                    a, _, _ = actor.act(torch.as_tensor(obs), deterministic=True)
                obs, _r, done, _i = env.step(a.numpy().tolist())
                if done:
                    break
            if done:
                break
    print("[ranking] Q(规划器) 高于对照的比例：")
    for k, (w, t) in win.items():
        print(f"  vs {k:<11s} {w}/{t} = {w / max(1, t):.1%}   mean gap {np.mean(gaps[k]):+.3f}")
    print("[增量优势 S_ReAd 分布] （门限 alpha 就从这里取）")
    for k, v in incs.items():
        v = np.asarray(v)
        print(f"  {k:<11s} mean {v.mean():+.4f}  p10 {np.percentile(v, 10):+.4f}  "
              f"p20 {np.percentile(v, 20):+.4f}  p50 {np.percentile(v, 50):+.4f}")
    return {"win": {k: w / max(1, t) for k, (w, t) in win.items()},
            "alpha_p20": float(np.percentile(np.asarray(incs["planner"]), 20))}


def eval_gating(args, critic, gated):
    """端到端：把"随机 LLM"的计划过一遍 ReAd 门限，比较任务指标。"""
    import random
    import torch
    torch.set_num_threads(1)
    rng = random.Random(args.seed + 31)
    actor = _load_actor(args.actor)
    env = _make_env(args, args.seed + 31)
    out, n_resample = [], 0
    for ep in range(args.episodes):
        obs = env.reset()
        while True:
            plan, _ = sample_plan(env, rng, kind="random")
            if gated:
                # S_ReAd(tau_i) 全部过阈值才下发，否则重采样（上限 retries 次）
                for _try in range(args.retries):
                    inc, _q = critic.score(env, plan)
                    if float(inc.min()) > args.alpha:
                        break
                    n_resample += 1
                    plan, _ = sample_plan(env, rng, kind="random")
            obs = env.set_plan(plan)
            done = False
            for _ in range(args.horizon):
                with torch.no_grad():
                    a, _, _ = actor.act(torch.as_tensor(obs), deterministic=True)
                obs, _r, done, info = env.step(a.numpy().tolist())
                if done:
                    break
            if done:
                break
        e = env.env
        out.append([float(getattr(e, "explored_rate", 0.0)),
                    float(getattr(e, "active_discovery_rate", 0.0)),
                    float(getattr(e, "active_rescuer_known_rate", 0.0)),
                    float((e.grid["burning_state"] == 3).sum()),
                    float(getattr(e, "n_contained_clusters", 0)),
                    float(info.get("n_depleted", 0))])
    m = np.mean(np.asarray(out), axis=0)
    tag = f"gated(alpha={args.alpha:+.3f})" if gated else "ungated"
    print(f"[{tag:<22s}] explored {m[0]:.3f} disc {m[1]:.3f} rknown {m[2]:.3f} "
          f"ext {m[3]:.0f} contained {m[4]:.2f} depleted {m[5]:.2f}"
          + (f"  resamples {n_resample}" if gated else ""))
    return m


# ============================================================ CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "train", "eval", "probe"])
    ap.add_argument("--actor", default="lowlevel/runs/w4_h384/actor_best.pt")
    ap.add_argument("--data", default="lowlevel/runs/read/dataset.pkl")
    ap.add_argument("--out", default="lowlevel/runs/read/read_critic.pt")
    ap.add_argument("--ckpt", default="lowlevel/runs/read/read_critic.pt")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=25)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--num-uavs", type=int, default=4)
    ap.add_argument("--num-rescuers", type=int, default=8)
    ap.add_argument("--fire-seeds", type=int, default=3)
    ap.add_argument("--spread-gain", type=float, default=1.0)
    ap.add_argument("--grid", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--a-hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--a-l2", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--w-disc", type=float, default=3.0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--v-epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--retries", type=int, default=8)
    ap.add_argument("--sem-episodes", type=int, default=20)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    if args.cmd == "collect":
        import multiprocessing as mp
        per = max(1, args.episodes // args.workers)
        ctx = mp.get_context("fork")
        q = ctx.Queue()
        procs = [ctx.Process(target=collect_episodes,
                             args=(args, args.seed + 1000 * w, per, q))
                 for w in range(args.workers)]
        for p in procs:
            p.start()
        rows = []
        for _ in procs:
            rows += q.get()
        for p in procs:
            p.join()
        with open(args.data, "wb") as f:
            pickle.dump(rows, f)
        rets = np.array([r["ret"] for r in rows])
        print(f"collected {len(rows)} segments -> {args.data} | "
              f"ret mean {rets.mean():+.3f} std {rets.std():.3f} "
              f"min {rets.min():+.3f} max {rets.max():+.3f}")
    elif args.cmd == "train":
        train(args)
    elif args.cmd == "probe":
        import multiprocessing as mp
        import copy as _copy
        critic = ReAdCritic(path=args.ckpt, device=args.device)
        sem_args = _copy.copy(args)
        sem_args.episodes = min(args.episodes, args.sem_episodes)
        probe_semantics(sem_args, critic)
        per = max(1, args.episodes // args.workers)
        ctx = mp.get_context("fork")
        q = ctx.Queue()
        procs = [ctx.Process(target=probe_worker,
                             args=(args, args.seed + 100 * w + 7, per, q))
                 for w in range(args.workers)]
        for p in procs:
            p.start()
        pairs = []
        for _ in procs:
            pairs += q.get()
        for p in procs:
            p.join()
        report_pairs(pairs)
    else:
        critic = ReAdCritic(path=args.ckpt, device=args.device)
        rk = eval_ranking(args, critic)
        if args.alpha is None:
            # alpha 取规划器计划增量优势的 20 分位：让"和规划器一样好"的指令基本都放过
            args.alpha = rk["alpha_p20"]
            print(f"[gating] alpha 自动标定 = {args.alpha:+.4f}")
        eval_gating(args, critic, gated=False)
        eval_gating(args, critic, gated=True)


if __name__ == "__main__":
    main()

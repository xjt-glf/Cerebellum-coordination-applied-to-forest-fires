# -*- coding: utf-8 -*-
"""奖励分解对照：同一起点与同一 explore 指令，比较 Actor 实际轨迹与"贪心服从
目标"轨迹的逐步奖励构成——验证"不服从指定目标在训练奖励下是否最优"。

参考控制器（贪心）不是 Actor 的效果，只用于受控比较（任务书 §九）。
"""
import json
import math
import multiprocessing as mp
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "critic_advice"))

from harness import BENCH, Snapshot, make_env, load_actor, make_h_brain, task_metrics
from lowlevel.task_spec import UavOrder, TASK_FIRE_SENSING, SENSE_EXPLORE
from lowlevel.mission_demo import brain as h_brain

W = 60
_DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (0, 0)]


def greedy_action(u, tgt):
    dx, dy = tgt[0] - float(u.x), tgt[1] - float(u.y)
    best, bd = 8, 1e9
    for k, (ax, ay) in enumerate(_DIRS[:8]):
        d = math.hypot(dx - ax, dy - ay)
        if d < bd:
            best, bd = k, d
    return best


def run_one(seed, regime):
    import torch
    torch.set_num_threads(1)
    cfg = dict(BENCH)
    if regime == "scarce":
        cfg.update(uavs=3, rescuers=14)
    env = make_env(seed, cfg)
    env.reset()
    actor = load_actor(BENCH["actor"])
    brain = make_h_brain(cfg)
    obs, done = None, False
    while env.t < 200 and not done:
        if env.t % 20 == 0:
            from lowlevel.mission_demo import sense_state
            s = sense_state(env)
            plan, _, _ = brain.plan(s)
            obs = env.set_plan(plan)
        with torch.no_grad():
            a = actor.act(torch.as_tensor(obs, dtype=torch.float32),
                          deterministic=True)[0].numpy()
        obs, _r, done, _i = env.step(a.tolist())
    if done:
        return None
    snap = Snapshot(env)
    from lowlevel.mission_demo import sense_state
    s = sense_state(env)
    plan, _, _ = brain.plan(s)

    out = []
    for i in range(cfg["uavs"]):
        u = env.env.uavs[i]
        tgt = max([(25.0, 25.0), (25.0, 75.0), (75.0, 75.0), (75.0, 25.0)],
                  key=lambda q: math.hypot(q[0] - u.x, q[1] - u.y))
        orders = list(plan and env.orders)
        orders = [UavOrder(task=o.task, mode=o.mode, target=o.target,
                           ring_radius=o.ring_radius, rescuer_id=o.rescuer_id,
                           proactive_handover=o.proactive_handover) for o in orders]
        orders[i] = UavOrder(task=TASK_FIRE_SENSING, mode=SENSE_EXPLORE, target=tgt)

        # ---- 轨迹 1：Actor 实际 ----
        e1 = snap.restore()
        o1 = e1.set_plan(list(orders))
        parts_actor, d1 = [], math.hypot(u.x - tgt[0], u.y - tgt[1])
        for k in range(W):
            with torch.no_grad():
                a = actor.act(torch.as_tensor(o1, dtype=torch.float32),
                              deterministic=True)[0].numpy().tolist()
            o1, _r, _dn, info = e1.step(a)
            parts_actor.append(info["parts"][i])
        d_end_actor = math.hypot(e1.env.uavs[i].x - tgt[0], e1.env.uavs[i].y - tgt[1])

        # ---- 轨迹 2：贪心服从（受测机贪心直行，其余机仍用 Actor） ----
        e2 = snap.restore()
        o2 = e2.set_plan(list(orders))
        parts_greedy, obs2 = [], None
        for k in range(W):
            with torch.no_grad():
                a = actor.act(torch.as_tensor(o2, dtype=torch.float32),
                              deterministic=True)[0].numpy().tolist()
            a[i] = greedy_action(e2.env.uavs[i], tgt)
            o2, _r, _dn, info = e2.step(a)
            parts_greedy.append(info["parts"][i])
        d_end_greedy = math.hypot(e2.env.uavs[i].x - tgt[0], e2.env.uavs[i].y - tgt[1])

        def agg(parts):
            keys = ("shape", "sense", "refresh", "time", "team")
            return {k: float(sum(p.get(k, 0.0) for p in parts)) for k in keys}

        out.append(dict(regime=regime, seed=seed, uav=i, target=list(tgt),
                        d_init=round(d1, 1), d_end_actor=round(d_end_actor, 1),
                        d_end_greedy=round(d_end_greedy, 1),
                        actor_parts=agg(parts_actor), greedy_parts=agg(parts_greedy),
                        actor_total=float(sum(sum(p.values()) for p in parts_actor)),
                        greedy_total=float(sum(sum(p.values()) for p in parts_greedy))))
    return out


def _w(a):
    try:
        r = run_one(*a)
        return r if r else []
    except Exception as e:
        import traceback
        return [dict(error=str(e), trace=traceback.format_exc()[-600:])]


if __name__ == "__main__":
    jobs = [(60000 + k, rg) for rg in ("normal", "scarce") for k in range(6)]
    with mp.get_context("fork").Pool(12) as pool:
        res = pool.map(_w, jobs)
    rows = [r for chunk in res for r in chunk if isinstance(r, dict) and "error" not in r]
    with open(os.path.join(_HERE, "runs", "reward_decomp.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print(f"n={len(rows)}（60 步窗口，受测机为 explore 指定远目标）")
    for rg in ("normal", "scarce"):
        sub = [r for r in rows if r["regime"] == rg]
        if not sub:
            continue
        print(f"\n=== {rg} ===")
        print(f"  距目标：init {np.mean([r['d_init'] for r in sub]):5.1f} → "
              f"Actor末 {np.mean([r['d_end_actor'] for r in sub]):5.1f} | "
              f"贪心末 {np.mean([r['d_end_greedy'] for r in sub]):5.1f}")
        for k in ("shape", "sense", "time", "team"):
            av = np.mean([r["actor_parts"][k] for r in sub])
            gv = np.mean([r["greedy_parts"][k] for r in sub])
            print(f"  {k:8s}: Actor {av:+8.2f} | 贪心服从 {gv:+8.2f}")
        print(f"  总奖励  : Actor {np.mean([r['actor_total'] for r in sub]):+8.2f} | "
              f"贪心服从 {np.mean([r['greedy_total'] for r in sub]):+8.2f} "
              f"（差 {np.mean([r['actor_total'] - r['greedy_total'] for r in sub]):+.2f}）")

# -*- coding: utf-8 -*-
"""字段敏感性探针（验收协议 §4）：同一快照只改一个字段。

- explore 目标 A vs B：首步概率 L1 / argmax 翻转 / 60 步轨迹分叉 / 各自是否完成自己的目标；
- patrol ring=auto vs 显式 6/10：对应环距是否随之变化；
- relay 混合字段：rescuer_id 优先（契约 v2）。
纯 Actor、argmax、无 waypoint_assist。
"""
import argparse, copy, json, math, multiprocessing as mp, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "experiments", "critic_advice"))
sys.path.insert(0, _HERE)

from harness import Snapshot, load_actor
from lowlevel.task_spec import (UavOrder, TASK_FIRE_SENSING, TASK_COMM_RELAY,
                                SENSE_EXPLORE, SENSE_PATROL)
from lowlevel.obs_builder import shared_cache, resolve_order_target
from evaluate import build_snapshots, make_eval_env, run_case, gen_cases


def probe_job(args_tuple):
    seed, actor_path, regime = args_tuple
    import torch
    torch.set_num_threads(1)
    actor = load_actor(actor_path)
    out = []
    snaps = build_snapshots(seed, regime, actor)
    for t0, (snap, base_orders) in snaps.items():
        env = snap.restore()
        e = env.env
        rng = np.random.RandomState(seed)
        i = 0
        # ---- explore A vs B（两个合法远目标）----
        from evaluate import valid_explore_targets
        tg = valid_explore_targets(e, i, rng)
        far_targets = [v for k, v in tg.items() if k >= 1]
        if len(far_targets) >= 2:
            A, B = far_targets[0][0], far_targets[1][0]
            for tag, tgt in (("A", A), ("B", B)):
                cs = dict(task="explore", agent=i, stratum=1, target=list(tgt),
                          regime=regime, seed=seed, t0=t0, n_uavs=len(e.uavs),
                          base=[o.to_dict() for o in base_orders])
                rec = run_case(snap, base_orders, cs, actor, record_traj=True)
                out.append(dict(probe="explore_AB", which=tag, seed=seed, t0=t0,
                                target=list(tgt),
                                arrived=rec["first_arrival"] is not None,
                                own_region_frac=rec.get("own_region_frac"),
                                traj=rec.get("traj")))
        # ---- patrol auto vs 显式 ----
        c = shared_cache(e)
        if c["clusters"]:
            cl = c["clusters"][0]
            cent = (float(cl["centroid"][0]), float(cl["centroid"][1]))
            for ring in ("auto", 6.0, 10.0):
                cs = dict(task="patrol", agent=i, cluster=0, target=list(cent),
                          ring=ring, regime=regime, seed=seed, t0=t0,
                          n_uavs=len(e.uavs), base=[o.to_dict() for o in base_orders])
                rec = run_case(snap, base_orders, cs, actor, record_traj=False)
                out.append(dict(probe="patrol_ring", ring=str(ring), seed=seed, t0=t0,
                                target=list(cent),
                                ring_err_mean=rec.get("ring_err_mean"),
                                within2=rec.get("ring_within2"),
                                arrived=rec["first_arrival"] is not None))
        # ---- relay 混合字段：rescuer_id 优先 ----
        rs = list(e.rescuers)
        if len(rs) >= 2:
            r0, r1 = rs[0], rs[-1]
            cs = dict(task="relay", agent=i, rescuer=int(r0.rescuer_id),
                      interface="rescuer_id", tag="mixed",
                      extra_target=[float(r1.x), float(r1.y)],
                      regime=regime, seed=seed, t0=t0, n_uavs=len(e.uavs),
                      base=[o.to_dict() for o in base_orders])
            orders = copy.deepcopy(list(base_orders))
            o = UavOrder(task=TASK_COMM_RELAY, rescuer_id=int(r0.rescuer_id),
                         target=(float(r1.x), float(r1.y)))
            orders[i] = o
            env2 = snap.restore()
            env2.replan_every = 0
            obs = env2.set_plan(list(orders))
            e2 = env2.env
            tgt, _f, _rg = resolve_order_target(e2, i, o, shared_cache(e2))
            d_correct = math.hypot(tgt[0] - e2.uavs[i].x, tgt[1] - e2.uavs[i].y)
            followed_right_person = None
            for k in range(80):
                with torch.no_grad():
                    a = actor.act(torch.as_tensor(obs, dtype=torch.float32),
                                  deterministic=True)[0].numpy()
                obs, _r, done, _i = env2.step(a)
                dd = math.hypot(rs[0].x - e2.uavs[i].x, rs[0].y - e2.uavs[i].y)
                other = math.hypot(rs[-1].x - e2.uavs[i].x, rs[-1].y - e2.uavs[i].y)
                if dd < other and k > 10:
                    followed_right_person = True
                    break
                if other < dd - 6 and k > 30:
                    followed_right_person = False
                    break
                if done:
                    break
            out.append(dict(probe="relay_mixed", seed=seed, t0=t0,
                            resolved_is_target_person=(math.hypot(
                                tgt[0] - rs[0].x, tgt[1] - rs[0].y) < 1e-6),
                            followed_right_person=followed_right_person,
                            d_resolved=round(d_correct, 1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=515000)
    args = ap.parse_args()
    jobs = [(args.seed0 + k, args.actor, rg)
            for rg in ("normal", "scarce") for k in range(args.seeds)]
    recs = []
    with mp.get_context("fork").Pool(min(args.workers, len(jobs))) as pool:
        for rs in pool.imap_unordered(probe_job, jobs):
            recs.extend(rs)
    out_dir = os.path.join(_HERE, "runs", "eval", args.tag or "probe")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "field_probe.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")
    ab = [r for r in recs if r["probe"] == "explore_AB"]
    bys = {}
    for r in ab:
        bys.setdefault((r["seed"], r["t0"]), {})[r["which"]] = r
    n_pair = 0
    both = diff = 0
    traj_diffs = []
    for k, v in bys.items():
        if "A" in v and "B" in v:
            n_pair += 1
            both += int(v["A"]["arrived"] and v["B"]["arrived"])
            diff += int(v["A"]["arrived"] != v["B"]["arrived"])
            ta, tb = v["A"].get("traj"), v["B"].get("traj")
            if ta and tb:
                n = min(len(ta), len(tb))
                traj_diffs.append(sum(1 for x, y in zip(ta[:n], tb[:n]) if x != y))
    pr = [r for r in recs if r["probe"] == "patrol_ring"]
    mx = [r for r in recs if r["probe"] == "relay_mixed"]
    summary = dict(
        explore_AB_pairs=n_pair, both_arrived=both, one_arrived=diff,
        mean_traj_diff=(float(np.mean(traj_diffs)) if traj_diffs else None),
        patrol_ring={str(r["ring"]): dict(err=round(r["ring_err_mean"], 2) if r["ring_err_mean"] else None,
                                          arrived=r["arrived"]) for r in pr[:9]},
        relay_mixed=dict(n=len(mx),
                         resolved_correct=sum(1 for r in mx if r["resolved_is_target_person"]),
                         followed_right=sum(1 for r in mx if r["followed_right_person"])),
    )
    print(json.dumps(summary, indent=1))
    with open(os.path.join(out_dir, "field_probe_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()

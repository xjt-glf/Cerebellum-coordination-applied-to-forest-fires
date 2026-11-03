# -*- coding: utf-8 -*-
"""修正口径的技能/整局验收（lowlevel_training_acceptance.md）。

主指标全部在 纯 Actor、deterministic argmax、无 waypoint_assist/教师接管 下测量。

与旧 diag_skills.py 的差异（计划 §3.1，必须修正的验收口径）：
- relay 服从 = 受测机到指定人员的**本机链路**（距离 <= uav_comm_radius，人员移动逐步测量），
  另报全队覆盖，不再用 r.is_covered 冒充本机服从；
- explore 目标只抽**合法目标**（区域未知比例 >= 0.6、能量可往返），按 8-25/25-50/50-90
  距离分层；已完成区域不会成为服从失败样本；
- d_init / e_init 在动作**前**记录（set_plan 之后、第一步之前）；
- patrol 环半径每步用 resolve_order_target 的实际解析值（auto 随火点演化）；
- 窗口 W = min(160, max(60, 2*chebyshev+20))，不晚于环境终止，中断记删失。

用法：
    python experiments/lowlevel_training/evaluate.py --actor <actor.pt> --tag mytag \
        --split quick --workers 8
    python experiments/lowlevel_training/evaluate.py --actor <actor.pt> --tag mytag \
        --split final --workers 10
    python experiments/lowlevel_training/evaluate.py --actor <actor.pt> --tag mytag \
        --mission --episodes 20 --workers 8
"""
import argparse
import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "experiments", "critic_advice"))

from harness import BENCH, Snapshot, make_env, load_actor, make_h_brain, task_metrics
from lowlevel.task_spec import (
    UavOrder, TASK_FIRE_SENSING, TASK_COMM_RELAY, TASK_RECHARGE,
    SENSE_EXPLORE, SENSE_PATROL,
)
from lowlevel.obs_builder import shared_cache, resolve_order_target, assigned_station
from lowlevel.env_wrapper import TaskConditionedEnv
from experiments.lowlevel_training.goal_manager import (
    GoalManager, chebyshev_steps, EXPLORE_ARRIVAL_RADIUS,
)

REGIMES = {
    "normal": dict(),
    "scarce": dict(uavs=3, rescuers=14),
}
DIST_STRATA = ((8.0, 25.0), (25.0, 50.0), (50.0, 90.0))


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ================================================================ 快照与用例
def make_eval_env(seed, regime, replan_every=0):
    c = dict(BENCH, **REGIMES[regime])
    return TaskConditionedEnv(
        num_uavs=c["uavs"], num_rescuers=c["rescuers"],
        n_fire_seeds=c["fires"], n_fire_seeds_range=None,
        grid_size=(c["grid"], c["grid"]), max_steps=c["steps"],
        spread_steps=c["spread_steps"], spread_gain=c["spread_gain"],
        seed=seed, task_sampler="planner", replan_every=replan_every)


def build_snapshots(seed, regime, actor):
    """旧 Actor + H 协议推进到 t0 ∈ {0, 200}，产出快照 + 该时刻 H 基线计划。"""
    import torch
    env = make_eval_env(seed, regime)
    env.reset()
    brain = make_h_brain({})
    replan = 20
    snaps = {}
    obs = None
    done = False
    for t0 in (0, 200):
        while env.t < t0 and not done:
            if env.t % replan == 0:
                from lowlevel.mission_demo import sense_state
                s = sense_state(env)
                plan, _, _ = brain.plan(s)
                obs = env.set_plan(plan)
            with torch.no_grad():
                a = actor.act(torch.as_tensor(obs, dtype=torch.float32),
                              deterministic=True)[0].numpy()
            obs, _r, done, _i = env.step(a)
        if done:
            break
        snap = Snapshot(env)
        from lowlevel.mission_demo import sense_state
        s = sense_state(env)
        plan, _, _ = brain.plan(s)
        env.set_plan(plan)
        base_orders = copy.deepcopy(list(env.orders))
        snaps[t0] = (snap, base_orders)
        env = snap.restore()
        obs = None
    return snaps


def valid_explore_targets(e, i, rng, n_try=120):
    """按距离分层抽合法 explore 目标：区域(11x11)未知比例 >= 0.6，能量可往返。"""
    u = e.uavs[i]
    ux, uy = float(u.x), float(u.y)
    M = shared_cache(e)["M"]
    H, W = M.shape
    cost = e.energy_cost_per_step(i) if hasattr(e, "energy_cost_per_step") else 2.5
    out = {}
    for si, (lo, hi) in enumerate(DIST_STRATA):
        found = None
        tries = 0
        while tries < n_try and found is None:
            tries += 1
            ang = rng.uniform(0, 2 * math.pi)
            d = rng.uniform(lo, min(hi, 88))
            cx = int(min(max(6, ux + d * math.cos(ang)), H - 7))
            cy = int(min(max(6, uy + d * math.sin(ang)), W - 7))
            dd = math.hypot(cx - ux, cy - uy)
            if dd < 6 or dd > 95:
                continue
            if float(u.e) < cost * 2 * chebyshev_steps(ux, uy, cx, cy):
                continue
            reg = M[max(0, cx - 5):min(H, cx + 6), max(0, cy - 5):min(W, cy + 6)]
            if reg.size and float((reg < 0).mean()) >= 0.6:
                found = ((float(cx), float(cy)), float((reg < 0).mean()))
        if found is not None:
            out[si] = found
    return out


def gen_cases(snap, base_orders, regime, seed, t0, rng):
    """为一个快照生成全部受测用例。"""
    env = snap.restore()
    e = env.env
    n_uav = len(e.uavs)
    cases = []
    for i in range(n_uav):
        u = e.uavs[i]
        targets = valid_explore_targets(e, i, rng)
        for si, (tgt, frac) in targets.items():
            cases.append(dict(task="explore", agent=i, stratum=si,
                              target=list(tgt), unknown_frac=round(frac, 3)))
        if not targets:
            cases.append(dict(task="explore", agent=i, stratum=-1, target=None,
                              skip="no_valid_region"))
        c = shared_cache(e)
        for ci, cl in enumerate(c["clusters"][:3]):
            cent = (float(cl["centroid"][0]), float(cl["centroid"][1]))
            cl_r = math.sqrt(max(1.0, float(cl["size"])) / math.pi) + 2.0
            d = math.hypot(cent[0] - u.x, cent[1] - u.y)
            if d > 85:
                continue
            cases.append(dict(task="patrol", agent=i, cluster=ci,
                              target=list(cent), ring="auto", ring_base=round(cl_r, 2),
                              cluster_size=int(cl["size"])))
            if cl_r + 1.5 <= 6.0:
                cases.append(dict(task="patrol", agent=i, cluster=ci,
                                  target=list(cent), ring=6.0, ring_base=round(cl_r, 2),
                                  cluster_size=int(cl["size"])))
            cases.append(dict(task="patrol", agent=i, cluster=ci,
                              target=list(cent), ring=min(10.0, round(cl_r + 4.0, 1)),
                              ring_base=round(cl_r, 2), cluster_size=int(cl["size"])))
        rs = list(e.rescuers)
        if rs:
            by_d = sorted(rs, key=lambda r: math.hypot(r.x - u.x, r.y - u.y))
            near, far = by_d[0], by_d[-1]
            cases.append(dict(task="relay", agent=i, rescuer=int(near.rescuer_id),
                              interface="rescuer_id", tag="near"))
            if far.rescuer_id != near.rescuer_id:
                cases.append(dict(task="relay", agent=i, rescuer=int(far.rescuer_id),
                                  interface="rescuer_id", tag="far"))
            cases.append(dict(task="relay", agent=i, target=[float(far.x), float(far.y)],
                              interface="point", tag="far_point"))
        st, st_d = assigned_station(e, i)
        cases.append(dict(task="charge", agent=i,
                          station=(list(map(float, st)) if st else None)))
    for cs in cases:
        cs.update(regime=regime, seed=seed, t0=t0,
                  n_uavs=n_uav, base=[o.to_dict() for o in base_orders])
    return cases


def order_for_case(cs):
    if cs["task"] == "explore":
        return UavOrder(task=TASK_FIRE_SENSING, mode=SENSE_EXPLORE,
                        target=tuple(cs["target"]))
    if cs["task"] == "patrol":
        return UavOrder(task=TASK_FIRE_SENSING, mode=SENSE_PATROL,
                        target=tuple(cs["target"]),
                        ring_radius=None if cs["ring"] == "auto" else float(cs["ring"]))
    if cs["task"] == "relay":
        o = UavOrder(task=TASK_COMM_RELAY, proactive_handover=True)
        if cs["interface"] == "rescuer_id":
            o.rescuer_id = int(cs["rescuer"])
        else:
            o.target = tuple(cs["target"])
        return o
    return UavOrder(task=TASK_RECHARGE)


# ================================================================ 单用例 rollout
def run_case(snap, base_orders, cs, actor, record_traj=False):
    """受测机换成用例指令，其余机保持 H 基线指令（整段冻结），argmax 跑 W 步。"""
    import torch
    i = cs["agent"]
    env = snap.restore()
    orders = copy.deepcopy(list(base_orders))
    orders[i] = order_for_case(cs)
    env.replan_every = 0
    obs = env.set_plan(list(orders))
    e = env.env
    u = e.uavs[i]
    tgt0, _f, _r0 = resolve_order_target(e, i, orders[i], shared_cache(e))
    d_init_raw = (math.hypot(tgt0[0] - u.x, tgt0[1] - u.y)
                  if tgt0 is not None else None)
    e0 = float(u.e)
    cheb = (chebyshev_steps(u.x, u.y, *cs["target"]) if cs.get("target") else 0)
    W = int(min(160, max(60, 2 * int(cheb) + 20))) if cs["task"] != "charge" else 120
    W = int(min(W, env.max_steps - env.t))

    gm = GoalManager(grid_size=e.grid_size)
    e._goal_t = env.t
    gm.sync(e, orders, env.t)
    g = gm.goals[i]

    rec = dict(case=cs, steps_run=0, first_arrival=None, arrival_time=None,
               d_init=d_init_raw if d_init_raw is not None else (g.d_init if g else None),
               e_init=e0, window=W, done_mid=False, traj=None)
    own_seen_n, unknown_n = 0, 1
    if cs["task"] == "explore" and g is not None and g.initial_unknown is not None:
        unknown_n = max(1, int(g.initial_unknown.sum()))
    in_band_after, own_link_after, team_cover_after = [], [], []
    ring_errs, ring_vals, d_tos = [], [], []
    traj = []
    on_station_t = None
    recharged = False
    for k in range(W):
        with torch.no_grad():
            a = actor.act(torch.as_tensor(obs, dtype=torch.float32),
                          deterministic=True)[0].numpy()
        obs, _r, done, info = env.step(a)
        e._goal_t = env.t
        ev = gm.update(e, orders, env.t)[i]
        u = e.uavs[i]
        rec["steps_run"] = k + 1
        d_tos.append(ev.get("d_now"))
        if cs["task"] == "explore":
            if g.own_seen is not None:
                own_seen_n = int((g.own_seen & g.initial_unknown).sum())
            if ev.get("arrived") and rec["first_arrival"] is None:
                rec["first_arrival"] = env.t - cs["t0"]
                rec["arrival_time"] = k + 1
        elif cs["task"] == "patrol":
            ring_errs.append(ev.get("ring_err"))
            ring_vals.append(ev.get("ring"))
            if ev.get("arrived") and rec["first_arrival"] is None:
                rec["first_arrival"] = env.t - cs["t0"]
                rec["arrival_time"] = k + 1
            if rec["arrival_time"] is not None and k + 1 > rec["arrival_time"]:
                in_band_after.append(bool(ev.get("in_band")))
        elif cs["task"] == "relay":
            if ev.get("arrived") and rec["first_arrival"] is None:
                rec["first_arrival"] = env.t - cs["t0"]
                rec["arrival_time"] = k + 1
            rid = cs.get("rescuer")
            if rid is not None:
                rr = next((r for r in e.rescuers
                           if int(r.rescuer_id) == int(rid)), None)
                if rr is not None:
                    linked = math.hypot(rr.x - u.x, rr.y - u.y) <= float(
                        getattr(e, "uav_comm_radius", 12))
                    if rec["arrival_time"] is not None and k + 1 > rec["arrival_time"]:
                        own_link_after.append(bool(linked))
                        team_cover_after.append(bool(rr.is_covered))
            else:
                # 固定点：到位后 30 步距点 <= 5 占比
                if rec["arrival_time"] is not None and k + 1 > rec["arrival_time"]:
                    own_link_after.append(bool(ev.get("d_now", 1e9) <= 5.0))
                    team_cover_after.append(True)
        elif cs["task"] == "charge":
            on_st = bool(e.at_station(u.x, u.y)) if hasattr(e, "at_station") else False
            if on_st and on_station_t is None:
                on_station_t = k + 1
            if getattr(u, "recharged", False):
                recharged = True
        if record_traj:
            traj.append([float(u.x), float(u.y)])
        if done:
            rec["done_mid"] = True
            break

    if cs["task"] == "explore":
        M = shared_cache(e)["M"]
        team_frac = (float((g.initial_unknown & (M >= 0.0)).sum()) / unknown_n
                     if g.initial_unknown is not None else 0.0)
        rec["own_region_frac"] = own_seen_n / unknown_n
        rec["team_region_frac"] = team_frac
        rec["d_final"] = d_tos[-1] if d_tos else None
        rec["progress_vs_shortest"] = (
            (rec["d_init"] - (d_tos[-1] if d_tos else rec["d_init"])) / max(1.0, cheb)
            if rec["d_init"] is not None and cheb > 0 else None)
    elif cs["task"] == "patrol":
        rec["ring_err_mean"] = float(np.mean(ring_errs)) if ring_errs else None
        rec["ring_within2"] = (float(np.mean([r < 2.0 for r in ring_errs]))
                               if ring_errs else 0.0)
        rec["in_band_after_frac"] = float(np.mean(in_band_after)) if in_band_after else 0.0
        rec["n_in_band_steps"] = len(in_band_after)
        rec["sectors"] = len(g.sectors)
        rec["resolved_ring_minmax"] = (min(ring_vals), max(ring_vals)) if ring_vals else None
    elif cs["task"] == "relay":
        rec["own_link_after_frac"] = (float(np.mean(own_link_after))
                                      if own_link_after else 0.0)
        rec["team_cover_after_frac"] = (float(np.mean(team_cover_after))
                                        if team_cover_after else 0.0)
        rec["d_final"] = d_tos[-1] if d_tos else None
    elif cs["task"] == "charge":
        rec["on_station_t"] = on_station_t
        rec["recharged"] = bool(recharged)
        rec["e_final"] = float(e.uavs[i].e)
    if record_traj:
        rec["traj"] = traj
    return rec


# ================================================================ 整局评估
def run_mission(seed, actor_path, regime="normal", protocol="goal_update"):
    """完整回合。protocol:
    - 'legacy_h'  : 旧 H 协议原样（每 20 步重发态势驱动的静态目标）；
    - 'goal_update'（默认，验收 §5 主协议）：H 决定任务，目标用"完成/失效更新"管理器
      维护——explore 目标区域完成(>=80% 已观测)/失效时换下一个合法未知区域，
      patrol 火点消失时退回 H 的下一目标。新旧 Actor 用同一协议，结果分别列出。
    """
    import torch
    import random as _random
    from experiments.lowlevel_training.sampler import TaskSampler
    from lowlevel.task_spec import TeamPlan, UavOrder
    actor = load_actor(actor_path)
    env = make_eval_env(seed, regime)
    env.reset()
    brain = make_h_brain({})
    sampler = TaskSampler(role_mode="randomized")
    rng = _random.Random(seed)
    obs = None
    done = False
    gm = GoalManager(grid_size=(100, 100))
    cur_orders = list(env.orders)
    cov_hist, aoi_hist = [], []

    while not done:
        if env.t % 20 == 0:
            from lowlevel.mission_demo import sense_state
            s = sense_state(env)
            plan, _, _ = brain.plan(s)
            h_orders = TeamPlan.from_llm_json(plan, env.num_uavs).orders
            if protocol == "goal_update":
                e = env.env
                e._goal_t = env.t
                gm.sync(e, cur_orders, env.t)
                new_orders = list(h_orders)
                for i in range(env.num_uavs):
                    ho = h_orders[i]
                    task, mode = int(ho.task), int(ho.mode)
                    g = gm.goals.get(i)
                    keep_cur = False
                    if (g is not None and not g.completed and env.t <= g.deadline_t):
                        same_kind = (
                            (g.kind == "explore" and task == 0 and mode == 0) or
                            (g.kind == "patrol" and task == 0 and mode == 1) or
                            (g.kind == "relay" and task == 1) or
                            (g.kind == "charge" and task == 2))
                        if same_kind:
                            if g.kind == "patrol":
                                keep_cur = len(shared_cache(e)["clusters"]) > 0
                            elif g.kind == "explore" and g.initial_unknown is not None:
                                left = float((g.initial_unknown &
                                              (shared_cache(e)["M"] < 0.0)).sum())
                                total = max(1, int(g.initial_unknown.sum()))
                                keep_cur = (not g.arrived) or left > 0.2 * total
                            else:
                                keep_cur = True
                    if keep_cur and i < len(cur_orders):
                        new_orders[i] = cur_orders[i]
                    elif task == 0 and mode == 0:
                        tgt, frac = sampler.sample_explore_target(
                            e, rng, float(e.uavs[i].x), float(e.uavs[i].y), "S3",
                            float(e.uavs[i].e))
                        if tgt is not None:
                            new_orders[i] = UavOrder(task=0, mode=0, target=tgt)
                        elif i < len(cur_orders):
                            new_orders[i] = cur_orders[i]
                        else:
                            new_orders[i] = ho
                    else:
                        new_orders[i] = ho
                cur_orders = new_orders
                obs = env.set_plan(list(cur_orders))
            else:
                cur_orders = list(h_orders)
                obs = env.set_plan(plan)
        with torch.no_grad():
            a = actor.act(torch.as_tensor(obs, dtype=torch.float32),
                          deterministic=True)[0].numpy()
        obs, r, done, info = env.step(a)
        cov_hist.append(float(getattr(env.env, "coverage_rate", 0.0)))
        aoi_hist.append(float(np.mean([r.AoI for r in env.env.rescuers]))
                        if env.env.rescuers else 0.0)
        if protocol == "goal_update":
            e = env.env
            e._goal_t = env.t
            gm.update(e, cur_orders, env.t)
    m = task_metrics(env)
    m["cov_mean"] = float(np.mean(cov_hist)) if cov_hist else 0.0
    m["aoi_mean_series"] = float(np.mean(aoi_hist)) if aoi_hist else 0.0
    m["aoi_p95_series"] = float(np.percentile(aoi_hist, 95)) if aoi_hist else 0.0
    m["seed"] = seed
    m["regime"] = regime
    m["steps_run"] = env.t
    m["cov_mean_proxy"] = float(getattr(env.env, "coverage_rate", 0.0))
    return m


# ================================================================ workers
def _worker_skill(job):
    seed, regime, actor_path, split = job
    try:
        import torch
        torch.set_num_threads(1)
        actor = load_actor(actor_path)
        rng = np.random.RandomState(seed * 7 + 11)
        snaps = build_snapshots(seed, regime, actor)
        recs = []
        for t0, (snap, base_orders) in snaps.items():
            cases = gen_cases(snap, base_orders, regime, seed, t0, rng)
            for cs in cases:
                if cs.get("skip"):
                    recs.append(dict(case=cs, skipped=cs["skip"]))
                    continue
                recs.append(run_case(snap, base_orders, cs, actor,
                                     record_traj=(split == "final")))
        return recs
    except Exception as ex:
        import traceback
        return [dict(error=f"{type(ex).__name__}: {ex}",
                     trace=traceback.format_exc()[-800:], seed=seed, regime=regime)]


def _worker_mission(job):
    seed, actor_path, regime, protocol = job
    try:
        import torch
        torch.set_num_threads(1)
        return run_mission(seed, actor_path, regime, protocol)
    except Exception as ex:
        import traceback
        return dict(error=str(ex), trace=traceback.format_exc()[-500:], seed=seed)


# ================================================================ 汇总
def summarize(recs, out_path, n_err):
    ok = [r for r in recs if "error" not in r and "skipped" not in r]

    def sel(task, **kw):
        s = [r for r in ok if r["case"]["task"] == task]
        for k, v in kw.items():
            s = [r for r in s if r["case"].get(k) == v]
        return s

    S = {}
    ex = sel("explore")
    valid = [r for r in ex if r["case"].get("stratum", -1) >= 0]
    if valid:
        S["explore_arrival"] = float(np.mean([r["first_arrival"] is not None
                                              for r in valid]))
        S["explore_region_own"] = float(np.mean(
            [r.get("own_region_frac", 0) >= 0.8 for r in valid]))
        S["explore_n"] = len(valid)
        for si, name in enumerate(("near", "mid", "far")):
            sub = [r for r in valid if r["case"]["stratum"] == si]
            if sub:
                S[f"explore_arrival_{name}"] = float(np.mean(
                    [r["first_arrival"] is not None for r in sub]))
    pa = sel("patrol")
    if pa:
        S["patrol_arrival"] = float(np.mean([r["first_arrival"] is not None for r in pa]))
        arr = [r for r in pa if r["first_arrival"] is not None
               and r["n_in_band_steps"] >= 5]
        S["patrol_inband_after"] = (float(np.mean([r["in_band_after_frac"] for r in arr]))
                                    if arr else 0.0)
        S["patrol_n"] = len(pa)
        S["patrol_sectors_mean"] = float(np.mean([r["sectors"] for r in pa]))
        for rg in REGIMES:
            sub = [r for r in pa if r["case"]["regime"] == rg]
            if sub:
                S[f"patrol_arrival_{rg}"] = float(np.mean(
                    [r["first_arrival"] is not None for r in sub]))
    rl = sel("relay")
    if rl:
        S["relay_arrival"] = float(np.mean([r["first_arrival"] is not None for r in rl]))
        arr = [r for r in rl if r["first_arrival"] is not None]
        S["relay_own_after"] = (float(np.mean([r["own_link_after_frac"] for r in arr]))
                                if arr else 0.0)
        S["relay_team_after"] = (float(np.mean([r["team_cover_after_frac"] for r in arr]))
                                 if arr else 0.0)
        S["relay_n"] = len(rl)
    ch = sel("charge")
    if ch:
        S["charge_success"] = float(np.mean([bool(r.get("recharged")) for r in ch]))
        S["charge_n"] = len(ch)
    for i in range(4):
        sub = [r for r in valid if r["case"]["agent"] == i]
        if sub:
            S[f"exp_arrival_uav{i}"] = float(np.mean(
                [r["first_arrival"] is not None for r in sub]))
        sub = [r for r in pa if r["case"]["agent"] == i]
        if sub:
            S[f"pat_arrival_uav{i}"] = float(np.mean(
                [r["first_arrival"] is not None for r in sub]))
        sub = [r for r in rl if r["case"]["agent"] == i]
        if sub:
            S[f"relay_arrival_uav{i}"] = float(np.mean(
                [r["first_arrival"] is not None for r in sub]))
    S["n_errors"] = n_err
    S["n_records"] = len(ok)
    print(json.dumps(S, indent=1, default=float))
    return S


def evaluate_skills(actor_path, out_dir, split="quick", workers=8, seeds=None,
                    regimes=("normal", "scarce")):
    if split == "quick":
        seeds = seeds or list(range(510000, 510008))
    elif split == "full":
        seeds = seeds or list(range(511000, 511012))
    else:
        seeds = seeds or list(range(520000, 520020))
    jobs = [(s, rg, actor_path, split) for rg in regimes for s in seeds]
    recs = []
    with mp.get_context("fork").Pool(min(workers, len(jobs))) as pool:
        for rs in pool.imap_unordered(_worker_skill, jobs):
            recs.extend(rs)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"skills_{split}.jsonl")
    n_err = sum(1 for r in recs if "error" in r)
    with open(out, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")
    S = summarize(recs, out, n_err)
    S["_actor"] = actor_path
    S["_sha256"] = sha256_of(actor_path)
    S["_split"] = split
    with open(os.path.join(out_dir, f"skills_{split}_summary.json"), "w") as f:
        json.dump(S, f, indent=1, default=float)
    return S


def evaluate_mission(actor_path, out_dir, episodes=20, workers=8, seed0=530000,
                      protocol="goal_update"):
    jobs = [(seed0 + k, actor_path, "normal", protocol) for k in range(episodes // 2)]
    jobs += [(seed0 + 500 + k, actor_path, "scarce", protocol)
             for k in range(episodes - episodes // 2)]
    ms = []
    with mp.get_context("fork").Pool(min(workers, len(jobs))) as pool:
        for m in pool.imap_unordered(_worker_mission, jobs):
            if "error" not in m:
                ms.append(m)
    agg = {}
    if ms:
        for k in ("explored", "burned_cells", "extinguished_cells", "contained",
                  "discovery", "rknown", "depleted", "mean_aoi", "cov_mean",
                  "aoi_mean_series", "aoi_p95_series"):
            vals = [m.get(k) for m in ms if m.get(k) is not None]
            if vals:
                agg[k] = float(np.mean(vals))
        for rg in ("normal", "scarce"):
            sub = [m for m in ms if m["regime"] == rg]
            if sub:
                agg[f"{rg}.n"] = len(sub)
                agg[f"{rg}.burned"] = float(np.mean([m["burned_cells"] for m in sub]))
                agg[f"{rg}.ext"] = float(np.mean([m["extinguished_cells"] for m in sub]))
                agg[f"{rg}.depleted"] = float(np.mean([m["depleted"] for m in sub]))
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"mission_{protocol}.jsonl")
    with open(out, "w") as f:
        for m in ms:
            f.write(json.dumps(m, ensure_ascii=False, default=float) + "\n")
    agg["_actor"] = actor_path
    agg["_sha256"] = sha256_of(actor_path)
    print(json.dumps(agg, indent=1))
    agg["_protocol"] = protocol
    with open(os.path.join(out_dir, f"mission_summary_{protocol}.json"), "w") as f:
        json.dump(agg, f, indent=1, default=float)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--split", default="quick", choices=["quick", "full", "final"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--mission", action="store_true")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--mission-seed0", type=int, default=530000)
    ap.add_argument("--protocol", default="goal_update", choices=["goal_update", "legacy_h"])
    ap.add_argument("--out-root", default=os.path.join(_HERE, "runs", "eval"))
    args = ap.parse_args()
    out_dir = os.path.join(args.out_root, args.tag or os.path.basename(args.actor))
    print(f"[eval] actor={args.actor} sha256={sha256_of(args.actor)[:16]}... "
          f"split={args.split} workers={args.workers}", flush=True)
    if args.mission:
        evaluate_mission(args.actor, out_dir, args.episodes, args.workers,
                         args.mission_seed0, args.protocol)
    else:
        evaluate_skills(args.actor, out_dir, args.split, args.workers, args.seeds)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""训练前必须通过的短检查（H200 计划 §3、§12）。

1. 混合字段契约：relay 同时给 target 与 rescuer_id 时，观测编码 = rescuer_id 解析的
   人员位置（契约 v2）；单字段/默认目标/charge 各自正确。
2. GoalEnv 基本运行：obs 形状/范围、奖励有限、info 字段齐全、目标事件推进。
3. 到访奖励去重：同一计划重发不重置一次性奖励；换目标才换 goal。
4. 奖励排序：同一 explore 目标，教师服从轨迹的 RewardV2 总回报 > 不服从（悬停/自由探索）
   轨迹——这是根因 1 修复的直接验证。
5. patrol 环半径：auto 半径随火点演化使用每步实际解析值。
6. GAE 不跨 episode、值归一化（mappo 现有逻辑抽查）。
"""
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "experiments", "lowlevel_training"))

import numpy as np
import torch

from lowlevel.task_spec import UavOrder, TASK_COMM_RELAY, TASK_RECHARGE
from lowlevel.obs_builder import build_obs, shared_cache, OBS_DIM, OBS_SLICES
from experiments.lowlevel_training.goal_env import GoalEnv
from experiments.lowlevel_training.teacher import teacher_action

FAILED = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name} {detail}")
    if not cond:
        FAILED.append(name)


def make_env(seed=123, **kw):
    return GoalEnv(seed=seed, num_uavs=4, num_rescuers=10, n_fire_seeds=3,
                  reward_version="P1", role_mode="randomized", **kw)


# ---------------------------------------------------------------- 1. 契约
def test_contract():
    env = GoalEnv(seed=42, reward_version="P1")
    env.reset()
    e = env.env
    c = shared_cache(e)
    # 取一个人员位置
    r = e.rescuers[0]
    person = (float(r.x), float(r.y))
    other = (float(e.rescuers[-1].x), float(e.rescuers[-1].y))
    u = e.uavs[0]
    ux, uy = float(u.x), float(u.y)

    # 混合字段：target=other, rescuer_id=r.rescuer_id -> 解析目标应为 person
    o = UavOrder(task=TASK_COMM_RELAY, target=other, rescuer_id=int(r.rescuer_id))
    from lowlevel.obs_builder import resolve_order_target
    tgt, _f, _rg = resolve_order_target(e, 0, o, c)
    check("contract.resolve_rescuer_first", math.hypot(tgt[0] - person[0], tgt[1] - person[1]) < 1e-6,
          f"resolved=({tgt[0]:.1f},{tgt[1]:.1f}) person=({person[0]:.1f},{person[1]:.1f})")
    vec = o.to_vector(ux, uy, 141.0, resolved_target=tgt)
    # 方向位应指向 person
    dxn, dyn = vec[2], vec[3]
    exp_dx = (person[0] - ux) / max(1e-6, math.hypot(person[0] - ux, person[1] - uy))
    exp_dy = (person[1] - uy) / max(1e-6, math.hypot(person[0] - ux, person[1] - uy))
    check("contract.v2_encodes_resolved", abs(dxn - exp_dx) < 0.02 and abs(dyn - exp_dy) < 0.02,
          f"encoded_dir=({dxn:.3f},{dyn:.3f}) expected=({exp_dx:.3f},{exp_dy:.3f})")
    # 显式目标位保持：有 target 或 rescuer_id 时为 1
    check("contract.explicit_flag", vec[7] == 1.0, f"flag={vec[7]}")
    # 单字段：只有 target -> 编码 target
    o2 = UavOrder(task=TASK_COMM_RELAY, target=other)
    obs2 = build_obs(e, 0, o2, cache=c)
    v2 = obs2[OBS_SLICES["order"]]
    exp_dx2 = (other[0] - ux) / max(1e-6, math.hypot(other[0] - ux, other[1] - uy))
    check("contract.target_only", abs(v2[2] - exp_dx2) < 0.02, f"dir_x={v2[2]:.3f} exp={exp_dx2:.3f}")
    # 默认（无字段）：resolved_target 兜底（最近未覆盖人员）
    o3 = UavOrder(task=TASK_COMM_RELAY)
    obs3 = build_obs(e, 0, o3, cache=c)
    v3 = obs3[OBS_SLICES["order"]]
    check("contract.default_flag0", v3[7] == 0.0, f"flag={v3[7]}")
    # charge：观测方向指向站点
    o4 = UavOrder(task=TASK_RECHARGE)
    obs4 = build_obs(e, 0, o4, cache=c)
    v4 = obs4[OBS_SLICES["order"]]
    from lowlevel.obs_builder import assigned_station
    st, _ = assigned_station(e, 0)
    exp_dx4 = (st[0] - ux) / max(1e-6, math.hypot(st[0] - ux, st[1] - uy))
    check("contract.charge_station", abs(v4[2] - exp_dx4) < 0.02, f"dir_x={v4[2]:.3f} exp={exp_dx4:.3f}")


# ---------------------------------------------------------------- 2. GoalEnv
def test_goalenv():
    env = make_env(seed=7)
    obs = env.reset()
    check("env.obs_shape", obs.shape == (4, 355), str(obs.shape))
    check("env.obs_finite", np.isfinite(obs).all())
    tot = np.zeros(4)
    ev_keys_seen = set()
    for k in range(60):
        a = np.random.randint(0, 9, size=4)
        obs, r, done, info = env.step(a)
        check("env.reward_finite", np.isfinite(r).all()) if k == 0 else None
        tot += r
        ge = info.get("goal_events") or {}
        for i, ev in ge.items():
            if "d_now" in ev:
                ev_keys_seen.update(ev.keys())
        if done:
            env.reset()
    check("env.reward_finite", np.isfinite(tot).all(), f"60-step return={np.round(tot,2)}")
    need = {"d_now", "progress", "g_arrived", "g_completed"}
    check("env.goal_events", need.issubset(ev_keys_seen), f"keys={sorted(ev_keys_seen)[:8]}...")
    check("env.info_tasks", "tasks" in info and "modes" in info)


# ---------------------------------------------------------------- 3. 去重
def test_dedup():
    env = make_env(seed=99)
    env.reset()
    # 给 UAV0 一个显式 explore 目标
    from lowlevel.task_spec import TASK_FIRE_SENSING, SENSE_EXPLORE
    e = env.env
    c = shared_cache(e)
    M = c["M"]
    # 找一个未知比例高的区域
    tgt = None
    for _ in range(200):
        x, y = np.random.randint(10, 90, 2)
        reg = M[max(0, x - 5):x + 6, max(0, y - 5):y + 6]
        if (reg < 0).mean() >= 0.8 and math.hypot(x - e.uavs[0].x, y - e.uavs[0].y) > 15:
            tgt = (float(x), float(y))
            break
    if tgt is None:
        check("dedup.setup", False, "no valid region found")
        return
    orders = list(env.orders)
    orders[0] = UavOrder(task=TASK_FIRE_SENSING, mode=SENSE_EXPLORE, target=tgt)
    env.set_plan(list(orders))
    # 教师飞到目标
    arr_bonus_count = 0
    for k in range(120):
        cache = shared_cache(env.env)
        acts = [teacher_action(env.env, i, env.orders[i], cache) for i in range(4)]
        obs, r, done, info = env.step(acts)
        p = info["parts"][0] if "parts" in info else {}
        if "arrival" in p:
            arr_bonus_count += 1
        g = env.goal_mgr.goals[0]
        if g.arrived:
            break
    g = env.goal_mgr.goals[0]
    check("dedup.first_arrival", g.arrived, f"d_init={g.d_init:.1f} t={env.t}")
    check("dedup.arrival_once", arr_bonus_count == 1, f"arrival bonuses={arr_bonus_count}")
    key0 = g.goal_key
    # 重发同一计划：不换 goal、不重置事件
    env.set_plan(list(orders))
    g2 = env.goal_mgr.goals[0]
    check("dedup.resend_keeps_goal", g2.goal_key == key0 and g2.arrived and g2.arrival_t == g.arrival_t,
          f"key={g2.goal_key} arrived={g2.arrived}")
    # 换目标：换 goal
    orders[0] = UavOrder(task=TASK_FIRE_SENSING, mode=SENSE_EXPLORE, target=(tgt[0] + 20, tgt[1]))
    env.set_plan(list(orders))
    g3 = env.goal_mgr.goals[0]
    check("dedup.new_target_new_goal", g3.goal_key != key0 and not g3.arrived,
          f"key={g3.goal_key}")
    # 完成判定：own_seen 覆盖率
    fr = env.goal_mgr.goals[0]
    check("dedup.region_masks", fr.initial_unknown is not None and fr.own_seen is not None)


# ---------------------------------------------------------------- 4. 奖励排序
def test_reward_ordering():
    """同一 explore 目标：教师服从 vs 悬停不服从 vs 随机游走。V2 应让服从严格更优。"""
    from lowlevel.task_spec import TASK_FIRE_SENSING, SENSE_EXPLORE

    def run(mode, seed=31, steps=80):
        env = make_env(seed=seed)
        env.reset()
        # 固定初始计划：UAV0 explore 远目标，其余原样
        e = env.env
        c = shared_cache(e)
        M = c["M"]
        tgt = None
        rs = np.random.RandomState(0)
        for _ in range(300):
            x, y = rs.randint(10, 90, 2)
            reg = M[max(0, x - 5):x + 6, max(0, y - 5):y + 6]
            if (reg < 0).mean() >= 0.8 and 30 < math.hypot(x - e.uavs[0].x, y - e.uavs[0].y) < 55:
                tgt = (float(x), float(y))
                break
        if tgt is None:
            return None, None
        orders = list(env.orders)
        orders[0] = UavOrder(task=TASK_FIRE_SENSING, mode=SENSE_EXPLORE, target=tgt)
        env.set_plan(list(orders))
        ret = 0.0
        for k in range(steps):
            if mode == "teacher":
                cache = shared_cache(env.env)
                acts = [teacher_action(env.env, i, env.orders[i], cache) for i in range(4)]
            elif mode == "hover":
                acts = [8] * 4
            else:
                acts = list(np.random.randint(0, 9, 4))
            obs, r, done, info = env.step(acts)
            ret += float(r[0])
            if done:
                break
        g = env.goal_mgr.goals[0]
        return ret, (g.arrived, g.completed, g.d_init)

    rt, gt = run("teacher")
    rh, gh = run("hover")
    rr, gr = run("random")
    if rt is None or rh is None:
        check("reward.setup", False, "no valid far target")
        return
    check("reward.teacher_arrives", bool(gt[0]), f"teacher={gt}")
    check("reward.hover_not_arrive", not gh[0], f"hover={gh}")
    check("reward.v2_teacher_beats_hover", rt > rh + 1.0,
          f"teacher={rt:.2f} hover={rh:.2f}")
    if rr is not None:
        check("reward.v2_teacher_beats_random", rt > rr,
              f"random={rr:.2f}")


# ---------------------------------------------------------------- 5. patrol 环半径
def test_patrol_ring():
    from lowlevel.task_spec import TASK_FIRE_SENSING, SENSE_PATROL
    env = make_env(seed=55)
    env.reset()
    # 用教师探索一段时间发现火点（随机游走发现太慢）
    for k in range(200):
        cache = shared_cache(env.env)
        acts = [teacher_action(env.env, i, env.orders[i], cache) for i in range(4)]
        obs, r, done, info = env.step(acts)
        if done:
            env.reset()
    c = shared_cache(env.env)
    if not c["clusters"]:
        check("patrol.no_cluster", False, "no cluster after 200 teacher steps")
        return
    cl = c["clusters"][0]
    orders = list(env.orders)
    orders[0] = UavOrder(task=TASK_FIRE_SENSING, mode=SENSE_PATROL,
                         target=tuple(cl["centroid"]), ring_radius=None)
    env.set_plan(list(orders))
    rings_seen = []
    for k in range(40):
        cache = shared_cache(env.env)
        acts = [teacher_action(env.env, i, env.orders[i], cache) for i in range(4)]
        obs, r, done, info = env.step(acts)
        ev = (info.get("goal_events") or {}).get(0)
        if ev and "ring" in ev:
            rings_seen.append(ev["ring"])
        if done:
            break
    g = env.goal_mgr.goals[0]
    check("patrol.goal_created", g.kind == "patrol", f"kind={g.kind}")
    check("patrol.ring_resolved", len(rings_seen) > 0, f"n={len(rings_seen)}")
    check("patrol.teacher_in_band", g.arrived, f"arrived={g.arrived} sectors={len(g.sectors)}")
    # 环半径每步记录（auto 可变）
    check("patrol.per_step_ring", len(set(rings_seen)) >= 1,
          f"distinct_rings={len(set(rings_seen))}")


# ---------------------------------------------------------------- 6. GAE / 边界
def test_gae():
    from lowlevel.mappo import RolloutBuffer
    dev = torch.device("cpu")
    T, E, N, D, G = 6, 2, 4, 355, 8
    buf = RolloutBuffer(T, E, N, D, G, dev)
    obs = torch.zeros(E, N, D)
    glb = torch.zeros(E, G)
    buf.start(obs, glb)
    for t in range(T):
        rew = torch.zeros(E, N)
        done = torch.zeros(E)
        if t == 2:
            rew[0, :] = 1.0            # env0 在 t=2 有奖励，且该转移结束后 done
            done[0] = 1.0              # 存进 done[t+1]，切断 t=3 之后的回传
        buf.add(obs, glb, torch.zeros(E, N, dtype=torch.long),
                torch.zeros(E, N), torch.zeros(E, N), rew, done)
    buf.compute_gae(torch.zeros(E, N), 0.99, 0.95)
    a = buf.adv[:T, 0, 0]
    # t=2 的奖励只属于 t=2：done[3]=1 切断后续 bootstrap；t=1 通过 GAE 折回 γλ·1；
    # t=3 是新 episode，与 t=2 的奖励无关
    ok = (abs(float(a[2]) - 1.0) < 1e-6 and abs(float(a[3])) < 1e-6
          and abs(float(a[1]) - 0.99 * 0.95) < 1e-4)
    check("gae.no_cross_episode", ok, f"adv={np.round(a.tolist(), 4)}")


if __name__ == "__main__":
    torch.set_num_threads(1)
    test_contract()
    test_goalenv()
    test_dedup()
    test_reward_ordering()
    test_patrol_ring()
    test_gae()
    print("\n==== SUMMARY ====")
    if FAILED:
        print(f"FAILED: {FAILED}")
        sys.exit(1)
    print("ALL CHECKS PASSED")

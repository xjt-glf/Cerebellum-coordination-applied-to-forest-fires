# -*- coding: utf-8 -*-
"""示范数据生成（H200 计划 §5.3 P3 / §9 R2 的行为克隆热身）。

教师 = 合法导航控制器：只用控制时可知状态（自身位置/电量、指令解析目标、
9x9 局部地图），动作遵守同一碰撞与能量规则（8 方向 1 格/步 + 悬停）。
- explore：飞向目标区域，区域内系统覆盖（朝最近的未见区域格移动）；
- patrol：环带外逼近、环带内切向绕行（固定方向）；
- relay：飞向指定人员/固定点，进入通信半径后跟随；
- charge：飞向分配站点，入站后悬停充电。

教师只用于生成 BC 热身数据，正式验收与部署全部由 Actor 输出动作。
"""
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lowlevel.task_spec import (
    TASK_FIRE_SENSING, TASK_COMM_RELAY, TASK_RECHARGE,
    SENSE_EXPLORE, SENSE_PATROL,
)
from lowlevel.obs_builder import resolve_order_target, shared_cache

# 9 动作：0-7 八方向（与 低层环境.py 的 action_set 一致），8 悬停
MOVES = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]


def teacher_action(env_raw, i, order, cache=None):
    """合法导航教师。返回动作 id ∈ {0..8}。"""
    cache = cache if cache is not None else shared_cache(env_raw)
    env = env_raw
    u = env.uavs[i]
    ux, uy = float(u.x), float(u.y)
    H, W = env.grid_size
    M = cache["M"]
    tgt, is_fire, ring = resolve_order_target(env, i, order, cache)
    task, mode = int(order.task), int(order.mode)

    def step_pos(a):
        dx, dy = MOVES[a]
        nx, ny = ux + dx, uy + dy
        ok = (0 <= nx < H and 0 <= ny < W)
        return (nx, ny), ok

    def dist_to(x, y):
        return math.hypot(x - tgt[0], y - tgt[1]) if tgt is not None else 1e9

    # ---- 子目标 ----
    if task == TASK_RECHARGE:
        if env.at_station(ux, uy):
            return 8
        sub = tgt
    elif task == TASK_COMM_RELAY:
        r_comm = float(getattr(env, "uav_comm_radius", 12))
        if tgt is not None and dist_to(ux, uy) <= r_comm * 0.8:
            return 8                      # 已在覆盖内，驻留
        sub = tgt
    elif task == TASK_FIRE_SENSING and mode == SENSE_PATROL and is_fire:
        ring_eff = float(ring if ring is not None else 6.0)
        d = dist_to(ux, uy)
        err = abs(d - ring_eff)
        if err <= 2.0:
            # 环带内：切向绕行（固定顺时针）
            ang = math.atan2(uy - tgt[1], ux - tgt[0])
            tang = ang + math.pi / 2.0
            sub = (tgt[0] + ring_eff * math.cos(tang), tgt[1] + ring_eff * math.sin(tang))
        elif d > ring_eff:
            sub = tgt                     # 太远，靠近
        else:
            # 太近（在火里），沿径向外推
            ang = math.atan2(uy - tgt[1], ux - tgt[0])
            sub = (tgt[0] + (ring_eff + 2.0) * math.cos(ang),
                   tgt[1] + (ring_eff + 2.0) * math.sin(ang))
    else:
        # explore（显式目标）：先到区域，再覆盖区域内未见格
        if tgt is None:
            return 8
        d = dist_to(ux, uy)
        hw = 5
        if d > 4.0:
            sub = tgt
        else:
            # 最近的区域未知格（自己的认知图内未知 = -1）
            best, bd = None, 1e9
            x0, x1 = max(0, int(tgt[0]) - hw), min(H, int(tgt[0]) + hw + 1)
            y0, y1 = max(0, int(tgt[1]) - hw), min(W, int(tgt[1]) + hw + 1)
            r_sense = int(getattr(env, "sense_radius", 4))
            for x in range(x0, x1):
                for y in range(y0, y1):
                    if M[x, y] < 0.0:
                        dd = math.hypot(x - ux, y - uy)
                        if dd < bd:
                            best, bd = (float(x), float(y)), dd
            if best is None or bd <= r_sense * 0.5:
                return 8
            sub = best
    if sub is None:
        return 8

    # ---- 贪心选动作：最小化到子目标距离，越界大罚，火格小罚 ----
    best_a, best_c = 8, 1e18
    for a in range(9):
        if a == 8:
            (nx, ny), ok = (ux, uy), True
        else:
            (nx, ny), ok = step_pos(a)
        if not ok:
            continue
        c = math.hypot(nx - sub[0], ny - sub[1])
        xi, yi = int(round(nx)), int(round(ny))
        if 0 <= xi < H and 0 <= yi < W:
            bs = env.grid["burning_state"]
            if 0 < bs[xi, yi] < 2:
                c += 1.5                  # 尽量别穿火
        if c < best_c - 1e-9:
            best_c, best_a = c, a
    return best_a


# ------------------------------------------------------------ 数据生成
def _gen_episode(args):
    """跑一个教师 episode，返回 (obs_list, act_list)。在 worker 进程内执行。"""
    seed, n_steps, kw = args
    try:
        from .goal_env import GoalEnv
    except ImportError:
        from experiments.lowlevel_training.goal_env import GoalEnv
    env = GoalEnv(seed=seed, **kw)
    obs = env.reset()
    X, Y = [], []
    for _ in range(n_steps):
        cache = shared_cache(env.env)
        acts = [teacher_action(env.env, i, env.orders[i], cache)
                for i in range(env.num_uavs)]
        X.append(obs.copy())
        Y.append(list(acts))
        obs, _r, done, _info = env.step(acts)
        if done:
            obs = env.reset()
    env.close()
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.int64)


def generate_teacher_data(out_path, target_transitions=100000, workers=8,
                          n_steps_per_env=600, seed0=700000, env_kw=None):
    """并行生成教师示范（单机 transition 计数 = steps × num_uavs）。"""
    import multiprocessing as mp
    env_kw = dict(env_kw or {})
    n_uav = int(env_kw.get("num_uavs", 4))
    n_eps = int(math.ceil(target_transitions / (n_steps_per_env * n_uav))) + 2
    jobs = [(seed0 + k, n_steps_per_env, env_kw) for k in range(n_eps)]
    Xs, Ys = [], []
    with mp.get_context("fork").Pool(min(workers, n_eps)) as pool:
        for X, Y in pool.imap_unordered(_gen_episode, jobs):
            Xs.append(X.reshape(-1, X.shape[-1]))
            Ys.append(Y.reshape(-1))
    X = np.concatenate(Xs)
    Y = np.concatenate(Ys)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, obs=X, act=Y)
    return out_path, X.shape[0]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_HERE, "runs", "teacher_bc.npz"))
    ap.add_argument("--target", type=int, default=100000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    p, n = generate_teacher_data(args.out, args.target, args.workers)
    print(f"[teacher] saved {n} transitions -> {p}")

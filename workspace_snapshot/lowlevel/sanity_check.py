# -*- coding: utf-8 -*-
"""训练前自检：确认新场景下的奖励是可学的。

判据：一个只会"照着指令目标直线飞"的贪心策略，必须在各类指令下都明显优于随机。
如果贪心都拿不到分，说明奖励的引导项有问题，先修奖励再训练。

    python3 lowlevel/sanity_check.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lowlevel.env_wrapper import TaskConditionedEnv
from lowlevel.obs_builder import OBS_SLICES
from lowlevel.task_spec import (
    UavOrder, TASK_FIRE_SENSING, TASK_COMM_RELAY, TASK_RECHARGE,
    SENSE_EXPLORE, SENSE_PATROL,
)

# 与 Env/低层环境.py step() 里的动作表一致；动作 8 = 原地悬停
DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]


def greedy_action(obs_i, order):
    """照着 order 段里的目标方向走。

    order 段布局见 UavOrder.to_vector：[mode(2), tdx, tdy, tdist, ring, ring_err, ...]
    patrol 模式下：环误差为正（太远）就往里飞，为负（太近）就往外飞，
    误差很小就沿切向绕圈。
    """
    seg = obs_i[OBS_SLICES["order"]]
    tdx, tdy, tdist = float(seg[2]), float(seg[3]), float(seg[4])
    if abs(tdx) < 1e-6 and abs(tdy) < 1e-6:
        return 8
    ring_err = float(seg[6])
    target_is_fire = float(seg[8]) > 0.5
    # 只有"目标确实是火点"才走绕圈逻辑；没有火点时 patrol 退化成搜寻，
    # 否则贪心会围着一个未探索块的中心一直转，反而不如随机（实测 explored 0.12）
    if target_is_fire:
        if abs(ring_err) < 0.02:                 # 已经在环上 → 沿切向绕
            tdx, tdy = -tdy, tdx
        elif ring_err < 0:                       # 离火点太近 → 往外
            tdx, tdy = -tdx, -tdy
    best, best_dot = 8, 0.2
    for a, (dx, dy) in enumerate(DIRS):
        n = (dx * dx + dy * dy) ** 0.5
        dot = (dx * tdx + dy * tdy) / n
        if dot > best_dot:
            best, best_dot = a, dot
    return best


CASES = {
    "explore x4": [UavOrder(task=TASK_FIRE_SENSING, mode=SENSE_EXPLORE)] * 4,
    "relay x4": [UavOrder(task=TASK_COMM_RELAY)] * 4,
    "recharge x4": [UavOrder(task=TASK_RECHARGE)] * 4,
    "explore2+relay1+charge1": [
        UavOrder(task=TASK_FIRE_SENSING, mode=SENSE_EXPLORE),
        UavOrder(task=TASK_FIRE_SENSING, mode=SENSE_EXPLORE),
        UavOrder(task=TASK_COMM_RELAY), UavOrder(task=TASK_RECHARGE)],
}


def patrol_at_fires(env):
    """给出"绕真实火点巡逻"的指令：目标取初始火点中心，环半径 6。

    这是唯一能真正检验环半径势函数的用例——高层给不出火点时 patrol 会退化成搜寻。
    """
    seeds = list(getattr(env.env, "fire_seeds", []))
    h, w = env.env.fire_seed_hw
    orders = []
    for i in range(env.num_uavs):
        if seeds:
            cx, cy = seeds[i % len(seeds)]
            tgt = (float(cx) + h / 2.0, float(cy) + w / 2.0)
        else:
            tgt = None
        orders.append(UavOrder(task=TASK_FIRE_SENSING, mode=SENSE_PATROL,
                               target=tgt, ring_radius=None))
    return orders


def patrol_mission(env):
    """稳态阶段的真实编队：2 台绕火点巡逻 + 1 台中继 + 1 台返航补给。"""
    orders = patrol_at_fires(env)[:2]
    orders.append(UavOrder(task=TASK_COMM_RELAY, rescuer_id=1))
    orders.append(UavOrder(task=TASK_RECHARGE))
    return orders


# 说明：不测"patrol 但高层没给火点"这种指令。规划器在没有已知火点时给的是
# explore（见 env_wrapper._planner_orders），那是分布外的指令；贪心基线在这种
# 指令下会去绕一个未探索块的中心，反而不如随机乱走，据此判定奖励有问题是误判。
CASES["patrol x4 @fire"] = patrol_at_fires
CASES["patrol2+relay1+charge1"] = patrol_mission

# 只对"规划器真的会下发的编队"做通过/不通过判定。
# `relay x4` 是分布外指令：规划器只派 1–2 台中继，4 台全去中继时它们会盯上同一批
# 人员（人员初期都聚在基站附近），扎堆惩罚每步都触发，贪心因此比随机还差——
# 这是奖励在正确地抑制扎堆，不是奖励有问题。保留该行仅作参考。
GATED = {"explore x4", "recharge x4", "explore2+relay1+charge1",
         "patrol x4 @fire", "patrol2+relay1+charge1"}

# 每个用例看哪个量。`recharge x4` 不能看回报：全队停在补给点 600 步，谁都不去
# 搜寻/中继，人员拿不到火情、火自由蔓延，全队共享的过火面积惩罚会把回报压到很低——
# 这是奖励在**正确地**表达"把所有人都派去充电是坏决策"。该用例真正要验的是
# "给了 tau=2 会不会真的去充电"，看耗尽数。
# 列索引与 rollout() 的返回顺序一致：
# 0 return, 1 explored, 2 disc, 3 burned, 4 cov, 5 rknown, 6 ext, 7 depl
GATE_METRIC = {"recharge x4": (7, "lower")}


def rollout(kind, orders, seed, max_steps=600):
    env = TaskConditionedEnv(num_uavs=4, max_steps=max_steps, seed=seed,
                             task_sampler="mix", replan_every=0)
    if callable(orders):
        env.reset()
        obs = env.set_plan(orders(env))
    else:
        obs = env.reset(plan=[UavOrder(**vars(o)) for o in orders])
    rng = np.random.RandomState(seed)
    tot = 0.0
    cov, disc, rk = [], [], []
    for _ in range(max_steps):
        if kind == "random":
            a = rng.randint(0, 9, size=4)
        else:
            a = [greedy_action(obs[i], env.orders[i]) for i in range(4)]
        obs, r, done, info = env.step(a)
        tot += float(np.sum(r))
        cov.append(info["coverage_rate"])
        disc.append(info["fire_discovery_rate"])
        rk.append(info["rescuer_known_fire_rate"])
        if done:
            break
    env.close()
    # disc / rknown / cov 取回合均值：末值的分母是"当前真实燃烧格数"，
    # 人员灭火能力增强后回合末只剩个别燃烧格，末值会剧烈抖动
    return (tot, info["explored_rate"], float(np.mean(disc)),
            info["burned_area_rate"], float(np.mean(cov)),
            float(np.mean(rk)), info["n_extinguished"], info["n_depleted"])


def main():
    n_ep = int(os.environ.get("SANITY_EPISODES", 3))
    print(f"{'case':24s} {'policy':8s} {'return':>9s} {'explored':>9s} {'disc':>6s} "
          f"{'burned':>7s} {'cov':>6s} {'rknown':>7s} {'ext':>5s} {'depl':>5s}")
    print("-" * 92)
    ok = True
    for name, orders in CASES.items():
        got = {}
        for kind in ("random", "greedy"):
            rs = [rollout(kind, orders, 7000 + s) for s in range(n_ep)]
            a = np.array(rs, dtype=np.float64)
            got[kind] = a.mean(axis=0)
            print(f"{name:24s} {kind:8s} {a[:,0].mean():9.2f} {a[:,1].mean():9.3f} "
                  f"{a[:,2].mean():6.3f} {a[:,3].mean():7.4f} {a[:,4].mean():6.3f} "
                  f"{a[:,5].mean():7.3f} {a[:,6].mean():5.0f} {a[:,7].mean():5.2f}")
        col, direction = GATE_METRIC.get(name, (0, "higher"))
        worse = (got["greedy"][col] <= got["random"][col] if direction == "higher"
                 else got["greedy"][col] >= got["random"][col])
        if worse:
            if name in GATED:
                ok = False
                print(f"  !! {name}: 贪心在判定指标上没有优于随机，奖励引导项需要检查")
            else:
                print(f"  ~  {name}: 分布外指令，仅供参考（不参与判定）")
        print("-" * 92)
    print("\nRESULT:", "PASS - 奖励可学，可以开始训练" if ok else "FAIL - 先修奖励")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

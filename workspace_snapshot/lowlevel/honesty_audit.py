# -*- coding: utf-8 -*-
"""诚实性审计：检查"分批次充电"是不是真的由策略学出来的。

背景
----
历史实现里每步能耗是 `uav.e -= idx+1`，即第 i 架每步耗 i+1。同构机队（都是
e=600, d=5）本来不该有这种差异，它的唯一效果是让各机在**不同时刻**耗尽电量，
于是"看起来"出现了分批返航充电——但这是物理设定造成的，不是策略学到的。
现在能耗已统一为常数（`env.energy_cost`，默认 2.5 = mean(1,2,3,4)，保持机队
总能耗与历史实验一致）。

本脚本做两件事：
1. 报告 uniform / legacy_index 两种能耗下的"耗尽时刻离散度"。
   legacy 下离散度大是物理造成的；uniform 下如果仍然大，才是策略的功劳。
2. 报告通信中断步数、覆盖率、同时在站机数，作为错峰效果的客观指标。

    python3 lowlevel/honesty_audit.py
    python3 lowlevel/honesty_audit.py --actor lowlevel/runs/xxx/actor_best.pt
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lowlevel.env_wrapper import TaskConditionedEnv
from lowlevel.task_spec import UavOrder
from lowlevel.sanity_check import greedy_action


def rollout(taus, mode, seed, kind="greedy", actor=None, device="cpu",
            num_uavs=4, max_steps=600, energy_cost=2.5):
    env = TaskConditionedEnv(num_uavs=num_uavs, max_steps=max_steps, seed=seed,
                             replan_every=0)
    obs = env.reset(plan=[UavOrder(task=int(t)) for t in taus])
    env.env.energy_cost_mode = mode
    env.env.energy_cost = energy_cost
    e0 = float(getattr(env.env, "uav_energy", 600.0))
    for u in env.env.uavs:            # reset 后重置电量，保证两种模式起点一致
        u.e = e0

    rng = np.random.RandomState(seed)
    depletion_step = [None] * num_uavs
    blackout, cov_hist, on_station = 0, [], []
    for t in range(max_steps):
        if kind == "random":
            a = rng.randint(0, 9, size=num_uavs)
        elif kind == "greedy":
            a = [greedy_action(obs[i], env.orders[i]) for i in range(num_uavs)]
        else:
            import torch
            with torch.no_grad():
                o = torch.as_tensor(obs, dtype=torch.float32, device=device)
                a = actor.act(o, deterministic=True)[0].cpu().numpy()
        obs, r, done, info = env.step(a)
        for i, e in enumerate(info["energies"]):
            if e <= 0.0 and depletion_step[i] is None:
                depletion_step[i] = t
        blackout += int(info["blackout"])
        cov_hist.append(info["coverage_rate"])
        on_station.append(info["n_on_station"])
        if done:
            break
    env.close()

    ds = [d for d in depletion_step if d is not None]
    return {
        "n_depleted": len(ds),
        "depletion_std": float(np.std(ds)) if len(ds) >= 2 else 0.0,
        "depletion_spread": float(max(ds) - min(ds)) if len(ds) >= 2 else 0.0,
        "blackout_steps": blackout,
        "mean_cov": float(np.mean(cov_hist)),
        "max_on_station": int(max(on_station)) if on_station else 0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--actor", default=None)
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--num-uavs", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--energy-cost", type=float, default=2.5)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    kinds = ["random", "greedy"]
    actor = None
    if args.actor and os.path.exists(args.actor):
        import torch
        from lowlevel.models import Actor
        from lowlevel.obs_builder import OBS_DIM
        actor = Actor(OBS_DIM, 9).to(args.device)
        actor.load_state_dict(torch.load(args.actor, map_location=args.device))
        actor.eval()
        kinds.append("policy")

    taus = [1, 1, 1, 1]     # 全队中继：最能暴露"同时耗尽 -> 通信全断"的场景
    print("场景：4 机全部执行通信中继（tau=1），观察耗尽时刻是否离散\n")
    hdr = (f"{'energy_mode':14s} {'policy':8s} {'depl':>5s} {'depl_std':>9s} "
           f"{'depl_spread':>12s} {'blackout':>9s} {'mean_cov':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for mode in ("legacy_index", "uniform"):
        for kind in kinds:
            rs = [rollout(taus, mode, 50_000 + s, kind, actor, args.device,
                          args.num_uavs, args.max_steps, args.energy_cost)
                  for s in range(args.episodes)]
            f = lambda k: np.mean([x[k] for x in rs])
            print(f"{mode:14s} {kind:8s} {f('n_depleted'):5.1f} "
                  f"{f('depletion_std'):9.1f} {f('depletion_spread'):12.1f} "
                  f"{f('blackout_steps'):9.1f} {f('mean_cov'):9.3f}")
        print("-" * len(hdr))

    print("""
怎么读这张表
------------
* legacy_index 行的 depl_spread 会很大（≈ 能耗差异直接决定），但这**不是**策略的
  功劳——换成随机策略同样很大。这就是原来"分批充电"假象的来源。
* uniform 行才是诚实的基线：随机/贪心策略下 depl_spread 应接近 0（同时耗尽），
  blackout 很高。只有训练后的 policy 把 depl_spread 拉开、blackout 压低，
  才能说错峰是学出来的。
* 注意：**决定"谁在什么时候去充电"的是高层**（tau=2 的分配），低层只负责执行。
  所以这张表在 uniform 下能否改善，最终取决于高层观测里是否包含队友电量信息。
  详见 lowlevel/README.md 第 10 节。
""")


if __name__ == "__main__":
    main()

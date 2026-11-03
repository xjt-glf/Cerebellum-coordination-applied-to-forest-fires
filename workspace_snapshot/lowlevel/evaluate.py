# -*- coding: utf-8 -*-
"""验收脚本：在若干"真实编队指令"下对比 random / greedy / policy。

    python3 lowlevel/evaluate.py --actor lowlevel/runs/<run>/actor_best.pt

判据（接大模型之前）：
    - 每个指令组合下 policy 回报都显著高于 random；
    - 搜寻编队的**全地图探索率** ≳ 0.75；
    - 巡逻+中继编队的 火情发现率 与 人员已知火情比例 ≳ 0.70，覆盖率 ≳ 0.70；
    - 全队返航指令下 n_depleted ≈ 0；
    - 指令敏感性：换目标点/换人员，指标要跟着变（说明 advice 真的被用上了）。
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lowlevel.env_wrapper import TaskConditionedEnv
from lowlevel.models import Actor
from lowlevel.obs_builder import OBS_DIM
from lowlevel.sanity_check import greedy_action, CASES
from lowlevel.task_spec import (
    UavOrder, TASK_FIRE_SENSING, TASK_COMM_RELAY, TASK_RECHARGE,
    SENSE_EXPLORE, SENSE_PATROL,
)

METRICS = ("return", "explored", "disc", "burned", "cov", "rknown", "depl")


def rollout(kind, orders, seed, actor=None, device="cpu", num_uavs=4, max_steps=600,
            replan_every=0, task_sampler="planner"):
    env = TaskConditionedEnv(num_uavs=num_uavs, max_steps=max_steps, seed=seed,
                             task_sampler=task_sampler, replan_every=replan_every)
    if orders is None:
        obs = env.reset()                       # 让内置规划器全程接管
    elif callable(orders):
        env.reset()
        obs = env.set_plan(orders(env))
    else:
        obs = env.reset(plan=[UavOrder(**vars(o)) for o in orders])
    rng = np.random.RandomState(seed)
    tot, cov, disc, rk = 0.0, [], [], []
    for _ in range(max_steps):
        if kind == "random":
            a = rng.randint(0, 9, size=num_uavs)
        elif kind == "greedy":
            a = [greedy_action(obs[i], env.orders[i]) for i in range(num_uavs)]
        else:
            with torch.no_grad():
                o = torch.as_tensor(obs, dtype=torch.float32, device=device)
                a = actor.act(o, deterministic=True)[0].cpu().numpy()
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


def table(name, arms, episodes, actor, device, args, orders):
    out = {}
    for kind in arms:
        rs = [rollout(kind, orders, 20_000 + s, actor, device,
                      args.num_uavs, args.max_steps,
                      replan_every=args.replan_every)
              for s in range(episodes)]
        a = np.array(rs, dtype=np.float64)
        out[kind] = a.mean(axis=0)
        print(f"{name:26s} {kind:8s} {a[:,0].mean():9.2f} {a[:,1].mean():9.3f} "
              f"{a[:,2].mean():6.3f} {a[:,3].mean():7.4f} {a[:,4].mean():6.3f} "
              f"{a[:,5].mean():7.3f} {a[:,6].mean():5.0f} {a[:,7].mean():5.2f}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--actor", default=None)
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--num-uavs", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--replan-every", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--skip-baselines", action="store_true")
    args = p.parse_args()

    actor = None
    arms = [] if args.skip_baselines else ["random", "greedy"]
    if args.actor and os.path.exists(args.actor):
        sd = torch.load(args.actor, map_location=args.device)
        hidden = int(sd["head.weight"].shape[1])
        actor = Actor(OBS_DIM, 9, hidden=hidden).to(args.device)
        actor.load_state_dict(sd)
        actor.eval()
        arms.append("policy")
        print(f"[loaded] {args.actor} (hidden={hidden})")
    else:
        print("[warn] 未提供 --actor，只跑 random / greedy 基线")

    print(f"\n{'case':26s} {'policy':8s} {'return':>9s} {'explored':>9s} {'disc':>6s} "
          f"{'burned':>7s} {'cov':>6s} {'rknown':>7s} {'ext':>5s} {'depl':>5s}")
    print("-" * 94)
    cases = dict(CASES)
    # 追加"全程交给内置规划器"这一档：最接近部署形态（高层每 25 步重规划一次）
    for name, orders in cases.items():
        table(name, arms, args.episodes, actor, args.device, args, orders)
        print("-" * 94)
    args.replan_every = 25
    table("planner (replan/25)", arms, args.episodes, actor, args.device, args, None)
    print("-" * 94)

    if actor is None:
        return
    # ---------------- 指令敏感性 ----------------
    print("\n指令敏感性 1：巡逻环半径（patrol x4 @fire，改 ring_radius）")
    args.replan_every = 0
    for rr in (4.0, 8.0, 14.0):
        def mk(env, rr=rr):
            from lowlevel.sanity_check import patrol_at_fires
            os_ = patrol_at_fires(env)
            for o in os_:
                o.ring_radius = rr
            return os_
        rs = [rollout("policy", mk, 30_000 + s, actor, args.device,
                      args.num_uavs, args.max_steps) for s in range(args.episodes)]
        a = np.array(rs, dtype=np.float64)
        print(f"  ring={rr:4.1f}  disc={a[:,2].mean():.3f} burned={a[:,3].mean():.4f} "
              f"ret={a[:,0].mean():8.2f}")

    print("\n指令敏感性 2：搜寻目标（explore x4，把目标指到地图不同角落）")
    for tgt in ((15.0, 15.0), (85.0, 85.0), (15.0, 85.0)):
        orders = [UavOrder(task=TASK_FIRE_SENSING, mode=SENSE_EXPLORE, target=tgt)] * 4
        rs = [rollout("policy", orders, 40_000 + s, actor, args.device,
                      args.num_uavs, args.max_steps) for s in range(args.episodes)]
        a = np.array(rs, dtype=np.float64)
        print(f"  target={tgt}  explored={a[:,1].mean():.3f} ret={a[:,0].mean():8.2f}")

    print("\n指令敏感性 3：中继指定不同人员（relay x4，改 rescuer_id）")
    for rid in (1, 4, 8):
        orders = [UavOrder(task=TASK_COMM_RELAY, rescuer_id=rid)] * 4
        rs = [rollout("policy", orders, 50_000 + s, actor, args.device,
                      args.num_uavs, args.max_steps) for s in range(args.episodes)]
        a = np.array(rs, dtype=np.float64)
        print(f"  rescuer_id={rid}  cov={a[:,4].mean():.3f} rknown={a[:,5].mean():.3f} "
              f"ret={a[:,0].mean():8.2f}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""部署兼容性检查：选中 Actor 通过 policy_api.LowLevelController 加载并实跑。"""
import argparse, json, math, os, sys
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _ROOT)

from lowlevel.policy_api import LowLevelController
from experiments.critic_advice.harness import BENCH, task_metrics
from lowlevel.env_wrapper import TaskConditionedEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor", required=True)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=999001)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ctl = LowLevelController(args.actor)
    sd = torch.load(args.actor, map_location="cpu")
    if isinstance(sd, dict) and "actor" in sd:
        sd = sd["actor"]
    print(f"[deploy_check] loaded {args.actor}: hidden={sd['head.weight'].shape[1]} "
          f"obs_dim_in={sd['base.net.1.weight'].shape[1]} "
          f"n_actions={sd['head.weight'].shape[0]}")
    env = TaskConditionedEnv(num_uavs=4, num_rescuers=10, n_fire_seeds=3,
                             grid_size=(100, 100), max_steps=600,
                             seed=args.seed, task_sampler="planner", replan_every=0)
    env.reset()
    # 合法任务指令（不经 LLM，直接 JSON 下发）
    plan = {"uavs": [
        {"task": "explore", "target": [75.0, 25.0]},
        {"task": "relay", "rescuer_id": 3, "handover": True},
        {"task": "charge"},
        {"task": "patrol", "target": None, "ring_radius": "auto"},
    ]}
    ctl.set_plan(plan, num_uavs=4)
    ok_steps = 0
    dists = []
    for t in range(args.steps):
        acts = ctl.act(env.env)          # 纯 Actor，无 waypoint_assist
        assert len(acts) == 4 and all(0 <= a <= 8 for a in acts)
        env.step(acts)
        ok_steps += 1
        dists.append(math.hypot(env.env.uavs[0].x - 75.0, env.env.uavs[0].y - 25.0))
    m = task_metrics(env)
    res = dict(actor=args.actor, steps=ok_steps,
               uav0_dist_to_explore_target=round(dists[-1], 1),
               uav0_dist_min=round(min(dists), 1),
               explored=round(m["explored"], 3), n_depleted=m["depleted"],
               actions_valid=True)
    print(json.dumps(res, indent=1))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
    assert ok_steps == args.steps


if __name__ == "__main__":
    main()

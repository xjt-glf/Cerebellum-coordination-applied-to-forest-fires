"""把一个回合里各奖励分项按任务累加起来，用于校准权重。"""
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lowlevel.env_wrapper import TaskConditionedEnv
from lowlevel.sanity_check import CASES, greedy_action
from lowlevel.task_spec import UavOrder, TASK_NAMES


def run(name, kind="greedy", seed=7000, max_steps=600):
    orders = CASES[name]
    env = TaskConditionedEnv(num_uavs=4, max_steps=max_steps, seed=seed,
                             task_sampler="mix", replan_every=0)
    if callable(orders):
        env.reset()
        obs = env.set_plan(orders(env))
    else:
        obs = env.reset(plan=[UavOrder(**vars(o)) for o in orders])
    rng = np.random.RandomState(seed)
    acc = collections.defaultdict(lambda: collections.defaultdict(float))
    tot = collections.defaultdict(float)
    for _ in range(max_steps):
        a = (rng.randint(0, 9, size=4) if kind == "random"
             else [greedy_action(obs[i], env.orders[i]) for i in range(4)])
        obs, r, done, info = env.step(a)
        for i, pt in enumerate(info["parts"]):
            t = TASK_NAMES[int(env.orders[i].task)]
            for k, v in pt.items():
                acc[t][k] += float(v)
            tot[t] += float(r[i])
        if done:
            break
    print(f"\n=== {name} / {kind} (seed {seed}) ===")
    for t in sorted(acc):
        items = sorted(acc[t].items(), key=lambda kv: -abs(kv[1]))
        print(f"  {t:13s} total={tot[t]:9.1f}  " +
              "  ".join(f"{k}={v:+.1f}" for k, v in items))


if __name__ == "__main__":
    for nm in ("explore2+relay1+charge1", "explore x4"):
        for kd in ("greedy", "random"):
            run(nm, kd)

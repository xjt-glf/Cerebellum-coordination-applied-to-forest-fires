# -*- coding: utf-8 -*-
"""Trace task exposure over one episode with the baseline actor under the new sampler."""
import math, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "experiments", "lowlevel_training"))
import numpy as np
import torch
torch.set_num_threads(1)

from experiments.lowlevel_training.goal_env import GoalEnv
from experiments.critic_advice.harness import load_actor

actor = load_actor()
env = GoalEnv(seed=1234, reward_version="P1", num_uavs=4, num_rescuers=10, n_fire_seeds=3)
obs = env.reset()
task_steps = {"E": 0, "P": 0, "R": 0, "C": 0}
arrival = {"E": 0, "P": 0, "R": 0, "C": 0}
clusters_first = None
for k in range(600):
    with torch.no_grad():
        a = actor.act(torch.as_tensor(obs, dtype=torch.float32), deterministic=True)[0].numpy()
    obs, r, done, info = env.step(a)
    for i, t in enumerate(info["tasks"]):
        key = "P" if (t == 0 and info["modes"][i] == 1) else ("E" if t == 0 else ("R" if t == 1 else "C"))
        task_steps[key] += 1
        ge = (info.get("goal_events") or {}).get(str(i)) or (info.get("goal_events") or {}).get(i) or {}
        if ge.get("g_arrived"):
            arrival[key] += 1
    if clusters_first is None and info["n_clusters"] > 0:
        clusters_first = k
    if done:
        break
tot = sum(task_steps.values())
print("task exposure:", {k: round(v / tot, 3) for k, v in task_steps.items()}, "clusters first seen at t =", clusters_first)
print("arrived-step fraction:", {k: round(v / max(1, task_steps[k]), 3) for k, v in arrival.items()})
print("final explored:", round(info["explored_rate"], 3), "n_clusters:", info["n_clusters"])

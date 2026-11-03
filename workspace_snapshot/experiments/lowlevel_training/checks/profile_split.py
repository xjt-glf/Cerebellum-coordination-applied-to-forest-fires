# -*- coding: utf-8 -*-
"""Isolate vec.step vs actor time."""
import os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "experiments", "lowlevel_training"))
import numpy as np
import torch
torch.set_num_threads(1)

from lowlevel.vec_env import SubprocVecEnv
from lowlevel.models import Actor
from experiments.lowlevel_training.goal_env import GoalEnv


def make_fn(seed):
    def _f():
        return GoalEnv(seed=seed, reward_version="P1", role_mode="randomized",
                       num_uavs=4, num_rescuers=10, n_fire_seeds=3)
    return _f


for device in ("cpu", "cuda:0"):
    vec = SubprocVecEnv([make_fn(5000 + i) for i in range(16)], envs_per_worker=2)
    actor = Actor(355, 9, hidden=384, n_layers=2).to(device)
    obs, glb, _ = vec.reset()
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    pre = np.random.randint(0, 9, size=(16, 4))
    for _ in range(20):
        vec.step(pre)
    t_env = t_act = 0.0
    n = 60
    for k in range(n):
        ta = time.time()
        with torch.no_grad():
            a, _, _ = actor.act(obs_t.reshape(-1, 355))
        a_np = a.view(16, 4).cpu().numpy()
        t_act += time.time() - ta
        ta = time.time()
        obs, rew, done, glb, infos = vec.step(a_np)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        t_env += time.time() - ta
    print(f"device={device}: act {1000*t_act/n:.1f}ms  vec.step {1000*t_env/n:.1f}ms "
          f"-> {16/t_env*n:.0f} sps")
    vec.close()

# -*- coding: utf-8 -*-
"""Profile: SubprocVecEnv throughput with the goal-conditioned env."""
import argparse, os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "experiments", "lowlevel_training"))
import numpy as np
import torch

from lowlevel.vec_env import SubprocVecEnv
from lowlevel.models import Actor
from experiments.lowlevel_training.goal_env import GoalEnv


def make_fn(seed):
    def _f():
        return GoalEnv(seed=seed, reward_version="P1", role_mode="randomized",
                       num_uavs=4, num_rescuers=10, n_fire_seeds=3)
    return _f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epw", type=int, default=2)
    ap.add_argument("--secs", type=float, default=45)
    args = ap.parse_args()
    torch.set_num_threads(1)
    n_envs = args.workers * args.epw
    vec = SubprocVecEnv([make_fn(5000 + i) for i in range(n_envs)],
                        envs_per_worker=args.epw)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    actor = Actor(355, 9, hidden=384, n_layers=2).to(device)
    opt = torch.optim.Adam(actor.parameters(), lr=1e-4)
    obs, glb, _ = vec.reset()
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    glb_t = torch.as_tensor(glb, dtype=torch.float32, device=device)
    steps = 0
    t0 = time.time()
    warm = 10
    times = []
    while time.time() - t0 < args.secs:
        ta = time.time()
        with torch.no_grad():
            a, lp, _ = actor.act(obs_t.reshape(-1, 355))
        a = a.view(n_envs, 4)
        obs, rew, done, glb, infos = vec.step(a.cpu().numpy())
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        glb_t = torch.as_tensor(glb, dtype=torch.float32, device=device)
        steps += n_envs
        if steps > warm * n_envs:
            times.append(time.time() - ta)
    dt = time.time() - t0
    sps = (steps - warm * n_envs) / (dt - 0.5)
    print(f"workers={args.workers} epw={args.epw} envs={n_envs}: "
          f"{sps:.0f} env-steps/s, mean step {1000*np.mean(times):.1f}ms "
          f"(p95 {1000*np.percentile(times,95):.1f}ms)")
    vec.close()


if __name__ == "__main__":
    main()

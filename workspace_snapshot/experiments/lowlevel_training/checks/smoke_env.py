# -*- coding: utf-8 -*-
"""Smoke test: env + baseline actor forward/step in container."""
import sys, os
sys.path.insert(0, "/ssd3/xiangjutao/hrl_dqn_improve")
sys.path.insert(0, "/ssd3/xiangjutao/hrl_dqn_improve/experiments/critic_advice")
import torch
import numpy as np

from harness import make_env, load_actor, BENCH

env = make_env(60000)
obs = env.reset()
print("obs shape:", obs.shape)
actor = load_actor()
o = torch.as_tensor(obs, dtype=torch.float32)
a = actor.act(o, deterministic=True)[0].numpy()
obs2, r, done, info = env.step(a.tolist())
print("step ok, rewards:", np.round(r, 3), "done:", done, "tasks:", info["tasks"])
print("explored:", round(info["explored_rate"], 3), "t:", info["t"])
print("obs range:", float(obs.min()), float(obs.max()))
print("action space sample:", a.tolist())
# check actor param count
n = sum(p.numel() for p in actor.parameters())
print("actor params:", n)

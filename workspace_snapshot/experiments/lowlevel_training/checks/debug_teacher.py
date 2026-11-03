# -*- coding: utf-8 -*-
"""Debug: trace teacher trajectory + reward parts under V2."""
import math, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "experiments", "lowlevel_training"))
import numpy as np
import torch
torch.set_num_threads(1)

from lowlevel.task_spec import UavOrder, TASK_FIRE_SENSING, SENSE_EXPLORE
from lowlevel.obs_builder import shared_cache
from experiments.lowlevel_training.goal_env import GoalEnv
from experiments.lowlevel_training.teacher import teacher_action

env = GoalEnv(seed=31, reward_version="P1", num_uavs=4, num_rescuers=10, n_fire_seeds=3)
env.reset()
e = env.env
c = shared_cache(e)
M = c["M"]
rs = np.random.RandomState(0)
tgt = None
for _ in range(300):
    x, y = rs.randint(10, 90, 2)
    reg = M[max(0, x - 5):x + 6, max(0, y - 5):y + 6]
    if (reg < 0).mean() >= 0.8 and 30 < math.hypot(x - e.uavs[0].x, y - e.uavs[0].y) < 55:
        tgt = (float(x), float(y))
        break
print("target:", tgt, "uav0 pos:", (e.uavs[0].x, e.uavs[0].y))
orders = list(env.orders)
orders[0] = UavOrder(task=TASK_FIRE_SENSING, mode=SENSE_EXPLORE, target=tgt)
env.set_plan(list(orders))
g0 = env.goal_mgr.goals[0]
print("goal:", g0.goal_key, "d_init:", g0.d_init, "deadline_t:", g0.deadline_t)

ret = 0.0
for k in range(80):
    cache = shared_cache(env.env)
    acts = [teacher_action(env.env, i, env.orders[i], cache) for i in range(4)]
    obs, r, done, info = env.step(acts)
    ret += float(r[0])
    p = info["parts"][0]
    u = env.env.uavs[0]
    ge = (info.get("goal_events") or {}).get(0, {})
    if k % 10 == 0 or k < 5:
        print(f"t={k} pos=({u.x:.0f},{u.y:.0f}) act={acts[0]} d={ge.get('d_now', -1):.1f} "
              f"r={r[0]:+.3f} parts={ {kk: round(vv,3) for kk,vv in p.items()} } "
              f"order0={env.orders[0].to_dict()}")
    if done:
        print("done at", k)
        break
g = env.goal_mgr.goals[0]
print("final goal:", g.goal_key, "arrived:", g.arrived, "completed:", g.completed, "ret:", round(ret, 2))

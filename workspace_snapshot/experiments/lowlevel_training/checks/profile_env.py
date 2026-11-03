# -*- coding: utf-8 -*-
"""Profile single GoalEnv step vs TaskConditionedEnv baseline."""
import cProfile, pstats, io, os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "experiments", "lowlevel_training"))
import numpy as np

from experiments.lowlevel_training.goal_env import GoalEnv

env = GoalEnv(seed=77, reward_version="P1", num_uavs=4, num_rescuers=10, n_fire_seeds=3)
env.reset()
# warmup
for _ in range(20):
    env.step(np.random.randint(0, 9, 4))

t0 = time.time()
N = 100
for _ in range(N):
    env.step(np.random.randint(0, 9, 4))
print(f"plain: {1000*(time.time()-t0)/N:.1f} ms/step")

pr = cProfile.Profile()
pr.enable()
for _ in range(N):
    env.step(np.random.randint(0, 9, 4))
pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(18)
print(s.getvalue())

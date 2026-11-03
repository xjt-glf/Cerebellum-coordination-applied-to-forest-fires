# -*- coding: utf-8 -*-
"""并行向量化环境（多进程）。

TaskConditionedEnv 单步约 12 ms，训练需要百万级步，必须并行。
这里实现一个最小可用的 SubprocVecEnv，不引入额外依赖。

**每个 worker 承载多个环境**（`envs_per_worker`）。原来一进程一环境时，
主进程每步都要等**最慢**的那个 worker：env.step 的耗时是重尾分布
（`低层环境.py` 里救援人员重分配会走 `bfs_path_avoid_fire`），64 个 worker
取 max 后单步高达 200+ ms，实测吞吐只有独立进程基准的 1/15。
让一个 worker 顺序跑 k 个环境，重尾在 k 个环境内被平均掉，单步耗时收敛到
k×均值，同时把管道往返次数降到 1/k。
"""
import multiprocessing as mp
import numpy as np


def _worker(remote, parent_remote, env_fns):
    parent_remote.close()
    envs = [fn() for fn in env_fns]
    keys = ("coverage_rate", "comm_coverage_rate", "explored_rate", "fire_discovery_rate",
            "burned_area_rate", "rescuer_known_fire_rate", "delivered_cells",
            "max_AoI", "n_depleted", "n_clusters", "n_contained", "t")
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step_full":
                O, R, D, G, I = [], [], [], [], []
                for env, act in zip(envs, data):
                    obs, rew, done, info = env.step(act)
                    payload = {k: info[k] for k in keys}
                    payload["tasks"] = list(info["tasks"])
                    # 兼容扩展：目标条件训练（experiments/lowlevel_training）的
                    # 逐步目标事件与模式。旧奖励 Trial 没有这些键，缺省为空。
                    payload["modes"] = list(info.get("modes", []))
                    payload["goal_events"] = info.get("goal_events", {})
                    if done:
                        obs = env.reset()
                    O.append(obs); R.append(rew); D.append(done)
                    G.append(env.global_features()); I.append(payload)
                remote.send((np.stack(O), np.stack(R),
                             np.array(D, dtype=bool), np.stack(G), I))
            elif cmd == "reset":
                O = [env.reset(*(data or ())) for env in envs]
                remote.send((np.stack(O),
                             np.stack([env.global_features() for env in envs]),
                             np.array([list(env.taus) for env in envs])))
            elif cmd == "set_plan":
                for env in envs:
                    env.set_plan(data)
                remote.send(True)
            elif cmd == "spec":
                e = envs[0]
                remote.send((e.obs_dim, e.n_actions, e.num_uavs,
                             e.GLOBAL_DIM, e.max_steps))
            elif cmd == "close":
                for env in envs:
                    env.close()
                remote.close()
                break
    except (KeyboardInterrupt, EOFError):
        pass


class SubprocVecEnv:
    def __init__(self, env_fns, start_method=None, envs_per_worker=1):
        self.n_envs = len(env_fns)
        k = max(1, int(envs_per_worker))
        groups = [env_fns[i:i + k] for i in range(0, self.n_envs, k)]
        self.groups = [len(g) for g in groups]
        self.bounds = np.cumsum([0] + self.groups)
        self.n_workers = len(groups)
        ctx = mp.get_context(start_method or ("fork" if hasattr(mp, "get_all_start_methods")
                                              and "fork" in mp.get_all_start_methods()
                                              else "spawn"))
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.n_workers)])
        self.ps = []
        for wr, r, fns in zip(self.work_remotes, self.remotes, groups):
            p = ctx.Process(target=_worker, args=(wr, r, fns), daemon=True)
            p.start()
            wr.close()
            self.ps.append(p)
        self.remotes[0].send(("spec", None))
        (self.obs_dim, self.n_actions, self.num_uavs,
         self.global_dim, self.max_steps) = self.remotes[0].recv()

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        outs = [r.recv() for r in self.remotes]
        obs = np.concatenate([o[0] for o in outs])    # [E, N, D]
        glb = np.concatenate([o[1] for o in outs])    # [E, G]
        taus = np.concatenate([o[2] for o in outs])   # [E, N]
        return obs, glb, taus

    def step(self, actions):
        """actions: [E, N] -> obs [E,N,D], rew [E,N], done [E], glb [E,G], infos"""
        for i, r in enumerate(self.remotes):
            chunk = actions[self.bounds[i]:self.bounds[i + 1]]
            r.send(("step_full", [list(a) for a in chunk]))
        outs = [r.recv() for r in self.remotes]
        obs = np.concatenate([o[0] for o in outs])
        rew = np.concatenate([o[1] for o in outs])
        done = np.concatenate([o[2] for o in outs])
        glb = np.concatenate([o[3] for o in outs])
        infos = [inf for o in outs for inf in o[4]]
        return obs, rew, done, glb, infos

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except Exception:
                pass
        for p in self.ps:
            p.join(timeout=2)


class DummyVecEnv:
    """单进程版本，便于调试与本机小规模验证。"""

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.n_envs = len(self.envs)
        e0 = self.envs[0]
        self.obs_dim, self.n_actions = e0.obs_dim, e0.n_actions
        self.num_uavs, self.global_dim, self.max_steps = (
            e0.num_uavs, e0.GLOBAL_DIM, e0.max_steps)

    def reset(self):
        obs = np.stack([e.reset() for e in self.envs])
        glb = np.stack([e.global_features() for e in self.envs])
        taus = np.array([e.taus for e in self.envs])
        return obs, glb, taus

    def step(self, actions):
        O, R, D, G, I = [], [], [], [], []
        for e, a in zip(self.envs, actions):
            o, r, d, info = e.step(list(a))
            payload = {k: info[k] for k in
                       ("coverage_rate", "comm_coverage_rate", "explored_rate", "fire_discovery_rate",
                        "burned_area_rate", "rescuer_known_fire_rate",
                        "delivered_cells", "n_extinguished",
            "max_AoI", "n_depleted", "n_clusters", "n_contained", "t")}
            payload["tasks"] = list(info["tasks"])
            payload["modes"] = list(info.get("modes", []))
            payload["goal_events"] = info.get("goal_events", {})
            if d:
                o = e.reset()
            O.append(o); R.append(r); D.append(d)
            G.append(e.global_features()); I.append(payload)
        return (np.stack(O), np.stack(R), np.array(D, dtype=bool),
                np.stack(G), I)

    def close(self):
        for e in self.envs:
            e.close()

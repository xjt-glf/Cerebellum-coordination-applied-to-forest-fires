"""Isolated episode adapter, CPU simulation and deterministic frozen movement Actor."""
import collections
import copy
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from experiments.feedback_attribution.joint_runtime import JointRuntime, load_actor
from lowlevel.obs_builder import shared_cache, resolve_order_target
from lowlevel.task_spec import UavOrder
from .protocol import VERSION, digest, resolve_target

ROOT = Path(__file__).resolve().parents[2]
LOWLEVEL = ROOT / "research_outputs/lowlevel_training_final/actor_selected.pt"
EXPECTED = "dfa5cdd0f4b673567c04220af9d82ee55f16920d55398a02471a0d535d59982e"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_actor():
    assert file_hash(LOWLEVEL) == EXPECTED, "Frozen lowlevel Actor changed"
    return load_actor(str(LOWLEVEL))


def avg(xs):
    return float(np.mean(xs)) if len(xs) else 0.0


class DistributedAllocatorEnv:
    def __init__(self, seed, n, actor=None, reward_version="v1_count", trajectory=None, num_rescuers=10):
        self.seed, self.n, self.reward_version = seed, n, reward_version
        self.rt = JointRuntime(seed, cfg={"uavs": n, "rescuers": int(num_rescuers)}, actor=actor or frozen_actor())
        self.w, self.e = self.rt.env, self.rt.env.env
        self.rt.charging.goals = None  # Independent, versioned goal protocol below.
        self.w.set_plan([UavOrder() for _ in range(n)])
        self.lowobs = self.w._obs()
        self.ages = [0] * n
        self.history = [collections.deque(maxlen=11) for _ in range(n)]
        self.protected = [False] * n
        self.depleted = set()
        self.covs, self.ucovs, self.aois, self.lost = [], [], [], np.zeros(len(self.e.rescuers), int)
        self.max_lost, self.exposure, self.overrides, self.switches = 0, 0, 0, 0
        self.components = collections.Counter()
        self.action_counts = np.zeros(3, int)
        self.last_delta, self.window_cov, self.window_protect = 0., 0., 0.
        self.timing = collections.Counter()
        self.message_bytes = 0
        self.started = time.monotonic()
        self.trace = gzip.open(trajectory, "wt", encoding="utf-8") if trajectory else None

    def valid(self, i):
        o, u = self.w.orders[i], self.e.uavs[i]
        if o.task == 0:
            if o.target is None or self.ages[i] >= 100:
                return False
            x, y = o.target
            if max(abs(x-u.x), abs(y-u.y)) <= 2:
                return False
            M = shared_cache(self.e)["M"]
            a, b = min(4, int(x / (M.shape[0]/5))), min(4, int(y / (M.shape[1]/5)))
            return bool((np.array_split(np.array_split(M, 5, axis=0)[a], 5, axis=1)[b] < 0).any())
        if o.task == 1:
            return o.rescuer_id in [int(r.rescuer_id) for r in self.e.rescuers]
        return True

    def snapshot(self):
        e, w = self.e, self.w
        M = shared_cache(e)["M"]
        regions = []
        for a, row in enumerate(np.array_split(M, 5, axis=0)):
            for b, block in enumerate(np.array_split(row, 5, axis=1)):
                regions.append(dict(id=a*5+b, center=[(a+.5)*M.shape[0]/5, (b+.5)*M.shape[1]/5],
                                    unknown=int((block < 0).sum()), fire=int(((block > 0)&(block < 2)).sum())))
        return dict(version=VERSION, t=w.t, seed=self.seed,
                    uavs=[dict(id=i, pos=[float(u.x), float(u.y)], energy=float(u.e),
                               task=int(w.orders[i].task), order=w.orders[i].to_dict(), valid=self.valid(i))
                          for i, u in enumerate(e.uavs)],
                    persons=[dict(id=int(r.rescuer_id), pos=[float(r.x), float(r.y)],
                                  covered=bool(r.is_covered), aoi=float(r.AoI)) for r in e.rescuers],
                    regions=regions)

    def observations(self):
        e, w, c = self.e, self.w, shared_cache(self.e)
        H, W = e.grid_size
        vectors = []
        for i, (u, o) in enumerate(zip(e.uavs, w.orders)):
            st = self.rt.charging.targets.get(i)
            if st is None:
                st = min(e.charging_stations, key=lambda st: self.rt.charging.steps_to_pad(u, st))
            steps = self.rt.charging.steps_to_pad(u, st)
            target = resolve_order_target(e, i, o, c)[0]
            em = float(getattr(u, "e_max", getattr(e, "uav_energy", 600))) or 600
            hist = self.history[i]
            stalled = len(hist) == 11 and len(set(hist)) == 1 and not e.at_station(u.x, u.y)
            vectors.append([u.x/H, u.y/W, u.e/em, np.clip((u.e-e.energy_cost_per_step(i)*steps)/em,-1,1),
                            steps/max(H,W), *[float(o.task == a) for a in range(3)],
                            (target[0]-u.x)/H if target else 0, (target[1]-u.y)/W if target else 0,
                            float(target is not None), min(self.ages[i]/100,1), float(not self.valid(i)),
                            float(self.protected[i]), float(stalled), float(u.e > 0 or e.at_station(u.x,u.y))])
        us = np.asarray(vectors, np.float32).reshape(-1,16)
        ps = np.asarray([[r.x/H, r.y/W, float(r.is_covered), min(float(r.AoI)/100,1),
                          max(abs(r.x-e.base_pos[0]),abs(r.y-e.base_pos[1]))/max(H,W),
                          min((max(abs(r.x-u.x),abs(r.y-u.y)) for u in e.uavs),default=0)/max(H,W),
                          0., 1.] for r in e.rescuers], np.float32).reshape(-1,8)
        M = c["M"]
        tiles = [b for row in np.array_split(M,5,axis=0) for b in np.array_split(row,5,axis=1)]
        maps = [float(f(b).mean()) for b in tiles for f in (lambda b:b<0, lambda b:(b>0)&(b<2))]
        cov = avg([r.is_covered for r in e.rescuers])
        energies = us[:,2]
        tasks = [o.task for o in w.orders]
        summary = [w.t/600, 1-w.t/600, float((M>=0).mean()), float(e.coverage_rate), cov,
                   avg([r.AoI for r in e.rescuers])/100,
                   max((r.AoI for r in e.rescuers),default=0)/100,
                   1-cov if e.rescuers else 0, len(c["clusters"])/25,
                   float(((M>0)&(M<2)).mean()), float((M<0).mean()),
                   *[tasks.count(a)/self.n for a in range(3)], float(energies.min()),
                   float(energies.mean()), self.last_delta, self.window_cov,
                   1-self.window_cov if w.t else 0, self.window_protect,
                   self.n/16, len(e.rescuers)/32]
        context = np.clip(np.asarray(summary+maps,np.float32),-1,1)
        assert context.shape == (72,)
        base = dict(persons=ps, person_mask=np.ones(len(ps),np.float32), context=context)
        local = [dict(base, self=us[i], mates=np.delete(us,i,axis=0),
                      mate_mask=np.ones(self.n-1,np.float32)) for i in range(self.n)]
        team = dict(base, mates=us, mate_mask=np.ones(self.n,np.float32))
        return local, team

    def step(self, actions, metadata=None):
        w, e, rt = self.w, self.e, self.rt
        snap = self.snapshot()
        t0 = w.t
        old_tasks = [o.task for o in w.orders]
        active = [u.e > 0 or e.at_station(u.x,u.y) for u in e.uavs]
        switches = sum(a != b and live for a,b,live in zip(actions,old_tasks,active)) if t0 else 0
        self.switches += switches
        started = time.perf_counter()
        messages = {i: int(a) for i,a in enumerate(actions)}
        plans = [resolve_target(copy.deepcopy(snap), messages, i) for i in range(self.n)]
        self.timing["targets"] += time.perf_counter()-started
        self.message_bytes += len(json.dumps(snap).encode()) + len(json.dumps(messages).encode())
        orders = [UavOrder.from_llm_json(p) for p in plans]
        for i,o in enumerate(orders):
            if o.to_dict() != w.orders[i].to_dict():
                self.ages[i] = 0
        self.lowobs = w.set_plan(orders)
        comp = collections.Counter(switch=-.02*switches/(max(1,sum(active)) if self.reward_version=="v2_normalized" else 1))
        steps = []
        known0 = int((shared_cache(e)["M"]>=0).sum())
        window_c, window_o = [], 0
        for _ in range(min(20, 600-w.t)):
            B = int((e.grid["burning_state"]>0).sum())
            known = int((shared_cache(e)["M"]>=0).sum())
            before = [o.task for o in w.orders]
            started = time.perf_counter()
            ob, notes = rt.charging.sync()
            self.timing["protection"] += time.perf_counter()-started
            if ob is not None:
                self.lowobs = ob
            self.protected = [a != o.task for a,o in zip(before,w.orders)]
            overrides = sum(self.protected)
            self.overrides += overrides
            window_o += overrides
            started = time.perf_counter()
            with torch.no_grad():
                moves = rt.actor.act(torch.as_tensor(self.lowobs), deterministic=True)[0].numpy()
            self.timing["lowlevel_actor"] += time.perf_counter()-started
            actual = [o.to_dict() for o in w.orders]
            self.lowobs, _, done, info = w.step(moves)
            rt.goals.current = copy.deepcopy(w.orders)
            rt.goals.update()
            for i,u in enumerate(e.uavs):
                self.ages[i] += 1
                self.history[i].append((float(u.x),float(u.y)))
            stranded = {i for i,u in enumerate(e.uavs) if u.e<=0 and not e.at_station(u.x,u.y)}
            delta_d = len(stranded-self.depleted)
            self.depleted |= stranded
            L = int(e._active_true_burning_mask().sum())
            C = avg([r.is_covered for r in e.rescuers])
            X = max(0,int((shared_cache(e)["M"]>=0).sum())-known)
            parts = dict(burn=-.020*max(0,int((e.grid["burning_state"]>0).sum())-B),
                         exposure=-.002*L/100, uncovered=-(.100 if self.reward_version == 'v3_coverage' else .035 if self.reward_version == 'v4_balanced' else .010)*(1-C),
                         explore=(.020 if self.reward_version == 'v4_balanced' else .005)*X/100,
                         depletion=-5.*delta_d/(max(1,sum(active)) if self.reward_version=="v2_normalized" else 1))
            comp.update(parts)
            self.exposure += L
            self.covs.append(C)
            window_c.append(C)
            self.ucovs.append(float(e.coverage_rate))
            self.aois.extend(float(r.AoI) for r in e.rescuers)
            self.lost = np.where([not r.is_covered for r in e.rescuers],self.lost+1,0)
            self.max_lost = max(self.max_lost,int(self.lost.max(initial=0)))
            steps.append(dict(t=w.t, actual=actual, notes=notes, station_targets=rt.charging.targets,
                              reward=parts, L=L, C=C, X=X, new_depleted=delta_d,
                              pos=[[float(u.x),float(u.y),float(u.e)] for u in e.uavs]))
        self.last_delta = (int((shared_cache(e)["M"]>=0).sum())-known0)/e.grid["burning_state"].size
        self.window_cov, self.window_protect = avg(window_c), window_o/(self.n*len(steps))
        self.components.update(comp)
        self.action_counts += np.bincount(actions,minlength=3)
        terminated = w.t == 600
        if self.trace:
            rec = dict(scenario_seed=self.seed, num_uavs=self.n, t=t0, obs_version="obs_v2",
                       protocol_version=VERSION, public_snapshot_hash=digest(snap),
                       broadcast=snap, actions=messages, requested_plans=plans, goal_applied_plans=plans,
                       lowlevel_actor_hash=EXPECTED, physical_steps=steps, reward_components=dict(comp),
                       terminated=terminated, truncated=False, **(metadata or {}))
            self.trace.write(json.dumps(rec,ensure_ascii=False)+"\n")
            self.trace.flush()
        return sum(comp.values()), terminated

    def metrics(self):
        bs = self.e.grid["burning_state"]
        return dict(seed=self.seed,n=self.n,t=self.w.t,terminated=self.w.t==600,truncated=False,
                    burned_cells=int((bs>0).sum()), burning_cells=int(((bs>0)&(bs<2)).sum()),
                    burned_out_cells=int((bs==2).sum()), extinguished_cells=int((bs==3).sum()),
                    uncontained_exposure=self.exposure, explored=float((shared_cache(self.e)["M"]>=0).mean()),
                    cov_mean=avg(self.covs),uav_cov_mean=avg(self.ucovs),
                    aoi_p95=float(np.percentile(self.aois,95)) if self.aois else 0,
                    aoi_mean=avg(self.aois),longest_disconnect=self.max_lost,
                    depleted=len(self.depleted), switches=self.switches,
                    override_per_100_uav_steps=100*self.overrides/max(1,self.n*self.w.t),
                    reward_components=dict(self.components),action_counts=self.action_counts.tolist(),
                    message_bytes=self.message_bytes,broadcast_rounds=2*(self.w.t//20),
                    seconds=time.monotonic()-self.started,timing=dict(self.timing))

    def close(self):
        if self.trace:
            self.trace.close()

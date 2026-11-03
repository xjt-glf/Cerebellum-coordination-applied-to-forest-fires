"""Additional execution contracts and deployment latency evidence."""
import copy,inspect,json,time
import numpy as np
import torch
from .environment import DistributedAllocatorEnv,ROOT,frozen_actor,file_hash,EXPECTED
from .model import Actor,Critic,stack_obs,tensor_obs
from .protocol import resolve_all,resolve_target,VERSION
from experiments.feedback_attribution.joint_runtime import JointRuntime
from lowlevel.task_spec import UavOrder

def main():
    torch.set_num_threads(1);low=frozen_actor();actor=Actor().eval();critic=Critic().eval()
    out=ROOT/'research_outputs/local_allocator_mappo_20261103'
    checks={}
    for n in (1,3,4,6,8):
        env=DistributedAllocatorEnv(979200+n,n,low);local,team=env.observations()
        for o in local: assert actor.act(o) in (0,1,2)
        # Batch mixed fleet widths and empty people sets retain exact set semantics.
        empty=copy.deepcopy(local[0]);empty['persons']=np.zeros((0,8),np.float32);empty['person_mask']=np.zeros(0,np.float32)
        batch=stack_obs([local[0],empty])
        assert torch.isfinite(actor(tensor_obs(batch))).all()
        assert torch.isfinite(critic(tensor_obs(stack_obs([team])))).all()
        snap=env.snapshot();actions={i:0 for i in range(n)};plans=resolve_all(snap,actions)
        # Retention must reserve higher-ID old targets before lower-ID new targets.
        snap['uavs'][-1]['order']=dict(task='explore',target=plans[n-1]['target'])
        snap['uavs'][-1]['task']=0;snap['uavs'][-1]['valid']=True
        assert resolve_all(snap,actions)[n-1]['target']==plans[n-1]['target']
        bad=dict(actions);bad.pop(0)
        try: resolve_all(snap,bad)
        except ValueError: pass
        else: raise AssertionError('Missing broadcast accepted')
        snap['persons']=[]
        assert all(p['task']=='relay' and 'rescuer_id' not in p for p in resolve_all(snap,{i:1 for i in range(n)}).values())
        # Common protection must force a low-energy, off-pad vehicle to recharge.
        u=env.e.uavs[0];u.x,u.y,u.e=50.,50.,1.
        env.w.set_plan([UavOrder() for _ in range(n)])
        ob,notes=env.rt.charging.sync()
        assert env.w.orders[0].task==2 and ob is not None
        checks[n]=dict(independent_actor=True,empty_people=True,missing_broadcast_rejected=True,retention=True,protection_override=True)
        env.close()
    # Compare independent physical-step adapter against original runtime for fixed plan.
    env=DistributedAllocatorEnv(979250,4,low)
    snap=env.snapshot();acts={0:0,1:0,2:1,3:2};plans=resolve_all(snap,acts)
    env.step([0,0,1,2]);expected=[(u.x,u.y,u.e) for u in env.e.uavs];truth=env.e.grid['burning_state'].copy()
    rt=JointRuntime(979250,cfg={'uavs':4},actor=low)
    rt.apply_joint_plan(dict(uavs=[plans[i] for i in range(4)],source=VERSION));rt.run_to(20)
    checks['runtime_parity']=dict(uavs=np.allclose(expected,[(u.x,u.y,u.e) for u in rt.env.env.uavs]),fire=np.array_equal(truth,rt.env.env.grid['burning_state']))
    assert all(checks['runtime_parity'].values())
    # Measure local inference and each vehicle independently resolving whole plan.
    local,_=env.observations();snap=env.snapshot();acts={i:i%3 for i in range(4)}
    for _ in range(100): actor.act(local[0]);resolve_target(snap,acts,0)
    times={'actor_local_cpu_ms':[],'targets_local_cpu_ms':[],'actor_batched_cpu_ms':[]}
    for _ in range(500):
        t=time.perf_counter();actor.act(local[0]);times['actor_local_cpu_ms'].append((time.perf_counter()-t)*1000)
        t=time.perf_counter();resolve_target(snap,acts,0);times['targets_local_cpu_ms'].append((time.perf_counter()-t)*1000)
        t=time.perf_counter()
        with torch.no_grad(): actor(tensor_obs(stack_obs(local)))
        times['actor_batched_cpu_ms'].append((time.perf_counter()-t)*1000)
    latency={k:dict(zip(('p50','p95','p99'),np.percentile(v,[50,95,99]).tolist())) for k,v in times.items()}
    (out/'extra_contracts.json').write_text(json.dumps(dict(checks=checks,passed=True,latency=latency,measurement=dict(device='CPU',threads=1,warmup=100,repeats=500,fleet=4,network_latency='not simulated; no physical link timing')),indent=2))
    modules=['experiments.feedback_attribution.joint_runtime','lowlevel.env_wrapper','lowlevel.obs_builder','lowlevel.models','webapp.goal_protocol','webapp.charge_protocol']
    import importlib
    paths={m:str(Path(importlib.import_module(m).__file__).resolve()) for m in modules}
    paths['terrain']=str(ROOT/'terrain_height.csv');paths['simulation']=inspect.getfile(type(env.e))
    (out/'runtime_source_provenance.json').write_text(json.dumps({k:dict(path=v,sha256=file_hash(v)) for k,v in paths.items()},indent=2))
    print(json.dumps(dict(passed=True,checks=checks,latency=latency)),flush=True)

from pathlib import Path
if __name__=='__main__': main()

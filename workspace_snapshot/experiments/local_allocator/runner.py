"""CPU whole episode worker; no per-step GPU IPC."""
import random
import time
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .environment import DistributedAllocatorEnv, frozen_actor
from .model import Actor, Critic, tensor_obs, stack_obs
from .protocol import baseline
_LOW=None

def init_worker():
    global _LOW
    torch.set_num_threads(1)
    _LOW=frozen_actor()

def append_json(path, obj):
    with open(path,'a') as f: f.write(json.dumps(obj,ensure_ascii=False)+'\n')

def episode(job):
    if _LOW is None: init_worker()
    seed,n=job['seed'],job['n']
    ts=job.get('train_seed',0)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    actor,critic=Actor().eval(),Critic().eval()
    if job.get('actor') is not None: actor.load_state_dict(job['actor'])
    if job.get('critic') is not None: critic.load_state_dict(job['critic'])
    trajectory=job.get('trajectory')
    if trajectory: Path(trajectory).parent.mkdir(parents=True,exist_ok=True)
    env=DistributedAllocatorEnv(seed,n,_LOW,job.get('reward_version','v1_count'),trajectory)
    transitions=[]
    ah=hashlib.sha256(b''.join(x.numpy().tobytes() for x in actor.state_dict().values())).hexdigest()
    try:
        for k in range(30):
            local,team=env.observations()
            with torch.no_grad():
                forward_start=time.perf_counter()
                logits=actor(tensor_obs(stack_obs(local)))
                actor_seconds=time.perf_counter()-forward_start
                forward_start=time.perf_counter()
                value=float(critic(tensor_obs(stack_obs([team])))[0])
                critic_seconds=time.perf_counter()-forward_start
                probs=logits.softmax(-1).numpy()
            kind=job.get('kind','mappo')
            if kind!='mappo': actions=baseline(env.snapshot(),kind,job.get('threshold',260),job.get('relay_mode','one'),ts)
            elif job.get('deterministic'): actions=logits.argmax(-1).tolist()
            else:
                actions=[int(np.random.default_rng(np.random.SeedSequence([ts,seed,k*20,i])).choice(3,p=p.astype(float)/p.sum(dtype=float))) for i,p in enumerate(probs)]
            oldlog=np.log(probs[np.arange(n),actions]+1e-30)
            reward,done=env.step(actions,dict(train_seed=ts,run_id=job.get('run_id',kind),actor_hash=ah,task_logits=logits.tolist(),probs=probs.tolist(),actor_batch_forward_seconds=actor_seconds,critic_team_forward_seconds=critic_seconds,own_obs_hashes=[hashlib.sha256(b''.join(v.tobytes() for v in o.values())).hexdigest() for o in local]))
            transitions.append(dict(local=local,team=team,actions=actions,logp=oldlog,active=[o['self'][-1] for o in local],value=value,reward=reward,done=done))
        assert done and env.w.t==600
        metrics=dict(env.metrics(),kind=kind,train_seed=ts,actor_hash=ah,threshold=job.get('threshold',260),relay_mode=job.get('relay_mode','one'))
        advantage,nextvalue=0.,0.
        for tr in reversed(transitions):
            delta=tr['reward']+.97*nextvalue-tr['value']
            advantage=delta+.97*.95*advantage
            tr['advantage'],tr['return']=advantage,advantage+tr['value']
            nextvalue=tr['value']
        return (transitions if job.get('training') else []),metrics
    finally: env.close()

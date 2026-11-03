"""Contract, mask, leakage, independent local target and real terminal checks."""
import concurrent.futures as cf
import multiprocessing as mp
import copy,json,time
import numpy as np
import torch
from .environment import DistributedAllocatorEnv,frozen_actor,file_hash,LOWLEVEL,EXPECTED,ROOT
from .model import Actor,stack_obs,tensor_obs
from .protocol import resolve_all

def target_process(args): return resolve_all(*args)

def main():
    torch.set_num_threads(1);actor=Actor().eval();low=frozen_actor();results=[]
    for n in (1,3,4,5,6,7,8):
        start=time.monotonic();env=DistributedAllocatorEnv(979000+n,n,low)
        local,team=env.observations();logits=actor(tensor_obs(stack_obs(local))).detach()
        assert logits.shape==(n,3)
        padded=stack_obs(local);expected=actor(tensor_obs(padded)).detach()
        for entity,mask in (('mates','mate_mask'),('persons','person_mask')):
            perm=np.arange(padded[entity].shape[1])[::-1]
            padded[entity]=padded[entity][:,perm].copy();padded[mask]=padded[mask][:,perm].copy()
            padded[entity]=np.concatenate((padded[entity],np.ones_like(padded[entity][:,:1])*999),axis=1)
            padded[mask]=np.concatenate((padded[mask],np.zeros_like(padded[mask][:,:1])),axis=1)
        assert torch.allclose(expected,actor(tensor_obs(padded)),atol=1e-6)
        snap=env.snapshot();acts={i:0 for i in range(n)};plans=resolve_all(snap,acts)
        assert len({tuple(o['target']) for o in plans.values()})==n
        relay=resolve_all(snap,{i:1 for i in range(n)})
        assert len({o['rescuer_id'] for o in relay.values()})==min(n,len(snap['persons']))
        rev=copy.deepcopy(snap);rev['uavs'].reverse();rev['persons'].reverse()
        assert plans==resolve_all(rev,dict(reversed(list(acts.items()))))
        with cf.ProcessPoolExecutor(2,mp_context=mp.get_context('spawn')) as pool:
            assert all(p==plans for p in pool.map(target_process,[(snap,acts),(rev,acts)]))
        saved=env.e.grid['burning_state'].copy();env.e.grid['burning_state'][...]=3
        env.e.fire_discovery_rate=.123;env.e.burned_area_rate=.987
        after,_=env.observations()
        assert all(np.array_equal(a[k],b[k]) for a,b in zip(local,after) for k in a)
        assert plans==resolve_all(env.snapshot(),acts)
        assert torch.equal(logits,actor(tensor_obs(stack_obs(after))).detach())
        env.e.grid['burning_state'][...]=saved
        env.e.uavs[0].e-=10;changed,_=env.observations()
        assert not np.array_equal(changed[0]['self'],local[0]['self'])
        r,done=env.step([i%3 for i in range(n)])
        assert env.w.t==20 and not done and np.isfinite(r)
        results.append(dict(n=n,seconds=time.monotonic()-start,passed=True));env.close()
        print(json.dumps(results[-1]),flush=True)
    hashes=[]
    for seed in (979050,979050,979051):
        env=DistributedAllocatorEnv(seed,3,low);initial=env.e.grid['burning_state'].tobytes()
        for k in range(30):
            _,done=env.step([0,1,2]);assert done==(k==29)
        hashes.append((initial,json.dumps(env.metrics()['reward_components'],sort_keys=True),env.e.grid['burning_state'].tobytes()));env.close()
    assert hashes[0]==hashes[1] and hashes[0]!=hashes[2]
    assert file_hash(LOWLEVEL)==EXPECTED
    (ROOT/'research_outputs/local_allocator_mappo_20261103/contracts.json').write_text(json.dumps(dict(passed=True,smoke=results,deterministic_replay=True,true_terminal=True,leakage=True,multiprocess_targets=True,padding=True,lowlevel_hash=EXPECTED),indent=2))
    print('ALL_CONTRACTS_PASSED',flush=True)

if __name__=='__main__': main()

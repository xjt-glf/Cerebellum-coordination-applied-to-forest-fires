"""GPU MAPPO updates; CPU whole episode rollouts, team critic once per time."""
import argparse
import concurrent.futures as cf
import multiprocessing as mp
import json,os,random,time
from pathlib import Path
import numpy as np
import torch
from .environment import ROOT,LOWLEVEL,EXPECTED,file_hash
from .model import Actor,Critic,stack_obs,tensor_obs
from .runner import episode,init_worker,append_json
OUT=ROOT/'research_outputs/local_allocator_mappo_20261103'

def cpu_state(model): return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def update(actor,critic,ao,co,transitions,device):
    local,actions,logp,advantages=[],[],[],[]
    for tr in transitions:
        for i,o in enumerate(tr['local']):
            if tr['active'][i]:
                local.append(o);actions.append(tr['actions'][i]);logp.append(tr['logp'][i]);advantages.append(tr['advantage'])
    obs=tensor_obs(stack_obs(local),device)
    a=torch.tensor(actions,device=device)
    lp=torch.tensor(np.array(logp),device=device,dtype=torch.float32)
    adv=torch.tensor(advantages,device=device,dtype=torch.float32)
    adv=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-6)
    team=tensor_obs(stack_obs([t['team'] for t in transitions]),device)
    returns=torch.tensor([t['return'] for t in transitions],device=device,dtype=torch.float32)
    logs=[]
    for epoch in range(4):
        for idx in torch.randperm(len(a),device=device).split(256):
            dist=torch.distributions.Categorical(logits=actor({k:v[idx] for k,v in obs.items()}))
            ratio=(dist.log_prob(a[idx])-lp[idx]).exp()
            ent=dist.entropy().mean()
            loss=-torch.minimum(ratio*adv[idx],ratio.clamp(.8,1.2)*adv[idx]).mean()-.01*ent
            ao.zero_grad();loss.backward()
            gn=torch.nn.utils.clip_grad_norm_(actor.parameters(),.5);ao.step()
            logs.append([float(loss.detach()),float(ent.detach()),float(gn),float(((ratio-1).abs()>.2).float().mean().detach())])
        for idx in torch.randperm(len(transitions),device=device).split(256):
            vl=.5*(critic({k:v[idx] for k,v in team.items()})-returns[idx]).square().mean()
            co.zero_grad();vl.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),.5);co.step()
    return dict(policy_loss=float(np.mean(logs,axis=0)[0]),entropy=float(np.mean(logs,axis=0)[1]),actor_grad_norm=float(np.mean(logs,axis=0)[2]),clip_fraction=float(np.mean(logs,axis=0)[3]),value_loss=float(vl.detach()),team_transitions=len(transitions),agent_samples=len(a))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,required=True);p.add_argument('--index',type=int,required=True)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--batch-episodes',type=int,default=16)
    p.add_argument('--episodes',type=int,default=2000);p.add_argument('--seconds',type=int,default=5100)
    p.add_argument('--device',default='cuda:0');p.add_argument('--version',default='v1_count');p.add_argument('--resume')
    args=p.parse_args();torch.set_num_threads(1)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    run=OUT/f'{args.version}_s{args.seed}';run.mkdir(parents=True,exist_ok=True)
    (run/'config.json').write_text(json.dumps(dict(vars(args),gamma=.97,gae_lambda=.95,epochs=4,clip=.2,lr=.0003,entropy=.01,normalization='obs_v2 fixed physical scales; no fitted normalization',resume_boundary='complete batches only',lowlevel_hash=EXPECTED,scenario_seeds=list(range(980000+args.index*2000,982000+args.index*2000))),indent=2))
    actor,critic=Actor().to(args.device),Critic().to(args.device)
    ao,co=torch.optim.Adam(actor.parameters(),lr=3e-4),torch.optim.Adam(critic.parameters(),lr=3e-4)
    completed,updates=0,0
    if args.resume:
        ck=torch.load(args.resume,map_location=args.device,weights_only=False)
        actor.load_state_dict(ck['actor']);critic.load_state_dict(ck['critic'])
        ao.load_state_dict(ck['actor_optimizer']);co.load_state_dict(ck['critic_optimizer'])
        completed,updates=ck['episodes'],ck['updates']
        random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);torch.set_rng_state(ck['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in ck['cuda_rng']])
    (run/'pid.json').write_text(json.dumps(dict(pid=os.getpid(),started=time.time())))
    start=time.monotonic()
    with cf.ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn'),initializer=init_worker) as pool:
        while completed<args.episodes and time.monotonic()-start<args.seconds:
            count=min(args.batch_episodes,args.episodes-completed)
            sd,cd=cpu_state(actor),cpu_state(critic)
            jobs=[dict(seed=980000+args.index*2000+j,n=3+j%4,train_seed=args.seed,actor=sd,critic=cd,training=True,reward_version=args.version,run_id=run.name,trajectory=str(run/'trajectories'/f'episode_{j:05d}.jsonl.gz')) for j in range(completed,completed+count)]
            results=list(pool.map(episode,jobs))
            stats=update(actor,critic,ao,co,[t for trs,m in results for t in trs],args.device)
            for _,m in results: append_json(run/'episodes.jsonl',m)
            completed+=count;updates+=1
            ck=dict(actor=cpu_state(actor),critic=cpu_state(critic),actor_optimizer=ao.state_dict(),critic_optimizer=co.state_dict(),episodes=completed,updates=updates,config=vars(args),normalization='fixed physical obs_v2',python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),env_rng='No live env at complete episode batch boundary; future seeds fixed in config',lowlevel_hash=file_hash(LOWLEVEL))
            path=run/f'checkpoint_{completed:05d}.pt';torch.save(ck,path);torch.save(ck,run/'latest.pt')
            stats.update(episodes=completed,updates=updates,seconds=time.monotonic()-start,mean_burned=float(np.mean([m['burned_cells'] for _,m in results])))
            append_json(run/'updates.jsonl',stats)
            (run/'STATUS.md').write_text(f'# {run.name}\n\nPID {os.getpid()}, {args.device}, {args.workers} CPU workers.\n\nCompleted {completed}/{args.episodes}, updates {updates}, elapsed {stats["seconds"]:.1f}s.\nCheckpoint {path}. Resume only at completed batch boundaries.\nTraining reward is not acceptance.\n')
            print(json.dumps(stats),flush=True)
    assert file_hash(LOWLEVEL)==EXPECTED
    print(f'TRAINING_FINISHED episodes={completed} updates={updates}',flush=True)

if __name__=='__main__': main()

"""Frozen-model diagnostic evaluation: ID rotation, observation ablations, execution."""
import collections,concurrent.futures as cf,copy,gzip,json,multiprocessing as mp,time
import numpy as np
import torch
from .environment import DistributedAllocatorEnv,frozen_actor,EXPECTED
from .evaluate import OUT,paired_report
from .model import Actor
from .protocol import baseline,resolve_target
from .runner import append_json

def diagnostic(job):
    torch.set_num_threads(1)
    low=frozen_actor();actor=Actor().eval()
    if job.get('checkpoint'):
        ck=torch.load(job['checkpoint'],map_location='cpu',weights_only=False);actor.load_state_dict(ck['actor'])
    env=DistributedAllocatorEnv(job['seed'],job['n'],low)
    mode=job['mode'];n=job['n'];segments={};current={};physical=[]
    original_step=env.w.step
    def measured_step(moves):
        before=[float(u.e) for u in env.e.uavs]
        result=original_step(moves)
        for i,(u,o) in enumerate(zip(env.e.uavs,env.w.orders)):
            key=json.dumps([o.task,o.target,o.rescuer_id])
            if current.get(i)!=key:
                current[i]=key;segments[(i,len(physical))]=dict(uav=i,task=o.task,order=o.to_dict(),steps=0,success=False,relay_covered_steps=0)
            entry=next(v for k,v in reversed(segments.items()) if k[0]==i)
            entry['steps']+=1
            if o.task==0 and o.target:
                success=max(abs(o.target[0]-u.x),abs(o.target[1]-u.y))<=2
            elif o.task==1:
                p=next((r for r in env.e.rescuers if r.rescuer_id==o.rescuer_id),None)
                success=p is not None and (u.x-p.x)**2+(u.y-p.y)**2<=env.e.uav_comm_radius**2
                entry['relay_covered_steps']+=int(success)
            else: success=env.e.at_station(u.x,u.y) and u.e>before[i]
            entry['success']|=bool(success)
        physical.append(dict(t=env.w.t,uavs=[[float(u.x),float(u.y),float(u.e)] for u in env.e.uavs],persons=[[int(r.rescuer_id),float(r.x),float(r.y),bool(r.is_covered),float(r.AoI)] for r in env.e.rescuers],orders=[o.to_dict() for o in env.w.orders]))
        return result
    env.w.step=measured_step
    if mode=='id_rotate':
        from . import environment
        def rotated(snapshot,actions,self_id):
            remap={i:(i+1)%n for i in range(n)};snap=copy.deepcopy(snapshot)
            for u in snap['uavs']: u['id']=remap[u['id']]
            return resolve_target(snap,{remap[i]:a for i,a in actions.items()},remap[self_id])
        environment.resolve_target=rotated
    for k in range(30):
        local,_=env.observations()
        if job.get('kind')=='h3': actions=baseline(env.snapshot(),'h3',job['threshold'],job['relay_mode'])
        else:
            for o in local:
                if mode=='no_teammates': o['mate_mask']=np.zeros_like(o['mate_mask'])
                if mode=='no_persons': o['person_mask']=np.zeros_like(o['person_mask'])
            actions=[actor.act(o) for o in local]
        env.step(actions)
    if mode=='id_rotate':
        from . import environment
        environment.resolve_target=resolve_target
    m=dict(env.metrics(),mode=mode,kind=job.get('kind','mappo'),train_seed=job.get('train_seed'),
        execution_segments=list(segments.values()),metric_definition='Actual protected order segments; success is any within2 explore target / within own UAV radius of assigned live person / at pad with energy increase. Includes right-censored segments; relay covered fraction counts every actual relay step, not arrival-conditional 30-step skill rate.')
    rates={}
    for task in (0,1,2):
        ss=[v for v in segments.values() if v['task']==task]
        rates[task]=dict(segments=len(ss),any_success_rate=float(np.mean([s['success'] for s in ss])) if ss else None,steps=sum(s['steps'] for s in ss),relay_coverage_fraction=sum(s['relay_covered_steps'] for s in ss)/max(1,sum(s['steps'] for s in ss)))
    m['execution_rates']=rates
    d=OUT/'diagnostics';d.mkdir(exist_ok=True)
    path=d/f'{job.get("train_seed","h3")}_{mode}_n{n}_s{job["seed"]}.jsonl.gz'
    with gzip.open(path,'wt') as f:
        for row in physical:f.write(json.dumps(row)+'\n')
    env.close();return m

def main():
    while not (OUT/'frozen_manifest.json').exists(): time.sleep(30)
    manifest=json.loads((OUT/'frozen_manifest.json').read_text());jobs=[]
    for seed,selected in manifest['selected'].items():
        for mode in ('normal','id_rotate','no_teammates','no_persons'):
            for n in range(3,9):
                for j in range(2):
                    scenario=(987000+(n-3)*40+j) if n<=6 else (988000+(n-7)*40+j)
                    jobs.append(dict(train_seed=seed,checkpoint=selected['checkpoint'],seed=scenario,n=n,mode=mode))
    for n in range(3,9):
        for j in range(2):
            scenario=(987000+(n-3)*40+j) if n<=6 else (988000+(n-7)*40+j)
            jobs.append(dict(kind='h3',train_seed='h3',seed=scenario,n=n,mode='normal',threshold=manifest['h3']['threshold'],relay_mode=manifest['h3']['relay_mode']))
    results=[]
    with cf.ProcessPoolExecutor(16,mp_context=mp.get_context('spawn')) as pool:
        for f in cf.as_completed([pool.submit(diagnostic,j) for j in jobs]):
            m=f.result();append_json(OUT/'robustness.jsonl',m);results.append(m)
    reports={}
    for seed in manifest['selected']:
        normal=[x for x in results if x['train_seed']==seed and x['mode']=='normal']
        reports[seed]={mode:paired_report([x for x in results if x['train_seed']==seed and x['mode']==mode],normal) for mode in ('id_rotate','no_teammates','no_persons')}
    (OUT/'robustness_summary.json').write_text(json.dumps(dict(reports=reports,warning='Two heldout scenes per fleet only. Frozen-model inference ablations, no retraining; ID rotation rotates UAV labels in target arbitration, not movement Actor ID one-hot. No model selection uses these results.'),indent=2))
    print('ROBUSTNESS_FINISHED',flush=True)

if __name__=='__main__': main()

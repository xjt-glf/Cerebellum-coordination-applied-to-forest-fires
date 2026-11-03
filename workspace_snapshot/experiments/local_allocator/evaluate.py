"""Paired evaluation with scenario-level bootstrap. No heldout tuning."""
import argparse,concurrent.futures as cf,json,multiprocessing as mp
from pathlib import Path
import numpy as np
import torch
from .environment import ROOT
from .runner import episode,init_worker,append_json
OUT=ROOT/'research_outputs/local_allocator_mappo_20261103'

def evaluation_jobs(kind,ns,count,start,tag,checkpoint=None,threshold=260,relay_mode='one'):
    state=torch.load(checkpoint,map_location='cpu',weights_only=False) if checkpoint else {}
    return [dict(seed=start+(n-min(ns))*40+j,n=n,kind=kind,deterministic=True,actor=state.get('actor'),critic=state.get('critic'),threshold=threshold,relay_mode=relay_mode,train_seed=state.get('config',{}).get('seed',0),run_id=tag,trajectory=str(OUT/'evaluation'/tag/f'n{n}_s{start+(n-min(ns))*40+j}.jsonl.gz')) for n in ns for j in range(count)]

def execute(jobs,output,workers=8):
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    existing=[json.loads(s) for s in output.read_text().splitlines() if s] if output.exists() else []
    keys={(m['seed'],m['n']) for m in existing}
    pending=[j for j in jobs if (j['seed'],j['n']) not in keys]
    with cf.ProcessPoolExecutor(workers,mp_context=mp.get_context('spawn'),initializer=init_worker) as pool:
        for fut in cf.as_completed([pool.submit(episode,j) for j in pending]):
            _,m=fut.result();append_json(output,m);existing.append(m)
            print(json.dumps(dict(output=str(output),completed=len(existing),seed=m['seed'],n=m['n'],burned=m['burned_cells'],seconds=m['seconds'])),flush=True)
    return existing

def paired_report(model,baseline):
    bm={(x['n'],x['seed']):x for x in baseline};report={}
    metrics=('burned_cells','depleted','cov_mean','uav_cov_mean','aoi_p95','longest_disconnect','explored','burning_cells','burned_out_cells','extinguished_cells','uncontained_exposure','override_per_100_uav_steps')
    for n in sorted({x['n'] for x in model}):
        pairs=[(x,bm[n,x['seed']]) for x in model if x['n']==n and (n,x['seed']) in bm]
        if not pairs: continue
        stats={}
        for key in metrics:
            a=np.array([x[key] for x,b in pairs],float);b=np.array([b[key] for x,b in pairs],float);d=a-b
            rng=np.random.default_rng(61203);boot=d[rng.integers(0,len(d),size=(2000,len(d)))].mean(1)
            stats[key]=dict(model_mean=float(a.mean()),h3_mean=float(b.mean()),paired_mean=float(d.mean()),paired_median=float(np.median(d)),paired_95ci=np.percentile(boot,[2.5,97.5]).tolist())
        reduction=-stats['burned_cells']['paired_mean']/max(1,stats['burned_cells']['h3_mean'])
        passed=reduction>=.05 and stats['depleted']['paired_mean']<=.1 and stats['cov_mean']['paired_mean']>=-.03
        report[n]=dict(episodes=len(pairs),provisional=len(pairs)<30,burn_reduction=reduction,engineering_thresholds_passed=passed,metrics=stats,failures=[dict(seed=x['seed'],burned_delta=x['burned_cells']-b['burned_cells']) for x,b in pairs if x['burned_cells']>b['burned_cells']])
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',default='h3');p.add_argument('--checkpoint');p.add_argument('--ns',default='3,4,5,6');p.add_argument('--count',type=int,default=4);p.add_argument('--start',type=int,default=986000);p.add_argument('--tag',required=True);p.add_argument('--workers',type=int,default=8);p.add_argument('--threshold',type=int,default=260);p.add_argument('--relay-mode',default='one');a=p.parse_args()
    execute(evaluation_jobs(a.kind,list(map(int,a.ns.split(','))),a.count,a.start,a.tag,a.checkpoint,a.threshold,a.relay_mode),OUT/f'{a.tag}.jsonl',a.workers)

if __name__=='__main__': main()

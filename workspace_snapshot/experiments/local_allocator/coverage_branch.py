"""Independent reward v3: development-first selection, then immutable evaluation."""
import concurrent.futures as cf,datetime,json,multiprocessing as mp,os,subprocess,sys,time
import numpy as np
import torch
from .evaluate import OUT,evaluation_jobs,execute,paired_report
from .environment import ROOT,file_hash,LOWLEVEL,EXPECTED
from .runner import episode,init_worker,append_json

def load(p):return [json.loads(s) for s in p.read_text().splitlines() if s] if p.exists() else []

def main():
    torch.set_num_threads(1)
    config=dict(version='v3_coverage',created=str(datetime.datetime.now()),trigger='Development communication guard failed in v1_count; no heldout results accessed',change='uncovered penalty increases from .010 to .100 per physical step; all other components unchanged',train_seeds=[1103,1104,1105],training_scene_pools='same as v1, paired reward branch',budget_seconds_per_seed=1100,workers=16,development_count_per_fleet=8,heldout_count_per_fleet=30,main_v1_preserved=True)
    (OUT/'v3_coverage_config.json').write_text(json.dumps(config,indent=2))
    children=[]
    for index,seed in enumerate((1103,1104,1105)):
        log=open(OUT/f'v3_s{seed}_training.log','a')
        cmd=[sys.executable,'-u','-m','experiments.local_allocator.train','--seed',str(seed),'--index',str(index),'--workers','16','--batch-episodes','16','--seconds','1100','--device',f'cuda:{index+4}','--version','v3_coverage']
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
        children.append((proc,log))
    for proc,log in children:
        if proc.wait()!=0: raise RuntimeError('v3 learner failed')
        log.close()
    baseline=load(OUT/'dev_h3_selected.jsonl');selected={};allcandidates={}
    for seed in (1103,1104,1105):
        run=OUT/f'v3_coverage_s{seed}';paths=sorted(run.glob('checkpoint_*.pt'))
        picked=sorted(set([paths[len(paths)//2],paths[-1]]));candidates=[]
        for ck in picked:
            tag=f'dev_v3_s{seed}_{ck.stem}'
            ms=execute(evaluation_jobs('mappo',[3,4,5,6],8,986000,tag,ck),OUT/(tag+'.jsonl'),16)
            report=paired_report(ms,baseline)
            guard=all(v['metrics']['depleted']['paired_mean']<=.1 and v['metrics']['cov_mean']['paired_mean']>=-.03 for v in report.values())
            score=float(np.mean([v['metrics']['burned_cells']['paired_mean']/max(1,v['metrics']['burned_cells']['h3_mean']) for v in report.values()]))
            candidates.append(dict(checkpoint=str(ck),guard_pass=guard,score=score,report=report))
        allcandidates[seed]=candidates;selected[seed]=min(candidates,key=lambda c:(not c['guard_pass'],c['score']))
        (OUT/'v3_development_selection.json').write_text(json.dumps(allcandidates,indent=2))
    h3=json.loads((OUT/'h3_selection.json').read_text())['selected']
    manifest=dict(frozen_at=str(datetime.datetime.now()),reward_version='v3_coverage',selected=selected,h3=h3,
        checkpoint_sha256={s:file_hash(c['checkpoint']) for s,c in selected.items()},source_sha256={str(p):file_hash(p) for p in (ROOT/'experiments/local_allocator').glob('*.py')},lowlevel_sha256=file_hash(LOWLEVEL),heldout_tuning=False,provisional='8 development scenes per fleet')
    (OUT/'v3_frozen_manifest.json').write_text(json.dumps(manifest,indent=2))
    jobs=[]
    for seed,candidate in selected.items():
        tag=f'heldout_v3_s{seed}'
        part=evaluation_jobs('mappo',[3,4,5,6],30,987000,tag,candidate['checkpoint'])+evaluation_jobs('mappo',[7,8],30,988000,tag,candidate['checkpoint'])
        jobs.extend([dict(j,_tag=tag) for j in part])
    with cf.ProcessPoolExecutor(24,mp_context=mp.get_context('spawn'),initializer=init_worker) as pool:
        fs={pool.submit(episode,j):j['_tag'] for j in jobs}
        for i,f in enumerate(cf.as_completed(fs)):
            _,m=f.result();append_json(OUT/(fs[f]+'.jsonl'),m)
            if i%24==0:print(f'v3 frozen evaluation {i+1}/{len(fs)}',flush=True)
    # H3 paired run is shared with primary workflow and must complete first.
    while len(load(OUT/'heldout_h3.jsonl'))<180:time.sleep(20)
    baseline=load(OUT/'heldout_h3.jsonl')
    reports={seed:paired_report(load(OUT/f'heldout_v3_s{seed}.jsonl'),baseline) for seed in selected}
    (OUT/'v3_paired_results.json').write_text(json.dumps(reports,indent=2))
    for seed,rec in selected.items():
        ck=torch.load(rec['checkpoint'],map_location='cpu',weights_only=False)
        directory=OUT/'deployment'/f'v3_s{seed}';directory.mkdir(parents=True,exist_ok=True)
        torch.save(ck['actor'],directory/'actor_selected.pt');torch.save(ck['critic'],directory/'critic_selected.pt')
    assert file_hash(LOWLEVEL)==EXPECTED
    print('V3_BRANCH_FINISHED',flush=True)

if __name__=='__main__':main()

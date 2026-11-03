"""Pre-heldout v4 balanced reward fork. Preserve parents and all failed branches."""
import concurrent.futures as cf,datetime,json,multiprocessing as mp,os,subprocess,sys,time
import numpy as np
import torch
from .evaluate import OUT,evaluation_jobs,execute,paired_report
from .environment import ROOT,LOWLEVEL,EXPECTED,file_hash
from .runner import episode,init_worker,append_json

def load(p):return [json.loads(s) for s in p.read_text().splitlines() if s] if p.exists() else []

def main():
    torch.set_num_threads(1);parents={s:OUT/f'v1_count_s{s}/checkpoint_00512.pt' for s in (1103,1104,1105)}
    while not all(p.exists() for p in parents.values()):time.sleep(15)
    config=dict(version='v4_balanced',created=str(datetime.datetime.now()),trigger='Development-only v3 checkpoint128 had .94-.99 total coverage but more burned cells; no heldout accessed',uncovered_penalty=.035,exploration_per100=.020,parent_episodes=512,maximum_new_episodes=256,parent_checkpoints={s:dict(path=str(p),sha256=file_hash(p)) for s,p in parents.items()},all_other_rewards='v1_count',selection='development-only midpoint/latest, 8 per N',comparison='reward continuation fork, not cold-start equal-budget trial')
    (OUT/'v4_balanced_config.json').write_text(json.dumps(config,indent=2));children=[]
    for index,seed in enumerate((1103,1104,1105)):
        stream=open(OUT/f'v4_s{seed}_training.log','a');device=(7,3,7)[index]
        cmd=[sys.executable,'-u','-m','experiments.local_allocator.train','--seed',str(seed),'--index',str(index),'--workers','16','--batch-episodes','16','--episodes','768','--seconds','1100','--device',f'cuda:{device}','--version','v4_balanced','--resume',str(parents[seed])]
        proc=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'));children.append((proc,stream))
    for proc,stream in children:
        if proc.wait()!=0:raise RuntimeError('v4 learner failed')
        stream.close()
    baseline=load(OUT/'dev_h3_selected.jsonl');selected={};allcandidates={}
    for seed in (1103,1104,1105):
        paths=sorted((OUT/f'v4_balanced_s{seed}').glob('checkpoint_*.pt'));candidates=[]
        for path in sorted(set([paths[len(paths)//2],paths[-1]])):
            tag=f'dev_v4_s{seed}_{path.stem}';ms=execute(evaluation_jobs('mappo',[3,4,5,6],8,986000,tag,path),OUT/(tag+'.jsonl'),16);report=paired_report(ms,baseline)
            guard=all(v['metrics']['depleted']['paired_mean']<=.1 and v['metrics']['cov_mean']['paired_mean']>=-.03 for v in report.values());score=float(np.mean([v['metrics']['burned_cells']['paired_mean']/max(1,v['metrics']['burned_cells']['h3_mean']) for v in report.values()]));candidates.append(dict(checkpoint=str(path),guard_pass=guard,score=score,report=report))
        allcandidates[seed]=candidates;selected[seed]=min(candidates,key=lambda c:(not c['guard_pass'],c['score']))
        (OUT/'v4_development_selection.json').write_text(json.dumps(allcandidates,indent=2))
    manifest=dict(frozen_at=str(datetime.datetime.now()),selected=selected,checkpoint_sha256={s:file_hash(c['checkpoint']) for s,c in selected.items()},source_sha256={str(p):file_hash(p) for p in (ROOT/'experiments/local_allocator').glob('*.py')},config=config,lowlevel_sha256=file_hash(LOWLEVEL),heldout_tuning=False)
    (OUT/'v4_frozen_manifest.json').write_text(json.dumps(manifest,indent=2));jobs=[]
    for seed,candidate in selected.items():
        tag=f'heldout_v4_s{seed}';part=evaluation_jobs('mappo',[3,4,5,6],30,987000,tag,candidate['checkpoint'])+evaluation_jobs('mappo',[7,8],30,988000,tag,candidate['checkpoint']);jobs.extend([dict(j,_tag=tag) for j in part])
    with cf.ProcessPoolExecutor(24,mp_context=mp.get_context('spawn'),initializer=init_worker) as pool:
        fs={pool.submit(episode,j):j['_tag'] for j in jobs}
        for i,f in enumerate(cf.as_completed(fs)):
            _,m=f.result();append_json(OUT/(fs[f]+'.jsonl'),m)
            if i%24==0:print(f'v4 frozen evaluation {i+1}/{len(fs)}',flush=True)
    while len(load(OUT/'heldout_h3.jsonl'))<180:time.sleep(20)
    baseline=load(OUT/'heldout_h3.jsonl');reports={s:paired_report(load(OUT/f'heldout_v4_s{s}.jsonl'),baseline) for s in selected}
    (OUT/'v4_paired_results.json').write_text(json.dumps(reports,indent=2));lines=['# v4 平衡奖励续训分支','',str(datetime.datetime.now()),'', '因v3开发集出现通信与过火权衡，失联惩罚固定为0.035、新探索/100格奖励固定为0.020；其余与v1一致。每种子从原版512局checkpoint续训，分支实际新增局数单列，不能把继承的512局当作本分支新增。参数在留出前固定；开发8场景/N选择，未按留出跨版本回选。','', '|种子|新增训练局数|继承局数|N|配对局数|过火下降|覆盖差|耗尽差|点估计门槛|','|---|---:|---:|---:|---:|---:|---:|---:|---|']
    exports={}
    for seed,rs in reports.items():
        count=len(load(OUT/f'v4_balanced_s{seed}/episodes.jsonl'))
        for n,r in rs.items():lines.append(f'|{seed}|{count}|512|{n}|{r["episodes"]}|{r["burn_reduction"]:.2%}|{r["metrics"]["cov_mean"]["paired_mean"]:.3f}|{r["metrics"]["depleted"]["paired_mean"]:.3f}|{r["engineering_thresholds_passed"]}|')
        ck=torch.load(selected[seed]['checkpoint'],map_location='cpu',weights_only=False);d=OUT/'deployment'/f'v4_s{seed}';d.mkdir(parents=True,exist_ok=True);torch.save(ck['actor'],d/'actor_selected.pt');torch.save(ck['critic'],d/'critic_selected.pt');exports[seed]=dict(actor_sha256=file_hash(d/'actor_selected.pt'),critic_sha256=file_hash(d/'critic_selected.pt'))
    lines+=['','完整原始数据、置信区间与失败场景：v4_paired_results.json 和 heldout_v4_s*.jsonl。7/8为未见规模外推。整体实现限制和provisional判定仍见final_report.md；不因某单元点估计通过而宣称全面验收。']
    (OUT/'v4_report.md').write_text('\n'.join(lines)+'\n');(OUT/'v4_deployment_exports.json').write_text(json.dumps(exports,indent=2));assert file_hash(LOWLEVEL)==EXPECTED
    print('V4_BRANCH_FINISHED',flush=True)

if __name__=='__main__':main()

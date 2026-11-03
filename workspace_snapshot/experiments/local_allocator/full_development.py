"""Extend frozen selected models to all40 development scenes, without reselection."""
import concurrent.futures as cf,json,multiprocessing as mp,time
from .evaluate import OUT,evaluation_jobs,paired_report
from .runner import episode,init_worker,append_json

def load(p):return [json.loads(s) for s in p.read_text().splitlines() if s] if p.exists() else []

def main():
    while not (OUT/'frozen_manifest.json').exists():time.sleep(30)
    manifest=json.loads((OUT/'frozen_manifest.json').read_text());h3=manifest['h3'];jobs=[]
    policies=[('h3',None,'full_dev_h3')]+[('mappo',x['checkpoint'],f'full_dev_s{s}') for s,x in manifest['selected'].items()]
    for kind,ck,tag in policies:
        jobs.extend([dict(j,_tag=tag) for j in evaluation_jobs(kind,[3,4,5,6],40,986000,tag,ck,h3['threshold'],h3['relay_mode'])])
    with cf.ProcessPoolExecutor(16,mp_context=mp.get_context('spawn'),initializer=init_worker) as pool:
        fs={pool.submit(episode,j):j['_tag'] for j in jobs}
        for i,f in enumerate(cf.as_completed(fs)):
            _,m=f.result();append_json(OUT/(fs[f]+'.jsonl'),m)
            if i%32==0:print(f'Full development diagnostic {i+1}/{len(fs)}',flush=True)
    baseline=load(OUT/'full_dev_h3.jsonl')
    reports={s:paired_report(load(OUT/f'full_dev_s{s}.jsonl'),baseline) for s in manifest['selected']}
    (OUT/'full_development_diagnostic.json').write_text(json.dumps(dict(reports=reports,note='40 development scenes per fleet for already frozen models. This is supplementary diagnostic evidence, not retrospective selection. Original selection used8 scenes per fleet and remains provisional.'),indent=2))
    print('FULL_DEVELOPMENT_FINISHED',flush=True)

if __name__=='__main__':main()

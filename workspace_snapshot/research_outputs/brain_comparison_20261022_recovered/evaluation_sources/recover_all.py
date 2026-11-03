import sys,os,json,subprocess,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'research_outputs/brain_comparison_20261022_recovered';OUT.mkdir(exist_ok=True)
def job(seed):
    with open(OUT/f'recover_{seed}.log','a') as log:
        code=subprocess.call([sys.executable,str(ROOT/'experiments/brain_comparison/recover.py'),str(seed)],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=log)
    print(seed,code,flush=True);return [seed,code]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    result=list(pool.map(job,list(range(730000,730010))))
(OUT/'batch_finished.json').write_text(json.dumps(result))

import os,sys,time,json,subprocess,concurrent.futures,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'research_outputs/brain_comparison_20261022';OUT.mkdir(exist_ok=True)
seeds=list(range(730000,730010))
manifest={'seeds':seeds,'arms':['heuristic','llm'],'grid':[100,100],'uavs':4,'rescuers':10,'initial_fires':3,'steps':600,'spread_steps':40,'spread_gain':1.,'randomize_wind':True,'replan_every':20,'actor':'research_outputs/lowlevel_training_final/actor_selected.pt','actor_execution':'argmax; no waypoint assist','goal_protocol':'same explicit-change acceptance for both planners','charge_protocol':'webapp.ChargeProtocol reserve_steps=12, both arms','charge_at':300,'max_chargers':2,'explore_until':.7,'llm_input':'full_environment truth; heuristic receives same state but its implementation ignores full_environment','llm_temperature':.2,'api_timeout_seconds':180,'transport_max_attempts':3,'time_model':'synchronous paused simulation during planning; API latency recorded separately','fixed_horizon':True,'started_at':time.strftime('%Y-%m-%d %H:%M:%S %z'),'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'experiments/brain_comparison/run.py',ROOT/'webapp/brains.py',ROOT/'webapp/goal_protocol.py',ROOT/'webapp/charge_protocol.py',ROOT/'webapp/world_snapshot.py',ROOT/'lowlevel/env_wrapper.py',ROOT/'Env/低层环境.py']}}
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
def job(arm,seed):
    path=OUT/f'{arm}_{seed}'
    if (path/'summary.json').exists():return (arm,seed,'exists')
    with open(OUT/f'{arm}_{seed}.log','a') as log:
        code=subprocess.call([sys.executable,str(ROOT/'experiments/brain_comparison/run.py'),arm,str(seed)],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=log)
    print(arm,seed,'exit',code,flush=True);return arm,seed,code
# Independent processes isolate the environment's process-global NumPy RNG and chdir.
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as llmpool,concurrent.futures.ThreadPoolExecutor(max_workers=2) as hpool:
    jobs=[llmpool.submit(job,'llm',s) for s in seeds]+[hpool.submit(job,'heuristic',s) for s in seeds]
    results=[f.result() for f in concurrent.futures.as_completed(jobs)]
(OUT/'finished.json').write_text(json.dumps(results))

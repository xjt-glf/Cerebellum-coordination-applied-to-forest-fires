"""Fixed-seed API-vs-heuristic evaluation. No training; no live server mutation."""
import os, sys, json, time, gzip, hashlib, traceback, urllib.request, urllib.error
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from webapp.server import RunCfg
from webapp.brains import LLMBrain,HeuristicBrain,_CTX
from webapp.goal_protocol import GoalProtocol
from webapp.charge_protocol import ChargeProtocol
from webapp.world_snapshot import full_snapshot
from lowlevel.mission_demo import sense_state
from lowlevel.env_wrapper import TaskConditionedEnv
from experiments.critic_advice.harness import load_actor,task_metrics
OUT=ROOT/'research_outputs/brain_comparison_20261022'
ACTOR=ROOT/'research_outputs/lowlevel_training_final/actor_selected.pt'
def clean(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,Path): return str(x)
    raise TypeError(type(x).__name__)
def dump(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,default=clean,indent=2));tmp.replace(path)
def append(path,obj):
    with open(path,'a') as f:f.write(json.dumps(obj,ensure_ascii=False,default=clean)+'\n')
def zipped(path,obj):
    with gzip.open(path,'wt',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,default=clean)
class CommonGoals(GoalProtocol):
    def apply(self,plan,source='heuristic',event=None):
        # Identical explicit-change acceptance, regardless of planner identity.
        return super().apply(plan,source='llm',event=event)
class RecordedLLM(LLMBrain):
    def _call(self,user_text):
        body={'model':self.model,'temperature':self.temperature,'messages':[{'role':'system','content':self.SYSTEM},{'role':'user','content':user_text}]}
        zipped(self.path/f'call_{self.tick:04d}_request.json.gz',body)
        for attempt in range(1,4):
            start=time.monotonic()
            try:
                req=urllib.request.Request(self.url,data=json.dumps(body).encode(),method='POST',headers={'Content-Type':'application/json','Authorization':'Bearer '+self.api_key})
                with urllib.request.urlopen(req,timeout=self.timeout,context=_CTX) as response:obj=json.loads(response.read().decode())
                zipped(self.path/f'call_{self.tick:04d}_response.json.gz',obj)
                self.last_usage=obj.get('usage',{})
                append(self.path/'api_attempts.jsonl',{'t':self.tick,'attempt':attempt,'ok':True,'seconds':time.monotonic()-start,'usage':self.last_usage,'returned_model':obj.get('model')})
                return obj['choices'][0]['message']['content']
            except Exception as ex:
                # Exception strings contain no Authorization header; redact defensively.
                msg=str(ex).replace(self.api_key,'[REDACTED]') if self.api_key else str(ex)
                append(self.path/'api_attempts.jsonl',{'t':self.tick,'attempt':attempt,'ok':False,'seconds':time.monotonic()-start,'error':msg})
                retry=not isinstance(ex,urllib.error.HTTPError) or ex.code in (408,429,500,502,503,504)
                if not retry or attempt==3:raise
                time.sleep(2**attempt)
def run(arm,seed):
    torch.set_num_threads(1)
    folder=OUT/f'{arm}_{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'summary.json').exists():return
    started=time.monotonic();cfg=RunCfg()
    env=TaskConditionedEnv(num_uavs=4,max_steps=600,spread_steps=40,grid_size=(100,100),n_fire_seeds=3,n_fire_seeds_range=None,randomize_wind=True,spread_gain=1,num_rescuers=10,seed=seed,task_sampler='planner',replan_every=0,terminate_on_all_depleted=False)
    obs=env.reset();actor=load_actor(str(ACTOR));goals=CommonGoals(env,seed);charging=ChargeProtocol(env,goals)
    hb=HeuristicBrain(explore_until=.7,charge_at=300,max_chargers=2)
    brain=hb if arm=='heuristic' else RecordedLLM(cfg.base_url,cfg.api_key,cfg.model,temperature=cfg.temperature,timeout=180,fallback=hb)
    if arm=='llm':brain.path=folder;brain.full_state=True
    initial=full_snapshot(env,goals=goals)
    zipped(folder/'initial_state.json.gz',initial)
    initial_hash=hashlib.sha256(json.dumps(initial,default=clean,sort_keys=True).encode()).hexdigest()
    recent=[];rows=[];aoi=[];decisions=[];blackouts=np.zeros(10,dtype=int);max_blackout=0;depleted_ever=set();switches=0;previous=None
    for step in range(600):
        if step%20==0:
            state=sense_state(env);state['full_environment']=full_snapshot(env,goals=goals,recent_decisions=recent)
            if arm=='llm':brain.tick=step
            else:zipped(folder/f'state_{step:04d}.json.gz',state)
            ts=time.monotonic();plan,reason,meta=brain.plan(state)
            elapsed=time.monotonic()-ts
            obs,applied,notes=goals.apply(plan,source=meta.get('source'))
            record={'t':step,'requested_plan':plan,'plan':applied,'reason':reason,'meta':meta,'seconds':elapsed,'goal_notes':notes,'brief':{k:v for k,v in state.items() if k!='full_environment'}}
            append(folder/'decisions.jsonl',record);decisions.append(record);recent.append(record)
            print(f'{arm} seed={seed} t={step} source={meta.get("source")} seconds={elapsed:.1f}',flush=True)
        new_obs,notes=charging.sync()
        if new_obs is not None:obs=new_obs
        if notes:append(folder/'charge_overrides.jsonl',{'t':step,'notes':notes,'orders':[x.to_dict() for x in env.orders]})
        current=[x.to_dict() for x in env.orders]
        if previous is not None:switches+=sum(a!=b for a,b in zip(previous,current))
        previous=current
        with torch.no_grad():actions=actor.act(torch.as_tensor(obs,dtype=torch.float32),deterministic=True)[0].numpy()
        obs,reward,done,info=env.step(actions);goals.update()
        metrics=task_metrics(env);rows.append(metrics)
        person_aoi=[float(r.AoI) for r in env.env.rescuers];aoi.extend(person_aoi)
        blackouts=np.where([r.is_covered for r in env.env.rescuers],0,blackouts+1);max_blackout=max(max_blackout,int(blackouts.max()))
        depleted_ever.update(i for i,u in enumerate(env.env.uavs) if u.e<=0)
        append(folder/'steps.jsonl',{'metrics':metrics,'actions':actions,'orders':current,'person_aoi':person_aoi,'uavs':[{'id':i,'x':float(u.x),'y':float(u.y),'energy':float(u.e)} for i,u in enumerate(env.env.uavs)]})
        dump(folder/'progress.json',{'arm':arm,'seed':seed,'t':env.t,'elapsed_seconds':time.monotonic()-started,'api_calls':getattr(brain,'n_call',0),'api_fallbacks':getattr(brain,'n_fail',0)})
        if done and env.t<600:raise RuntimeError(f'Premature done at {env.t}')
    final=task_metrics(env)
    final.update(arm=arm,seed=seed,initial_hash=initial_hash,actor_sha256=hashlib.sha256(ACTOR.read_bytes()).hexdigest(),steps=env.t,wall_seconds=time.monotonic()-started,mean_coverage=float(np.mean([r['coverage'] for r in rows])),mean_person_aoi=float(np.mean(aoi)),p95_person_aoi=float(np.percentile(aoi,95)),max_blackout_steps=max_blackout,depleted_ever=len(depleted_ever),instruction_changes=switches,planning_calls=len(decisions),api_fallbacks=getattr(brain,'n_fail',0),planning_seconds=sum(r['seconds'] for r in decisions))
    dump(folder/'summary.json',final)
def main():
    import fcntl
    arm=sys.argv[1];seed=int(sys.argv[2])
    folder=OUT/f'{arm}_{seed}';folder.mkdir(parents=True,exist_ok=True)
    lock=open(folder/'run.lock','w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    try:run(arm,seed)
    except Exception:
        folder=OUT/f'{arm}_{seed}';folder.mkdir(parents=True,exist_ok=True)
        (folder/'FAILED.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':main()

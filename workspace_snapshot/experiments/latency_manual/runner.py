"""Interactive Codex planner, paused vs injected asynchronous delay.
One process maintains two live JointRuntime objects. No copy/snapshot and no API.
JSON lines: initial shared joint plan; then {"A": joint plan, "B": joint plan}.
The planner receives only compact observable state at each boundary. B holds the
previous task for 5 environment ticks before the submitted decision is applied.
"""
import copy, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
REMOTE=ROOT/'远端机器文件夹'
sys.path.insert(0,str(REMOTE))
from experiments.feedback_attribution.joint_runtime import JointRuntime, load_actor, task_metrics
from lowlevel.obs_builder import shared_cache

SEED=int(os.environ.get('LATENCY_SEED','975001')); DELAY=5; PERIOD=20; END=600
OUT=ROOT/'research_outputs/latency_manual_20261103'/f'seed_{SEED}'
OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1)
actor=load_actor(str(ROOT/'research_outputs/lowlevel_training_final/actor_selected.pt'))
rts={k:JointRuntime(seed=SEED,regime='normal',actor=actor) for k in ('A','B')}

def jsave(path,data):
 path.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=lambda x: x.item() if isinstance(x,np.generic) else str(x)))

def state(rt):
 s=rt.observable_state(); sense=s['sense']; m=shared_cache(rt.env.env)['M']; unknown=m<0
 tiles=[[int(unknown[x:x+20,y:y+20].sum()) for y in range(0,100,20)] for x in range(0,100,20)]
 active={str(k):{'task':v.get('task'),'target':v.get('target'),'state':v.get('state'),'sector':v.get('approach_sector'),'index':v.get('index')} for k,v in s['active'].items()}
 pending={str(k):{'task':v.get('task'),'target':v.get('target'),'state':v.get('state')} for k,v in s['pending'].items()}
 # s.discovery/burned/rescuer_known may use ground truth: exclude them.
 return {'t':sense['t'],'explored':sense['explored'],'coverage':sense['coverage'],
  'max_aoi':max([p['aoi'] for p in sense['rescuers']],default=0), 'uavs':sense['uavs'],
  'orders':s['uav_orders'], 'persons':[{k:p.get(k) for k in ('id','x','y','c','aoi','task','state','target','pending')} for p in s['persons']],
  'known_uav_fire_clusters':sense['clusters'], 'known_person_fire_clusters':s['known_fire_clusters'],
  'person_active':active,'person_pending':pending,'unknown_tiles_20x20':tiles,
  'charging_stations':sense['charging_stations'], 'base':sense['base_pos']}

def validate(p):
 if not isinstance(p,dict) or len(p.get('uavs',[]))!=4 or not isinstance(p.get('persons',[]),list):raise ValueError('Each plan requires exactly 4 uav orders and a persons list')
 p=copy.deepcopy(p);p['source']='codex_joint';p['reason']=str(p.get('reason','Codex jointly plans from observed state'))
 return p

def report():
 out={k:state(rt) for k,rt in rts.items()}
 # Compact live view for a human planner; the complete, auditable state remains
 # in decisions.jsonl. Do not expose truth-only fields here.
 brief={}
 for k,s in out.items():
  brief[k]={
   't':s['t'],'explored':s['explored'],'coverage':s['coverage'],'max_aoi':s['max_aoi'],
   'uavs':[{'id':u['id'],'p':u['pos'],'e':u['e'],'m':u['margin']} for u in s['uavs']],
   'fires':[{'c':f['centroid'],'n':f['size']} for f in s['known_uav_fire_clusters']],
   'people':[{k2:p[k2] for k2 in ('id','x','y','aoi','c','task','state')} for p in s['persons'] if p['aoi']>20 or p['task']!='auto']
  }
 print(json.dumps({'state':brief},ensure_ascii=False,separators=(',',':')),flush=True)
 return out

meta={'seed':SEED,'actorsha256':hashlib.sha256((ROOT/'research_outputs/lowlevel_training_final/actor_selected.pt').read_bytes()).hexdigest(),
 'period':PERIOD,'delay_steps_B':DELAY,'start':'identical initial plan at t=0; no initial delay',
 'max_steps':END,'real_time_mapping':'none; fixed delay in simulated steps','planner':'Codex manually submits each explicit joint plan; no LLM API, no heuristic brain',
 'state_scope':'observed summaries, known maps, orders and events; no truth-fire clusters or truth metrics in planner state',
 'limitation':'one seed, same Codex session and common context across arms; exploratory; subjective planner differs per arm'}
jsave(OUT/'metadata.json',meta)
states=report()
for line in sys.stdin:
 try:
  command=json.loads(line); t=rts['A'].env.t
  if t!=rts['B'].env.t:raise RuntimeError('arms at different observations')
  if t==0:
   plan=validate(command['initial']);plans={'A':copy.deepcopy(plan),'B':copy.deepcopy(plan)}
  else:plans={k:validate(command[k]) for k in rts}
  record={'observation_tick':t,'observations':states,'plans':plans,'applied':{},'at_application':{}}
  for k,rt in rts.items():
   if k=='B' and t>0:rt.run_to(min(t+DELAY,END),brain=None)
   record['at_application'][k]=state(rt)
   rt.apply_joint_plan(plans[k]);record['applied'][k]=copy.deepcopy(rt.decisions[-1])
   rt.run_to(min(t+PERIOD,END),brain=None)
  with open(OUT/'decisions.jsonl','a',encoding='utf8') as f:f.write(json.dumps(record,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')
  states=report()
  if rts['A'].env.t>=END:
   result={k:{**task_metrics(rt.env),'cov_mean':float(np.mean(rt.cov_hist)),'aoi_mean':float(np.mean(rt.aoi_hist)),'steps_run':len(rt.cov_hist),'charge_rewrites':rt.n_rewrites_charge,'orders_issued':len(rt.decisions)} for k,rt in rts.items()}
   jsave(OUT/'result.json',result);print(json.dumps({'done':result},ensure_ascii=False),flush=True);break
 except Exception as exc:
  print(json.dumps({'error':str(exc)},ensure_ascii=False),flush=True)

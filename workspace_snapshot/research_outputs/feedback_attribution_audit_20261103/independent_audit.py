"""Independent read-only audit of the remote agent's archived data and runtime.
Run from this file's original project checkout with Python 3.13 and the local venv:
PYTHONPATH=.venv/lib/python3.13/site-packages /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 research_outputs/feedback_attribution_audit_20261103/independent_audit.py
The remote folder is read-only; output goes only alongside this script.
"""
from pathlib import Path
import collections, copy, json, os, sys, glob, hashlib
import numpy as np
from scipy.stats import spearmanr
here=Path(__file__).resolve().parent
root=here.parent.parent
remote=root/'远端机器文件夹'
sys.path.insert(0,str(remote));os.chdir(remote)
from experiments.feedback_attribution.joint_runtime import JointRuntime,load_actor
from experiments.feedback_attribution.joint_planner import joint_heuristic_brain
from experiments.feedback_attribution.collector import alternatives,run_branch
from experiments.feedback_attribution.train import load_rows,EXECUTION,MATURITY,DEFAULT_METRICS
from experiments.feedback_attribution.evaluate import f1
actor=load_actor();rt=JointRuntime(seed=941101,actor=actor)
rt.run_to(20,brain=joint_heuristic_brain)
sit=rt.observable_state(); plan=joint_heuristic_brain(sit);snap=rt.snapshot('normal'); cloned=JointRuntime.restore(copy.deepcopy(snap),actor=actor)
links={"personnel_env_is_running_env":cloned.personnel.env is cloned.env.env,
       "goals_env_is_running_env":cloned.goals.env is cloned.env,
       "charging_wrapper_is_running_env":cloned.charging.wrapper is cloned.env,
       "charging_goals_is_active_goals":cloned.charging.goals is cloned.goals,
       "env_controller_is_personnel":cloned.env.env.personnel_controller is cloned.personnel}
source_cases={}
for source in ('manual_main_session','joint_heuristic','codex_joint','shared_model'):
 live=JointRuntime(seed=941101,actor=actor)
 live.apply_joint_plan({'source':source,'uavs':[{'id':0,'task':'fire_sensing','mode':'explore','target':[25,25]}], 'persons':[]})
 source_cases[source]={'requested':live.decisions[-1]['requested_uavs'][0], 'actual':live.decisions[-1]['actual_uavs'][0]}
cases=[]; baseline,_,_=run_branch(snap,actor,plan,20,{'kind':'noop'})
for name,alternative,edit in alternatives(sit,plan):
 if name not in {'uav_charge','uav_explore_target','uav_relay_target','person_retreat'}:continue
 x=JointRuntime.restore(copy.deepcopy(snap),actor=actor);x.apply_joint_plan(copy.deepcopy(alternative))
 values,_,labels=run_branch(snap,actor,alternative,20,edit)
 cases.append(dict(candidate=name,requested=alternative['uavs'][0],running_env_order=x.env.orders[0].to_dict(),goal_copy_order=x.goals.env.orders[0].to_dict(),delta={k:values[k]-v for k,v in baseline.items() if values[k]!=v},execution_label=labels['execution']))
counts=collections.defaultdict(collections.Counter)
for p in glob.glob('research_outputs/feedback_attribution_final/data/*.jsonl'):
 for line in open(p,encoding='utf8'):
  r=json.loads(line)
  if r['horizon'] not in (20,60,120):continue
  c=counts[(r['alternative'],r['horizon'])];y=r['task_metric_delta'];c['n']+=1
  c['all_zero']+=int(all(v==0 for v in y.values()));c['zero_ext']+=int(y['extinguished_cells']==0);c['zero_cov']+=int(y['coverage']==0);c['overridden']+=int(r['supervision']['E']['execution']=='overridden')
rows,_=load_rows(['research_outputs/feedback_attribution_final/data/*.jsonl']);train=[r for r in rows if r['split']=='train'];test=[r for r in rows if r['split']=='test'];preds=json.load(open('research_outputs/feedback_attribution_final/model_shared/predictions.json'));assert len(preds)==len(test)
f1s={}
for name,classes in [('execution',EXECUTION),('maturity',MATURITY)]:
 c=collections.defaultdict(collections.Counter)
 for r in train:c[(r['candidate'],r['horizon'])][r[name]]+=1
 truth=np.array([r[name] for r in test]);prior=np.array([c[(r['candidate'],r['horizon'])].most_common(1)[0][0] if c[(r['candidate'],r['horizon'])] else 0 for r in test]);pred=np.array([np.argmax(p[name+'_prob']) for p in preds])
 f1s[name]={'class_counts':dict(collections.Counter(int(x) for x in truth)),'training_candidate_and_horizon_majority_f1':f1(truth,prior,len(classes)),'model_f1':f1(truth,pred,len(classes))}
ranks={}
for name in ['burned_cells','extinguished_cells','coverage']:
 i=DEFAULT_METRICS.index(name); y=[r['cf'][i] for r in test];p=[r['counterfactual'][name] for r in preds]
 ranks[name]={'zero_fraction':sum(v==0 for v in y)/len(y),'spearman_tie_aware':float(spearmanr(y,p).statistic)}
out={'seed':941101,'boundary':20,'object_links':links,'source_cases':source_cases,'counterexamples':cases,'data_counts':{'%s_h%d'%k:dict(v) for k,v in sorted(counts.items())},'test_samples':len(test),'f1':f1s,'corrected_ranks':ranks,'model_hashes':{str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('research_outputs/feedback_attribution_final/model_shared').glob('checkpoint_*.pt')}}
(here/'audit_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps({'links':links,'candidate_counts':len(cases),'data_groups':len(counts),'test_samples':len(test),'f1':f1s,'ranks':ranks},ensure_ascii=False))

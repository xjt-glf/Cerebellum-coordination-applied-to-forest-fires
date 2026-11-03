"""Replay saved decisions only, add instrumentation, and require exact original trace.
No new policy trial, API request, training, or optimization.
"""
import sys,json,math,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/brain_comparison'))
import numpy as np
import run as runner
ARM=sys.argv[1];SEED=int(sys.argv[2]);OUTPUT=ROOT/'research_outputs/comprehensive_comparison_3seeds_20261022'
SOURCE=(ROOT/'research_outputs/codex_brain_3runs_20261022' if ARM=='codex' else ROOT/'research_outputs/brain_comparison_20261022_recovered')/f'{ARM}_{SEED}'
FOLDER=OUTPUT/'instrumented_replays'/f'{ARM}_{SEED}';runner.OUT=FOLDER.parent
saved={r['t']:r for r in [json.loads(x) for x in (SOURCE/'decisions.jsonl').read_text().splitlines()]}
OriginalEnv=runner.TaskConditionedEnv
class MeasuredEnv(OriginalEnv):
    def step(self,actions):
        e=self.env;prev=np.array([[u.x,u.y] for u in e.uavs]);prevp=np.array([[r.x,r.y] for r in e.rescuers]);pre_e=np.array([u.e for u in e.uavs])
        result=super().step(actions)
        bs=e.grid['burning_state'];burn=(bs>0)&(bs<2);active=e._active_true_burning_mask(bs)
        team=e.team_map_info;person=e.rescuers_map_info;tb=(team>0)&(team<2);pb=(person>0)&(person<2)
        pos=np.array([[u.x,u.y] for u in e.uavs]);rpos=np.array([[r.x,r.y] for r in e.rescuers]);energy=np.array([u.e for u in e.uavs]);cost=np.array([e.energy_cost_per_step(i) for i in range(4)])
        # Ideal 8-neighbour travel lower bound to any true pad, no collision delays included.
        padsteps=np.array([min(max(0,math.ceil(max(abs(u.x-st[0]),abs(u.y-st[1]))-e.charge_radius)) for st in e.charging_stations) for u in e.uavs])
        runner.append(FOLDER/'extra_steps.jsonl',{'t':self.t,'true_burning':int(burn.sum()),'true_active_burning':int(active.sum()),'burned_out':int((bs==2).sum()),'team_tp':int((tb&burn).sum()),'team_fp':int((tb&~burn).sum()),'person_tp':int((pb&burn).sum()),'person_fp':int((pb&~burn).sum()),'active_team_tp':int((tb&active).sum()),'active_person_tp':int((pb&active).sum()),'comm_coverage_rate':float(e.comm_coverage_rate),'blocked':[bool(u.blocked) for u in e.uavs],'oob':[bool(u.oob) for u in e.uavs],'at_pad':[bool(e.at_station(u.x,u.y)) for u in e.uavs],'recharged':[bool(u.recharged) for u in e.uavs],'return_margin':(energy-cost*padsteps).tolist(),'displacement':(pos-prev).tolist(),'rescuer_distance':np.linalg.norm(rpos-prevp,axis=1).tolist(),'energy_replenished':np.maximum(energy-np.maximum(pre_e-cost,0),0).tolist()})
        return result
class SavedBrain:
    def __init__(self,*args,**kwargs):self.n_call=0;self.n_fail=0
    def plan(self,s,event=None):
        self.n_call+=1;r=saved[s['t']];return r['requested_plan'],r['reason'],dict(r['meta'],instrumentation_replay=True)
runner.TaskConditionedEnv=MeasuredEnv;runner.RecordedLLM=SavedBrain;runner.HeuristicBrain=SavedBrain
runner.run(ARM,SEED)
old=[json.loads(x) for x in (SOURCE/'steps.jsonl').read_text().splitlines()];new=[json.loads(x) for x in (FOLDER/'steps.jsonl').read_text().splitlines()]
assert len(old)==len(new)==600 and old==new,'Instrumentation altered the recorded trajectory'
assert len((FOLDER/'extra_steps.jsonl').read_text().splitlines())==600
(FOLDER/'verification.json').write_text(json.dumps({'arm':ARM,'seed':SEED,'exact_600_transitions_match_original':True,'new_api_calls':0,'new_independent_trials':0,'source':str(SOURCE)},indent=2))
print('INSTRUMENTATION VERIFIED',ARM,SEED)

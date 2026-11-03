"""Replay the *same recorded A plans* in both arms with 0/5-tick delay.
This is open-loop high-level plan playback, not fresh Codex decisions for B.
"""
import argparse, copy, json, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'远端机器文件夹'))
from experiments.feedback_attribution.joint_runtime import JointRuntime,load_actor,task_metrics
parser=argparse.ArgumentParser();parser.add_argument('--recorded-b',action='store_true');parser.add_argument('--seed',type=int,default=975001);args=parser.parse_args()
seed=args.seed;out=ROOT/'research_outputs/latency_manual_20261103'/f'seed_{seed}'
records=[json.loads(s) for s in open(out/'decisions.jsonl',encoding='utf8')]
assert len(records)==30 and [r['observation_tick'] for r in records]==list(range(0,600,20))
torch.set_num_threads(1);actor=load_actor(str(ROOT/'research_outputs/lowlevel_training_final/actor_selected.pt'))
rts={k:JointRuntime(seed=seed,regime='normal',actor=actor) for k in ('A','B')}
for record in records:
 t=record['observation_tick']
 for k,rt in rts.items():
  p=record['plans'][k] if args.recorded_b else record['plans']['A']
  assert rt.env.t==t
  if k=='B' and t:rt.run_to(t+5,brain=None)
  rt.apply_joint_plan(copy.deepcopy(p));rt.run_to(t+20,brain=None)
result={k:{**task_metrics(rt.env),'cov_mean':float(np.mean(rt.cov_hist)), 'aoi_mean':float(np.mean(rt.aoi_hist)), 'steps_run':len(rt.cov_hist), 'charge_rewrites':rt.n_rewrites_charge} for k,rt in rts.items()}
original=json.load(open(out/'result.json'))
for key in ('burned_cells','burning_cells','extinguished_cells','contained','explored','coverage','mean_aoi','cov_mean','aoi_mean'):
 if abs(result['A'][key]-original['A'][key])>1e-8:raise RuntimeError(f'A replay mismatch {key}: {result["A"][key]} vs {original["A"][key]}')
 if args.recorded_b and abs(result['B'][key]-original['B'][key])>1e-8:raise RuntimeError(f'B replay mismatch {key}: {result["B"][key]} vs {original["B"][key]}')
description='Both manually submitted plan sequences replayed.' if args.recorded_b else 'Recorded A plans applied to both arms, not an on-policy B planner; action schedule is fixed even when B state diverges.'
filename='recorded_plan_replay.json' if args.recorded_b else 'matched_plan_replay.json'
(out/filename).write_text(json.dumps({'description':description, 'same_seed':seed,'delay_steps':5,'verified_A_replay':True,'verified_B_replay':bool(args.recorded_b),'results':result},ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False))

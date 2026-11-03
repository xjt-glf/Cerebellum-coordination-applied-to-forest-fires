"""Replay a saved agent episode without any model/API call; verify every transition."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/brain_comparison'))
import run as runner
SEED=int(sys.argv[1]);SOURCE=ROOT/'research_outputs/codex_brain_3runs_20261022'/f'codex_{SEED}'
runner.OUT=SOURCE.parent/'replay_verification'
class SavedCommands:
    def __init__(self,*args,**kwargs):self.n_call=0;self.n_fail=0
    def plan(self,state,event=None):
        self.n_call+=1;command=json.loads((SOURCE/f'command_{state["t"]:04d}.json').read_text());assert command['seed']==SEED
        return command['plan'],[command['reason']],{'source':'codex-agent','external_api_calls':0,'replay_only':True}
runner.RecordedLLM=SavedCommands
runner.run('codex',SEED)
replay=runner.OUT/f'codex_{SEED}'
old=[json.loads(x) for x in (SOURCE/'steps.jsonl').read_text().splitlines()]
new=[json.loads(x) for x in (replay/'steps.jsonl').read_text().splitlines()]
assert len(old)==len(new)==600 and old==new,'Full trajectory differs'
assert json.loads((SOURCE/'summary.json').read_text())['initial_hash']==json.loads((replay/'summary.json').read_text())['initial_hash']
(replay/'verification.json').write_text(json.dumps({'seed':SEED,'all_600_transitions_identical':True,'actions_orders_metrics_positions_energy_aoi_identical':True,'external_api_calls':0,'not_a_new_policy_trial':True},indent=2))
print('REPLAY VERIFIED',SEED)

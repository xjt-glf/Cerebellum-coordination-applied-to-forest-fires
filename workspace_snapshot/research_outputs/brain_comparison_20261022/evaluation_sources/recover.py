"""After API access is restored: replay paid, pre-failure decisions, then resume API.
Creates a separate pure-API run; never edits the original mixed-system evidence.
Usage: python experiments/brain_comparison/recover.py 730000
"""
import sys,json,gzip,shutil,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run as runner
SOURCE=runner.OUT
runner.OUT=runner.ROOT/'research_outputs/brain_comparison_20261022_recovered'
seed=int(sys.argv[1]);source=SOURCE/f'llm_{seed}'
records=[json.loads(x) for x in (source/'decisions.jsonl').read_text().splitlines()]
cutoff=next((r['t'] for r in records if r['meta'].get('source')!='llm'),600)
BaseLLM=runner.RecordedLLM
class ReplayingLLM(BaseLLM):
    def _call(self,user_text):
        if self.tick<cutoff:
            with gzip.open(source/f'call_{self.tick:04d}_request.json.gz','rt') as f:old=json.load(f)
            assert old['messages'][1]['content']==user_text,f'Non-identical reconstructed state at {self.tick}'
            with gzip.open(source/f'call_{self.tick:04d}_response.json.gz','rt') as f:obj=json.load(f)
            self.last_usage=obj.get('usage',{})
            for suffix in ['request','response']:shutil.copy2(source/f'call_{self.tick:04d}_{suffix}.json.gz',self.path/f'call_{self.tick:04d}_{suffix}.json.gz')
            runner.append(self.path/'replayed_calls.jsonl',{'t':self.tick,'source':str(source),'usage_already_paid':self.last_usage})
            return obj['choices'][0]['message']['content']
        return super()._call(user_text)
    def plan(self,s,event=None):
        result=super().plan(s,event)
        if result[2].get('source')!='llm':raise RuntimeError('API unavailable: fail-closed recovery; no heuristic decision applied. '+result[2].get('error',''))
        return result
runner.RecordedLLM=ReplayingLLM
try:
    runner.run('llm',seed)
except Exception:
    import traceback
    path=runner.OUT/f'llm_{seed}';path.mkdir(parents=True,exist_ok=True);(path/'FAILED.txt').write_text(traceback.format_exc());raise
new=runner.OUT/f'llm_{seed}'
original_steps=[json.loads(x) for x in (source/'steps.jsonl').read_text().splitlines()]
new_steps=[json.loads(x) for x in (new/'steps.jsonl').read_text().splitlines()]
assert original_steps[:cutoff]==new_steps[:cutoff],'Paid prefix trajectory replay mismatch'
(new/'recovery_audit.json').write_text(json.dumps({'source':str(source),'replayed_steps':cutoff,'prefix_trajectory_identical':True,'live_api_from_t':cutoff},indent=2))
print('Recovery completed and paid-prefix trajectory verified',seed,cutoff)

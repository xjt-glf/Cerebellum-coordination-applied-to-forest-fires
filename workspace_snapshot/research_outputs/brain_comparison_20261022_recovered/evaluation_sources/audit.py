"""Delivery integrity checks, including credential exclusion; emits no credentials."""
import json,gzip,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from webapp.server import RunCfg
OUT=ROOT/'research_outputs/brain_comparison_20261022'
def lines(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def main():
    checks=[];hashes={};api_sources={}
    for seed in range(730000,730010):
        for arm in ['heuristic','llm']:
            path=OUT/f'{arm}_{seed}';summary=json.loads((path/'summary.json').read_text());dec=lines(path/'decisions.jsonl');steps=lines(path/'steps.jsonl')
            assert len(dec)==30 and [r['t'] for r in dec]==list(range(0,600,20)),(arm,seed,'decision times')
            assert len(steps)==600 and [r['metrics']['t'] for r in steps]==list(range(1,601)),(arm,seed,'steps')
            assert summary['steps']==600
            assert all(len(r['actions'])==4 and all(0<=a<9 for a in r['actions']) for r in steps)
            assert all(len(r['plan']['uavs'])==4 for r in dec)
            assert summary['burned_cells']==steps[-1]['metrics']['burned_cells']
            hashes[arm,seed]=summary['initial_hash']
            api_sources[f'{arm}_{seed}']={source:sum(r['meta'].get('source')==source for r in dec) for source in set(r['meta'].get('source') for r in dec)}
            if arm=='llm':
                for r in dec:
                    with gzip.open(path/f'call_{r["t"]:04d}_request.json.gz','rt') as f:request=json.load(f)
                    assert request['model']=='qwen-plus' and len(request['messages'])==2
                    assert '完整环境信息' in request['messages'][1]['content']
                    if r['meta'].get('source')=='llm':assert (path/f'call_{r["t"]:04d}_response.json.gz').exists()
            checks.append(f'{arm}_{seed}: 600 steps, 30 decisions, valid actions and plan sizes')
        assert hashes['heuristic',seed]==hashes['llm',seed]
    key=RunCfg().api_key.encode();checked=0
    for path in OUT.rglob('*'):
        if not path.is_file():continue
        data=gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes()
        assert not key or key not in data,f'Credential in {path.name}'
        checked+=1
    result={'status':'passed','episodes':20,'paired_seeds':10,'total_environment_steps':12000,'total_planning_decisions':600,'initial_fingerprints_match':True,'credential_scan_files':checked,'credential_found':False,'sources':api_sources,'checks':checks}
    (OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print('Audit passed:',len(checks),'episodes;',checked,'files scanned; no credentials.')
if __name__=='__main__':main()

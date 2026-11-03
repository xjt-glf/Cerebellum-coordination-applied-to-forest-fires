"""Development-only normalized reward comparison, fixed baseline, model export."""
import json,time
import torch
from .environment import ROOT,file_hash
from .evaluate import OUT,execute,evaluation_jobs,paired_report

def load(p): return [json.loads(s) for s in p.read_text().splitlines() if s] if p.exists() else []

def main():
    torch.set_num_threads(1)
    fixed=execute(evaluation_jobs('fixed',[3,4,5,6],4,986000,'dev_fixed'),OUT/'dev_fixed.jsonl',8)
    norm=OUT/'v2_normalized_s2103'
    while not (norm/'checkpoint_00128.pt').exists(): time.sleep(30)
    ck=norm/'checkpoint_00128.pt'
    jobs=evaluation_jobs('mappo',[3,4,5,6],8,986000,'dev_v2_normalized_128',ck)
    ms=execute(jobs,OUT/'dev_v2_normalized_128.jsonl',12)
    count=OUT/'v1_count_s1103/checkpoint_00128.pt'
    refs=execute(evaluation_jobs('mappo',[3,4,5,6],8,986000,'dev_v1_count_128',count),OUT/'dev_v1_count_128.jsonl',12)
    report=paired_report(ms,refs)
    (OUT/'normalized_reward_development.json').write_text(json.dumps(dict(report=report,v1_checkpoint=str(count),v2_checkpoint=str(ck),note='Development only. Different optimizer/initialization seeds 1103 vs2103 confound causal reward comparison; same 128 scene seeds and episode counts. No heldout used. Count version remains predeclared main experiment.',v2_sha256=file_hash(ck)),indent=2))
    while not (OUT/'frozen_manifest.json').exists(): time.sleep(30)
    manifest=json.loads((OUT/'frozen_manifest.json').read_text())
    exports={}
    for seed,rec in manifest['selected'].items():
        checkpoint=torch.load(rec['checkpoint'],map_location='cpu',weights_only=False)
        directory=OUT/'deployment'/f's{seed}';directory.mkdir(parents=True,exist_ok=True)
        torch.save(checkpoint['actor'],directory/'actor_selected.pt');torch.save(checkpoint['critic'],directory/'critic_selected.pt')
        exports[seed]=dict(actor=str(directory/'actor_selected.pt'),actor_sha256=file_hash(directory/'actor_selected.pt'),critic_sha256=file_hash(directory/'critic_selected.pt'),source_checkpoint=rec['checkpoint'])
    (OUT/'deployment_exports.json').write_text(json.dumps(exports,indent=2))
    print('DEVELOPMENT_EXTRAS_FINISHED',flush=True)

if __name__=='__main__': main()

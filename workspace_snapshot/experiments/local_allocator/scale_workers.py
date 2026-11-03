"""Recover existing learners at complete batch boundaries with 16 CPU workers."""
import datetime,json,os,signal,subprocess,sys,time
from .evaluate import OUT
from .environment import file_hash

def main():
    expected=OUT/'recovery_probe_fixed_s1103/checkpoint_00032.pt'
    while not expected.exists(): time.sleep(10)
    import torch
    old=torch.load(OUT/'v1_count_s1103/checkpoint_00032.pt',map_location='cpu',weights_only=False)
    probe=torch.load(expected,map_location='cpu',weights_only=False)
    diffs={name:max(float((old[name][k]-probe[name][k]).abs().max()) for k in old[name]) for name in ('actor','critic')}
    result=dict(exact=all(v==0 for v in diffs.values()),max_abs_difference=diffs,
        reference_sha256=file_hash(OUT/'v1_count_s1103/checkpoint_00032.pt'),recovered_sha256=file_hash(expected),
        note='Resumed from episode16, repeated next 16 complete scenes with 4 workers versus original8; compared actor and critic after PPO update32. CUDA RNG restored as CPU ByteTensor.')
    (OUT/'recovery_verification.json').write_text(json.dumps(result,indent=2))
    if not result['exact']: raise RuntimeError(f'Recovery differs: {diffs}')
    records=[]
    for seed in (1103,1104,1105):
        run=OUT/f'v1_count_s{seed}'
        pid=json.loads((run/'pid.json').read_text())['pid']
        (run/'config_initial_8workers.json').write_text((run/'config.json').read_text())
        records.append(dict(seed=seed,old_pid=pid))
        try: os.kill(pid,signal.SIGINT)
        except ProcessLookupError: pass
    # SIGINT exits after the executor finishes pending episodes. Latest checkpoint
    # remains a COMPLETE batch; unconsumed interrupted rollout is replayed.
    for rec in records:
        start=time.monotonic()
        while os.path.exists(f'/proc/{rec["old_pid"]}'):
            if time.monotonic()-start>180: raise RuntimeError('Old learner did not exit, will not overlap writers')
            time.sleep(5)
    deadline=datetime.datetime(2026,11,3,21,51).timestamp()
    children=[]
    for index,rec in enumerate(records):
        run=OUT/f'v1_count_s{rec["seed"]}'
        args=[sys.executable,'-u','-m','experiments.local_allocator.train','--seed',str(rec['seed']),'--index',str(index),'--workers','16','--batch-episodes','16','--seconds',str(max(1,int(deadline-time.time()))),'--device',f'cuda:{index}','--resume',str(run/'latest.pt')]
        stream=open(run/'resumed_training.log','a')
        proc=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
        rec.update(new_pid=proc.pid,command=args,checkpoint=str(run/'latest.pt'),resumed_at=str(datetime.datetime.now()))
        children.append((proc,stream))
    (OUT/'worker_scaling.json').write_text(json.dumps(records,indent=2))
    for proc,stream in children: proc.wait();stream.close()
    print('SCALED_TRAINING_FINISHED',flush=True)

if __name__=='__main__': main()

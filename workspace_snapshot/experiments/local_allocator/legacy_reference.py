"""Original patrol-capable heuristic: explicitly an unequal-interface reference."""
import concurrent.futures as cf,gzip,json,multiprocessing as mp,time
import numpy as np
import torch
from .environment import frozen_actor
from .evaluate import OUT
from .runner import append_json
from experiments.feedback_attribution.joint_runtime import JointRuntime
from lowlevel.mission_demo import brain

def reference(job):
    torch.set_num_threads(1);seed,n=job
    rt=JointRuntime(seed,cfg={'uavs':n},actor=frozen_actor())
    physics=[];original=rt.env.step;covered=[]
    def measured(moves):
        result=original(moves)
        e=rt.env.env
        covered.append(float(np.mean([r.is_covered for r in e.rescuers])))
        physics.append(dict(t=rt.env.t,uavs=[[float(u.x),float(u.y),float(u.e)] for u in e.uavs],persons=[[int(r.rescuer_id),float(r.x),float(r.y),float(r.AoI)] for r in e.rescuers],orders=[o.to_dict() for o in rt.env.orders]))
        return result
    rt.env.step=measured
    start=time.monotonic();metrics=rt.run_to(600,brain=lambda situation:brain(situation['sense'])[0])
    metrics.pop('new_events',None)
    metrics.update(seed=seed,n=n,total_cov_mean=float(np.mean(covered)),seconds=time.monotonic()-start,reference_only=True,task_interface='original H supports patrol; original GoalProtocol; NOT comparable main three-action target protocol')
    directory=OUT/'legacy_reference';directory.mkdir(exist_ok=True)
    with gzip.open(directory/f'n{n}_s{seed}.jsonl.gz','wt') as f:
        f.write(json.dumps(dict(decisions=rt.decisions))+'\n')
        for row in physics:f.write(json.dumps(row)+'\n')
    return metrics

def main():
    jobs=[(986000+(n-3)*40+j,n) for n in range(3,7) for j in range(4)]
    with cf.ProcessPoolExecutor(8,mp_context=mp.get_context('spawn')) as pool:
        for f in cf.as_completed([pool.submit(reference,j) for j in jobs]):append_json(OUT/'legacy_reference.jsonl',f.result())
    print('LEGACY_REFERENCE_FINISHED',flush=True)

if __name__=='__main__':main()

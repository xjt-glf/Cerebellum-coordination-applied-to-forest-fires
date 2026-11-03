"""Interactive agent brain: every 20 steps requires an explicit human/agent JSON file.
No heuristic planning, no external LLM API, no rollout search, no action overrides.
Spatial summaries describe current state only; the agent selects every plan.
"""
import os,sys,json,time,math,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/brain_comparison'))
import numpy as np
import run as runner
runner.OUT=ROOT/'research_outputs/codex_brain_3runs_20261022'
SEED=int(sys.argv[1]);FOLDER=runner.OUT/f'codex_{SEED}'
def compact(s):
    full=s['full_environment'];e=None
    # Decode observed map only to count unknowns by 20x20 tiles. No candidate scoring.
    m=full['maps']['team_known_fire']['rows_x'];grid=np.array([list(r) for r in m]);unknown=grid=='?'
    tiles=[[int(unknown[x:x+20,y:y+20].sum()) for y in range(0,100,20)] for x in range(0,100,20)]
    true=full['maps']['true_fire']['rows_x'];tg=np.array([list(r) for r in true]);known=grid!='?'
    clusters=[]
    for c in full['true_fire_clusters']:
        cells=c['boundary_cells'];unseen=sum(grid[int(x),int(y)]=='?' for x,y in cells)
        clusters.append({'xy':[round(float(v),1) for v in c['centroid']],'size':c['size'],'contained':c['contained'],'unknown_boundary':unseen})
    us=[]
    for u in full['uavs']:
        gp=u['goal_progress'] or {};order=u['current_order'];kind=order['mode'] if order['task']=='fire_sensing' else order['task']
        us.append({'id':u['id'],'xy':u['position'],'E':u['energy'],'task':kind,'target':u['resolved_target'],'d':round(u['distance_to_target'] or 0,1),'done':gp.get('completed'),'arrived':gp.get('arrived'),'roi':round(gp.get('own_region_completion',0),2),'link':u['personnel_in_own_link']})
    people=[{'id':r['id'],'xy':[round(float(v),1) for v in r['position']],'aoi':r['aoi'],'link':r['uav_links'],'status':r['status']} for r in full['personnel']]
    return {'seed':SEED,'t':s['t'],'metrics':{k:s[k] for k in ['explored','discovery','rescuer_known','coverage','burned']},'true_clusters':clusters,'known_clusters':s['clusters'],'uavs':us,'people':people,'unknown_tiles_x_rows_y_cols_centers_10_30_50_70_90':tiles,'wind':{k:v for k,v in full['rules'].items() if 'wind' in k}}
class InteractiveBrain:
    def __init__(self,*args,**kwargs):self.n_call=0;self.n_fail=0
    def plan(self,s,event=None):
        self.n_call+=1;t=s['t'];start=time.monotonic();data=compact(s)
        runner.dump(FOLDER/'waiting.json',data)
        cmd=FOLDER/f'command_{t:04d}.json'
        while not cmd.exists():
            if time.monotonic()-start>3600:raise TimeoutError('No explicit agent decision received')
            time.sleep(.15)
        command=json.loads(cmd.read_text());assert command['seed']==SEED and command['t']==t
        assert len(command['plan']['uavs'])==4
        return command['plan'],[command['reason']],{'source':'codex-agent','external_api_calls':0,'tool_assisted':True,'decision_mode':'explicit per-boundary agent JSON; no automatic planning'}
runner.RecordedLLM=InteractiveBrain
if __name__=='__main__':
    try:runner.run('codex',SEED)
    except Exception:
        FOLDER.mkdir(parents=True,exist_ok=True);(FOLDER/'FAILED.txt').write_text(traceback.format_exc());raise

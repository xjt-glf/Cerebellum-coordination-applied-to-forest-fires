"""Explicit interactive Codex short-route experiment. No API, no training."""
import sys,time,json,hashlib,traceback,copy,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/brain_comparison'))
import run as base
import numpy as np,torch
from lowlevel.env_wrapper import TaskConditionedEnv
from lowlevel.mission_demo import sense_state
from webapp.world_snapshot import full_snapshot
from webapp.charge_protocol import ChargeProtocol
from webapp.route_converter import ShortRouteConverter
SEED=int(sys.argv[1]);MODE=sys.argv[2] if len(sys.argv)>2 else 'online'
OUT=ROOT/'research_outputs/codex_short_routes_3runs_20261022';FOLDER=OUT/(f'codex_{SEED}' if MODE=='online' else f'{MODE}_{SEED}')
class MeasuredEnv(TaskConditionedEnv):
    """Read-only instrumentation for saved-command replays."""
    def step(self,actions):
        e=self.env;prev=np.array([[u.x,u.y] for u in e.uavs])
        result=super().step(actions)
        bs=e.grid['burning_state'];burn=(bs>0)&(bs<2);active=e._active_true_burning_mask(bs)
        team=e.team_map_info;person=e.rescuers_map_info;tb=(team>0)&(team<2);pb=(person>0)&(person<2)
        pos=np.array([[u.x,u.y] for u in e.uavs]);energy=np.array([u.e for u in e.uavs]);cost=np.array([e.energy_cost_per_step(i) for i in range(4)])
        padsteps=np.array([min(max(0,math.ceil(max(abs(u.x-st[0]),abs(u.y-st[1]))-e.charge_radius)) for st in e.charging_stations) for u in e.uavs])
        base.append(FOLDER/'extra_steps.jsonl',{'t':self.t,'true_burning':int(burn.sum()),'true_active_burning':int(active.sum()),'team_tp':int((tb&burn).sum()),'person_tp':int((pb&burn).sum()),'active_team_tp':int((tb&active).sum()),'active_person_tp':int((pb&active).sum()),'comm_coverage_rate':float(e.comm_coverage_rate),'blocked':[bool(u.blocked) for u in e.uavs],'oob':[bool(u.oob) for u in e.uavs],'at_pad':[bool(e.at_station(u.x,u.y)) for u in e.uavs],'return_margin':(energy-cost*padsteps).tolist(),'displacement':(pos-prev).tolist()})
        return result
def summarize(s,converter):
    f=s['full_environment'];m=np.array([list(r) for r in f['maps']['team_known_fire']['rows_x']]);tile=[[int((m[x:x+20,y:y+20]=='?').sum()) for y in range(0,100,20)] for x in range(0,100,20)]
    return {'seed':SEED,'t':s['t'],'metrics':[s[k] for k in ['explored','discovery','rescuer_known','coverage','burned']],'fires':[[*[round(float(v),1) for v in c['centroid']],c['size'],c['contained']] for c in f['true_fire_clusters']],'U':[[u['id'],*u['position'],u['energy'],u['current_order']['mode'] if u['current_order']['task']=='fire_sensing' else u['current_order']['task'],u['resolved_target'],u['personnel_in_own_link'],u['blocked']] for u in f['uavs']],'P':[[p['id'],*[round(float(v),1) for v in p['position']],p['aoi'],p['uav_links']] for p in f['personnel']],'unknown_tiles':tile,'route_feedback':converter.feedback()}
def run():
    torch.set_num_threads(1);FOLDER.mkdir(parents=True,exist_ok=True)
    if (FOLDER/'summary.json').exists():return
    Env=TaskConditionedEnv if MODE=='online' else MeasuredEnv
    env=Env(num_uavs=4,max_steps=600,spread_steps=40,grid_size=(100,100),n_fire_seeds=3,n_fire_seeds_range=None,randomize_wind=True,spread_gain=1,num_rescuers=10,seed=SEED,task_sampler='planner',replan_every=0,terminate_on_all_depleted=False)
    obs=env.reset();actor=base.load_actor(str(base.ACTOR));goals=base.CommonGoals(env,SEED);charge=ChargeProtocol(env,goals);converter=ShortRouteConverter(env,goals,enabled=MODE!='off')
    initial=full_snapshot(env,goals=goals);base.zipped(FOLDER/'initial_state.json.gz',initial);initial_hash=hashlib.sha256(json.dumps(initial,default=base.clean,sort_keys=True).encode()).hexdigest()
    recent=[];metrics=[];aois=[];blackout=np.zeros(10,int);max_blackout=0;ever=set();last_order=None;changes=0;start=time.monotonic();last_events=0
    for step in range(600):
        if step%20==0:
            state=sense_state(env);state['full_environment']=full_snapshot(env,goals=goals,recent_decisions=recent);base.zipped(FOLDER/f'state_{step:04d}.json.gz',state)
            wait=time.monotonic()
            if MODE=='online':
                base.dump(FOLDER/'waiting.json',summarize(state,converter));path=FOLDER/f'command_{step:04d}.json'
                while not path.exists():
                    if time.monotonic()-wait>3600:raise TimeoutError('Agent command missing')
                    time.sleep(.1)
            else:path=OUT/f'codex_{SEED}'/f'command_{step:04d}.json'
            command=json.loads(path.read_text());assert command['seed']==SEED and command['t']==step
            if MODE=='off':
                # Same open-loop commands, use first point only, no dynamic queue or route validation against diverged positions.
                plan=copy.deepcopy(command['plan'])
                for o in plan['uavs']:
                    if 'route' in o:o['target']=o.pop('route')[0]
                obs,applied,notes=goals.apply(plan,source='codex-agent')
            else:obs,applied,notes=converter.apply(command['plan'])
            row={'t':step,'requested_plan':command['plan'],'plan':applied,'reason':[command['reason']],'meta':{'source':'codex-agent','external_api_calls':0,'route_converter':MODE!='off','fixed_commands_replay':MODE!='online'},'seconds':time.monotonic()-wait,'goal_notes':notes,'brief':{k:v for k,v in state.items() if k!='full_environment'}}
            base.append(FOLDER/'decisions.jsonl',row);recent.append(row)
        new,notes=charge.sync()
        if new is not None:obs=new
        if notes:base.append(FOLDER/'charge_overrides.jsonl',{'t':step,'notes':notes,'orders':[o.to_dict() for o in env.orders]})
        new=converter.tick()
        if new is not None:obs=new
        for event in converter.events[last_events:]:base.append(FOLDER/'route_events.jsonl',event)
        last_events=len(converter.events)
        orders=[o.to_dict() for o in env.orders]
        if last_order is not None:changes+=sum(x!=y for x,y in zip(last_order,orders))
        last_order=orders
        with torch.no_grad():actions=actor.act(torch.as_tensor(obs,dtype=torch.float32),deterministic=True)[0].numpy()
        obs,_,done,info=env.step(actions);goals.update()
        if env.t%20==0:converter.finish_window()
        for event in converter.events[last_events:]:base.append(FOLDER/'route_events.jsonl',event)
        last_events=len(converter.events)
        m=base.task_metrics(env);metrics.append(m);age=[float(p.AoI) for p in env.env.rescuers];aois.extend(age)
        blackout=np.where([r.is_covered for r in env.env.rescuers],0,blackout+1);max_blackout=max(max_blackout,int(blackout.max()));ever.update(i for i,u in enumerate(env.env.uavs) if u.e<=0)
        base.append(FOLDER/'steps.jsonl',{'metrics':m,'actions':actions,'orders':orders,'person_aoi':age,'uavs':[{'id':i,'x':float(u.x),'y':float(u.y),'energy':float(u.e)} for i,u in enumerate(env.env.uavs)]})
        base.dump(FOLDER/'progress.json',{'t':env.t,'external_api_calls':0,'agent_decisions':len(recent)})
        if done and env.t<600:raise RuntimeError('Early stop')
    final=base.task_metrics(env);final.update(seed=SEED,arm='short_routes_'+MODE,steps=env.t,initial_hash=initial_hash,actor_sha256=hashlib.sha256(base.ACTOR.read_bytes()).hexdigest(),mean_coverage=float(np.mean([m['coverage'] for m in metrics])),mean_person_aoi=float(np.mean(aois)),p95_person_aoi=float(np.percentile(aois,95)),max_blackout_steps=max_blackout,depleted_ever=len(ever),instruction_changes=changes,planning_calls=30,external_api_calls=0,wall_seconds=time.monotonic()-start)
    base.dump(FOLDER/'summary.json',final)
    if MODE=='replay':
        original=[json.loads(x) for x in (OUT/f'codex_{SEED}'/'steps.jsonl').read_text().splitlines()];actual=[json.loads(x) for x in (FOLDER/'steps.jsonl').read_text().splitlines()];assert original==actual
        base.dump(FOLDER/'verification.json',{'all_600_steps_exact_match':True,'external_api_calls':0})
if __name__=='__main__':
    try:run()
    except Exception:FOLDER.mkdir(parents=True,exist_ok=True);(FOLDER/'FAILED.txt').write_text(traceback.format_exc());raise

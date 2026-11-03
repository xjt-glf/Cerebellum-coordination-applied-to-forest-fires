"""Short, within-horizon polylines -> existing single-target Actor interface.
Only changes UavOrder targets, never actor actions. Single-waypoint plans keep
GoalProtocol behavior. A route owns its points and is never randomly retargeted.
"""
import copy,math
from lowlevel.task_spec import UavOrder,TASK_FIRE_SENSING,TASK_RECHARGE

def validate_route(points,start,horizon=20,shape=(100,100)):
    if not isinstance(points,list) or not 2<=len(points)<=4:raise ValueError('route requires 2..4 points')
    p=[]
    for q in points:
        if not isinstance(q,(list,tuple)) or len(q)!=2:raise ValueError('point must be [x,y]')
        x,y=map(float,q)
        if not math.isfinite(x+y) or not (1<=x<=shape[0]-2 and 0<=y<=shape[1]-2):raise ValueError('point outside flyable domain')
        if p and math.hypot(x-p[-1][0],y-p[-1][1])<4:raise ValueError('interior points must be separated by >=4 cells')
        p.append((x,y))
    seq=[tuple(map(float,start))]+p
    cost=sum(max(abs(b[0]-a[0]),abs(b[1]-a[1])) for a,b in zip(seq,seq[1:]))
    if cost>horizon+1e-6:raise ValueError(f'route ideal travel {cost:.2f} exceeds horizon {horizon}')
    vectors=[(b[0]-a[0],b[1]-a[1]) for a,b in zip(seq,seq[1:])]
    if not any(abs(a[0]*b[1]-a[1]*b[0])>1e-6 for a,b in zip(vectors,vectors[1:])):raise ValueError('straight path: use a single target instead')
    return p,cost

class ShortRouteConverter:
    def __init__(self,env,goals,horizon=20,enabled=True):
        self.env=env;self.goals=goals;self.horizon=horizon;self.enabled=enabled;self.routes={};self.events=[]
    def event(self,i,kind,**kw):self.events.append(dict(t=self.env.t,uav=i,event=kind,**kw))
    def apply(self,plan,source='codex-agent'):
        ordinary={'uavs':[]};new={}
        for i,raw in enumerate(plan['uavs']):
            q=copy.deepcopy(raw)
            if q.get('task')=='patrol':raise ValueError('patrol disabled in this experiment')
            if 'route' in q:
                if q.get('task')!='explore':raise ValueError('route currently supported only for explore')
                u=self.env.env.uavs[i];points,cost=validate_route(q.pop('route'),(u.x,u.y),self.horizon,self.env.env.grid_size)
                q['target']=list(points[0]);new[i]=dict(points=points,index=0,start=(u.x,u.y),segment_start=(u.x,u.y),created=self.env.t,expires=self.env.t+self.horizon,last_distance=None,stagnant=0,done=False)
                self.event(i,'issued',points=points,ideal_steps=cost,enabled=self.enabled)
            ordinary['uavs'].append(q)
        for i,r in self.routes.items():
            if not r['done']:self.event(i,'superseded',old_index=r['index'])
        self.routes=new if self.enabled else {}
        obs,applied,notes=self.goals.apply(ordinary,source=source)
        # Explicit ownership: force the first route point, including repeated routes.
        orders=copy.deepcopy(self.env.orders);changed=False
        for i in new:
            wanted=UavOrder(task=TASK_FIRE_SENSING,mode=0,target=tuple(ordinary['uavs'][i]['target']))
            if orders[i].to_dict()!=wanted.to_dict():orders[i]=wanted;changed=True
        if changed:
            obs=self.env.set_plan(orders);self.goals.current=copy.deepcopy(orders);self.goals.manager.sync(self.env.env,orders,self.env.t)
            applied={'uavs':[o.to_dict() for o in orders]}
        return obs,applied,notes
    def tick(self):
        orders=copy.deepcopy(self.env.orders);changed=False
        for i,r in self.routes.items():
            if r['done']:continue
            if self.env.t>=r['expires']:
                r['done']=True;self.event(i,'expired',index=r['index']);continue
            if orders[i].task==TASK_RECHARGE:
                r['done']=True;self.event(i,'charge_preempted',index=r['index']);continue
            u=self.env.env.uavs[i];p=r['points'][r['index']];d=math.hypot(u.x-p[0],u.y-p[1])
            r['stagnant']=r['stagnant']+1 if r['last_distance'] is not None and d>=r['last_distance']-.05 else 0
            r['last_distance']=d
            a=r['segment_start'];vx,vy=p[0]-a[0],p[1]-a[1];norm=vx*vx+vy*vy
            along=((u.x-a[0])*vx+(u.y-a[1])*vy)/norm if norm>0 else 0
            cross=abs((u.x-a[0])*vy-(u.y-a[1])*vx)/math.sqrt(norm) if norm>0 else d
            why='arrived' if d<=2.5 else ('passed' if along>=1 and cross<=2.5 else ('near_stall' if d<=4 and r['stagnant']>=4 else None))
            if why is None:continue
            self.event(i,'point_reached',index=r['index'],point=p,distance=d,reason=why)
            if r['index']==len(r['points'])-1:
                r['done']=True;self.event(i,'completed',elapsed=self.env.t-r['created']);continue
            r['index']+=1;r['segment_start']=(u.x,u.y);r['stagnant']=0;r['last_distance']=None
            orders[i]=UavOrder(task=TASK_FIRE_SENSING,mode=0,target=r['points'][r['index']]);changed=True
            self.event(i,'target_switched',index=r['index'],target=r['points'][r['index']])
        if not changed:return None
        obs=self.env.set_plan(orders);self.goals.current=copy.deepcopy(orders);self.goals.manager.sync(self.env.env,orders,self.env.t)
        return obs
    def finish_window(self):
        # The 20th action may reach the final point immediately before replanning.
        for i,r in self.routes.items():
            if r['done'] or self.env.t<r['expires']:continue
            u=self.env.env.uavs[i];p=r['points'][r['index']];d=math.hypot(u.x-p[0],u.y-p[1])
            r['done']=True
            if self.env.orders[i].task==TASK_RECHARGE:self.event(i,'charge_preempted',index=r['index'])
            elif r['index']==len(r['points'])-1 and d<=2.5:
                self.event(i,'point_reached',index=r['index'],point=p,distance=d,reason='arrived_at_deadline');self.event(i,'completed',elapsed=self.env.t-r['created'])
            else:self.event(i,'expired',index=r['index'])
    def feedback(self):
        return {i:{'index':r['index'],'count':len(r['points']),'done':r['done'],'expires':r['expires'],'remaining':r['points'][r['index']:]} for i,r in self.routes.items()}

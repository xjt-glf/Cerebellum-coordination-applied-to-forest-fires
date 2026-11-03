"""Matched-seed descriptive analysis; no planning, training or API calls."""
import json,csv,gzip,hashlib,collections,sys,html,base64
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from webapp.route_converter import validate_route
OUT=ROOT/'research_outputs/codex_short_routes_3runs_20261022';SEEDS=[730000,730001,730002]
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
def writecsv(p,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def measured(mode,s):
    f=OUT/f'{mode}_{s}';raw=lines(f/'steps.jsonl');extra=lines(f/'extra_steps.jsonl');su=read(f/'summary.json')
    assert len(raw)==len(extra)==600
    m={k:np.array([r['metrics'][k] for r in raw]) for k in raw[0]['metrics']};x={k:np.array([r[k] for r in extra]) for k in extra[0]}
    aoi=np.array([r['person_aoi'] for r in raw]);energy=np.array([[u['energy'] for u in r['uavs']] for r in raw]);disp=x['displacement'];moving=np.linalg.norm(disp,axis=2)>0
    active=x['true_active_burning'];idx=np.flatnonzero(active>0);ct=600 if active[-1]>0 else int(idx[-1])+2 if len(idx) else 1
    initial=305 if s==730000 else 299
    def ratio(a,b):return float(a/b) if b else None
    roles=collections.Counter(o['mode'] if o['task']=='fire_sensing' else o['task'] for r in raw for o in r['orders'])
    r=dict(arm='routes' if mode=='replay' else 'first_only',seed=s,affected=int(m['burned_cells'][-1]),new_affected=int(m['burned_cells'][-1])-initial,extinguished=int(m['extinguished_cells'][-1]),burning_final=int(x['true_burning'][-1]),active_final=int(active[-1]),active_auc=int(active.sum()),burning_auc=int(x['true_burning'].sum()),contain_time=ct,explore_final=float(m['explored'][-1]),explore_mean=float(m['explored'].mean()),team_recall=ratio(x['team_tp'].sum(),x['true_burning'].sum()),person_recall=ratio(x['person_tp'].sum(),x['true_burning'].sum()),active_team_recall=ratio(x['active_team_tp'].sum(),active.sum()),active_person_recall=ratio(x['active_person_tp'].sum(),active.sum()),uav_coverage=float(m['coverage'].mean()),comm_coverage=float(x['comm_coverage_rate'].mean()),aoi_mean=float(aoi.mean()),aoi_p95=float(np.percentile(aoi,95)),aoi_p99=float(np.percentile(aoi,99)),aoi_over20=float((aoi>20).mean()),blackout_max=su['max_blackout_steps'],blocked_rate=float(x['blocked'].mean()),oob_rate=float(x['oob'].mean()),path_length=float(np.linalg.norm(disp,axis=2).sum()),move_rate=float(moving.mean()),instruction_changes=su['instruction_changes'],min_energy=float(energy.min()),min_margin=float(x['return_margin'].min()),unsafe_margin_rate=float((x['return_margin']<0).mean()),depleted_ever=su['depleted_ever'],role_explore=roles['explore']/2400,role_patrol=roles['patrol']/2400,role_relay=roles['comm_relay']/2400,role_charge=roles['recharge']/2400)
    r['controlled_final']=int(active[-1]==0)
    r['burnout']=r['affected']-r['extinguished']-r['burning_final'];consecutive=moving[:-1]&moving[1:];r['reverse_rate']=ratio((consecutive&np.all(disp[1:]==-disp[:-1],axis=2)).sum(),consecutive.sum())
    return r,m,x
METRICS=[('affected','累计过火格数 ↓','格'),('new_affected','新增过火格数 ↓','格'),('active_auc','未合围燃烧暴露积分 ↓','格·步'),('contain_time','稳定合围耗时 ↓','步'),('burnout','最终烧尽格数 ↓','格'),('extinguished','最终扑灭格数','格'),('burning_final','最终余火格数 ↓','格'),('explore_final','最终探索率 ↑','%'),('explore_mean','全程平均探索率 ↑','%'),('active_person_recall','未合围火人员认知率 ↑','%'),('person_recall','全部燃烧人员认知率 ↑','%'),('uav_coverage','平均UAV通信覆盖 ↑','%'),('comm_coverage','含基站平均通信可用率 ↑','%'),('aoi_mean','平均人员AoI ↓','步'),('aoi_p95','人员AoI P95 ↓','步'),('aoi_p99','人员AoI P99 ↓','步'),('aoi_over20','AoI>20人员时间占比 ↓','%'),('blackout_max','最长连续中断 ↓','步'),('blocked_rate','被其他无人机阻挡占比 ↓','%'),('oob_rate','越界尝试占比 ↓','%'),('reverse_rate','相邻移动立即折返率','%'),('path_length','四机总实际航程','格'),('instruction_changes','实际逐机指令变化数','次'),('min_energy','全程最低电量','电量单位'),('min_margin','最低理想返航余量','电量单位'),('depleted_ever','曾耗尽架数 ↓','架'),('role_explore','探索任务时间占比','%'),('role_relay','中继任务时间占比','%'),('role_charge','充电任务时间占比','%')]
def fmt(v,unit):return f'{v*100:.2f}%' if unit=='%' else f'{v:,.2f}'
def main():
    METRICS.extend([('active_final','终局未合围燃烧格数 ↓','格'),('controlled_final','终局无未合围火比例 ↑','%')])
    old=read(ROOT/'research_outputs/comprehensive_comparison_3seeds_20261022/analysis.json')['records'];records=[dict(r,arm='single') for r in old if r['arm']=='codex'];traces={};audit=[];routed=[];route_stats=[]
    actor=hashlib.sha256((ROOT/'research_outputs/lowlevel_training_final/actor_selected.pt').read_bytes()).hexdigest()
    for s in SEEDS:
        online=OUT/f'codex_{s}';raw=lines(online/'steps.jsonl');dec=lines(online/'decisions.jsonl');events=lines(online/'route_events.jsonl');su=read(online/'summary.json');oldsu=read(ROOT/f'research_outputs/codex_brain_3runs_20261022/codex_{s}/summary.json')
        assert len(raw)==600 and len(dec)==30 and len(list(online.glob('command_*.json')))==30
        assert read(OUT/f'replay_{s}/verification.json')['all_600_steps_exact_match']
        assert actor==su['actor_sha256']==oldsu['actor_sha256']
        assert su['initial_hash']==oldsu['initial_hash']==read(OUT/f'off_{s}/summary.json')['initial_hash']
        for mode in ['replay','off']:
            r,m,x=measured(mode,s);records.append(r);traces[r['arm'],s]=(m,x)
        with gzip.open(online/'initial_state.json.gz','rt') as f:initial=json.load(f)
        grouped={};current={}
        for e in events:
            if e['event']=='issued':current[e['uav']]=(e['t'],e['uav']);grouped[current[e['uav']]]=[]
            elif e['uav'] in current:grouped[current[e['uav']]].append(e)
        for d in dec:
            start=d['t'];cmd=read(online/f'command_{start:04d}.json');assert cmd['plan']==d['requested_plan']
            for i,o in enumerate(cmd['plan']['uavs']):
                assert o['task']!='patrol'
                if 'route' not in o:continue
                u=initial['uavs'][i]['position'] if start==0 else [raw[start-1]['uavs'][i][k] for k in ['x','y']]
                _,cost=validate_route(o['route'],u)
                ev=grouped[start,i]
                complete=next((e for e in ev if e['event']=='completed'),None);end=raw[start+19]['uavs'][i];end_distance=float(np.linalg.norm(np.array([end['x'],end['y']])-o['route'][-1]))
                pts=[e for e in ev if e['event']=='point_reached']
                routed.append(dict(seed=s,t=start,uav=i,points=len(o['route']),ideal_steps=cost,completed=complete is not None,elapsed=complete['elapsed'] if complete else None,terminal_hold_steps=20-complete['elapsed'] if complete else 0,all_points_strict_arrival=len(pts)==len(o['route']) and all(e['distance']<=2.5 for e in pts),final_distance=end_distance,switches=sum(e['event']=='target_switched' for e in ev),near_stall_points=sum(e.get('reason')=='near_stall' for e in pts)))
        rs=[r for r in routed if r['seed']==s];route_stats.append(dict(seed=s,routes=len(rs),completed=sum(r['completed'] for r in rs),all_points_strict=sum(r['all_points_strict_arrival'] for r in rs),switches=sum(r['switches'] for r in rs),near_stall_points=sum(r['near_stall_points'] for r in rs),terminal_hold_steps=sum(r['terminal_hold_steps'] for r in rs),mean_ideal_steps=float(np.mean([r['ideal_steps'] for r in rs])),max_ideal_steps=max(r['ideal_steps'] for r in rs)))
        assert all(o['mode']!='patrol' for r in raw for o in r['orders'] if o['task']=='fire_sensing')
        assert all(0<=a<=8 for r in raw for a in r['actions'])
        audit.append(dict(seed=s,steps=600,agent_plans=30,initial_state_matches_old=True,actor_sha256=actor,replay_exact=True,api_calls=0))
    arms=['single','routes','first_only'];means={a:{k:float(np.mean([r[k] for r in records if r['arm']==a])) for k,_,_ in METRICS} for a in arms}
    writecsv(OUT/'metrics_by_episode.csv',[{k:r[k] for k in ['arm','seed']+[v[0] for v in METRICS]} for r in records]);writecsv(OUT/'route_details.csv',routed);writecsv(OUT/'route_summary.csv',route_stats)
    paired=[]
    for s in SEEDS:
        for b in ['single','first_only']:
            n=next(r for r in records if r['seed']==s and r['arm']=='routes');o=next(r for r in records if r['seed']==s and r['arm']==b)
            for k,_,_ in METRICS:paired.append(dict(seed=s,baseline=b,metric=k,routes=n[k],baseline_value=o[k],difference=n[k]-o[k]))
    writecsv(OUT/'paired_differences.csv',paired)
    (OUT/'analysis.json').write_text(json.dumps(dict(records=records,means=means,route_stats=route_stats),ensure_ascii=False,indent=2));(OUT/'audit.json').write_text(json.dumps(dict(checks=audit,online_trials=3,replay_checks=3,fixed_command_ablations=3,new_api_calls=0,valid_routes=len(routed)),indent=2))
    plots(records,traces,routed)
    n,b,f=[means[a] for a in arms];pct=(n['affected']/b['affected']-1)*100;offpct=(n['affected']/f['affected']-1)*100
    text=['# 冻结Actor短航迹转化器：三局对比实验','',f'**结论：接口可用，但尚未证明明显、稳定的任务提升。** 与上一轮由我规划的单航点三局相比，平均过火从{b["affected"]:.1f}降至{n["affected"]:.1f}格（{pct:+.1f}%）；逐局是两局略差、一局改善。不能把这一均值变化全部归因于转化器。','',f'在保存同一批指令、关闭自动切点且只执行首点的重放中，开启航迹后的平均过火变化为{offpct:+.1f}%。这一对照用于检查当前这批航迹的贡献，不是另一个会适应新状态的大脑。','','## 实验设置','','- 我在每次收到当前状态后亲自给出指令：3个种子730000–730002，每局600步，每20步决策，共90条显式计划；没有用启发式替代我规划。','- 使用现有actor_selected.pt，冻结权重、argmax执行，不重新训练，不调用千问API。初态哈希与此前三局一致，公共目标管理及充电保护保持一致。','- 每次保存全量100×100环境快照；我读取当前火区、人员、无人机、电量、5×5未知区计数与上一段执行反馈。没有查询未来轨迹或搜索候选计划收益。','- 直线任务继续输出一个航点，可持续多个窗口；仅局部转弯输出2–4点折线。折线含当前位置的各段切比雪夫距离之和不得超过20，连续点间距至少4格，纯共线序列拒绝。该预算是理想航程下界，不保证Actor或避碰必定20步走完。','- 转化器每个底层步检查目标：距离≤2.5切点，也允许通过目标垂线且横向偏差≤2.5；距离≤4且连续4次无进展可切点，单独记录。末点完成后本窗口保持，充电可抢占，窗口结束不自动跨窗继续。','- 航迹是稀疏折线提示，没有训练曲率/速度跟踪；2.5格容差会切弯，不能称为精确平滑曲线跟踪。','- 本次仅把新接口接入实验运行器，未修改现有前端和千问提示词。','','## 逐局结果','','| 种子 | 旧单航点过火 | 新航迹过火 | 新−旧 | 仅首点重放过火 | 旧探索率 | 新探索率 |','|---|---:|---:|---:|---:|---:|---:|']
    for s in SEEDS:
        o,nw,fo=[next(r for r in records if r['seed']==s and r['arm']==a) for a in arms];text.append(f'| {s} | {o["affected"]} | {nw["affected"]} | {nw["affected"]-o["affected"]:+} | {fo["affected"]} | {o["explore_final"]:.2%} | {nw["explore_final"]:.2%} |')
    text+=['','## 多指标均值','','每局先计算指标，再对三局等权平均。↓越低越好；未标方向的量只描述行为，不做越大越好的假设。','','| 指标 | 旧单航点（自适应） | 新航迹（自适应） | 同新指令仅首点（固定重放） |','|---|---:|---:|---:|']
    for k,label,unit in METRICS:text.append('| '+label+' | '+' | '.join(fmt(means[a][k],unit) for a in arms)+' |')
    text+=['','过火包含燃烧、烧尽和扑灭；合围不等于余火归零。稳定合围时刻来自每步真值，取最后一次未合围燃烧之后的首步，旧报告20步快照时刻与此可相差1–19步。未合围火人员认知率使用燃烧格·步加权，避免后半局默认1掩盖早期问题。AoI的单位是环境步，不是真实秒。返航余量是到最近真实充电站的无阻挡最短步下界，不是安全保证。','','## 航迹执行','','| 种子 | 航迹数 | 按切换规则完成 | 全部点均≤2.5格 | 窗口内自动切点 | 近点停滞放行 | 末点完成后保持机步 |','|---|---:|---:|---:|---:|---:|---:|']
    for r in route_stats:text.append(f'| {r["seed"]} | {r["routes"]} | {r["completed"]} | {r["all_points_strict"]} | {r["switches"]} | {r["near_stall_points"]} | {r["terminal_hold_steps"]} |')
    text+=['','“完成”遵循接口到达/通过/近点停滞规则；严格到达列另外约束每个点距离≤2.5，不能混用。末点保持机步不自动等于浪费，可能仍在观测或通信；它说明队列仍可能提前结束，并未消除全部驻留。总航程、折返率也受中继和充电的有意停留影响。','','![对比结果](comparison.png)','','![首个窗口的实际航迹与目标](route_examples.png)','','## 可以怎样解释','','1. 不重训即可用现有探索接口执行短折线，巡逻标签可以在这一接口中停用。转化器没有代替Actor决定动作，也没有修改电量或环境动力学。','2. 上一轮单航点与本轮航迹的大脑选择不同，信息刷新、人员分组和充电交班也不同；两者比较的是完整方案，不是纯转化器因果增益。','3. 固定指令首点对照去掉后续航点，能够检查这批序列是否有帮助；但后续指令来自开启航迹时的反馈，关闭后的状态已经分叉，因此不是重新适应状态的公平单航点最优基线。不能从它估计通用收益百分比。','4. 这三个旧种子不是新留出集，n=3不报告统计显著性、不声称主创新成立。所有仿真在大脑思考时暂停，因此没有验证推理延迟、异步过期和真实飞行实时性。','5. 下一步值得保留转化器作为执行接口；主研究仍应围绕建议的状态条件价值、可执行性和反馈纠偏。要证明轨迹有用，需要新种子、预算相同的自适应单点/多点比较，并单独检验曲线扫描的场景条件。','','## 代码、数据与复现','','- webapp/route_converter.py：独立转化器；experiments/route_brain/：实验桥接、控制、测试和报告脚本。','- codex_<seed>/：90条我给出的command文件、完整决策/理由、600步动作和实际指令、30个全量快照、航迹事件与充电抢占记录。','- replay_<seed>/：加测仪表的确定性重放，逐步动作/指令/状态与在线轨迹全部一致，verification.json可核验。','- off_<seed>/：同计划只执行航迹首点的固定指令对照，不能当作新增自适应大脑实验。','- metrics_by_episode.csv、paired_differences.csv、route_details.csv、route_summary.csv、analysis.json、audit.json：逐局指标、配对差值、逐条航迹及审计。','- REPRODUCE.md：运行方式与依赖；无需外部API。']
    (OUT/'report.md').write_text('\n'.join(text)+'\n')
    parts=[]
    for line in text:
        if line.startswith('!['):
            image=line[line.index('(')+1:-1];parts.append('<img style="max-width:100%" src="data:image/png;base64,'+base64.b64encode((OUT/image).read_bytes()).decode()+'">')
        else:parts.append(html.escape(line))
    (OUT/'report.html').write_text('<meta charset="utf-8"><title>短航迹三局实验</title><style>body{max-width:1200px;margin:35px auto;font:15px/1.8 monospace;white-space:pre-wrap;padding:20px;color:#203041}img{white-space:normal}</style>'+ '\n'.join(parts))
    print(json.dumps(dict(means=means,route_stats=route_stats),ensure_ascii=False,indent=2))
def plots(records,traces,routes):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(14,4));arms=['single','routes','first_only'];labels=['Prior single waypoint','New short routes','Same commands: first only']
    for ax,(key,title) in zip(axs,[('affected','Final affected cells (lower better)'),('explore_final','Final explored fraction'),('aoi_mean','Mean person AoI (steps)')]):
        for s in SEEDS:ax.plot(range(3),[next(r[key] for r in records if r['arm']==a and r['seed']==s) for a in arms],'-o',label=str(s))
        ax.set_xticks(range(3),labels,rotation=12,fontsize=8);ax.set_title(title,fontsize=10);ax.grid(alpha=.25)
    axs[0].legend();fig.suptitle('3 matched seeds; adaptive comparison + fixed-command ablation (not significance)');fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    for ax,s,i in zip(axs,SEEDS,[1,3,1]):
        online=OUT/f'codex_{s}';raw=lines(online/'steps.jsonl');off=lines(OUT/f'off_{s}/steps.jsonl');cmd=read(online/'command_0000.json');route=np.array(cmd['plan']['uavs'][i]['route'])
        with gzip.open(online/'initial_state.json.gz','rt') as f:init=json.load(f)
        origin=init['uavs'][i]['position']
        for data,label,color in [(raw,'Short route','#137bac'),(off,'First target only','#df7e24')]:
            xy=np.array([origin]+[[r['uavs'][i][k] for k in ['x','y']] for r in data[:20]]);ax.plot(xy[:,0],xy[:,1],'.-',label=label,color=color)
        ideal=np.vstack([origin,route]);ax.plot(ideal[:,0],ideal[:,1],'k--',alpha=.4);ax.scatter(route[:,0],route[:,1],marker='x',s=70,color='black')
        for j,p in enumerate(route):ax.annotate(str(j+1),p)
        ax.set_aspect('equal');ax.grid(alpha=.25);ax.set_title(f'Seed {s}, UAV {i}, t=0..20');ax.set_xlabel('x');ax.set_ylabel('y')
    axs[0].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'route_examples.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()

"""Source-backed comprehensive descriptive comparison; n=3 paired seeds, no API calls."""
import sys,json,gzip,csv,math,collections,hashlib,base64,html,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/brain_comparison'))
from lowlevel.task_spec import TeamPlan
OUT=ROOT/'research_outputs/comprehensive_comparison_3seeds_20261022';SEEDS=[730000,730001,730002];ARMS=['heuristic','llm','codex'];NAMES={'heuristic':'启发式','llm':'千问 API','codex':'Codex agent'}
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def source(a,s):return (ROOT/'research_outputs/codex_brain_3runs_20261022' if a=='codex' else ROOT/'research_outputs/brain_comparison_20261022_recovered')/f'{a}_{s}'
def runs(mask):
    mask=np.asarray(mask,bool);pad=np.r_[False,mask,False];start=np.flatnonzero(~pad[:-1]&pad[1:]);end=np.flatnonzero(pad[:-1]&~pad[1:]);return (end-start).tolist()
def ratio(a,b):return float(a/b) if b else None
REG=[]
def add(key,label,group,unit,direction,definition,core=True):REG.append(dict(key=key,label=label,group=group,unit=unit,direction=direction,definition=definition,core=core))
# Fixed metric definitions, before aggregation. No weighted composite score.
for args in [
('affected','最终累计过火格数','任务','格','low','t600时 burning_state>0，含燃烧/烧尽/扑灭'),
('new_affected','新增过火格数','任务','格','low','最终累计过火减t0初始过火，同种子初始相同'),
('burnout','最终烧尽格数','任务','格','low','t600时 burning_state==2；不等于所有过火'),
('extinguished','最终已扑灭格数','任务','格','high','t600时 burning_state==3；需与受灾面积联合判断'),
('burning_final','最终仍燃烧格数','任务','格','low','t600时0<burning_state<2，含已合围余火'),
('active_final','最终未合围燃烧格数','任务','格','low','真值活火连通区中仍有健康可燃邻居的格数'),
('active_auc','未合围燃烧暴露积分','任务','格·步','low','sum(t1..600)真实未合围燃烧格数，不含已合围余火'),
('burning_auc','全部燃烧暴露积分','任务','格·步','low','sum(t1..600)全部真实燃烧格数，含已合围余火'),
('affected_auc','累计受灾面积积分','任务','格·步','low','sum(t1..600)累计过火格数，越早受灾越大'),
('controlled_final','终局无未合围燃烧格','任务','success','high','t600真值未合围燃烧为0；烧尽也可能满足，非独立成功定义'),
('contain_time','稳定无未合围火耗时（600步截断）','任务','步','low','最后一次真值未合围燃烧>0之后的首步；未完成记600并另报命中'),
('peak_burning','燃烧格数峰值','任务','格','low','max(t1..600)真实燃烧格数'),
('late_new_affected','最后200步新增过火','任务','格','low','t600累计过火减t400累计过火'),
('explore_final','最终探索率','信息','rate','high','t600全队已知格占10000格比例'),
('explore_mean','全程平均探索率','信息','rate','high','600步探索率均值，衡量探索形成的时效'),
('team_recall','火情发现率（燃烧格·步加权）','信息','rate','high','sum团队地图正确标燃烧TP / sum真实燃烧格，分母不人为补1'),
('person_recall','人员火情认知率（燃烧格·步加权）','信息','rate','high','sum人员共享地图正确标燃烧TP / sum真实燃烧格'),
('active_team_recall','未合围火发现率（加权）','信息','rate','high','sum团队TP与真值未合围火交集 / sum真值未合围燃烧格'),
('active_person_recall','未合围火人员认知率（加权）','信息','rate','high','sum人员TP与真值未合围火交集 / sum真值未合围燃烧格'),
('team_precision','团队燃烧标记精确率（加权）','信息','rate','high','sum团队TP / sum团队TP+FP；FP含过时燃烧标记'),
('person_precision','人员燃烧标记精确率（加权）','信息','rate','high','sum人员TP / sum人员TP+FP；不等于整张地图准确率'),
('final_person_recall','最终人员火情认知率','信息','rate','high','原指标t600，真实燃烧为0时按环境定义返回1'),
('knowledge_gap','发现与人员认知差（加权）','信息','rate','descriptive','加权团队召回减加权人员召回；不是纯链路因果损失'),
('uav_coverage','平均无人机通信覆盖率','通信','rate','high','按环境coverage_rate，仅计UAV贡献，不含基站'),
('comm_coverage','平均通信可用率（含基站）','通信','rate','high','环境comm_coverage_rate，基站或任意UAV均可'),
('aoi_mean','人员×时间平均AoI','通信','步','low','600×10个person_aoi均值，字段按通信阶段记录'),
('aoi_p95','人员×时间AoI P95','通信','步','low','每局6000个人员时间样本的95分位，再跨局取均值'),
('aoi_p99','人员×时间AoI P99','通信','步','low','每局6000个人员时间样本的99分位，再跨局取均值'),
('aoi_over20','AoI>20的人员时间占比','通信','rate','low','count(person_aoi>20)/6000'),
('worst_person_aoi','最差人员的平均AoI','通信','步','low','先每人600步平均，再取该局10人最大值'),
('blackout_max','最长连续通信中断','通信','步','low','每人AoI>0的连续段最大长度；含基站口径'),
('long_blackouts','长度≥20步的中断段数','通信','次','low','每人AoI>0连续段中长度≥20的段数，终局段右删失'),
('blocked_rate','碰撞阻挡的无人机时间占比','执行','rate','low','环境u.blocked为真次数/2400，非真实相撞事故数'),
('oob_rate','越界动作尝试占比','执行','rate','low','环境u.oob为真次数/2400，环境拒绝动作'),
('path_length','四机实际总航程','执行','格','descriptive','逐步实际位移欧氏长度之和，斜步sqrt(2)，不评价越少越好'),
('move_rate','实际移动的无人机时间占比','执行','rate','descriptive','非零实际位移/2400，停留可能是有意中继/补给'),
('hover_action_rate','悬停动作占比','执行','rate','descriptive','action==8次数/2400，与被阻挡不动区别开'),
('reverse_rate','相邻移动立即折返率','执行','rate','descriptive','连续两步均移动且后一位移恰为前一步反向/连续移动对数'),
('explore_stall','有效探索窗口无进展占比','执行','rate','low','20步内固定显式explore目标、初始距>5且全段未改派，末距未减少超过1格的窗口占比；非通用服从率'),
('instruction_changes','实际逐机指令变更数','执行','次','descriptive','相邻步实际order字典变化总数，包含公共协议变化'),
('protocol_rewrite','规划被目标管理改写比例','执行','rate','descriptive','每次规划规范化requested与applied任意机不同的次数/30，合法自动更新也计入'),
('safety_charge_changes','安全协议强制改充电次数','执行','次','low','每步目标管理后/原上一指令非charge，但公共充电保护后变charge的逐机次数'),
('min_energy','全程最低剩余电量','能源','电量单位','descriptive','全部2400个步后电量的最小值，非耗能效率'),
('min_margin','最低理想返航余量','能源','电量单位','descriptive','电量减最近真实站点无阻挡8邻域最少步数×2.5，负值有风险，非安全保证'),
('unsafe_margin_rate','理想返航余量<0占比','能源','rate','low','负返航余量的无人机时间/2400，不含碰撞绕行额外耗能'),
('depleted_ever','曾在步后耗尽的无人机数','能源','架','low','4架中任意步后energy<=0的数量'),
('zero_energy_rate','步后零电量时间占比','能源','rate','low','energy<=0的无人机时间/2400'),
('stranded_steps','站外零电量无人机步数','能源','机·步','low','energy<=0且不在任何真实补给坪的次数'),
('pad_entry_visits','进入补给坪次数','能源','次','descriptive','每机at_pad从假到真计1，进出抖动会重复计入，不是有效补给会话'),
('at_pad_rate','处于补给坪时间占比','能源','rate','descriptive','4机×600步里落在真实补给坪的比例，含满电停留'),
('role_explore','探索指令时间占比','分配','rate','descriptive','实际task=fire_sensing,mode=explore的无人机时间/2400，含用于位置保持的航点'),
('role_patrol','巡逻指令时间占比','分配','rate','descriptive','实际task=fire_sensing,mode=patrol的无人机时间/2400'),
('role_relay','中继指令时间占比','分配','rate','descriptive','实际task=comm_relay的无人机时间/2400'),
('role_charge','补给指令时间占比','分配','rate','descriptive','实际task=recharge的无人机时间/2400；途经充电不一定是该任务'),
]:add(*args)

def compute(a,s):
    folder=source(a,s);enriched=OUT/'instrumented_replays'/f'{a}_{s}';assert read(enriched/'verification.json')['exact_600_transitions_match_original']
    raw=lines(folder/'steps.jsonl');extra=lines(enriched/'extra_steps.jsonl');summ=read(folder/'summary.json');dec=lines(folder/'decisions.jsonl')
    assert len(raw)==len(extra)==600
    m={k:np.array([r['metrics'][k] for r in raw],float) for k in raw[0]['metrics']}
    x={k:np.array([r[k] for r in extra]) for k in extra[0]}
    with gzip.open(folder/'initial_state.json.gz','rt') as f:initial=json.load(f)
    initial_fire=sum(c in 'fFbe' for row in initial['maps']['true_fire']['rows_x'] for c in row)
    energy=np.array([[u['energy'] for u in r['uavs']] for r in raw]);aoi=np.array([r['person_aoi'] for r in raw]);actions=np.array([r['actions'] for r in raw]);disp=x['displacement'];moving=np.linalg.norm(disp,axis=2)>0
    active=x['true_active_burning'];last_active=np.flatnonzero(active>0);control=None if active[-1]>0 else (int(last_active[-1])+2 if len(last_active) else 1)
    r={'arm':a,'seed':s,'affected':int(m['burned_cells'][-1]),'new_affected':int(m['burned_cells'][-1]-initial_fire),'burnout':int(x['burned_out'][-1]),'extinguished':int(m['extinguished_cells'][-1]),'burning_final':int(x['true_burning'][-1]),'active_final':int(active[-1]),'active_auc':float(active.sum()),'burning_auc':float(x['true_burning'].sum()),'affected_auc':float(m['burned_cells'].sum()),'controlled_final':int(active[-1]==0),'contain_time':control if control is not None else 600,'containment_exact_t':control,'peak_burning':float(x['true_burning'].max()),'late_new_affected':float(m['burned_cells'][-1]-m['burned_cells'][399]),'explore_final':float(m['explored'][-1]),'explore_mean':float(m['explored'].mean()),'team_recall':ratio(x['team_tp'].sum(),x['true_burning'].sum()),'person_recall':ratio(x['person_tp'].sum(),x['true_burning'].sum()),'active_team_recall':ratio(x['active_team_tp'].sum(),active.sum()),'active_person_recall':ratio(x['active_person_tp'].sum(),active.sum()),'team_precision':ratio(x['team_tp'].sum(),x['team_tp'].sum()+x['team_fp'].sum()),'person_precision':ratio(x['person_tp'].sum(),x['person_tp'].sum()+x['person_fp'].sum()),'final_person_recall':float(m['rknown'][-1]),'uav_coverage':float(m['coverage'].mean()),'comm_coverage':float(x['comm_coverage_rate'].mean()),'aoi_mean':float(aoi.mean()),'aoi_p95':float(np.percentile(aoi,95)),'aoi_p99':float(np.percentile(aoi,99)),'aoi_over20':float((aoi>20).mean()),'worst_person_aoi':float(aoi.mean(axis=0).max()),'blocked_rate':float(x['blocked'].mean()),'oob_rate':float(x['oob'].mean()),'path_length':float(np.linalg.norm(disp,axis=2).sum()),'move_rate':float(moving.mean()),'hover_action_rate':float((actions==8).mean()),'instruction_changes':summ['instruction_changes'],'min_energy':float(energy.min()),'min_margin':float(x['return_margin'].min()),'unsafe_margin_rate':float((x['return_margin']<0).mean()),'depleted_ever':int(np.any(energy<=0,axis=0).sum()),'zero_energy_rate':float((energy<=0).mean()),'stranded_steps':int(((energy<=0)&~x['at_pad']).sum()),'at_pad_rate':float(x['at_pad'].mean())}
    r['knowledge_gap']=r['team_recall']-r['person_recall']
    blackouts=[v for i in range(10) for v in runs(aoi[:,i]>0)];r['blackout_max']=max(blackouts,default=0);r['long_blackouts']=sum(v>=20 for v in blackouts)
    consecutive=moving[:-1]&moving[1:];reverse=consecutive&np.all(disp[1:]==-disp[:-1],axis=2);r['reverse_rate']=ratio(reverse.sum(),consecutive.sum())
    initial_pad=np.array([any(max(abs(u['position'][0]-st[0]),abs(u['position'][1]-st[1]))<=1 for st in initial['rules']['charging_stations']) for u in initial['uavs']]);r['pad_entry_visits']=int((x['at_pad']&~np.vstack([initial_pad,x['at_pad'][:-1]])).sum())
    role=collections.Counter();changes=0;safety=0;rewrites=0;decmap={row['t']:row for row in dec};prev=None
    for idx,row in enumerate(raw):
        for o in row['orders']:role[o['mode'] if o['task']=='fire_sensing' else ('relay' if o['task']=='comm_relay' else 'charge')]+=1
        before=decmap[idx]['plan']['uavs'] if idx in decmap else prev
        if before:safety+=sum(o['task']=='recharge' and b['task']!='recharge' for b,o in zip(before,row['orders']))
        if prev is not None:changes+=sum(b!=o for b,o in zip(prev,row['orders']))
        prev=row['orders']
    assert changes==summ['instruction_changes']
    for key in ['explore','patrol','relay','charge']:r['role_'+key]=role[key]/2400
    for row in dec:
        normalized=json.loads(json.dumps([o.to_dict() for o in TeamPlan.from_llm_json(row['requested_plan'],4).orders]))
        rewrites+=int(normalized!=row['plan']['uavs'])
    r['protocol_rewrite']=rewrites/30;r['safety_charge_changes']=safety
    tested=stalled=0
    initialxy=np.array([u['position'] for u in initial['uavs']]);positions=np.array([[[u['x'],u['y']] for u in row['uavs']] for row in raw])
    for start in range(0,600,20):
        for i in range(4):
            order=raw[start]['orders'][i]
            if order['task']!='fire_sensing' or order['mode']!='explore' or order.get('target') is None:continue
            if not all(raw[j]['orders'][i]==order for j in range(start,start+20)):continue
            p0=initialxy[i] if start==0 else positions[start-1,i];target=np.array(order['target']);d0=np.linalg.norm(p0-target)
            if d0<=5:continue
            tested+=1;stalled+=int(np.linalg.norm(positions[start+19,i]-target)>=d0-1)
    r['explore_stall']=ratio(stalled,tested);r['explore_stall_tested_windows']=tested;r['explore_stall_windows']=stalled
    r['milestones']={}
    for field,thresholds in [('extinguished_cells',[1,100,200]),('explored',[.5,.7,.8]),('discovery',[.5,.9]),('rknown',[.5,.9])]:
        for threshold in thresholds:
            hit=np.flatnonzero(m[field]>=threshold);r['milestones'][f'{field}>={threshold}']=int(hit[0])+1 if len(hit) else None
    # Trajectory and source consistency checks.
    assert r['affected']==r['burnout']+r['extinguished']+r['burning_final']
    assert np.allclose(x['comm_coverage_rate'],(aoi==0).mean(axis=1))
    assert abs(sum(r['role_'+k] for k in ['explore','patrol','relay','charge'])-1)<1e-12
    assert r['depleted_ever']==summ['depleted_ever']
    return r,{'base':m,'extra':x,'aoi':aoi}

def fmt(v,reg):
    if v is None:return '—'
    if reg['unit']=='rate':return f'{100*v:.1f}%'
    if reg['unit']=='success':return f'{round(v*3)}/3'
    if abs(v)>=10000:return f'{v:,.0f}'
    return f'{v:.2f}'
def csvwrite(path,records):
    keys=list(dict.fromkeys(k for r in records for k in r));
    with open(path,'w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(records)
def main():
    OUT.mkdir(exist_ok=True);records=[];trajectories={}
    for a in ARMS:
        for s in SEEDS:
            r,t=compute(a,s);records.append(r);trajectories[a,s]=t
            baseline=read(source('heuristic',s)/'summary.json');current=read(source(a,s)/'summary.json');assert baseline['initial_hash']==current['initial_hash'] and baseline['actor_sha256']==current['actor_sha256']
    stats=[];means={a:{} for a in ARMS};paired=[]
    for reg in REG:
        row=dict(reg)
        for a in ARMS:
            vals=np.array([r[reg['key']] for r in records if r['arm']==a and r[reg['key']] is not None],float);means[a][reg['key']]=float(vals.mean()) if len(vals) else None
            row.update({a+'_mean':means[a][reg['key']],a+'_sd':float(vals.std(ddof=1)) if len(vals)>1 else None,a+'_min':float(vals.min()) if len(vals) else None,a+'_max':float(vals.max()) if len(vals) else None})
        for base in ['heuristic','llm']:
            for s in SEEDS:
                c=next(r for r in records if r['arm']=='codex' and r['seed']==s);b=next(r for r in records if r['arm']==base and r['seed']==s)
                if c[reg['key']] is not None and b[reg['key']] is not None:paired.append({'metric':reg['key'],'baseline':base,'seed':s,'codex':c[reg['key']],'baseline_value':b[reg['key']],'difference_codex_minus_baseline':c[reg['key']]-b[reg['key']]})
        stats.append(row)
    csvwrite(OUT/'metrics_by_episode.csv',[{k:v for k,v in r.items() if k!='milestones'} for r in records]);csvwrite(OUT/'metrics_summary.csv',stats);csvwrite(OUT/'paired_differences.csv',paired);csvwrite(OUT/'metric_definitions.csv',REG)
    (OUT/'analysis.json').write_text(json.dumps({'seeds':SEEDS,'metric_count':len(REG),'means':means,'records':records,'summary':stats,'new_api_calls':0},ensure_ascii=False,indent=2))
    make_plots(trajectories,records,means)
    write_report(records,stats,means)
    write_html()
    audit={'status':'passed','original_independent_episodes':9,'additional_independent_episodes':0,'new_api_calls':0,'seeds':SEEDS,'metric_count':len(REG),'instrumented_transition_records':5400,'all_nine_trajectories_identical_to_original':True,'initial_state_actor_and_horizon_match':True,'checks':['area accounting: affected=burning+burnout+extinguished','comm_coverage_rate=share AoI==0','role fractions sum to1','depleted counts match original summaries','instruction switches match original summaries'],'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'experiments/comprehensive_comparison').glob('*.py')}}
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    src=OUT/'evaluation_sources';src.mkdir(exist_ok=True)
    for p in (ROOT/'experiments/comprehensive_comparison').glob('*.py'):shutil.copy2(p,src/p.name)
    print('Metric count',len(REG));print(json.dumps(means,ensure_ascii=False,indent=2))

def make_plots(ts,records,means):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    colors={'heuristic':'#2674ab','llm':'#dd7d28','codex':'#239470'}
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    specs=[('base','burned_cells','Cumulative affected cells'),('extra','true_active_burning','True uncontained burning cells'),('base','explored','Explored map fraction'),('base','coverage','UAV-only coverage'),('extra','comm_coverage_rate','Any-link coverage (including base)'),('base','mean_aoi','Mean person AoI')]
    for ax,(kind,key,title) in zip(axes.ravel(),specs):
        for a in ARMS:
            values=np.array([ts[a,s][kind][key] for s in SEEDS]);mu=values.mean(axis=0);ax.plot(range(1,601),mu,label={'heuristic':'Heuristic','llm':'Qwen API','codex':'Codex agent'}[a],color=colors[a]);ax.fill_between(range(1,601),values.min(axis=0),values.max(axis=0),color=colors[a],alpha=.09)
        ax.set_title(title);ax.set_xlabel('Environment step');ax.grid(alpha=.2)
    axes[0,0].legend();fig.suptitle('Three matched seeds: mean curves; shaded range = min-max across seeds, not CI');fig.tight_layout();fig.savefig(OUT/'process_metrics.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,(key,title) in zip(axes,[('affected','Final cumulative affected cells'),('active_auc','Uncontained burning exposure (cell-steps)'),('aoi_p95','Per-episode person-time AoI P95')]):
        for s in SEEDS:ax.plot(range(3),[next(r[key] for r in records if r['arm']==a and r['seed']==s) for a in ARMS],'-o',label=str(s))
        ax.set_xticks(range(3),['Heuristic','Qwen API','Codex agent']);ax.set_title(title,fontsize=10);ax.grid(alpha=.2)
    axes[0].legend(title='Seed');fig.tight_layout();fig.savefig(OUT/'paired_metrics.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4));x=np.arange(3);bottom=np.zeros(3)
    for key,label,color in [('role_explore','Explore','#2674ab'),('role_patrol','Patrol','#9a69b5'),('role_relay','Relay','#239470'),('role_charge','Charge','#dd7d28')]:
        vals=[means[a][key] for a in ARMS];axes[0].bar(x,vals,bottom=bottom,label=label,color=color);bottom+=vals
    axes[0].set_xticks(x,['Heuristic','Qwen API','Codex agent']);axes[0].set_ylim(0,1.05);axes[0].set_title('Applied order allocation (UAV-time)');axes[0].legend(fontsize=8)
    for i,a in enumerate(ARMS):axes[1].scatter([next(r['explore_final'] for r in records if r['arm']==a and r['seed']==s) for s in SEEDS],[next(r['affected'] for r in records if r['arm']==a and r['seed']==s) for s in SEEDS],label=NAMES[a] if False else ['Heuristic','Qwen API','Codex agent'][i],s=65)
    axes[1].set_xlabel('Final explored fraction');axes[1].set_ylabel('Final affected cells');axes[1].set_title('Exploration and loss: no single composite score');axes[1].legend();axes[1].grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'allocation_tradeoff.png',dpi=170);plt.close(fig)

def write_report(records,stats,means):
    h,l,c=[means[a] for a in ARMS];loss=(c['affected']/h['affected']-1)*100;exposure=(c['active_auc']/h['active_auc']-1)*100
    text=['# 无人机大小脑协同：三种高层规划的多指标对比报告','','**比较对象：启发式、千问 API、Codex agent；相同三个种子，每局600步。** 本报告扩展已有实验，未重新规划、调参或调用模型。通过严格重放补记缺失量，九条轨迹共5,400个完整转移均与原数据一致。','',f'共计算 **{len(REG)} 项评价指标**，覆盖任务控制、感知回传、通信时效、能源、执行和角色分配。指标存在相关性，不等于{len(REG)}次独立证据；不拼凑任意加权“总分”。','','## 主要判断','',f'Codex agent 在这三个场景的累计过火均低于另外两组，均值 **{c["affected"]:.1f}格**，相对启发式 **{loss:+.1f}%**；未合围燃烧暴露积分相对变化 **{exposure:+.1f}%**。平均无人机通信覆盖率为 **{c["uav_coverage"]:.1%}**，启发式 **{h["uav_coverage"]:.1%}**。',f'其代价是最终探索率均值 **{c["explore_final"]:.1%}**，低于启发式 **{h["explore_final"]:.1%}**。终局好看不能代替过程评价，需结合最长中断、低电量和阻挡指标阅读。','','这些结果支持“当前带反馈与工具分析的高层方案有进一步研究价值”，尚不足以证明Codex裸模型强于千问，或证明某一种纠偏措施具有独立贡献。n=3为描述性配对结果，不给显著性结论。','','## 1. 实验与口径','','- 场景：100×100、4架无人机、10名人员、3个初始火源、随机风；种子730000–730002。','- 同一冻结Actor、纯argmax、每20步规划、相同目标管理与充电保护；每局600步，无提前结束。初态与权重哈希逐对一致。','- 千问使用全量当前真值及现有提示词，正式组无回退。Codex使用同类当前真值的工具摘要、持续上下文，并提前知道Actor短板与既有实验结论；启发式只消费自身已有简报字段。有效信息与工具预算不相等。','- 三个种子是旧基准前三个，运行前固定，但不是新的留出集。原10局千问结论不能直接用不同样本的均值与本3局混比。','- 下表默认是每局先计算指标、再对三个种子取等权均值；区间范围见CSV。事件计数为每局均值。','- UAV覆盖率只算无人机贡献；通信可用率含基站。AoI按环境通信阶段字段记录，不能把当前位置的几何覆盖与这一字段逐格强行对齐。','- 信息召回以真实燃烧格·步加权，避免大量“已无火/已合围”时刻用默认1抬高结果；同时单列未合围火口径。','- 合围意味着不能继续向健康格扩散，并不意味着余火被扑灭；全部烧尽也可能满足“无未合围火”，必须与过火和烧尽量联合评价。','','## 2. 分维度指标结果','']
    for group,title in [('任务','任务结果与火势过程'),('信息','感知与信息回传'),('通信','通信可靠性与信息时效'),('能源','补给与能源安全'),('执行','规划执行与运动行为'),('分配','实际任务分配')]:
        text+=['### '+title,'','| 指标 | 启发式 | 千问 API | Codex agent |','|---|---:|---:|---:|']
        for reg in REG:
            if reg['group']!=group:continue
            mark={'low':' ↓','high':' ↑','descriptive':' ·描述项'}[reg['direction']]
            text.append('| '+reg['label']+mark+' | '+' | '.join(fmt(means[a][reg['key']],reg) for a in ARMS)+' |')
        text.append('')
        if group=='任务':text+=['积分以格·步计，低值表示较少/较短暴露，不是shaping reward。受限合围时间对未完成局记600，并同时报告终局完成局数；它不能被解释为未完成局真实完成耗时。','']
        if group=='信息':text+=['燃烧标记精确率的FP包括“过去有火、现在已烧尽/扑灭，但认知图仍标燃烧”的过时信息。它不是火势强度误差、也不是全图准确率。发现与认知差可能还受人员自己的观测与融合影响，不能当作纯通信损失的因果量。','']
        if group=='能源':text+=['返航余量是忽略碰撞延误的可达下界，非负不保证安全。补给坪进出次数会受站边抖动影响；进入补给坪不等于一次独立高效补给。模型每步基础能耗固定，因此不把总航程或剩余电量冒充真实节能成绩。','']
        if group=='执行':text+=['折返、停留、切换本身不一定有害：中继、补给和观察都可能需要停留。探索无进展只统计同一显式目标且起始距离>5的稳定20步片段，不等于通用指令服从率；被提前改派片段和到位后的驻留不进入分母。','']
        if group=='分配':text+=['这里只统计实际指令标签。Codex部分explore航点兼顾观测与定位覆盖，不能直接把explore标签时间全部当成有效搜图时间。','']
    text+=['## 3. 逐种子配对结果','','| 种子 | 规划器 | 过火 | 烧尽 | 扑灭 | 未合围暴露积分 | 稳定合围步 | 探索率 | UAV覆盖 | AoI P95 | 最长中断 | 耗尽架数 |','|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|']
    for s in SEEDS:
        for a in ARMS:
            r=next(r for r in records if r['arm']==a and r['seed']==s);ct=str(r['containment_exact_t']) if r['containment_exact_t'] is not None else '600步未完成'
            text.append(f'| {s} | {NAMES[a]} | {r["affected"]} | {r["burnout"]} | {r["extinguished"]} | {r["active_auc"]:,.0f} | {ct} | {r["explore_final"]:.1%} | {r["uav_coverage"]:.1%} | {r["aoi_p95"]:.1f} | {r["blackout_max"]} | {r["depleted_ever"]} |')
    text+=['','稳定合围时间来自新增逐步真值记录，替代旧报告每20步快照首次确认时刻；两者口径不同，差1–19步不表示数据冲突。','','## 4. 过程图与取舍','','阴影为三个种子的最小—最大范围，不是置信区间。','', '![任务与通信过程](process_metrics.png)','','![配对指标](paired_metrics.png)','','![任务分配与探索取舍](allocation_tradeoff.png)','','## 5. 阶段里程碑','','首次达到阈值不表示此后一直保持；信息率会因火势变化下降。`—`表示600步内未到达，不能按0步处理，也不只对到达局取均值冒充全组速度。','','| 种子 | 规划器 | 首次扑灭 | 扑灭100格 | 扑灭200格 | 探索50% | 探索70% | 探索80% | 人员认知90% |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    keys=['extinguished_cells>=1','extinguished_cells>=100','extinguished_cells>=200','explored>=0.5','explored>=0.7','explored>=0.8','rknown>=0.9']
    for s in SEEDS:
        for a in ARMS:
            r=next(r for r in records if r['arm']==a and r['seed']==s);text.append('| '+str(s)+' | '+NAMES[a]+' | '+' | '.join(str(r['milestones'][k]) if r['milestones'][k] is not None else '—' for k in keys)+' |')
    text+=['','## 6. 从数据中能得出什么','','- 优先看累计过火、烧尽、未合围暴露与合围耗时，判断是否更早抑制了火势；再看探索与有效火情召回，判断信息获取的取舍。最终探索率高不自动代表灭火更好。','- 通信应同时看覆盖均值、AoI尾部、最差人员和长中断事件。均值改善不代表每个人始终在线，不能省略第三局等困难场景。','- Codex在反馈中调整了感知位置、人员分组、补给交接、占位冲突和固定点位置。任务与过程指标可以检验结果是否改善，但本轮未做这些措施的独立消融，不能给各措施分配贡献百分比。','- 高频重规划不是本轮自动成立的解释：三组周期相同，实际切换由指令与公共协议共同决定。协议改写也不自动等于模型出错，可能是合法的目标完成更新。','- 信息组织、工具计算、已知Actor能力和持续反馈都与模型类型一起变化。因此本轮可比较整体方案价值，不能孤立归因于参数规模、提示词或模型品牌。','','## 7. 不能据现有数据报告的指标','','没有人员伤亡/真实救援完成定义、真实飞行动力学功耗、实际飞行秒与步长映射，也没有异步推理期间环境演化，因此不虚构“救援成功率”“真实节能率”“实时安全率”或决策过期损失。也没有对原始九局重新采样模型输出。','','## 8. 调用成本与速度口径','']
    from finish_pure import latency
    llm_times=[];codex_times=[];ht=[];prompt_tokens=completion_tokens=0
    for s in SEEDS:
        for d in lines(source('llm',s)/'decisions.jsonl'):
            llm_times.append(latency(source('llm',s),d['t']))
            with gzip.open(source('llm',s)/f'call_{d["t"]:04d}_response.json.gz','rt') as f:u=json.load(f).get('usage',{});prompt_tokens+=u.get('prompt_tokens',0);completion_tokens+=u.get('completion_tokens',0)
        codex_times.extend(d['seconds'] for d in lines(source('codex',s)/'decisions.jsonl'));ht.extend(d['seconds'] for d in lines(source('heuristic',s)/'decisions.jsonl'))
    costs={'formal_qwen_calls':90,'formal_qwen_prompt_tokens':prompt_tokens,'formal_qwen_completion_tokens':completion_tokens,'qwen_historical_decision_median_s':float(np.median(llm_times)),'qwen_historical_decision_p95_s':float(np.percentile(llm_times,95)),'heuristic_planning_median_s':float(np.median(ht)),'codex_interactive_wait_median_s':float(np.median(codex_times)),'codex_interactive_wait_p95_s':float(np.percentile(codex_times,95)),'codex_external_api_calls':0,'new_api_calls_for_this_report':0};(OUT/'cost_accounting.json').write_text(json.dumps(costs,indent=2))
    text+=['| 项目 | 数值 | 解释 |','|---|---:|---|',f'| 原三局千问正式决策 | 90次 | 来自原实验，含已付费有效前缀，非本次新调用 |',f'| 原三局千问输入/输出token | {prompt_tokens:,} / {completion_tokens:,} | 正式响应usage之和，不含初次故障混合轨迹的额外支出 |',f'| 千问历史规划中位/P95 | {np.median(llm_times):.2f}s / {np.percentile(llm_times,95):.2f}s | 不把免费重放耗时当推理速度 |',f'| Codex交互等待中位/P95 | {np.median(codex_times):.2f}s / {np.percentile(codex_times,95):.2f}s | 包含读状态、工具交互、三局并行调度等待，不能作为裸模型推理排名 |','| 本次扩展报告新增外部API调用 | 0 | 九条已存计划的确定性重放 |','','Codex会话额度并非免费，现有记录没有可靠token账单，未估计货币费用。所有仿真均在规划时暂停，速度表不是实时部署测试。','','## 9. 交付与复现','','- `report.md`与`report.html`：同一完整报告；HTML内嵌图片，可独立打开。','- `metrics_by_episode.csv`：9局全部指标；`metrics_summary.csv`：三组均值、样本标准差与范围；`paired_differences.csv`：每种子Codex减两种基线的差值。','- `metric_definitions.csv`：每项指标的单位、方向和精确定义；`analysis.json`：机器可读结果、里程碑与原始指标。','- `instrumented_replays/<arm>_<seed>/extra_steps.jsonl`：新增逐步量；`verification.json`：与原始600步完整轨迹一致；`audit.json`：全局核验。','- 原始决策与轨迹继续保留在 `research_outputs/codex_brain_3runs_20261022/` 和 `research_outputs/brain_comparison_20261022_recovered/`，未改写。','- 源码 `experiments/comprehensive_comparison/enrich.py` 与 `analyze.py`，副本和哈希保存在本目录；环境依赖沿用项目Python3.13，不需API。','','### 指标字典','','| ID | 指标 | 单位 | 精确定义 |','|---|---|---|---|']
    for reg in REG:text.append('| '+reg['key']+' | '+reg['label']+' | '+reg['unit']+' | '+reg['definition']+' |')
    (OUT/'report.md').write_text('\n'.join(text)+'\n')

def write_html():
    # Minimal local renderer for the authored subset; no external assets or publishing.
    md=(OUT/'report.md').read_text().splitlines();parts=[];in_table=False;in_list=False
    for line in md:
        if line.startswith('|'):
            if not in_table:parts.append('<div class="table-wrap"><table>');in_table=True;header=True
            if set(line.replace('|','').replace(':','').replace('-','').replace(' ',''))==set():continue
            tag='th' if header else 'td';cells=line.strip('|').split('|');parts.append('<tr>'+''.join(f'<{tag}>{html.escape(c.strip())}</{tag}>' for c in cells)+'</tr>');header=False;continue
        if in_table:parts.append('</table></div>');in_table=False
        if line.startswith('!['):
            label=line[2:line.index(']')];p=OUT/line[line.index('(')+1:line.rindex(')')];parts.append(f'<figure><img alt="{html.escape(label)}" src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}"><figcaption>{html.escape(label)}</figcaption></figure>');continue
        if line.startswith('#'):
            n=len(line)-len(line.lstrip('#'));parts.append(f'<h{n}>'+html.escape(line[n:].strip())+f'</h{n}>');continue
        if line:parts.append('<p>'+html.escape(line).replace('**','')+'</p>')
    if in_table:parts.append('</table></div>')
    css='body{font:16px/1.75 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;color:#203041;background:#f2f5f8;margin:0}main{max-width:1280px;margin:30px auto;padding:42px;background:white;border-radius:12px}h1{font-size:30px}h2{margin-top:42px;border-bottom:2px solid #dce6ef;padding-bottom:8px}h3{margin-top:30px;color:#19658a}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #dce4ea;padding:8px 12px;text-align:left}th{background:#eaf1f7}tr:nth-child(even){background:#f8fafb}.table-wrap{overflow:auto;margin:18px 0}img{width:100%;height:auto}figcaption{font-size:13px;color:#617386;text-align:center}p{margin:12px 0}@media print{body{background:white}main{padding:10px;margin:0}h2{break-after:avoid}tr{break-inside:avoid}}'
    (OUT/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>无人机高层规划多指标对比报告</title><style>'+css+'</style><main>'+''.join(parts)+'</main></html>')
if __name__=='__main__':main()

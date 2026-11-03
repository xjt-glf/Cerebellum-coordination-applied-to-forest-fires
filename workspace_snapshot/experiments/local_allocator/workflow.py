"""Actual supervision, development-only selection, immutable heldout freeze/report."""
import concurrent.futures as cf
import datetime,hashlib,json,multiprocessing as mp,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from .environment import ROOT,LOWLEVEL,EXPECTED,file_hash
from .evaluate import OUT,evaluation_jobs,execute,paired_report
from .runner import episode,init_worker,append_json

def load(path): return [json.loads(s) for s in Path(path).read_text().splitlines() if s] if Path(path).exists() else []

def status(stage):
    rows=[]
    for r in sorted(OUT.glob('v*_s*')):
        if (r/'pid.json').exists():
            p=json.loads((r/'pid.json').read_text());eps=load(r/'episodes.jsonl');ups=load(r/'updates.jsonl')
            rows.append(dict(run=r.name,pid=p['pid'],episodes=len(eps),updates=len(ups),checkpoint=str(r/'latest.pt')))
    text='# MAPPO 真实运行状态\n\n'+str(datetime.datetime.now())+'\n\n'+stage+'\n\n'
    text+='容器 xiangjutao-kda-bench，8×H800，torch 2.13；CPU 整局 worker，GPU 更新。\n\n'
    text+='```json\n'+json.dumps(rows,indent=2)+'\n```\n\n'
    text+='契约结果 contracts.json；奖励审计 audit_random.jsonl / audit_h3.jsonl；workflow.log 为监督日志。\n'
    text+='全部轨迹为 gzip JSONL，完整 checkpoint 含 actor/critic/优化器/RNG；仅支持完整批次边界恢复。\n'
    text+='留出仅在 frozen_manifest.json 写入后运行；未完成数量不算通过。底层 SHA256 '+EXPECTED+'。\n'
    (OUT/'STATUS.md').write_text(text)

def audit_summary():
    result={}
    for kind in ('random','h3'):
        data=load(OUT/f'audit_{kind}.jsonl');result[kind]={}
        for n in range(3,7):
            ms=[m for m in data if m['n']==n]
            if not ms: continue
            keys=set(k for m in ms for k in m['reward_components'])
            result[kind][n]=dict(episodes=len(ms),mean_seconds=float(np.mean([m['seconds'] for m in ms])),
                burned=float(np.mean([m['burned_cells'] for m in ms])),components={k:dict(mean=float(np.mean([m['reward_components'].get(k,0) for m in ms])),p05=float(np.percentile([m['reward_components'].get(k,0) for m in ms],5)),p95=float(np.percentile([m['reward_components'].get(k,0) for m in ms],95))) for k in keys})
    (OUT/'reward_audit.json').write_text(json.dumps(result,indent=2))

def main():
    torch.set_num_threads(1)
    status('训练运行中；开始开发集 H3 小网格搜索，不读取留出。')
    audit_summary()
    splits=dict(training=[list(range(980000+i*2000,982000+i*2000)) for i in range(3)],
        audit={n:list(range(978000+(n-3)*40,978020+(n-3)*40)) for n in range(3,7)},
        development={n:list(range(986000+(n-3)*40,986040+(n-3)*40)) for n in range(3,7)},
        heldout={n:list(range(987000+(n-3)*40,987030+(n-3)*40)) for n in range(3,7)},
        extrapolation={n:list(range(988000+(n-7)*40,988030+(n-7)*40)) for n in (7,8)})
    (OUT/'splits.json').write_text(json.dumps(splits,indent=2))
    grid=[]
    for th in (220,260,300):
        for mode in ('one','ratio'):
            tag=f'dev_h3_{th}_{mode}'
            jobs=evaluation_jobs('h3',[3,4,5,6],4,986000,tag,threshold=th,relay_mode=mode)
            grid.extend([dict(j,_tag=tag) for j in jobs])
    # Concurrent grid jobs, preserve each exact candidate's raw paired results.
    with cf.ProcessPoolExecutor(16,mp_context=mp.get_context('spawn'),initializer=init_worker) as pool:
        fs={pool.submit(episode,j):j['_tag'] for j in grid}
        for f in cf.as_completed(fs):
            _,m=f.result();append_json(OUT/(fs[f]+'.jsonl'),m)
    candidates=[]
    for th in (220,260,300):
        for mode in ('one','ratio'):
            m=load(OUT/f'dev_h3_{th}_{mode}.jsonl')
            candidates.append(dict(threshold=th,relay_mode=mode,burned_mean=float(np.mean([x['burned_cells'] for x in m])),episodes=len(m)))
    h3=min(candidates,key=lambda x:(x['burned_mean'],x['threshold'],x['relay_mode']))
    (OUT/'h3_selection.json').write_text(json.dumps(dict(grid=candidates,selected=h3,budget='6 candidates x 4 fleet sizes x 4 development scenes = 96 episodes; provisional subset of 40'),indent=2))
    dev_h3=execute(evaluation_jobs('h3',[3,4,5,6],8,986000,'dev_h3_selected',threshold=h3['threshold'],relay_mode=h3['relay_mode']),OUT/'dev_h3_selected.jsonl',16)
    status('H3 开发选择已完成；正在评估各训练种子早期 checkpoint。训练继续。')
    selections={}
    # Budgeted development search: initial ~100, midpoint, then later latest.
    for seed in (1103,1104,1105):
        run=OUT/f'v1_count_s{seed}'
        paths=sorted(run.glob('checkpoint_*.pt'))
        if not paths: raise RuntimeError(f'No checkpoint for {seed}')
        picked=sorted(set([min(paths,key=lambda p:abs(int(p.stem.split('_')[-1])-100)),paths[len(paths)//2],paths[-1]]))
        selections[seed]=[]
        for path in picked:
            tag=f'dev_s{seed}_{path.stem}'
            ms=execute(evaluation_jobs('mappo',[3,4,5,6],8,986000,tag,path),OUT/(tag+'.jsonl'),16)
            report=paired_report(ms,dev_h3)
            guards=all(v['metrics']['depleted']['paired_mean']<=.1 and v['metrics']['cov_mean']['paired_mean']>=-.03 for v in report.values())
            score=float(np.mean([v['metrics']['burned_cells']['paired_mean']/max(1,v['metrics']['burned_cells']['h3_mean']) for v in report.values()]))
            selections[seed].append(dict(checkpoint=str(path),guard_pass=guards,score=score,report=report))
            (OUT/'development_selection.json').write_text(json.dumps(selections,indent=2))
    status('早期 checkpoint 开发评估完成；继续训练至晚期候选冻结。开发样本 8/机数，预算限制为 provisional。')
    # Leave enough wall clock for final paired evaluation before the two-hour handoff.
    freeze_at=datetime.datetime(2026,11,3,21,22).timestamp()
    while time.time()<freeze_at:
        status('持续监督训练；等待 21:22 冻结晚期候选。所有当前训练进度见下方。')
        time.sleep(min(30,freeze_at-time.time()))
    for seed in (1103,1104,1105):
        run=OUT/f'v1_count_s{seed}';path=sorted(run.glob('checkpoint_*.pt'))[-1]
        if str(path) not in {x['checkpoint'] for x in selections[seed]}:
            tag=f'dev_s{seed}_{path.stem}'
            ms=execute(evaluation_jobs('mappo',[3,4,5,6],8,986000,tag,path),OUT/(tag+'.jsonl'),16)
            report=paired_report(ms,dev_h3)
            guards=all(v['metrics']['depleted']['paired_mean']<=.1 and v['metrics']['cov_mean']['paired_mean']>=-.03 for v in report.values())
            score=float(np.mean([v['metrics']['burned_cells']['paired_mean']/max(1,v['metrics']['burned_cells']['h3_mean']) for v in report.values()]))
            selections[seed].append(dict(checkpoint=str(path),guard_pass=guards,score=score,report=report))
    (OUT/'development_selection.json').write_text(json.dumps(selections,indent=2))
    selected={s:min(v,key=lambda c:(not c['guard_pass'],c['score'])) for s,v in selections.items()}
    source={str(p):file_hash(p) for p in (ROOT/'experiments/local_allocator').glob('*.py')}
    manifest=dict(frozen_at=str(datetime.datetime.now()),selected=selected,h3=h3,lowlevel_sha256=file_hash(LOWLEVEL),source_sha256=source,
        checkpoint_sha256={s:file_hash(c['checkpoint']) for s,c in selected.items()},heldout_tuning=False,limitations='8 development episodes per fleet; H3 grid 4 per fleet; provisional selection budget')
    (OUT/'frozen_manifest.json').write_text(json.dumps(manifest,indent=2))
    status('模型、H3 和源码已冻结；正在运行 3..6 留出与 7/8 外推，各机数 30 个场景，禁止据此调参。')
    # Evaluate all policies in the SAME process pool, avoiding repeated imports.
    alljobs=[]
    for kind,ck,tag in [('h3',None,'heldout_h3')]+[('mappo',v['checkpoint'],f'heldout_s{s}') for s,v in selected.items()]:
        jobs=evaluation_jobs(kind,[3,4,5,6],30,987000,tag,ck,h3['threshold'],h3['relay_mode'])
        jobs+=evaluation_jobs(kind,[7,8],30,988000,tag,ck,h3['threshold'],h3['relay_mode'])
        alljobs.extend([dict(j,_tag=tag) for j in jobs])
    with cf.ProcessPoolExecutor(24,mp_context=mp.get_context('spawn'),initializer=init_worker) as pool:
        fs={pool.submit(episode,j):j['_tag'] for j in alljobs}
        for i,f in enumerate(cf.as_completed(fs)):
            _,m=f.result();append_json(OUT/(fs[f]+'.jsonl'),m)
            if i%24==0:
                print(f'heldout complete {i+1}/{len(fs)}',flush=True)
                status(f'冻结评测正在运行：已完成 {i+1}/{len(fs)} 局；不据留出结果重新选择。')
    baseline=load(OUT/'heldout_h3.jsonl')
    reports={s:paired_report(load(OUT/f'heldout_s{s}.jsonl'),baseline) for s in selected}
    (OUT/'paired_results.json').write_text(json.dumps(reports,indent=2))
    assert file_hash(LOWLEVEL)==EXPECTED
    lines=['# 三动作 MAPPO 训练评测报告','',str(datetime.datetime.now()),'',
        '结论：以逐机数配对结果为准；工程验收不能由训练奖励或总体平均代替。开发选择仅 8 场景/机数、H3 网格仅 4 场景/机数，整体标 provisional。',
        '', '训练：CPU 独立进程整局采样，冻结 9 动作底层 Actor，GPU 训练新 3 动作共享 Actor 与每团队一次 Critic。每 batch 16 局/480 团队高层转移，4 PPO epoch；小于建议 2048 以适应吞吐。',
        '', '观测：自身16、队友16/实体、人员8/实体、22摘要+50已知地图。绝对坐标、固定物理归一化；known_fire_flag统一0。两轮理想同步广播。目标协议由各机独立重复计算，全局ID仅用于目标破对称。',
        '', '奖励 v1_count 保留指南权重，真值仅用于奖励与评估。600为真正terminal且不bootstrap。旧底层Actor本身包含历史特权特征，未修改其输入/权重；新分配器Actor/Critic/广播/目标通过真值隔离测试。',
        '', '冻结选择见 frozen_manifest.json。完整分项与95%场景bootstrap区间见 paired_results.json；原始逐局数据和所有 gzip 轨迹保留。',
        '', '|训练种子|实际训练局数|N|留出局数|累计过火相对H3下降|耗尽增量|总通信增量|门槛|', '|---|---:|---:|---:|---:|---:|---:|---|']
    for s,report in reports.items():
        count=len(load(OUT/f'v1_count_s{s}'/'episodes.jsonl'))
        for n,v in report.items():
            lines.append(f'|{s}|{count}|{n}|{v["episodes"]}|{v["burn_reduction"]:.2%}|{v["metrics"]["depleted"]["paired_mean"]:.3f}|{v["metrics"]["cov_mean"]["paired_mean"]:.3f}|{v["engineering_thresholds_passed"]}|')
    lines+=['','7/8为未见规模外推，不与3..6平均。未完成事项：人员/队友消融重训、ID轮换性能评估、原始patrol启发式H额外参考、逐机执行成功率/目标完成指标、推理P50/P95/P99、完整40场景开发集及严格每100局评估、2000局/种子。归一化奖励版本开发比较需独立版本，不据留出回选。','',
        '开始/结束低层SHA256：'+EXPECTED,'',
        '复现：容器工作目录 /workspace/hrl_dqn_improve，python3 -m experiments.local_allocator.checks；训练 python3 -m experiments.local_allocator.train --seed 1103 --index 0 --workers 8 --batch-episodes 16 --seconds 4800 --device cuda:0；恢复增加 --resume 对应latest.pt。']
    (OUT/'training_report.md').write_text('\n'.join(lines)+'\n')
    status('冻结留出/外推评测完成；结果见 training_report.md、paired_results.json。训练进程若未到预算仍在继续，选中模型保持冻结。')
    print('WORKFLOW_FINISHED',flush=True)

if __name__=='__main__':
    try: main()
    except BaseException:
        traceback.print_exc();status('监督/评估进程失败，完整异常见 workflow.log；已有训练checkpoint与数据保留。');raise

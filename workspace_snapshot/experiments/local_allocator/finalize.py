"""Generate reviewable final evidence, plots and honest guide acceptance gaps."""
import datetime,json,os,time
from pathlib import Path
import numpy as np
import torch
from .environment import ROOT,LOWLEVEL,EXPECTED,file_hash
from .evaluate import OUT,paired_report

def load(path):return [json.loads(s) for s in path.read_text().splitlines() if s] if path.exists() else []

def collect():
    rows=[]
    for seed in (1103,1104,1105):
        run=OUT/f'v1_count_s{seed}';episodes=load(run/'episodes.jsonl');updates=load(run/'updates.jsonl')
        pid=json.loads((run/'pid.json').read_text())['pid']
        last=sorted(run.glob('checkpoint_*.pt'))[-1]
        rows.append(dict(seed=seed,episodes=len(episodes),updates=len(updates),pid=pid,running=os.path.exists(f'/proc/{pid}'),checkpoint=str(last),sha256=file_hash(last),fleet_counts={n:sum(m['n']==n for m in episodes) for n in range(3,7)},physical_steps=sum(m['t'] for m in episodes),team_transitions=sum(m['t']//20 for m in episodes),agent_samples=sum(m.get('agent_samples',0) for m in updates)))
    manifest=json.loads((OUT/'frozen_manifest.json').read_text()) if (OUT/'frozen_manifest.json').exists() else {}
    baseline=load(OUT/'heldout_h3.jsonl')
    reports={seed:paired_report(load(OUT/f'heldout_s{seed}.jsonl'),baseline) for seed in (1103,1104,1105)}
    source_ok={p:file_hash(p)==h for p,h in manifest.get('source_sha256',{}).items()}
    checkpoint_ok={s:file_hash(manifest['selected'][s]['checkpoint'])==h for s,h in manifest.get('checkpoint_sha256',{}).items()}
    result=dict(time=str(datetime.datetime.now()),training=rows,reports=reports,frozen_source_hash_verified=source_ok,frozen_checkpoint_hash_verified=checkpoint_ok,lowlevel_sha256=file_hash(LOWLEVEL),lowlevel_unchanged=file_hash(LOWLEVEL)==EXPECTED,
        overall='provisional / incomplete guide acceptance; inspect per-fleet task thresholds',
        known_gaps=['Main selection uses 8 rather than 40 development scenes per fleet; H3 grid uses4.',
            'Training may end below2000 per seed; complete episodes reported separately.',
            'Not every100 episodes evaluated; budgeted early/mid/latest snapshots selected.',
            'Status protection_active feature is last-step transition override, not persistent latch.',
            'message_bytes counts minimal target snapshot/task messages, excludes full observation/history payload and recipient fanout.',
            'Target goal age resets on target change, not task-only age.',
            'Broadcast validates version, participant set and action range; no explicit packet timestamp/hash disagreement handling.',
            'No-teammate/no-person ablations are frozen-model inference perturbations, not separately trained ablations.',
            'Execution success diagnostic is per actual protected order segment, not original lowlevel skill acceptance rate.'])
    (OUT/'final_acceptance.json').write_text(json.dumps(result,indent=2))
    return result,manifest

def plots(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for seed in (1103,1104,1105):
        updates=load(OUT/f'v1_count_s{seed}'/'updates.jsonl')
        for ax,key,label in zip(axs.flat,('mean_burned','entropy','value_loss','actor_grad_norm'),('Training batch burned cells','Policy entropy','Team critic loss','Actor gradient norm')):
            ax.plot([u['episodes'] for u in updates],[u[key] for u in updates],label=str(seed));ax.set(xlabel='Completed training episodes',ylabel=label);ax.grid(alpha=.2)
    axs[0,0].legend(title='Training seed');fig.tight_layout();fig.savefig(OUT/'training_curves.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5))
    for index,(seed,report) in enumerate(result['reports'].items()):
        xs=[];ys=[];lower=[];upper=[]
        for n,r in report.items():
            m=r['metrics']['burned_cells'];lo,hi=m['paired_95ci'];mean=m['paired_mean']
            xs.append(int(n)+(index-1)*.17);ys.append(mean);lower.append(mean-lo);upper.append(hi-mean)
        if xs:ax.errorbar(xs,ys,yerr=[lower,upper],fmt='o',capsize=3,label=str(seed))
    ax.axhline(0,color='black',linewidth=.8);ax.axvline(6.5,color='grey',linestyle=':');ax.set(xticks=list(range(3,9)),xlabel='UAV count (7/8 unseen in training)',ylabel='Burned cells: MAPPO minus paired H3 (95% scene bootstrap)');ax.legend(title='Training seed');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'heldout_paired_burn.png',dpi=160);plt.close(fig)

def write_report(result,manifest):
    lines=['# MAPPO 两小时实施与验收交付','',result['time'],'',
        '结论：已实现、实际启动并监督三种子训练与冻结配对评测；整体按 provisional/未完整通过指南验收交付。是否达到任务指标见下表，未完成或实现口径差异不能当作通过。','',
        '容器 xiangjutao-kda-bench；实际硬件 8×NVIDIA H800，torch 2.13.0+cu130。CPU worker 整局仿真及冻结底层 Actor 推理，GPU 仅批量训练新三动作 MAPPO。','',
        '|种子|实际完整局数|3/4/5/6 机局数|更新数|物理步|PID仍运行|','|---|---:|---|---:|---:|---|']
    for r in result['training']:lines.append(f'|{r["seed"]}|{r["episodes"]}|{list(r["fleet_counts"].values())}|{r["updates"]}|{r["physical_steps"]}|{r["running"]}|')
    lines+=['','完整 checkpoint 支持批次边界恢复；实际恢复测试从16局重跑到32局，Actor/Critic 最大绝对差均为0，见 recovery_verification.json。CUDA RNG ByteTensor 恢复修复前失败日志也保留。8→16 workers 的配置、旧/新PID见 worker_scaling.json。','',
        '每局600物理步、每20步高层决策；共享Actor每机3 logits，逐机clipped PPO，团队Critic/GAE每团队时间步一次。每批16局/480团队转移、4epoch、minibatch256、gamma .97、lambda .95、学习率3e-4、熵.01、梯度裁剪.5。','',
        '开发选 H3 与模型；先冻结清单再留出。H3 配置：'+json.dumps(manifest.get('h3',{}),ensure_ascii=False)+'。','',
        '|训练种子|N|配对局数|过火下降|过火差95%区间|耗尽差|总覆盖差|任务及守护点估计门槛|','|---|---:|---:|---:|---|---:|---:|---|']
    all_pass=True
    for seed,report in result['reports'].items():
        for n,v in report.items():
            m=v['metrics'];lo,hi=m['burned_cells']['paired_95ci']
            lines.append(f'|{seed}|{n}|{v["episodes"]}|{v["burn_reduction"]:.2%}|[{lo:.1f}, {hi:.1f}]|{m["depleted"]["paired_mean"]:.3f}|{m["cov_mean"]["paired_mean"]:.3f}|{v["engineering_thresholds_passed"]}|')
            if int(n)<=6:all_pass &= v['engineering_thresholds_passed'] and v['episodes']>=30
    lines+=['','主规模3..6各自要求累计过火下降至少5%、耗尽增量≤0.1、总通信覆盖下降≤3个百分点；置信区间跨0不能宣称稳定改善。7/8仅为未见规模外推。该表的门槛仅为任务点估计，不替代完整工程契约或稳定性验收。','',
        '累计过火、活火、烧尽、扑灭、未合围暴露积分、探索、UAV/总通信、AoI尾部、最长失联、站外耗尽与保护覆盖分开保存在原始逐局JSONL及 final_acceptance.json。完整失败场景配对差保存在 paired_results.json。','',
        '契约：contracts.json 与 extra_contracts.json 包括3..8机20物理步、N=1 mask、非零padding、实体重排、真值干预隔离、独立进程目标一致、有效目标先保留去重、缺动作广播拒绝、空人员、充电保护、完整600步同seed重放/不同seed差异、原运行器固定计划逐步适配一致性。','',
        '计算延迟见 extra_contracts.json：单线程CPU，预热100次、测量500次，单机Actor、本地目标计算、四机批量Actor各有P50/P95/P99。两轮广播是假设；没有测量真实无线延迟。','',
        '额外结果：normalized_reward_development.json 为独立奖励v2开发比较（128局，各初始化种子不同，不能作严格因果消融）；legacy_reference.jsonl 为可patrol的旧H，动作/协议不同，仅作参考；robustness_summary.json 为每N两个冻结场景上的ID轮换与无队友/无人员推理消融，不能声称重训消融。','',
        '实现及验收剩余差异：','']
    lines += ['- '+x for x in result['known_gaps']]
    lines += ['', '观测16维自身/队友、8维人员集合、22维摘要和50维已知图；人员known_fire_flag因缺逐人公开事实置0；Actor不含ID、真值统计或奖励。目标按ID破对称，系统并非严格置换等变。广播提供理想公共地图和人员视图，不能推断真实丢包/延迟/分区鲁棒性。冻结底层 Actor 保留历史输入设计与技能限制。','',
        '底层冻结 SHA256：`'+result['lowlevel_sha256']+'`；与指南相同：'+str(result['lowlevel_unchanged'])+'。选中 checkpoint 和源码哈希在 frozen_manifest.json，部署 actor/critic 及哈希在 deployment_exports.json。所有路径的宿主根为 /ssd3/xiangjutao/hrl_dqn_improve，容器根为 /workspace/hrl_dqn_improve。','',
        '复现命令（容器内，工作目录 /workspace/hrl_dqn_improve）：','', '```bash','python3 -m experiments.local_allocator.checks','python3 -m experiments.local_allocator.extra_checks','python3 -m experiments.local_allocator.train --seed 1103 --index 0 --workers 16 --batch-episodes 16 --seconds 3422 --device cuda:0 --resume /workspace/hrl_dqn_improve/research_outputs/local_allocator_mappo_20261103/v1_count_s1103/latest.pt','python3 -m experiments.local_allocator.evaluate --kind mappo --checkpoint <frozen_checkpoint_absolute_path> --count 30 --start 987000 --tag reproduction --workers 16','```','',
        '训练曲线 training_curves.png；配对过火差及区间 heldout_paired_burn.png。STATUS.md、workflow.log、worker_scaling.log 和各run/resumed_training.log包含真实进度及失败痕迹。']
    (OUT/'final_report.md').write_text('\n'.join(lines)+'\n')

def main():
    deadline=datetime.datetime(2026,11,3,21,53).timestamp()
    while time.time()<deadline:time.sleep(min(30,deadline-time.time()))
    result,manifest=collect();plots(result);write_report(result,manifest)
    (OUT/'STATUS.md').write_text('# 最终交付状态\n\n'+result['time']+'\n\n真实训练、评测和剩余差异见 final_report.md 与 final_acceptance.json。整体 provisional，不宣称收敛或通过全部门槛。\n\n'+json.dumps(result['training'],ensure_ascii=False,indent=2)+'\n\n低层actor SHA256 '+result['lowlevel_sha256']+'\n')
    print('FINAL_ARTIFACTS_WRITTEN',flush=True)

if __name__=='__main__':main()

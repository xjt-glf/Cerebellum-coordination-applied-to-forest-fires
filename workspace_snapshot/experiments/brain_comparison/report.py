"""Aggregate independent paired episodes, never treat steps as independent samples."""
import sys,json,csv,gzip,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'research_outputs/brain_comparison_20261022'
METRICS=[('burned_cells','最终累计过火格数','lower'),('extinguished_cells','最终已扑灭格数','higher'),('rknown','最终人员火情认知率','higher'),('explored','最终探索率','higher'),('mean_coverage','全程平均通信覆盖率','higher'),('p95_person_aoi','人员×时间 AoI 的 P95','lower'),('max_blackout_steps','最长连续未覆盖步数','lower'),('depleted_ever','曾耗尽的无人机数','lower'),('instruction_changes','指令改变次数（逐机）','descriptive')]
def read_lines(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
def main():
    manifest=json.loads((OUT/'manifest.json').read_text());seeds=manifest['seeds'];summaries={}
    for arm in ['heuristic','llm']:
        for seed in seeds:
            p=OUT/f'{arm}_{seed}'/'summary.json'
            if p.exists():summaries[(arm,seed)]=json.loads(p.read_text())
    valid=[s for s in seeds if all((a,s) in summaries for a in ['heuristic','llm'])]
    for s in valid:
        assert summaries['heuristic',s]['initial_hash']==summaries['llm',s]['initial_hash'],f'Initial mismatch {s}'
        assert summaries['heuristic',s]['actor_sha256']==summaries['llm',s]['actor_sha256']
        assert summaries['heuristic',s]['steps']==summaries['llm',s]['steps']==600
    if not valid:raise RuntimeError('No completed matched pairs')
    records=list(summaries.values())
    with open(OUT/'episodes.csv','w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    rng=np.random.default_rng(42599);stats=[]
    for key,label,direction in METRICS:
        h=np.array([summaries['heuristic',s][key] for s in valid]);l=np.array([summaries['llm',s][key] for s in valid]);d=l-h
        boot=np.mean(rng.choice(d,size=(20000,len(d)),replace=True),axis=1);ci=np.quantile(boot,[.025,.975])
        signs=np.array([[1 if (i>>j)&1 else -1 for j in range(len(d))] for i in range(2**len(d))])
        p=float(np.mean(np.abs((signs*d).mean(axis=1))>=abs(d.mean())-1e-12))
        wins=int(np.sum(d<0 if direction=='lower' else d>0));ties=int(np.sum(np.isclose(d,0)))
        stats.append(dict(metric=key,label=label,direction=direction,heuristic_mean=float(h.mean()),llm_mean=float(l.mean()),delta_mean=float(d.mean()),bootstrap95=ci.tolist(),signflip_two_sided_p=p,llm_wins=wins,ties=ties,pairs=len(d)))
    calls=[];attempts=[]
    for s in seeds:
        folder=OUT/f'llm_{s}'
        for row in read_lines(folder/'decisions.jsonl'):
            calls.append({'seed':s,'t':row['t'],'source':row['meta'].get('source'),'seconds':row['seconds'],'usage':row['meta'].get('usage',{}),'prompt_chars':row['meta'].get('prompt_chars',0)})
        attempts.extend(read_lines(folder/'api_attempts.jsonl'))
    api={'logical_calls':len(calls),'successful_llm_decisions':sum(c['source']=='llm' for c in calls),'fallback_decisions':sum(c['source']!='llm' for c in calls),'http_attempts':len(attempts),'failed_attempts':sum(not x['ok'] for x in attempts),'latency_median_s':float(np.median([c['seconds'] for c in calls])) if calls else None,'latency_p95_s':float(np.percentile([c['seconds'] for c in calls],95)) if calls else None,'prompt_tokens':sum(x.get('usage',{}).get('prompt_tokens',0) for x in attempts if x['ok']),'completion_tokens':sum(x.get('usage',{}).get('completion_tokens',0) for x in attempts if x['ok']),'returned_models':sorted({str(x.get('returned_model')) for x in attempts if x['ok']})}
    result={'paired_seeds':valid,'stats':stats,'api':api};(OUT/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,(key,label) in zip(axes,[('burned_cells','Cumulative affected cells'),('mean_coverage','Mean communication coverage'),('p95_person_aoi','Person-time AoI P95')]):
        for s in valid:
            ax.plot([0,1],[summaries['heuristic',s][key],summaries['llm',s][key]],'-o',alpha=.55)
        ax.set_xticks([0,1],['Heuristic','API LLM']);ax.set_title(label);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'paired_results.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,(key,title) in zip(axes,[('burned_cells','Cumulative affected cells'),('coverage','Communication coverage'),('mean_aoi','Mean person AoI')]):
        for arm,color in [('heuristic','#2872b5'),('llm','#df7930')]:
            values=np.array([[row['metrics'][key] for row in read_lines(OUT/f'{arm}_{s}'/'steps.jsonl')] for s in valid])
            ax.plot(np.arange(1,601),values.mean(axis=0),label=arm,color=color)
        ax.set_title(title);ax.set_xlabel('Environment step');ax.grid(alpha=.2)
    axes[0].legend();fig.tight_layout();fig.savefig(OUT/'mean_trajectories.png',dpi=170);plt.close(fig)
    st=stats[0];delta=st['delta_mean'];h=st['heuristic_mean'];pct=100*delta/h if h else 0
    conclusion=f'主指标累计过火格数：启发式均值 {h:.1f}，大模型均值 {st["llm_mean"]:.1f}；大模型相对变化 {pct:+.1f}%。'
    ci=st['bootstrap95']
    if ci[0]<=0<=ci[1]:conclusion+='配对差的 95% bootstrap 区间跨零，这 10 个场景尚不足以支持明确优势。'
    else:conclusion+=('在这组种子上，大模型的过火面积较低。' if delta<0 else '在这组种子上，启发式的过火面积较低。')+'结论仅限本协议和当前场景族。'
    if api['fallback_decisions']:conclusion+=f'其中 {api["fallback_decisions"]} 次决策回退为启发式，因此 LLM 组实际是带回退的 API 系统。'
    lines=['# API 大模型与启发式规划：10 次配对实验','',conclusion,'','## 实验是否完整','',f'完成 {len(valid)}/10 对，共 {len(records)}/20 局。每局 600 步，每 20 步规划一次。初始状态指纹、Actor SHA-256 和步数已逐对核验。',f'使用权重：`{records[0]["actor_sha256"]}`。没有重新训练或更换低层模型。','','## 协议与可解释范围','','- 100×100 地图，4 架无人机，10 名人员，3 个初始火源；随机风；种子 730000–730009。','- 双方共用目标管理、最近真实补给点保护、纯 Actor argmax；无航点动作接管，不因合围、活火归零或全机耗尽提前停局。','- 目标协议统一为：显式改变计划立即接受；重复未完成目标保留，已完成/失效探索目标按同一采样器更新。网页原先按来源区别处理，本实验只在独立 runner 中消除了该差异。','- LLM 调用项目既有 API、模型名 qwen-plus、temperature=0.2，使用现有提示词与完整真值快照（含全格火情、地形、人员、无人机、能量和规则）。启发式收到同一个状态结构，但现有规则只消费简报。','- 因此这是当前系统的效果对比，不是有效信息量完全相等的纯推理能力实验；LLM 若有收益，不能排除真值信息的贡献。','- 仿真在调用 API 时暂停。延迟独立记录，没有模拟推理期间火势继续演化，因此没有检验决策过期问题。','- 每个场景每组只跑一次。环境/Actor 固定，API temperature=0.2 仍可能有随机性，未估计同种子多次 API 波动。','','## 指标结果','','差值统一为“大模型 − 启发式”。区间按场景为单位重采样 20,000 次；n=10，区间只是小样本不确定性描述。AoI 与覆盖使用整局，不把 6,000 个时间步当独立试验。','','| 指标 | 启发式均值 | 大模型均值 | 配对均差 | 95% bootstrap 区间 | 大模型胜/平/负 |','|---|---:|---:|---:|---|---|']
    for r in stats:
        ci=r['bootstrap95'];wins=r['llm_wins'];ties=r['ties'];wt='—' if r['direction']=='descriptive' else f'{wins}/{ties}/{len(valid)-wins-ties}'
        lines.append(f'| {r["label"]} | {r["heuristic_mean"]:.4f} | {r["llm_mean"]:.4f} | {r["delta_mean"]:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {wt} |')
    lines+=['','累计过火为最终 `burning_state > 0` 的格数，包含燃烧、烧尽和已扑灭区域；它衡量累计受灾足迹。已扑灭数单独列出，更多已扑灭不自动代表损失更小。最终人员认知率可能饱和，需结合全程通信/AoI 看。','', '![配对结果](paired_results.png)','','![平均轨迹](mean_trajectories.png)','','## 逐种子主指标','','| 种子 | 启发式过火 | LLM 过火 | 差值 | LLM 回退 |','|---|---:|---:|---:|---:|']
    for s in valid:
        hr=summaries['heuristic',s];lr=summaries['llm',s];lines.append(f'| {s} | {hr["burned_cells"]} | {lr["burned_cells"]} | {lr["burned_cells"]-hr["burned_cells"]:+d} | {lr["api_fallbacks"]} |')
    lines+=['','## API 成本与失败','','```json',json.dumps(api,ensure_ascii=False,indent=2),'```','','token 数按 API usage 原样累计；没有价格配置，未虚构货币费用。所有传输尝试及失败有日志；失败后启发式回退没有被当作 LLM 成功。','','## 数据与复现','','- `manifest.json`：冻结配置、代码哈希、种子和协议。','- `episodes.csv`、`analysis.json`：逐局指标、配对统计与 API 汇总；精确符号翻转检验仅作探索，未多重比较校正。','- `{arm}_{seed}/initial_state.json.gz`：初态完整信息；`steps.jsonl`：每步指标、位置、电量、指令、动作和人员 AoI。','- `llm_*/call_*_request.json.gz` / `response.json.gz`：实际消息和供应商原始响应，不含 Authorization 或 API key；`api_attempts.jsonl`：时延、usage 与重试。','- `decisions.jsonl`、`charge_overrides.jsonl`：请求计划、实际应用计划、目标管理/安全修正、解析回退。','- 实验源码：`experiments/brain_comparison/run.py`、`launch.py`、`report.py`。环境使用项目已有 Python 3.13 依赖；见 `REPRODUCE.md`。','','## 对“大模型价值”的判断','','现阶段应按任务结果与调用开销一起判断。结果只能回答当前提示词、完整信息模式、现有执行器及标准场景是否获益；不能证明所有大模型或未来 Critic 方法无价值。若 LLM 仍输，优先检查逐步指令频繁切换、人员链路选择与执行器巡逻局限的实际轨迹证据，不能仅凭成绩把原因归为模型能力或提示词。若 LLM 赢，也应补充同信息量和真实延迟对照，才能支持论文中的机制主张。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print(conclusion);print(json.dumps(api,ensure_ascii=False));print('pairs',len(valid))
if __name__=='__main__':main()

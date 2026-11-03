"""Create the authoritative full-horizon report after all pure API recoveries finish."""
import sys,json,gzip,shutil,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];ORIGINAL=ROOT/'research_outputs/brain_comparison_20261022';OUT=ROOT/'research_outputs/brain_comparison_20261022_recovered'
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
def latency(folder,t):
    record=next(r for r in rows(folder/'decisions.jsonl') if r['t']==t)
    replay=next((r for r in rows(folder/'replayed_calls.jsonl') if r['t']==t),None)
    return latency(Path(replay['source']),t) if replay else record['seconds']
def main():
    for seed in range(730000,730010):
        pure=OUT/f'llm_{seed}';assert (pure/'summary.json').exists()
        dec=rows(pure/'decisions.jsonl');assert len(dec)==30 and all(d['meta'].get('source')=='llm' for d in dec)
        # Effective historical latency is distinct from the cost-free local replay wall time.
        s=json.loads((pure/'summary.json').read_text());s['historical_planning_seconds']=sum(latency(pure,d['t']) for d in dec)
        (pure/'summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2))
        destination=OUT/f'heuristic_{seed}'
        if not destination.exists():shutil.copytree(ORIGINAL/f'heuristic_{seed}',destination)
        hs=json.loads((destination/'summary.json').read_text());hs['historical_planning_seconds']=hs['planning_seconds'];(destination/'summary.json').write_text(json.dumps(hs,ensure_ascii=False,indent=2))
    manifest=json.loads((ORIGINAL/'manifest.json').read_text());manifest.update(status='complete: 10 pure API full episodes + 10 heuristic episodes',recovery='Paid valid prefix replayed with exact prompt and full trajectory checks; no heuristic fallback in formal API data',recovery_concurrency=4,source_directory=str(ORIGINAL),intermediate_failure='78 heuristic fallback decisions in original runs; excluded from formal API episodes',final_source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'experiments/brain_comparison').glob('*.py')})
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    import report,behavior,audit
    report.OUT=OUT;behavior.OUT=OUT;audit.OUT=OUT
    report.main();behavior.main()
    effective=[];tokens_in=0;tokens_out=0;used_replay=0
    for seed in range(730000,730010):
        folder=OUT/f'llm_{seed}';used_replay+=len(rows(folder/'replayed_calls.jsonl'))
        for d in rows(folder/'decisions.jsonl'):
            effective.append(latency(folder,d['t']))
            with gzip.open(folder/f'call_{d["t"]:04d}_response.json.gz','rt') as f:obj=json.load(f)
            u=obj.get('usage',{});tokens_in+=u.get('prompt_tokens',0);tokens_out+=u.get('completion_tokens',0)
    attempts=[]
    for root in [ORIGINAL,OUT]:
        for p in root.rglob('api_attempts.jsonl'):attempts.extend(rows(p))
    cost={'formal_decisions':300,'formal_fallbacks':0,'paid_prefix_replayed_decisions':used_replay,'formal_prompt_tokens_including_paid_prefix':tokens_in,'formal_completion_tokens_including_paid_prefix':tokens_out,'formal_decision_historical_latency_median_s':float(np.median(effective)),'formal_decision_historical_latency_p95_s':float(np.percentile(effective,95)),'all_experiment_http_attempts_including_discarded_mixed_runs':len(attempts),'all_experiment_successful_requests':sum(a['ok'] for a in attempts),'all_experiment_prompt_tokens':sum(a.get('usage',{}).get('prompt_tokens',0) for a in attempts if a['ok']),'all_experiment_completion_tokens':sum(a.get('usage',{}).get('completion_tokens',0) for a in attempts if a['ok']),'all_experiment_http_errors':sum(not a['ok'] for a in attempts),'health_probe_note':'One extra small health request failed HTTP 403; no usage returned; excluded from rollout counts.'}
    (OUT/'api_accounting.json').write_text(json.dumps(cost,ensure_ascii=False,indent=2))
    analysis=json.loads((OUT/'analysis.json').read_text());analysis['recovery_http_api_only']=analysis['api'];analysis['api']={'logical_calls':300,'successful_llm_decisions':300,'fallback_decisions':0,'prompt_tokens':tokens_in,'completion_tokens':tokens_out,'latency_median_s':cost['formal_decision_historical_latency_median_s'],'latency_p95_s':cost['formal_decision_historical_latency_p95_s'],'accounting':'Formal 300 decision responses, including already-paid replay prefix; see api_accounting.json for all actual expenditure'};(OUT/'analysis.json').write_text(json.dumps(analysis,ensure_ascii=False,indent=2))
    body=(OUT/'report.md').read_text()
    # Base report statistics only see newly executed attempts. Replace its misleading latency/usage block.
    start=body.index('## API 成本与失败');end=body.index('## 数据与复现',start)
    accounting='## API 成本、故障恢复与统计口径\n\n```json\n'+json.dumps(cost,ensure_ascii=False,indent=2)+'\n```\n\n正式组全部 300 次决策均由真实 API 生成；恢复时前缀读取原始响应，输入逐字核对一致，整段动作/轨迹逐条核验一致。前缀只付费一次，没有重新采样或挑选更优方案。恢复后的新决策发生于原首次失败处，任何 API 回退均立即停局，未进入正式成绩。\n\n初次 10 路并发时出现 5 次 429 及后续间歇性 403，导致 78 次启发式回退；这些原始混合局已排除。将恢复批次降低至 4 路（开始时另有 1 路验证）并有限重试后补齐所有纯 API 局。接口恢复与并发变化同时发生，无法仅凭本轮证据确定 403 的服务商根因。所有故障日志仍在原目录。\n\n时延按原调用历史计算，不把免费本地重放的约零秒当作模型推理时间；正式决策 token 与所有实际支出 token 分别统计，后者还包含后来被排除的混合轨迹调用。未配置供应商价格，未虚构金额。\n\n'
    body=body[:start]+accounting+body[end:]
    st=json.loads((OUT/'analysis.json').read_text())['stats'][0];ci=st['bootstrap95'];pct=100*st['delta_mean']/st['heuristic_mean']
    headline=f'完整正式实验已完成：**10 局纯 API 大模型 + 10 局启发式**，每局 600 步、同种子配对、同新 Actor、无 API 回退。累计过火：启发式均值 {st["heuristic_mean"]:.1f}，大模型 {st["llm_mean"]:.1f}，变化 {pct:+.1f}%；配对差 95% bootstrap 区间 [{ci[0]:+.1f}, {ci[1]:+.1f}] 格。'
    if ci[0]<=0<=ci[1]:headline+='该区间跨零，不能认定累计过火有稳定差异。'
    elif st['delta_mean']>0:headline+='在当前配置和这组场景中，启发式的累计过火更少。'
    else:headline+='在当前配置和这组场景中，大模型的累计过火更少。'
    prefix='# 大模型与启发式：正式 10×2 局对比报告\n\n'+headline+'\n\n初次实验曾受 API 故障影响；已通过严格重放与续跑补齐纯 API 组。本报告的任务指标只使用最终纯组；初次混合数据保留作故障证据，未混入成绩。\n\n'
    body=prefix+body.split('\n',2)[2]
    body+='\n## 如何解释大模型价值\n\n本轮评估的是“现有 qwen-plus 提示词 + 全量真值输入 + 当前低层执行器 + 同步规划”的整体方案。不要把任务分配表现直接等同于底层指令执行能力。平均覆盖上升未必意味着新火情发现与回传改善：中继收到的可能是过时地图，探索与巡逻的时间分配也会影响任务。行为表与逐步轨迹提供关联证据；没有做对应因果消融，不能宣称已证实某个失败机制。\n\n如果累计过火或灭火结果没有优势，当前保留启发式作为部署基线更合适；大模型是否适合作 Critic、事件处理或选择性介入需要另外设计实验。本轮没有测试自然语言事件、真实时间推理延迟或场景外泛化，也没有在同种子上重复多次 API 采样。\n'
    (OUT/'report.md').write_text(body)
    sources=OUT/'evaluation_sources';sources.mkdir(exist_ok=True)
    for p in (ROOT/'experiments/brain_comparison').glob('*.py'):shutil.copy2(p,sources/p.name)
    shutil.copy2(ORIGINAL/'REPRODUCE.md',OUT/'REPRODUCE.md')
    with open(OUT/'REPRODUCE.md','a') as f:f.write('\n正式报告生成：运行 `python experiments/brain_comparison/finish_pure.py`。`episodes.csv` 的 planning_seconds 为本地本次执行耗时（含免费重放），historical_planning_seconds 为对应实际 API 调用的历史耗时总和；API 计费口径见 api_accounting.json。\n')
    audit.main()
    old=ORIGINAL/'report.md';old.write_text('> **已由正式完整实验替代。** 请阅读 `../brain_comparison_20261022_recovered/report.md`。本文件保留初次 API 故障与混合轨迹的分析，不作为最终纯大模型对比结论。\n\n'+old.read_text())
    print('AUTHORITATIVE REPORT:',OUT/'report.md');print(headline)
if __name__=='__main__':main()

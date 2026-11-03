"""Build truthful report separating pure prefix from provider-failure mixed missions."""
import os,sys,json,subprocess,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'research_outputs/brain_comparison_20261022'
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]
def main():
    for script in ['report.py','behavior.py']:
        subprocess.run([sys.executable,str(ROOT/'experiments/brain_comparison'/script)],check=True,env=os.environ.copy())
    recovery=json.loads((OUT/'recovery_points.json').read_text());horizon=min(r['first_fallback_t'] for r in recovery)
    samples=[]
    for seed in range(730000,730010):
        for arm in ['heuristic','llm']:
            folder=OUT/f'{arm}_{seed}';data=rows(folder/'steps.jsonl')[:horizon];decisions=[r for r in rows(folder/'decisions.jsonl') if r['t']<horizon]
            if arm=='llm':assert all(d['meta'].get('source')=='llm' for d in decisions)
            m=dict(data[-1]['metrics']);m.update(seed=seed,arm=arm,horizon=horizon,mean_coverage=float(np.mean([r['metrics']['coverage'] for r in data])),p95_person_aoi=float(np.percentile([v for r in data for v in r['person_aoi']],95)),mean_rknown=float(np.mean([r['metrics']['rknown'] for r in data])),planning_calls=len(decisions))
            samples.append(m)
    from report import METRICS
    metrics=[m for m in METRICS if m[0] in ['burned_cells','extinguished_cells','rknown','explored','mean_coverage','p95_person_aoi']]
    metrics.append(('mean_rknown','前 380 步平均人员火情认知率','higher'))
    rng=np.random.default_rng(52799);stats=[]
    for key,label,direction in metrics:
        h=np.array([r[key] for r in samples if r['arm']=='heuristic']);l=np.array([r[key] for r in samples if r['arm']=='llm']);d=l-h;ci=np.quantile(np.mean(rng.choice(d,size=(20000,10),replace=True),axis=1),[.025,.975]);wins=int(sum(d<0 if direction=='lower' else d>0));ties=int(sum(np.isclose(d,0)))
        stats.append(dict(metric=key,label=label.replace('最终',f'第 {horizon} 步'),heuristic_mean=float(h.mean()),llm_mean=float(l.mean()),delta=float(d.mean()),ci95=ci.tolist(),llm_wins=wins,ties=ties))
    (OUT/'pure_prefix_analysis.json').write_text(json.dumps({'horizon':horizon,'selection':'Earliest provider failure across all 10 seeds; post-hoc auxiliary horizon, not original full-horizon primary test','pure_pairs':10,'records':samples,'stats':stats},ensure_ascii=False,indent=2))
    mainstat=stats[0];change=mainstat['delta']/mainstat['heuristic_mean']*100
    api=json.loads((OUT/'analysis.json').read_text())['api'];attempts=[r for p in OUT.glob('llm_*/api_attempts.jsonl') for r in rows(p)];ok=[r['seconds'] for r in attempts if r['ok']]
    api['successful_http_median_s']=float(np.median(ok));api['successful_http_p95_s']=float(np.percentile(ok,95));api['error_counts']={k:sum((not r['ok']) and k in r.get('error','') for r in attempts) for k in ['429','403']}
    text=['# 大模型与启发式对比实验报告','','**实验状态：部分完成。** 20 局仿真都已跑完，但 10 局 API 组后段均遇到服务商权限拒绝（`HTTP 403 / AccessDenied.Unpurchased`），共 78 次决策回退。**不能把它们当成 10 局全程纯大模型实验。** 一次独立的小请求健康检查也返回同样的 403；本地无法修复服务商模型访问权限。','','可以判断的是：**当前配置尚未体现出替代启发式的收益。** 在全部 10 对均未发生回退的前 380 步，LLM 组平均累计过火 '+f'{mainstat["llm_mean"]:.1f} 格，启发式 {mainstat["heuristic_mean"]:.1f} 格，变化 {change:+.1f}%。'+ ' 全程 API+回退系统的过火也更高。不能据此断言大模型方法本身无价值，也不能把后段差异全部归因于大模型。','','## 不含回退的共同前缀：10 对 × 380 步','','前 380 步为所有场景中首次 API 失败前的最长共同区间：每局前 19 次规划全部来自真实 qwen-plus API，双方使用同一新 Actor 和协议。380 是由接口故障事后确定的辅助窗口，**不是预先设计的 600 步主实验**。它有助于排除回退污染，但不能代替完整任务验收。','','| 指标 | 启发式 | 纯 API LLM | LLM − 启发式 | 配对 95% bootstrap 区间 | LLM 胜/平/负 |','|---|---:|---:|---:|---|---|']
    for r in stats:
        ci=r['ci95'];text.append(f'| {r["label"]} | {r["heuristic_mean"]:.4f} | {r["llm_mean"]:.4f} | {r["delta"]:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {r["llm_wins"]}/{r["ties"]}/{10-r["llm_wins"]-r["ties"]} |')
    text+=['','累计过火不是短窗 shaped reward，而是到第 380 步实际累计受灾格数。但短于完整任务，后续排名仍可能改变。所有区间按 10 个场景重采样，未按时间步伪造大样本。','','## API 故障与结果边界','',f'- 300 次逻辑决策中，222 次成功生成计划；78 次回退。305 次 HTTP 尝试中，5 次限流（429，均恢复），78 次权限拒绝（403，未恢复）。',f'- 成功 HTTP 调用中位耗时 {api["successful_http_median_s"]:.2f} 秒，P95 {api["successful_http_p95_s"]:.2f} 秒；总输入 {api["prompt_tokens"]:,} token，输出 {api["completion_tokens"]:,} token。独立健康检查未成功，无额外 token usage。','- 服务商错误说明模型访问资格/购买状态有问题；它没有提供足以确认“余额不足”的证据，不能把错误原因直接说成欠费。','- `api_health_check.json`、逐调用日志及原始决策都已保存。恢复脚本会重放已付费的有效前缀，检查输入一致，再从首次失败处调用 API，失败即停，不再混入启发式。','- 不使用 API 故障之后偶尔成功的调用来拼接“纯 LLM”轨迹，因为它们的环境状态已受先前启发式回退影响。','','## 600 步全程结果：API+启发式回退系统','','下文所有 API 组全程统计均含回退。用途是描述实际系统效果，不是全程纯 LLM 能力结论。','']
    body=(OUT/'report.md').read_text();body=body.split('\n',2)[2]
    body=body.replace('大模型均值','API混合组均值').replace('大模型相对变化','API混合组相对变化').replace('大模型胜/平/负','API混合组胜/平/负').replace('LLM 组','API混合组').replace('大模型的过火面积','API混合组的过火面积')
    text.append(body)
    text+=['','## 当前处置建议','','当前演示与性能基线继续保留启发式。先恢复 qwen-plus 的模型访问资格，再完成原定 600 步纯 API 任务；不要根据这次受故障影响的结果去重训底层。现有纯前缀已显示“给全部数据”并没有自动带来收益，下一轮应比较信息组织与任务选择，特别是高层决策对应的实际观测/回传效果。这里只是待检验方向，未用本轮数据调参后重复报分。','','## 继续运行与交付核验','','恢复接口后可执行 `experiments/brain_comparison/recover.py <seed>`（730000–730009），输出到单独的 `research_outputs/brain_comparison_20261022_recovered/`。已付费决策从磁盘重放，不重复调用；在首次失败处开始新的 API 调用，并在任何回退发生前立即停止。恢复结果不得和原混合数据混写。','', '`audit.json` 记录 20 局步数、600 次规划、初态一致性与凭据排除检查；`pure_prefix_analysis.json` 为辅助纯前缀统计；`recovery_points.json` 为各局恢复边界。']
    (OUT/'report.md').write_text('\n'.join(text)+'\n')
    # Preserve runnable evaluation sources, deliberately excluding server.py credentials.
    sources=OUT/'evaluation_sources';sources.mkdir(exist_ok=True)
    for p in (ROOT/'experiments/brain_comparison').glob('*.py'):(sources/p.name).write_bytes(p.read_bytes())
    (OUT/'api_summary.json').write_text(json.dumps(api,ensure_ascii=False,indent=2))
    print('PURE PREFIX',json.dumps(stats,ensure_ascii=False))
if __name__=='__main__':main()

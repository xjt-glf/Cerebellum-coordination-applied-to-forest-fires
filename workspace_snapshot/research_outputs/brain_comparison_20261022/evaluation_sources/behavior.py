import json,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'research_outputs/brain_comparison_20261022'
def rows(path):return [json.loads(x) for x in path.read_text().splitlines()]
def main():
    episodes=[]
    for arm in ['heuristic','llm']:
        for seed in range(730000,730010):
            folder=OUT/f'{arm}_{seed}';s=json.loads((folder/'summary.json').read_text());steps=rows(folder/'steps.jsonl');dec=rows(folder/'decisions.jsonl');roles=collections.Counter()
            for row in steps:
                for order in row['orders']:
                    role=order['mode'] if order['task']=='fire_sensing' else ('relay' if order['task']=='comm_relay' else 'charge')
                    roles[role]+=1
            first70=next((r['metrics']['t'] for r in steps if r['metrics']['explored']>=.7),None)
            record={'arm':arm,'seed':seed,'role_uav_steps':dict(roles),'first_70pct_explored_step':first70,'mean_all_fire_knowledge':float(np.mean([r['metrics']['rknown'] for r in steps])),'mean_all_fire_discovery':float(np.mean([r['metrics']['discovery'] for r in steps])),'mean_extinguished_cells':float(np.mean([r['metrics']['extinguished_cells'] for r in steps])),'instruction_changes':s['instruction_changes'],'charge_override_events':len(rows(folder/'charge_overrides.jsonl')) if (folder/'charge_overrides.jsonl').exists() else 0}
            episodes.append(record)
    (OUT/'behavior.json').write_text(json.dumps(episodes,ensure_ascii=False,indent=2))
    text=['\n## 轨迹行为核查（关联证据，不等于因果消融）','', '以下按真实应用指令统计，包含公共目标更新和充电保护；每组共 24,000 个无人机×时间步。','', '| 全程行为 | 启发式 | API 大模型 |','|---|---:|---:|']
    means={}
    for arm in ['heuristic','llm']:
        a=[r for r in episodes if r['arm']==arm];means[arm]={'roles':{k:sum(r['role_uav_steps'].get(k,0) for r in a)/24000 for k in ['explore','patrol','relay','charge']},'knowledge':np.mean([r['mean_all_fire_knowledge'] for r in a]),'switches':np.mean([r['instruction_changes'] for r in a]),'reached70':sum(r['first_70pct_explored_step'] is not None for r in a)}
    for key,label in [('explore','探索指令占比'),('patrol','巡逻指令占比'),('relay','中继指令占比'),('charge','充电指令占比')]:text.append(f'| {label} | {means["heuristic"]["roles"][key]:.1%} | {means["llm"]["roles"][key]:.1%} |')
    text.extend([f'| 全程平均人员火情认知率（全部真实燃烧格口径） | {means["heuristic"]["knowledge"]:.3f} | {means["llm"]["knowledge"]:.3f} |',f'| 到达 70% 探索的局数 | {means["heuristic"]["reached70"]}/10 | {means["llm"]["reached70"]}/10 |',f'| 每局逐机指令变化均数 | {means["heuristic"]["switches"]:.1f} | {means["llm"]["switches"]:.1f} |','','人员认知率分母会随火势和扑灭变化，不能单独用来给“中继质量”作因果结论；这里与过火、覆盖和 AoI 联合阅读。'])
    summaries={(arm,s):json.loads((OUT/f'{arm}_{s}'/'summary.json').read_text()) for arm in ['heuristic','llm'] for s in range(730000,730010)}
    ds=sorted((summaries['llm',s]['burned_cells']-summaries['heuristic',s]['burned_cells'],s) for s in range(730000,730010))
    text.extend(['','### 两个代表性场景',''])
    for delta,seed in [ds[0],ds[-1]]:
        text.extend([f'种子 **{seed}**：LLM − 启发式累计过火差 {delta:+d} 格。'])
        for arm in ['heuristic','llm']:
            s=summaries[arm,seed];b=next(r for r in episodes if r['arm']==arm and r['seed']==seed)
            text.append(f'- {arm}：过火 {s["burned_cells"]}，扑灭 {s["extinguished_cells"]}，探索 {s["explored"]:.1%}，平均覆盖 {s["mean_coverage"]:.1%}，AoI P95 {s["p95_person_aoi"]:.1f}，指令变化 {s["instruction_changes"]}；角色步数 `{json.dumps(b["role_uav_steps"],ensure_ascii=False)}`。')
        text.append('')
    text.extend(['这是按主指标选出的最好/最差差值案例，不能当作独立复现。完整逐局轨迹均已保留，未只保留有利场景。',''])
    with open(OUT/'report.md','a') as f:f.write('\n'.join(text))
    print(json.dumps(means,ensure_ascii=False,default=float))
if __name__=='__main__':main()

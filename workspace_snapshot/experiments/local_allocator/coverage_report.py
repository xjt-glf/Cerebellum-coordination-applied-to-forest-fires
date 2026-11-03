"""Report v3 independently; never select versions using heldout results."""
import datetime,json,os,time
from pathlib import Path
import numpy as np
from .evaluate import OUT,paired_report
from .environment import file_hash,LOWLEVEL,EXPECTED

def load(p):return [json.loads(s) for s in p.read_text().splitlines() if s] if p.exists() else []

def generate():
    manifest=json.loads((OUT/'v3_frozen_manifest.json').read_text()) if (OUT/'v3_frozen_manifest.json').exists() else {}
    baseline=load(OUT/'heldout_h3.jsonl');reports={s:paired_report(load(OUT/f'heldout_v3_s{s}.jsonl'),baseline) for s in (1103,1104,1105)}
    training=[]
    for seed in (1103,1104,1105):
        run=OUT/f'v3_coverage_s{seed}';ms=load(run/'episodes.jsonl');up=load(run/'updates.jsonl')
        pid=json.loads((run/'pid.json').read_text())['pid'] if (run/'pid.json').exists() else None
        training.append(dict(seed=seed,episodes=len(ms),updates=len(up),physical_steps=sum(x['t'] for x in ms),fleet_counts={n:sum(m['n']==n for m in ms) for n in range(3,7)},pid=pid,running=bool(pid and os.path.exists(f'/proc/{pid}'))))
    exports={}
    for p in (OUT/'deployment').glob('v3_s*/*.pt'):exports[str(p)]=file_hash(p)
    result=dict(time=str(datetime.datetime.now()),training=training,reports=reports,manifest=manifest,exports_sha256=exports,lowlevel_sha256=file_hash(LOWLEVEL),overall='Independent v3 reward branch; developmental selection provisional; no cross-version heldout selection')
    (OUT/'v3_final_acceptance.json').write_text(json.dumps(result,indent=2))
    lines=['# v3 通信奖励修正分支报告','',result['time'],'',
        '触发依据：v1 的开发集通信守护失败；在任何留出结果之前把每物理步失联比例惩罚从0.010提高到0.100，其他奖励不变。配置、启动PID、开发选择、冻结时间与哈希均单列保存，不覆盖v1/v2。三个初始化训练种子仍为1103/1104/1105，各自使用与v1相同的训练场景池；训练预算较短，不能与v1宣称等步数比较。','',
        '模型仅按开发集选择；留出各N30场景，7/8单列外推。没有根据v1或v3的留出结果回选奖励版本。整体实现差异与开发样本预算仍适用 final_report.md 和 specifications.json 的限制，不能把任务点估计门槛当作全部工程验收。','',
        '|种子|训练局数|3/4/5/6机分布|更新|','|---|---:|---|---:|']
    for r in training:lines.append(f'|{r["seed"]}|{r["episodes"]}|{list(r["fleet_counts"].values())}|{r["updates"]}|')
    lines+=['','|种子|N|留出配对局数|过火下降|过火差95%区间|总通信差|耗尽差|点估计门槛|','|---|---:|---:|---:|---|---:|---:|---|']
    for s,rs in reports.items():
        for n,r in rs.items():
            m=r['metrics'];ci=m['burned_cells']['paired_95ci']
            lines.append(f'|{s}|{n}|{r["episodes"]}|{r["burn_reduction"]:.2%}|[{ci[0]:.1f},{ci[1]:.1f}]|{m["cov_mean"]["paired_mean"]:.3f}|{m["depleted"]["paired_mean"]:.3f}|{r["engineering_thresholds_passed"]}|')
    lines+=['','完整均值、中位数、bootstrap95%区间与失败种子见 v3_paired_results.json / v3_final_acceptance.json；逐局结果为 heldout_v3_s1103/1104/1105.jsonl；完整轨迹在 evaluation/heldout_v3_s*/。','',
        'checkpoint 位于 v3_coverage_s*/checkpoint_*.pt，部署 Actor/Critic 位于 deployment/v3_s*/。哈希、选中checkpoint与源码在 v3_frozen_manifest.json。','',
        '资源限制：根分区90G满盘曾导致一次docker exec失败，重试与TMPDIR移到/ssd3后继续；未清理其他任务数据。resource_constraints.json保留具体错误。','',
        '底层Actor SHA256：'+result['lowlevel_sha256']+'，是否仍冻结：'+str(result['lowlevel_sha256']==EXPECTED)+'。']
    (OUT/'v3_report.md').write_text('\n'.join(lines)+'\n')
    with open(OUT/'STATUS.md','a') as f:f.write('\n\nv3 独立分支结果已写入 v3_report.md / v3_final_acceptance.json；原版完整结果见 final_report.md。不据留出回选版本。\n')
    print('V3_FINAL_REPORT_WRITTEN',flush=True)

def main():
    until=datetime.datetime(2026,11,3,21,54).timestamp()
    while time.time()<until:time.sleep(min(30,until-time.time()))
    generate()

if __name__=='__main__':main()

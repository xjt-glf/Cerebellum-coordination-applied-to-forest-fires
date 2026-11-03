# critic_advice：第一方向（Critic 给精确可操作建议）的研究代码

对照《创新点探索指南.md》第 5/6 节设计。全部代码只新增，未改动任何现有源码与交付权重。

## 文件

| 文件 | 作用 |
|---|---|
| harness.py | 配对执行核心：环境快照（deepcopy+三套 RNG）、统一协议 rollout、任务指标直算、H brain 构造 |
| edits.py | 指令级修改生成器（7 类规则编辑 + 随机对照），全部只用可观测证据 |
| collect_pairs.py | 采集配对分支数据（基线整局 + 每决策边界快照 + 每编辑分支） |
| analyze_pairs.py | 筛选分析（分类型均值/CI、失败分类、noop 确定性检查） |
| analyze_episode.py | 整局窗口分析（跨窗口相关性、级联结构） |
| predict_gain.py | 收益预测器（按种子切分 28/4/8，回归+分类+决策规则+标定） |
| policy_experiment.py | 复合策略整局配对实验（H 参数变体，按种子配对） |

## 复现命令（容器 xiangjutao-211，CPU 即可）

```bash
cd /workspace/xiangjutao/hrl_dqn_improve

# 1. 阶段 A 基线冒烟
python deliverable/simulate.py --episodes 1 --fires 3 --rescuers 10 --uavs 4 \
    --steps 600 --replan-every 20 --seed 20000 --quiet

# 2. 配对采集（40 种子，~12 分钟 / 20 workers）
python experiments/critic_advice/collect_pairs.py --run screen40 --seeds 40 --seed0 30000 --workers 20

# 3. 整局窗口版（同键，分支跑到回合结束，~35 分钟）
python experiments/critic_advice/collect_pairs.py --run screen40_ep --seeds 40 --seed0 30000 --workers 20 --window-long 600

# 4. 分析
python experiments/critic_advice/analyze_pairs.py --run screen40
python experiments/critic_advice/analyze_episode.py          # 读 runs/screen40_ep

# 5. 收益预测器（Δext@60 / Δcov20 / ... 按种子切分）
python experiments/critic_advice/predict_gain.py --run screen40 --target ext60

# 6. 复合策略配对实验
python experiments/critic_advice/policy_experiment.py --seeds 40 --seed0 31000 --workers 40

# 7. EACC v1（第二轮）
python experiments/critic_advice/gen_data.py --mode exec --regimes normal,scarce --seeds 100 --seed-base 40000:41000 --workers 24
python experiments/critic_advice/gen_data.py --mode episode --regimes normal,scarce --seeds 40 --seed-base 42000:43000 --workers 24
python experiments/critic_advice/exec_model.py --model exec --ensemble 3
python experiments/critic_advice/exec_model.py --model value --ensemble 3
python experiments/critic_advice/exec_model.py --model scalar --ensemble 3
python experiments/critic_advice/critic_loop.py --stage expl --policies base,random,rule,scalar,eacc,eacc_pos

# 8. v2 规划器级建议（oracle 探针 + 选择器 + 闭环）
python experiments/critic_advice/oracle_switch_probe.py        # 2400 局，产出 oracle_switch.json
# 切换点 ctx（并入 selector 训练，见 research_outputs/critic_v1_report.md §7.2 内联脚本）
python experiments/critic_advice/v2_loop.py                    # selector vs base vs random_sw
```

## 数据目录

- runs/screen40/：40 种子 × 29 边界 × 适用编辑，+20/+60 双检查点（4343 条）
- runs/screen40_ep/：同键，长窗口=整局（4343 条）
- runs/policy_ep/：复合策略实验（results.json + confirm_P1.json）
- runs/screen1_bugctx_invalid/：**作废批次**（编辑上下文取在 H 计划下发之前的 bug，仅留档）

每个 run 的 config.json 记录完整显式配置、actor SHA-256、全部相关源码哈希与命令行。

## 口径备忘

- 指标从环境状态直算（harness.task_metrics），不用低层 shaped return（set_plan 会 clear_shape 造成虚假差）。
- noop 检查 = 原计划原样重发，验证配对差可精确归因于计划本身（screen40/screen40_ep 全部零差）。
- 配对差按种子聚类报 CI；删失（steps_run 不一致）单列不入均值。
- 报告在 /research_outputs/{phase_a_audit,phase_b_screening}.md 与 innovation_report.md。

## 第二轮新增（EACC）

| 文件 | 作用 |
|---|---|
| critic_data.py | 决策上下文 v2（含认知粗视图）、机制指标、参数化编辑空间、特征编码（580 维） |
| gen_data.py | exec/episode 双数据集采集（双 regime，配对标签） |
| exec_model.py | 执行模型 / 价值头 / 标量头训练与评估（按种子切分） |
| critic_loop.py | v1 闭环（6 臂 + 频率消融臂） |
| oracle_switch_probe.py | 规划器级配置切换的 oracle 探针 |
| v2_loop.py | v2 闭环（学习选择器 vs 基线 vs 随机切换） |

第二轮报告：/research_outputs/candidates_design.md（三候选与选型）、/research_outputs/critic_v1_report.md（完整实验记录）。

# 阶段 A 核对记录（权重、代码事实、基线复现）

日期：2026-10-13
执行环境：容器 xiangjutao-211（python 3.12.3, numpy 2.3.5, torch 2.11.0+cu130，仅用 CPU）
项目根：/ssd3/xiangjutao/hrl_dqn_improve（Linux 侧；指南中的 /Users/xiangjutao/Desktop/ 路径是同一项目的 Mac 视图，`.venv` 为 Mac 环境，Linux 上不可用）

## 1. 权重核对

| 权重 | SHA-256（实测） | 指南记录 | 一致 |
|---|---|---|---|
| deliverable/model/actor_best.pt | e70e3f6169704b7b4027974049d3505a9113e2d6cfd38ceef93e46b460271168 | 同 | 是 |
| deliverable/model/read_critic.pt | 9de6df75c5912c2ef81b68cf33def32718600cd56102353ad3e3a42649df66cf | 同 | 是 |

项目当前无 Git 仓库。本轮所有新增代码在 `experiments/`、`research_outputs/` 独立目录，未改动任何现有源码与交付权重；`experiments/critic_advice/runs/*/config.json` 记录了源码与权重哈希作为版本清单。

## 2. 指南第 3 节 10 条事实逐条核对结果

| # | 事实 | 核对位置 | 结论 |
|---|---|---|---|
| 1 | 网页同步等待 | webapp/server.py:282-287：`brain.plan` 返回后才继续 `env.step`，等待期间环境不推进 | 属实。第一阶段实验沿用同步设置隔离反馈质量 |
| 2 | 现有门限实验没有调用真实 LLM | lowlevel/read_critic.py:744-787：`eval_gating` 用 `sample_plan(kind="random")` 随机计划 + 重采样 | 属实。只能当评分器诊断 |
| 3 | 历史配对检验含 Critic 自举值 | lowlevel/read_critic.py:545-567：`_rollout_branch` 返回 r + γV(s′) − V(s)，V 来自被检 Critic 自身 | 属实。本轮配对执行改为真实有限窗口任务指标差（experiments/critic_advice/harness.py） |
| 4 | 前缀价值差 ≠ 个体因果贡献 | read_critic.py:251-275：同一 segment 的整段回报复用到所有前缀特征 | 属实。本轮直接做指令级干预的配对执行，不再解释前缀差 |
| 5 | waypoint_assist 覆盖动作 | webapp/server.py:301-307：仅 relay+target+无 rescuer_id 的定点驻留机被确定性寻路接管；默认关闭 | 属实。主实验全部关闭 |
| 6 | 活火归零 ≠ 灭火成功 | webapp/server.py:145-156、314-317：`_active_burning` 只数未合围燃烧格，烧尽格不计；seen_fire 闩锁防开局误判 | 属实。烧尽也会触发"火情清零"。研究 runner 用固定 600 步 + 独立记录 ext/contained/burned，不用提前收队当成功率 |
| 7 | relay 中 rescuer_id 优先 + 自动选目标 | lowlevel/obs_builder.py:266-279：rescuer_id 命中者优先；无显式目标时自动选最近未覆盖人员/人员质心 | 属实。实验记录 raw plan / applied orders / resolved target 三层 |
| 8 | HeuristicBrain 忽略 event | webapp/brains.py:84-93 | 属实。语言事件能力单列，不入本轮主比较 |
| 9 | 策略可知 vs 评价真值 | lowlevel/obs_builder.py:73-148：shared_cache 全部来自全队认知图 team_map_info；sense_state 同源；burned/discovery 等评价口径才用真值 | 属实。修改生成器只读 sense_state（可观测） |
| 10 | advice 字段未必被底层利用 | task_spec.py:73-111（10 维指令编码含解析目标/环误差/交接位）；obs_builder.py:349-351 | 属实。配对实验同时测"指令变了但执行没变"的比例（Actor 无响应率） |

## 3. 本轮新增的关键事实（指南未写、影响实验设计）

**F1. 默认路径下环境逐步演化完全确定。**
- 火蔓延 `update_burning_state`（Env/低层环境.py:652-697）为向量化确定性更新，无随机数；
- 人员移动 `move_rescuers_one_step`（2077-2165）BFS/贪心确定性；
- 全文件仅 init（280-440 行）消费全局 np.random；`random.shuffle`（1894-1895）只在 legacy 分支 `assign_rescuer_targets` 的真值路径上，默认 `rescuer_knows_only_shared_map=True` 时在 1845 行提前 return，不会执行。
- 推论：配合确定性 actor（argmax），**同一快照出发的两个分支，唯一差异就是下发的计划**。配对 A/B 无需公共随机数管理即可精确归因。已在 noop 检查中实测验证（同计划重发 → +20/+60 所有指标完全相等）。

**F2. 低层 set_plan 会重置势函数 shaping（env_wrapper.py:218-219）。**
跨分支比较不能用低层 shaped return（计划变更本身会造成 shaping 电位清零的虚假差）。本轮指标全部直接从环境状态计算（harness.task_metrics），不用 shaped return。

**F3. 配对执行成本。**
单线程 7.8 ms/步（128 sps）、快照 deepcopy 3.1 ms。一个 60 步分支 ≈ 0.5 s。容器 192 核 CPU、内存 2 TB：完全不需要 GPU。

**F4. webapp/server.py:104 内嵌了一个千问 API key（默认 base_url maas.qianwenaiapi.com）。**
本轮未使用；产出物一律不携带该 key。后续真实 LLM 闭环实验前需先把它挪到环境变量/本地配置。

## 4. H 基线复现（显式基准配置）

配置：100×100、4 机、3 火点、10 人、600 步、规划周期 20、spread_steps 40、spread_gain 1.0、seed 显式；H 参数与 deliverable/simulate.py 默认一致（charge_at=260、explore_until=0.70、max_chargers=2）。冻结 actor（哈希见上）确定性执行。

种子 20000 单回合（deliverable/simulate.py 冒烟）：探索 0.882、发现 1.000、人员已知 1.000、覆盖率均值 0.480、过火 0.0487、扑灭 245 格、合围 2、耗尽 0。
screen1 批次（seed 30000-30009，experiments/critic_advice/runs/screen1/episodes.jsonl）：见筛选报告。

## 5. 与 L2 基线（ReAd）的机制差异定位

ReAd（ACL 2025 Findings，本地 .codex-research/02.txt）：Critic 回归出联合/个体优势分，反馈形式是"Critic: [Advantage Score]，低于阈值请重规划"——标量分数 + 重规划触发，无定位依据、无具体编辑建议。本项目 read_critic.py 即此风格（ReAd-S 增量优势 + 门限 alpha）。
本轮第一方向的主张空间：**依据驱动的具体编辑建议**（定位到机/指令、给可验证证据、给可执行 edit、预测收益经过配对执行校准）在同预算下超过"分数低→重规划"。screen1 即回答"编辑空间里是否存在可检出的真实收益差"这一前置问题。

## 6. 阶段 A 结论

- 基线可复现、执行协议与指标口径已在 runner 中显式固定；
- 未发现基线被刻意削弱的迹象（H 参数取 simulate.py 文档默认）；
- 可重复执行瓶颈的定位交给 screen1 配对数据回答（见 phase_b_screening.md）。

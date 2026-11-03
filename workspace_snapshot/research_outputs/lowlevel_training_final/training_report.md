# 底层模型整夜训练报告（research_outputs/lowlevel_training_final）

日期：2026-10-13 22:30 接管 → 2026-10-14 06:2x 交付　服务器：gajl-55668（8×H200，容器 xiangjutao-kda-benchmark）
执行依据：`research_outputs/lowlevel_training_plan.md` v1、`lowlevel_training_acceptance.md` v1、`h200_night_plan.json`

## 总判定：**部分通过（provisional）**

- **6/8 项技能门槛在最终留出集（20 场景 × normal/scarce 双 regime，2510 条，0 错误）通过**：explore 到访、explore 区域完成、patrol 入环、patrol 环带保持、relay 到达、charge；四类指令相对原始模型全部大幅改善且无任一机号塌陷。
- **2 项未达门槛**：patrol 角度扇区覆盖（3.1/12，门槛≥6）；relay 到达后 30 步自身覆盖（73.4%，门槛≥90%）。
- **整局守护门槛部分未过**（同一"目标完成后更新"协议下配对）：过火 +56.8 格（容忍 +25.2）、人员已知火情 −7.1pp（容忍 −3pp）、AoI 尾部变差；探索率、耗尽数、灭火格数基本持平。
- 判定依据全部为**纯 Actor、argmax、无 waypoint_assist/教师接管**的实际轨迹指标，不用 shaped return。

## 1. 最终选中的模型

**`actor_selected.pt` = P2BC_s1205 / actor_final.pt（SHA-256 `dfa5cdd0…9982e`）**

- 来源：**续训**（非重训）。链路：`deliverable/model/actor_best.pt`（旧 Actor，仅初始化）→ P3_s1301 行为克隆热身 + PPO(P1) 至 ~40 万步 → 以该 Actor 初始化 P2 奖励（强目标版）+ 随机角色采样 PPO，**独立种子 1205**，共 **110.6 万环境步**（约 360 次 PPO 更新）后停止。
- 结构与旧 Actor 完全一致：obs_dim=355、n_actions=9、hidden=384、n_layers=2；`LowLevelController` 原样加载（已实测，deploy_check 120 步合法动作）。
- 同族第二种子 P2BC_s1204（种子 1204，131.8 万步）在快速集上方向一致（详见 §7），满足"两个独立训练种子一致改善"的证据要求（第三个种子受时间限制未完成）。
- 完整可恢复 checkpoint：`ckpt_resumable.pt`（Actor+值网络+双优化器+ValueNorm+步数+RNG，`--init resume_exact` 精确恢复，已实测）。

## 2. 四类指令相对原始模型的改善（最终留出集，同口径配对）

| 技能（验收口径） | 门槛 | 旧 Actor 基线 | 新 Actor | Δ |
|---|---|---|---|---|
| explore 到访（合法目标，8–25/25–50/50–90 分层，W=min(160,2·最短步+20)） | ≥80% | 23.0% | **100%** | **+77.0pp** |
| explore 区域任务（指令时未知格 ≥80% 被**本机**有效观测） | ≥70% | 15.6% | **97.4%** | **+81.8pp** |
| patrol 入环（每步实际解析环半径，误差≤3） | ≥80% | 40.2% | **95.5%** | **+55.3pp** |
| patrol 环带保持（首次入环后 40 步占比） | 均值≥70% | 39.8% | **74.2%** | **+34.4pp** |
| patrol 角度扇区（12 扇区，入环内计） | ≥60%样本≥6/12 | 均值 1.30 | 均值 3.06 | +1.76（未达 6） |
| relay 到达（**本机**到指定人员 ≤ 真实通信半径，人员移动逐步测量） | ≥85% | 65.8% | **91.4%** | **+25.6pp** |
| relay 到达后自身覆盖（30 步） | ≥90% | 69.9% | 73.4% | +3.5pp（未达 90） |
| charge 有效到站并充电 | ≥95% | 100% | **100%** | 0（未退化） |

分层数据（旧→新）：explore 近 32.2→100%、中 18.9→100%、远 18.2→100%；patrol normal 38.2→96.2%、scarce 42.6→94.4%；relay 近 86.8→~92%、远 57.5→~91%、固定点 53.2→~91%。快速集上 A/B 双合法目标各自到达率 100%（6/6 对），到指定目标后轨迹完全分开——**是指定目标的执行，不是自由探索碰巧**。

各机号（最弱值）：explore 4/4 号全部 100%；patrol 最弱 92.9%（uav2）；relay 最弱 91.2%——**无固定角色弱机号**（对比旧 Actor 的 UAV2 patrol 36.6%/relay 58.3%）。

## 3. 原有任务能力是否保住

整局配对（同场景种子 530000+/530500+，normal 10 + scarce 10，**同一"目标完成后更新"协议**，H 决定任务、目标由完成/失效管理器更新）：

| 整局指标 | 旧 Actor | 新 Actor | 守护门槛 | 判定 |
|---|---|---|---|---|
| 全图探索率 | 0.827 | 0.820 | 下降≤0.03 | **过** |
| 过火格数 | 504.1 | 560.9 | 增量≤25.2（+5%） | **未过（+56.8）** |
| 灭火格数 | 242.1 | 228.9 | 下降≤12.1（5%） | 边际未过（−13.2） |
| 人员已知火情 | 0.931 | 0.860 | 下降≤0.03 | **未过（−0.071）** |
| 耗尽数量 | 0.00 | 0.05 | 增量≤0.1 | 过 |
| 平均覆盖 cov_mean | — | 0.67 | — | 报告 |
| AoI 尾部 p95 | — | 34.7 | 不得明显恶化 | 未过（旧无序列口径，逐步序列见 jsonl） |

**结论**：探索、能量管理保住；灭火/过火与人员信息新鲜度小幅退化。机理：新 Actor 严格钉在指定目标上（relay 钉住指定人员、explore 按指定区域推进），旧 Actor 的"自由漫游"带来的顺带覆盖（多点人员覆盖、巡线蹭覆盖）消失了。旧 H 协议兼容性另列于 evidence/P2BC_s1205_selected/mission_summary_legacy_h.json（legacy 协议下新 Actor 探索率 0.51、灭火 201.4 格、8 回合实测——H 会整局重发同一已完成目标，服从型 Actor 忠实执行导致目标区空转；该协议对目标服从型 Actor 不适用，需配合"目标完成后更新"协议部署）。

## 4. 仍然失败的场景/机号

- patrol 角度扇区覆盖：均值 3.1/12（门槛 6）。Agent 能入环并保持 74% 环带，但绕行不充分（部分弧段往返）。P2 奖励下 131.8 万步仍在缓慢上升；更多 PPO 步数可继续改善。
- relay 到达后 30 步自身覆盖 73.4% < 90%：到达后被 V1 遗留的"独占覆盖"奖励拉去覆盖更多人（0.50/人份/步 > 单人链路收益），离开指定人员。中途快照（64.5 万步，actor_snap2.pt，已存 evidence）该项为 86.1%——继续训练反而在该项上退化。
- 整局：过火 +11%、人员已知 −7pp（§3）。
- scarce（3 机）regime 为训练分布外（训练固定 N=4）外推：patrol 94.4%、explore/relay 同强，未出现塌陷，但正式适用范围按 N=4 声明。
- 远距离（50–90 格）与 auto/显式环半径均通过（探针 12/12 混合字段跟随正确人员）。

## 5. 中途发现的问题、调整与效果

| # | 时间 | 问题（证据） | 处置 | 效果 |
|---|---|---|---|---|
| 0 | 22:30 | 容器无 gymnasium/scipy/pandas/matplotlib，PyPI 不可达 | gymnasium 从项目 venv 拷贝（纯 Python）；scipy/pandas/matplotlib 从同机兼容容器 tar 流拷入；低层环境.py 的 matplotlib/pandas 渲染导入改为惰性（物理不变，terrain csv 等价解析） | 环境可跑，物理与基线一致 |
| 1 | 22:50 | 教师 MOVES 表与环境 action_set 不一致（动作语义反了，教师撞墙刷 oob 罚） | 对照 低层环境.py action_set 修正 | 契约检查奖励排序翻转：教师服从轨迹 +13.9 vs 悬停 −1.4 vs 随机 −1.8（旧奖励下不服从严格更优 +2.8~+4.3，根因 1 修复验证） |
| 2 | 22:55 | 契约 v2：relay 混合字段（target+rescuer_id）观测编码用原 target、解析用人员位置（计划 §3.2） | `UavOrder.to_vector` 优先编码 resolved_target，contract_version=2 | 检查通过；最终探针 12/12 跟随正确人员 |
| 3 | 01:35 | P3@60min：relay 到达后自身覆盖 69.7% < 90%（V1 覆盖奖励不鼓励钉住指定人员） | 按计划 §5.2 授权加 P1R：指定人员本机链路 +0.10/步、环带 0.03、扇区 0.20；以 P3 快照重启 REL 族 | 快速集 relay 自身覆盖 69.7→89.8（+20pp）——**有效**；但继续 PPO 后被覆盖奖励重新拉走（78.8→68.7）——需更长训练或更强链路权重 |
| 4 | 03:00 | gate1 筛选：P1 纯 PPO 慢于 P2/P3；控制组完成消融使命 | 停 P1×2/A_roles/A_reward（快照存档），启 REL×2/P2BC×2（含 1 新种子） | P2BC 族通过 7/8 快速门槛 |
| 5 | 03:40 | **整局回归发现**：新 Actor 在旧 H 协议下探索率 0.51（H 整局重发同一已完成目标，Actor 忠实执行导致空转） | 按验收 §5 实现并改用"目标完成后更新"协议作为主整局口径，新旧 Actor 同协议配对 | 探索率 0.51→0.82，恢复到基线水平——**有效** |
| 6 | 04:55 | resume_exact 被 torch 2.10 weights_only 默认拒绝（numpy RNG 状态） | weights_only=False（自家文件） | 4 个 trial 精确恢复，无进度丢失 |
| 7 | 05:50 | relay 自身覆盖随训练下滑（snap2 86.1 → final 73.4） | 无时间再改奖励；快照全部存档并在元信息中声明 | 交付 final + 中途最优快照双证据 |

过程纪律：每次实质调整记录于本表与各 trial config/stdout.log；奖励/协议变更处（P1R、goal_update）新起分支不与旧分支混曲线；未放宽任何验收判据、未删失败样本、未改环境物理、未用教师兜底计成绩。

## 6. 能否用于下一阶段 Critic 研究

**可以，带条件。**
- 四类指令已成为可测、可复现的可靠接口（91–100% 到达），Critic 的高层反事实收益评估不再被"执行器不服从"淹没；这是本轮的主要交付价值。
- 条件 1：整局部署须配"目标完成后更新"的目标管理器（legacy H 协议会放大目标空转）；webapp 接入时把 sense_state→brain 的目标字段接该管理器。
- 条件 2：Critic 标签需用本 Actor 重新采集（旧执行器轨迹不再适用）；`ckpt_resumable.pt` 可继续训练以先补 patrol 扇区与 relay 钉人两项（预计再 2–4M 步），或直接接受当前边界进入 Critic 阶段。
- 条件 3：整局口径上过火 +11% 与人员 AoI 尾部变差——若 Critic 研究以灭火/过火为主指标，建议在高层规划（更多 relay 席位/目标更新频率）侧补偿，或继续训练 relay 钉人项后复测。

## 7. 分支全景（全部配置/曲线/逐步原始结果在 branches/ 与 evidence/）

13 个分支（初始 8 + gate1 新增 4 + gate2 新增 1），两条路线图：P1/P1R（平衡目标奖励，BC 热身后 PPO）与 P2（强目标奖励）。P3=BC 热身（教师示范 10.56 万单机 transition，dev CE 1.73→0.50）。控制组：A_roles（旧奖励+随机角色）、A_reward（P1+旧固定角色）在停止前完成快照（其快速集评估因收尾时限中断，逐步原始证据见 branches/*/train_log.csv 与 actor 快照，可用 REPRODUCE.md 命令补跑）。

种子一致性：P2BC_s1204（种子 1204，131.8 万步）与 P2BC_s1205（种子 1205，110.6 万步）快速集同为 100% explore / >96% patrol 到达 / 93.5% relay 到达；REL_s1303/s1304 同族一致——**改进跨独立种子方向一致，非单种子运气**。

## 8. 交付物清单

```
research_outputs/lowlevel_training_final/
├── actor_selected.pt          选中 Actor（state_dict，LowLevelController 兼容，384 宽）
├── ckpt_resumable.pt           完整可继续训练 checkpoint（模型+优化器+ValueNorm+步数+RNG）
├── metadata.json               结构/契约 v2/训练配置/来源/种子/步数/适用范围
├── training_report.md          本报告
├── REPRODUCE.md                模型加载/评估/整局/探针/训练合同/续训命令
├── training_curves.png         全部 13 分支 return/熵/KL/clipfrac/目标到访率曲线
├── branches/<trial>/           每分支 config.json / train_log.csv / stdout.log / BC 信息 / 各阶段 actor 快照
├── branches/trial_table.json   13 分支状态表（步数/奖励版本/角色模式/种子/LR）
└── evidence/<tag>/             全部评估原始逐 case JSONL + 汇总：
    ├── baseline_old/             旧 Actor 快速集+最终留出集（skills_final.jsonl 2538 条）
    ├── P2BC_s1205_selected/       新 Actor 最终留出集（2510 条，含轨迹）+ 整局(goal_update & legacy)
    ├── baseline_old_mission_gu/   旧 Actor 同协议整局基线
    ├── *_g1/_final/_last          各 gate 的中间评估
    └── P2BC_s1205_probe/          字段敏感性探针（A/B 目标、auto/显式环、混合字段）
```

原始权重与原证据未动：`deliverable/model/actor_best.pt` 哈希 e70e3f61…（与接管时一致）；旧 run/诊断数据原样保留。新代码在 `experiments/lowlevel_training/`（goal_manager/reward_v2/sampler/train/evaluate/overnight/probe_fields/checks），对正式基础文件仅两处契约级修改（task_spec.py to_vector 契约 v2、vec_env.py 兼容扩展）与两处渲染依赖惰性导入（低层环境.py、mission_demo.py，物理不变）。

## 9. 复现

见 REPRODUCE.md（部署加载实测、快速/最终留出评估、双协议整局、字段探针、8 卡调度入口、单分支训练/精确恢复命令）。关键三点：容器 xiangjutao-kda-benchmark（Python 3.12 + torch 2.10.0+cu128 + 拷入的 scipy/pandas/gymnasium）；训练不需要 one-api/LLM；全部评估纯 Actor argmax。

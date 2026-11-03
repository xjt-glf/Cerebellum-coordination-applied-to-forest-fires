# MAPPO 两小时实施与验收交付

2026-11-03 21:55:25.763680

结论：已实现、实际启动并监督三种子训练与冻结配对评测；整体按 provisional/未完整通过指南验收交付。是否达到任务指标见下表，未完成或实现口径差异不能当作通过。

容器 xiangjutao-kda-bench；实际硬件 8×NVIDIA H800，torch 2.13.0+cu130。CPU worker 整局仿真及冻结底层 Actor 推理，GPU 仅批量训练新三动作 MAPPO。

|种子|实际完整局数|3/4/5/6 机局数|更新数|物理步|PID仍运行|
|---|---:|---|---:|---:|---|
|1103|1616|[404, 404, 404, 404]|101|969600|False|
|1104|1536|[384, 384, 384, 384]|96|921600|False|
|1105|1376|[344, 344, 344, 344]|86|825600|False|

完整 checkpoint 支持批次边界恢复；实际恢复测试从16局重跑到32局，Actor/Critic 最大绝对差均为0，见 recovery_verification.json。CUDA RNG ByteTensor 恢复修复前失败日志也保留。8→16 workers 的配置、旧/新PID见 worker_scaling.json。

每局600物理步、每20步高层决策；共享Actor每机3 logits，逐机clipped PPO，团队Critic/GAE每团队时间步一次。每批16局/480团队转移、4epoch、minibatch256、gamma .97、lambda .95、学习率3e-4、熵.01、梯度裁剪.5。

开发选 H3 与模型；先冻结清单再留出。H3 配置：{"threshold": 300, "relay_mode": "one", "burned_mean": 484.25, "episodes": 16}。

|训练种子|N|配对局数|过火下降|过火差95%区间|耗尽差|总覆盖差|任务及守护点估计门槛|
|---|---:|---:|---:|---|---:|---:|---|
|1103|3|30|-16.04%|[57.3, 121.4]|0.100|0.021|False|
|1103|4|30|-5.18%|[1.4, 57.1]|0.100|0.067|False|
|1103|5|30|-10.91%|[25.4, 81.3]|0.133|0.091|False|
|1103|6|30|-4.94%|[8.0, 41.8]|0.267|0.147|False|
|1103|7|30|-5.55%|[8.2, 46.3]|0.533|0.134|False|
|1103|8|30|-2.38%|[2.5, 19.5]|0.667|0.097|False|
|1104|3|30|-12.07%|[38.6, 95.5]|0.000|0.040|False|
|1104|4|30|-6.53%|[11.4, 60.5]|0.200|0.147|False|
|1104|5|30|-9.67%|[24.4, 68.3]|0.200|0.152|False|
|1104|6|30|-2.50%|[-2.8, 30.4]|0.400|0.206|False|
|1104|7|30|-4.33%|[1.4, 43.1]|1.167|0.205|False|
|1104|8|30|-2.78%|[2.6, 23.9]|1.267|0.192|False|
|1105|3|30|-9.71%|[31.5, 78.5]|0.000|0.103|False|
|1105|4|30|-3.63%|[-3.5, 42.1]|0.133|0.182|False|
|1105|5|30|-9.18%|[21.1, 65.6]|0.200|0.189|False|
|1105|6|30|-3.33%|[-2.7, 38.1]|0.633|0.224|False|
|1105|7|30|-2.99%|[1.4, 29.2]|0.733|0.227|False|
|1105|8|30|-3.23%|[4.6, 25.7]|1.433|0.205|False|

主规模3..6各自要求累计过火下降至少5%、耗尽增量≤0.1、总通信覆盖下降≤3个百分点；置信区间跨0不能宣称稳定改善。7/8仅为未见规模外推。该表的门槛仅为任务点估计，不替代完整工程契约或稳定性验收。

累计过火、活火、烧尽、扑灭、未合围暴露积分、探索、UAV/总通信、AoI尾部、最长失联、站外耗尽与保护覆盖分开保存在原始逐局JSONL及 final_acceptance.json。完整失败场景配对差保存在 paired_results.json。

契约：contracts.json 与 extra_contracts.json 包括3..8机20物理步、N=1 mask、非零padding、实体重排、真值干预隔离、独立进程目标一致、有效目标先保留去重、缺动作广播拒绝、空人员、充电保护、完整600步同seed重放/不同seed差异、原运行器固定计划逐步适配一致性。

计算延迟见 extra_contracts.json：单线程CPU，预热100次、测量500次，单机Actor、本地目标计算、四机批量Actor各有P50/P95/P99。两轮广播是假设；没有测量真实无线延迟。

额外结果：normalized_reward_development.json 为独立奖励v2开发比较（128局，各初始化种子不同，不能作严格因果消融）；legacy_reference.jsonl 为可patrol的旧H，动作/协议不同，仅作参考；robustness_summary.json 为每N两个冻结场景上的ID轮换与无队友/无人员推理消融，不能声称重训消融。

实现及验收剩余差异：

- Main selection uses 8 rather than 40 development scenes per fleet; H3 grid uses4.
- Training may end below2000 per seed; complete episodes reported separately.
- Not every100 episodes evaluated; budgeted early/mid/latest snapshots selected.
- Status protection_active feature is last-step transition override, not persistent latch.
- message_bytes counts minimal target snapshot/task messages, excludes full observation/history payload and recipient fanout.
- Target goal age resets on target change, not task-only age.
- Broadcast validates version, participant set and action range; no explicit packet timestamp/hash disagreement handling.
- No-teammate/no-person ablations are frozen-model inference perturbations, not separately trained ablations.
- Execution success diagnostic is per actual protected order segment, not original lowlevel skill acceptance rate.

观测16维自身/队友、8维人员集合、22维摘要和50维已知图；人员known_fire_flag因缺逐人公开事实置0；Actor不含ID、真值统计或奖励。目标按ID破对称，系统并非严格置换等变。广播提供理想公共地图和人员视图，不能推断真实丢包/延迟/分区鲁棒性。冻结底层 Actor 保留历史输入设计与技能限制。

底层冻结 SHA256：`dfa5cdd0f4b673567c04220af9d82ee55f16920d55398a02471a0d535d59982e`；与指南相同：True。选中 checkpoint 和源码哈希在 frozen_manifest.json，部署 actor/critic 及哈希在 deployment_exports.json。所有路径的宿主根为 /ssd3/xiangjutao/hrl_dqn_improve，容器根为 /workspace/hrl_dqn_improve。

复现命令（容器内，工作目录 /workspace/hrl_dqn_improve）：

```bash
python3 -m experiments.local_allocator.checks
python3 -m experiments.local_allocator.extra_checks
python3 -m experiments.local_allocator.train --seed 1103 --index 0 --workers 16 --batch-episodes 16 --seconds 3422 --device cuda:0 --resume /workspace/hrl_dqn_improve/research_outputs/local_allocator_mappo_20261103/v1_count_s1103/latest.pt
python3 -m experiments.local_allocator.evaluate --kind mappo --checkpoint <frozen_checkpoint_absolute_path> --count 30 --start 987000 --tag reproduction --workers 16
```

训练曲线 training_curves.png；配对过火差及区间 heldout_paired_burn.png。STATUS.md、workflow.log、worker_scaling.log 和各run/resumed_training.log包含真实进度及失败痕迹。

终验补充：发现 model.py 在21:37出现未预期的独立Actor.act padding包装回退。已保留原始漂移副本并恢复冻结归档中的精确版本，三份冻结清单的源码哈希再次全部匹配，空集合部署调用复测通过。1800局评测走未改变的批量Actor.forward，权重均匹配冻结选中模型；诊断在漂移前完成。原因未确认，不把早先编辑器超时直接当作原因。详见source_drift_incident.json和source_drift_model_213718.tar.gz。

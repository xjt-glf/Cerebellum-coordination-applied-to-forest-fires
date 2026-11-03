# 独立只读审计发现

- 奖励 L 必须使用 `int(e._active_true_burning_mask().sum())`。该方法返回二维 bool mask，排除已合围燃烧连通域；不能替换成所有活火格。
- 总通信 `comm_coverage_rate` 包括基地和任何 UAV 的几何覆盖，不要求 UAV 当前任务为 relay。`coverage_rate` 仅 UAV 覆盖；人员 `is_covered` 和 AoI 属于总通信口径。无人时在适配层显式定义为零，避免环境残留属性。
- 环境 `uav_id` 为 1-based，底层 Actor/orders 索引为 0-based；人员 `rescuer_id` 为 1-based。公开 ID 与数组索引必须显式转换。
- 冻结底层观测 `lowlevel/obs_builder.py:466` 包含 `fire_discovery_rate`、`burned_area_rate`、`rescuer_known_fire_rate`、`fire_diff` 真值统计。这是遗留底层限制；新高层 obs_v2 应隔离，但不得宣称端到端无真值泄漏。不要静默置零底层字段，因为会改变冻结执行器行为和训练分布。
- 低层 `MAX_AGENTS=8`；超过八架没有有效身份 one-hot。7/8 架仍需完整执行测试，不能仅凭初始化宣称泛化。
- `shared_cache(e)["M"]` 是允许的 UAV 已知图融合，适用于指南理想广播假设；不可使用真实 burning_state 生成高层地图、摘要或目标。

来源：独立 Explore 审计；未更改代码、权重或启动额外训练。

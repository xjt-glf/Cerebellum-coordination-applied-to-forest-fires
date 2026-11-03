# 逐页叙事
## 01 高层大脑的决策价值
- Role: evidence
- Message: Codex方案在已测三局优于启发式，模型与上下文差异引出研究问题。
- Visual: 零基线柱图＋右侧上下文比较与研究问题。
- Text: 过火553.7/533.0/445.3格；相对启发式−19.6%；3种子、冻结Actor、600步。
- Evidence: comprehensive_comparison_3seeds_20261022/analysis.json；十局正式千问对照作为可见补充。
- Source visual: 精确数值驱动的生成图，不引用参考图实验内容。
- Repair handle: 三柱数值、同种子口径与右侧待验证上下文问题。
## 02 大小脑闭环与长期记忆
- Role: method-overview
- Message: 决策输出结构化任务与建议，编码后由冻结RL执行；拟新增评分与记忆反馈闭环。
- Visual: 环境、决策LLM、任务编码、Actor为主路径，评分LLM和RAG为反馈路径。
- Text: 355维输入、9个离散动作、每20步评价、已实现/拟新增。
- Evidence: task_spec/obs_builder/world_snapshot/ChargeProtocol；研究方案文档。
- Source visual: 用户图只做关系参考，全新无衬线平面布局。
- Repair handle: 评分LLM不能直接输出物理动作；向量化不是虚构可训练嵌入。
## 03 反馈可能错判决策
- Role: problem
- Message: 20步状态差不等于决策贡献，充电造成的短期覆盖下降可能是必要代价。
- Visual: t0下发充电→t20覆盖暂降/仍在途中→后续到站，搭配错误批评与合理等待。
- Text: 案例示意；计划错误/执行偏差/收益未兑现。
- Evidence: 当前模型移动与能量规则；因果和评价时域分析。
- Source visual: 生成的机制示意，不能伪装为真实轨迹实验。
- Repair handle: 不凭20步未到判不服从；不可给示意图虚构实测分数。
## 04 执行感知的评价与记忆
- Role: mechanism-detail
- Message: 任务承诺、执行核验、时机判断与经验回填共同减少错误反馈传播。
- Visual: 四步闭环，上方主步骤，下方工作记忆到长期经验的分支，底部消融序列。
- Text: 保持/等待/局部修正；未兑现进工作记忆，已验证经验进RAG。
- Evidence: llm_critic_memory_novelty_20261024.md；Reflexion/ACE/COUNTERMEM相关工作。
- Source visual: 新设计方法图，明确候选机制尚待验证。
- Repair handle: 待验证与已验证经验不得混写；方法增益标记为待验证。

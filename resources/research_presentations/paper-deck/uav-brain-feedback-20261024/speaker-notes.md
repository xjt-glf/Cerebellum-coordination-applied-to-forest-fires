# 讲稿与来源
## 1 实验与上下文问题
数据为同三个历史种子的启发式、千问与Codex单航点方案，并非最新短航迹组。过火均值553.6667/533/445.3333；445.3333相对553.6667降低19.5665%。同Actor argmax、目标/充电协议、600步，Codex每20步自行决策。3种子为探索性结果。千问也有全量快照和最近5次决策，不能称其无上下文。完整10对中千问反而较差，不能推广所有大模型胜过启发式。
来源：research_outputs/comprehensive_comparison_3seeds_20261022/analysis.json、report.md；research_outputs/brain_comparison_20261022_recovered/report.md。
## 2 系统结构
现有task_spec确定性编码目标条件，与状态构成355维观测，Actor输出9种离散动作。评分LLM和RAG闭环为拟研究内容，非已部署功能。底层权重冻结；不采用参考图的任务嵌入网络与双Q结构。充电保护依然有效。
来源：lowlevel/task_spec.py、obs_builder.py、webapp/route_converter.py、charge_protocol.py。
## 3 反馈误判
充电示例是机制示意，无实测分数或曲线。前后变化不自动是决策因果贡献；记录实际轨迹、抢占和执行时间。20步内未到远目标不等于失败。
## 4 候选机制
本页为待验证方案，不宣称双LLM、RAG、反思或反事实验证本身原创。
相关工作：Reflexion https://arxiv.org/abs/2303.11366；ACE https://arxiv.org/abs/2510.04618；COUNTERMEM https://arxiv.org/abs/2609.31874；FRAMe https://arxiv.org/abs/2607.06964。
设计来源：research_outputs/llm_critic_memory_novelty_20261024.md。

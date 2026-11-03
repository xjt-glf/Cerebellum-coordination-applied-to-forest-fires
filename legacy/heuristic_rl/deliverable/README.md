# 无人机—地面人员协同灭火：低层强化学习策略（交付展示）

## 1. 这是什么

一个**分层强化学习**系统的低层执行器。上层（现在是规则规划器，将来接大模型
LLM Task Planner + ReAd critic）一次性给出全队无人机的任务与目标；低层是一个
**参数共享的 MAPPO 策略**，每架无人机独立读自己的观测与指令，输出 9 个离散动作之一。

任务链条：**无人机搜寻分散的火点 → 把火势认知地图回传给地面人员 → 人员绕火场外圈
把边界灭成隔火带 → 火场被合围后不再需要处理**。中间还要维持对人员的通信覆盖、
并轮换返航补给。

关键设定：人员对火情**一无所知**，全部认知来自无人机的回传。所以"无人机把地图传给人员"
是灭火能够发生的**必要前提**，而不是一个可选的加分项。

## 2. 文件说明

```
deliverable/
├── README.md                 本文件（总览 + 运行方法）
├── 1_S-A-R设计.md            状态空间 S / 动作空间 A / 奖励函数 R 的完整设计
├── 2_训练与效果.md            训练配置、曲线解读、验收数值、对照实验
├── simulate.py               ★ 仿真入口（可指定火点数/人员数/无人机数等参数）
├── model/
│   ├── actor_best.pt         成果权重（仅 Actor，部署只需这个，1.1 MB）
│   └── read_critic.pt        ReAd critic Q_θ(s, τ)：给上层任务分配打分的门限模型（0.6 MB）
├── curves/
│   ├── reward_curve_all.png  8 个超参变体的训练回报对比
│   ├── reward_curve_best.png 成果版本：训练回报 + 确定性评估回报
│   ├── metrics_best.png      成果版本 6 项评估指标随训练的变化
│   └── train_log_w4_h384.csv 原始训练日志（每次 update 一行）
└── demo/
    ├── mission_3fire_10r.gif            3 火点 / 10 人 / 4 机 / 600 步
    ├── mission_3fire_10r_commands.md    上层每 20 步下发的 JSON 指令与态势
    ├── mission_5fire_16r_6uav.gif       5 火点 / 16 人 / 6 机 / 500 步（全部合围）
    └── mission_random_baseline.gif      随机动作对照（全部无人机耗尽、无人灭火）
```

## 3. 怎么跑

环境：Python 3.10+，需要 `torch numpy scipy matplotlib pandas pillow`。
在项目根目录（`hrl_dqn_improve/`）下执行。

```bash
# 默认配置：100x100 地图、3 个火点、10 名人员、4 架无人机、600 步
python3 deliverable/simulate.py

# 换环境参数
python3 deliverable/simulate.py --fires 5 --rescuers 16 --uavs 6 --steps 500 --seed 31

# 同时输出 GIF 和上层指令日志
python3 deliverable/simulate.py --fires 3 --rescuers 10 --uavs 4 \
        --gif out.gif --fps 6

# 多回合统计均值（单回合方差较大，正式结论请看均值）
python3 deliverable/simulate.py --episodes 5

# 随机动作对照基线
python3 deliverable/simulate.py --random-policy
```

主要参数（`--help` 可看全部）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--fires` | 3 | 初始火点数量（位置每回合随机、互相分散） |
| `--rescuers` | 10 | 地面救援人员数量 |
| `--uavs` | 4 | 无人机数量，**支持 1–8**（观测里 agent-id one-hot 上限 8） |
| `--grid` | 100 | 任务区域边长 |
| `--steps` | 600 | 一回合时间步数 |
| `--spread-gain` | 1.0 | 火蔓延速度增益 |
| `--replan-every` | 20 | 上层多少步重规划一次 |
| `--charge-at` | 260 | 电量低于该值排队去补给（满电 600） |
| `--explore-until` | 0.70 | 探索率达到该值后，感知任务从搜寻转为绕火点巡逻 |

**同一套权重可以直接换编队规模**，不需要重训：Actor 是参数共享的，Critic 用
`[自身观测, 队友观测均值, 全局特征]` 的置换不变形式，所以 1–8 架都能跑。
实测 6 机 / 16 人 / 5 火点下：全图探索率 0.978、火情发现率 1.000、5 处火全部合围、
扑灭 432 格、0 台耗尽。

## 4. 一句话结果

| 指标 | 随机动作 | 训练后策略 |
|---|---|---|
| 全图探索率 | 0.180 | **0.918** |
| 火情发现率 | 0.034 | **0.661** |
| 人员已知火情比例 | 0.004 | **0.631** |
| 无人机通信覆盖率 | 0.089 | **0.685** |
| 扑灭格数 | 14 | **434** |
| 耗尽的无人机数 | 4.00 | **0.00** |

（100x100 地图、600 步、上层规划器每 25 步重规划，4 回合均值。详见 `2_训练与效果.md`）

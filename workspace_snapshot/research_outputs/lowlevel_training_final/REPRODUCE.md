# 复现命令（lowlevel_training_final）

所有命令在项目根 `/ssd3/xiangjutao/hrl_dqn_improve` 下执行；服务器容器：
`xiangjutao-kda-benchmark`（Python 3.12 + torch 2.10.0+cu128）。

## 1. 模型加载（部署兼容）

```bash
# A. 现有部署接口（webapp/policy_api 同一条路径）
python3 - << 'PY'
import sys; sys.path.insert(0, "/ssd3/xiangjutao/hrl_dqn_improve")
from lowlevel.policy_api import LowLevelController
ctl = LowLevelController("research_outputs/lowlevel_training_final/actor_selected.pt")
ctl.set_plan({"uavs": [
    {"task": "explore", "target": [75.0, 25.0]},
    {"task": "relay", "rescuer_id": 3, "handover": True},
    {"task": "charge"},
    {"task": "patrol", "target": None, "ring_radius": "auto"}]}, num_uavs=4)
# acts = ctl.act(env)  # env 是原始 GridEnv；返回 List[int] 长度=机数，元素 0..8
PY
# B. 实跑验证（合法任务指令、纯 Actor、无 waypoint_assist）
python3 experiments/lowlevel_training/checks/deploy_check.py \
    --actor research_outputs/lowlevel_training_final/actor_selected.pt --steps 120 \
    --out research_outputs/lowlevel_training_final/evidence/deploy_check.json
```

## 2. 技能验收（修正口径：本机链路 / 合法目标分层 / 每步解析环半径）

```bash
# 快速开发集（固定 8 场景种子 510000-510007，normal+scarce 双 regime）
python3 experiments/lowlevel_training/evaluate.py \
    --actor research_outputs/lowlevel_training_final/actor_selected.pt \
    --tag selected --split quick --workers 8
# 最终留出集（种子 520000 起 20 场景 × 2 regime，含轨迹）
python3 experiments/lowlevel_training/evaluate.py \
    --actor research_outputs/lowlevel_training_final/actor_selected.pt \
    --tag selected --split final --workers 12
# 旧 Actor 基线（同口径对照）
python3 experiments/lowlevel_training/evaluate.py \
    --actor deliverable/model/actor_best.pt --tag baseline_old --split final --workers 12
```

## 3. 整局回归（同一"目标完成后更新"协议，新旧 Actor 分别列出）

```bash
python3 experiments/lowlevel_training/evaluate.py \
    --actor research_outputs/lowlevel_training_final/actor_selected.pt \
    --tag selected --mission --protocol goal_update --episodes 20 --workers 8
python3 experiments/lowlevel_training/evaluate.py \
    --actor research_outputs/lowlevel_training_final/actor_selected.pt \
    --tag selected --mission --protocol legacy_h --episodes 20 --workers 8
python3 experiments/lowlevel_training/evaluate.py \
    --actor deliverable/model/actor_best.pt \
    --tag baseline_old --mission --protocol goal_update --episodes 20 --workers 8
```

## 4. 字段敏感性（验收 §4）

```bash
python3 experiments/lowlevel_training/probe_fields.py \
    --actor research_outputs/lowlevel_training_final/actor_selected.pt \
    --tag selected_probe --workers 4 --seeds 3
```

## 5. 训练合同（本轮实现的入口）

```bash
# 8 卡初筛（h200_night_plan.json 的 initial_trials）
python3 experiments/lowlevel_training/overnight.py \
    --plan research_outputs/h200_night_plan.json --wall-hours 8 launch
python3 experiments/lowlevel_training/overnight.py --plan research_outputs/h200_night_plan.json status

# 单 trial 训练（例：选中分支的配置）
CUDA_VISIBLE_DEVICES=0 python3 experiments/lowlevel_training/train.py \
    --trial-id P2BC_s1205 --gpu 0 --reward P2 --roles randomized \
    --init actor_only --init-actor <P3_s1301/actor_final_snapshot.pt> --seed 1205 \
    --total-steps 40000000 --n-envs 24 --envs-per-worker 2
# 精确恢复（模型+优化器+ValueNorm+步数+RNG）
CUDA_VISIBLE_DEVICES=0 python3 experiments/lowlevel_training/train.py \
    --trial-id P2BC_s1205 --init resume_exact \
    --init-resume-ckpt <run>/ckpt_latest.pt --reward P2 --seed 1205 ...

# 训练前契约检查（混合字段/去重/奖励排序/GAE 边界）
python3 experiments/lowlevel_training/checks/contract_check.py

# 行为克隆教师数据（P3/R2 分支用）
python3 experiments/lowlevel_training/teacher.py --out experiments/lowlevel_training/runs/teacher_bc.npz \
    --target 100000 --workers 12
```

## 6. 继续训练选中模型

```bash
CUDA_VISIBLE_DEVICES=0 python3 experiments/lowlevel_training/train.py \
    --trial-id selected_continue --gpu 0 --reward P2 --roles randomized \
    --init resume_exact \
    --init-resume-ckpt research_outputs/lowlevel_training_final/ckpt_resumable.pt \
    --seed 1205 --total-steps 40000000 --n-envs 24 --envs-per-worker 2
```

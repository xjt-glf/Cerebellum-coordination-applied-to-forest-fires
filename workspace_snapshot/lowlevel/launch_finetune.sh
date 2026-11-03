#!/bin/bash
# 续训阶段（可复用）：从已有权重接着训，任务采样固定为 mix+single。
#   v3: INIT=lowlevel/runs/v2_mb2/ckpt_best.pt PREFIX=v3 SEED0=10
#       动机 = v2 验收里 recharge×4 仍有 2 台耗尽、sensing×4 探索率 0.726 略低于 0.75，
#       而纯 mix 采样下"全队同任务"只占 1.2%，正是验收考的分布。
#   v4: INIT=lowlevel/runs/v3_lr2e4_ent5/ckpt_best.pt PREFIX=v4 SEED0=20
#       动机 = 修正 obs_builder.station_assignment（补给点全局去冲突匹配）后需要重训，
#       旧实现下那 2 台耗尽的机每步拿到的补给点目标都在翻转。
set -u
cd /workspace/xiangjutao/hrl_dqn_improve
mkdir -p lowlevel/runs/logs

INIT=${INIT:-lowlevel/runs/v2_mb2/ckpt_best.pt}
PREFIX=${PREFIX:-v3}
SEED0=${SEED0:-10}
STEPS=${STEPS:-2000000}
ENVS=${ENVS:-128}
EPW=${EPW:-4}
ROLLOUT=${ROLLOUT:-64}

launch() {  # gpu run_name extra_args...
  local gpu=$1; local name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  nohup python3 -u lowlevel/train_lowlevel.py \
      --n-envs $ENVS --envs-per-worker $EPW --rollout $ROLLOUT --total-steps $STEPS \
      --task-sampler mix+single --device cuda --n-minibatch 2 --resume "$INIT" \
      --eval-every 20 --eval-episodes 8 \
      --run-name "$name" "$@" \
      > lowlevel/runs/logs/$name.log 2>&1 &
  echo "launched $name on gpu$gpu pid=$!"
}

launch 0 ${PREFIX}_lr3e4      --seed $((SEED0 + 0)) --lr 3e-4 --entropy-coef 0.01
launch 1 ${PREFIX}_lr2e4_ent5 --seed $((SEED0 + 1)) --lr 2e-4 --entropy-coef 0.005
launch 2 ${PREFIX}_lr3e4_ent2 --seed $((SEED0 + 2)) --lr 3e-4 --entropy-coef 0.02
launch 3 ${PREFIX}_lr1e4_ent3 --seed $((SEED0 + 3)) --lr 1e-4 --entropy-coef 0.003

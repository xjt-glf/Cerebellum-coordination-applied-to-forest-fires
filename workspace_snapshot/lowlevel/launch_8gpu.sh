#!/bin/bash
# 第三轮训练（人员 0.5 格/步、绕火场外圈灭火、合围后不再侦查之后）。
# 以上一轮最优（lr 5e-4 / ent 0.01 / mb 2 / gamma 0.995）为中心做小范围扫参，
# 并用一个变体试更高频的重规划（--replan-every 12）来扩宽指令分布。
# 不再保留 mix+single 变体：两轮实测它都在退（回报涨、任务指标跌）。
set -u
cd /workspace/xiangjutao/hrl_dqn_improve
mkdir -p lowlevel/runs/logs

STEPS=${STEPS:-16000000}
ENVS=${ENVS:-96}
EPW=${EPW:-4}
ROLLOUT=${ROLLOUT:-64}
PREFIX=${PREFIX:-w4}
INIT=${INIT:-}

launch() {  # gpu name extra...
  local gpu=$1; local name=$2; shift 2
  local resume=""
  [ -n "$INIT" ] && resume="--resume $INIT"
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  nohup python3 -u lowlevel/train_lowlevel.py \
      --n-envs $ENVS --envs-per-worker $EPW --rollout $ROLLOUT --total-steps $STEPS \
      --device cuda --eval-every 40 --eval-episodes 5 $resume \
      --run-name "${PREFIX}_${name}" "$@" \
      > lowlevel/runs/logs/${PREFIX}_${name}.log 2>&1 &
  echo "launched ${PREFIX}_${name} on gpu$gpu pid=$!"
}

launch 0 base      --seed 300 --lr 5e-4 --entropy-coef 0.01  --n-minibatch 2 --gamma 0.995
launch 1 base2     --seed 301 --lr 5e-4 --entropy-coef 0.01  --n-minibatch 2 --gamma 0.995
launch 2 ent015    --seed 302 --lr 5e-4 --entropy-coef 0.015 --n-minibatch 2 --gamma 0.995
launch 3 lr3e4     --seed 303 --lr 3e-4 --entropy-coef 0.01  --n-minibatch 2 --gamma 0.995
launch 4 mb4       --seed 304 --lr 5e-4 --entropy-coef 0.01  --n-minibatch 4 --gamma 0.995
launch 5 h384      --seed 305 --lr 5e-4 --entropy-coef 0.01  --n-minibatch 2 --gamma 0.995 --hidden 384
launch 6 g999      --seed 306 --lr 5e-4 --entropy-coef 0.01  --n-minibatch 2 --gamma 0.999
launch 7 replan12  --seed 307 --lr 5e-4 --entropy-coef 0.01  --n-minibatch 2 --gamma 0.995 --replan-every 12

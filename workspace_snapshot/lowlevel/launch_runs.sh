#!/bin/bash
# 并行启动多组低层 MAPPO 训练（每组独占一块 GPU），最后取 eval 最优的一组作为成果版本。
# 采样是 CPU 瓶颈（env.step ~13ms 且重尾），GPU 只跑 actor/critic（act+update < 5% 时间），
# 所以提速靠两件事：worker 内批量跑环境（envs-per-worker）+ 多 GPU 并行跑多个变体。
set -u
cd /workspace/xiangjutao/hrl_dqn_improve
mkdir -p lowlevel/runs/logs

STEPS=${STEPS:-4000000}
ENVS=${ENVS:-128}
EPW=${EPW:-4}
ROLLOUT=${ROLLOUT:-64}

launch() {  # gpu run_name extra_args...
  local gpu=$1; local name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  nohup python3 -u lowlevel/train_lowlevel.py \
      --n-envs $ENVS --envs-per-worker $EPW --rollout $ROLLOUT --total-steps $STEPS \
      --task-sampler mix --device cuda \
      --eval-every 20 --eval-episodes 8 \
      --run-name "$name" "$@" \
      > lowlevel/runs/logs/$name.log 2>&1 &
  echo "launched $name on gpu$gpu pid=$!"
}

launch 0 v2_s0          --seed 0 --lr 5e-4 --entropy-coef 0.01
launch 1 v2_s1          --seed 1 --lr 5e-4 --entropy-coef 0.01
launch 2 v2_lr3e4_ent02 --seed 2 --lr 3e-4 --entropy-coef 0.02
launch 3 v2_mb2         --seed 3 --lr 5e-4 --entropy-coef 0.01 --n-minibatch 2

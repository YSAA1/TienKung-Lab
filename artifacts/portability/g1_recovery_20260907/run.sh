#!/bin/bash
set -euo pipefail
cd /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-recovery-20260907
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,3
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
isaac_nv=/home/nubot/isaac-sim-standalone-5.1.0-linux-x86_64/kit/python/lib/python3.11/site-packages/nvidia
export LD_LIBRARY_PATH="$isaac_nv/nvjitlink/lib:$isaac_nv/cusparse/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
bash scripts/nubot_run.sh -m torch.distributed.run --nproc_per_node=2 --master_port=29618 \
  artifacts/portability/g1_recovery_20260907/train.py \
  --task g1_loco_teacher --num_envs 2048 --seed 42 --distributed --headless \
  --max_iterations 30000 --experiment_name g1_recovery_30k --run_name g1_recovery_30k \
  --amp_expert_manifest "$PWD/legged_lab/envs/g1/datasets/motion_amp_expert_t4_rob2rob_full17_v1/_manifest.json" \
  > artifacts/portability/g1_recovery_20260907/train.log 2>&1

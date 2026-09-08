#!/usr/bin/env bash
# Independent recipe experiment; preserve the core-fix baseline on GPU0/2.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,3
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
isaac_nv="$HOME/isaac-sim-standalone-5.1.0-linux-x86_64/kit/python/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="$isaac_nv/nvjitlink/lib:$isaac_nv/cusparse/lib:${LD_LIBRARY_PATH:-}"
mkdir -p artifacts/portability/g1_vital_motion_20260908
bash scripts/nubot_run.sh -m torch.distributed.run --nproc_per_node=2 --master_port=29643 \
  legged_lab/scripts/train.py --task g1_loco_teacher --g1_motion_experiment vital_v1 \
  --num_envs 2048 --seed 42 --distributed --headless --max_iterations 30000 \
  --experiment_name g1_vital_motion --run_name vital_motion_v1 \
  --amp_expert_manifest "$PWD/legged_lab/envs/g1/datasets/motion_amp_expert_t4_rob2rob_full17_v1/_manifest.json" \
  > artifacts/portability/g1_vital_motion_20260908/train.log 2>&1

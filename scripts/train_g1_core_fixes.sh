#!/usr/bin/env bash
# Cold start after the core-fix acceptance gates; radial progress matches arm B.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
isaac_nv="$HOME/isaac-sim-standalone-5.1.0-linux-x86_64/kit/python/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="$isaac_nv/nvjitlink/lib:$isaac_nv/cusparse/lib:${LD_LIBRARY_PATH:-}"
bash scripts/nubot_run.sh -m torch.distributed.run --nproc_per_node=2 --master_port=29633 \
  legged_lab/scripts/train.py --task g1_loco_teacher --g1_progress_ab B \
  --num_envs 2048 --seed 42 --distributed --headless --max_iterations 30000 \
  --experiment_name g1_core_fixes --run_name g1_core_fixes_B30000 \
  --amp_expert_manifest "$PWD/legged_lab/envs/g1/datasets/motion_amp_expert_t4_rob2rob_full17_v1/_manifest.json" \
  > artifacts/portability/g1_core_fixes_20260908/train.log 2>&1

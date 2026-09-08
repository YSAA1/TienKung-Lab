#!/usr/bin/env bash
# One cold-start arm per tmux session; both use the same frozen checkout.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
arm="${1:?Expected A or B}"
case "$arm" in
  A) gpu_pair=0,2; master_port=29620 ;;
  B) gpu_pair=1,3; master_port=29621 ;;
  *) echo "Expected A or B" >&2; exit 2 ;;
esac
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu_pair"
isaac_nv="$HOME/isaac-sim-standalone-5.1.0-linux-x86_64/kit/python/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="$isaac_nv/nvjitlink/lib:$isaac_nv/cusparse/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
bash scripts/nubot_run.sh -m torch.distributed.run --nproc_per_node=2 --master_port="$master_port" \
  legged_lab/scripts/train.py --task g1_loco_teacher --g1_progress_ab "$arm" \
  --num_envs 2048 --seed 42 --distributed --headless --max_iterations 30000 \
  --experiment_name "g1_progress_${arm}" --run_name "g1_progress_${arm}" \
  --amp_expert_manifest "$PWD/legged_lab/envs/g1/datasets/motion_amp_expert_t4_rob2rob_full17_v1/_manifest.json" \
  > "artifacts/portability/g1_progress_ab_20260908/train_${arm}.log" 2>&1

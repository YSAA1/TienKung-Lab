#!/usr/bin/env bash
# G1 VITAL v3: curated LAFAN unitree_v6 AMP + end-to-end ramped plant DR.
# GPU0/2 only (vital_v2 still holds GPU1/3). Cold start, 30k iterations.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
case "$PWD" in
  *TienKung-Lab-g1-vital-v3-20260911) ;;
  *) echo "refusing to run outside the isolated v3 worktree: $PWD" >&2; exit 1 ;;
esac
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
isaac_nv="$HOME/isaac-sim-standalone-5.1.0-linux-x86_64/kit/python/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="$isaac_nv/nvjitlink/lib:$isaac_nv/cusparse/lib:${LD_LIBRARY_PATH:-}"
mkdir -p artifacts/portability/g1_vital_v3_20260911
bash scripts/nubot_run.sh -m torch.distributed.run --nproc_per_node=2 --master_port=29647 \
  legged_lab/scripts/train.py --task g1_loco_teacher --g1_motion_experiment vital_v3 \
  --num_envs 2048 --seed 42 --distributed --headless --max_iterations 30000 \
  --experiment_name g1_vital_motion --run_name vital_motion_v3 \
  --amp_expert_manifest "$PWD/legged_lab/envs/g1/datasets/motion_amp_expert_unitree_v6/_manifest.json" \
  > artifacts/portability/g1_vital_v3_20260911/train.log 2>&1

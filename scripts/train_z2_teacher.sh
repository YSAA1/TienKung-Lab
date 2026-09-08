#!/usr/bin/env bash
# Frozen-source cold start after the Z2 asset/AMP acceptance checks.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 90
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
isaac_nv="$HOME/isaac-sim-standalone-5.1.0-linux-x86_64/kit/python/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="$isaac_nv/nvjitlink/lib:$isaac_nv/cusparse/lib:${LD_LIBRARY_PATH:-}"
mkdir -p artifacts/z2_migration/formal_v1
printf '%s\n' "$$" > artifacts/z2_migration/formal_v1/launcher.pid
bash scripts/nubot_run.sh -m torch.distributed.run --nproc_per_node=2 --master_port=29653 \
  legged_lab/scripts/train.py --task z2_loco_teacher --num_envs 2048 --seed 42 \
  --distributed --headless --max_iterations 30000 --logger tensorboard \
  --experiment_name z2_loco_teacher_sparse --run_name z2_source_usd_teacher_v1 \
  --amp_expert_manifest "$PWD/legged_lab/envs/z2/datasets/motion_amp_expert/_manifest.json" \
  > artifacts/z2_migration/formal_v1/train.log 2>&1
training_exit=$?
printf '%s\n' "$training_exit" > artifacts/z2_migration/formal_v1/exit_code.txt
exit "$training_exit"

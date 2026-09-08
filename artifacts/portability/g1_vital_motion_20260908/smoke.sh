#!/usr/bin/env bash
set -uo pipefail
cd /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-vital-motion-20260908
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,3
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
isaac_nv="$HOME/isaac-sim-standalone-5.1.0-linux-x86_64/kit/python/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="$isaac_nv/nvjitlink/lib:$isaac_nv/cusparse/lib:${LD_LIBRARY_PATH:-}"
out=artifacts/portability/g1_vital_motion_20260908
mkdir -p "$out/smoke"
timeout --signal=INT --kill-after=30s 600s bash scripts/nubot_run.sh -m torch.distributed.run \
  --nproc_per_node=2 --master_port=29642 "$out/smoke.py" \
  --task g1_loco_teacher --g1_motion_experiment vital_v1 --num_envs 32 --seed 42 --distributed --headless \
  --max_iterations 2 --experiment_name g1_vital_motion_smoke --run_name smoke \
  --amp_expert_manifest "$PWD/legged_lab/envs/g1/datasets/motion_amp_expert_t4_rob2rob_full17_v1/_manifest.json" \
  > "$out/smoke.log" 2>&1
code=$?
echo "$code" > "$out/smoke.exit"
exit "$code"

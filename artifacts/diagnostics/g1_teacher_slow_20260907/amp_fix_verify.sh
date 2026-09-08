#!/usr/bin/env bash
set -euo pipefail
root=/home/nubot/phn_ws/t4_train/TienKung-Lab-g1-amp-fix-20260907
out=/tmp/g1_teacher_audit_20260907
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
timeout -k 15s 120s bash "$root/scripts/nubot_run.sh" -m torch.distributed.run --nproc_per_node=2 --master_port=29603 \
  "$out/reproduce_amp_distributed.py" "$out/amp_ddp_after.json" --require-consistent >"$out/amp_ddp_after.log" 2>&1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2
isaac_nv=/home/nubot/isaac-sim-standalone-5.1.0-linux-x86_64/kit/python/lib/python3.11/site-packages/nvidia
export LD_LIBRARY_PATH="$isaac_nv/nvjitlink/lib:$isaac_nv/cusparse/lib:${LD_LIBRARY_PATH:-}"
timeout -k 15s 420s bash "$root/scripts/nubot_run.sh" -m torch.distributed.run --nproc_per_node=2 --master_port=29604 \
  "$out/amp_fix_probe_train.py" --task g1_loco_teacher --num_envs 128 --distributed --headless --seed 42 \
  --max_iterations 3 --run_name amp_fix_runtime >"$out/amp_fix_runtime.log" 2>&1
echo 'AMP_FIX_VERIFIED' >"$out/amp_fix_verify_exit.txt"

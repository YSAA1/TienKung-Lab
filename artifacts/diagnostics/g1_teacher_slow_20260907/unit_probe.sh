#!/usr/bin/env bash
set -euo pipefail
root=/home/nubot/phn_ws/t4_train/TienKung-Lab-g1-action-units-20260907
out=/tmp/g1_teacher_audit_20260907
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
cd "$root"
timeout -k 15s 900s bash scripts/nubot_run.sh "$out/unit_reward_train.py" \
  --task g1_loco_teacher --num_envs 512 --seed 42 --max_iterations 200 --run_name audit_unit_rewards --headless \
  >"$out/action_ab_unit_rewards.log" 2>&1
echo 'unit_rewards completed' >>"$out/action_ab_exit.txt"

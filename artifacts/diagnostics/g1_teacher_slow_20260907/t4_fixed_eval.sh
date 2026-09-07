#!/usr/bin/env bash
set -u
root=/home/nubot/phn_ws/t4_train/TienKung-Lab-s11b-upright-tbslim
run=2026-08-22_01-12-08_t_sparse_lightlp_s11b_upright_tbslim
out=/tmp/g1_teacher_audit_20260907
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
cd "$root" || exit 1
sha256sum legged_lab/scripts/eval_t4_hurdle.py "logs/t4_loco_teacher_sparse/$run/model_8000.pt" >"$out/t4_m8000_lineage.sha256"
for terrain in flat stepping_stones; do
  extra=()
  if [[ "$terrain" != flat ]]; then extra=(--spawn_y_offset_m 0 --spawn_yaw_deg 0); fi
  timeout -k 15s 240s bash scripts/nubot_run.sh legged_lab/scripts/eval_t4_hurdle.py \
    --task t4_loco_teacher_sparse --num_envs 32 --episodes 32 --load_run "$run" \
    --checkpoint "logs/t4_loco_teacher_sparse/$run/model_8000.pt" \
    --output "$out/t4_m8000_${terrain}.json" --difficulty 0 --terrain_type "$terrain" \
    --command_vx 0.7 --headless "${extra[@]}" >"$out/t4_m8000_${terrain}.log" 2>&1
  echo "$terrain exit=$?" >>"$out/t4_m8000_exit.txt"
done

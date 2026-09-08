#!/usr/bin/env bash
set -u
Z2_ROOT=/home/nubot/phn_ws/t4_train/TienKung-Lab-z2-teacher-20260909
G1_ROOT=/home/nubot/phn_ws/t4_train/TienKung-Lab-g1-vital-motion-20260908
OUT="$Z2_ROOT/artifacts/z2_migration/formal_v1/evaluation/g1/1000"
mkdir -p "$OUT"
cd "$G1_ROOT" || exit 2
export CUDA_VISIBLE_DEVICES=0
bash scripts/nubot_run.sh "$Z2_ROOT/artifacts/z2_migration/formal_v1/evaluation/run-milestone-evidence.py" --mode eval --profile vital_v1 --project-root "$G1_ROOT" --evidence-manifest "$OUT/flat.lineage.json" --amp-expert-manifest "$G1_ROOT/legged_lab/envs/g1/datasets/motion_amp_expert_t4_rob2rob_full17_v1/_manifest.json" --task g1_loco_teacher --num_envs 32 --episodes 64 --load_run 2026-09-08_16-25-01_vital_motion_v1 --checkpoint "$G1_ROOT/logs/g1_vital_motion/2026-09-08_16-25-01_vital_motion_v1/model_1000.pt" --output "$OUT/flat.json" --terrain_type flat --difficulty 0.0 --command_vx 0.7 --device cuda:0 --headless > "$OUT/flat.log" 2>&1
RESULT=$?
printf '%s\n' "$RESULT" > "$OUT/flat.exit"
exit "$RESULT"

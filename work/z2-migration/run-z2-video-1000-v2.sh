#!/usr/bin/env bash
set -u
ROOT=/home/nubot/phn_ws/t4_train/TienKung-Lab-z2-teacher-20260909
OUT="$ROOT/artifacts/z2_migration/formal_v1/evaluation/z2/1000"
mkdir -p "$OUT"
cd "$ROOT" || exit 2
export CUDA_VISIBLE_DEVICES=2
bash scripts/nubot_run.sh "$ROOT/artifacts/z2_migration/formal_v1/evaluation/run-milestone-evidence.py" --mode play --profile registry --project-root "$ROOT" --evidence-manifest "$OUT/flat.replay-video.lineage.json" --amp-expert-manifest "$ROOT/legged_lab/envs/z2/datasets/motion_amp_expert/_manifest.json" --task z2_loco_teacher --checkpoint_path "$ROOT/logs/z2_loco_teacher_sparse/2026-09-09_03-45-12_z2_source_usd_teacher_v1/model_1000.pt" --terrain --terrain_types flat --difficulty 0.0 --command_vx 0.7 --num_envs 1 --seed 42 --duration 20 --record "$OUT/flat.replay.mp4" --device cuda:0 --headless > "$OUT/flat.replay-video.log" 2>&1
RESULT=$?
printf '%s\n' "$RESULT" > "$OUT/flat.replay-video.exit"
exit "$RESULT"

#!/usr/bin/env bash
set -u
shopt -s nullglob
ROOT=/home/nubot/phn_ws/t4_train/TienKung-Lab-z2-actionrate-20260909
OLD=/home/nubot/phn_ws/t4_train/TienKung-Lab-z2-teacher-20260909
BASE="$ROOT/artifacts/z2_migration/formal_v1"
mkdir -p "$BASE/monitor"
while true; do
  CHECKPOINTS=("$ROOT"/logs/z2_loco_teacher_sparse/*_z2_actionrate_001_v1/model_0.pt)
  if [ "${#CHECKPOINTS[@]}" -gt 0 ]; then break; fi
  if [ -f "$BASE/exit_code.txt" ]; then exit 2; fi
  sleep 10
done
if [ "${#CHECKPOINTS[@]}" != 1 ]; then exit 3; fi
RUN_DIR=$(dirname "${CHECKPOINTS[0]}")
printf '%s\n' "$RUN_DIR" > "$BASE/run_dir.txt"
cd "$ROOT" || exit 2
exec bash scripts/nubot_run.sh "$OLD/monitor-early-training.py" --run-dir "$RUN_DIR" --output-dir "$BASE/monitor" --training-exit "$BASE/exit_code.txt" > "$BASE/monitor/monitor.log" 2>&1

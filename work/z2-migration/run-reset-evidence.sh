#!/usr/bin/env bash
set -u
ROOT=/home/nubot/phn_ws/t4_train/TienKung-Lab-z2-reset-20260909
OLD=/home/nubot/phn_ws/t4_train/TienKung-Lab-z2-teacher-20260909
BASE="$ROOT/artifacts/z2_migration/formal_v1"
WRAPPER="$OLD/artifacts/z2_migration/formal_v1/evaluation/run-milestone-evidence.py"
AMP="$ROOT/legged_lab/envs/z2/datasets/motion_amp_expert/_manifest.json"
export CUDA_VISIBLE_DEVICES=0
cd "$ROOT" || exit 2
mkdir -p "$BASE/evaluation"
for ITER in 1000 2000 3000; do
  while [ ! -f "$BASE/monitor/iteration_$ITER.json" ]; do
    if [ -f "$BASE/exit_code.txt" ]; then exit 2; fi
    sleep 30
  done
  RUN_DIR=$(cat "$BASE/run_dir.txt")
  OUT="$BASE/evaluation/z2/$ITER"
  mkdir -p "$OUT"
  for TERRAIN in flat stepping_stones; do
    bash scripts/nubot_run.sh "$WRAPPER" --mode eval --profile registry --project-root "$ROOT" --evidence-manifest "$OUT/$TERRAIN.lineage.json" --amp-expert-manifest "$AMP" --task z2_loco_teacher --num_envs 32 --episodes 64 --load_run unused_absolute_checkpoint --checkpoint "$RUN_DIR/model_$ITER.pt" --output "$OUT/$TERRAIN.json" --terrain_type "$TERRAIN" --difficulty 0.0 --command_vx 0.7 --device cuda:0 --headless > "$OUT/$TERRAIN.log" 2>&1
    RESULT=$?
    printf '%s\n' "$RESULT" > "$OUT/$TERRAIN.exit"
    if [ "$RESULT" != 0 ]; then exit "$RESULT"; fi
  done
  if [ "$ITER" = 3000 ]; then
    for TERRAIN in flat stepping_stones; do
      bash scripts/nubot_run.sh "$WRAPPER" --mode play --profile registry --project-root "$ROOT" --evidence-manifest "$OUT/$TERRAIN.video.lineage.json" --amp-expert-manifest "$AMP" --task z2_loco_teacher --checkpoint_path "$RUN_DIR/model_$ITER.pt" --terrain --terrain_types "$TERRAIN" --difficulty 0.0 --command_vx 0.7 --num_envs 1 --seed 42 --duration 20 --record "$OUT/$TERRAIN.mp4" --device cuda:0 --headless > "$OUT/$TERRAIN.video.log" 2>&1
      RESULT=$?
      printf '%s\n' "$RESULT" > "$OUT/$TERRAIN.video.exit"
      if [ "$RESULT" != 0 ]; then exit "$RESULT"; fi
    done
  fi
  echo "Completed evidence $ITER"
done
printf '0\n' > "$BASE/evaluation/batch.exit"

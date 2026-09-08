#!/usr/bin/env bash
set -u
Z2_ROOT=/home/nubot/phn_ws/t4_train/TienKung-Lab-z2-teacher-20260909
G1_ROOT=/home/nubot/phn_ws/t4_train/TienKung-Lab-g1-vital-motion-20260908
BASE="$Z2_ROOT/artifacts/z2_migration/formal_v1"
WRAPPER="$BASE/evaluation/run-milestone-evidence.py"
export CUDA_VISIBLE_DEVICES=0
for ITER in 1000 2000 3000; do
  while [ ! -f "$BASE/monitor/iteration_$ITER.json" ]; do
    if [ -f "$BASE/exit_code.txt" ]; then echo "Training ended before $ITER"; exit 2; fi
    sleep 30
  done
  for ROBOT in g1 z2; do
    if [ "$ROBOT" = g1 ]; then
      ROOT="$G1_ROOT"; PROFILE=vital_v1; RUN=2026-09-08_16-25-01_vital_motion_v1; EXP=g1_vital_motion
      AMP="$ROOT/legged_lab/envs/g1/datasets/motion_amp_expert_t4_rob2rob_full17_v1/_manifest.json"
    else
      ROOT="$Z2_ROOT"; PROFILE=registry; RUN=2026-09-09_03-45-12_z2_source_usd_teacher_v1; EXP=z2_loco_teacher_sparse
      AMP="$ROOT/legged_lab/envs/z2/datasets/motion_amp_expert/_manifest.json"
    fi
    OUT="$BASE/evaluation/$ROBOT/$ITER"
    mkdir -p "$OUT"
    cd "$ROOT" || exit 2
    for TERRAIN in flat stepping_stones; do
      if [ -f "$OUT/$TERRAIN.exit" ] && [ "$(cat "$OUT/$TERRAIN.exit")" = 0 ] && [ -s "$OUT/$TERRAIN.json" ]; then continue; fi
      bash scripts/nubot_run.sh "$WRAPPER" --mode eval --profile "$PROFILE" --project-root "$ROOT" --evidence-manifest "$OUT/$TERRAIN.lineage.json" --amp-expert-manifest "$AMP" --task "${ROBOT}_loco_teacher" --num_envs 32 --episodes 64 --load_run "$RUN" --checkpoint "$ROOT/logs/$EXP/$RUN/model_$ITER.pt" --output "$OUT/$TERRAIN.json" --terrain_type "$TERRAIN" --difficulty 0.0 --command_vx 0.7 --device cuda:0 --headless > "$OUT/$TERRAIN.log" 2>&1
      RESULT=$?
      printf '%s\n' "$RESULT" > "$OUT/$TERRAIN.exit"
      if [ "$RESULT" != 0 ]; then echo "Failed $ROBOT $ITER $TERRAIN"; exit "$RESULT"; fi
    done
    if [ "$ITER" = 3000 ]; then
      for TERRAIN in flat stepping_stones; do
        bash scripts/nubot_run.sh "$WRAPPER" --mode play --profile "$PROFILE" --project-root "$ROOT" --evidence-manifest "$OUT/$TERRAIN.video.lineage.json" --amp-expert-manifest "$AMP" --task "${ROBOT}_loco_teacher" --checkpoint_path "$ROOT/logs/$EXP/$RUN/model_$ITER.pt" --terrain --terrain_types "$TERRAIN" --difficulty 0.0 --command_vx 0.7 --num_envs 1 --seed 42 --duration 20 --record "$OUT/$TERRAIN.mp4" --device cuda:0 --headless > "$OUT/$TERRAIN.video.log" 2>&1
        RESULT=$?
        printf '%s\n' "$RESULT" > "$OUT/$TERRAIN.video.exit"
        if [ "$RESULT" != 0 ]; then echo "Failed video $ROBOT $ITER $TERRAIN"; exit "$RESULT"; fi
      done
    fi
  done
  echo "Completed evidence $ITER"
done
printf '0\n' > "$BASE/evaluation/batch.exit"

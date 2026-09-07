#!/usr/bin/env bash
# One bounded cold-start run, then fixed behavior JSON and continuous replays.
# Invoke from tmux with scripts/nubot_run.sh available in this source snapshot.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p artifacts/portability/v4 logs
run_name=g1_official_teacher_v4
shopt -s nullglob
previous=(logs/g1_loco_teacher_sparse/*_${run_name}/model_*.pt)
if (( ${#previous[@]} )); then
  echo "Existing teacher-v4 checkpoints found; inspect lineage before restarting." >&2
  exit 1
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,3
isaac_nv=/home/nubot/isaac-sim-standalone-5.1.0-linux-x86_64/kit/python/lib/python3.11/site-packages/nvidia
export LD_LIBRARY_PATH="$isaac_nv/nvjitlink/lib:$isaac_nv/cusparse/lib:${LD_LIBRARY_PATH:-}"
printf 'TRAIN_START %s\n' "$(date -Is)" >artifacts/portability/v4/supervisor.log
if bash scripts/nubot_run.sh -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=29589 \
  legged_lab/scripts/train.py --task=g1_loco_teacher --num_envs=2048 --distributed --headless \
  --run_name "$run_name" --max_iterations 30000 >logs/g1_teacher_v4.log 2>&1; then
  printf 'TRAIN_DONE %s\n' "$(date -Is)" >>artifacts/portability/v4/supervisor.log
else
  printf 'TRAIN_FAILED %s\n' "$(date -Is)" >>artifacts/portability/v4/supervisor.log
  exit 1
fi
checkpoints=(logs/g1_loco_teacher_sparse/*_${run_name}/model_29999.pt)
if (( ${#checkpoints[@]} != 1 )); then
  echo 'Expected exactly one final checkpoint.' >&2
  exit 1
fi
checkpoint="${checkpoints[0]}"
sha256sum "$checkpoint" >artifacts/portability/v4/final_checkpoint.sha256
export CUDA_VISIBLE_DEVICES=1
for terrain in flat stepping_stones raised_pillars; do
  difficulties=(0)
  extra=()
  if [[ "$terrain" != flat ]]; then
    difficulties=(0 0.85)
    extra=(--spawn_y_offset_m 0 --spawn_yaw_deg 0)
  fi
  for difficulty in "${difficulties[@]}"; do
    stem="final_${terrain}_d${difficulty}"
    timeout -k 15s 480s bash scripts/nubot_run.sh legged_lab/scripts/eval_locomotion.py \
      --task g1_loco_teacher --num_envs 32 --episodes 32 --load_run "$run_name" \
      --checkpoint "$checkpoint" --output "artifacts/portability/v4/${stem}.json" \
      --difficulty "$difficulty" --terrain_type "$terrain" --command_vx 0.7 --headless "${extra[@]}" \
      >"artifacts/portability/v4/${stem}.log" 2>&1
  done
done
for terrain in stepping_stones raised_pillars; do
  timeout -k 15s 480s bash scripts/nubot_run.sh legged_lab/scripts/play.py \
    --task g1_loco_teacher --num_envs 1 --seed 42 --checkpoint_path "$checkpoint" \
    --command_vx 0.7 --terrain --terrain_types "$terrain" --difficulty 0 \
    --record "artifacts/portability/v4/final_${terrain}.mp4" --duration 16 --headless \
    >"artifacts/portability/v4/final_${terrain}_replay.log" 2>&1
done
printf 'EVALUATION_DONE %s\n' "$(date -Is)" >>artifacts/portability/v4/supervisor.log

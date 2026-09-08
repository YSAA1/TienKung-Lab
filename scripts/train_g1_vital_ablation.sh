#!/usr/bin/env bash
# Single-factor G1 motion ablation. Keep the full vital_v1 run as the positive control.
set -euo pipefail

profile="${1:?profile required}"
gpu="${2:?gpu index required}"
max_iterations="${3:-2000}"

case "$profile" in
  vital_termination_only|vital_gait_gate_off_only|vital_action_rate_only|vital_no_action_rate|vital_no_gait_gate_off) ;;
  *)
    echo "unknown ablation profile: $profile" >&2
    exit 2
    ;;
esac

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
isaac_nv="$HOME/isaac-sim-standalone-5.1.0-linux-x86_64/kit/python/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="$isaac_nv/nvjitlink/lib:$isaac_nv/cusparse/lib:${LD_LIBRARY_PATH:-}"

artifact_dir="artifacts/portability/g1_vital_ablation_20260908"
mkdir -p "$artifact_dir"
cat > "$artifact_dir/${profile}_launch.json" <<JSON
{
  "profile": "$profile",
  "gpu": "$gpu",
  "max_iterations": $max_iterations,
  "num_envs": 2048,
  "seed": 42,
  "experiment_name": "g1_vital_ablation",
  "run_name": "$profile",
  "amp_expert_manifest": "$PWD/legged_lab/envs/g1/datasets/motion_amp_expert_t4_rob2rob_full17_v1/_manifest.json"
}
JSON

bash scripts/nubot_run.sh legged_lab/scripts/train.py \
  --task g1_loco_teacher \
  --g1_motion_experiment "$profile" \
  --num_envs 2048 \
  --seed 42 \
  --headless \
  --max_iterations "$max_iterations" \
  --experiment_name g1_vital_ablation \
  --run_name "$profile" \
  --amp_expert_manifest "$PWD/legged_lab/envs/g1/datasets/motion_amp_expert_t4_rob2rob_full17_v1/_manifest.json" \
  > "$artifact_dir/${profile}.log" 2>&1

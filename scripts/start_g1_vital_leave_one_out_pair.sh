#!/usr/bin/env bash
# Launch the leave-one-out VITAL ablation after single-factor profiles failed.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p artifacts/portability/g1_vital_ablation_20260908

tmux new-session -d -s g1-ablate-no-action-rate \
  "cd '$PWD' && bash scripts/train_g1_vital_ablation.sh vital_no_action_rate 0 2000"

tmux new-session -d -s g1-ablate-no-gait-gate \
  "cd '$PWD' && bash scripts/train_g1_vital_ablation.sh vital_no_gait_gate_off 2 2000"

if ! tmux has-session -t g1-ablation-tb 2>/dev/null; then
  tmux new-session -d -s g1-ablation-tb \
    "cd '$PWD' && python3 -m tensorboard.main --logdir logs/g1_vital_ablation --host 0.0.0.0 --port 8044 --load_fast=false > artifacts/portability/g1_vital_ablation_20260908/tensorboard.log 2>&1"
fi

tmux ls

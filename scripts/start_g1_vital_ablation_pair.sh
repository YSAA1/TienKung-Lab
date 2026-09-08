#!/usr/bin/env bash
# Launch the requested two-way G1 VITAL ablation on freed GPUs.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p artifacts/portability/g1_vital_ablation_20260908

tmux new-session -d -s g1-ablate-action \
  "cd '$PWD' && bash scripts/train_g1_vital_ablation.sh vital_action_rate_only 0 2000"

tmux new-session -d -s g1-ablate-termination \
  "cd '$PWD' && bash scripts/train_g1_vital_ablation.sh vital_termination_only 2 2000"

if ! tmux has-session -t g1-ablation-tb 2>/dev/null; then
  tmux new-session -d -s g1-ablation-tb \
    "cd '$PWD' && python3 -m tensorboard.main --logdir logs/g1_vital_ablation --host 0.0.0.0 --port 8044 --load_fast=false > artifacts/portability/g1_vital_ablation_20260908/tensorboard.log 2>&1"
fi

tmux ls

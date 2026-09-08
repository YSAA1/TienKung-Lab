#!/usr/bin/env bash
set -euo pipefail
# Isaac FK 70D Z2 experts. Parent launches this in tmux. Does not start training.
OUT="${1:-legged_lab/envs/z2/datasets/motion_amp_expert}"
exec bash scripts/nubot_run.sh legged_lab/scripts/generate_z2_amp_expert.py --output-dir "$OUT" --headless

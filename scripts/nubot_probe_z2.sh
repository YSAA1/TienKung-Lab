#!/usr/bin/env bash
set -euo pipefail
# Isaac plant probe for the Z2 teacher. Parent launches this in tmux.
# Does not start training.
OUT="${1:-artifacts/z2_migration/isaac_probe.json}"
exec bash scripts/nubot_run.sh legged_lab/scripts/probe_z2_isaac.py --output "$OUT" --headless

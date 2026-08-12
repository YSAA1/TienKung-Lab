#!/usr/bin/env bash
# Launch a repo Python entry point under the nubot Isaac Sim runtime.
#
# The Isaac Sim 5.1 standalone bundle links against a newer libstdc++ than the
# system provides, so a conda env contributes only its lib directory to
# LD_LIBRARY_PATH while the interpreter stays Isaac Sim's own python.sh.
#
# Usage: scripts/nubot_run.sh legged_lab/scripts/train.py --task t4_loco_teacher ...
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACSIM_ROOT="${ISAACSIM_ROOT:-$HOME/isaac-sim-standalone-5.1.0-linux-x86_64}"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-$HOME/IsaacLab}"
ISAACLAB_LIB_ENV="${ISAACLAB_LIB_ENV:-$HOME/anaconda3/envs/isaaclab}"

if [ ! -x "$ISAACSIM_ROOT/python.sh" ]; then
  echo "missing Isaac Sim launcher at $ISAACSIM_ROOT/python.sh; set ISAACSIM_ROOT" >&2
  exit 1
fi
if [ ! -d "$ISAACLAB_ROOT/source/isaaclab" ]; then
  echo "missing IsaacLab sources at $ISAACLAB_ROOT/source; set ISAACLAB_ROOT" >&2
  exit 1
fi

export LD_LIBRARY_PATH="$ISAACLAB_LIB_ENV/lib:${LD_LIBRARY_PATH:-}"

python_path="$REPO_ROOT:$REPO_ROOT/rsl_rl"
for package in isaaclab isaaclab_assets isaaclab_rl isaaclab_tasks; do
  python_path="$python_path:$ISAACLAB_ROOT/source/$package"
done
export PYTHONPATH="$python_path:${PYTHONPATH:-}"

cd "$REPO_ROOT"
exec "$ISAACSIM_ROOT/python.sh" "$@"

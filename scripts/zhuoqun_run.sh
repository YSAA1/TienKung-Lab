#!/usr/bin/env bash
# Launch a repo Python entry point under the zhuoqun Isaac Sim runtime.
#
# W2 has not discovered a working Isaac Sim / IsaacLab install on zhuoqun yet.
# Do not copy nubot absolute paths as if they were facts. Set the three
# variables below after probing the machine; the script refuses to guess.
#
# Usage: ISAACSIM_ROOT=... ISAACLAB_ROOT=... ISAACLAB_LIB_ENV=... \
#        scripts/zhuoqun_run.sh legged_lab/scripts/train.py --task t4_walk --headless
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -z "${ISAACSIM_ROOT:-}" ] || [ -z "${ISAACLAB_ROOT:-}" ] || [ -z "${ISAACLAB_LIB_ENV:-}" ]; then
  echo "zhuoqun Isaac runtime is not frozen. Set ISAACSIM_ROOT, ISAACLAB_ROOT, and ISAACLAB_LIB_ENV." >&2
  echo "Last probe (2026-08-13): no python.sh / IsaacLab source under /home/zhuoqun; GPUs were occupied by MjLab stair traversal." >&2
  exit 1
fi

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

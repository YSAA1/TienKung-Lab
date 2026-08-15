#!/usr/bin/env bash
# Play a pulled checkpoint on this desktop without the zhuoqun Docker stack.
#
# env_isaaclab already has Isaac Sim 5.1. Force the in-repo rsl_rl ahead of
# PHP-main/whole_body_tracking on sys.path.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-/home/ssy/anaconda3/envs/env_isaaclab}"
if [ ! -x "$CONDA_ENV/bin/python" ]; then
  echo "missing $CONDA_ENV/bin/python" >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
exec "$CONDA_ENV/bin/python" "$REPO_ROOT/legged_lab/scripts/play.py" "$@"

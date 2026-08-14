#!/usr/bin/env bash
# Launch a repo Python entry point in the local t4-isaac-jammy:v2 container.
#
# Same contract as scripts/zhuoqun_run.sh (Isaac Sim 5.1 standalone + IsaacLab
# 2.1.0 + in-repo rsl_rl), plus X11 so GUI play works on this desktop.
#
# Usage:
#   scripts/local_run.sh legged_lab/scripts/play.py --task t4_vault_mimic --num_envs=1
#   scripts/local_run.sh legged_lab/scripts/play.py --task t4_vault_mimic --num_envs=1 --headless --record artifacts/eval/g1_play.mp4
# Long / GPU jobs must be wrapped in tmux.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACSIM_ROOT="${ISAACSIM_ROOT:-$HOME/isaac-sim-standalone-5.1.0-linux-x86_64}"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-$HOME/IsaacLab}"
ISAACLAB_LIB_DIR="${ISAACLAB_LIB_DIR:-$HOME/isaaclab-libs/lib}"
DOCKER_IMAGE="${DOCKER_IMAGE:-t4-isaac-jammy:v2}"
CACHE_ROOT="${CACHE_ROOT:-$HOME/docker/isaac-sim}"

if [ ! -x "$ISAACSIM_ROOT/python.sh" ] || [ ! -x "$ISAACSIM_ROOT/kit/kit" ]; then
  echo "Isaac Sim 5.1 standalone is incomplete at $ISAACSIM_ROOT (need kit/). Run scripts/setup_local_isaac_docker.sh" >&2
  exit 1
fi
if [ ! -d "$ISAACLAB_ROOT/source/isaaclab" ]; then
  echo "missing IsaacLab 2.1 sources at $ISAACLAB_ROOT/source; run scripts/setup_local_isaac_docker.sh" >&2
  exit 1
fi
if [ ! -d "$ISAACLAB_LIB_DIR" ]; then
  echo "missing libstdc++ dir at $ISAACLAB_LIB_DIR; run scripts/setup_local_isaac_docker.sh" >&2
  exit 1
fi
if ! docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
  echo "missing docker image $DOCKER_IMAGE; run scripts/setup_local_isaac_docker.sh" >&2
  exit 1
fi

mkdir -p \
  "$CACHE_ROOT/cache/kit" "$CACHE_ROOT/cache/ov" "$CACHE_ROOT/cache/pip" \
  "$CACHE_ROOT/cache/glcache" "$CACHE_ROOT/cache/computecache" \
  "$CACHE_ROOT/logs" "$CACHE_ROOT/data"

python_path="$REPO_ROOT:$REPO_ROOT/rsl_rl"
for package in isaaclab isaaclab_assets isaaclab_rl isaaclab_tasks; do
  python_path="$python_path:$ISAACLAB_ROOT/source/$package"
done

docker_args=(
  --rm
  --gpus all
  --network host
  --ipc host
  --ulimit memlock=-1
  --ulimit stack=67108864
  -e ACCEPT_EULA=Y
  -e PRIVACY_CONSENT=Y
  -e OMNI_KIT_ACCEPT_EULA=YES
  -e PYTHONUNBUFFERED=1
  -e "LD_LIBRARY_PATH=$ISAACLAB_LIB_DIR"
  -e "PYTHONPATH=$python_path"
  -v "$ISAACSIM_ROOT:$ISAACSIM_ROOT"
  -v "$ISAACLAB_ROOT:$ISAACLAB_ROOT"
  -v "$ISAACLAB_LIB_DIR:$ISAACLAB_LIB_DIR"
  -v "$REPO_ROOT:$REPO_ROOT"
  -v "$CACHE_ROOT/cache/kit:/root/.cache/kit"
  -v "$CACHE_ROOT/cache/ov:/root/.cache/ov"
  -v "$CACHE_ROOT/cache/pip:/root/.cache/pip"
  -v "$CACHE_ROOT/cache/glcache:/root/.cache/nvidia/GLCache"
  -v "$CACHE_ROOT/cache/computecache:/root/.nv/ComputeCache"
  -v "$CACHE_ROOT/logs:/root/.nvidia-omniverse/logs"
  -v "$CACHE_ROOT/data:/root/.local/share/ov/data"
  -w "$REPO_ROOT"
)
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  docker_args+=(-e "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
fi
if [ -n "${DISPLAY:-}" ]; then
  docker_args+=(-e "DISPLAY=$DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix)
  if [ -n "${XAUTHORITY:-}" ] && [ -f "$XAUTHORITY" ]; then
    docker_args+=(-e XAUTHORITY=/tmp/.Xauthority -v "$XAUTHORITY:/tmp/.Xauthority:ro")
  fi
fi

exec docker run "${docker_args[@]}" "$DOCKER_IMAGE" "$ISAACSIM_ROOT/python.sh" "$@"

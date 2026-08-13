#!/usr/bin/env bash
# Launch a repo Python entry point under the zhuoqun Isaac Sim runtime.
#
# zhuoqun's host OS cannot run Isaac Sim 5.1 natively, so the runtime lives in
# the `t4-isaac-jammy` container (Ubuntu 22.04 userspace + Vulkan/GL libs) with
# the host's Isaac Sim standalone, IsaacLab sources, and this repo bind-mounted
# at identical paths. A host-provided libstdc++ (isaaclab-libs) is prepended to
# LD_LIBRARY_PATH exactly like nubot_run.sh does with its conda env.
#
# Usage: scripts/zhuoqun_run.sh legged_lab/scripts/train_t4_vault_mimic.py --headless ...
# Long runs must be wrapped in tmux on the host.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACSIM_ROOT="${ISAACSIM_ROOT:-$HOME/isaac-sim-standalone-5.1.0-linux-x86_64}"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-$HOME/IsaacLab}"
ISAACLAB_LIB_DIR="${ISAACLAB_LIB_DIR:-$HOME/isaaclab-libs/lib}"
# v2 adds libxt6 (MaterialX/GPU-foundation dependency), vulkan-tools, zenity.
DOCKER_IMAGE="${DOCKER_IMAGE:-t4-isaac-jammy:v2}"
CACHE_ROOT="${CACHE_ROOT:-$HOME/docker/isaac-sim}"

if [ ! -x "$ISAACSIM_ROOT/python.sh" ]; then
  echo "missing Isaac Sim launcher at $ISAACSIM_ROOT/python.sh; set ISAACSIM_ROOT" >&2
  exit 1
fi
if [ ! -d "$ISAACLAB_ROOT/source/isaaclab" ]; then
  echo "missing IsaacLab sources at $ISAACLAB_ROOT/source; set ISAACLAB_ROOT" >&2
  exit 1
fi
if [ ! -d "$ISAACLAB_LIB_DIR" ]; then
  echo "missing libstdc++ dir at $ISAACLAB_LIB_DIR; set ISAACLAB_LIB_DIR" >&2
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

exec docker run "${docker_args[@]}" "$DOCKER_IMAGE" "$ISAACSIM_ROOT/python.sh" "$@"

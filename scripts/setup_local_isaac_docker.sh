#!/usr/bin/env bash
# Provision the local Isaac play Docker to match zhuoqun's training stack:
# Isaac Sim 5.1 standalone + IsaacLab 2.1.0 + t4-isaac-jammy:v2.
#
# Downloads use China-reachable mirrors / multi-connection, not Tailscale rsync.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Always install under $HOME. Do not inherit ISAACLAB_ROOT from the shell —
# that often points at the conda/2.3.2 checkout used by local training.
ISAACSIM_ROOT="${T4_ISAACSIM_ROOT:-$HOME/isaac-sim-standalone-5.1.0-linux-x86_64}"
ISAACLAB_ROOT="${T4_ISAACLAB_ROOT:-$HOME/IsaacLab}"
ISAACLAB_LIBS="${T4_ISAACLAB_LIBS:-$HOME/isaaclab-libs}"
DOCKER_IMAGE="${DOCKER_IMAGE:-t4-isaac-jammy:v2}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$HOME/Downloads}"
ISAACSIM_ZIP="$DOWNLOAD_DIR/isaac-sim-standalone-5.1.0-linux-x86_64.zip"
ISAACSIM_URL="${ISAACSIM_URL:-https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip}"
ISAACLAB_URL="${ISAACLAB_URL:-https://ghfast.top/https://github.com/isaac-sim/IsaacLab/archive/refs/tags/v2.1.0.tar.gz}"

need_sudo() {
  cat <<'EOF' >&2

本机还不能用 docker。nvidia-container-toolkit 已装好，还差把当前用户加进 docker 组。
在终端执行（会要一次密码），然后重新开一个 shell 再跑本脚本：

  sudo usermod -aG docker "$USER"
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  newgrp docker

EOF
}

have_isaacsim() {
  [[ -x "$ISAACSIM_ROOT/python.sh" && -x "$ISAACSIM_ROOT/kit/kit" ]]
}

have_isaaclab() {
  [[ -f "$ISAACLAB_ROOT/VERSION" && "$(tr -d '[:space:]' < "$ISAACLAB_ROOT/VERSION")" == "2.1.0" && -f "$ISAACLAB_ROOT/source/isaaclab/isaaclab/utils/io/yaml.py" ]]
}

ensure_libs() {
  mkdir -p "$ISAACLAB_LIBS/lib"
  if [[ ! -e "$ISAACLAB_LIBS/lib/libstdc++.so.6" ]]; then
    local src="$HOME/anaconda3/envs/env_isaaclab/lib"
    cp -a "$src"/libstdc++.so* "$src"/libgcc_s.so* "$ISAACLAB_LIBS/lib/"
  fi
}

download_isaaclab() {
  if have_isaaclab; then
    echo "[setup] IsaacLab 2.1.0 already at $ISAACLAB_ROOT"
    return
  fi
  echo "[setup] download IsaacLab v2.1.0 via $ISAACLAB_URL"
  local tmp
  tmp="$(mktemp -d)"
  curl -fL --retry 3 --retry-delay 2 -o "$tmp/IsaacLab-2.1.0.tar.gz" "$ISAACLAB_URL"
  tar -xzf "$tmp/IsaacLab-2.1.0.tar.gz" -C "$tmp"
  rm -rf "$ISAACLAB_ROOT"
  mv "$tmp"/IsaacLab-2.1.0 "$ISAACLAB_ROOT"
  rm -rf "$tmp"
  have_isaaclab
}

download_isaacsim() {
  if have_isaacsim; then
    echo "[setup] Isaac Sim 5.1 already at $ISAACSIM_ROOT"
    return
  fi
  mkdir -p "$DOWNLOAD_DIR"
  echo "[setup] download Isaac Sim 5.1 (8.2 GiB) with aria2c 16 connections"
  echo "        $ISAACSIM_URL"
  aria2c -c -x 16 -s 16 -k 1M --file-allocation=none \
    --console-log-level=notice \
    -d "$DOWNLOAD_DIR" \
    -o "$(basename "$ISAACSIM_ZIP")" \
    "$ISAACSIM_URL"
  echo "[setup] unzip Isaac Sim -> $ISAACSIM_ROOT"
  rm -rf "$ISAACSIM_ROOT"
  mkdir -p "$ISAACSIM_ROOT"
  unzip -q "$ISAACSIM_ZIP" -d "$ISAACSIM_ROOT"
  if [[ ! -x "$ISAACSIM_ROOT/python.sh" ]]; then
    # some zips wrap a single top-level directory
    local inner
    inner="$(find "$ISAACSIM_ROOT" -mindepth 2 -maxdepth 2 -name python.sh -print -quit)"
    if [[ -n "$inner" ]]; then
      shopt -s dotglob
      mv "$(dirname "$inner")"/* "$ISAACSIM_ROOT"/
      shopt -u dotglob
    fi
  fi
  if [[ -x "$ISAACSIM_ROOT/post_install.sh" ]]; then
    (cd "$ISAACSIM_ROOT" && ./post_install.sh)
  fi
  have_isaacsim
}

echo "[setup] IsaacLab 2.1.0"
download_isaaclab
ln -sfn "$ISAACSIM_ROOT" "$ISAACLAB_ROOT/_isaac_sim"

echo "[setup] isaaclab-libs (libstdc++)"
ensure_libs

echo "[setup] Isaac Sim 5.1 standalone"
download_isaacsim

if [[ ! -x "$ISAACSIM_ROOT/python.sh" || ! -d "$ISAACSIM_ROOT/kit" ]]; then
  echo "Isaac Sim incomplete at $ISAACSIM_ROOT" >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  if getent group docker | grep -qw "${USER:-}"; then
    exec sg docker -c "$(printf '%q ' "$0" "$@")"
  fi
  need_sudo
  exit 2
fi

echo "[setup] build $DOCKER_IMAGE"
docker build -t "$DOCKER_IMAGE" "$REPO_ROOT/docker/t4-isaac-jammy"

if [[ -n "${DISPLAY:-}" ]]; then
  xhost +SI:localuser:root >/dev/null 2>&1 || xhost +local: >/dev/null 2>&1 || true
fi

echo "[setup] IsaacLab pip deps into kit python (do not pin/replace torch)"
# IsaacLab 2.1 setup.py wants torch==2.5.1; Isaac Sim 5.1 already ships torch.
# flatdict 4.0.1 is sdist-only and needs pkg_resources at build time.
PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
"$REPO_ROOT/scripts/local_run.sh" -m pip install -q -i "$PIP_INDEX" 'setuptools<81' wheel
"$REPO_ROOT/scripts/local_run.sh" -m pip install -q -i "$PIP_INDEX" --no-build-isolation \
  'flatdict==4.0.1' 'prettytable==3.3.0' 'gymnasium' 'einops' \
  GitPython 'onnx==1.16.1' 'tensorboard==2.18.0' 'numpy<2' \
  h5py hydra-core moviepy 'protobuf>=3.20.2,<5.0.0'

echo "[setup] smoke: torch + CUDA inside the container"
"$REPO_ROOT/scripts/local_run.sh" -c 'import torch,sys; print(sys.version.split()[0], torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count())'

echo "[setup] ready:"
echo "  scripts/local_run.sh legged_lab/scripts/play.py --task t4_vault_mimic --num_envs=1 \\"
echo "    --load_run 2026-08-13_09-30-13 --checkpoint model_28500.pt"

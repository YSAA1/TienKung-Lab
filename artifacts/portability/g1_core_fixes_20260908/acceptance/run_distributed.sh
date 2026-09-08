#!/usr/bin/env bash
set -uo pipefail
cd /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-core-fixes-20260908
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
isaac_nv="$HOME/isaac-sim-standalone-5.1.0-linux-x86_64/kit/python/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="$isaac_nv/nvjitlink/lib:$isaac_nv/cusparse/lib:${LD_LIBRARY_PATH:-}"
out=artifacts/portability/g1_core_fixes_20260908/acceptance
python3 - <<'PY' || exit 1
import json
from pathlib import Path
out = Path('artifacts/portability/g1_core_fixes_20260908/acceptance')
for name in ('g1_mixed', 't4_mixed', 'g1_flat', 'reset_cache', 't4_reset_cache'):
    assert (out / (name + '.exit')).read_text().strip() == '0', name
    assert json.loads((out / (name + '.json')).read_text())['ok'], name
PY
timeout --signal=INT --kill-after=30s 900s bash scripts/nubot_run.sh -m torch.distributed.run \
  --nproc_per_node=2 --master_port=29632 "$out/distributed_probe.py" \
  --task g1_loco_teacher --g1_progress_ab B --num_envs 32 --seed 42 --distributed --headless \
  --max_iterations 2 --experiment_name g1_core_fixes_smoke --run_name integration \
  --amp_expert_manifest "$PWD/legged_lab/envs/g1/datasets/motion_amp_expert_t4_rob2rob_full17_v1/_manifest.json" \
  > "$out/distributed.log" 2>&1
code=$?
printf '%s\n' "$code" > "$out/distributed.exit"
exit "$code"

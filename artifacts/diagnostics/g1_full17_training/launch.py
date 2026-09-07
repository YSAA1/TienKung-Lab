# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

"""Launch the user-requested two-GPU, 30000-update full-data G1 training on nubot."""

import datetime
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/nubot/phn_ws/t4_train/TienKung-Lab-g1-unitree-v6-20260907")
OUT = ROOT / "artifacts/portability/g1_full17_30k_20260907"
DATA = ROOT / "legged_lab/envs/g1/datasets/motion_amp_expert_t4_rob2rob_full17_v1"
EVIDENCE = ROOT / "artifacts/portability/t4_rob2rob_full17_v1"
assert not OUT.exists()
validation = json.loads((EVIDENCE / "isaac_validation.json").read_text())
manifest = json.loads((DATA / "_manifest.json").read_text())
assert validation["passed"] and validation["total_frames"] == 3303
assert len(manifest["clips"]) == 17 and set(manifest["clips"]) == set(validation["clips"])
assert json.loads((EVIDENCE / "visual_review.json").read_text())["accepted_for_data_ablation"]
for name, clip in manifest["clips"].items():
    assert hashlib.sha256((DATA / f"{name}.txt").read_bytes()).hexdigest() == clip["sha256"]
    assert clip["sha256"] == validation["clips"][name]["sha256"]
reference = json.loads((ROOT / "artifacts/portability/v6/source_manifest.json").read_text())
for name, expected in reference["files"].items():
    assert hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == expected, name
subprocess.run(["tmux", "has-session", "-t", "g1-teacher-v6"], check=True)
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 29608))
source = Path("/tmp/g1_full17_formal_train.py").read_bytes()
assert hashlib.sha256(source).hexdigest() == "92c1666aed86d68e3d895f9fd322a7f471a6947788134a795f4afce0b0f3c274"
OUT.mkdir()
(OUT / "train.py").write_bytes(source)
(OUT / "train_snapshot.py.txt").write_bytes(source)
stopped = []
for group, session, relative in (
    ("control", "g1-reset-control", "artifacts/portability/reset_xy_ablation_20260907/control"),
    ("full17", "g1-amp-full17", "artifacts/portability/rob2rob_full17_ablation_20260907/full17"),
):
    path = ROOT / relative
    pid = json.loads((path / "ready.json").read_text())["pid"]
    proc = Path(f"/proc/{pid}/cmdline")
    command = proc.read_bytes().replace(b"\0", b" ").decode()
    assert str(ROOT) in command and f"--group {group}" in command, command
    stopped.append(
        {
            "pid": pid,
            "session": session,
            "command": command,
            "checkpoints": sorted(p.name for p in path.glob("model_*.pt")),
        }
    )
    os.kill(pid, signal.SIGTERM)
    for _ in range(40):
        if not proc.exists():
            break
        time.sleep(0.25)
    else:
        assert proc.read_bytes().replace(b"\0", b" ").decode() == command
        os.kill(pid, signal.SIGKILL)
    if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
        subprocess.run(["tmux", "kill-session", "-t", session], check=True)
if subprocess.run(["tmux", "has-session", "-t", "g1-amp-full17-tb"], capture_output=True).returncode == 0:
    subprocess.run(["tmux", "kill-session", "-t", "g1-amp-full17-tb"], check=True)
time.sleep(2)
with socket.socket() as sock:
    sock.bind(("0.0.0.0", 8038))
(OUT / "stopped_ablation_jobs.json").write_text(json.dumps(stopped, indent=2))
script = f"""#!/bin/bash
set -euo pipefail
cd {ROOT}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2
export PYTHONPATH="$PWD:${{PYTHONPATH:-}}"
isaac_nv=/home/nubot/isaac-sim-standalone-5.1.0-linux-x86_64/kit/python/lib/python3.11/site-packages/nvidia
export LD_LIBRARY_PATH="$isaac_nv/nvjitlink/lib:$isaac_nv/cusparse/lib:${{LD_LIBRARY_PATH:-}}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
bash scripts/nubot_run.sh -m torch.distributed.run --nproc_per_node=2 --master_port=29608 \\
  {OUT}/train.py --task g1_loco_teacher --num_envs 2048 --seed 42 --distributed --headless \\
  --max_iterations 30000 --experiment_name g1_full17_30k --run_name g1_full17_30k \\
  --amp_expert_manifest {DATA}/_manifest.json > {OUT}/train.log 2>&1
date -Is > {OUT}/completed.txt
"""
(OUT / "run.sh").write_text(script)
subprocess.run(["bash", "-n", str(OUT / "run.sh")], check=True)
lineage = {
    "created_at": datetime.datetime.now().astimezone().isoformat(),
    "request": "Full 17 T4 motions, two GPUs, cold full training for 30000 updates; no ablation.",
    "runtime_source_commit": "c125f74d50b9d1aef70ae6ce8bd8b3fcb7b26d52",
    "verified_runtime_files": len(reference["files"]),
    "train_sha256": hashlib.sha256(source).hexdigest(),
    "data_manifest_sha256": hashlib.sha256((DATA / "_manifest.json").read_bytes()).hexdigest(),
    "physical_gpus": [0, 2],
    "world_size": 2,
    "num_envs_per_rank": 2048,
    "global_num_envs": 4096,
    "updates": 30000,
    "steps_per_env": 24,
    "seed": 42,
    "cold_start": True,
    "amp_clips": 17,
    "amp_frames": 3303,
    "class_probabilities": manifest["class_probabilities"],
    "log_root": str(ROOT / "logs/g1_full17_30k"),
    "tensorboard": "http://100.100.188.39:8038/#scalars",
    "behavior_success_verified": False,
}
(OUT / "lineage.json").write_text(json.dumps(lineage, indent=2))
subprocess.run(["tmux", "new-session", "-d", "-s", "g1-full17-30k", f"bash {OUT}/run.sh"], check=True)
subprocess.run(
    [
        "tmux",
        "new-session",
        "-d",
        "-s",
        "g1-full17-30k-tb",
        (
            f"/usr/bin/python3 -m tensorboard.main --logdir {ROOT}/logs/g1_full17_30k --host 0.0.0.0 --port 8038"
            f" --load_fast=false > {OUT}/tensorboard.log 2>&1"
        ),
    ],
    check=True,
)
print(json.dumps(lineage))

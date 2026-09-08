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

"""Run once on nubot to replace the random-velocity treatment with the data ablation."""

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
OUT = ROOT / "artifacts/portability/rob2rob_amp_ablation_20260907"
OLD = ROOT / "artifacts/portability/reset_xy_ablation_20260907"
DATA = ROOT / "legged_lab/envs/g1/datasets/motion_amp_expert_t4_rob2rob_v1"
TRAIN = ROOT / "artifacts/diagnostics/g1_curated_ablation/train.py"
assert not OUT.exists()
for session in ("g1-reset-control", "g1-teacher-v6"):
    subprocess.run(["tmux", "has-session", "-t", session], check=True)
validation = json.loads((ROOT / "artifacts/portability/t4_rob2rob_v1/isaac_validation.json").read_text())
assert validation["passed"] and validation["total_frames"] == 287
review = json.loads((ROOT / "artifacts/portability/t4_rob2rob_v1/visual_review.json").read_text())
assert review["accepted_for_data_ablation"]
for name, clip in validation["clips"].items():
    assert hashlib.sha256((DATA / f"{name}.txt").read_bytes()).hexdigest() == clip["sha256"]
reference = json.loads((ROOT / "artifacts/portability/v6/source_manifest.json").read_text())
for name, expected in reference["files"].items():
    assert hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == expected, name
with socket.socket() as sock:
    sock.bind(("0.0.0.0", 8037))

pid = json.loads((OLD / "random_xy/ready.json").read_text())["pid"]
cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
assert str(ROOT) in cmdline and "--group random_xy" in cmdline, cmdline
OUT.mkdir()
stopped = {
    "pid": pid,
    "command": cmdline,
    "time": datetime.datetime.now().astimezone().isoformat(),
    "checkpoints": sorted(p.name for p in (OLD / "random_xy").glob("model_*.pt")),
    "reason": "User requested new AMP data ablation immediately after data validation.",
}
os.kill(pid, signal.SIGTERM)
for _ in range(40):
    if not Path(f"/proc/{pid}").exists():
        break
    time.sleep(0.25)
else:
    # Recheck identity before escalating this exact experiment process.
    assert Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode() == cmdline
    os.kill(pid, signal.SIGKILL)
if subprocess.run(["tmux", "has-session", "-t", "g1-reset-random_xy"], capture_output=True).returncode == 0:
    subprocess.run(["tmux", "kill-session", "-t", "g1-reset-random_xy"], check=True)
(OUT / "stopped_random_xy.json").write_text(json.dumps(stopped, indent=2))
(OUT / "control").symlink_to(OLD / "control", target_is_directory=True)
dest = OUT / "rob2rob"
dest.mkdir()
script = (OLD / "run_random_xy.sh").read_text()
script = script.replace(str(OLD / "random_xy"), str(dest))
script = script.replace(str(ROOT / "artifacts/diagnostics/g1_reset_ablation/train.py"), str(TRAIN))
script = script.replace("--group random_xy", "--group rob2rob")
script = script.replace(str(OLD / "start.allowed"), str(OUT / "start.allowed"))
script = script.replace("--load_run random_xy", "--load_run rob2rob")
(OUT / "run_rob2rob.sh").write_text(script)
subprocess.run(["bash", "-n", str(OUT / "run_rob2rob.sh")], check=True)
manifest = {
    "created_at": datetime.datetime.now().astimezone().isoformat(),
    "runtime_source_commit": "c125f74d50b9d1aef70ae6ce8bd8b3fcb7b26d52",
    "verified_runtime_files": len(reference["files"]),
    "train_sha256": hashlib.sha256(TRAIN.read_bytes()).hexdigest(),
    "data_manifest_sha256": hashlib.sha256((DATA / "_manifest.json").read_bytes()).hexdigest(),
    "control": str(OLD / "control"),
    "treatment": str(dest),
    "seed": 42,
    "num_envs": 2048,
    "updates": 4000,
    "steps_per_env": 24,
    "cold_start": True,
    "difference": "AMP expert dataset only; both reset root velocities remain zero.",
    "shared_class_probabilities": {"walk_forward": 0.625, "run": 0.375},
    "limitations": [
        "Single seed exploratory data ablation.",
        "Control starts earlier; compare equal updates/samples.",
        "Two short retargeted clips; no dynamic-trackability guarantee.",
    ],
    "tensorboard": "http://100.100.188.39:8037/#scalars",
}
(OUT / "lineage.json").write_text(json.dumps(manifest, indent=2))
subprocess.run(["tmux", "new-session", "-d", "-s", "g1-amp-rob2rob", f"bash {OUT}/run_rob2rob.sh"], check=True)
subprocess.run(
    [
        "tmux",
        "new-session",
        "-d",
        "-s",
        "g1-amp-data-tb",
        (
            f"/usr/bin/python3 -m tensorboard.main --logdir {OUT} --host 0.0.0.0 --port 8037 --load_fast=false >"
            f" {OUT}/tensorboard.log 2>&1"
        ),
    ],
    check=True,
)
print(json.dumps(manifest))

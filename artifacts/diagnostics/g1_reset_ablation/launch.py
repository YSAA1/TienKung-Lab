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

"""Run on nubot system Python after copying train.py next to this file."""

import datetime
import hashlib
import json
import socket
import subprocess
from pathlib import Path

ROOT = Path("/home/nubot/phn_ws/t4_train/TienKung-Lab-g1-unitree-v6-20260907")
OUT = ROOT / "artifacts/portability/reset_xy_ablation_20260907"
TRAIN = ROOT / "artifacts/diagnostics/g1_reset_ablation/train.py"
assert not OUT.exists(), f"Refusing to overwrite existing experiment: {OUT}"
assert subprocess.run(["tmux", "has-session", "-t", "g1-teacher-v6"], capture_output=True).returncode == 0
gpu_rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
gpu_ids = dict(row.strip().split(", ") for row in gpu_rows.splitlines())
apps = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
    text=True,
)
assert all(gpu_ids[str(gpu)] not in apps for gpu in (0, 2)), apps
with socket.socket() as check_port:
    check_port.bind(("0.0.0.0", 8036))
reference = json.loads((ROOT / "artifacts/portability/v6/source_manifest.json").read_text())
verified = {}
for name, expected in reference["files"].items():
    digest = hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert digest == expected, name
    verified[name] = digest
OUT.mkdir(parents=True)
base = (ROOT / "artifacts/portability/v6/early_evaluation/run.sh").read_text().split("exec >")[0]
manifest = {
    "created_at": datetime.datetime.now().astimezone().isoformat(),
    "runtime_source_commit": "c125f74d50b9d1aef70ae6ce8bd8b3fcb7b26d52",
    "train_script_sha256": hashlib.sha256(TRAIN.read_bytes()).hexdigest(),
    "verified_runtime_file_count": len(verified),
    "seed": 42,
    "envs_per_group": 2048,
    "updates_per_group": 4000,
    "steps_per_env": 24,
    "samples_per_group": 2048 * 4000 * 24,
    "cold_start": True,
    "single_task_difference": "reset root linear velocities x/y: zero vs Uniform(-0.5, 0.5) m/s",
    "evaluation": "Same frozen G1 task defaults for both groups: zero root velocity and official joint pose",
    "limitations": [
        "One paired seed is an exploratory causal comparison, not multi-seed proof.",
        "Each group uses one GPU; comparisons to historical two-GPU v6 are contextual only.",
        "The model_3999 file represents 4000 completed updates.",
    ],
    "groups": {},
    "tensorboard": "http://100.100.188.39:8036/#scalars",
}
for group, gpu in (("control", 0), ("random_xy", 2)):
    dest = OUT / group
    dest.mkdir()
    script = OUT / f"run_{group}.sh"
    lines = [
        base.replace("CUDA_VISIBLE_DEVICES=0", f"CUDA_VISIBLE_DEVICES={gpu}"),
        "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4",
        f"exec > {dest}/supervisor.log 2>&1",
        (
            f"bash scripts/nubot_run.sh {TRAIN} --group {group} --output {dest} --start-file {OUT}/start.allowed"
            f" --headless > {dest}/train.log 2>&1"
        ),
        f"test -s {dest}/model_3999.pt",
        f"sha256sum {dest}/model_3999.pt > {dest}/checkpoint.sha256",
    ]
    for terrain in ("flat", "stepping_stones", "raised_pillars"):
        extra = "" if terrain == "flat" else " --spawn_y_offset_m 0 --spawn_yaw_deg 0"
        lines.append(
            "timeout -k 15s 480s bash scripts/nubot_run.sh legged_lab/scripts/eval_locomotion.py --task"
            f" g1_loco_teacher --num_envs 32 --episodes 32 --seed 42 --load_run {group} --checkpoint"
            f" {dest}/model_3999.pt --terrain_type {terrain} --difficulty 0 --command_vx .7 --headless{extra} --output"
            f" {dest}/{terrain}.json > {dest}/{terrain}.log 2>&1"
        )
        lines.append(f"test -s {dest}/{terrain}.json")
    for terrain in ("flat", "stepping_stones"):
        video = dest / f"{terrain}.mp4"
        lines.append(
            "timeout -k 15s 480s bash scripts/nubot_run.sh legged_lab/scripts/play.py --task g1_loco_teacher"
            f" --num_envs 1 --seed 42 --checkpoint_path {dest}/model_3999.pt --command_vx .7 --duration 16 --headless"
            f" --terrain --terrain_types {terrain} --difficulty 0 --record {video} > {dest}/{terrain}_replay.log 2>&1"
        )
        lines.extend(
            [
                f"if [[ ! -s {video} ]]; then",
                f"  test -s {video.with_suffix('.gif')}",
                (
                    "  env -u LD_LIBRARY_PATH /usr/bin/ffmpeg -hide_banner -loglevel error -y -i"
                    f" {video.with_suffix('.gif')} -vf 'setpts=N/(12.5*TB)' -r 12.5 -c:v libx264 -crf 18 -pix_fmt"
                    f" yuv420p -movflags +faststart {video}"
                ),
                "fi",
                f"test -s {video}",
                f"sha256sum {video} > {video.with_suffix('.sha256')}",
            ]
        )
    lines.append(f"date -Is > {dest}/completed.txt")
    script.write_text("\n".join(lines) + "\n")
    subprocess.run(["bash", "-n", str(script)], check=True)
    manifest["groups"][group] = {
        "gpu": gpu,
        "tmux": f"g1-reset-{group}",
        "directory": str(dest),
    }
(OUT / "lineage.json").write_text(json.dumps(manifest, indent=2))
for group in manifest["groups"]:
    subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            f"g1-reset-{group}",
            f"bash {OUT}/run_{group}.sh",
        ],
        check=True,
    )
subprocess.run(
    [
        "tmux",
        "new-session",
        "-d",
        "-s",
        "g1-reset-ab-tb",
        (
            f"/usr/bin/python3 -m tensorboard.main --logdir {OUT} --host 0.0.0.0 --port 8036 --load_fast=false >"
            f" {OUT}/tensorboard.log 2>&1"
        ),
    ],
    check=True,
)
print(json.dumps(manifest))

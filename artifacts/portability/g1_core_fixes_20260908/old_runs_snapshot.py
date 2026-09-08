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

"""Inventory old A/B processes and retained artifacts; never stops or deletes anything."""

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

OLD = Path("/home/nubot/phn_ws/t4_train/TienKung-Lab-g1-progress-ab-20260908")
OUT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--stopped", action="store_true")
args = parser.parse_args()
processes = []
for proc in Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    try:
        command = (proc / "cmdline").read_bytes().decode().split("\0")
        if "legged_lab/scripts/train.py" not in command or Path(os.readlink(proc / "cwd")) != OLD:
            continue
        environment = dict(
            item.split("=", 1) for item in (proc / "environ").read_bytes().decode().split("\0") if "=" in item
        )
        processes.append(
            {
                "pid": int(proc.name),
                "command": command,
                "environment": {key: environment.get(key) for key in ("CUDA_VISIBLE_DEVICES", "LOCAL_RANK", "RANK")},
            }
        )
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        continue
groups = {}
for arm in ("A", "B"):
    directory = OLD / "logs" / f"g1_progress_{arm}"
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    records = []
    for path in files:
        item = {"path": str(path), "size": path.stat().st_size}
        if args.stopped:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            item["sha256"] = digest.hexdigest()
        records.append(item)
    groups[arm] = {"checkpoint_count": sum(path.suffix == ".pt" for path in files), "files": records}
result = {"time": datetime.now().astimezone().isoformat(), "root": str(OLD), "processes": processes, "groups": groups}
if args.stopped:
    assert not processes, processes
    before = json.loads((OUT / "old_before_stop.json").read_text())
    for arm in ("A", "B"):
        assert groups[arm]["checkpoint_count"] >= before["groups"][arm]["checkpoint_count"]
(OUT / ("old_stopped.json" if args.stopped else "old_before_stop.json")).write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {"processes": processes, "checkpoints": {arm: group["checkpoint_count"] for arm, group in groups.items()}}
    )
)

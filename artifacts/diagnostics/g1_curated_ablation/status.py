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

"""Snapshot live status without changing trainers; run with nubot system Python."""

import datetime
import json
import math
import subprocess
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

OUT = Path(
    "/home/nubot/phn_ws/t4_train/TienKung-Lab-g1-unitree-v6-20260907/artifacts/portability/rob2rob_amp_ablation_20260907"
)
result = {"time": datetime.datetime.now().astimezone().isoformat(), "groups": {}}
for group in ("control", "rob2rob"):
    path = OUT / group
    status = {
        "session_alive": (
            subprocess.run(
                ["tmux", "has-session", "-t", {"control": "g1-reset-control", "rob2rob": "g1-amp-rob2rob"}[group]],
                capture_output=True,
            ).returncode
            == 0
        ),
        "checkpoints": sorted(file.name for file in path.glob("model_*.pt")),
        "complete": (path / "completed.txt").exists(),
    }
    events = sorted(path.glob("events.out.tfevents.*"))
    if events:
        acc = EventAccumulator(str(events[-1]), size_guidance={"scalars": 100})
        acc.Reload()
        status["metrics"] = {}
        for tag in (
            "Train/mean_reward",
            "Train/mean_episode_length",
            "Loss/learning_rate",
            "Loss/value_function",
            "Loss/surrogate",
            "Loss/amp",
            "Policy/mean_noise_std",
            "Reset/collapsed",
        ):
            if tag not in acc.Tags()["scalars"]:
                continue
            rows = acc.Scalars(tag)
            values = [row.value for row in rows[-20:]]
            status["metrics"][tag] = {
                "step": rows[-1].step,
                "last": rows[-1].value,
                "mean_last20": sum(values) / len(values),
                "finite": all(math.isfinite(value) for value in values),
            }
    with (path / "train.log").open("rb") as log:
        log.seek(0, 2)
        log.seek(max(0, log.tell() - 900))
        status["log_tail"] = log.read().decode(errors="replace")
    result["groups"][group] = status
result["v6_session_alive"] = (
    subprocess.run(["tmux", "has-session", "-t", "g1-teacher-v6"], capture_output=True).returncode == 0
)
(OUT / "latest_health.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result))

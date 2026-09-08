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

"""Read the formal core-fix run without touching training state."""

import json
import math
from datetime import datetime
from pathlib import Path

import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
events = sorted((ROOT / "logs/g1_core_fixes").glob("*/events.out.tfevents.*"), key=lambda path: path.stat().st_mtime)
assert events, "formal run has no TensorBoard event file yet"
event = events[-1]
run = event.parent
acc = EventAccumulator(str(event), size_guidance={"scalars": 30}).Reload()
tags = acc.Tags()["scalars"]
metrics = {}
for tag in tags:
    values = acc.Scalars(tag)
    if values and tag.startswith(("Loss/", "Policy/", "Perf/", "Progress/all/")):
        metrics[tag] = {"step": values[-1].step, "value": values[-1].value}
env = yaml.load((run / "params/env.yaml").read_text(), Loader=yaml.BaseLoader)
agent = yaml.load((run / "params/agent.yaml").read_text(), Loader=yaml.BaseLoader)
assert agent["max_iterations"] == "30000"
assert agent["resume"].lower() == "false"
assert agent["num_steps_per_env"] == "24" and env["scene"]["num_envs"] == "2048"
assert env["lightlp_promotion_distance"] == "max_radial"
assert env["sparse_command_min_speed_scale"] == "1.0"
assert len(agent["amp_motion_files"]) == 17
assert all(float(value) == 0.05 for value in agent["min_normalized_std"])
finite = all(math.isfinite(row.value) for tag in tags for row in acc.Scalars(tag)[-20:])
assert finite, "nonfinite recent training scalars"
result = {
    "time": datetime.now().astimezone().isoformat(),
    "run": str(run),
    "last_iteration": max(value["step"] for value in metrics.values()),
    "recent_scalars_finite": finite,
    "metrics": metrics,
    "checkpoints": [str(path) for path in sorted(run.glob("model_*.pt"))],
    "config_verified": {
        "budget": 30000,
        "resume": False,
        "envs_per_rank": 2048,
        "steps_per_update": 24,
        "progress": "max_radial",
        "expert_count": 17,
    },
    "scope": "startup and numerical health only; no behavior capability claim",
}
(OUT / "training_status.json").write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {
            key: result[key]
            for key in ("run", "last_iteration", "recent_scalars_finite", "checkpoints", "config_verified")
        }
    )
)

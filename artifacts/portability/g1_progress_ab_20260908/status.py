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

"""Read-only live TensorBoard/configuration snapshot, using nubot system Python."""

import argparse
import datetime
import json
import math
import subprocess
from pathlib import Path

import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()
result = {"time": datetime.datetime.now().astimezone().isoformat(), "groups": {}}
configs = {}
for arm in ("A",) if args.smoke else ("A", "B"):
    name = f"g1_progress_{'smoke_' if args.smoke else ''}{arm}"
    runs = sorted((ROOT / "logs" / name).glob("*/events.out.tfevents.*"))
    state = {
        "session_alive": (
            subprocess.run(
                ["tmux", "has-session", "-t", "g1-progress-smoke" if args.smoke else f"g1-progress-{arm}"],
                capture_output=True,
            ).returncode
            == 0
        )
    }
    if runs:
        event = runs[-1]
        run = event.parent
        acc = EventAccumulator(str(event), size_guidance={"scalars": 40}).Reload()
        tags = acc.Tags()["scalars"]
        state["run"] = str(run)
        state["checkpoints"] = sorted(p.name for p in run.glob("model_*.pt"))
        state["metrics"] = {}
        for tag in tags:
            if tag.startswith(
                ("Loss/", "Progress/all/", "RewardMix/all/", "Perf/", "Policy/", "Train/mean_episode_length")
            ):
                if tag.endswith("/time"):
                    continue
                values = acc.Scalars(tag)
                if values:
                    state["metrics"][tag] = {"step": values[-1].step, "last": values[-1].value}
        state["all_recent_scalars_finite"] = all(
            math.isfinite(row.value) for tag in tags for row in acc.Scalars(tag)[-20:]
        )
        state["progress_tag_count"] = sum(t.startswith("Progress/") for t in tags)
        state["reward_mix_tag_count"] = sum(t.startswith("RewardMix/") for t in tags)
        env = yaml.load((run / "params/env.yaml").read_text(), Loader=yaml.BaseLoader)
        agent = yaml.load((run / "params/agent.yaml").read_text(), Loader=yaml.BaseLoader)
        configs[arm] = {"env": env, "agent": agent}
        state["contract"] = {
            "speed_scale": env["sparse_command_min_speed_scale"],
            "promotion": env["lightlp_promotion_distance"],
            "monitor": env["progress_monitor_enabled"],
            "num_envs": env["scene"]["num_envs"],
            "resume": agent["resume"],
            "max_iterations": agent["max_iterations"],
            "steps_per_env": agent["num_steps_per_env"],
            "seed": agent["seed"],
            "save_interval": agent["save_interval"],
            "experts": len(agent["amp_motion_files"]),
            "lin_weight": env["reward"]["track_lin_vel_xy_exp"]["weight"],
        }
    result["groups"][arm] = state


def differences(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        rows = []
        for key in sorted(set(left) | set(right)):
            rows.extend(differences(left.get(key), right.get(key), f"{prefix}.{key}".strip(".")))
        return rows
    return [] if left == right else [{"key": prefix, "A": left, "B": right}]


if "A" in configs and "B" in configs:
    result["config_differences"] = differences(configs["A"], configs["B"])
result["gpu_processes"] = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
        "--format=csv,noheader",
    ],
    text=True,
)
dest = OUT / ("smoke_health.json" if args.smoke else "latest_health.json")
dest.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))

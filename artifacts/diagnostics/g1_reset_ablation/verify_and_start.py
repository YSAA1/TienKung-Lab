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

"""Run on nubot after both ready.json files exist; release the verified pair."""

import datetime
import json
from pathlib import Path

import yaml

OUT = Path(
    "/home/nubot/phn_ws/t4_train/TienKung-Lab-g1-unitree-v6-20260907/artifacts/portability/reset_xy_ablation_20260907"
)
groups = ("control", "random_xy")
if not all((OUT / group / "ready.json").exists() for group in groups):
    for group in groups:
        path = OUT / group
        print(group, "ready", (path / "ready.json").exists())
        log = path / "train.log"
        if log.exists():
            print(log.read_text(errors="replace")[-1300:])
    raise SystemExit(2)
ready = [json.loads((OUT / group / "ready.json").read_text()) for group in groups]
hash_keys = [key for key in ready[0] if key.endswith("sha256")]
for key in hash_keys:
    assert ready[0][key] == ready[1][key], f"Initial state mismatch: {key}"
env = [yaml.load((OUT / group / "params/env.yaml").read_text(), Loader=yaml.BaseLoader) for group in groups]
velocities = [item["domain_rand"]["events"]["reset_base"]["params"].pop("velocity_range") for item in env]
assert velocities[0] == {}
assert velocities[1] == {"x": ["-0.5", "0.5"], "y": ["-0.5", "0.5"]}
assert env[0] == env[1], "Other environment configuration differences found"
agent = [yaml.load((OUT / group / "params/agent.yaml").read_text(), Loader=yaml.BaseLoader) for group in groups]
assert agent[0] == agent[1], "Agent configuration mismatch"
result = {
    "time": datetime.datetime.now().astimezone().isoformat(),
    "sole_config_difference": "domain_rand.events.reset_base.params.velocity_range (x/y only)",
    "velocity_ranges": velocities,
    "identical_initial_hash_fields": hash_keys,
    "ready": dict(zip(groups, ready)),
    "training_success_verified": False,
}
(OUT / "pair_verification.json").write_text(json.dumps(result, indent=2))
if not (OUT / "start.allowed").exists():
    (OUT / "start.allowed").write_text(result["time"])
print(json.dumps(result))

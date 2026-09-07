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

"""Release the data treatment only after matching the original cold control."""

import datetime
import json
from pathlib import Path

import yaml

ROOT = Path("/home/nubot/phn_ws/t4_train/TienKung-Lab-g1-unitree-v6-20260907")
OUT = ROOT / "artifacts/portability/rob2rob_amp_ablation_20260907"
CONTROL = ROOT / "artifacts/portability/reset_xy_ablation_20260907/control"
TREATMENT = OUT / "rob2rob"
assert (TREATMENT / "ready.json").exists(), "Treatment still initializing"
ready = [json.loads((p / "ready.json").read_text()) for p in (CONTROL, TREATMENT)]
matched = [key for key in ready[0] if key.endswith("sha256") and key != "initial_expert_samples_sha256"]
for key in matched:
    assert ready[0][key] == ready[1][key], key
assert ready[0]["initial_expert_samples_sha256"] != ready[1]["initial_expert_samples_sha256"]
for r in ready:
    assert max(r["root_velocity_abs_max"]) < 1e-6
    assert r["num_envs"] == 2048 and r["iterations"] == 4000 and not r["resume"]
envs = [yaml.load((p / "params/env.yaml").read_text(), Loader=yaml.BaseLoader) for p in (CONTROL, TREATMENT)]
assert envs[0] == envs[1], "Environment differs"
agents = [yaml.load((p / "params/agent.yaml").read_text(), Loader=yaml.BaseLoader) for p in (CONTROL, TREATMENT)]
data_settings = []
for agent in agents:
    data_settings.append({key: agent.pop(key) for key in ("amp_expert_dir", "amp_motion_files")})
assert agents[0] == agents[1], "Agent settings differ beyond the dataset"
result = {
    "time": datetime.datetime.now().astimezone().isoformat(),
    "matched_initial_hashes": matched,
    "data_settings": data_settings,
    "ready": dict(zip(("control", "rob2rob"), ready)),
    "sole_difference": "AMP expert data",
    "behavior_success_verified": False,
}
(OUT / "pair_verification.json").write_text(json.dumps(result, indent=2))
if not (OUT / "start.allowed").exists():
    (OUT / "start.allowed").write_text(result["time"])
print(json.dumps(result))

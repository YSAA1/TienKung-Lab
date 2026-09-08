"""Reproduce physical exploration and regularizer units from saved evidence."""

import json
from pathlib import Path

from legged_lab.assets.t4.constants import T4_JOINT_NAMES
from legged_lab.assets.unitree_g1.constants import G1_29DOF_JOINT_NAMES

root = Path(__file__).resolve().parent
checkpoints = json.loads((root / "checkpoint_stats.json").read_text())
stds = {row["index"]: row["std"] for row in checkpoints if row["checkpoint"].endswith("model_8000.pt")}
audit = json.loads((root / "g1_m8000_flat.json").read_text())["audit"]
scales = dict(zip(audit["joint_names"], audit["action_scale"][0]))
pairs = (
    ("hip_pitch", "J_hip_l_pitch", "left_hip_pitch_joint"),
    ("knee", "J_knee_l_pitch", "left_knee_joint"),
    ("ankle_pitch", "J_ankle_l_pitch", "left_ankle_pitch_joint"),
    ("waist_yaw", "J_waist_yaw", "waist_yaw_joint"),
)
rows = []
for role, t4, g1 in pairs:
    scale = scales[g1]
    t4_std = stds[2][T4_JOINT_NAMES.index(t4)]
    g1_std = stds[4][G1_29DOF_JOINT_NAMES.index(g1)]
    rows.append({
        "role": role,
        "t4_scale_rad": 0.25,
        "g1_scale_rad": scale,
        "t4_target_noise_std_rad": 0.25 * t4_std,
        "g1_target_noise_std_rad": scale * g1_std,
        "g1_vs_t4_penalty_for_same_target_delta": (0.25 / scale) ** 2,
    })

# Same physical target change with different action coordinates: the unchanged
# raw-action reward is not invariant to the new action transformation.
delta_rad = 0.1
waist_pitch_scale = scales["waist_pitch_joint"]
result = {
    "source_checkpoint_iteration": 8000,
    "rows": rows,
    "waist_pitch_scale_rad": waist_pitch_scale,
    "waist_pitch_penalty_ratio": (0.25 / waist_pitch_scale) ** 2,
    "same_delta_rad": delta_rad,
    "old_action_rate_cost_one_joint": 0.1 * (delta_rad / 0.25) ** 2,
    "current_waist_pitch_action_rate_cost_one_joint": 0.1 * (delta_rad / waist_pitch_scale) ** 2,
    "interpretation": "Exact unit and checkpoint comparison; not an isolated training ablation proving sole causality.",
}
assert result["waist_pitch_penalty_ratio"] > 18.0
assert rows[0]["g1_vs_t4_penalty_for_same_target_delta"] > 2.0
assert rows[2]["g1_vs_t4_penalty_for_same_target_delta"] < 0.34
(root / "action_units.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))

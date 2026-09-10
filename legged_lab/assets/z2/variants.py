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

"""Documented Z2 DoF and 29DoF PD variants. Not mixed into the teacher plant."""

from __future__ import annotations

# Values are the _make_z2_cfg kwargs from upstream z2.py. The 29DoF AMP
# registration chain (Z2Switch29DofAmpBaseEnvCfg) uses WALK_POSE_DAMPED.
Z2_29DOF_PD_RECIPES: dict[str, dict[str, float]] = {
    "Z2_29DOF_CFG": {
        "ankle_effort_limit_sim": 75.0,
        "ankle_velocity_limit_sim": 10.0,
        "hip_pitch_knee_damping": 4.0,
        "hip_yaw_stiffness": 60.0,
        "hip_yaw_damping": 2.5,
        "ankle_stiffness": 60.0,
        "ankle_damping": 2.0,
    },
    "Z2_29DOF_RELAXED_ANKLE_CFG": {
        "ankle_effort_limit_sim": 150.0,
        "ankle_velocity_limit_sim": 12.0,
        "hip_pitch_knee_damping": 4.0,
        "hip_yaw_stiffness": 60.0,
        "hip_yaw_damping": 2.5,
        "ankle_stiffness": 60.0,
        "ankle_damping": 2.0,
    },
    "Z2_29DOF_STRICT_ACTION_RATE_PD_CFG": {
        "ankle_effort_limit_sim": 150.0,
        "ankle_velocity_limit_sim": 12.0,
        "hip_pitch_knee_damping": 4.0,
        "hip_yaw_stiffness": 75.0,
        "hip_yaw_damping": 3.5,
        "ankle_stiffness": 50.0,
        "ankle_damping": 3.0,
    },
    "Z2_29DOF_WALK_POSE_DAMPED_PD_CFG": {
        "ankle_effort_limit_sim": 150.0,
        "ankle_velocity_limit_sim": 12.0,
        "hip_pitch_knee_damping": 6.0,
        "hip_yaw_stiffness": 75.0,
        "hip_yaw_damping": 3.5,
        "ankle_stiffness": 50.0,
        "ankle_damping": 4.0,
    },
}

Z2_TEACHER_PD_RECIPE = "Z2_29DOF_WALK_POSE_DAMPED_PD_CFG"

Z2_DOF_VARIANTS: dict[str, dict[str, str]] = {
    "29": {
        "urdf": "assembly_urdf_29/assembly.urdf",
        "usd": "usd/assembly_29dof/assembly.usd",
        "mjcf": "assembly_mjcf_29/assembly.xml",
        "hands": "sphere hands, fixed joints",
        "neck": "neck_fixed",
        "waist": "yaw/pitch/roll actuated",
        "wrists": "roll/pitch/yaw actuated",
    },
    "23": {
        "urdf": "assembly_urdf/assembly.urdf",
        "usd": "usd/assembly/assembly.usd",
        "hands": "revo2 dexterous, not this teacher",
        "waist": "actuated",
        "wrists": "not in policy",
    },
    "20": {
        "urdf": "assembly_urdf/assembly_20dofs_refine.urdf",
        "usd": "usd/assembly_20dofs_refine/assembly_20dofs_refine.usd",
        "waist": "not actuated",
        "wrists": "not in policy",
    },
}

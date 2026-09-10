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
# Modifications are licensed under the BSD-3-Clause license,
# and is distributed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

"""Isaac-free Z2 MuJoCo sim2sim contract.

PD/effort numbers match ``Z2_29DOF_WALK_POSE_DAMPED_PD_CFG`` in
``legged_lab/assets/z2/z2.py`` — the plant used by ``z2_loco_teacher`` — not
the generic ``Z2_29DOF_CFG`` (ankle 75 Nm). MJCF actuators in
``mjcf/assembly.xml`` are torque motors named after the joints, so this module
rewrites them into position servos like the G1 contract does.
"""

from __future__ import annotations

import numpy as np

from legged_lab.assets.z2.constants import Z2_29DOF_JOINT_NAMES

ISAAC_FRICTION = (1.0, 0.005, 0.0001)


def isaac_pd_gains(joint: str) -> tuple[float, float, float]:
    """Return ``(kp, kd, effort)`` for one Z2 joint name (walk-pose damped plant)."""
    if joint == "waist_yaw_joint":
        return 200.0, 5.0, 90.0
    if joint in ("waist_pitch_joint", "waist_roll_joint"):
        return 80.0, 4.0, 75.0
    if joint.endswith("_hip_pitch_joint") or joint.endswith("_knee_joint"):
        return 120.0, 6.0, 130.0
    if joint.endswith("_hip_roll_joint"):
        return 120.0, 4.0, 132.0
    if joint.endswith("_hip_yaw_joint"):
        return 75.0, 3.5, 70.0
    if joint.endswith("_ankle_pitch_joint") or joint.endswith("_ankle_roll_joint"):
        # Walk-pose damped plant raises the generic ankle effort 75 -> 150 Nm.
        return 50.0, 4.0, 150.0
    if joint.endswith("_shoulder_pitch_joint"):
        return 50.0, 1.5, 70.0
    if (
        joint.endswith(("_shoulder_roll_joint", "_shoulder_yaw_joint", "_elbow_joint", "_wrist_roll_joint"))
    ):
        return 40.0, 1.0, 36.0
    if joint.endswith(("_wrist_pitch_joint", "_wrist_yaw_joint")):
        return 40.0, 1.0, 12.0
    raise ValueError(joint)


def apply_isaac_pd(model, joint_names: tuple[str, ...] = Z2_29DOF_JOINT_NAMES) -> None:
    """Rewrite MJCF torque motors into the Isaac implicit PD/effort contract."""
    import mujoco

    for name in joint_names:
        kp, kd, effort = isaac_pd_gains(name)
        actuator = model.actuator(name)
        dof = int(model.jnt_dofadr[model.joint(name).id])
        a = int(actuator.id)
        model.actuator_gaintype[a] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_gainprm[a, :] = 0.0
        model.actuator_gainprm[a, 0] = kp
        model.actuator_biastype[a] = mujoco.mjtBias.mjBIAS_AFFINE
        model.actuator_biasprm[a, :] = 0.0
        model.actuator_biasprm[a, 1] = -kp
        model.actuator_ctrllimited[a] = 0
        model.actuator_forcelimited[a] = 1
        model.actuator_forcerange[a] = (-effort, effort)
        model.dof_damping[dof] = kd
        # The upstream MJCF default joint carries frictionloss=0.1; Isaac has none.
        model.dof_frictionloss[dof] = 0.0


def apply_isaac_contact_friction(model) -> None:
    """Give every colliding geom Isaac-style friction and condim>=3."""
    friction = np.asarray(ISAAC_FRICTION, dtype=np.float64)
    for i in range(model.ngeom):
        if int(model.geom_contype[i]) == 0 and int(model.geom_conaffinity[i]) == 0:
            continue
        model.geom_condim[i] = max(int(model.geom_condim[i]), 3)
        model.geom_friction[i] = friction

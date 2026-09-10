"""Isaac-free G1 MuJoCo sim2sim contract.

PD numbers match ``official_velocity_g1.py`` ImplicitActuatorCfg. Actuator names
in ``xmls/g1_actuated.xml`` are the joint names.
"""

from __future__ import annotations

import numpy as np

from legged_lab.assets.unitree_g1.constants import G1_29DOF_JOINT_NAMES

ISAAC_FRICTION = (1.0, 0.005, 0.0001)


def isaac_pd_gains(joint: str) -> tuple[float, float, float]:
    """Return ``(kp, kd, effort)`` for one G1 joint name."""
    if joint.endswith(("_wrist_pitch_joint", "_wrist_yaw_joint")):
        return 40.0, 1.0, 5.0
    if joint == "waist_yaw_joint":
        return 200.0, 5.0, 88.0
    if "hip_pitch" in joint or "hip_yaw" in joint:
        return 100.0, 2.0, 88.0
    if "hip_roll" in joint:
        return 100.0, 2.0, 139.0
    if "knee" in joint:
        return 150.0, 4.0, 139.0
    if "ankle" in joint:
        return 40.0, 2.0, 25.0
    if joint in {"waist_roll_joint", "waist_pitch_joint"}:
        return 40.0, 5.0, 25.0
    if "shoulder" in joint or "elbow" in joint or joint.endswith("wrist_roll_joint"):
        return 40.0, 1.0, 25.0
    raise ValueError(joint)


def apply_isaac_pd(model, joint_names: tuple[str, ...] = G1_29DOF_JOINT_NAMES) -> None:
    """Rewrite MJCF position servos to the official G1 velocity PD/effort limits."""
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
        model.dof_frictionloss[dof] = 0.0


def apply_isaac_contact_friction(model) -> None:
    """Give every colliding geom Isaac mu=1 sliding friction and condim=3."""
    friction = np.asarray(ISAAC_FRICTION, dtype=np.float64)
    for i in range(model.ngeom):
        if int(model.geom_contype[i]) == 0 and int(model.geom_conaffinity[i]) == 0:
            continue
        model.geom_condim[i] = max(int(model.geom_condim[i]), 3)
        model.geom_friction[i] = friction

"""Isaac-free T4 MuJoCo sim2sim contract.

Numbers match ``t4.py`` ImplicitActuatorCfg. ``t4_std.xml`` stays a torque-motor
deployment asset; this module rewrites motors into position servos and applies
Isaac contact friction at load time.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from legged_lab.assets.t4.constants import T4_JOINT_NAMES

# Sliding / torsional / rolling. Sliding 1.0 matches Isaac ground mu=1;
# the last two copy official tienkung2_lite/mjcf/tienkung.xml.
ISAAC_FRICTION = (1.0, 0.005, 0.0001)

# URDF sphere-hand pose on AL7/AR7 (t4_std.urdf PHP plant).
SPHERE_HAND_POS = (0.0, 0.0, -0.031)
SPHERE_HAND_EULER = (0.0, 3.14159265359, 0.0)

# URDF left/right_foot_link collision boxes. MuJoCo box size is half-extent.
ISAAC_FOOT_BOXES: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...] = (
    ((0.089, 0.0, -0.027), (0.119 * 0.5, 0.077 * 0.5, 0.015 * 0.5)),
    ((-0.0355, 0.0, -0.027), (0.05 * 0.5, 0.069 * 0.5, 0.015 * 0.5)),
    ((0.0095, 0.0, -0.027), (0.04 * 0.5, 0.066 * 0.5, 0.015 * 0.5)),
)

_FOOT_CAPSULE_RE = re.compile(
    r'\s*<geom name="(?:left|right)_foot\d+_collision"[^/]*/>\n',
)
_HAND_BOX_RE = re.compile(
    r'<geom size="0\.025 0\.015 0\.05" pos="0 0 -0\.02" type="box"[^/]*/>',
)
_GROUND_RE = re.compile(r'<geom name="ground"[^/]*/>')


def isaac_pd_gains(joint: str) -> tuple[float, float, float]:
    """Return ``(kp, kd, effort)`` for one T4 joint name."""
    if joint.startswith("J_arm"):
        idx = int(joint[-2:])
        kp = 20.0 if idx <= 5 else 10.0
        effort = 36.0 if idx <= 4 else 12.0
        return kp, 1.0, effort
    if joint == "J_waist_yaw":
        return 50.0, 2.0, 120.0
    if "hip" in joint:
        return (100.0, 4.0, 130.0) if joint.endswith("pitch") else (50.0, 2.0, 120.0)
    if "knee" in joint:
        return 100.0, 4.0, 130.0
    if "ankle" in joint:
        return (80.0, 4.0, 72.0) if joint.endswith("pitch") else (20.0, 1.0, 72.0)
    raise ValueError(joint)


def apply_isaac_pd(model, joint_names: tuple[str, ...] = T4_JOINT_NAMES) -> None:
    """Rewrite torque motors into Isaac ImplicitActuator position servos.

    ``dof_damping`` is set to ``kd`` (not added to the XML default 0.05).
    ``dof_frictionloss`` is cleared. Armature is left as authored in the MJCF.
    """
    import mujoco

    for name in joint_names:
        kp, kd, effort = isaac_pd_gains(name)
        actuator = model.actuator(f"M{name[1:]}")
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


def rewrite_vault_mjcf(xml: str, *, meshdir: Path, box_pos, box_size) -> str:
    """Patch ``t4_std.xml`` into a G1 vault scene with Isaac contact surfaces."""
    xml = xml.replace('meshdir="../meshes/"', f'meshdir="{meshdir}"')
    if 'name="half_sphere"' not in xml:
        xml = xml.replace(
            '<mesh name="torso_link"',
            '<mesh name="half_sphere" file="half_sphere.obj"/>\n    <mesh name="torso_link"',
            1,
        )
    friction = f'friction="{" ".join(str(v) for v in ISAAC_FRICTION)}"'
    xml = _GROUND_RE.sub(
        f'<geom name="ground" type="plane" pos="0 0 0" size="0 0 1" '
        f'material="matplane" condim="3" {friction}/>',
        xml,
        count=1,
    )
    xml = _FOOT_CAPSULE_RE.sub("", xml)
    foot_geoms = []
    for side in ("left", "right"):
        for i, (pos, half) in enumerate(ISAAC_FOOT_BOXES, start=1):
            foot_geoms.append(
                f'<geom name="{side}_foot{i}_collision" type="box" '
                f'pos="{pos[0]} {pos[1]} {pos[2]}" '
                f'size="{half[0]} {half[1]} {half[2]}" '
                f'condim="3" {friction}/>'
            )
        marker = f'<site name="{side}_foot"'
        xml = xml.replace(marker, "".join(foot_geoms) + marker, 1)
        foot_geoms.clear()

    def _hand_geom(name: str) -> str:
        return (
            f'<geom name="{name}" type="mesh" mesh="half_sphere" '
            f'pos="{SPHERE_HAND_POS[0]} {SPHERE_HAND_POS[1]} {SPHERE_HAND_POS[2]}" '
            f'euler="{SPHERE_HAND_EULER[0]} {SPHERE_HAND_EULER[1]} {SPHERE_HAND_EULER[2]}" '
            f'condim="3" {friction} density="0"/>'
        )

    hands = iter((_hand_geom("left_sphere_hand"), _hand_geom("right_sphere_hand")))
    xml = _HAND_BOX_RE.sub(lambda _m: next(hands), xml, count=2)

    hx, hy, hz = (s / 2.0 for s in box_size)
    px, py, pz = box_pos
    box = (
        f'<geom name="vault_box" type="box" pos="{px} {py} {pz}" '
        f'size="{hx} {hy} {hz}" rgba="0.75 0.45 0.15 1" condim="3" {friction}/>'
    )
    xml = xml.replace("</worldbody>", box + "</worldbody>")
    return xml

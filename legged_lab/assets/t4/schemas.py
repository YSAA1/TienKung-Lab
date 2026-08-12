"""Versioned T4 observation contracts for the privileged teacher and depth student.

This module lives in the pure-Python asset package, next to the joint-order truth
it depends on, so offline tools and contract tests can import it without an
IsaacLab runtime. Runtime code, the AMP expert generator, the evaluator and the
export path must all read their shapes and orders from here instead of restating
literals.
"""

from __future__ import annotations

from legged_lab.assets.t4.constants import T4_JOINT_NAMES

NUM_T4_JOINTS = len(T4_JOINT_NAMES)

POLICY_ROLES = ("teacher", "student")

# ---------------------------------------------------------------------------
# AMP state: q27 + dq27 + hands_root6 + feet_root6
# ---------------------------------------------------------------------------

AMP_SCHEMA_VERSION = "t4_amp.v1"

AMP_FIELDS: tuple[tuple[str, int], ...] = (
    ("joint_pos", NUM_T4_JOINTS),
    ("joint_vel", NUM_T4_JOINTS),
    ("hand_pos_root", 6),
    ("foot_pos_root", 6),
)

AMP_FRAME_DIM = sum(width for _, width in AMP_FIELDS)
AMP_TRANSITION_DIM = 2 * AMP_FRAME_DIM

# End-effector features are taken from the migrated T4 asset, using the MJCF
# site offsets so expert and runtime observations share one kinematic chain.
AMP_HAND_BODIES = ("AL7", "AR7")
AMP_FOOT_BODIES = ("left_foot_link", "right_foot_link")
AMP_HAND_SITE_OFFSET = (0.005, 0.0, -0.114)
AMP_FOOT_SITE_OFFSET = (0.047, 0.0, -0.0345)

# `t4_run` stays held out until it passes an independent joint-limit and motion
# quality review; see the M0 audit artifact.
AMP_HELD_OUT_MOTIONS = ("t4_run",)

# Explicit behaviour-class weights. File counts must never decide the mixture.
AMP_MOTION_CLASS_WEIGHTS: dict[str, float] = {
    "stand": 1.0,
    "walk_forward": 1.0,
    "walk_backward": 0.6,
    "walk_lateral": 0.8,
    "turn": 0.8,
    "jog": 0.6,
}

AMP_MOTION_CLASSES: dict[str, str] = {
    "t4_stand": "stand",
    "t4_walk_forward": "walk_forward",
    "t4_walk_left": "walk_lateral",
    "t4_side_left": "walk_lateral",
    "t4_side_right": "walk_lateral",
    "t4_left_rotate": "turn",
    "t4_right_rotate": "turn",
    "t4_jog_forward": "jog",
    "t4_jog_backward": "jog",
    "t4_jog_left": "jog",
    "t4_jog_right": "jog",
    "B4_-_Stand_to_Walk_backwards_stageii": "walk_backward",
    "B9_-__Walk_turn_left_90_stageii": "turn",
    "B10_-__Walk_turn_left_45_stageii": "turn",
    "B13_-__Walk_turn_right_90_stageii": "turn",
    "B14_-__Walk_turn_right_45_t2_stageii": "turn",
    "B15_-__Walk_turn_around_stageii": "turn",
}


def amp_field_slice(name: str) -> tuple[int, int]:
    """Return the ``[start, end)`` column range of one AMP field."""
    start = 0
    for field_name, width in AMP_FIELDS:
        if field_name == name:
            return start, start + width
        start += width
    raise KeyError(f"unknown AMP field {name!r}; known={[field for field, _ in AMP_FIELDS]}")


def amp_motion_class(motion_stem: str) -> str:
    """Return the declared behaviour class of a motion stem."""
    try:
        return AMP_MOTION_CLASSES[motion_stem]
    except KeyError as error:
        raise KeyError(
            f"motion {motion_stem!r} has no declared AMP behaviour class; add it to AMP_MOTION_CLASSES "
            "or hold it out explicitly"
        ) from error


def amp_motion_weight(motion_stem: str) -> float:
    """Return the per-file sampling weight implied by the explicit class weights."""
    motion_class = amp_motion_class(motion_stem)
    class_members = [stem for stem, name in AMP_MOTION_CLASSES.items() if name == motion_class]
    return AMP_MOTION_CLASS_WEIGHTS[motion_class] / len(class_members)


AMP_FORMAL_EXPERT_DIR = "legged_lab/envs/t4/datasets/motion_amp_expert"


def amp_expert_files(expert_dir: str = AMP_FORMAL_EXPERT_DIR) -> list[str]:
    """Return the AMP expert files implied by the declared motion classes.

    The list comes from the frozen class table rather than from a directory
    listing, so a missing or half-generated expert set fails loudly at load time
    instead of silently shrinking the behaviour mixture.
    """
    stems = sorted(stem for stem in AMP_MOTION_CLASSES if stem not in AMP_HELD_OUT_MOTIONS)
    return [f"{expert_dir}/{stem}.txt" for stem in stems]


# ---------------------------------------------------------------------------
# Teacher local terrain privilege
# ---------------------------------------------------------------------------

TEACHER_TERRAIN_SCHEMA_VERSION = "t4_teacher_terrain.v1"

TEACHER_SCAN_BODY = "Trunk"
TEACHER_SCAN_RESOLUTION = 0.1
# Forward-asymmetric window. The IsaacLab grid pattern is centred on the sensor
# frame, so the forward bias comes from the sensor offset rather than the size.
TEACHER_SCAN_SIZE = (1.4, 1.2)
TEACHER_SCAN_OFFSET = (0.9, 0.0)
TEACHER_SCAN_FORWARD_RANGE = (0.2, 1.6)
TEACHER_SCAN_LATERAL_RANGE = (-0.6, 0.6)
TEACHER_SCAN_ORDERING = "xy"
TEACHER_SCAN_SHAPE = (
    round(TEACHER_SCAN_SIZE[0] / TEACHER_SCAN_RESOLUTION) + 1,
    round(TEACHER_SCAN_SIZE[1] / TEACHER_SCAN_RESOLUTION) + 1,
)
TEACHER_SCAN_DIM = TEACHER_SCAN_SHAPE[0] * TEACHER_SCAN_SHAPE[1]
TEACHER_SCAN_HEIGHT_OFFSET = 0.5
TEACHER_SCAN_CLIP = (-1.0, 1.0)
# A ray that finds no ground means a drop deeper than the caster reaches, so the
# invalid fill is the positive clip bound rather than zero.
TEACHER_SCAN_INVALID_VALUE = 1.0
TEACHER_SCAN_HISTORY_LENGTH = 1

# Fields that would make the teacher unlearnable for a depth student.
TEACHER_FORBIDDEN_PRIVILEGE_FIELDS = (
    "global_map",
    "route_progress",
    "future_gate",
    "gate_pose",
    "success_label",
    "terrain_id",
    "terrain_type",
    "terrain_difficulty",
    "contact_truth",
    "teacher_latent",
)

# ---------------------------------------------------------------------------
# Depth student preprocessing
# ---------------------------------------------------------------------------

DEPTH_SCHEMA_VERSION = "t4_depth.v1"

DEPTH_SENSOR_SIZE = (270, 480)
DEPTH_POLICY_SIZE = (48, 64)
DEPTH_CLIP_RANGE = (0.2, 3.0)
DEPTH_INVALID_VALUE = 1.0
DEPTH_NORMALIZED_RANGE = (0.0, 1.0)
DEPTH_HISTORY_LENGTH = 3
# Depth refreshes every third policy step; intermediate steps reuse the newest
# frame, which is also what the deployment stack does.
DEPTH_UPDATE_DECIMATION = 3
DEPTH_RESIZE_MODE = "area"

# Camera pose candidate that keeps the visible ground band roughly aligned with
# the teacher scan window. Camera height above ground is about 1.32 m and the
# vertical half-FOV is about 28 deg, so a 35 deg downward pitch first sees the
# ground near 0.67 m ahead. The 0.2-0.67 m near band is intentionally left to
# depth history plus proprioception; it can never be directly visible from a
# torso-mounted camera.
DEPTH_CAMERA_BODY = "Trunk"
DEPTH_CAMERA_SITE = "forward_camera"
DEPTH_CAMERA_SITE_POS = (0.085, 0.0, 0.42)
DEPTH_CAMERA_PITCH_DEG = 35.0
DEPTH_NEAREST_VISIBLE_GROUND = 0.67

# ---------------------------------------------------------------------------
# Proprioceptive observation shared by teacher and student
# ---------------------------------------------------------------------------

PROPRIO_SCHEMA_VERSION = "t4_proprio.v1"

PROPRIO_FIELDS: tuple[tuple[str, int], ...] = (
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("velocity_command", 3),
    ("joint_pos", NUM_T4_JOINTS),
    ("joint_vel", NUM_T4_JOINTS),
    ("previous_action", NUM_T4_JOINTS),
    ("gait_phase_sin", 2),
    ("gait_phase_cos", 2),
    ("gait_air_ratio", 2),
)
PROPRIO_FRAME_DIM = sum(width for _, width in PROPRIO_FIELDS)
PROPRIO_HISTORY_LENGTH = 10

CRITIC_EXTRA_FIELDS: tuple[tuple[str, int], ...] = (
    ("base_lin_vel", 3),
    ("feet_contact", 2),
)
CRITIC_FRAME_DIM = PROPRIO_FRAME_DIM + sum(width for _, width in CRITIC_EXTRA_FIELDS)

TEACHER_ACTOR_OBS_DIM = PROPRIO_FRAME_DIM * PROPRIO_HISTORY_LENGTH + TEACHER_SCAN_DIM
STUDENT_ACTOR_OBS_DIM = (
    PROPRIO_FRAME_DIM * PROPRIO_HISTORY_LENGTH + DEPTH_POLICY_SIZE[0] * DEPTH_POLICY_SIZE[1] * DEPTH_HISTORY_LENGTH
)


def proprio_field_slice(name: str) -> tuple[int, int]:
    """Return the ``[start, end)`` column range of one proprioceptive field."""
    start = 0
    for field_name, width in PROPRIO_FIELDS:
        if field_name == name:
            return start, start + width
        start += width
    raise KeyError(f"unknown proprio field {name!r}; known={[field for field, _ in PROPRIO_FIELDS]}")


def assert_no_privilege_leakage(role: str, field_names: tuple[str, ...] | list[str]) -> None:
    """Fail fast when a deployable role is fed teacher-only or route-truth fields."""
    if role not in POLICY_ROLES:
        raise ValueError(f"unknown policy role {role!r}; known={POLICY_ROLES}")
    if role == "teacher":
        leaked = [name for name in field_names if name in TEACHER_FORBIDDEN_PRIVILEGE_FIELDS]
        if leaked:
            raise ValueError(f"teacher privilege schema contains non-transferable fields: {sorted(leaked)}")
        return
    forbidden = set(TEACHER_FORBIDDEN_PRIVILEGE_FIELDS) | {"height_scan", "teacher_scan", "elevation_map"}
    leaked = [name for name in field_names if name in forbidden]
    if leaked:
        raise ValueError(f"student observation schema leaks privileged fields: {sorted(leaked)}")


def observation_manifest() -> dict:
    """Machine-checkable manifest of every frozen T4 observation contract."""
    return {
        "joint_order": list(T4_JOINT_NAMES),
        "num_actions": NUM_T4_JOINTS,
        "amp": {
            "schema_version": AMP_SCHEMA_VERSION,
            "fields": [{"name": name, "width": width} for name, width in AMP_FIELDS],
            "frame_dim": AMP_FRAME_DIM,
            "transition_dim": AMP_TRANSITION_DIM,
            "hand_bodies": list(AMP_HAND_BODIES),
            "foot_bodies": list(AMP_FOOT_BODIES),
            "hand_site_offset": list(AMP_HAND_SITE_OFFSET),
            "foot_site_offset": list(AMP_FOOT_SITE_OFFSET),
            "held_out_motions": list(AMP_HELD_OUT_MOTIONS),
            "class_weights": dict(AMP_MOTION_CLASS_WEIGHTS),
        },
        "proprio": {
            "schema_version": PROPRIO_SCHEMA_VERSION,
            "fields": [{"name": name, "width": width} for name, width in PROPRIO_FIELDS],
            "frame_dim": PROPRIO_FRAME_DIM,
            "history_length": PROPRIO_HISTORY_LENGTH,
        },
        "teacher_terrain": {
            "schema_version": TEACHER_TERRAIN_SCHEMA_VERSION,
            "body": TEACHER_SCAN_BODY,
            "resolution": TEACHER_SCAN_RESOLUTION,
            "size": list(TEACHER_SCAN_SIZE),
            "offset": list(TEACHER_SCAN_OFFSET),
            "forward_range": list(TEACHER_SCAN_FORWARD_RANGE),
            "lateral_range": list(TEACHER_SCAN_LATERAL_RANGE),
            "ordering": TEACHER_SCAN_ORDERING,
            "shape": list(TEACHER_SCAN_SHAPE),
            "dim": TEACHER_SCAN_DIM,
            "height_offset": TEACHER_SCAN_HEIGHT_OFFSET,
            "clip": list(TEACHER_SCAN_CLIP),
            "invalid_value": TEACHER_SCAN_INVALID_VALUE,
            "history_length": TEACHER_SCAN_HISTORY_LENGTH,
            "forbidden_fields": list(TEACHER_FORBIDDEN_PRIVILEGE_FIELDS),
        },
        "depth": {
            "schema_version": DEPTH_SCHEMA_VERSION,
            "sensor_size": list(DEPTH_SENSOR_SIZE),
            "policy_size": list(DEPTH_POLICY_SIZE),
            "clip_range": list(DEPTH_CLIP_RANGE),
            "invalid_value": DEPTH_INVALID_VALUE,
            "normalized_range": list(DEPTH_NORMALIZED_RANGE),
            "history_length": DEPTH_HISTORY_LENGTH,
            "update_decimation": DEPTH_UPDATE_DECIMATION,
            "resize_mode": DEPTH_RESIZE_MODE,
            "camera_body": DEPTH_CAMERA_BODY,
            "camera_site": DEPTH_CAMERA_SITE,
            "camera_site_pos": list(DEPTH_CAMERA_SITE_POS),
            "camera_pitch_deg": DEPTH_CAMERA_PITCH_DEG,
            "nearest_visible_ground": DEPTH_NEAREST_VISIBLE_GROUND,
        },
        "actor_obs_dim": {
            "teacher": TEACHER_ACTOR_OBS_DIM,
            "student": STUDENT_ACTOR_OBS_DIM,
        },
        "critic_frame_dim": CRITIC_FRAME_DIM,
    }

"""Robot-independent locomotion observation fields and sensor contracts."""

from __future__ import annotations

from dataclasses import dataclass


def proprio_fields(num_joints: int) -> tuple[tuple[str, int], ...]:
    return (
        ("base_ang_vel", 3), ("projected_gravity", 3), ("velocity_command", 3),
        ("joint_pos", num_joints), ("joint_vel", num_joints), ("previous_action", num_joints),
        ("gait_phase_sin", 2), ("gait_phase_cos", 2), ("gait_air_ratio", 2),
    )


def field_slice(fields, name: str) -> tuple[int, int]:
    start = 0
    for field, width in fields:
        if field == name:
            return start, start + width
        start += width
    raise KeyError(f"unknown observation field {name!r}")


@dataclass(frozen=True)
class ObservationLayout:
    num_joints: int
    actor_history: int = 10
    critic_history: int = 10
    scan_history: int = 1
    scan_shape: tuple[int, int] = (15, 13)
    foot_scan_shape: tuple[int, int] = (5, 3)
    actor_contact: bool = False
    critic_foot_scan: bool = False
    critic_immunity: bool = False

    def __post_init__(self):
        if min(self.num_joints, self.actor_history, self.critic_history, self.scan_history,
               *self.scan_shape, *self.foot_scan_shape) <= 0:
            raise ValueError("observation dimensions and histories must be positive")

    @property
    def proprio_dim(self) -> int:
        return sum(width for _, width in proprio_fields(self.num_joints))

    @property
    def critic_frame_dim(self) -> int:
        return self.proprio_dim + 5

    @property
    def scan_dim(self) -> int:
        return self.scan_shape[0] * self.scan_shape[1]

    @property
    def foot_scan_dim(self) -> int:
        return self.foot_scan_shape[0] * self.foot_scan_shape[1]

    @property
    def actor_dim(self) -> int:
        return self.proprio_dim * self.actor_history + self.scan_dim * self.scan_history + 2 * self.actor_contact

    @property
    def critic_dim(self) -> int:
        return (self.critic_frame_dim * self.critic_history + self.scan_dim * self.scan_history
                + 2 * self.foot_scan_dim * self.critic_foot_scan + int(self.critic_immunity))

    @classmethod
    def from_cfg(cls, cfg):
        scan = cfg.scene.height_scanner
        foot = cfg.scene.foot_scanner
        return cls(
            num_joints=len(cfg.robot_spec.joint_names),
            actor_history=cfg.robot.actor_obs_history_length,
            critic_history=cfg.robot.critic_obs_history_length,
            scan_history=getattr(cfg, "teacher_scan_history_length", 1),
            scan_shape=tuple(round(s / scan.resolution) + 1 for s in scan.size),
            foot_scan_shape=tuple(round(s / foot.resolution) + 1 for s in foot.size),
            actor_contact=getattr(cfg, "append_actor_feet_contact", False),
            critic_foot_scan=getattr(cfg, "append_critic_foot_scan", False),
            critic_immunity=getattr(cfg, "append_critic_immunity", False),
        )

POLICY_ROLES = ("teacher", "student")

TEACHER_SCAN_RESOLUTION = 0.1

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

TEACHER_SCAN_INVALID_VALUE = 1.0

TEACHER_SCAN_HISTORY_LENGTH = 1

FOOT_SCAN_RESOLUTION = 0.04

FOOT_SCAN_SIZE = (0.16, 0.08)

FOOT_SCAN_SHAPE = (
    round(FOOT_SCAN_SIZE[0] / FOOT_SCAN_RESOLUTION) + 1,
    round(FOOT_SCAN_SIZE[1] / FOOT_SCAN_RESOLUTION) + 1,
)

FOOT_SCAN_DIM = FOOT_SCAN_SHAPE[0] * FOOT_SCAN_SHAPE[1]

FOOT_SCAN_BOTH_DIM = 2 * FOOT_SCAN_DIM

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

DEPTH_SENSOR_SIZE = (270, 480)

DEPTH_POLICY_SIZE = (48, 64)

DEPTH_CLIP_RANGE = (0.2, 3.0)

DEPTH_INVALID_VALUE = 1.0

DEPTH_NORMALIZED_RANGE = (0.0, 1.0)

DEPTH_HISTORY_LENGTH = 3

DEPTH_UPDATE_DECIMATION = 3

DEPTH_RESIZE_MODE = "area"

PROPRIO_HISTORY_LENGTH = 10

CRITIC_EXTRA_FIELDS: tuple[tuple[str, int], ...] = (
    ("base_lin_vel", 3),
    ("feet_contact", 2),
)

TEACHER_PAPER_CONTACT_DIM = 2

TEACHER_SPARSE_SCAN_HISTORY_LENGTH = 5

TEACHER_SPARSE_CONTACT_DIM = 2

TEACHER_SPARSE_IMMUNITY_DIM = 1

STUDENT_PROPRIO_HISTORY_LENGTH = 1

STUDENT_DEPTH_HISTORY_LENGTH = 1

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

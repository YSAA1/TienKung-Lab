"""Explicit biped semantics supplied by each robot, independent of Isaac Sim."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class LocomotionRobotSpec:
    name: str
    joint_names: tuple[str, ...]
    feet: tuple[str, str]
    hands: tuple[str, str]
    torso: str
    diagnostic_bodies: tuple[str, ...]
    # Semantic order: hip roll, hip pitch, hip yaw, knee, ankle pitch, ankle roll.
    left_leg: tuple[str, ...]
    right_leg: tuple[str, ...]
    ankles: tuple[str, ...]
    hand_site_offset: tuple[float, float, float]
    foot_site_offset: tuple[float, float, float]
    mirror_indices: tuple[int, ...]
    mirror_signs: tuple[float, ...]
    nominal_feet_distance: float
    reward_bodies: dict[str, tuple[str, ...]]
    reward_joints: dict[str, tuple[str, ...]]

    @property
    def amp_frame_dim(self) -> int:
        return 2 * len(self.joint_names) + 3 * (len(self.hands) + len(self.feet))

    @property
    def diagnostic_joints(self) -> tuple[str, ...]:
        return self.left_leg[1:4:2] + self.right_leg[1:4:2]

    def validate(self) -> None:
        n = len(self.joint_names)
        if not self.name or n == 0 or len(set(self.joint_names)) != n:
            raise ValueError("robot spec requires a name and unique ordered policy joints")
        if len(set(self.feet)) != 2 or len(set(self.hands)) != 2:
            raise ValueError(f"{self.name}: locomotion requires two ordered feet and two ordered hands")
        if len(self.left_leg) != 6 or len(self.right_leg) != 6:
            raise ValueError(f"{self.name}: leg semantics must be roll, pitch, yaw, knee, ankle pitch, ankle roll")
        for label, names in (("left leg", self.left_leg), ("right leg", self.right_leg), ("ankles", self.ankles)):
            if len(set(names)) != len(names) or not set(names) <= set(self.joint_names):
                raise ValueError(f"{self.name}: invalid {label} joint names {names}")
        if set(self.left_leg) & set(self.right_leg):
            raise ValueError(f"{self.name}: left and right legs overlap")
        if sorted(self.mirror_indices) != list(range(n)) or len(self.mirror_signs) != n:
            raise ValueError(f"{self.name}: mirror must be a permutation of all policy joints")
        for i, j in enumerate(self.mirror_indices):
            if (self.mirror_indices[j] != i or self.mirror_signs[i] not in (-1.0, 1.0)
                    or self.mirror_signs[i] * self.mirror_signs[j] != 1.0):
                raise ValueError(f"{self.name}: mirror is not an involution at joint {self.joint_names[i]}")
        if not math.isfinite(self.nominal_feet_distance) or self.nominal_feet_distance <= 0:
            raise ValueError(f"{self.name}: nominal foot separation must be positive")
        for offset in (self.hand_site_offset, self.foot_site_offset):
            if len(offset) != 3 or not all(math.isfinite(v) for v in offset):
                raise ValueError(f"{self.name}: AMP site offsets must have three finite coordinates")

    def validate_articulation(self, robot) -> None:
        self.validate()
        actual = tuple(robot.joint_names)
        if len(actual) != len(self.joint_names) or set(actual) != set(self.joint_names):
            raise ValueError(
                f"{self.name}: articulation/policy joint mismatch; "
                f"missing={sorted(set(self.joint_names) - set(actual))}, "
                f"unexpected={sorted(set(actual) - set(self.joint_names))}"
            )
        bodies = set(robot.body_names)
        required = set(self.feet + self.hands + self.diagnostic_bodies + (self.torso,))
        if not required <= bodies:
            raise ValueError(f"{self.name}: articulation missing bodies {sorted(required - bodies)}")

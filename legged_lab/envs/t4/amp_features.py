"""Single 66D T4 AMP feature builder shared by runtime and expert generation.

The AMP contract requires expert transitions and policy transitions to come from
one implementation. Both the training env and the offline expert generator build
their frames through :class:`T4AmpFeatureBuilder` so a schema change can never
drift between the two paths.
"""

from __future__ import annotations

import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.utils.math import quat_apply, quat_conjugate

from legged_lab.assets.t4.constants import T4_JOINT_NAMES
from legged_lab.assets.t4.schemas import (
    AMP_FOOT_BODIES,
    AMP_FOOT_SITE_OFFSET,
    AMP_FRAME_DIM,
    AMP_HAND_BODIES,
    AMP_HAND_SITE_OFFSET,
    NUM_T4_JOINTS,
)


def site_pos_in_root_frame(
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    site_offset_b: torch.Tensor,
) -> torch.Tensor:
    """Express body-local site positions in the root frame.

    Args:
        root_pos_w: World root positions, shape ``(N, 3)``.
        root_quat_w: World root orientations as ``wxyz``, shape ``(N, 4)``.
        body_pos_w: World body positions, shape ``(N, B, 3)``.
        body_quat_w: World body orientations as ``wxyz``, shape ``(N, B, 4)``.
        site_offset_b: Site offset in each body frame, shape ``(3,)``.

    Returns:
        Root-frame site positions, shape ``(N, B * 3)``.
    """
    num_envs, num_bodies = body_pos_w.shape[0], body_pos_w.shape[1]
    offset = site_offset_b.expand(num_envs * num_bodies, 3)
    site_w = body_pos_w.reshape(-1, 3) + quat_apply(body_quat_w.reshape(-1, 4), offset)
    site_w = site_w.reshape(num_envs, num_bodies, 3) - root_pos_w.unsqueeze(1)
    root_quat = quat_conjugate(root_quat_w).unsqueeze(1).expand(num_envs, num_bodies, 4)
    site_b = quat_apply(root_quat.reshape(-1, 4), site_w.reshape(-1, 3))
    return site_b.reshape(num_envs, num_bodies * 3)


class T4AmpFeatureBuilder:
    """Builds the frozen ``q27 + dq27 + hands_root6 + feet_root6`` AMP state."""

    def __init__(self, robot: Articulation, device: str | torch.device):
        self.robot = robot
        self.device = device
        self.joint_ids = self._resolve_joints(robot)
        self.hand_body_ids = self._resolve_bodies(robot, AMP_HAND_BODIES)
        self.foot_body_ids = self._resolve_bodies(robot, AMP_FOOT_BODIES)
        self.hand_site_offset = torch.tensor(AMP_HAND_SITE_OFFSET, dtype=torch.float32, device=device)
        self.foot_site_offset = torch.tensor(AMP_FOOT_SITE_OFFSET, dtype=torch.float32, device=device)

    @staticmethod
    def _resolve_joints(robot: Articulation) -> list[int]:
        joint_ids, joint_names = robot.find_joints(name_keys=list(T4_JOINT_NAMES), preserve_order=True)
        if tuple(joint_names) != T4_JOINT_NAMES:
            raise RuntimeError(f"resolved T4 joint order {tuple(joint_names)} does not match T4_JOINT_NAMES")
        if len(joint_ids) != NUM_T4_JOINTS:
            raise RuntimeError(f"expected {NUM_T4_JOINTS} T4 joints, resolved {len(joint_ids)}")
        return joint_ids

    @staticmethod
    def _resolve_bodies(robot: Articulation, names: tuple[str, ...]) -> list[int]:
        body_ids, body_names = robot.find_bodies(name_keys=list(names), preserve_order=True)
        if tuple(body_names) != names:
            raise RuntimeError(f"resolved bodies {tuple(body_names)} do not match requested {names}")
        return body_ids

    def compute(self) -> torch.Tensor:
        """Return the current AMP state, shape ``(num_envs, 66)``."""
        data = self.robot.data
        root_pos_w = data.root_state_w[:, 0:3]
        root_quat_w = data.root_state_w[:, 3:7]
        hand_pos_root = site_pos_in_root_frame(
            root_pos_w,
            root_quat_w,
            data.body_pos_w[:, self.hand_body_ids, :],
            data.body_quat_w[:, self.hand_body_ids, :],
            self.hand_site_offset,
        )
        foot_pos_root = site_pos_in_root_frame(
            root_pos_w,
            root_quat_w,
            data.body_pos_w[:, self.foot_body_ids, :],
            data.body_quat_w[:, self.foot_body_ids, :],
            self.foot_site_offset,
        )
        state = torch.cat(
            [
                data.joint_pos[:, self.joint_ids],
                data.joint_vel[:, self.joint_ids],
                hand_pos_root,
                foot_pos_root,
            ],
            dim=-1,
        )
        if state.shape[-1] != AMP_FRAME_DIM:
            raise RuntimeError(f"AMP state width {state.shape[-1]} does not match schema width {AMP_FRAME_DIM}")
        return state

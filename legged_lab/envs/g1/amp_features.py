"""70D G1 AMP feature builder shared by runtime and expert generation."""

from __future__ import annotations

import torch
from isaaclab.assets.articulation import Articulation

from legged_lab.assets.unitree_g1.constants import G1_29DOF_JOINT_NAMES, NUM_G1_29DOF_JOINTS
from legged_lab.assets.unitree_g1.schemas import (
    AMP_FOOT_BODIES,
    AMP_FOOT_SITE_OFFSET,
    AMP_FRAME_DIM,
    AMP_HAND_BODIES,
    AMP_HAND_SITE_OFFSET,
)
from legged_lab.envs.t4.amp_features import site_pos_in_root_frame


class G1AmpFeatureBuilder:
    """Builds the frozen ``q29 + dq29 + hands_root6 + feet_root6`` AMP state."""

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
        joint_ids, joint_names = robot.find_joints(name_keys=list(G1_29DOF_JOINT_NAMES), preserve_order=True)
        if tuple(joint_names) != G1_29DOF_JOINT_NAMES:
            raise RuntimeError(f"resolved G1 joint order {tuple(joint_names)} does not match G1_29DOF_JOINT_NAMES")
        if len(joint_ids) != NUM_G1_29DOF_JOINTS:
            raise RuntimeError(f"expected {NUM_G1_29DOF_JOINTS} G1 joints, resolved {len(joint_ids)}")
        return joint_ids

    @staticmethod
    def _resolve_bodies(robot: Articulation, names: tuple[str, ...]) -> list[int]:
        body_ids, body_names = robot.find_bodies(name_keys=list(names), preserve_order=True)
        if tuple(body_names) != names:
            raise RuntimeError(f"resolved bodies {tuple(body_names)} do not match requested {names}")
        return body_ids

    def compute(self) -> torch.Tensor:
        """Return the current AMP state, shape ``(num_envs, 70)``."""
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

"""T4 online depth distillation environment.

The physics, rewards and reset semantics are inherited from the Stage E teacher.
The environment exposes two observation streams: a deployable depth/proprio
student input and the frozen teacher's proprio/HeightScan input for supervision.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from isaaclab.utils import configclass

from legged_lab.assets.t4.schemas import (
    DEPTH_CAMERA_SITE_POS,
    DEPTH_CLIP_RANGE,
    DEPTH_HISTORY_LENGTH,
    DEPTH_INVALID_VALUE,
    DEPTH_POLICY_SIZE,
    DEPTH_UPDATE_DECIMATION,
    PROPRIO_FRAME_DIM,
    PROPRIO_HISTORY_LENGTH,
    STUDENT_ACTOR_OBS_DIM,
    TEACHER_ACTOR_OBS_DIM,
    depth_camera_ros_quat_wxyz,
)
from legged_lab.envs.base.base_config import BaseSceneCfg
from legged_lab.envs.t4.t4_env import T4LocoEnv
from legged_lab.envs.base.base_config import FootScannerCfg
from legged_lab.envs.t4.teacher_cfg import T4LocoTeacherEnvCfg, T4SparseTeacherRewardCfg
from legged_lab.terrains import T4_STAGE_E_SPARSE_TERRAINS_CFG
from legged_lab.sensors.camera.camera_cfg import CameraCfg
from legged_lab.sensors.camera.camera_cfgs import D455CameraCfg


@configclass
class T4LocoDepthStudentEnvCfg(T4LocoTeacherEnvCfg):
    """Stage E physics with the schema head-height depth camera."""

    policy_role: str = "student"

    def __post_init__(self):
        # Keep the same local HeightScan for teacher supervision, while the
        # student receives only the camera stream through the env below.
        # Head-height Trunk site + 35 deg down. Do not inherit the stock D455
        # pelvis offset: that rolls the image 90 deg and is off-contract.
        self.scene.depth_camera = D455CameraCfg(
            prim_body_name="Trunk/depth_camera",
            width=480,
            height=270,
            debug_vis=False,
            data_types=["distance_to_image_plane"],
            offset=CameraCfg.OffsetCfg(
                pos=DEPTH_CAMERA_SITE_POS,
                rot=depth_camera_ros_quat_wxyz(),
                convention="ros",
            ),
        )
        # The nubot headless image does not ship the Isaac Nucleus visual
        # materials referenced by the generic SceneCfg. Camera sensors only
        # need the collision terrain, so keep rendering self-contained.
        self.scene.disable_visual_assets = True
        self.noise.add_noise = False


@configclass
class T4LocoDepthStudentFtEnvCfg(T4LocoDepthStudentEnvCfg):
    """Sparse-mix student FT: LightLP rewards, existing depth student obs."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain_generator = T4_STAGE_E_SPARSE_TERRAINS_CFG
        self.scene.max_init_terrain_level = 5
        self.scene.foot_scanner = FootScannerCfg(enable=True)
        self.reward = T4SparseTeacherRewardCfg()
        self.append_actor_feet_contact = False


class T4LocoDepthDistillEnv(T4LocoEnv):
    """T4 teacher rollout with a depth-only deployable observation stream."""

    def __init__(self, cfg: T4LocoDepthStudentEnvCfg, headless):
        # T4LocoEnv's teacher initialization builds the shared physics/reward
        # path and buffers. Switch the role after construction so the existing
        # teacher contract remains untouched.
        cfg.policy_role = "teacher"
        super().__init__(cfg, headless)
        cfg.policy_role = "student"
        self.policy_role = "student"
        self.depth_camera = self.scene["depth_camera"]
        self.depth_history = torch.zeros(
            self.num_envs,
            DEPTH_HISTORY_LENGTH,
            DEPTH_POLICY_SIZE[0],
            DEPTH_POLICY_SIZE[1],
            dtype=torch.float32,
            device=self.device,
        )
        self.depth_update_counter = 0
        self._last_teacher_obs = None

    def _read_depth_frame(self) -> torch.Tensor:
        raw = self.depth_camera.data.output["distance_to_image_plane"].squeeze(-1)
        raw = torch.nan_to_num(raw, nan=DEPTH_INVALID_VALUE, posinf=DEPTH_INVALID_VALUE, neginf=DEPTH_CLIP_RANGE[0])
        raw = torch.clamp(raw, DEPTH_CLIP_RANGE[0], DEPTH_CLIP_RANGE[1])
        normalized = (raw - DEPTH_CLIP_RANGE[0]) / (DEPTH_CLIP_RANGE[1] - DEPTH_CLIP_RANGE[0])
        resized = F.interpolate(
            normalized.unsqueeze(1), size=DEPTH_POLICY_SIZE, mode="area", align_corners=None
        ).squeeze(1)
        return resized

    def _update_depth_history(self, env_ids=None):
        frame = self._read_depth_frame()
        if self.depth_update_counter == 0:
            self.depth_history[:] = frame.unsqueeze(1)
        elif self.depth_update_counter % DEPTH_UPDATE_DECIMATION == 0:
            self.depth_history[:, :-1] = self.depth_history[:, 1:].clone()
            self.depth_history[:, -1] = frame
        self.depth_update_counter += 1

    def compute_observations(self):
        current_actor_obs, current_critic_obs = self.compute_current_observations()
        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)
        proprio_history = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        height_scan = self.compute_teacher_terrain_privilege()
        teacher_obs = torch.cat([proprio_history, height_scan], dim=-1)
        if teacher_obs.shape[-1] != TEACHER_ACTOR_OBS_DIM:
            raise RuntimeError(f"teacher observation width {teacher_obs.shape[-1]} != {TEACHER_ACTOR_OBS_DIM}")

        self._update_depth_history()
        student_obs = torch.cat([proprio_history, self.depth_history.flatten(start_dim=1)], dim=-1)
        if student_obs.shape[-1] != STUDENT_ACTOR_OBS_DIM:
            raise RuntimeError(f"student observation width {student_obs.shape[-1]} != {STUDENT_ACTOR_OBS_DIM}")
        student_obs = torch.clip(student_obs, -self.clip_obs, self.clip_obs)
        self._last_teacher_obs = torch.clip(teacher_obs, -self.clip_obs, self.clip_obs)
        return student_obs, self._last_teacher_obs

    def get_observations(self):
        student_obs, teacher_obs = self.compute_observations()
        self.extras["observations"] = {"teacher": teacher_obs}
        return student_obs, self.extras

    def step(self, actions: torch.Tensor):
        student_obs, rewards, dones, infos = super().step(actions)
        self.extras["observations"] = {"teacher": self._last_teacher_obs}
        return student_obs, rewards, dones, self.extras

    def reset(self, env_ids):
        super().reset(env_ids)
        if hasattr(self, "depth_history") and len(env_ids):
            self.depth_history[env_ids] = 0.0

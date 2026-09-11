# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""G1 online depth distillation environment (B2 of the vision-student plan).

The physics, rewards and reset semantics are inherited from the G1 LightLP
sparse teacher (``G1LocoTeacherEnvCfg``); this module only attaches the
deployable depth stream, mirroring the T4 S12 pattern from
``legged_lab/envs/t4/depth_student_env.py``. G1 joins through
``G1_LOCOMOTION`` (``LocomotionRobotSpec``) and never imports T4 task code.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from legged_lab.envs.g1.depth_student_contract import (
    g1_depth_camera_ros_quat_wxyz,
    G1_DEPTH_CAMERA_SITE_POS,
    G1_STUDENT_RENDER_SIZE,
)
from legged_lab.envs.g1.teacher_cfg import G1LocoTeacherEnvCfg
from legged_lab.locomotion.mdp.camera_extrinsic import LIGHTLP_CAMERA_ORI_JITTER_RAD, LIGHTLP_CAMERA_POS_JITTER_M
from legged_lab.sensors.camera.camera_cfg import CameraCfg
from legged_lab.sensors.camera.camera_cfgs import TiledD455CameraCfg


def _g1_student_depth_camera(
    *,
    update_period: float = 0.02,
    sensor_noise_enable: bool = False,
) -> TiledD455CameraCfg:
    # Tiled RTX D455 at native policy resolution (48x64); LightLP §VI noise is
    # applied in Python, keep Isaac SensorNoiseCfg off so the two do not stack.
    height, width = G1_STUDENT_RENDER_SIZE
    cfg = TiledD455CameraCfg(
        prim_body_name="torso_link",
        width=width,
        height=height,
        debug_vis=False,
        data_types=["distance_to_image_plane"],
        offset=CameraCfg.OffsetCfg(
            pos=G1_DEPTH_CAMERA_SITE_POS,
            rot=g1_depth_camera_ros_quat_wxyz(),
            convention="ros",
        ),
    )
    cfg.sensor_noise.enable = bool(sensor_noise_enable)
    cfg.update_period = update_period
    cfg.enable_depth_camera = True
    return cfg


@configclass
class G1LocoSparseDepthStudentEnvCfg(G1LocoTeacherEnvCfg):
    """G1 sparse teacher MDP with a deployable depth/proprio student stream."""

    policy_role: str = "student"
    # Same per-env camera mounting-tolerance DR as the T4 S12 student line
    # (consumed by LightLPDepthDistillationEnv's extrinsic jitter).
    student_camera_pos_jitter_m: float = LIGHTLP_CAMERA_POS_JITTER_M
    student_camera_ori_jitter_rad: float = LIGHTLP_CAMERA_ORI_JITTER_RAD

    def __post_init__(self):
        super().__post_init__()
        # Keep the local HeightScan for teacher supervision; the student only
        # receives the camera stream through LightLPDepthDistillationEnv.
        self.scene.depth_camera = _g1_student_depth_camera()
        self.scene.disable_visual_assets = True
        self.noise.add_noise = True
        self.noise.noise_scales.height_scan = 0.0
        self.student_depth_noise = True
        self.sparse_curriculum_demote = False
        self.random_level_reset_fraction = 0.10
        self.random_level_reset_min_level = None
        self.random_level_reset_max_level = None


@configclass
class G1LocoSparseDepthStudentReprFirstEnvCfg(G1LocoSparseDepthStudentEnvCfg):
    """Representation stage starts on hard sparse rows; action stage returns to teacher reset mix."""

    random_level_reset_fraction: float = 0.50
    random_level_reset_min_level: int | None = 6
    random_level_reset_max_level: int | None = None

    def __post_init__(self):
        super().__post_init__()
        self.random_level_reset_fraction = 0.50
        self.random_level_reset_min_level = 6
        self.random_level_reset_max_level = None

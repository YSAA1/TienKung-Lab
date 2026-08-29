"""T4 online depth distillation environment.

The physics, rewards and reset semantics are inherited from the Stage E teacher.
The environment exposes two observation streams: a deployable depth/proprio
student input and the frozen teacher's proprio/HeightScan input for supervision.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp

from legged_lab.assets.t4.schemas import (
    DEPTH_CAMERA_SITE_POS,
    DEPTH_CLIP_RANGE,
    DEPTH_HISTORY_LENGTH,
    DEPTH_INVALID_VALUE,
    DEPTH_POLICY_SIZE,
    DEPTH_UPDATE_DECIMATION,
    PROPRIO_FRAME_DIM,
    STAGE_E_STUDENT_ACTOR_OBS_DIM,
    STUDENT_ACTOR_OBS_DIM,
    STUDENT_DEPTH_HISTORY_LENGTH,
    TEACHER_ACTOR_OBS_DIM,
    TEACHER_SPARSE_ACTOR_OBS_DIM,
    depth_camera_ros_quat_wxyz,
)
from legged_lab.envs.base.base_config import EventCfg, FootScannerCfg
from legged_lab.envs.t4.mdp.camera_extrinsic import (
    LIGHTLP_CAMERA_ORI_JITTER_RAD,
    LIGHTLP_CAMERA_POS_JITTER_M,
    apply_camera_local_offset,
    capture_nominal_camera_pose,
    compose_camera_offset,
)
from legged_lab.envs.t4.mdp.depth_noise import (
    LIGHTLP_DEPTH_BLOCK_COUNT,
    LIGHTLP_DEPTH_BLOCK_REFRESH,
    LIGHTLP_DEPTH_DELAY_STEPS,
    LIGHTLP_DEPTH_HOLD_STEPS,
    LIGHTLP_DEPTH_SCALE_JITTER,
    apply_metric_depth_noise,
    apply_depth_boundary_corruption,
    apply_normalized_block_dropout,
    depth_refresh_plan,
    edge_biased_block_rows,
    scaled_block_hw,
    stamp_rectangular_blocks,
)
from legged_lab.envs.t4.t4_env import T4LocoEnv
from legged_lab.envs.t4.teacher_cfg import T4LocoSparseTeacherEnvCfg, T4LocoTeacherEnvCfg, T4SparseTeacherRewardCfg
from legged_lab.sensors.camera.camera_cfg import CameraCfg
from legged_lab.sensors.camera.camera_cfgs import D455CameraCfg, TiledD455CameraCfg
from legged_lab.terrains import T4_STAGE_E_SPARSE_TERRAINS_CFG

# Native policy resolution. Tiled RTX renders 48x64; no 72x128 capture.
SPARSE_STUDENT_RENDER_SIZE = DEPTH_POLICY_SIZE


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


def _t4_student_depth_camera(
    *,
    update_period: float = 0.02,
    sensor_noise_enable: bool = False,
) -> TiledD455CameraCfg:
    # Tiled RTX D455 at native policy resolution. LightLP §VI noise is applied
    # in Python; keep Isaac SensorNoiseCfg off so the two models do not stack.
    height, width = SPARSE_STUDENT_RENDER_SIZE
    cfg = TiledD455CameraCfg(
        prim_body_name="Trunk/depth_camera",
        width=width,
        height=height,
        debug_vis=False,
        data_types=["distance_to_image_plane"],
        offset=CameraCfg.OffsetCfg(
            pos=DEPTH_CAMERA_SITE_POS,
            rot=depth_camera_ros_quat_wxyz(),
            convention="ros",
        ),
    )
    cfg.sensor_noise.enable = bool(sensor_noise_enable)
    cfg.update_period = update_period
    cfg.enable_depth_camera = True
    return cfg


@configclass
class T4LocoSparseDepthStudentEnvCfg(T4LocoSparseTeacherEnvCfg):
    """S12 sparse teacher MDP with a deployable depth/proprio student stream."""

    policy_role: str = "student"
    student_depth_noise: bool = True
    student_depth_hold_steps: int = LIGHTLP_DEPTH_HOLD_STEPS
    student_depth_delay_steps: tuple[int, int] = LIGHTLP_DEPTH_DELAY_STEPS
    # 50 Hz tiled RTX so the 30–60 ms delay model is not quantized to 16.7 Hz.
    student_depth_camera_update_period: float = 0.02
    student_depth_d455_sensor_noise: bool = False
    student_depth_dropout_after_resize: bool = True
    student_camera_pos_jitter_m: float = LIGHTLP_CAMERA_POS_JITTER_M
    student_camera_ori_jitter_rad: float = LIGHTLP_CAMERA_ORI_JITTER_RAD
    student_depth_boundary_corruption: bool = False
    student_depth_boundary_probability: float = 0.08
    student_depth_boundary_threshold_m: float = 0.08
    sparse_curriculum_demote: bool = False
    random_level_reset_fraction: float = 0.50
    random_level_reset_min_level: int = 6

    def __post_init__(self):
        super().__post_init__()
        self.scene.depth_camera = _t4_student_depth_camera(
            update_period=self.student_depth_camera_update_period,
            sensor_noise_enable=self.student_depth_d455_sensor_noise,
        )
        # Skip Nucleus marble/sky assets; keep local robot USD + terrain mesh
        # so RTX depth can self-occlude.
        self.scene.disable_visual_assets = True
        self.noise.add_noise = True
        self.noise.noise_scales.height_scan = 0.0
        self.student_depth_noise = True
        self.sparse_curriculum_demote = False
        self.random_level_reset_fraction = 0.50
        self.random_level_reset_min_level = 6


@configclass
class T4LocoSparseDepthStudentFtEnvCfg(T4LocoSparseDepthStudentEnvCfg):
    """Same deploy-domain env for gated D4c task RL. Not launched in this slice."""

    student_depth_camera_update_period: float = 0.02

    def __post_init__(self):
        super().__post_init__()
        self.student_depth_noise = True


@configclass
class T4TargetedFtEventCfg(EventCfg):
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    joint_effort = EventTerm(
        func=mdp.randomize_joint_effort_limits,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "effort_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class T4LocoSparseDepthStudentTargetedFtEnvCfg(T4LocoSparseDepthStudentFtEnvCfg):
    """Targeted-only sim-to-sim robustness domain; default S12/evaluator stays frozen."""

    student_depth_boundary_corruption: bool = True

    def __post_init__(self):
        super().__post_init__()
        base_events = self.domain_rand.events
        targeted_events = T4TargetedFtEventCfg()
        for name in ("physics_material", "add_base_mass", "reset_base", "reset_robot_joints", "push_robot"):
            setattr(targeted_events, name, getattr(base_events, name))
        self.domain_rand.events = targeted_events
        self.domain_rand.action_delay.enable = True
        self.domain_rand.action_delay.params = {"min_delay": 0, "max_delay": 2}
        self.student_depth_boundary_corruption = True
        generator = copy.deepcopy(T4_STAGE_E_SPARSE_TERRAINS_CFG)
        stones = generator.sub_terrains["stepping_stones"]
        stones.targeted_layout_seed = True
        pillars = generator.sub_terrains["raised_pillars"]
        pillars.targeted_manufacturing_variation = True
        pillars.xy_jitter_m = 0.015
        pillars.height_jitter_m = 0.015
        pillars.top_tilt_rad = 0.035
        pillars.diameter_scale_jitter = 0.04
        pillars.pitch_scale_jitter = 0.03
        self.scene.terrain_generator = generator


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
        pending = self._pending_post_reset_rtx_mask()
        writable = ~pending
        if torch.any(writable):
            if self.depth_update_counter == 0:
                self.depth_history[writable] = frame[writable].unsqueeze(1)
            elif self.depth_update_counter % DEPTH_UPDATE_DECIMATION == 0:
                hist = self.depth_history[writable]
                self.depth_history[writable, :-1] = hist[:, 1:].clone()
                self.depth_history[writable, -1] = frame[writable]
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
        if student_obs.shape[-1] != STAGE_E_STUDENT_ACTOR_OBS_DIM:
            raise RuntimeError(f"student observation width {student_obs.shape[-1]} != {STAGE_E_STUDENT_ACTOR_OBS_DIM}")
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


class T4LocoSparseDepthDistillEnv(T4LocoDepthDistillEnv):
    """S12 sparse teacher rollouts with a GRU-ready depth student observation."""

    def __init__(self, cfg: T4LocoSparseDepthStudentEnvCfg, headless):
        super().__init__(cfg, headless)
        self.depth_history = torch.zeros(
            self.num_envs,
            STUDENT_DEPTH_HISTORY_LENGTH,
            DEPTH_POLICY_SIZE[0],
            DEPTH_POLICY_SIZE[1],
            dtype=torch.float32,
            device=self.device,
        )
        self.student_depth_noise = bool(getattr(cfg, "student_depth_noise", False))
        self.student_depth_hold_steps = int(getattr(cfg, "student_depth_hold_steps", LIGHTLP_DEPTH_HOLD_STEPS))
        self.student_depth_dropout_after_resize = bool(getattr(cfg, "student_depth_dropout_after_resize", True))
        self.student_depth_boundary_corruption = bool(getattr(cfg, "student_depth_boundary_corruption", False))
        self.student_depth_boundary_probability = float(getattr(cfg, "student_depth_boundary_probability", 0.0))
        self.student_depth_boundary_threshold_m = float(
            getattr(cfg, "student_depth_boundary_threshold_m", 0.08)
        )
        delay = getattr(cfg, "student_depth_delay_steps", LIGHTLP_DEPTH_DELAY_STEPS)
        self.student_depth_delay_lo = int(delay[0])
        self.student_depth_delay_hi = int(delay[1])
        max_delay = max(self.student_depth_delay_hi, 1)
        render_h, render_w = SPARSE_STUDENT_RENDER_SIZE
        self._depth_delay_buf = torch.zeros(
            max_delay + 1, self.num_envs, render_h, render_w, dtype=torch.float32, device=self.device
        )
        self._depth_delay_index = 0
        self._depth_delay_steps = torch.randint(
            self.student_depth_delay_lo,
            self.student_depth_delay_hi + 1,
            (self.num_envs,),
            device=self.device,
        )
        self._depth_scale = torch.ones(self.num_envs, 1, 1, device=self.device)
        if self.student_depth_dropout_after_resize:
            mask_h, mask_w = DEPTH_POLICY_SIZE
        else:
            mask_h, mask_w = SPARSE_STUDENT_RENDER_SIZE
        self._depth_block_mask = torch.zeros(self.num_envs, mask_h, mask_w, dtype=torch.bool, device=self.device)
        self._last_policy_depth = torch.zeros(
            self.num_envs, DEPTH_POLICY_SIZE[0], DEPTH_POLICY_SIZE[1], dtype=torch.float32, device=self.device
        )
        self._depth_fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._refresh_depth_hist = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        captured = capture_nominal_camera_pose(self.depth_camera)
        if captured is None:
            raise RuntimeError(
                "tiled depth camera has no per-env pose API after spawn; "
                "refusing to train without Table II extrinsic DR"
            )
        nom_pos, nom_quat, pose_mode = captured
        self._depth_cam_nominal_pos = nom_pos.to(device=self.device, dtype=torch.float32)
        self._depth_cam_nominal_quat = nom_quat.to(device=self.device, dtype=torch.float32)
        self._depth_cam_pose_mode = pose_mode
        if self.student_depth_noise:
            self._resample_depth_corruption()
        self._apply_camera_extrinsic_jitter(torch.arange(self.num_envs, device=self.device))

    def _apply_camera_extrinsic_jitter(self, env_ids):
        if env_ids is None or len(env_ids) == 0:
            return
        if not hasattr(self, "_depth_cam_nominal_pos") or not hasattr(self, "_depth_cam_pose_mode"):
            return
        pos_m = float(getattr(self.cfg, "student_camera_pos_jitter_m", 0.0) or 0.0)
        ori_rad = float(getattr(self.cfg, "student_camera_ori_jitter_rad", 0.0) or 0.0)
        if pos_m <= 0.0 and ori_rad <= 0.0:
            return
        n = int(len(env_ids))
        dpos = (torch.rand(n, 3, device=self.device) * 2.0 - 1.0) * pos_m
        drpy = (torch.rand(n, 3, device=self.device) * 2.0 - 1.0) * ori_rad
        nom_pos = self._depth_cam_nominal_pos.detach().cpu().tolist()
        nom_quat = tuple(self._depth_cam_nominal_quat.detach().cpu().tolist())
        pos = torch.empty(n, 3, device=self.device, dtype=torch.float32)
        quat = torch.empty(n, 4, device=self.device, dtype=torch.float32)
        for i in range(n):
            composed_pos, composed_quat = compose_camera_offset(
                nom_pos, nom_quat, dpos[i].tolist(), drpy[i].tolist()
            )
            pos[i] = torch.tensor(composed_pos, device=self.device, dtype=torch.float32)
            quat[i] = torch.tensor(composed_quat, device=self.device, dtype=torch.float32)
        apply_camera_local_offset(
            self.depth_camera, env_ids, pos, quat, mode=self._depth_cam_pose_mode
        )

    def _resample_depth_corruption(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return
        draws = torch.rand(len(env_ids), device=self.device)
        self._depth_scale[env_ids, 0, 0] = (1.0 - LIGHTLP_DEPTH_SCALE_JITTER) + 2.0 * LIGHTLP_DEPTH_SCALE_JITTER * draws
        self._depth_block_mask[env_ids] = False
        height, width = self._depth_block_mask.shape[-2:]
        block_h, block_w = scaled_block_hw(height, width)
        for _ in range(LIGHTLP_DEPTH_BLOCK_COUNT):
            row_draw = torch.rand(len(env_ids), device=self.device)
            row = edge_biased_block_rows(row_draw, height, block_h)
            col = torch.randint(0, max(1, width - block_w + 1), (len(env_ids),), device=self.device)
            stamp_rectangular_blocks(self._depth_block_mask, env_ids, row, col, block_h, block_w)

    def _ingest_metric_depth(self) -> torch.Tensor:
        raw = self.depth_camera.data.output["distance_to_image_plane"].squeeze(-1)
        raw = torch.nan_to_num(raw, nan=DEPTH_INVALID_VALUE, posinf=DEPTH_INVALID_VALUE, neginf=DEPTH_CLIP_RANGE[0])
        if not self.student_depth_noise:
            return raw
        if self.depth_update_counter % LIGHTLP_DEPTH_BLOCK_REFRESH == 0:
            self._resample_depth_corruption()
        gaussian = torch.randn_like(raw)
        metric_dropout = None if self.student_depth_dropout_after_resize else self._depth_block_mask
        raw = apply_metric_depth_noise(
            raw,
            gaussian=gaussian,
            scale=self._depth_scale,
            dropout_mask=metric_dropout,
            dropout_value=DEPTH_CLIP_RANGE[1],
        )
        if self.student_depth_boundary_corruption:
            raw = apply_depth_boundary_corruption(
                raw,
                dropout_draw=torch.rand_like(raw),
                false_hit_draw=torch.rand_like(raw),
                probability=self.student_depth_boundary_probability,
                edge_threshold_m=self.student_depth_boundary_threshold_m,
                invalid_depth_m=DEPTH_CLIP_RANGE[1],
            )
        buf_len = self._depth_delay_buf.shape[0]
        slot = self._depth_delay_index % buf_len
        self._depth_delay_buf[slot] = raw
        if torch.any(self._depth_fresh):
            self._depth_delay_buf[:, self._depth_fresh] = raw[self._depth_fresh]
            self._depth_fresh[:] = False
        env_index = torch.arange(self.num_envs, device=self.device)
        delayed_index = (self._depth_delay_index - self._depth_delay_steps) % buf_len
        raw = self._depth_delay_buf[delayed_index, env_index]
        self._depth_delay_index += 1
        return raw

    def _metric_to_policy_depth(self, raw: torch.Tensor, dropout_mask=None) -> torch.Tensor:
        raw = torch.clamp(raw, DEPTH_CLIP_RANGE[0], DEPTH_CLIP_RANGE[1])
        normalized = (raw - DEPTH_CLIP_RANGE[0]) / (DEPTH_CLIP_RANGE[1] - DEPTH_CLIP_RANGE[0])
        if tuple(normalized.shape[-2:]) != DEPTH_POLICY_SIZE:
            resized = F.interpolate(
                normalized.unsqueeze(1), size=DEPTH_POLICY_SIZE, mode="area", align_corners=None
            ).squeeze(1)
        else:
            resized = normalized
        if self.student_depth_noise and self.student_depth_dropout_after_resize:
            mask = self._depth_block_mask if dropout_mask is None else dropout_mask
            resized = apply_normalized_block_dropout(resized, mask, fill_value=1.0)
        return resized

    def _read_depth_frame(self) -> torch.Tensor:
        return self._metric_to_policy_depth(self._ingest_metric_depth())

    def _update_depth_history(self, env_ids=None):
        hold = self.student_depth_hold_steps if self.student_depth_noise else DEPTH_UPDATE_DECIMATION
        refresh = self._refresh_depth_hist if hasattr(self, "_refresh_depth_hist") else None
        pending = self._pending_post_reset_rtx_mask()
        if self.student_depth_noise and refresh is not None:
            reset_any = bool(torch.any(refresh).item())
        else:
            reset_any = False
        ingest, postprocess, write_all = depth_refresh_plan(
            self.depth_update_counter, hold, reset_any, self.student_depth_noise
        )
        if ingest:
            raw = self._ingest_metric_depth()
            if postprocess:
                if write_all or refresh is None:
                    frame = self._metric_to_policy_depth(raw)
                    self._last_policy_depth = frame
                else:
                    frame = self._last_policy_depth
                    reset_mask = self._depth_block_mask[refresh]
                    frame[refresh] = self._metric_to_policy_depth(raw[refresh], dropout_mask=reset_mask)
                    self._last_policy_depth = frame
            else:
                frame = self._last_policy_depth
        else:
            frame = self._last_policy_depth
        if torch.any(pending):
            frame = frame.clone()
            frame[pending] = 0.0
            self._last_policy_depth = self._last_policy_depth.clone()
            self._last_policy_depth[pending] = 0.0
        writable = ~pending
        if write_all:
            if self.depth_update_counter == 0:
                if torch.any(writable):
                    self.depth_history[writable] = frame[writable].unsqueeze(1)
            elif torch.any(writable):
                hist = self.depth_history[writable]
                self.depth_history[writable, :-1] = hist[:, 1:].clone()
                self.depth_history[writable, -1] = frame[writable]
        elif self.student_depth_noise and refresh is not None and torch.any(refresh & writable):
            write_reset = refresh & writable
            self.depth_history[write_reset] = frame[write_reset].unsqueeze(1)
        if hasattr(self, "_refresh_depth_hist"):
            self._refresh_depth_hist[:] = False
        self.depth_update_counter += 1

    def compute_observations(self):
        teacher_obs, _critic_obs = T4LocoEnv.compute_observations(self)
        if teacher_obs.shape[-1] != TEACHER_SPARSE_ACTOR_OBS_DIM:
            raise RuntimeError(
                f"sparse teacher observation width {teacher_obs.shape[-1]} != {TEACHER_SPARSE_ACTOR_OBS_DIM}"
            )
        proprio_frame = self.actor_obs_buffer.buffer[:, -1]
        if proprio_frame.shape[-1] != PROPRIO_FRAME_DIM:
            raise RuntimeError(f"student proprio width {proprio_frame.shape[-1]} != {PROPRIO_FRAME_DIM}")
        self._update_depth_history()
        depth_frame = self.depth_history[:, -1].flatten(start_dim=1)
        student_obs = torch.cat([proprio_frame, depth_frame], dim=-1)
        if student_obs.shape[-1] != STUDENT_ACTOR_OBS_DIM:
            raise RuntimeError(f"student observation width {student_obs.shape[-1]} != {STUDENT_ACTOR_OBS_DIM}")
        student_obs = torch.clip(student_obs, -self.clip_obs, self.clip_obs)
        self._last_teacher_obs = teacher_obs
        return student_obs, self._last_teacher_obs

    def reset(self, env_ids):
        super().reset(env_ids)
        if hasattr(self, "_depth_delay_buf") and len(env_ids):
            self._depth_delay_buf[:, env_ids] = 0.0
            self._depth_fresh[env_ids] = True
            self._refresh_depth_hist[env_ids] = True
            self._depth_delay_steps[env_ids] = torch.randint(
                self.student_depth_delay_lo,
                self.student_depth_delay_hi + 1,
                (len(env_ids),),
                device=self.device,
            )
            if self.student_depth_noise:
                self._resample_depth_corruption(env_ids)
            self._apply_camera_extrinsic_jitter(env_ids)

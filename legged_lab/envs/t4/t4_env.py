# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

"""T4 27-DoF locomotion environment for the privileged teacher `pi_teacher`.

The observation layout, the AMP state and the teacher terrain privilege are all
read from :mod:`legged_lab.assets.t4.schemas`, and every policy-facing joint vector
is expressed in ``T4_JOINT_NAMES`` order regardless of the simulator's internal
joint ordering, so training, export and MuJoCo deployment share one contract.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils  # type: ignore
import numpy as np
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import EventManager, RewardManager
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sim import PhysxCfg, SimulationContext
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
from isaaclab.utils.math import euler_xyz_from_quat, quat_rotate_inverse, yaw_quat

from legged_lab.assets.t4.constants import T4_JOINT_NAMES
from legged_lab.assets.t4.schemas import (
    FOOT_SCAN_BOTH_DIM,
    FOOT_SCAN_DIM,
    NUM_T4_JOINTS,
    PROPRIO_FRAME_DIM,
    TEACHER_PAPER_CONTACT_DIM,
    TEACHER_SCAN_CLIP,
    TEACHER_SCAN_DIM,
    TEACHER_SCAN_INVALID_VALUE,
    TEACHER_SPARSE_CONTACT_DIM,
    TEACHER_SPARSE_SCAN_HISTORY_LENGTH,
    assert_no_privilege_leakage,
    proprio_field_slice,
)
from legged_lab.envs.t4.amp_features import T4AmpFeatureBuilder
from legged_lab.envs.t4.curriculum import (
    LIGHTLP_TRACKING_WELL_THRESHOLD,
    STANDING_COMMAND_THRESHOLD,
    gait_tracking_scale,
    lightlp_terrain_level_moves,
    terrain_level_moves,
)
from legged_lab.envs.t4.mdp.sparse_signals import (
    LIGHTLP_IMMUNITY_FRAC,
    LIGHTLP_IMMUNITY_PERIOD,
    LOG_COUNT_SUFFIX,
    SPARSE_FOOTHOLD_GENTLE_YAW_RANGE,
    SPARSE_FOOTHOLD_STRAIGHT_YAW_PROB,
    SPARSE_FOOTHOLD_VX_RANGE,
    impact_immunity_from_draws,
    lightlp_sparse_promotion_guard,
    lightlp_timeout_and_reset,
    mask_recent_push_accel,
    monitor_outcome_flags,
    random_level_reset_high,
    random_level_reset_low,
    random_level_reset_mask,
    sample_sparse_foothold_velocity,
    mask_sparse_curriculum_demote,
    sparse_curriculum_moves,
    sparse_pit_fall_mask,
    terrain_aware_commands_enabled,
    tilt_from_upright_rad,
)
from legged_lab.envs.t4.terrain_columns import (
    HURDLE_TERRAIN_NAMES,
    SPARSE_FOOTHOLD_NAMES,
    assign_curriculum_columns,
    columns_named,
    name_to_columns,
    unique_names,
)
from legged_lab.envs.t4.teacher_cfg import T4LocoTeacherEnvCfg
from legged_lab.terrains.stepping_stone_layout import (
    T4_FOOT_SCAN_RESOLUTION,
    T4_FOOT_SCAN_SIZE,
    T4_FOOTHOLD_PITCH_RANGE,
    T4_PILLAR_DIAMETER_RANGE,
    T4_PILLAR_PITCH_RANGE,
    T4_STONE_BORDER_WIDTH,
    T4_SPARSE_RIM_WIDTH,
    T4_STONE_PLATFORM_WIDTH,
    T4_STONE_TILE_SIZE,
    T4_STONE_WIDTH_RANGE,
    foot_scan_local_offsets,
)
from legged_lab.utils.env_utils.scene import SceneCfg, rtx_render_due, sensor_is_warp_raycast
from rsl_rl.env import VecEnv


class T4LocoEnv(VecEnv):
    """PPO + AMP locomotion env for the T4 27-DoF humanoid."""

    _TERRAIN_METRIC_FIELDS = (
        "promotion_rate",
        "timeout_success_rate",
        "success_rate",
        "reach_1m_rate",
        "reach_2m_rate",
        "reach_4m_rate",
        "fall_rate",
        "timeout_rate",
        "pit_fall_rate",
        "progress_m",
    )
    _TERRAIN_METRIC_BANDS = ("easy", "mid", "hard")
    _TERRAIN_METRIC_EPISODE_ALPHA = 0.01

    def __init__(self, cfg: T4LocoTeacherEnvCfg, headless):
        self.cfg = cfg
        self.headless = headless
        self.device = self.cfg.device
        self.physics_dt = self.cfg.sim.dt
        self.step_dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_envs = self.cfg.scene.num_envs
        self.policy_role = self.cfg.policy_role
        self.seed(cfg.scene.seed)

        if self.policy_role != "teacher":
            raise NotImplementedError(
                f"policy_role={self.policy_role!r} is frozen in schemas but not implemented yet; "
                "the depth student arrives with Stage S"
            )
        assert_no_privilege_leakage(self.policy_role, ["teacher_scan"])

        sim_cfg = sim_utils.SimulationCfg(
            device=cfg.device,
            dt=cfg.sim.dt,
            render_interval=cfg.sim.decimation,
            physx=PhysxCfg(gpu_max_rigid_patch_count=cfg.sim.physx.gpu_max_rigid_patch_count),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )
        self.sim = SimulationContext(sim_cfg)

        scene_cfg = SceneCfg(config=cfg.scene, physics_dt=self.physics_dt, step_dt=self.step_dt)
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()

        self.robot: Articulation = self.scene["robot"]
        self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]
        if not self.cfg.scene.height_scanner.enable_height_scan:
            raise ValueError("the T4 teacher requires the local terrain privilege; enable the height scanner")
        self.height_scanner: RayCaster = self.scene.sensors["height_scanner"]

        command_cfg = UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=self.cfg.commands.resampling_time_range,
            rel_standing_envs=self.cfg.commands.rel_standing_envs,
            rel_heading_envs=self.cfg.commands.rel_heading_envs,
            heading_command=self.cfg.commands.heading_command,
            heading_control_stiffness=self.cfg.commands.heading_control_stiffness,
            debug_vis=self.cfg.commands.debug_vis,
            ranges=self.cfg.commands.ranges,
        )
        self.command_generator = UniformVelocityCommand(cfg=command_cfg, env=self)
        self.reward_manager = RewardManager(self.cfg.reward, self)
        self.hurdle_terrain_type_id = None
        self.hurdle_terrain_type_ids: list[int] = []
        self.sparse_foothold_type_ids: list[int] = []
        self.stone_column_ids: list[int] = []
        self.pillar_column_ids: list[int] = []
        self.terrain_column_names: list[str] = []
        self.terrain_name_to_columns: dict[str, list[int]] = {}
        generator = getattr(self.cfg.scene, "terrain_generator", None)
        sub = getattr(generator, "sub_terrains", None) if generator is not None else None
        self.terrain_type_names = list(sub.keys()) if sub else []
        num_cols = int(getattr(generator, "num_cols", 0) or 0) if generator is not None else 0
        if sub and num_cols > 0:
            proportions = {name: float(getattr(cfg, "proportion", 0.0)) for name, cfg in sub.items()}
            if bool(getattr(generator, "curriculum", False)):
                self.terrain_column_names = assign_curriculum_columns(proportions, num_cols)
            elif len(sub) == 1:
                # play/eval often disable curriculum and keep one sub-terrain; every column is that type.
                only = next(iter(sub))
                self.terrain_column_names = [only] * num_cols
            if self.terrain_column_names:
                self.terrain_type_names = unique_names(self.terrain_column_names, preferred_order=list(sub.keys()))
                self.terrain_name_to_columns = name_to_columns(self.terrain_column_names, self.terrain_type_names)
                self.sparse_foothold_type_ids = columns_named(self.terrain_column_names, *SPARSE_FOOTHOLD_NAMES)
                self.stone_column_ids = columns_named(self.terrain_column_names, "stepping_stones")
                self.pillar_column_ids = columns_named(self.terrain_column_names, "raised_pillars")
                self.hurdle_terrain_type_ids = columns_named(self.terrain_column_names, *HURDLE_TERRAIN_NAMES)
                self.hurdle_terrain_type_id = (
                    self.hurdle_terrain_type_ids[0] if self.hurdle_terrain_type_ids else None
                )

        self.init_buffers()

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.event_manager = EventManager(self.cfg.domain_rand.events, self)
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        self.reset_env_ids = env_ids
        self.reset(env_ids)
        self.extras.pop("log", None)

    """
    Setup.
    """

    def init_buffers(self):
        self.extras = {}

        self.max_episode_length_s = self.cfg.scene.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.step_dt)
        self.num_actions = NUM_T4_JOINTS
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations

        if self.robot.num_joints != NUM_T4_JOINTS:
            raise RuntimeError(f"expected {NUM_T4_JOINTS} T4 joints, articulation reports {self.robot.num_joints}")

        self.action_scale = self.cfg.robot.action_scale
        self.action_buffer = DelayBuffer(
            self.cfg.domain_rand.action_delay.params["max_delay"], self.num_envs, device=self.device
        )
        self.action_buffer.compute(
            torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        )
        if self.cfg.domain_rand.action_delay.enable:
            time_lags = torch.randint(
                low=self.cfg.domain_rand.action_delay.params["min_delay"],
                high=self.cfg.domain_rand.action_delay.params["max_delay"] + 1,
                size=(self.num_envs,),
                dtype=torch.int,
                device=self.device,
            )
            self.action_buffer.set_time_lag(time_lags, torch.arange(self.num_envs, device=self.device))

        self.robot_cfg = SceneEntityCfg(name="robot")
        self.robot_cfg.resolve(self.scene)
        self.termination_contact_cfg = SceneEntityCfg(
            name="contact_sensor", body_names=self.cfg.robot.terminate_contacts_body_names
        )
        self.termination_contact_cfg.resolve(self.scene)
        self.feet_cfg = SceneEntityCfg(name="contact_sensor", body_names=self.cfg.robot.feet_body_names)
        self.feet_cfg.resolve(self.scene)
        self.diagnostic_contact_body_names = ("Trunk", "Shank_Left", "Shank_Right")
        self.diagnostic_contact_cfg = SceneEntityCfg(
            name="contact_sensor",
            body_names=list(self.diagnostic_contact_body_names),
            preserve_order=True,
        )
        self.diagnostic_contact_cfg.resolve(self.scene)

        # Policy-facing joint vectors always use the frozen T4 order; the simulator
        # order is an implementation detail of the USD conversion.
        self.t4_joint_ids, resolved_joint_names = self.robot.find_joints(
            name_keys=list(T4_JOINT_NAMES), preserve_order=True
        )
        if tuple(resolved_joint_names) != T4_JOINT_NAMES:
            raise RuntimeError(f"resolved joint order {tuple(resolved_joint_names)} does not match T4_JOINT_NAMES")

        self.feet_body_ids, _ = self.robot.find_bodies(
            name_keys=["left_foot_link", "right_foot_link"], preserve_order=True
        )
        self.left_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                "J_hip_l_roll",
                "J_hip_l_pitch",
                "J_hip_l_yaw",
                "J_knee_l_pitch",
                "J_ankle_l_pitch",
                "J_ankle_l_roll",
            ],
            preserve_order=True,
        )
        self.right_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                "J_hip_r_roll",
                "J_hip_r_pitch",
                "J_hip_r_yaw",
                "J_knee_r_pitch",
                "J_ankle_r_pitch",
                "J_ankle_r_roll",
            ],
            preserve_order=True,
        )
        self.ankle_joint_ids, _ = self.robot.find_joints(
            name_keys=["J_ankle_l_pitch", "J_ankle_r_pitch", "J_ankle_l_roll", "J_ankle_r_roll"], preserve_order=True
        )

        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise

        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.sim_step_counter = 0
        self.last_rtx_sim_step = -1
        self.schedule_rtx_render = True
        self._rtx_reset_sim_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.pit_fall_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        metric_shape = (len(self.terrain_type_names), 1 + len(self._TERRAIN_METRIC_BANDS))
        self.terrain_metric_ema = torch.zeros(
            *metric_shape,
            len(self._TERRAIN_METRIC_FIELDS),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.terrain_metric_initialized = torch.zeros(
            *metric_shape, dtype=torch.bool, device=self.device, requires_grad=False
        )
        self.terrain_metric_episodes = torch.zeros(
            *metric_shape, dtype=torch.long, device=self.device, requires_grad=False
        )

        self.gait_phase = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.gait_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.gait_cycle = torch.full(
            (self.num_envs,), self.cfg.gait.gait_cycle, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.phase_ratio = torch.tensor(
            [self.cfg.gait.gait_air_ratio_l, self.cfg.gait.gait_air_ratio_r], dtype=torch.float, device=self.device
        ).repeat(self.num_envs, 1)
        self.phase_offset = torch.tensor(
            [self.cfg.gait.gait_phase_offset_l, self.cfg.gait.gait_phase_offset_r],
            dtype=torch.float,
            device=self.device,
        ).repeat(self.num_envs, 1)
        self.gait_reward_scale = torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        # Peak radial displacement from the terrain origin within the current episode.
        # The terrain curriculum judges traversal on this instead of the final position,
        # so walking across the stairs and returning still counts as a completed crossing.
        self.episode_max_radial_dist = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.episode_path_length = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_tracking_sum = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.episode_tracking_steps = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        # Path length needs the previous control-step pose, while evaluators need
        # a terminal snapshot that survives the automatic reset in ``step()``.
        self.prev_step_root_pos_w = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.terminal_root_pos_w = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.terminal_root_quat_w = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.terminal_episode_max_radial_dist = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.terminal_feet_pos_w = torch.zeros(
            self.num_envs,
            len(self.feet_body_ids),
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.terminal_feet_contact = torch.zeros(
            self.num_envs,
            len(self.feet_body_ids),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.last_root_accel_mps2 = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_tilt_rad = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.terminal_root_lin_vel_w = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.terminal_root_accel_mps2 = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.terminal_tilt_rad = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.terminal_diagnostic_contact_force_n = torch.zeros(
            self.num_envs,
            len(self.diagnostic_contact_cfg.body_ids),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.action = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.action_t4 = torch.zeros_like(self.action)
        self.avg_feet_force_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.avg_feet_speed_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.foot_accel_ema = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.prev_foot_lin_vel_w = torch.zeros(
            self.num_envs, len(self.feet_body_ids), 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.use_algebraic_sparse_scan = bool(getattr(self.cfg, "use_algebraic_sparse_scan", False))
        self.soft_sparse_terrain = bool(getattr(self.cfg, "soft_sparse_terrain", False))
        self.use_lightlp_terminations = bool(getattr(self.cfg, "use_lightlp_terminations", False))
        self.terminate_on_pit_fall = bool(getattr(self.cfg, "terminate_on_pit_fall", True))
        self.append_critic_immunity = bool(getattr(self.cfg, "append_critic_immunity", False))
        self.teacher_scan_history_length = int(getattr(self.cfg, "teacher_scan_history_length", 1))
        self.append_critic_foot_scan = bool(getattr(self.cfg, "append_critic_foot_scan", False))
        self.impact_immunity = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._lightlp_step = 0
        self.prev_root_lin_vel_w = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self._push_step_marker = torch.full((self.num_envs,), -(10**9), dtype=torch.long, device=self.device)
        if self.use_lightlp_terminations:
            self._resample_impact_immunity()
        self._foot_scan_local = torch.tensor(
            foot_scan_local_offsets(T4_FOOT_SCAN_SIZE, T4_FOOT_SCAN_RESOLUTION),
            dtype=torch.float,
            device=self.device,
        )
        self.sparse_tile_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._sparse_command = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self._sparse_command_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_reason_masks: dict[str, torch.Tensor] = {}

        self.amp_builder = T4AmpFeatureBuilder(self.robot, self.device)
        self.init_obs_buffer()

    def init_obs_buffer(self):
        if self.add_noise:
            actor_obs, _ = self.compute_current_observations()
            noise_vec = torch.zeros_like(actor_obs[0])
            noise_scales = self.cfg.noise.noise_scales
            for field, scale in (
                ("base_ang_vel", noise_scales.ang_vel * self.obs_scales.ang_vel),
                ("projected_gravity", noise_scales.projected_gravity * self.obs_scales.projected_gravity),
                ("joint_pos", noise_scales.joint_pos * self.obs_scales.joint_pos),
                ("joint_vel", noise_scales.joint_vel * self.obs_scales.joint_vel),
            ):
                start, end = proprio_field_slice(field)
                noise_vec[start:end] = scale
            self.noise_scale_vec = noise_vec

            height_scan_noise_vec = torch.zeros(TEACHER_SCAN_DIM, dtype=torch.float, device=self.device)
            height_scan_noise_vec[:] = noise_scales.height_scan * self.obs_scales.height_scan
            self.height_scan_noise_vec = height_scan_noise_vec

        self.actor_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.actor_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        self.critic_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.critic_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        self.scan_obs_buffer = CircularBuffer(
            max_len=max(1, self.teacher_scan_history_length), batch_size=self.num_envs, device=self.device
        )

    """
    Observations.
    """

    def compute_current_observations(self):
        robot = self.robot
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        ang_vel = robot.data.root_ang_vel_b
        projected_gravity = robot.data.projected_gravity_b
        command = self.command_generator.command
        joint_pos = (robot.data.joint_pos - robot.data.default_joint_pos)[:, self.t4_joint_ids]
        joint_vel = (robot.data.joint_vel - robot.data.default_joint_vel)[:, self.t4_joint_ids]
        previous_action = self.action_buffer._circular_buffer.buffer[:, -1, :]
        root_lin_vel = robot.data.root_lin_vel_b
        feet_contact = torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 0.5

        current_actor_obs = torch.cat(
            [
                ang_vel * self.obs_scales.ang_vel,
                projected_gravity * self.obs_scales.projected_gravity,
                command * self.obs_scales.commands,
                joint_pos * self.obs_scales.joint_pos,
                joint_vel * self.obs_scales.joint_vel,
                previous_action * self.obs_scales.actions,
                torch.sin(2 * torch.pi * self.gait_phase),
                torch.cos(2 * torch.pi * self.gait_phase),
                self.phase_ratio,
            ],
            dim=-1,
        )
        if current_actor_obs.shape[-1] != PROPRIO_FRAME_DIM:
            raise RuntimeError(
                f"proprio width {current_actor_obs.shape[-1]} does not match schema width {PROPRIO_FRAME_DIM}"
            )
        current_critic_obs = torch.cat(
            [current_actor_obs, root_lin_vel * self.obs_scales.lin_vel, feet_contact], dim=-1
        )
        return current_actor_obs, current_critic_obs

    def _columns_mask(self, terrain_types: torch.Tensor, column_ids: list[int]) -> torch.Tensor:
        mask = torch.zeros(terrain_types.shape[0], dtype=torch.bool, device=terrain_types.device)
        for column_id in column_ids:
            mask |= terrain_types == column_id
        return mask

    def refresh_sparse_tile_mask(self) -> torch.Tensor:
        """Update and return the per-env mask of stepping-stone / raised-pillar tiles."""
        terrain_types = getattr(getattr(self.scene, "terrain", None), "terrain_types", None)
        if self.sparse_foothold_type_ids and terrain_types is not None:
            self.sparse_tile_mask = self._columns_mask(terrain_types, self.sparse_foothold_type_ids)
        else:
            self.sparse_tile_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return self.sparse_tile_mask

    def compute_teacher_terrain_privilege(self):
        """Forward-asymmetric local height scan, clipped and invalid-filled."""
        height_scan = (
            self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
            - self.height_scanner.data.ray_hits_w[..., 2]
            - self.cfg.normalization.height_scan_offset
        )
        height_scan = torch.nan_to_num(
            height_scan,
            nan=TEACHER_SCAN_INVALID_VALUE,
            posinf=TEACHER_SCAN_INVALID_VALUE,
            neginf=TEACHER_SCAN_INVALID_VALUE,
        )
        height_scan = torch.clip(height_scan, TEACHER_SCAN_CLIP[0], TEACHER_SCAN_CLIP[1])
        if self.use_algebraic_sparse_scan and self.soft_sparse_terrain:
            height_scan = self._apply_algebraic_sparse_scan(height_scan)
        if height_scan.shape[-1] != TEACHER_SCAN_DIM:
            raise RuntimeError(
                f"teacher scan width {height_scan.shape[-1]} does not match schema width {TEACHER_SCAN_DIM}"
            )
        return height_scan * self.obs_scales.height_scan

    def _sparse_support_mask_xy(self, world_xy: torch.Tensor) -> torch.Tensor:
        """True-hole support mask for world XY points on sparse tiles ``[N, P]``."""
        sparse = self.sparse_tile_mask
        n_pts = world_xy.shape[1]
        on_support = torch.zeros(self.num_envs, n_pts, dtype=torch.bool, device=self.device)
        if not torch.any(sparse):
            return on_support
        difficulty = self.terrain_difficulty()
        terrain_types = self.scene.terrain.terrain_types
        origins = self.scene.env_origins[:, :2]
        c = 0.5 * T4_STONE_TILE_SIZE
        half_p = 0.5 * T4_STONE_PLATFORM_WIDTH
        tile_x = (world_xy[..., 0] - origins[:, 0:1]) + c
        tile_y = (world_xy[..., 1] - origins[:, 1:2]) + c
        rx = tile_x - c
        ry = tile_y - c
        on_platform = (rx.abs() <= half_p) & (ry.abs() <= half_p)
        is_stone = self._columns_mask(terrain_types, self.stone_column_ids)
        pitch = torch.where(
            is_stone,
            (1.0 - difficulty) * T4_FOOTHOLD_PITCH_RANGE[0] + difficulty * T4_FOOTHOLD_PITCH_RANGE[1],
            (1.0 - difficulty) * T4_PILLAR_PITCH_RANGE[0] + difficulty * T4_PILLAR_PITCH_RANGE[1],
        )
        half_support = 0.5 * torch.where(
            is_stone,
            (1.0 - difficulty) * T4_STONE_WIDTH_RANGE[0] + difficulty * T4_STONE_WIDTH_RANGE[1],
            (1.0 - difficulty) * T4_PILLAR_DIAMETER_RANGE[0] + difficulty * T4_PILLAR_DIAMETER_RANGE[1],
        )
        pitch = pitch.clamp_min(1.0e-3).unsqueeze(1)
        half_support = half_support.unsqueeze(1)
        ix = torch.round(rx / pitch)
        iy = torch.round(ry / pitch)
        dx = rx - ix * pitch
        dy = ry - iy * pitch
        max_ring = torch.floor((c - T4_STONE_BORDER_WIDTH) / pitch)
        within_lattice = (ix.abs() <= max_ring) & (iy.abs() <= max_ring)
        on_rect = (dx.abs() <= half_support) & (dy.abs() <= half_support)
        on_disk = (dx * dx + dy * dy) <= (half_support * half_support)
        on_foothold = torch.where(is_stone.unsqueeze(1), on_rect, on_disk) & within_lattice
        rim = T4_SPARSE_RIM_WIDTH
        tile = T4_STONE_TILE_SIZE
        in_overflow = (tile_x >= -rim) & (tile_x <= tile + rim) & (tile_y >= -rim) & (tile_y <= tile + rim)
        on_rim = in_overflow & ((tile_x <= rim) | (tile_x >= tile - rim) | (tile_y <= rim) | (tile_y >= tile - rim))
        on_support = on_platform | on_foothold | on_rim
        return on_support & sparse.unsqueeze(1)

    def _apply_algebraic_sparse_scan(self, height_scan: torch.Tensor) -> torch.Tensor:
        """True-hole map on soft sparse tiles: pad/footholds keep rays; gaps → invalid."""
        sparse = self.sparse_tile_mask
        if not torch.any(sparse):
            return height_scan
        # Prefer hit XY when finite; otherwise cast from sensor pose + ray pattern is unavailable,
        # so fall back to treating non-support using ray hit positions only.
        hit_xy = self.height_scanner.data.ray_hits_w[..., :2]
        finite = torch.isfinite(hit_xy).all(dim=-1)
        # When the ray misses, force a hole on sparse tiles.
        on_support = self._sparse_support_mask_xy(torch.nan_to_num(hit_xy, nan=0.0))
        hole = sparse.unsqueeze(1) & (~on_support | ~finite)
        return torch.where(hole, torch.full_like(height_scan, TEACHER_SCAN_INVALID_VALUE), height_scan)

    def compute_foot_scan_privilege(self) -> torch.Tensor:
        """Concatenated left/right downward foot scans for the critic only."""
        scanners = []
        for name in ("left_foot_scanner", "right_foot_scanner"):
            if name in self.scene.sensors:
                scanners.append(self.scene.sensors[name])
        if len(scanners) != 2:
            return torch.zeros(self.num_envs, FOOT_SCAN_BOTH_DIM, device=self.device)
        chunks = []
        for scanner in scanners:
            foot_z = scanner.data.pos_w[:, 2].unsqueeze(1)
            hits = scanner.data.ray_hits_w[..., 2]
            depth = foot_z - hits - self.cfg.normalization.height_scan_offset
            depth = torch.nan_to_num(
                depth,
                nan=TEACHER_SCAN_INVALID_VALUE,
                posinf=TEACHER_SCAN_INVALID_VALUE,
                neginf=TEACHER_SCAN_INVALID_VALUE,
            )
            depth = torch.clip(depth, TEACHER_SCAN_CLIP[0], TEACHER_SCAN_CLIP[1])
            if self.use_algebraic_sparse_scan and self.soft_sparse_terrain:
                hit_xy = scanner.data.ray_hits_w[..., :2]
                finite = torch.isfinite(hit_xy).all(dim=-1)
                on_support = self._sparse_support_mask_xy(torch.nan_to_num(hit_xy, nan=0.0))
                hole = self.sparse_tile_mask.unsqueeze(1) & (~on_support | ~finite)
                depth = torch.where(hole, torch.full_like(depth, TEACHER_SCAN_INVALID_VALUE), depth)
            if depth.shape[-1] != FOOT_SCAN_DIM:
                raise RuntimeError(f"foot scan width {depth.shape[-1]} != {FOOT_SCAN_DIM}")
            chunks.append(depth * self.obs_scales.height_scan)
        return torch.cat(chunks, dim=-1)

    def algebraic_foot_illegal_fractions(
        self,
        foot_pos_w: torch.Tensor | None = None,
        root_quat_w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-foot unsupported fractions from the analytic sparse lattice."""
        sparse = self.sparse_tile_mask
        if not torch.any(sparse):
            return torch.zeros(self.num_envs, len(self.feet_body_ids), device=self.device)
        if foot_pos_w is None:
            foot_pos_w = self.robot.data.body_pos_w[:, self.feet_body_ids, :]
        if root_quat_w is None:
            root_quat_w = self.robot.data.root_quat_w
        foot_pos = foot_pos_w[..., :2]
        yaw_q = yaw_quat(root_quat_w)
        w, x, y, z = yaw_q[:, 0], yaw_q[:, 1], yaw_q[:, 2], yaw_q[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        cos_y = torch.cos(yaw)
        sin_y = torch.sin(yaw)
        local = self._foot_scan_local
        penalties = []
        for foot_i in range(foot_pos.shape[1]):
            fx = foot_pos[:, foot_i, 0]
            fy = foot_pos[:, foot_i, 1]
            wx = fx.unsqueeze(1) + cos_y.unsqueeze(1) * local[:, 0] - sin_y.unsqueeze(1) * local[:, 1]
            wy = fy.unsqueeze(1) + sin_y.unsqueeze(1) * local[:, 0] + cos_y.unsqueeze(1) * local[:, 1]
            world_xy = torch.stack((wx, wy), dim=-1)
            on_support = self._sparse_support_mask_xy(world_xy)
            # Non-sparse envs have all-false support mask; force full support so frac is 0.
            on_support = on_support | (~sparse).unsqueeze(1)
            penalties.append((~on_support).float().mean(dim=-1))
        return torch.stack(penalties, dim=-1) * sparse.float().unsqueeze(-1)

    def algebraic_illegal_footstep(self, contact_threshold: float = 0.5) -> torch.Tensor:
        """Vectorized contacted-foot illegal fraction from the true-hole lattice."""
        feet_force_w = self.contact_sensor.data.net_forces_w
        if feet_force_w.ndim == 4:
            feet_force_w = feet_force_w[:, -1]
        feet_force = torch.norm(feet_force_w[:, self.feet_body_ids], dim=-1)
        in_contact = feet_force > contact_threshold
        fractions = self.algebraic_foot_illegal_fractions()
        return torch.sum(fractions * in_contact.float(), dim=-1)

    def update_foot_accel_penalty(
        self,
        tau_s: float = 0.06,
        threshold_mps2: float = 30.0,
        asset_cfg: SceneEntityCfg | None = None,
    ) -> torch.Tensor:
        """LightLP Eq. (5): leaky integral of summed per-foot |a| excess."""
        foot_vel = self.robot.data.body_lin_vel_w[:, self.feet_body_ids, :]
        accel = (foot_vel - self.prev_foot_lin_vel_w) / max(self.step_dt, 1.0e-6)
        self.prev_foot_lin_vel_w.copy_(foot_vel)
        magnitude = torch.norm(accel, dim=-1)
        excess_sum = torch.clamp(magnitude - threshold_mps2, min=0.0).sum(dim=-1)
        decay = math.exp(-self.step_dt / max(tau_s, 1.0e-6))
        self.foot_accel_ema = decay * self.foot_accel_ema + excess_sum
        return self.foot_accel_ema

    def compute_observations(self):
        current_actor_obs, current_critic_obs = self.compute_current_observations()
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec

        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)

        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)

        height_scan = self.compute_teacher_terrain_privilege()
        self.scan_obs_buffer.append(height_scan)
        scan_hist = self.scan_obs_buffer.buffer.reshape(self.num_envs, -1)
        # CircularBuffer stores oldest→newest; when history length is 1 this is one frame.
        if self.teacher_scan_history_length > 1:
            actor_scan = scan_hist
            critic_scan = scan_hist
        else:
            actor_scan = height_scan
            critic_scan = height_scan

        critic_obs = torch.cat([critic_obs, critic_scan], dim=-1)
        if self.add_noise:
            noisy = height_scan + (2 * torch.rand_like(height_scan) - 1) * self.height_scan_noise_vec
            if self.teacher_scan_history_length > 1:
                # Only noise the newest frame in the stacked scan history.
                actor_scan = scan_hist.clone()
                actor_scan[:, -TEACHER_SCAN_DIM:] = noisy
            else:
                actor_scan = noisy
        actor_obs = torch.cat([actor_obs, actor_scan], dim=-1)
        if getattr(self.cfg, "append_actor_feet_contact", False):
            net_contact_forces = self.contact_sensor.data.net_forces_w_history
            feet_contact = (
                torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 0.5
            )
            contact_dim = TEACHER_SPARSE_CONTACT_DIM if self.teacher_scan_history_length > 1 else TEACHER_PAPER_CONTACT_DIM
            if feet_contact.shape[-1] != contact_dim:
                raise RuntimeError(f"feet contact width {tuple(feet_contact.shape)} != {contact_dim}")
            actor_obs = torch.cat([actor_obs, feet_contact.float()], dim=-1)
        if self.append_critic_foot_scan:
            critic_obs = torch.cat([critic_obs, self.compute_foot_scan_privilege()], dim=-1)
        if self.append_critic_immunity:
            critic_obs = torch.cat([critic_obs, self.impact_immunity.float().unsqueeze(-1)], dim=-1)

        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)
        return actor_obs, critic_obs

    def get_observations(self):
        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}
        return actor_obs, self.extras

    def get_amp_obs_for_expert_trans(self):
        """66D AMP state built by the shared feature builder."""
        return self.amp_builder.compute()

    def _depth_camera_sensor(self):
        sensors = getattr(self.scene, "sensors", None)
        if isinstance(sensors, dict):
            camera = sensors.get("depth_camera")
            if camera is not None:
                return camera
        if sensors is not None:
            try:
                if "depth_camera" in sensors:
                    return sensors["depth_camera"]
            except TypeError:
                pass
        return getattr(self, "depth_camera", None)

    def _has_warp_depth_camera(self) -> bool:
        return sensor_is_warp_raycast(self._depth_camera_sensor())

    def _has_rtx_sensors(self) -> bool:
        camera = self._depth_camera_sensor()
        if camera is not None:
            return not sensor_is_warp_raycast(camera)
        checker = getattr(self.sim, "has_rtx_sensors", None)
        if callable(checker):
            return bool(checker())
        return False

    def _depth_camera_update_period(self) -> float:
        camera = None
        sensors = getattr(self.scene, "sensors", None)
        if isinstance(sensors, dict):
            camera = sensors.get("depth_camera")
        if camera is None:
            camera = getattr(self, "depth_camera", None)
        period = getattr(getattr(camera, "cfg", None), "update_period", None)
        if period is not None and float(period) > 0.0:
            return float(period)
        scene_cam = getattr(self.cfg.scene, "depth_camera", None)
        period = getattr(scene_cam, "update_period", None)
        if period is not None and float(period) > 0.0:
            return float(period)
        return float(self.step_dt)

    def _pending_post_reset_rtx_mask(self) -> torch.Tensor:
        reset_at = getattr(self, "_rtx_reset_sim_step", None)
        last = int(getattr(self, "last_rtx_sim_step", -1))
        if reset_at is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return reset_at >= last

    """
    Stepping.
    """

    def step(self, actions: torch.Tensor):
        delayed_actions = self.action_buffer.compute(actions)
        self.action_t4 = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)
        # Scatter the T4-ordered action into simulator joint order once, so rewards
        # that index `robot.data` and `env.action` stay in the same frame.
        self.action.zero_()
        self.action[:, self.t4_joint_ids] = self.action_t4
        processed_actions = self.action * self.action_scale + self.robot.data.default_joint_pos

        self.avg_feet_force_per_step.zero_()
        self.avg_feet_speed_per_step.zero_()
        schedule_ticks = bool(getattr(self, "schedule_rtx_render", True))
        has_rtx = schedule_ticks and self._has_rtx_sensors()
        has_warp = self._has_warp_depth_camera()
        sensor_period = self._depth_camera_update_period() if (has_rtx or has_warp) else self.step_dt
        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            self.robot.set_joint_position_target(processed_actions)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            sensor_due = rtx_render_due(self.sim_step_counter, self.physics_dt, sensor_period)
            if has_rtx and sensor_due:
                self.sim.render()
                self.last_rtx_sim_step = int(self.sim_step_counter)
            self.scene.update(dt=self.physics_dt)
            if has_warp and schedule_ticks and sensor_due:
                self.last_rtx_sim_step = int(self.sim_step_counter)

            self.avg_feet_force_per_step += torch.norm(
                self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids, :3], dim=-1
            )
            self.avg_feet_speed_per_step += torch.norm(self.robot.data.body_lin_vel_w[:, self.feet_body_ids, :], dim=-1)

        self.avg_feet_force_per_step /= self.cfg.sim.decimation
        self.avg_feet_speed_per_step /= self.cfg.sim.decimation

        if not self.headless:
            self.sim.render()

        self.episode_length_buf += 1
        if self.use_lightlp_terminations:
            self._lightlp_step += 1
            if self._lightlp_step % LIGHTLP_IMMUNITY_PERIOD == 0:
                self._resample_impact_immunity()
        radial_dist = torch.norm(self.robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2], dim=1)
        self.episode_max_radial_dist = torch.maximum(self.episode_max_radial_dist, radial_dist)
        step_delta = torch.norm(self.robot.data.root_pos_w[:, :2] - self.prev_step_root_pos_w[:, :2], dim=1)
        self.episode_path_length = self.episode_path_length + step_delta
        tracking = self._gait_tracking_scale()
        moving_cmd = torch.norm(self.command_generator.command[:, :2], dim=1) > STANDING_COMMAND_THRESHOLD
        self.episode_tracking_sum = self.episode_tracking_sum + tracking
        self.episode_tracking_steps = self.episode_tracking_steps + moving_cmd.float()
        self._update_gait()

        self.command_generator.compute(self.step_dt)
        self._enforce_terrain_aware_commands()
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.reset_buf, self.time_out_buf = self.check_reset()
        reward_buf = self.reward_manager.compute(self.step_dt)
        self.reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.prev_step_root_pos_w.copy_(self.robot.data.root_pos_w)
        if len(self.reset_env_ids) > 0:
            env_ids = self.reset_env_ids
            self.terminal_root_pos_w[env_ids] = self.robot.data.root_pos_w[env_ids]
            self.terminal_root_quat_w[env_ids] = self.robot.data.root_quat_w[env_ids]
            self.terminal_root_lin_vel_w[env_ids] = self.robot.data.root_lin_vel_w[env_ids]
            self.terminal_root_accel_mps2[env_ids] = self.last_root_accel_mps2[env_ids]
            self.terminal_tilt_rad[env_ids] = self.last_tilt_rad[env_ids]
            self.terminal_episode_max_radial_dist[env_ids] = self.episode_max_radial_dist[env_ids]
            self.terminal_feet_pos_w[env_ids] = self.robot.data.body_pos_w[env_ids][:, self.feet_body_ids, :]
            net_contact_forces = self.contact_sensor.data.net_forces_w_history
            diagnostic_contact_force = torch.max(
                torch.norm(net_contact_forces[:, :, self.diagnostic_contact_cfg.body_ids], dim=-1),
                dim=1,
            )[0]
            self.terminal_diagnostic_contact_force_n[env_ids] = diagnostic_contact_force[env_ids]
            feet_contact = (
                torch.max(
                    torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1),
                    dim=1,
                )[0]
                > 0.5
            )
            self.terminal_feet_contact[env_ids] = feet_contact[env_ids]
        self.reset(self.reset_env_ids)
        if len(self.reset_env_ids) == 0:
            self.extras.pop("log", None)

        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}
        return actor_obs, reward_buf, self.reset_buf, self.extras

    def _resample_impact_immunity(self) -> None:
        self.impact_immunity = impact_immunity_from_draws(
            torch.rand(self.num_envs, device=self.device), fraction=LIGHTLP_IMMUNITY_FRAC
        )

    def check_reset(self):
        if self.use_lightlp_terminations:
            return self._check_reset_lightlp()
        return self._check_reset_stage_e()

    def _check_reset_stage_e(self):
        net_contact_forces = self.contact_sensor.data.net_forces_w_history
        reset_buf = torch.any(
            torch.max(
                torch.norm(net_contact_forces[:, :, self.termination_contact_cfg.body_ids], dim=-1),
                dim=1,
            )[0]
            > 1.0,
            dim=1,
        )
        time_out_buf = self.episode_length_buf >= self.max_episode_length
        reset_buf |= time_out_buf

        # Orientation fall termination (VITAL parity) remains a geometric fallback:
        # the trunk now has collision, but a large tilt can occur before contact.
        roll, pitch, _ = euler_xyz_from_quat(self.robot.data.root_quat_w)
        roll = torch.atan2(torch.sin(roll), torch.cos(roll))
        pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch))
        reset_buf |= (torch.abs(pitch) > 1.0) | (torch.abs(roll) > 0.8)

        self.pit_fall_buf.zero_()
        is_sparse = self.refresh_sparse_tile_mask()
        if self.terminate_on_pit_fall and self.sparse_foothold_type_ids and torch.any(is_sparse):
            self.pit_fall_buf.copy_(
                sparse_pit_fall_mask(
                    self.robot.data.root_pos_w[:, 2],
                    self.scene.env_origins[:, 2],
                    is_sparse,
                    drop_threshold=0.5,
                    soft_terrain=self.soft_sparse_terrain,
                )
            )
            reset_buf |= self.pit_fall_buf

        return reset_buf, time_out_buf

    def _check_reset_lightlp(self):
        """LightLP §IV-C2: gather tensors, then the shared helper decides flags."""
        tile = float(self.cfg.scene.terrain_generator.size[0])
        offset = self.robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        net_contact_forces = self.contact_sensor.data.net_forces_w_history
        torso_hit = torch.any(
            torch.max(
                torch.norm(net_contact_forces[:, :, self.termination_contact_cfg.body_ids], dim=-1),
                dim=1,
            )[0]
            > 1.0,
            dim=1,
        )
        lin_vel = self.robot.data.root_lin_vel_w
        accel = torch.norm((lin_vel - self.prev_root_lin_vel_w) / max(self.step_dt, 1.0e-6), dim=1)
        self.prev_root_lin_vel_w.copy_(lin_vel)
        gravity_b = self.robot.data.projected_gravity_b
        self.last_root_accel_mps2.copy_(accel)
        self.last_tilt_rad.copy_(
            tilt_from_upright_rad(gravity_b[:, 0], gravity_b[:, 1], gravity_b[:, 2])
        )
        accel_for_gate = mask_recent_push_accel(
            accel, self.sim_step_counter, self._push_step_marker, int(self.cfg.sim.decimation)
        )
        reset_buf, time_out_buf, reasons = lightlp_timeout_and_reset(
            episode_timeout=self.episode_length_buf >= self.max_episode_length,
            offset_xy=offset,
            tile_size=tile,
            max_abs_joint_vel=torch.max(torch.abs(self.robot.data.joint_vel), dim=1)[0],
            torso_hit=torso_hit,
            accel_mps2=accel_for_gate,
            elapsed_s=self.episode_length_buf.float() * self.step_dt,
            gravity_gx=gravity_b[:, 0],
            gravity_gy=gravity_b[:, 1],
            gravity_gz=gravity_b[:, 2],
            fall_draws=torch.rand(self.num_envs, device=self.device),
            immunity=self.impact_immunity,
        )

        self.pit_fall_buf.zero_()
        is_sparse = self.refresh_sparse_tile_mask()
        if self.sparse_foothold_type_ids and torch.any(is_sparse):
            self.pit_fall_buf.copy_(
                sparse_pit_fall_mask(
                    self.robot.data.root_pos_w[:, 2],
                    self.scene.env_origins[:, 2],
                    is_sparse,
                    drop_threshold=0.5,
                    soft_terrain=False,
                )
            )
        self.reset_reason_masks = reasons
        return reset_buf, time_out_buf

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return

        self.avg_feet_force_per_step[env_ids] = 0.0
        self.avg_feet_speed_per_step[env_ids] = 0.0

        self.extras["log"] = dict()
        if self.cfg.scene.terrain_generator is not None and self.cfg.scene.terrain_generator.curriculum:
            self.extras["log"].update(self.update_terrain_levels(env_ids))

        self.scene.reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset",
                env_ids=env_ids,
                dt=self.step_dt,
                global_env_step_count=self.sim_step_counter // self.cfg.sim.decimation,
            )

        reward_extras = self.reward_manager.reset(env_ids)
        self.extras["log"].update(reward_extras)
        self.extras["log"].update(self.command_provenance_log())
        self.extras["time_outs"] = self.time_out_buf

        self.command_generator.reset(env_ids)
        self._resample_terrain_aware_commands(env_ids)
        self._enforce_terrain_aware_commands()
        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        self.scan_obs_buffer.reset(env_ids)
        self.foot_accel_ema[env_ids] = 0.0
        # Seed with post-reset foot velocity so the first EMA step is not v/dt spike.
        self.prev_foot_lin_vel_w[env_ids] = self.robot.data.body_lin_vel_w[env_ids][:, self.feet_body_ids, :]
        self.action_buffer.reset(env_ids)
        self.episode_length_buf[env_ids] = 0
        self.episode_max_radial_dist[env_ids] = 0.0
        self.episode_path_length[env_ids] = 0.0
        self.episode_tracking_sum[env_ids] = 0.0
        self.episode_tracking_steps[env_ids] = 0.0
        self.gait_time[env_ids] = 0.0

        self.scene.write_data_to_sim()
        self.sim.forward()
        if hasattr(self, "_rtx_reset_sim_step"):
            self._rtx_reset_sim_step[env_ids] = int(self.sim_step_counter)
        self.prev_step_root_pos_w[env_ids] = self.robot.data.root_pos_w[env_ids]

    def update_terrain_levels(self, env_ids):
        """Apply terrain curriculum and update recent per-bucket behavior metrics."""
        max_dist = self.episode_max_radial_dist[env_ids]
        path_length = self.episode_path_length[env_ids]
        tracking_steps = self.episode_tracking_steps[env_ids]
        tracking_mean = self.episode_tracking_sum[env_ids] / tracking_steps.clamp(min=1.0)
        tracking_mean = torch.where(tracking_steps > 0.0, tracking_mean, torch.zeros_like(tracking_mean))
        command_norm = torch.norm(self.command_generator.command[env_ids, :2], dim=1)
        # Horizon timeout resamples the command on the same step; use the episode
        # moving history so a fresh stand/move draw cannot flip promotion.
        if self.use_lightlp_terminations:
            command_norm = torch.where(
                tracking_steps > 0.0,
                torch.clamp(command_norm, min=STANDING_COMMAND_THRESHOLD + 1.0e-3),
                torch.zeros_like(command_norm),
            )
        moving = command_norm > STANDING_COMMAND_THRESHOLD
        tile_size = self.scene.terrain.cfg.terrain_generator.size[0]
        if self.use_lightlp_terminations:
            move_up, move_down = lightlp_terrain_level_moves(
                path_length=path_length,
                tracking_mean=tracking_mean,
                command_lin_vel_norm=command_norm,
                tile_size=tile_size,
                tracking_threshold=LIGHTLP_TRACKING_WELL_THRESHOLD,
            )
        else:
            move_up, move_down = terrain_level_moves(
                max_radial_dist=max_dist,
                command_lin_vel_norm=command_norm,
                episode_length_s=self.max_episode_length_s,
                tile_size=tile_size,
            )
        terrain_types = self.scene.terrain.terrain_types[env_ids].long()
        terrain_levels = self.scene.terrain.terrain_levels[env_ids].long()
        is_sparse = self._columns_mask(terrain_types, self.sparse_foothold_type_ids)
        timed_out = self.time_out_buf[env_ids]
        pit_fall = self.pit_fall_buf[env_ids]
        if self.use_lightlp_terminations:
            move_up = lightlp_sparse_promotion_guard(move_up, is_sparse, pit_fall)
        if not self.use_lightlp_terminations:
            move_up, move_down = sparse_curriculum_moves(
                move_up=move_up,
                move_down=move_down,
                is_sparse=is_sparse,
                moving=moving,
                timed_out=timed_out,
                pit_fall=pit_fall,
            )
        move_down = mask_sparse_curriculum_demote(
            move_down,
            is_sparse,
            demote_sparse=bool(getattr(self.cfg, "sparse_curriculum_demote", True)),
        )
        promotion, timeout_success, _fall = monitor_outcome_flags(move_up, timed_out)
        self._update_terrain_metrics(
            terrain_types=terrain_types,
            terrain_levels=terrain_levels,
            moving=moving,
            promotion=promotion,
            timeout_success=timeout_success,
            timed_out=timed_out,
            pit_fall=pit_fall,
            max_dist=max_dist,
        )
        self.scene.terrain.update_env_origins(env_ids, move_up, move_down)
        random_reset_logs = self._apply_random_level_resets(env_ids)
        n_all = torch.tensor(float(self.num_envs), device=self.device)
        n_reset = torch.tensor(float(len(env_ids)), device=self.device)
        logs = {
            "Curriculum/terrain_levels": torch.mean(self.scene.terrain.terrain_levels.float()),
            f"Curriculum/terrain_levels{LOG_COUNT_SUFFIX}": n_all,
            "Curriculum/episode_max_radial_dist": torch.mean(max_dist),
            f"Curriculum/episode_max_radial_dist{LOG_COUNT_SUFFIX}": n_reset,
            "Curriculum/episode_path_length": torch.mean(path_length),
            f"Curriculum/episode_path_length{LOG_COUNT_SUFFIX}": n_reset,
            "Curriculum/episode_tracking_mean": torch.mean(tracking_mean),
            f"Curriculum/episode_tracking_mean{LOG_COUNT_SUFFIX}": n_reset,
        }
        logs["Curriculum/promotion_rate"] = promotion.float().mean()
        logs[f"Curriculum/promotion_rate{LOG_COUNT_SUFFIX}"] = n_reset
        logs["Curriculum/timeout_success_rate"] = timeout_success.float().mean()
        logs[f"Curriculum/timeout_success_rate{LOG_COUNT_SUFFIX}"] = n_reset
        logs.update(self._reset_reason_log(env_ids))
        logs.update(self._terrain_metrics_log())
        logs.update(self._batch_terrain_outcome_log(env_ids, promotion, timeout_success, timed_out, terrain_types))
        logs.update(random_reset_logs)
        return logs

    def _reset_reason_log(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Fractions of this reset batch attributed to each LightLP terminator."""
        n = torch.tensor(float(len(env_ids)), device=self.device)
        logs = {"Reset/n": n}
        reasons = getattr(self, "reset_reason_masks", None)
        if reasons:
            for name, mask in reasons.items():
                logs[f"Reset/{name}"] = mask[env_ids].float().mean()
                logs[f"Reset/{name}{LOG_COUNT_SUFFIX}"] = n
        if len(env_ids) > 0:
            logs["Reset/pit_fall"] = self.pit_fall_buf[env_ids].float().mean()
            logs[f"Reset/pit_fall{LOG_COUNT_SUFFIX}"] = n
            logs["Reset/timeout"] = self.time_out_buf[env_ids].float().mean()
            logs[f"Reset/timeout{LOG_COUNT_SUFFIX}"] = n
        return logs

    def _apply_random_level_resets(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Place a fraction of resets on a random row, independent of performance."""
        logs: dict[str, torch.Tensor] = {}
        fraction = float(getattr(self.cfg, "random_level_reset_fraction", 0.0) or 0.0)
        n_reset = torch.tensor(float(max(len(env_ids), 1)), device=self.device)
        if fraction <= 0.0 or len(env_ids) == 0:
            return logs
        terrain = self.scene.terrain
        if getattr(terrain, "terrain_origins", None) is None:
            return logs
        draws = torch.rand(len(env_ids), device=self.device)
        pick = random_level_reset_mask(draws, fraction=fraction)
        logs["Curriculum/random_level_reset_frac"] = pick.float().mean()
        logs[f"Curriculum/random_level_reset_frac{LOG_COUNT_SUFFIX}"] = n_reset
        if not bool(pick.any()):
            return logs
        chosen = env_ids[pick]
        n_pick = torch.tensor(float(len(chosen)), device=self.device)
        logs["Curriculum/random_level_before_mean"] = terrain.terrain_levels[chosen].float().mean()
        logs[f"Curriculum/random_level_before_mean{LOG_COUNT_SUFFIX}"] = n_pick
        max_level = random_level_reset_high(
            int(terrain.max_terrain_level),
            getattr(self.cfg, "random_level_reset_max_level", None),
        )
        min_level = random_level_reset_low(
            getattr(self.cfg, "random_level_reset_min_level", None),
            max_level,
        )
        terrain.terrain_levels[chosen] = torch.randint(
            min_level, max_level, (len(chosen),), device=self.device, dtype=terrain.terrain_levels.dtype
        )
        terrain.env_origins[chosen] = terrain.terrain_origins[
            terrain.terrain_levels[chosen], terrain.terrain_types[chosen]
        ]
        logs["Curriculum/random_level_after_mean"] = terrain.terrain_levels[chosen].float().mean()
        logs[f"Curriculum/random_level_after_mean{LOG_COUNT_SUFFIX}"] = n_pick
        return logs

    def _update_terrain_metrics(
        self,
        *,
        terrain_types: torch.Tensor,
        terrain_levels: torch.Tensor,
        moving: torch.Tensor,
        promotion: torch.Tensor,
        timeout_success: torch.Tensor,
        timed_out: torch.Tensor,
        pit_fall: torch.Tensor,
        max_dist: torch.Tensor,
    ) -> None:
        """Maintain recent episode-weighted EMAs for TensorBoard monitoring."""
        if not self.terrain_type_names or not bool(moving.any()):
            return
        max_level = max(1, self.cfg.scene.terrain_generator.num_rows - 1)
        band_ids = torch.clamp((terrain_levels * 3) // (max_level + 1), min=0, max=2)
        fall = ~timed_out
        values = (
            promotion.float(),
            timeout_success.float(),
            timeout_success.float(),
            (max_dist >= 1.0).float(),
            (max_dist >= 2.0).float(),
            (max_dist >= 4.0).float(),
            fall.float(),
            timed_out.float(),
            pit_fall.float(),
            max_dist,
        )

        for type_id, name in enumerate(self.terrain_type_names):
            columns = self.terrain_name_to_columns.get(name, [])
            type_mask = moving & self._columns_mask(terrain_types, columns)
            self._update_terrain_metric_slot(type_id, 0, type_mask, values)
            if name not in SPARSE_FOOTHOLD_NAMES:
                continue
            for band_id in range(len(self._TERRAIN_METRIC_BANDS)):
                self._update_terrain_metric_slot(type_id, 1 + band_id, type_mask & (band_ids == band_id), values)

    def _update_terrain_metric_slot(self, type_id: int, slot: int, mask: torch.Tensor, values: tuple) -> None:
        count = int(mask.sum().item())
        if count == 0:
            return
        batch = torch.stack([value[mask].mean() for value in values])
        if self.terrain_metric_initialized[type_id, slot]:
            alpha = 1.0 - (1.0 - self._TERRAIN_METRIC_EPISODE_ALPHA) ** count
            self.terrain_metric_ema[type_id, slot].lerp_(batch, alpha)
        else:
            self.terrain_metric_ema[type_id, slot].copy_(batch)
            self.terrain_metric_initialized[type_id, slot] = True
        self.terrain_metric_episodes[type_id, slot] += count

    _TERRAIN_LOG_METRICS = ("reach_2m_rate", "reach_4m_rate", "progress_m")
    _TERRAIN_BAND_LOG_METRICS = ("reach_2m_rate", "progress_m")

    def _terrain_metrics_log(self) -> dict[str, torch.Tensor]:
        logs: dict[str, torch.Tensor] = {}
        keep = set(self._TERRAIN_LOG_METRICS)
        band_keep = set(self._TERRAIN_BAND_LOG_METRICS)
        for type_id, name in enumerate(self.terrain_type_names):
            values = self.terrain_metric_ema[type_id, 0]
            for metric_id, metric in enumerate(self._TERRAIN_METRIC_FIELDS):
                if metric not in keep:
                    continue
                logs[f"Terrain/{name}/{metric}"] = values[metric_id]
            logs[f"Terrain/{name}/episodes"] = self.terrain_metric_episodes[type_id, 0].float()
            if name not in SPARSE_FOOTHOLD_NAMES:
                continue
            for band_id, band in enumerate(self._TERRAIN_METRIC_BANDS, start=1):
                band_values = self.terrain_metric_ema[type_id, band_id]
                for metric_id, metric in enumerate(self._TERRAIN_METRIC_FIELDS):
                    if metric not in band_keep:
                        continue
                    logs[f"Terrain/{name}/{band}_{metric}"] = band_values[metric_id]
                logs[f"Terrain/{name}/{band}_episodes"] = self.terrain_metric_episodes[type_id, band_id].float()
        terrain = getattr(self.scene, "terrain", None)
        levels_all = getattr(terrain, "terrain_levels", None)
        if levels_all is not None:
            max_level = max(1, self.cfg.scene.terrain_generator.num_rows - 1)
            n_all = torch.tensor(float(self.num_envs), device=self.device)
            for level in range(max_level + 1):
                logs[f"Curriculum/level_{level}_frac"] = (levels_all == level).float().mean()
                logs[f"Curriculum/level_{level}_frac{LOG_COUNT_SUFFIX}"] = n_all
        return logs

    """
    Curriculum-coupled schedules.
    """

    def terrain_difficulty(self) -> torch.Tensor:
        """Per-env terrain difficulty in ``[0, 1]`` derived from the terrain level.

        This is the env-level curriculum signal only. Capability claims still come
        from the fixed evaluator, never from this value.
        """
        terrain_levels = getattr(self.scene.terrain, "terrain_levels", None)
        if terrain_levels is None:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        max_level = max(1, self.cfg.scene.terrain_generator.num_rows - 1)
        return (terrain_levels.float() / max_level).clamp(0.0, 1.0)

    @staticmethod
    def _decay_scale(difficulty: torch.Tensor, start: float, min_scale: float) -> torch.Tensor:
        ramp = ((difficulty - start) / max(1.0e-6, 1.0 - start)).clamp(0.0, 1.0)
        return 1.0 - ramp * (1.0 - min_scale)

    def amp_reward_coef_scale(self) -> torch.Tensor:
        """Per-env AMP style weight multiplier.

        The expert set is flat-ground only, so the style weight decays with terrain
        difficulty instead of punishing the stair and strong-rough gaits the task
        reward asks for. Sparse tiles force AMP to zero (LightLP §IV has no AMP).
        """
        schedule = self.cfg.amp_terrain_schedule
        if not schedule.enable:
            scale = torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        elif schedule.mode != "linear_decay":
            raise NotImplementedError(f"unsupported AMP terrain schedule mode {schedule.mode!r}")
        else:
            scale = self._decay_scale(self.terrain_difficulty(), schedule.decay_start_difficulty, schedule.min_scale)
        if self.sparse_foothold_type_ids:
            self.refresh_sparse_tile_mask()
            scale = scale * (~self.sparse_tile_mask).float()
        return scale

    def _planar_vel_yaw(self):
        """Root planar velocity in the yaw frame, matching track_lin_vel_xy_exp."""
        vel_yaw = quat_rotate_inverse(yaw_quat(self.robot.data.root_quat_w), self.robot.data.root_lin_vel_w[:, :3])
        return vel_yaw[:, :2]

    def _gait_tracking_scale(self):
        return gait_tracking_scale(
            self.command_generator.command[:, :2],
            self._planar_vel_yaw(),
            tracking_std=self.cfg.gait.tracking_std,
        )

    def _update_gait(self) -> None:
        gait = self.cfg.gait
        cmd_speed = torch.norm(self.command_generator.command[:, :2], dim=1)
        tracking_scale = self._gait_tracking_scale()
        moving = cmd_speed > STANDING_COMMAND_THRESHOLD

        if gait.mode == "command_conditioned":
            blend = (cmd_speed / max(1.0e-6, gait.reference_max_speed)).clamp(0.0, 1.0)
            self.gait_cycle = gait.slow_gait_cycle + blend * (gait.fast_gait_cycle - gait.slow_gait_cycle)
        elif gait.mode not in ("fixed_clock", "difficulty_relaxed"):
            raise NotImplementedError(f"unsupported gait mode {gait.mode!r}")

        # Freeze the clock on standing commands so the policy is not forced to march.
        self.gait_time = self.gait_time + (self.step_dt / self.gait_cycle) * moving.float()
        self.gait_phase[:, 0] = (self.gait_time + self.phase_offset[:, 0]) % 1.0
        self.gait_phase[:, 1] = (self.gait_time + self.phase_offset[:, 1]) % 1.0

        terrain_scale = torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        if gait.mode == "difficulty_relaxed":
            terrain_scale = self._decay_scale(
                self.terrain_difficulty(), gait.gait_relax_start_difficulty, gait.min_gait_reward_scale
            )
        self.gait_reward_scale = tracking_scale * terrain_scale
        # LightLP sparse mix: no periodic gait on stones / pillars.
        if self.sparse_foothold_type_ids:
            self.refresh_sparse_tile_mask()
            self.gait_reward_scale = self.gait_reward_scale * (~self.sparse_tile_mask).float()

    def _terrain_aware_commands_active(self) -> bool:
        if not bool(getattr(self.cfg, "terrain_aware_commands", False)):
            return False
        ranges = self.cfg.commands.ranges
        return terrain_aware_commands_enabled(
            enabled=True,
            lin_vel_x=ranges.lin_vel_x,
            lin_vel_y=ranges.lin_vel_y,
            ang_vel_z=ranges.ang_vel_z,
        )

    def _base_heading_w(self) -> torch.Tensor:
        heading = getattr(self.robot.data, "heading_w", None)
        if heading is not None:
            return heading
        return euler_xyz_from_quat(self.robot.data.root_quat_w)[2]

    def _command_storage(self) -> torch.Tensor:
        gen = self.command_generator
        stored = getattr(gen, "vel_command_b", None)
        command = gen.command
        return stored if stored is not None else command

    def _resample_terrain_aware_commands(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        if not self._terrain_aware_commands_active():
            self._sparse_command_active[env_ids] = False
            return
        self.refresh_sparse_tile_mask()
        is_sparse = self.sparse_tile_mask[env_ids]
        self._sparse_command_active[env_ids] = is_sparse
        if not bool(is_sparse.any()):
            return
        chosen = env_ids[is_sparse]
        n = int(chosen.numel())
        vx_u = torch.rand(n, device=self.device)
        yaw_mode_u = torch.rand(n, device=self.device)
        yaw_u = torch.rand(n, device=self.device)
        vx, vy, wz = sample_sparse_foothold_velocity(
            vx_u,
            yaw_mode_u,
            yaw_u,
            vx_range=getattr(self.cfg, "sparse_command_lin_vel_x", SPARSE_FOOTHOLD_VX_RANGE),
            straight_prob=float(
                getattr(self.cfg, "sparse_command_straight_yaw_prob", SPARSE_FOOTHOLD_STRAIGHT_YAW_PROB)
            ),
            gentle_yaw_range=getattr(self.cfg, "sparse_command_gentle_ang_vel_z", SPARSE_FOOTHOLD_GENTLE_YAW_RANGE),
        )
        self._sparse_command[chosen, 0] = vx
        self._sparse_command[chosen, 1] = vy
        self._sparse_command[chosen, 2] = wz

    def _enforce_terrain_aware_commands(self) -> None:
        if not self._terrain_aware_commands_active():
            return
        mask = self._sparse_command_active
        if not bool(mask.any()):
            return
        storage = self._command_storage()
        storage[mask] = self._sparse_command[mask]
        command = self.command_generator.command
        if command is not storage:
            command[mask] = self._sparse_command[mask]
        standing = getattr(self.command_generator, "is_standing_env", None)
        if standing is not None:
            standing[mask] = False
        heading_env = getattr(self.command_generator, "is_heading_env", None)
        if heading_env is not None:
            heading_env[mask] = False
        heading_target = getattr(self.command_generator, "heading_target", None)
        if heading_target is not None:
            heading_target[mask] = self._base_heading_w()[mask]

    def _batch_terrain_outcome_log(
        self,
        env_ids: torch.Tensor,
        promotion: torch.Tensor,
        timeout_success: torch.Tensor,
        timed_out: torch.Tensor,
        terrain_types: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """This-reset-batch rates with counts, so four ranks can be reduced by sum/count."""
        logs: dict[str, torch.Tensor] = {}
        if not self.terrain_type_names or len(env_ids) == 0:
            return logs
        for name in SPARSE_FOOTHOLD_NAMES:
            columns = self.terrain_name_to_columns.get(name, [])
            type_mask = self._columns_mask(terrain_types, columns)
            count = type_mask.sum()
            if int(count.item()) == 0:
                continue
            n = count.float()
            logs[f"Terrain/{name}/promotion_rate"] = promotion[type_mask].float().mean()
            logs[f"Terrain/{name}/promotion_rate{LOG_COUNT_SUFFIX}"] = n
        return logs

    def command_provenance_log(self) -> dict:
        """Log what actually reached the reward, not just what was requested."""
        command = self.command_generator.command
        n_all = torch.tensor(float(self.num_envs), device=self.device)
        vx = command[:, 0]
        wz = command[:, 2]
        lin = torch.norm(command[:, :2], dim=1)
        standing = (lin < 0.1).float()
        logs = {
            "Command/generated_lin_vel_x": torch.mean(vx),
            "Command/generated_lin_vel_y": torch.mean(command[:, 1]),
            "Command/generated_ang_vel_z": torch.mean(wz),
            "Command/standing_env_fraction": torch.mean(standing),
            "Command/bin_standing": torch.mean(standing),
            "Command/bin_vx_reverse": (vx < 0.0).float().mean(),
            "Command/bin_vx_low_forward": ((vx >= 0.0) & (vx < 0.6) & (lin >= 0.1)).float().mean(),
            "Command/bin_vx_sparse_forward": ((vx >= 0.6) & (vx <= 2.0)).float().mean(),
            "Command/bin_yaw_straight": (wz.abs() < 1.0e-6).float().mean(),
            "Command/bin_yaw_gentle": ((wz.abs() >= 1.0e-6) & (wz.abs() <= 0.3)).float().mean(),
            "Command/bin_yaw_full": (wz.abs() > 0.3).float().mean(),
            "Curriculum/terrain_difficulty": torch.mean(self.terrain_difficulty()),
            "Curriculum/amp_reward_coef_scale": torch.mean(self.amp_reward_coef_scale()),
            "Curriculum/gait_reward_scale": torch.mean(self.gait_reward_scale),
            "Curriculum/gait_tracking_scale": torch.mean(self._gait_tracking_scale()),
        }
        counted = [
            "Command/generated_lin_vel_x",
            "Command/generated_lin_vel_y",
            "Command/generated_ang_vel_z",
            "Command/standing_env_fraction",
            "Command/bin_standing",
            "Command/bin_vx_reverse",
            "Command/bin_vx_low_forward",
            "Command/bin_vx_sparse_forward",
            "Command/bin_yaw_straight",
            "Command/bin_yaw_gentle",
            "Command/bin_yaw_full",
            "Curriculum/terrain_difficulty",
            "Curriculum/amp_reward_coef_scale",
            "Curriculum/gait_reward_scale",
            "Curriculum/gait_tracking_scale",
        ]
        for key in counted:
            logs[f"{key}{LOG_COUNT_SUFFIX}"] = n_all
        is_sparse = self.sparse_tile_mask
        logs["Command/sparse_env_fraction"] = is_sparse.float().mean()
        logs[f"Command/sparse_env_fraction{LOG_COUNT_SUFFIX}"] = n_all
        if bool(is_sparse.any()):
            sparse_n = is_sparse.sum().float()
            logs["Command/sparse_mean_vx"] = command[is_sparse, 0].mean()
            logs["Command/sparse_zero_yaw_frac"] = (command[is_sparse, 2].abs() < 1.0e-6).float().mean()
            logs["Command/sparse_standing_frac"] = standing[is_sparse].mean()
            for key in ("Command/sparse_mean_vx", "Command/sparse_zero_yaw_frac", "Command/sparse_standing_frac"):
                logs[f"{key}{LOG_COUNT_SUFFIX}"] = sparse_n
        nonsparse = ~is_sparse
        if bool(nonsparse.any()):
            nonsparse_n = nonsparse.sum().float()
            logs["Command/nonsparse_reverse_frac"] = (command[nonsparse, 0] < 0.0).float().mean()
            logs["Command/nonsparse_standing_frac"] = standing[nonsparse].mean()
            for key in ("Command/nonsparse_reverse_frac", "Command/nonsparse_standing_frac"):
                logs[f"{key}{LOG_COUNT_SUFFIX}"] = nonsparse_n
        return logs

    @staticmethod
    def seed(seed: int = -1) -> int:
        try:
            import omni.replicator.core as rep  # type: ignore

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        return torch_utils.set_seed(seed)

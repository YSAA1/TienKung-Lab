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
    NUM_T4_JOINTS,
    PROPRIO_FRAME_DIM,
    TEACHER_SCAN_CLIP,
    TEACHER_SCAN_DIM,
    TEACHER_SCAN_INVALID_VALUE,
    assert_no_privilege_leakage,
    proprio_field_slice,
)
from legged_lab.envs.t4.amp_features import T4AmpFeatureBuilder
from legged_lab.envs.t4.curriculum import (
    STANDING_COMMAND_THRESHOLD,
    gait_tracking_scale,
    terrain_level_moves,
)
from legged_lab.envs.t4.teacher_cfg import T4LocoTeacherEnvCfg
from legged_lab.utils.env_utils.scene import SceneCfg
from rsl_rl.env import VecEnv


class T4LocoEnv(VecEnv):
    """PPO + AMP locomotion env for the T4 27-DoF humanoid."""

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

        self.init_buffers()

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.event_manager = EventManager(self.cfg.domain_rand.events, self)
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        self.reset_env_ids = env_ids
        self.reset(env_ids)

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
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

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
        if height_scan.shape[-1] != TEACHER_SCAN_DIM:
            raise RuntimeError(
                f"teacher scan width {height_scan.shape[-1]} does not match schema width {TEACHER_SCAN_DIM}"
            )
        return height_scan * self.obs_scales.height_scan

    def compute_observations(self):
        current_actor_obs, current_critic_obs = self.compute_current_observations()
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec

        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)

        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)

        height_scan = self.compute_teacher_terrain_privilege()
        critic_obs = torch.cat([critic_obs, height_scan], dim=-1)
        if self.add_noise:
            height_scan = height_scan + (2 * torch.rand_like(height_scan) - 1) * self.height_scan_noise_vec
        actor_obs = torch.cat([actor_obs, height_scan], dim=-1)

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
        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            self.robot.set_joint_position_target(processed_actions)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

            self.avg_feet_force_per_step += torch.norm(
                self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids, :3], dim=-1
            )
            self.avg_feet_speed_per_step += torch.norm(self.robot.data.body_lin_vel_w[:, self.feet_body_ids, :], dim=-1)

        self.avg_feet_force_per_step /= self.cfg.sim.decimation
        self.avg_feet_speed_per_step /= self.cfg.sim.decimation

        if not self.headless:
            self.sim.render()

        self.episode_length_buf += 1
        radial_dist = torch.norm(self.robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2], dim=1)
        self.episode_max_radial_dist = torch.maximum(self.episode_max_radial_dist, radial_dist)
        self._update_gait()

        self.command_generator.compute(self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.reset_buf, self.time_out_buf = self.check_reset()
        reward_buf = self.reward_manager.compute(self.step_dt)
        self.reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset(self.reset_env_ids)

        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}
        return actor_obs, reward_buf, self.reset_buf, self.extras

    def check_reset(self):
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

        # Orientation fall termination (VITAL parity): the URDF keeps collision
        # geometry only on the feet/hands, so trunk-contact termination can never
        # fire and falls would otherwise run out the full episode.
        roll, pitch, _ = euler_xyz_from_quat(self.robot.data.root_quat_w)
        roll = torch.atan2(torch.sin(roll), torch.cos(roll))
        pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch))
        reset_buf |= (torch.abs(pitch) > 1.0) | (torch.abs(roll) > 0.8)

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
        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        self.action_buffer.reset(env_ids)
        self.episode_length_buf[env_ids] = 0
        self.episode_max_radial_dist[env_ids] = 0.0
        self.gait_time[env_ids] = 0.0

        self.scene.write_data_to_sim()
        self.sim.forward()

    def update_terrain_levels(self, env_ids):
        """Apply the frozen curriculum algebra; see :mod:`legged_lab.envs.t4.curriculum`."""
        max_dist = self.episode_max_radial_dist[env_ids]
        move_up, move_down = terrain_level_moves(
            max_radial_dist=max_dist,
            command_lin_vel_norm=torch.norm(self.command_generator.command[env_ids, :2], dim=1),
            episode_length_s=self.max_episode_length_s,
            tile_size=self.scene.terrain.cfg.terrain_generator.size[0],
        )
        self.scene.terrain.update_env_origins(env_ids, move_up, move_down)
        return {
            "Curriculum/terrain_levels": torch.mean(self.scene.terrain.terrain_levels.float()),
            "Curriculum/episode_max_radial_dist": torch.mean(max_dist),
        }

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
        reward asks for. The schedule is frozen before Stage E starts.
        """
        schedule = self.cfg.amp_terrain_schedule
        if not schedule.enable:
            return torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        if schedule.mode != "linear_decay":
            raise NotImplementedError(f"unsupported AMP terrain schedule mode {schedule.mode!r}")
        return self._decay_scale(self.terrain_difficulty(), schedule.decay_start_difficulty, schedule.min_scale)

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

    def command_provenance_log(self) -> dict:
        """Log what actually reached the reward, not just what was requested."""
        command = self.command_generator.command
        return {
            "Command/generated_lin_vel_x": torch.mean(command[:, 0]),
            "Command/generated_lin_vel_y": torch.mean(command[:, 1]),
            "Command/generated_ang_vel_z": torch.mean(command[:, 2]),
            "Command/standing_env_fraction": torch.mean((torch.norm(command, dim=1) < 0.1).float()),
            "Curriculum/terrain_difficulty": torch.mean(self.terrain_difficulty()),
            "Curriculum/amp_reward_coef_scale": torch.mean(self.amp_reward_coef_scale()),
            "Curriculum/gait_reward_scale": torch.mean(self.gait_reward_scale),
            "Curriculum/gait_tracking_scale": torch.mean(self._gait_tracking_scale()),
        }

    @staticmethod
    def seed(seed: int = -1) -> int:
        try:
            import omni.replicator.core as rep  # type: ignore

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        return torch_utils.set_seed(seed)

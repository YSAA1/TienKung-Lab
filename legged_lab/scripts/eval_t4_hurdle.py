"""Fixed-command behavior and perception evaluator for T4 terrain buckets.

The evaluator keeps the command and reset seeds fixed, separates terminal causes,
and can ablate only the actor height-scan block. Optional ``--spawn_y_offset_m`` /
``--spawn_yaw_deg`` pin a pad-safe first-step pose on stepping stones and raised
pillars. It remains a diagnostic rather than the final zero-contact or exact-footstep
acceptance gate.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

from legged_lab.scripts.isaaclab_runtime_compat import (
    patch_missing_physx_material_attributes,
    patch_physx_backward_compatibility_setting,
)

parser = argparse.ArgumentParser(description="Evaluate T4 Stage E terrain progress.")
parser.add_argument("--task", default="t4_loco_teacher")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--load_run", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--difficulty", type=float, default=0.85)
parser.add_argument("--progress_m", type=float, default=3.0)
parser.add_argument("--command_vx", type=float, default=0.7)
parser.add_argument("--zero_actions", action="store_true")
parser.add_argument("--keep_randomization", action="store_true")
parser.add_argument(
    "--stochastic",
    action="store_true",
    help="Sample the PPO action distribution instead of using the deterministic actor mean.",
)
parser.add_argument(
    "--terrain_type",
    choices=("hurdles", "flat", "stepping_stones", "raised_pillars"),
    default="hurdles",
)
parser.add_argument(
    "--scan_mode",
    choices=("normal", "zero", "permuted"),
    default="normal",
    help="Actor scan intervention. Proprioception and contact flags are unchanged.",
)
parser.add_argument(
    "--scan_permutation_seed",
    type=int,
    default=0,
    help="Seed for the fixed within-frame spatial scan permutation.",
)
parser.add_argument(
    "--spawn_y_offset_m",
    type=float,
    default=None,
    help="Pin reset y relative to the tile center in meters. Unset keeps the seeded random reset.",
)
parser.add_argument(
    "--spawn_yaw_deg",
    type=float,
    default=None,
    help="Pin reset yaw error in degrees. Unset keeps the seeded random reset.",
)
parser.add_argument(
    "--disable_student_depth_noise",
    action="store_true",
    help="Eval-only: turn off student depth hold/delay/scale/dropout. Camera jitter unchanged.",
)

patch_physx_backward_compatibility_setting(AppLauncher)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
if "depth_student" in (args_cli.task or ""):
    args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner  # noqa: E402

from legged_lab.assets.t4.schemas import (  # noqa: E402
    TEACHER_SCAN_DIM,
    TEACHER_SCAN_INVALID_VALUE,
)
from legged_lab.envs import *  # noqa: F401,F403,E402
from legged_lab.scripts.recurrent_policy_eval import (  # noqa: E402
    evaluate_counterfactual_actions,
    reset_recurrent_policy,
)
from legged_lab.terrains.stepping_stone_layout import (  # noqa: E402
    T4_STONE_PLATFORM_WIDTH,
    resolve_pinned_sparse_spawn,
)
from legged_lab.utils import task_registry  # noqa: E402
from legged_lab.utils.cli_args import update_rsl_rl_cfg  # noqa: E402

patch_missing_physx_material_attributes()


def evaluate() -> dict:
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)
    env_cfg.device = args_cli.device
    env_cfg.sim.device = args_cli.device
    agent_cfg.device = args_cli.device
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.terrain_generator.curriculum = False
    env_cfg.scene.terrain_generator.difficulty_range = (
        args_cli.difficulty,
        args_cli.difficulty,
    )
    sub_terrains = env_cfg.scene.terrain_generator.sub_terrains
    if args_cli.terrain_type not in sub_terrains:
        raise ValueError(f"{args_cli.terrain_type} terrain is unavailable: {sorted(sub_terrains)}")
    terrain_cfg = sub_terrains[args_cli.terrain_type]
    terrain_cfg.proportion = 1.0
    if hasattr(terrain_cfg, "soft_fill"):
        terrain_cfg.soft_fill = False
    env_cfg.scene.terrain_generator.sub_terrains = {args_cli.terrain_type: terrain_cfg}
    env_cfg.commands.rel_standing_envs = float(abs(args_cli.command_vx) <= 1.0e-9)
    env_cfg.commands.rel_heading_envs = 0.0
    env_cfg.commands.heading_command = False
    env_cfg.commands.resampling_time_range = (20.0, 20.0)
    env_cfg.commands.ranges.lin_vel_x = (args_cli.command_vx, args_cli.command_vx)
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.ranges.heading = (0.0, 0.0)
    if hasattr(env_cfg, "terrain_aware_commands"):
        env_cfg.terrain_aware_commands = False
    env_cfg.noise.add_noise = False
    if args_cli.disable_student_depth_noise and hasattr(env_cfg, "student_depth_noise"):
        env_cfg.student_depth_noise = False
    if not args_cli.keep_randomization:
        # Keep the seeded reset pose/joint perturbations. Removing both leaves a
        # perfectly symmetric nominal start where deterministic locomotion
        # policies can remain at a stationary fixed point, including known-good
        # Stage E checkpoints. Mass/material randomization and pushes are not
        # part of this fixed-command behavior check.
        for name in ("physics_material", "add_base_mass", "push_robot"):
            setattr(env_cfg.domain_rand.events, name, None)
    pinned_spawn = resolve_pinned_sparse_spawn(
        args_cli.spawn_y_offset_m,
        args_cli.spawn_yaw_deg,
        terrain_type=args_cli.terrain_type,
    )
    if pinned_spawn is not None:
        env_cfg.domain_rand.events.reset_base.params["pose_range"] = pinned_spawn["pose_range"]
        env_cfg.domain_rand.events.reset_base.params["velocity_range"] = pinned_spawn["velocity_range"]
        env_cfg.domain_rand.events.reset_robot_joints.params["position_range"] = pinned_spawn["joint_position_range"]
        env_cfg.domain_rand.events.reset_robot_joints.params["velocity_range"] = pinned_spawn["joint_velocity_range"]
    env_cfg.scene.seed = agent_cfg.seed
    env_cfg.scene.terrain_generator.num_rows = 1
    env_cfg.scene.terrain_generator.num_cols = args_cli.num_envs

    env_class = task_registry.get_task_class(args_cli.task)
    env = env_class(env_cfg, headless=True)
    root = Path(args_cli.checkpoint)
    if not root.is_file():
        root = Path(
            get_checkpoint_path(
                os.path.abspath(os.path.join("logs", agent_cfg.experiment_name)),
                args_cli.load_run,
                args_cli.checkpoint,
            )
        )
    runner_class = eval(getattr(agent_cfg, "runner_class_name", "OnPolicyRunner"))
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=str(root.parent), device=env.device)
    runner.load(str(root), load_optimizer=False)
    deterministic_policy = runner.get_inference_policy(device=env.device)
    if args_cli.stochastic:

        def stochastic_policy(policy_obs):
            if runner.cfg["empirical_normalization"]:
                policy_obs = runner.obs_normalizer(policy_obs)
            return runner.alg.policy.act(policy_obs)

    obs, _ = env.get_observations()
    scan_history_length = int(getattr(env, "teacher_scan_history_length", 1))
    scan_start = env.actor_obs_buffer.buffer.shape[-1] * env.cfg.robot.actor_obs_history_length
    scan_end = scan_start + TEACHER_SCAN_DIM * scan_history_length
    if obs.shape[-1] < scan_end:
        raise RuntimeError(f"actor observation width {obs.shape[-1]} is smaller than scan end {scan_end}")
    permutation_generator = torch.Generator(device="cpu")
    permutation_generator.manual_seed(args_cli.scan_permutation_seed)
    scan_permutation = torch.randperm(TEACHER_SCAN_DIM, generator=permutation_generator).to(env.device)

    def apply_scan_mode(policy_obs: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "normal":
            return policy_obs
        transformed = policy_obs.clone()
        scan = transformed[:, scan_start:scan_end].reshape(transformed.shape[0], scan_history_length, TEACHER_SCAN_DIM)
        if mode == "zero":
            scan.zero_()
        elif mode == "permuted":
            scan.copy_(scan.index_select(-1, scan_permutation))
        else:
            raise ValueError(f"unknown scan mode {mode!r}")
        return transformed

    def current_feet_contact() -> torch.Tensor:
        net_contact_forces = env.contact_sensor.data.net_forces_w_history
        return (
            torch.max(
                torch.norm(net_contact_forces[:, :, env.feet_cfg.body_ids], dim=-1),
                dim=1,
            )[0]
            > 0.5
        )

    episode_steps = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)
    start_pos = env.robot.data.root_pos_w.clone()

    def forward_xy(quat):
        w, x, y, z = quat.unbind(-1)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)

    start_forward_xy = forward_xy(env.robot.data.root_quat_w)
    episode_max_forward_progress = torch.zeros(env.num_envs, device=env.device)
    initial_feet_z = env.robot.data.body_pos_w[:, env.feet_body_ids, 2].clone()
    prev_feet_contact = current_feet_contact()
    first_swing_foot = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    first_swing_max_lift = torch.zeros(env.num_envs, device=env.device)
    first_swing_complete = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    swing_peak_lift = torch.zeros(env.num_envs, len(env.feet_body_ids), device=env.device)
    first_off_platform_contact_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    first_off_platform_contact_illegal = torch.zeros(env.num_envs, device=env.device)
    first_off_platform_swing_peak_lift = torch.zeros(env.num_envs, device=env.device)
    first_off_platform_touchdown_lift = torch.zeros(env.num_envs, device=env.device)
    prev_policy_actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    have_prev_policy_action = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    successes = failures = completed = 0
    timeout_episodes = strict_completion_episodes = fall_or_early_termination_episodes = 0
    pit_fall_episodes = 0
    reset_reason_counts: dict[str, int] = {}
    progress_values: list[float] = []
    monitor_radial_progress_values: list[float] = []
    final_forward_progress_values: list[float] = []
    first_swing_lift_values: list[float] = []
    first_off_platform_illegal_values: list[float] = []
    first_off_platform_swing_peak_lift_values: list[float] = []
    first_off_platform_touchdown_lift_values: list[float] = []
    episode_records: list[dict] = []
    failure_steps: list[int] = []
    commanded_forward_sum = 0.0
    actual_forward_sum = 0.0
    velocity_sample_count = 0
    scan_sum = scan_square_sum = scan_count = scan_invalid_count = scan_nonfinite_count = 0.0
    scan_min = float("inf")
    scan_max = float("-inf")
    counterfactual_normal_zero_action_delta_sum = 0.0
    counterfactual_normal_permuted_action_delta_sum = 0.0
    counterfactual_action_samples = 0
    action_delta_sum = 0.0
    action_delta_samples = 0
    diagnostic_joint_names = [
        env.robot.joint_names[index]
        for index in (env.left_leg_ids[1], env.left_leg_ids[3], env.right_leg_ids[1], env.right_leg_ids[3])
    ]
    diagnostic_joint_indices = [env.policy_joint_names.index(name) for name in diagnostic_joint_names]
    joint_action_abs_sum = dict.fromkeys(diagnostic_joint_names, 0.0)
    joint_action_abs_max = dict.fromkeys(diagnostic_joint_names, 0.0)
    joint_action_samples = 0
    scan_invalid_value = TEACHER_SCAN_INVALID_VALUE * float(env.obs_scales.height_scan)
    scan_invalid_tensor = torch.tensor(scan_invalid_value, device=env.device)
    quota_base, quota_remainder = divmod(args_cli.episodes, env.num_envs)
    episode_quota = torch.full((env.num_envs,), quota_base, dtype=torch.long, device=env.device)
    if quota_remainder:
        episode_quota[:quota_remainder] += 1
    recorded_episode_counts = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    while completed < args_cli.episodes:
        active_envs = recorded_episode_counts < episode_quota
        active_count = int(active_envs.sum().item())
        if active_count == 0:
            break
        raw_scan = obs[:, scan_start:scan_end]
        active_scan = raw_scan[active_envs]
        finite_scan = active_scan[torch.isfinite(active_scan)]
        scan_nonfinite_count += float((~torch.isfinite(active_scan)).sum().item())
        if finite_scan.numel() > 0:
            scan_sum += float(finite_scan.sum().item())
            scan_square_sum += float(torch.square(finite_scan).sum().item())
            scan_count += float(finite_scan.numel())
            scan_min = min(scan_min, float(finite_scan.min().item()))
            scan_max = max(scan_max, float(finite_scan.max().item()))
            scan_invalid_count += float(torch.isclose(finite_scan, scan_invalid_tensor).sum().item())

        current_forward_xy = forward_xy(env.robot.data.root_quat_w)
        actual_forward = torch.sum(env.robot.data.root_lin_vel_w[:, :2] * current_forward_xy, dim=-1)
        commanded_forward_sum += float(env.command_generator.command[active_envs, 0].sum().item())
        actual_forward_sum += float(actual_forward[active_envs].sum().item())
        velocity_sample_count += active_count

        observations_by_mode = {mode: apply_scan_mode(obs, mode) for mode in ("normal", "zero", "permuted")}
        with torch.inference_mode():
            deterministic_actions = evaluate_counterfactual_actions(
                runner.alg.policy,
                deterministic_policy,
                observations_by_mode,
                selected_mode=args_cli.scan_mode,
            )
            if args_cli.zero_actions:
                actions = torch.zeros_like(deterministic_actions[args_cli.scan_mode])
            elif args_cli.stochastic:
                if getattr(runner.alg.policy, "is_recurrent", False):
                    mean = deterministic_actions[args_cli.scan_mode]
                    std = runner.alg.policy._bounded_std(mean)
                    actions = torch.distributions.Normal(mean, std).sample()
                else:
                    actions = stochastic_policy(observations_by_mode[args_cli.scan_mode])
            else:
                actions = deterministic_actions[args_cli.scan_mode]

        normal_zero_delta = torch.mean(
            torch.abs(deterministic_actions["normal"] - deterministic_actions["zero"]),
            dim=-1,
        )
        normal_permuted_delta = torch.mean(
            torch.abs(deterministic_actions["normal"] - deterministic_actions["permuted"]),
            dim=-1,
        )
        counterfactual_normal_zero_action_delta_sum += float(normal_zero_delta[active_envs].sum().item())
        counterfactual_normal_permuted_action_delta_sum += float(normal_permuted_delta[active_envs].sum().item())
        counterfactual_action_samples += active_count

        action_delta_mask = have_prev_policy_action & active_envs
        if torch.any(action_delta_mask):
            delta = torch.mean(torch.abs(actions - prev_policy_actions), dim=-1)
            action_delta_sum += float(delta[action_delta_mask].sum().item())
            action_delta_samples += int(action_delta_mask.sum().item())
        prev_policy_actions[active_envs] = actions[active_envs]
        have_prev_policy_action[active_envs] = True

        diagnostic_actions = torch.abs(actions[active_envs][:, diagnostic_joint_indices])
        for joint_i, joint_name in enumerate(diagnostic_joint_names):
            joint_action_abs_sum[joint_name] += float(diagnostic_actions[:, joint_i].sum().item())
            joint_action_abs_max[joint_name] = max(
                joint_action_abs_max[joint_name],
                float(diagnostic_actions[:, joint_i].max().item()),
            )
        joint_action_samples += active_count

        obs, _, dones, extras = env.step(actions)
        reset_recurrent_policy(runner.alg.policy, dones)
        episode_steps[active_envs] += 1

        done_tensor_ids = torch.nonzero(dones, as_tuple=False).flatten()
        transition_root_pos = env.robot.data.root_pos_w.clone()
        transition_root_quat = env.robot.data.root_quat_w.clone()
        transition_feet_pos = env.robot.data.body_pos_w[:, env.feet_body_ids, :].clone()
        transition_feet_contact = current_feet_contact()
        if len(done_tensor_ids) > 0:
            transition_root_pos[done_tensor_ids] = env.terminal_root_pos_w[done_tensor_ids]
            transition_root_quat[done_tensor_ids] = env.terminal_root_quat_w[done_tensor_ids]
            transition_feet_pos[done_tensor_ids] = env.terminal_feet_pos_w[done_tensor_ids]
            transition_feet_contact[done_tensor_ids] = env.terminal_feet_contact[done_tensor_ids]

        delta_xy = transition_root_pos[:, :2] - start_pos[:, :2]
        forward_progress = torch.sum(delta_xy * start_forward_xy, dim=-1)
        episode_max_forward_progress[active_envs] = torch.maximum(
            episode_max_forward_progress[active_envs], forward_progress[active_envs]
        )

        foot_lift = transition_feet_pos[:, :, 2] - initial_feet_z
        swing_tracking = active_envs.unsqueeze(1) & (~transition_feet_contact)
        swing_peak_lift = torch.where(
            swing_tracking,
            torch.maximum(swing_peak_lift, foot_lift),
            swing_peak_lift,
        )
        candidate_lift = foot_lift.masked_fill(transition_feet_contact, float("-inf"))
        best_lift, best_foot = torch.max(candidate_lift, dim=-1)
        select_first_swing = active_envs & (first_swing_foot < 0) & (best_lift > 0.015)
        first_swing_foot[select_first_swing] = best_foot[select_first_swing]
        selected_lift = foot_lift.gather(1, first_swing_foot.clamp(min=0).unsqueeze(1)).squeeze(1)
        has_first_swing = active_envs & (first_swing_foot >= 0)
        tracking_first_swing = has_first_swing & (~first_swing_complete)
        first_swing_max_lift = torch.where(
            tracking_first_swing,
            torch.maximum(first_swing_max_lift, selected_lift),
            first_swing_max_lift,
        )
        selected_contact = transition_feet_contact.gather(1, first_swing_foot.clamp(min=0).unsqueeze(1)).squeeze(1)
        first_swing_complete |= tracking_first_swing & selected_contact

        touchdown = transition_feet_contact & (~prev_feet_contact)
        local_feet_xy = transition_feet_pos[:, :, :2] - env.scene.env_origins[:, None, :2]
        off_platform = torch.any(torch.abs(local_feet_xy) > 0.5 * T4_STONE_PLATFORM_WIDTH, dim=-1)
        first_contact_candidates = (
            touchdown
            & off_platform
            & active_envs.unsqueeze(1)
            & (~first_off_platform_contact_seen).unsqueeze(1)
        )
        first_contact_envs = torch.any(first_contact_candidates, dim=-1)
        if torch.any(first_contact_envs):
            illegal_fractions = env.algebraic_foot_illegal_fractions(
                foot_pos_w=transition_feet_pos,
                root_quat_w=transition_root_quat,
            )
            candidate_illegal = illegal_fractions.masked_fill(~first_contact_candidates, float("inf"))
            first_illegal, first_contact_foot = torch.min(candidate_illegal, dim=-1)
            first_off_platform_contact_seen[first_contact_envs] = True
            first_off_platform_contact_illegal[first_contact_envs] = first_illegal[first_contact_envs]
            first_contact_swing_peak = swing_peak_lift.gather(1, first_contact_foot.unsqueeze(1)).squeeze(1)
            first_contact_touchdown_lift = foot_lift.gather(1, first_contact_foot.unsqueeze(1)).squeeze(1)
            first_off_platform_swing_peak_lift[first_contact_envs] = first_contact_swing_peak[first_contact_envs]
            first_off_platform_touchdown_lift[first_contact_envs] = first_contact_touchdown_lift[first_contact_envs]
        swing_peak_lift[touchdown] = 0.0

        done_ids = done_tensor_ids.tolist()
        for env_id in done_ids:
            if not bool(active_envs[env_id].item()):
                continue
            steps = int(episode_steps[env_id].item())
            terminal_delta_xy = env.terminal_root_pos_w[env_id, :2] - start_pos[env_id, :2]
            final_forward_progress = float(torch.dot(terminal_delta_xy, start_forward_xy[env_id]).item())
            progress = float(episode_max_forward_progress[env_id].item())
            monitor_radial_progress = float(env.terminal_episode_max_radial_dist[env_id].item())
            timed_out = bool(extras.get("time_outs", torch.zeros_like(dones))[env_id].item())
            progress_values.append(progress)
            monitor_radial_progress_values.append(monitor_radial_progress)
            final_forward_progress_values.append(final_forward_progress)
            if timed_out:
                timeout_episodes += 1
            pit_fall_buf = getattr(env, "pit_fall_buf", None)
            pit_fall = pit_fall_buf is not None and bool(pit_fall_buf[env_id].item())
            if pit_fall:
                pit_fall_episodes += 1

            reason_flags: dict[str, bool] = {}
            reason_masks = getattr(env, "reset_reason_masks", None) or {}
            for name, mask in reason_masks.items():
                flag = bool(mask[env_id].item())
                reason_flags[name] = flag
                reset_reason_counts[name] = reset_reason_counts.get(name, 0) + int(flag)
            if reason_flags:
                behavioral_failure = (
                    reason_flags.get("joint_vel", False)
                    or reason_flags.get("torso", False)
                    or reason_flags.get("accel", False)
                    or reason_flags.get("fall_over", False)
                    or pit_fall
                )
                clean_completion = (
                    reason_flags.get("horizon", False) or reason_flags.get("oob", False)
                ) and not behavioral_failure
            else:
                clean_completion = timed_out and not pit_fall
            if clean_completion:
                strict_completion_episodes += 1
            else:
                fall_or_early_termination_episodes += 1

            first_swing_lift = float(first_swing_max_lift[env_id].item())
            first_contact_seen = bool(first_off_platform_contact_seen[env_id].item())
            first_contact_illegal = (
                float(first_off_platform_contact_illegal[env_id].item()) if first_contact_seen else None
            )
            first_contact_swing_peak = (
                float(first_off_platform_swing_peak_lift[env_id].item()) if first_contact_seen else None
            )
            first_contact_touchdown_lift = (
                float(first_off_platform_touchdown_lift[env_id].item()) if first_contact_seen else None
            )
            first_swing_lift_values.append(first_swing_lift)
            if first_contact_illegal is not None:
                first_off_platform_illegal_values.append(first_contact_illegal)
                first_off_platform_swing_peak_lift_values.append(first_contact_swing_peak)
                first_off_platform_touchdown_lift_values.append(first_contact_touchdown_lift)

            if clean_completion and progress >= args_cli.progress_m:
                successes += 1
            else:
                failures += 1
                failure_steps.append(steps)
            episode_records.append(
                {
                    "env_id": env_id,
                    "env_episode_index": int(recorded_episode_counts[env_id].item()),
                    "steps": steps,
                    "progress_m": progress,
                    "monitor_radial_progress_m": monitor_radial_progress,
                    "final_forward_progress_m": final_forward_progress,
                    "clean_completion": clean_completion,
                    "pit_fall": pit_fall,
                    "reset_reasons": [name for name, flag in reason_flags.items() if flag],
                    "first_swing_max_lift_m": first_swing_lift,
                    "first_off_platform_contact_seen": first_contact_seen,
                    "first_off_platform_contact_illegal_fraction": first_contact_illegal,
                    "first_off_platform_swing_peak_lift_m": first_contact_swing_peak,
                    "first_off_platform_touchdown_lift_m": first_contact_touchdown_lift,
                }
            )
            recorded_episode_counts[env_id] += 1
            completed += 1

        prev_feet_contact.copy_(current_feet_contact())
        if len(done_tensor_ids) > 0:
            episode_steps[done_tensor_ids] = 0
            start_pos[done_tensor_ids] = env.robot.data.root_pos_w[done_tensor_ids]
            start_forward_xy[done_tensor_ids] = forward_xy(env.robot.data.root_quat_w[done_tensor_ids])
            episode_max_forward_progress[done_tensor_ids] = 0.0
            initial_feet_z[done_tensor_ids] = env.robot.data.body_pos_w[done_tensor_ids][:, env.feet_body_ids, 2]
            first_swing_foot[done_tensor_ids] = -1
            first_swing_max_lift[done_tensor_ids] = 0.0
            first_swing_complete[done_tensor_ids] = False
            swing_peak_lift[done_tensor_ids] = 0.0
            first_off_platform_contact_seen[done_tensor_ids] = False
            first_off_platform_contact_illegal[done_tensor_ids] = 0.0
            first_off_platform_swing_peak_lift[done_tensor_ids] = 0.0
            first_off_platform_touchdown_lift[done_tensor_ids] = 0.0
            have_prev_policy_action[done_tensor_ids] = False

    reach_1m_episodes = sum(progress >= 1.0 for progress in progress_values)
    reach_2m_episodes = sum(progress >= 2.0 for progress in progress_values)
    reach_4m_episodes = sum(progress >= 4.0 for progress in progress_values)
    scan_mean = scan_sum / max(1.0, scan_count)
    scan_variance = max(0.0, scan_square_sum / max(1.0, scan_count) - scan_mean * scan_mean)
    first_swing_detected_episodes = sum(value > 0.0 for value in first_swing_lift_values)
    first_off_platform_contact_episodes = len(first_off_platform_illegal_values)
    first_off_platform_legal_episodes = sum(value <= 1.0e-6 for value in first_off_platform_illegal_values)
    targeted_envs = episode_quota > 0
    targeted_quota = episode_quota[targeted_envs]
    targeted_recorded = recorded_episode_counts[targeted_envs]
    result = {
        "evaluator": "t4_terrain_perception_v3",
        "checkpoint": str(root),
        "task": args_cli.task,
        "seed": int(agent_cfg.seed),
        "difficulty": args_cli.difficulty,
        "requested_command_vx_mps": args_cli.command_vx,
        "zero_actions": args_cli.zero_actions,
        "spawn_pose_pinned": pinned_spawn is not None,
        "spawn_y_offset_m": None if pinned_spawn is None else pinned_spawn["y_offset_m"],
        "spawn_yaw_deg": None if pinned_spawn is None else pinned_spawn["yaw_deg"],
        "terrain_type": args_cli.terrain_type,
        "soft_sparse_terrain": False,
        "requested_episodes": args_cli.episodes,
        "completed_episodes": completed,
        "episode_sampling": "balanced_per_env_quota",
        "targeted_envs": int(targeted_envs.sum().item()),
        "recorded_unique_envs": int((recorded_episode_counts > 0).sum().item()),
        "episode_quota_min": int(targeted_quota.min().item()) if targeted_quota.numel() else 0,
        "episode_quota_max": int(targeted_quota.max().item()) if targeted_quota.numel() else 0,
        "recorded_episode_count_min": int(targeted_recorded.min().item()) if targeted_recorded.numel() else 0,
        "recorded_episode_count_max": int(targeted_recorded.max().item()) if targeted_recorded.numel() else 0,
        "strict_progress_successes": successes,
        "strict_progress_failures": failures,
        "strict_progress_success_rate": successes / max(1, completed),
        "progress_m": args_cli.progress_m,
        "progress_source": "episode_max_command_direction_displacement",
        "progress_mean_m": sum(progress_values) / max(1, len(progress_values)),
        "progress_min_m": min(progress_values, default=0.0),
        "progress_max_m": max(progress_values, default=0.0),
        "monitor_radial_progress_source": "terminal_episode_max_radial_dist_from_terrain_origin",
        "monitor_radial_progress_mean_m": sum(monitor_radial_progress_values)
        / max(1, len(monitor_radial_progress_values)),
        "monitor_radial_progress_min_m": min(monitor_radial_progress_values, default=0.0),
        "monitor_radial_progress_max_m": max(monitor_radial_progress_values, default=0.0),
        "final_forward_progress_mean_m": sum(final_forward_progress_values)
        / max(1, len(final_forward_progress_values)),
        "final_forward_progress_min_m": min(final_forward_progress_values, default=0.0),
        "final_forward_progress_max_m": max(final_forward_progress_values, default=0.0),
        "reach_1m_episodes": reach_1m_episodes,
        "reach_1m_rate": reach_1m_episodes / max(1, completed),
        "reach_2m_episodes": reach_2m_episodes,
        "reach_2m_rate": reach_2m_episodes / max(1, completed),
        "reach_4m_episodes": reach_4m_episodes,
        "reach_4m_rate": reach_4m_episodes / max(1, completed),
        "timeout_episodes": timeout_episodes,
        "strict_completion_episodes": strict_completion_episodes,
        "fall_or_early_termination_episodes": fall_or_early_termination_episodes,
        "pit_fall_episodes": pit_fall_episodes,
        "reset_reason_counts": reset_reason_counts,
        "failure_step_mean": sum(failure_steps) / max(1, len(failure_steps)),
        "stochastic_actions": args_cli.stochastic,
        "commanded_forward_mean_mps": commanded_forward_sum / max(1, velocity_sample_count),
        "actual_forward_mean_mps": actual_forward_sum / max(1, velocity_sample_count),
        "student_depth_noise": bool(getattr(env_cfg, "student_depth_noise", False)),
        "disable_student_depth_noise": bool(args_cli.disable_student_depth_noise),
        "scan_mode": args_cli.scan_mode,
        "scan_permutation_seed": args_cli.scan_permutation_seed,
        "scan_history_length": scan_history_length,
        "scan_slice": [scan_start, scan_end],
        "raw_scan_min": scan_min if scan_count else 0.0,
        "raw_scan_max": scan_max if scan_count else 0.0,
        "raw_scan_mean": scan_mean,
        "raw_scan_std": scan_variance**0.5,
        "raw_scan_at_invalid_sentinel_fraction": scan_invalid_count / max(1.0, scan_count),
        "raw_scan_nonfinite_fraction": scan_nonfinite_count / max(1.0, scan_count + scan_nonfinite_count),
        "counterfactual_normal_zero_action_l1_mean": counterfactual_normal_zero_action_delta_sum
        / max(1, counterfactual_action_samples),
        "counterfactual_normal_permuted_action_l1_mean": counterfactual_normal_permuted_action_delta_sum
        / max(1, counterfactual_action_samples),
        "action_delta_l1_mean": action_delta_sum / max(1, action_delta_samples),
        "diagnostic_joint_action_abs_mean": {
            name: value / max(1, joint_action_samples) for name, value in joint_action_abs_sum.items()
        },
        "diagnostic_joint_action_abs_max": joint_action_abs_max,
        "first_swing_detected_episodes": first_swing_detected_episodes,
        "first_swing_foot_max_lift_mean_m": sum(first_swing_lift_values) / max(1, len(first_swing_lift_values)),
        "first_swing_foot_max_lift_min_m": min(first_swing_lift_values, default=0.0),
        "first_swing_foot_max_lift_max_m": max(first_swing_lift_values, default=0.0),
        "first_off_platform_contact_episodes": first_off_platform_contact_episodes,
        "first_off_platform_contact_legal_episodes": first_off_platform_legal_episodes,
        "first_off_platform_contact_legal_rate": first_off_platform_legal_episodes
        / max(1, first_off_platform_contact_episodes),
        "first_off_platform_contact_illegal_fraction_mean": sum(first_off_platform_illegal_values)
        / max(1, first_off_platform_contact_episodes),
        "first_off_platform_swing_peak_lift_mean_m": sum(first_off_platform_swing_peak_lift_values)
        / max(1, first_off_platform_contact_episodes),
        "first_off_platform_touchdown_lift_mean_m": sum(first_off_platform_touchdown_lift_values)
        / max(1, first_off_platform_contact_episodes),
        "episode_records": episode_records,
        "exact_foothold_gate": False,
        "final_zero_contact_gate": False,
        "caveat": (
            "This diagnostic separates command-direction progress, training-monitor radial distance, "
            "terminal causes, scan interventions, and first-foot behavior. It does not prove exact full-sole "
            "placement, zero bar contact, or ordered corridor passage. Sparse tiles use real holes."
        ),
    }
    Path(args_cli.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args_cli.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    exit_code = 0
    try:
        evaluate()
    except BaseException:
        import traceback

        traceback.print_exc()
        exit_code = 1
    finally:
        import threading

        threading.Timer(90.0, os._exit, args=(exit_code,)).start()
        simulation_app.close()
        os._exit(exit_code)

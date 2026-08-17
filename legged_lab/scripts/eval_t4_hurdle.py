"""Short fixed-command diagnostic evaluator for Stage E terrain buckets.

This version is intentionally a progress gate, not the final zero-contact
or exact-footstep evaluator. It fixes a forward command, disables training
randomization, and requires an uninterrupted timeout plus forward displacement.
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
patch_physx_backward_compatibility_setting(AppLauncher)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner  # noqa: E402

from legged_lab.envs import *  # noqa: F401,F403,E402
from legged_lab.utils import task_registry  # noqa: E402
from legged_lab.utils.cli_args import update_rsl_rl_cfg  # noqa: E402

patch_missing_physx_material_attributes()


def evaluate() -> dict:
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.terrain_generator.curriculum = False
    env_cfg.scene.terrain_generator.difficulty_range = (args_cli.difficulty, args_cli.difficulty)
    sub_terrains = env_cfg.scene.terrain_generator.sub_terrains
    if args_cli.terrain_type not in sub_terrains:
        raise ValueError(f"{args_cli.terrain_type} terrain is unavailable: {sorted(sub_terrains)}")
    terrain_cfg = sub_terrains[args_cli.terrain_type]
    terrain_cfg.proportion = 1.0
    env_cfg.scene.terrain_generator.sub_terrains = {args_cli.terrain_type: terrain_cfg}
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.rel_heading_envs = 0.0
    env_cfg.commands.heading_command = False
    env_cfg.commands.resampling_time_range = (20.0, 20.0)
    env_cfg.commands.ranges.lin_vel_x = (0.7, 0.7)
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.ranges.heading = (0.0, 0.0)
    env_cfg.noise.add_noise = False
    if not args_cli.keep_randomization:
        # Keep the seeded reset pose/joint perturbations. Removing both leaves a
        # perfectly symmetric nominal start where deterministic locomotion
        # policies can remain at a stationary fixed point, including known-good
        # Stage E checkpoints. Mass/material randomization and pushes are not
        # part of this fixed-command behavior check.
        for name in ("physics_material", "add_base_mass", "push_robot"):
            setattr(env_cfg.domain_rand.events, name, None)
    env_cfg.scene.seed = agent_cfg.seed
    env_cfg.scene.terrain_generator.num_rows = 1
    env_cfg.scene.terrain_generator.num_cols = args_cli.num_envs

    env_class = task_registry.get_task_class(args_cli.task)
    env = env_class(env_cfg, headless=True)
    root = Path(args_cli.checkpoint)
    if not root.is_file():
        root = Path(get_checkpoint_path(os.path.abspath(os.path.join("logs", agent_cfg.experiment_name)), args_cli.load_run, args_cli.checkpoint))
    runner_class = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=str(root.parent), device=env.device)
    runner.load(str(root), load_optimizer=False)
    deterministic_policy = runner.get_inference_policy(device=env.device)
    if args_cli.stochastic:

        def policy(policy_obs):
            if runner.cfg["empirical_normalization"]:
                policy_obs = runner.obs_normalizer(policy_obs)
            return runner.alg.policy.act(policy_obs)

    else:
        policy = deterministic_policy

    obs, _ = env.get_observations()
    max_steps = int(round(env.max_episode_length))
    episode_steps = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)
    start_pos = env.robot.data.root_pos_w.clone()
    start_quat = env.robot.data.root_quat_w.clone()

    def forward_xy(quat):
        w, x, y, z = quat.unbind(-1)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)

    start_forward_xy = forward_xy(start_quat)
    successes = failures = completed = 0
    timeout_episodes = fall_or_early_termination_episodes = pit_fall_episodes = 0
    progress_values: list[float] = []
    failure_steps: list[int] = []
    commanded_forward_sum = 0.0
    actual_forward_sum = 0.0
    velocity_sample_count = 0
    while completed < args_cli.episodes:
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, dones, extras = env.step(actions)
        current_forward_xy = forward_xy(env.robot.data.root_quat_w)
        actual_forward = torch.sum(env.robot.data.root_lin_vel_w[:, :2] * current_forward_xy, dim=-1)
        commanded_forward_sum += float(env.command_generator.command[:, 0].sum().item())
        actual_forward_sum += float(actual_forward.sum().item())
        velocity_sample_count += env.num_envs
        episode_steps += 1
        done_ids = torch.nonzero(dones, as_tuple=False).flatten().tolist()
        for env_id in done_ids:
            steps = int(episode_steps[env_id].item())
            delta_xy = env.last_step_root_pos_w[env_id, :2] - start_pos[env_id, :2]
            progress = float(torch.dot(delta_xy, start_forward_xy[env_id]).item())
            timed_out = bool(extras.get("time_outs", torch.zeros_like(dones))[env_id].item())
            progress_values.append(progress)
            if timed_out:
                timeout_episodes += 1
            else:
                fall_or_early_termination_episodes += 1
            pit_fall_buf = getattr(env, "pit_fall_buf", None)
            if pit_fall_buf is not None and bool(pit_fall_buf[env_id].item()):
                pit_fall_episodes += 1
            if timed_out and progress >= args_cli.progress_m:
                successes += 1
            else:
                failures += 1
                failure_steps.append(steps)
            completed += 1
            episode_steps[env_id] = 0
            start_pos[env_id] = env.robot.data.root_pos_w[env_id]
            start_forward_xy[env_id] = forward_xy(env.robot.data.root_quat_w[env_id : env_id + 1])[0]
            if completed >= args_cli.episodes:
                break

    reach_1m_episodes = sum(progress >= 1.0 for progress in progress_values)
    reach_2m_episodes = sum(progress >= 2.0 for progress in progress_values)
    reach_4m_episodes = sum(progress >= 4.0 for progress in progress_values)
    result = {
        "evaluator": "t4_stage_e_terrain_progress_v2",
        "checkpoint": str(root),
        "task": args_cli.task,
        "seed": int(agent_cfg.seed),
        "difficulty": args_cli.difficulty,
        "terrain_type": args_cli.terrain_type,
        "requested_episodes": args_cli.episodes,
        "completed_episodes": completed,
        "strict_progress_successes": successes,
        "strict_progress_failures": failures,
        "strict_progress_success_rate": successes / max(1, completed),
        "progress_m": args_cli.progress_m,
        "progress_mean_m": sum(progress_values) / max(1, len(progress_values)),
        "progress_min_m": min(progress_values, default=0.0),
        "progress_max_m": max(progress_values, default=0.0),
        "reach_1m_episodes": reach_1m_episodes,
        "reach_1m_rate": reach_1m_episodes / max(1, completed),
        "reach_2m_episodes": reach_2m_episodes,
        "reach_2m_rate": reach_2m_episodes / max(1, completed),
        "reach_4m_episodes": reach_4m_episodes,
        "reach_4m_rate": reach_4m_episodes / max(1, completed),
        "timeout_episodes": timeout_episodes,
        "fall_or_early_termination_episodes": fall_or_early_termination_episodes,
        "pit_fall_episodes": pit_fall_episodes,
        "failure_step_mean": sum(failure_steps) / max(1, len(failure_steps)),
        "stochastic_actions": args_cli.stochastic,
        "commanded_forward_mean_mps": commanded_forward_sum / max(1, velocity_sample_count),
        "actual_forward_mean_mps": actual_forward_sum / max(1, velocity_sample_count),
        "exact_foothold_gate": False,
        "final_zero_contact_gate": False,
        "caveat": (
            "This progress diagnostic does not prove exact foothold placement, zero bar contact, "
            "or ordered corridor passage."
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

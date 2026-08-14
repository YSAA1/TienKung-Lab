# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

import argparse
import os

import torch
from isaaclab.app import AppLauncher

from legged_lab.scripts.isaaclab_runtime_compat import (
    patch_missing_physx_material_attributes,
    patch_physx_backward_compatibility_setting,
)
from legged_lab.utils import task_registry
from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--terrain",
    action="store_true",
    help="Keep the task terrain generator instead of flattening to a plane.",
)
parser.add_argument(
    "--difficulty",
    type=float,
    default=0.85,
    help="Fixed terrain difficulty in [0, 1] when --terrain is set.",
)
parser.add_argument("--record", type=str, default=None, help="Write an MP4 and exit instead of looping the GUI.")
parser.add_argument("--duration", type=float, default=12.0, help="Recorded seconds when --record is set.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
patch_physx_backward_compatibility_setting(AppLauncher)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# Start camera rendering
if "sensor" in args_cli.task:
    args_cli.enable_cameras = True
if args_cli.record:
    args_cli.enable_cameras = True
    args_cli.headless = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg

patch_missing_physx_material_attributes()


def play():
    runner: OnPolicyRunner
    env_cfg: BaseEnvCfg  # noqa:F405

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.scene.max_episode_length_s = 40.0
    env_cfg.scene.num_envs = 50
    env_cfg.scene.env_spacing = 2.5
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.ranges.lin_vel_x = (0.6, 0.6)
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)

    if args_cli.terrain:
        env_cfg.scene.terrain_generator.curriculum = False
        env_cfg.scene.terrain_generator.difficulty_range = (args_cli.difficulty, args_cli.difficulty)
    else:
        env_cfg.scene.terrain_generator = None
        env_cfg.scene.terrain_type = "plane"

    if env_cfg.scene.terrain_generator is not None:
        env_cfg.scene.terrain_generator.num_rows = 5
        env_cfg.scene.terrain_generator.num_cols = 5
        env_cfg.scene.terrain_generator.curriculum = False

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)

    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    runner_class: OnPolicyRunner | AmpOnPolicyRunner = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)

    policy = runner.get_inference_policy(device=env.device)

    if not args_cli.record:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(runner.alg.policy, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(
            runner.alg.policy, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
        )

    if not args_cli.headless:
        from legged_lab.utils.keyboard import Keyboard

        keyboard = Keyboard(env)  # noqa:F841

    obs, _ = env.get_observations()

    if args_cli.record:
        _record_play_video(env, policy, obs, args_cli.record, args_cli.duration)
        return

    while simulation_app.is_running():

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)


def _record_play_video(env, policy, obs, output_path: str, duration_s: float) -> None:
    import imageio.v2 as imageio
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import Camera, CameraCfg

    camera = Camera(
        CameraCfg(
            prim_path="/World/play_cam",
            update_period=0.0,
            height=480,
            width=960,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.08, 40.0),
            ),
        )
    )
    # Play.py creates the camera after the timeline is already playing, so the
    # SensorBase PLAY callback never fires. Initialize it explicitly.
    camera._initialize_callback(None)
    camera.reset()

    n_steps = max(1, int(round(duration_s / env.step_dt)))
    frames: list = []
    look_offset = torch.tensor([0.0, 0.0, 0.45], device=env.device)
    eye_offset = torch.tensor([-2.8, -2.2, 1.6], device=env.device)

    for _ in range(n_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        root = env.robot.data.root_pos_w[0]
        camera.set_world_poses_from_view(
            eyes=(root + eye_offset).unsqueeze(0),
            targets=(root + look_offset).unsqueeze(0),
        )
        env.sim.render()
        camera.update(dt=env.step_dt)
        rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy()
        if rgb.size:
            frames.append(rgb.copy())

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    imageio.mimwrite(output_path, frames, fps=max(1, int(round(1.0 / env.step_dt))), quality=8, macro_block_size=1)
    print(f"[INFO] wrote {output_path} ({len(frames)} frames)")


if __name__ == "__main__":
    play()
    if not args_cli.record:
        simulation_app.close()

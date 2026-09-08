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
import hashlib
import json
from pathlib import Path

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
    "--g1_progress_ab", choices=("A", "B"), help="Cold-start full-speed G1: A legacy path, B radial traversal"
)
parser.add_argument(
    "--amp_expert_manifest", type=Path, help="Use the exact AMP files and hashes in this dataset manifest"
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
patch_physx_backward_compatibility_setting(AppLauncher)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# Start camera rendering
if "sensor" in args_cli.task:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
import os
from datetime import datetime

import torch
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg

patch_missing_physx_material_attributes()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def train():
    runner: OnPolicyRunner | AmpOnPolicyRunner

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)
    env_class = task_registry.get_task_class(env_class_name)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.g1_progress_ab is not None:
        if env_class_name != "g1_loco_teacher" or agent_cfg.resume or args_cli.amp_expert_manifest is None:
            raise ValueError("G1 A/B requires g1_loco_teacher, cold start and an explicit full17 AMP manifest")
        env_cfg.sparse_command_min_speed_scale = 1.0
        env_cfg.lightlp_promotion_distance = "path_length" if args_cli.g1_progress_ab == "A" else "max_radial"
        env_cfg.progress_monitor_enabled = True
    if args_cli.amp_expert_manifest is not None:
        manifest_path = args_cli.amp_expert_manifest.resolve()
        manifest = json.loads(manifest_path.read_text())
        if not manifest["clips"]:
            raise ValueError("AMP manifest has no clips")
        files = []
        for name, clip in manifest["clips"].items():
            path = manifest_path.parent / f"{name}.txt"
            if hashlib.sha256(path.read_bytes()).hexdigest() != clip["sha256"]:
                raise ValueError(f"AMP clip hash mismatch: {path}")
            files.append(str(path))
        agent_cfg.amp_expert_dir = str(manifest_path.parent)
        agent_cfg.amp_motion_files = files
    env_cfg.scene.seed = agent_cfg.seed

    if args_cli.distributed:
        # The env reads `env_cfg.device`; setting only `sim.device` leaves every rank
        # building its buffers on cuda:0 and cross-device ops then fail.
        env_cfg.device = f"cuda:{app_launcher.local_rank}"
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.scene.seed = seed
        agent_cfg.seed = seed

    env = env_class(env_cfg, args_cli.headless)

    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    runner_class: OnPolicyRunner | AmpOnPolicyRunner = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        if args_cli.reset_optimizer:
            print("[INFO]: Loading model state with a fresh optimizer.")
        runner.load(resume_path, load_optimizer=not args_cli.reset_optimizer)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    exit_code = 0
    try:
        train()
    except BaseException:
        import traceback

        traceback.print_exc()
        exit_code = 1
    finally:
        import threading

        # Isaac shutdown can hang after the final checkpoint. Bound teardown so
        # the tmux supervisor can release GPUs and run the behavior evaluations.
        threading.Timer(30.0, os._exit, args=(exit_code,)).start()
        simulation_app.close()
        os._exit(exit_code)

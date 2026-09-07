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
    assert agent_cfg.max_iterations == 30000
    env_class = task_registry.get_task_class(env_class_name)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
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

    log_root_path = "artifacts/portability/v5/ddp_probe"
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

    import json
    from pathlib import Path
    import torch.distributed as dist
    def rank_difference(values):
        data = values.detach().to(device=env.device, dtype=torch.float64)
        lo, hi = data.clone(), data.clone()
        dist.all_reduce(lo, op=dist.ReduceOp.MIN)
        dist.all_reduce(hi, op=dist.ReduceOp.MAX)
        return float((hi - lo).abs().max())
    def vector(module):
        return torch.cat([p.detach().reshape(-1) for p in module.parameters()])
    norm = runner.alg.amp_normalizer
    norm_values = torch.cat((torch.tensor(norm.mean), torch.tensor(norm.var), torch.tensor([norm.count])))
    differences = {
        "policy": rank_difference(vector(runner.alg.policy)),
        "discriminator": rank_difference(vector(runner.alg.discriminator)),
        "amp_normalizer": rank_difference(norm_values),
    }
    assert max(differences.values()) < 1e-7, differences
    assert env.action_scale == 0.25
    assert env.cfg.robot.action_scale_effort_fraction is None
    assert "reference_scale" not in env.cfg.reward.action_rate_l2.params
    assert env.cfg.random_level_reset_fraction == 0.1
    assert env.cfg.random_level_reset_min_level is None and env.cfg.random_level_reset_max_level is None
    assert env.cfg.amp_terrain_schedule.decay_start_difficulty == 0.3
    assert env.cfg.amp_terrain_schedule.min_scale == 0.3
    obs, extra = env.get_observations()
    assert torch.isfinite(obs).all()
    assert env.num_actions == 29 and env.robot.num_joints == 29
    assert torch.isfinite(env.robot.data.joint_pos).all() and torch.isfinite(env.robot.data.applied_torque).all()
    result = {"rank": dist.get_rank(), "world_size": dist.get_world_size(), "device": str(env.device),
              "action_scale": env.action_scale, "rank_differences_after_training": differences,
              "amp_count": norm.count, "obs_shape": list(obs.shape),
              "formal_budget": 30000, "probe_iterations": agent_cfg.max_iterations,
              "resume": agent_cfg.resume, "learning_capability_proven": False}
    result["actuators"] = {name: {"class": type(motor).__name__, "stiffness": motor.stiffness[0].tolist(), "damping": motor.damping[0].tolist(), "effort_limit": motor.effort_limit[0].tolist()} for name, motor in env.robot.actuators.items()}
    result["num_sim_joints"] = env.robot.num_joints
    result["asset_path"] = env.cfg.scene.robot.spawn.asset_path
    Path("artifacts/portability/v5/runtime_rank%d.json" % dist.get_rank()).write_text(json.dumps(result, indent=2)+"\n")
    print("AMP_RUNTIME_PROBE", result, flush=True)



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

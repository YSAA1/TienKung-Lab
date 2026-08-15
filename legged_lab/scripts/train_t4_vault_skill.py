"""Distill the G1 vault mimic teacher into the 1155D G2 heightscan skill policy."""

import argparse

from isaaclab.app import AppLauncher

from legged_lab.scripts.isaaclab_runtime_compat import (
    patch_missing_physx_material_attributes,
    patch_physx_backward_compatibility_setting,
)

import legged_lab.utils.cli_args as cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Distill T4 G1 vault mimic into the G2 heightscan skill.")
parser.add_argument("--task", type=str, default="t4_vault_skill")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument(
    "--teacher_checkpoint",
    type=str,
    required=True,
    help="G1 mimic checkpoint whose actor weights become the frozen teacher.",
)
cli_args.add_rsl_rl_args(parser)
patch_physx_backward_compatibility_setting(AppLauncher)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402

import legged_lab.envs.t4.vault_skill  # noqa: F401, E402
from legged_lab.envs.t4.vault_mimic.rsl_rl_compat import RslRlVecEnvWrapper  # noqa: E402
from legged_lab.utils.cli_args import update_rsl_rl_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

patch_missing_physx_material_attributes()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def train():
    spec = gym.spec(args_cli.task)
    env_cfg = spec.kwargs["env_cfg_entry_point"]()
    agent_cfg = spec.kwargs["rsl_rl_cfg_entry_point"]()

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)

    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = agent_cfg.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    log_root_path = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    teacher_path = Path(args_cli.teacher_checkpoint)
    if not teacher_path.is_file():
        raise FileNotFoundError(f"G1 teacher checkpoint not found: {teacher_path}")
    print(f"[INFO] Loading G1 teacher from: {teacher_path}")
    runner.load(str(teacher_path), load_optimizer=False)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


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

        threading.Timer(120.0, os._exit, args=(exit_code,)).start()
        simulation_app.close()
        os._exit(exit_code)

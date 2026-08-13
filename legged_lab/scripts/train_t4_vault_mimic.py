"""Train the T4 1 m box-vault mimic teacher (G1) with the vendored RSL-RL.

The task is an IsaacLab manager-based env (``legged_lab.envs.t4.vault_mimic``),
so this entry mirrors ``legged_lab/scripts/train.py`` conventions but goes
through ``gym.make`` plus the legacy RSL-RL wrapper instead of task_registry.
"""

import argparse

from isaaclab.app import AppLauncher

from legged_lab.scripts.isaaclab_runtime_compat import (
    patch_missing_physx_material_attributes,
    patch_physx_backward_compatibility_setting,
)

import legged_lab.utils.cli_args as cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train the T4 vault mimic teacher with RSL-RL.")
parser.add_argument("--task", type=str, default="t4_vault_mimic", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
cli_args.add_rsl_rl_args(parser)
patch_physx_backward_compatibility_setting(AppLauncher)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402
from datetime import datetime  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402

import legged_lab.envs.t4.vault_mimic  # noqa: F401, E402  (gym task registration)
from legged_lab.envs.t4.vault_mimic.rsl_rl_compat import (  # noqa: E402
    RslRlVecEnvWrapper,
)
from legged_lab.utils.cli_args import update_rsl_rl_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

patch_missing_physx_material_attributes()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


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
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    train()
    simulation_app.close()

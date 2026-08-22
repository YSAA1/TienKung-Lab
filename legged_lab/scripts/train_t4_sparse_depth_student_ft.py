"""Short PPO FT of the S12 GRU depth student under LightLP camera noise."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
from isaaclab.app import AppLauncher

from legged_lab.scripts.isaaclab_runtime_compat import (
    patch_missing_physx_material_attributes,
    patch_physx_backward_compatibility_setting,
)
from legged_lab.utils.cli_args import add_rsl_rl_args, update_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner

parser = argparse.ArgumentParser(description="Fine-tune the S12 GRU depth student with depth noise.")
parser.add_argument("--student_checkpoint", type=str, required=True)
parser.add_argument("--teacher_checkpoint", type=str, default="")
parser.add_argument("--task_num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=None)
add_rsl_rl_args(parser)
patch_physx_backward_compatibility_setting(AppLauncher)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.utils.io import dump_yaml  # noqa: E402
from legged_lab.envs.t4.depth_student_cfg import T4SparseDepthStudentFtAgentCfg  # noqa: E402
from legged_lab.envs.t4.depth_student_env import (  # noqa: E402
    T4LocoSparseDepthDistillEnv,
    T4LocoSparseDepthStudentFtEnvCfg,
)

patch_missing_physx_material_attributes()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def train():
    env_cfg = T4LocoSparseDepthStudentFtEnvCfg()
    agent_cfg = T4SparseDepthStudentFtAgentCfg()
    env_cfg.scene.num_envs = args_cli.task_num_envs
    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed
    if args_cli.distributed:
        env_cfg.device = f"cuda:{app_launcher.local_rank}"
        env_cfg.sim.device = env_cfg.device
        agent_cfg.device = env_cfg.device
        env_cfg.scene.seed = agent_cfg.seed + app_launcher.local_rank
        agent_cfg.seed += app_launcher.local_rank
    else:
        env_cfg.device = agent_cfg.device
        env_cfg.sim.device = agent_cfg.device

    env = T4LocoSparseDepthDistillEnv(env_cfg, args_cli.headless)
    log_root = Path("logs") / agent_cfg.experiment_name
    log_dir = log_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir = Path(f"{log_dir}_{agent_cfg.run_name}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)

    student_path = Path(args_cli.student_checkpoint)
    if not student_path.is_file():
        raise FileNotFoundError(f"student checkpoint not found: {student_path}")
    print(f"[INFO] Loading S12 GRU depth student from: {student_path}")
    runner.load(str(student_path), load_optimizer=False)
    if args_cli.teacher_checkpoint:
        teacher_path = Path(args_cli.teacher_checkpoint)
        if teacher_path.is_file():
            print(f"[INFO] Teacher slot available at {teacher_path}")

    dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
    dump_yaml(str(log_dir / "params" / "agent.yaml"), agent_cfg)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    try:
        train()
    finally:
        simulation_app.close()

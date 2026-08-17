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
parser.add_argument(
    "--student_checkpoint",
    type=str,
    default=None,
    help="Optional G2 student checkpoint to warm-start after the G1 teacher is loaded.",
)
parser.add_argument(
    "--collect_mode",
    type=str,
    choices=("teacher", "student"),
    default=None,
    help="Override Distillation collect_mode. teacher=R3, student=R4 DAgger.",
)
parser.add_argument(
    "--pg_coef",
    type=float,
    default=None,
    help="Student-drive PPO-clip coefficient on tracking-return advantages. 0 keeps pure BC.",
)
parser.add_argument(
    "--teacher_mix",
    type=float,
    default=None,
    help="Fraction of envs that execute the frozen teacher action (DAgger beta).",
)
parser.add_argument(
    "--teacher_mix_end",
    type=float,
    default=None,
    help="Optional end value for a linear teacher_mix anneal.",
)
parser.add_argument(
    "--teacher_mix_decay_iters",
    type=int,
    default=None,
    help="Iterations over which teacher_mix anneals to teacher_mix_end.",
)
parser.add_argument(
    "--student_noise",
    type=float,
    default=None,
    help="Override student action std after checkpoint load.",
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
    if args_cli.collect_mode is not None:
        agent_cfg.algorithm.collect_mode = args_cli.collect_mode
    if args_cli.pg_coef is not None:
        agent_cfg.algorithm.pg_coef = args_cli.pg_coef
    if args_cli.teacher_mix is not None:
        agent_cfg.algorithm.teacher_mix = args_cli.teacher_mix
    if args_cli.teacher_mix_end is not None:
        agent_cfg.algorithm.teacher_mix_end = args_cli.teacher_mix_end
    if args_cli.teacher_mix_decay_iters is not None:
        agent_cfg.algorithm.teacher_mix_decay_iters = args_cli.teacher_mix_decay_iters

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
    print(
        f"[INFO] Distillation collect_mode={agent_cfg.algorithm.collect_mode} "
        f"pg_coef={agent_cfg.algorithm.pg_coef} "
        f"teacher_mix={agent_cfg.algorithm.teacher_mix}->{agent_cfg.algorithm.teacher_mix_end} "
        f"over {agent_cfg.algorithm.teacher_mix_decay_iters}"
    )
    runner.load(str(teacher_path), load_optimizer=False)
    if args_cli.student_checkpoint:
        student_path = Path(args_cli.student_checkpoint)
        if not student_path.is_file():
            raise FileNotFoundError(f"G2 student checkpoint not found: {student_path}")
        print(f"[INFO] Loading G2 student from: {student_path}")
        runner.load(str(student_path), load_optimizer=False)
        runner.current_learning_iteration = 0
    if args_cli.student_noise is not None:
        runner.alg.policy.std.data.fill_(float(args_cli.student_noise))
        print(f"[INFO] Student action std set to {args_cli.student_noise}")

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)
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

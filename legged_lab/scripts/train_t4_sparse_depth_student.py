"""Distill an S12 sparse HeightScan teacher into a GRU depth student.

Do not pass a Stage E 1155D checkpoint or ``stage_s_head35``. The teacher must
be a ``t4_loco_teacher_sparse`` actor (1937D) from the S12 lineage, and the
launch must pass matching ``--teacher_eval_manifest`` JSON (or an explicit
``--allow_ungated_teacher`` waiver). Open this after the teacher fixed-evaluator
gate, not from TB/reward.
"""

from __future__ import annotations

import argparse
import json
import os
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

parser = argparse.ArgumentParser(description="Distill the S12 sparse teacher into a GRU depth student.")
parser.add_argument("--teacher_checkpoint", type=str, required=True)
parser.add_argument(
    "--student_warmstart_checkpoint",
    type=str,
    default="",
    help="Optional pre-collapse GRU student checkpoint. Loads student weights only; resets critic and optimizer.",
)
parser.add_argument(
    "--teacher_eval_manifest",
    action="append",
    default=[],
    help="Fixed evaluator JSON that lists this teacher checkpoint. Repeatable.",
)
parser.add_argument(
    "--allow_ungated_teacher",
    action="store_true",
    help="Waive evaluator-manifest requirement after an explicit operator authorization.",
)
parser.add_argument("--task_num_envs", type=int, default=1024)
parser.add_argument("--seed", type=int, default=None)
add_rsl_rl_args(parser)
patch_physx_backward_compatibility_setting(AppLauncher)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

from legged_lab.assets.t4.student_lineage import (  # noqa: E402
    load_student_warmstart_checkpoint,
    require_sparse_teacher_checkpoint,
)

teacher_gate_info = require_sparse_teacher_checkpoint(
    Path(args_cli.teacher_checkpoint),
    eval_manifests=[Path(item) for item in args_cli.teacher_eval_manifest],
    allow_ungated=bool(args_cli.allow_ungated_teacher),
)

args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.utils.io import dump_yaml  # noqa: E402
from legged_lab.envs.t4.depth_student_cfg import T4SparseDepthStudentAgentCfg  # noqa: E402
from legged_lab.envs.t4.depth_student_env import (  # noqa: E402
    T4LocoSparseDepthDistillEnv,
    T4LocoSparseDepthStudentEnvCfg,
)

patch_missing_physx_material_attributes()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def train():
    env_cfg = T4LocoSparseDepthStudentEnvCfg()
    agent_cfg = T4SparseDepthStudentAgentCfg()
    if args_cli.task_num_envs is not None:
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

    teacher_path = Path(args_cli.teacher_checkpoint)
    if not teacher_path.is_file():
        raise FileNotFoundError(f"teacher checkpoint not found: {teacher_path}")
    print(f"[INFO] Loading frozen S12 sparse teacher from: {teacher_path}")
    runner.load(str(teacher_path), load_optimizer=False)
    warmstart_info = None
    if args_cli.student_warmstart_checkpoint:
        warmstart_path = Path(args_cli.student_warmstart_checkpoint)
        warmstart_info = load_student_warmstart_checkpoint(runner.alg.policy, warmstart_path)
        print(
            "[INFO] Warm-started student control stack "
            f"from {warmstart_path} (iter={warmstart_info['iter']}, keys={warmstart_info['loaded_keys']}); "
            "critic, Adam, and safe-update counters reset."
        )

    if int(os.getenv("RANK", "0")) == 0:
        log_dir.mkdir(parents=True, exist_ok=True)
        lineage = {
            "task": "t4_loco_sparse_depth_student",
            "teacher_checkpoint": str(teacher_path.resolve()),
            "teacher_gate": teacher_gate_info,
            "student_warmstart": warmstart_info,
            "algorithm": agent_cfg.algorithm.class_name,
            "optimizer_reset": True,
            "safe_update_counters_reset": True,
        }
        (log_dir / "student_lineage.json").write_text(
            json.dumps(lineage, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
    dump_yaml(str(log_dir / "params" / "agent.yaml"), agent_cfg)
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
        os._exit(exit_code)

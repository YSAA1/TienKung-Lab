# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""Train the G1 depth-only student by online distillation of a frozen sparse teacher.

Do not pass a non-sparse G1 checkpoint. The teacher must be a ``g1_loco_teacher``
LightLP sparse actor (1997D) from the vital lineage, and the launch must pass a
matching ``--teacher_eval_manifest`` JSON (or an explicit
``--allow_ungated_teacher`` waiver). Open this after the teacher fixed-evaluator
gate, not from TB/reward.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from legged_lab.scripts.isaaclab_runtime_compat import (
    patch_missing_physx_material_attributes,
    patch_physx_backward_compatibility_setting,
)

parser = argparse.ArgumentParser(description="Distill the G1 sparse teacher into a GRU depth student.")
parser.add_argument("--teacher_checkpoint", type=str, required=True)
parser.add_argument(
    "--student_warmstart_checkpoint",
    type=str,
    default="",
    help="Optional student control-stack warm start; resets critic/Adam/std like the T4 lineage gate.",
)
parser.add_argument(
    "--teacher_eval_manifest",
    action="append",
    default=[],
    help="Evaluator JSON matching the frozen teacher. Repeatable.",
)
parser.add_argument(
    "--allow_ungated_teacher",
    action="store_true",
    help="Explicit waiver; only for diagnostics, never for a formal student run.",
)
parser.add_argument("--task_num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=None)
from legged_lab.utils.cli_args import add_rsl_rl_args, update_rsl_rl_cfg  # noqa: E402

add_rsl_rl_args(parser)
patch_physx_backward_compatibility_setting()
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
from datetime import datetime  # noqa: E402

from legged_lab.envs.g1.depth_student_cfg import G1SparseDepthStudentAgentCfg  # noqa: E402
from legged_lab.envs.g1.depth_student_contract import G1_SPARSE_TEACHER_ACTOR_OBS_DIM  # noqa: E402
from legged_lab.envs.g1.depth_student_env import G1LocoSparseDepthStudentReprFirstEnvCfg  # noqa: E402
from legged_lab.locomotion.depth_env import LightLPDepthDistillationEnv  # noqa: E402

patch_missing_physx_material_attributes()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _require_g1_sparse_teacher(path: Path, eval_manifests, allow_ungated: bool) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model_state = blob.get("model_state_dict", blob)
    obs_dim = int(model_state["actor.0.weight"].shape[1])
    if obs_dim != G1_SPARSE_TEACHER_ACTOR_OBS_DIM:
        raise ValueError(
            f"{path} actor obs dim is {obs_dim}, expected {G1_SPARSE_TEACHER_ACTOR_OBS_DIM}; "
            "refusing to distill a non-sparse G1 teacher"
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    info = {"obs_dim": obs_dim, "sha256": digest, "manifests": [], "ungated": bool(allow_ungated)}
    if allow_ungated:
        return info
    manifests = [Path(item) for item in eval_manifests]
    if not manifests:
        raise ValueError("G1 student train requires --teacher_eval_manifest JSON (or --allow_ungated_teacher waiver)")
    teacher_resolved = path.resolve()
    for manifest_path in manifests:
        if not manifest_path.is_file():
            raise ValueError(f"teacher eval manifest not found: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("task") != "g1_loco_teacher":
            raise ValueError(f"{manifest_path} task={payload.get('task')!r}; expected 'g1_loco_teacher'")
        listed = payload.get("checkpoint")
        if not listed:
            raise ValueError(f"{manifest_path} is missing checkpoint")
        listed_path = Path(str(listed))
        if not listed_path.is_absolute():
            listed_path = manifest_path.parent / listed_path
        if listed_path.resolve() != teacher_resolved:
            raise ValueError(f"{manifest_path} points at {listed_path}, not the frozen teacher {path}")
        if payload.get("checkpoint_sha256") not in (None, digest):
            raise ValueError(f"{manifest_path} checkpoint_sha256 mismatch for {path}")
    info["manifests"] = [str(item) for item in manifests]
    return info


def train():
    teacher_gate_info = _require_g1_sparse_teacher(
        Path(args_cli.teacher_checkpoint),
        eval_manifests=args_cli.teacher_eval_manifest,
        allow_ungated=bool(args_cli.allow_ungated_teacher),
    )
    env_cfg = G1LocoSparseDepthStudentReprFirstEnvCfg()
    agent_cfg = G1SparseDepthStudentAgentCfg()
    if args_cli.task_num_envs is not None:
        env_cfg.scene.num_envs = args_cli.task_num_envs
    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
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
    env = LightLPDepthDistillationEnv(env_cfg, args_cli.headless)
    log_root = Path("logs") / agent_cfg.experiment_name
    log_dir = log_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir = Path(f"{log_dir}_{agent_cfg.run_name}")
    from rsl_rl.runners import OnPolicyRunner  # noqa: E402

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
    teacher_path = Path(args_cli.teacher_checkpoint)
    print(f"[INFO] Loading frozen G1 sparse teacher from: {teacher_path}")
    print(f"[INFO] Teacher gate: {json.dumps(teacher_gate_info)}")
    runner.load(str(teacher_path), load_optimizer=False)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)


if __name__ == "__main__":
    train()
    simulation_app.close()

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
parser.add_argument("--task_num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=None)
add_rsl_rl_args(parser)
patch_physx_backward_compatibility_setting(AppLauncher)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

from legged_lab.assets.t4.student_lineage import (  # noqa: E402
    load_student_warmstart_checkpoint,
    require_json_gate,
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
from legged_lab.envs.t4.depth_student_cfg import T4SparseDepthStudentReprFirstAgentCfg  # noqa: E402
from legged_lab.envs.t4.depth_student_env import (  # noqa: E402
    T4LocoSparseDepthDistillEnv,
    T4LocoSparseDepthStudentReprFirstEnvCfg,
)

patch_missing_physx_material_attributes()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def train():
    env_cfg = T4LocoSparseDepthStudentReprFirstEnvCfg()
    agent_cfg = T4SparseDepthStudentReprFirstAgentCfg()
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
    if bool(getattr(agent_cfg, "resume", False)) and args_cli.student_warmstart_checkpoint:
        raise ValueError("resume and --student_warmstart_checkpoint are mutually exclusive")
    warmstart_info = None
    resume_info = None
    if args_cli.student_warmstart_checkpoint:
        warmstart_path = Path(args_cli.student_warmstart_checkpoint)
        warmstart_info = load_student_warmstart_checkpoint(runner.alg.policy, warmstart_path)
        print(
            "[INFO] Warm-started student control stack "
            f"from {warmstart_path} (iter={warmstart_info['iter']}, keys={warmstart_info['loaded_keys']}); "
            "critic, Adam, and safe-update counters reset; "
            f"action std reset to init={warmstart_info['init_action_std']:.3f} "
            f"(source max={warmstart_info['source_action_std']['max']:.3f})."
        )
    elif bool(getattr(agent_cfg, "resume", False)):
        from isaaclab_tasks.utils import get_checkpoint_path

        ckpt_arg = str(getattr(agent_cfg, "load_checkpoint", "") or "")
        ckpt_path = Path(ckpt_arg)
        if ckpt_path.is_file():
            resume_path = str(ckpt_path.resolve())
        else:
            resume_path = get_checkpoint_path(
                os.path.abspath(str(log_root)),
                agent_cfg.load_run,
                agent_cfg.load_checkpoint,
            )
        load_optimizer = not bool(getattr(args_cli, "reset_optimizer", False))
        print(f"[INFO] Resuming student from: {resume_path} (load_optimizer={load_optimizer})")
        runner.load(resume_path, load_optimizer=load_optimizer)
        resume_info = {
            "path": resume_path,
            "iter": int(runner.current_learning_iteration),
            "load_optimizer": load_optimizer,
            "load_run": str(getattr(agent_cfg, "load_run", "")),
        }

    if int(os.getenv("RANK", "0")) == 0:
        log_dir.mkdir(parents=True, exist_ok=True)
        lineage = {
            "task": "t4_loco_sparse_depth_student",
            "phase": "representation",
            "switch_iter": None,
            "switch_reason": None,
            "repr_first": True,
            "teacher_checkpoint": str(teacher_path.resolve()),
            "teacher_sha256": teacher_gate_info["sha256"],
            "teacher_gate": teacher_gate_info,
            "capability_gate_json": [
                require_json_gate(item, label="teacher capability gate")
                for item in teacher_gate_info["manifests"]
            ],
            "student_warmstart": warmstart_info,
            "student_resume": resume_info,
            "algorithm": agent_cfg.algorithm.class_name,
            "teacher_mix": float(agent_cfg.algorithm.teacher_mix),
            "teacher_mix_end": float(agent_cfg.algorithm.teacher_mix_end),
            "teacher_mix_decay_iters": int(agent_cfg.algorithm.teacher_mix_decay_iters),
            "action_mix_decay_iters": int(getattr(agent_cfg.algorithm, "action_mix_decay_iters", 0)),
            "pg_coef": float(agent_cfg.algorithm.pg_coef),
            "pg_coef_ramp_iters": int(getattr(agent_cfg.algorithm, "pg_coef_ramp_iters", 0)),
            "pg_delay_iters": int(getattr(agent_cfg.algorithm, "pg_delay_iters", 0)),
            "schedule": str(agent_cfg.algorithm.schedule),
            "learning_rate": float(agent_cfg.algorithm.learning_rate),
            "behavior_coef": float(agent_cfg.algorithm.behavior_coef),
            "behavior_coef_end": float(agent_cfg.algorithm.behavior_coef_end),
            "behavior_coef_decay_iters": int(agent_cfg.algorithm.behavior_coef_decay_iters),
            "critic_warmup_iters": int(agent_cfg.algorithm.critic_warmup_iters),
            "recon_coef": float(agent_cfg.algorithm.recon_coef),
            "entropy_coef": float(agent_cfg.algorithm.entropy_coef),
            "repr_cap_iters": int(getattr(agent_cfg.algorithm, "repr_cap_iters", 0)),
            "action_std_reset": None,
            "max_iterations": int(agent_cfg.max_iterations),
            "enable_cameras": True,
            "num_envs": int(env_cfg.scene.num_envs),
            "student_depth_noise": bool(getattr(env_cfg, "student_depth_noise", False)),
            "student_camera_pos_jitter_m": float(getattr(env_cfg, "student_camera_pos_jitter_m", 0.0)),
            "student_camera_ori_jitter_rad": float(getattr(env_cfg, "student_camera_ori_jitter_rad", 0.0)),
            "noise_model": {
                "depth_noise": bool(getattr(env_cfg, "student_depth_noise", False)),
                "depth_hold_steps": int(getattr(env_cfg, "student_depth_hold_steps", 0)),
                "depth_delay_steps": list(getattr(env_cfg, "student_depth_delay_steps", ())),
                "camera_pos_jitter_m": float(getattr(env_cfg, "student_camera_pos_jitter_m", 0.0)),
                "camera_ori_jitter_rad": float(getattr(env_cfg, "student_camera_ori_jitter_rad", 0.0)),
            },
            "sparse_curriculum_demote": bool(getattr(env_cfg, "sparse_curriculum_demote", True)),
            "representation_curriculum": {
                "random_level_reset_fraction": 0.50,
                "random_level_reset_min_level": 6,
            },
            "action_curriculum": {
                "random_level_reset_fraction": 0.10,
                "random_level_reset_min_level": None,
            },
            "random_level_reset_fraction": float(getattr(env_cfg, "random_level_reset_fraction", 0.0)),
            "random_level_reset_min_level": getattr(env_cfg, "random_level_reset_min_level", None),
            "random_level_reset_max_level": getattr(env_cfg, "random_level_reset_max_level", None),
            "optimizer_reset": False if resume_info and resume_info.get("load_optimizer") else True,
            "safe_update_counters_reset": False if resume_info and resume_info.get("load_optimizer") else True,
            "last_probe": None,
        }

        def write_lineage(updates=None):
            payload = dict(lineage)
            if updates:
                payload.update(updates)
                lineage.update(updates)
            (log_dir / "student_lineage.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        def on_repr_switch(algorithm):
            probe = algorithm._repr_state.last_probe
            write_lineage(
                {
                    "phase": algorithm._repr_state.phase,
                    "switch_iter": algorithm._repr_state.switch_iter,
                    "switch_reason": algorithm._repr_state.switch_reason,
                    "random_level_reset_fraction": float(getattr(env.cfg, "random_level_reset_fraction", 0.10)),
                    "random_level_reset_min_level": getattr(env.cfg, "random_level_reset_min_level", None),
                    "last_probe": None
                    if probe is None
                    else {
                        "mse_global": probe.mse_global,
                        "mse_stepping_stones": probe.mse_stepping_stones,
                        "mse_raised_pillars": probe.mse_raised_pillars,
                        "occ_agree_stepping_stones": probe.occ_agree_stepping_stones,
                        "occ_agree_raised_pillars": probe.occ_agree_raised_pillars,
                    },
                }
            )

    else:

        def on_repr_switch(algorithm):
            return None

    attach = getattr(runner.alg, "attach_repr_runtime", None)
    if callable(attach):
        attach(env, on_switch=on_repr_switch)
    if int(os.getenv("RANK", "0")) == 0:
        state = getattr(runner.alg, "_repr_state", None)
        write_lineage(
            {
                "phase": getattr(state, "phase", "representation"),
                "switch_iter": getattr(state, "switch_iter", None),
                "switch_reason": getattr(state, "switch_reason", None),
                "random_level_reset_fraction": float(getattr(env.cfg, "random_level_reset_fraction", 0.0)),
                "random_level_reset_min_level": getattr(env.cfg, "random_level_reset_min_level", None),
            }
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

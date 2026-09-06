"""Start a gated S12 GRU student continuation: joint, deploy, targeted, or residual FT."""

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

parser = argparse.ArgumentParser(description="Continue the S12 GRU student in a new gated lineage.")
parser.add_argument("--mode", choices=("joint", "deploy_ft", "targeted_ft", "residual_ft", "plant_ft"), required=True)
parser.add_argument("--student_checkpoint", type=str, required=True, help="Best checkpoint selected by the prior gate.")
parser.add_argument("--teacher_checkpoint", type=str, required=True, help="Frozen S12 model_21500.pt teacher.")
parser.add_argument(
    "--teacher_eval_manifest",
    action="append",
    default=[],
    help="Gate-0 evaluator JSON matching the frozen teacher. Repeatable.",
)
parser.add_argument(
    "--capability_gate_json",
    action="append",
    required=True,
    help="Student-only evaluator JSON authorizing this phase transition. Repeatable.",
)
parser.add_argument("--task_num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=None)
add_rsl_rl_args(parser)
patch_physx_backward_compatibility_setting(AppLauncher)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

from legged_lab.assets.t4.student_lineage import (  # noqa: E402
    load_student_continuation_checkpoint,
    require_json_gate,
    require_sparse_teacher_checkpoint,
)

teacher_gate_info = require_sparse_teacher_checkpoint(
    Path(args_cli.teacher_checkpoint),
    eval_manifests=[Path(item) for item in args_cli.teacher_eval_manifest],
)
capability_gate_info = [require_json_gate(item) for item in args_cli.capability_gate_json]

args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.utils.io import dump_yaml  # noqa: E402
from legged_lab.envs.t4.depth_student_cfg import (  # noqa: E402
    T4SparseDepthStudentDeployFtAgentCfg,
    T4SparseDepthStudentFtAgentCfg,
    T4SparseDepthStudentPlantFtAgentCfg,
    T4SparseDepthStudentResidualFtAgentCfg,
    T4SparseDepthStudentTargetedFtAgentCfg,
)
from legged_lab.envs.t4.depth_student_env import T4LocoSparseDepthStudentFtEnvCfg, T4LocoSparseDepthStudentPlantFtEnvCfg, T4LocoSparseDepthStudentResidualFtEnvCfg, T4LocoSparseDepthStudentTargetedFtEnvCfg  # noqa: E402
from legged_lab.locomotion.depth_env import LightLPDepthDistillationEnv  # noqa: E402

patch_missing_physx_material_attributes()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def train():
    if args_cli.mode == "joint":
        phase = "joint"
        env_cfg = T4LocoSparseDepthStudentFtEnvCfg()
        agent_cfg = T4SparseDepthStudentFtAgentCfg()
    elif args_cli.mode == "deploy_ft":
        phase = "deploy_ft"
        env_cfg = T4LocoSparseDepthStudentFtEnvCfg()
        agent_cfg = T4SparseDepthStudentDeployFtAgentCfg()
    elif args_cli.mode == "residual_ft":
        phase = "residual_ft"
        env_cfg = T4LocoSparseDepthStudentResidualFtEnvCfg()
        agent_cfg = T4SparseDepthStudentResidualFtAgentCfg()
    elif args_cli.mode == "targeted_ft":
        phase = "targeted_ft"
        env_cfg = T4LocoSparseDepthStudentTargetedFtEnvCfg()
        agent_cfg = T4SparseDepthStudentTargetedFtAgentCfg()
    elif args_cli.mode == "plant_ft":
        phase = "plant_ft"
        env_cfg = T4LocoSparseDepthStudentPlantFtEnvCfg()
        agent_cfg = T4SparseDepthStudentPlantFtAgentCfg()
    else:
        raise ValueError(f"unsupported FT mode: {args_cli.mode}")
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

    env = LightLPDepthDistillationEnv(env_cfg, args_cli.headless)
    log_root = Path("logs") / agent_cfg.experiment_name
    log_dir = log_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir = Path(f"{log_dir}_{agent_cfg.run_name}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)

    student_path = Path(args_cli.student_checkpoint)
    if not student_path.is_file():
        raise FileNotFoundError(f"student checkpoint not found: {student_path}")
    teacher_path = Path(args_cli.teacher_checkpoint)
    print(
        "[INFO] Loading S12 GRU depth student "
        f"(FT critic_warmup={agent_cfg.algorithm.critic_warmup_iters}, "
        f"pg_coef={agent_cfg.algorithm.pg_coef}, "
        f"pg_coef_ramp_iters={getattr(agent_cfg.algorithm, 'pg_coef_ramp_iters', 0)}, "
        f"behavior_coef={agent_cfg.algorithm.behavior_coef}, "
        f"schedule={agent_cfg.algorithm.schedule}) from: {student_path}"
    )
    continuation = load_student_continuation_checkpoint(
        runner.alg.policy,
        student_path,
        teacher_path,
        reset_action_std=0.08,
    )
    reference_action_coef = float(getattr(agent_cfg.algorithm, "reference_action_coef", 0.0))
    if reference_action_coef > 0.0:
        runner.alg.set_reference_policy_from_current()
    runner.current_learning_iteration = 0
    if hasattr(runner.alg, "num_updates"):
        runner.alg.num_updates = 0
        if hasattr(runner.alg, "num_accepted_updates"):
            runner.alg.num_accepted_updates = 0
    print(
        f"[INFO] New {phase} lineage: full student/GRU/critic loaded, optimizer and iteration reset, "
        f"action std={continuation['reset_action_std']:.2f}; frozen teacher verified at {teacher_path}."
    )

    if int(os.getenv("RANK", "0")) == 0:
        log_dir.mkdir(parents=True, exist_ok=True)
        lineage = {
            "task": "t4_loco_sparse_depth_student_ft",
            "phase": phase,
            "parent_student_checkpoint": str(student_path.resolve()),
            "parent_student_sha256": continuation["sha256"],
            "parent_student_iter": continuation["iter"],
            "teacher_checkpoint": str(teacher_path.resolve()),
            "teacher_sha256": teacher_gate_info["sha256"],
            "teacher_gate": teacher_gate_info,
            "capability_gate_json": capability_gate_info,
            "algorithm": agent_cfg.algorithm.class_name,
            "teacher_mix": float(agent_cfg.algorithm.teacher_mix),
            "teacher_mix_end": float(agent_cfg.algorithm.teacher_mix_end),
            "teacher_mix_decay_iters": int(agent_cfg.algorithm.teacher_mix_decay_iters),
            "pg_coef": float(agent_cfg.algorithm.pg_coef),
            "pg_coef_ramp_iters": int(getattr(agent_cfg.algorithm, "pg_coef_ramp_iters", 0)),
            "behavior_coef": float(agent_cfg.algorithm.behavior_coef),
            "behavior_coef_end": float(agent_cfg.algorithm.behavior_coef_end),
            "reference_action_coef": reference_action_coef,
            "critic_warmup_iters": int(agent_cfg.algorithm.critic_warmup_iters),
            "recon_coef": float(agent_cfg.algorithm.recon_coef),
            "schedule": str(agent_cfg.algorithm.schedule),
            "learning_rate": float(agent_cfg.algorithm.learning_rate),
            "max_iterations": int(agent_cfg.max_iterations),
            "student_depth_noise": True,
            "random_level_reset_fraction": float(getattr(env_cfg, "random_level_reset_fraction", 0.0)),
            "random_level_reset_min_level": getattr(env_cfg, "random_level_reset_min_level", None),
            "noise_model": {
                "depth_noise": bool(getattr(env_cfg, "student_depth_noise", False)),
                "depth_hold_steps": int(getattr(env_cfg, "student_depth_hold_steps", 0)),
                "depth_delay_steps": list(getattr(env_cfg, "student_depth_delay_steps", ())),
                "camera_pos_jitter_m": float(getattr(env_cfg, "student_camera_pos_jitter_m", 0.0)),
                "camera_ori_jitter_rad": float(getattr(env_cfg, "student_camera_ori_jitter_rad", 0.0)),
                "depth_boundary_corruption": bool(
                    getattr(env_cfg, "student_depth_boundary_corruption", False)
                ),
            },
            "action_delay_steps": dict(getattr(env_cfg.domain_rand.action_delay, "params", {})),
            "actuator_domain_randomization": {
                "stiffness_scale": [0.9, 1.1],
                "damping_scale": [0.9, 1.1],
                "armature_scale": [0.8, 1.2],
                "effort_scale": [0.9, 1.1],
            },
            "terrain_manufacturing_variation": {
                "enabled": bool(
                    getattr(
                        env_cfg.scene.terrain_generator.sub_terrains.get("raised_pillars"),
                        "targeted_manufacturing_variation",
                        False,
                    )
                ),
                "independent_layout_seed": bool(
                    getattr(
                        env_cfg.scene.terrain_generator.sub_terrains.get("stepping_stones"),
                        "targeted_layout_seed",
                        False,
                    )
                ),
                "pillar_xy_jitter_m": float(
                    getattr(env_cfg.scene.terrain_generator.sub_terrains.get("raised_pillars"), "xy_jitter_m", 0.0)
                ),
                "pillar_height_jitter_m": float(
                    getattr(
                        env_cfg.scene.terrain_generator.sub_terrains.get("raised_pillars"),
                        "height_jitter_m",
                        0.0,
                    )
                ),
                "pillar_top_tilt_rad": float(
                    getattr(env_cfg.scene.terrain_generator.sub_terrains.get("raised_pillars"), "top_tilt_rad", 0.0)
                ),
                "pillar_diameter_scale_jitter": float(
                    getattr(
                        env_cfg.scene.terrain_generator.sub_terrains.get("raised_pillars"),
                        "diameter_scale_jitter",
                        0.0,
                    )
                ),
                "pillar_pitch_scale_jitter": float(
                    getattr(
                        env_cfg.scene.terrain_generator.sub_terrains.get("raised_pillars"),
                        "pitch_scale_jitter",
                        0.0,
                    )
                ),
            },
            "optimizer_reset": True,
            "freeze_actor_during_critic_warmup": True,
            "reset_learning_iteration": True,
            "action_std_reset": continuation["reset_action_std"],
            "entropy_coef": float(agent_cfg.algorithm.entropy_coef),
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

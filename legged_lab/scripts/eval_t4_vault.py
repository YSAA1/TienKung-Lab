"""Strict G1 vault evaluator: corridor + ordered gates + zip landing semantics.

An untrained / zero policy must emit a complete lineage JSON and be judged a
failure. Reward and episode length are never used as success substitutes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

from isaaclab.app import AppLauncher

from legged_lab.scripts.isaaclab_runtime_compat import (
    patch_missing_physx_material_attributes,
    patch_physx_backward_compatibility_setting,
)

parser = argparse.ArgumentParser(description="Evaluate a T4 vault checkpoint with the V2 corridor contract.")
parser.add_argument("--task", default="t4_vault_mimic_eval")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--episodes", type=int, default=16)
parser.add_argument("--load_run", default="")
parser.add_argument("--checkpoint", default="zero-policy")
parser.add_argument("--output", required=True)
parser.add_argument("--policy", choices=("checkpoint", "zero"), default="checkpoint")
parser.add_argument("--max_steps", type=int, default=345)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--bucket", default="g1_fixed_1m")
patch_physx_backward_compatibility_setting(AppLauncher)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402

import legged_lab.envs.t4.vault_mimic  # noqa: F401,E402
from legged_lab.assets.t4.vault_contract import T4_VAULT_ACTION_SCALE, T4_VAULT_MOTION_FILE  # noqa: E402
from legged_lab.envs.t4.vault_eval import (  # noqa: E402
    VaultBatchResult,
    build_evaluation_report,
    canonical_vault_corridor_layout,
    evaluate_vault_batch,
)
from legged_lab.envs.t4.vault_mimic.rsl_rl_compat import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

patch_missing_physx_material_attributes()

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git_lineage(repo_root: Path) -> tuple[str, bool, str]:
    if not (repo_root / ".git").exists():
        return "not-packaged", False, hashlib.sha256(b"").hexdigest()
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z"], cwd=repo_root, check=True, capture_output=True
        ).stdout
        diff = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=repo_root, check=True, capture_output=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "git-unavailable", False, hashlib.sha256(b"").hexdigest()
    return head.decode(), bool(status), hashlib.sha256(status + b"\0" + diff).hexdigest()


def _checkpoint_path(agent_cfg) -> str:
    if args_cli.policy == "zero":
        return "zero-policy"
    candidate = Path(args_cli.checkpoint)
    if candidate.is_file():
        return str(candidate)
    if not args_cli.load_run:
        raise ValueError("--load_run is required when --policy checkpoint")
    root = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
    return str(get_checkpoint_path(root, args_cli.load_run, args_cli.checkpoint))


def _independent_reset(env):
    """Start a trial batch from a warm RSI frame-0 state.

    Cold ``reset()`` + ``sim.forward()`` still leaves wrist body poses stale
    until PhysX has taken at least one step. A discarded zero-action step
    followed by a second reset matches the working post-termination path
    and keeps last-action / motion time from the previous batch out.
    """
    env.reset()
    zeros = torch.zeros((env.num_envs, env.num_actions), device=env.unwrapped.device)
    env.step(zeros)
    env.reset()
    unwrapped = env.unwrapped
    unwrapped.scene.write_data_to_sim()
    unwrapped.sim.forward()
    return env.get_observations()


def evaluate() -> dict:
    spec = gym.spec(args_cli.task)
    env_cfg = spec.kwargs["env_cfg_entry_point"]()
    agent_cfg = spec.kwargs["rsl_rl_cfg_entry_point"]()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device or agent_cfg.device
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))

    episode_length_steps = math.ceil(env_cfg.episode_length_s / (env_cfg.decimation * env_cfg.sim.dt))
    if args_cli.max_steps >= episode_length_steps:
        raise ValueError(
            f"--max_steps ({args_cli.max_steps}) must be below the timeout "
            f"({episode_length_steps}) so every early done is a hard violation."
        )

    checkpoint = _checkpoint_path(agent_cfg)
    if args_cli.policy == "checkpoint":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(Path(checkpoint).parent), device=env.device)
        runner.load(checkpoint, load_optimizer=False)
        policy = runner.get_inference_policy(device=env.device)
    else:

        def policy(obs):
            return torch.zeros((obs.shape[0], env.num_actions), device=env.device)

    layout = canonical_vault_corridor_layout()
    requested = args_cli.episodes
    completed = 0
    episode_results = []

    while completed < requested:
        obs, _ = _independent_reset(env)
        batch = min(args_cli.num_envs, requested - completed)
        active = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=env.device)
        active[:batch] = True
        hard_violation = torch.zeros_like(active)
        termination_step = torch.full((args_cli.num_envs,), -1, dtype=torch.int64, device=env.device)
        termination_terms: list[list[str]] = [[] for _ in range(args_cli.num_envs)]
        root_x_steps, root_y_steps, root_z_steps = [], [], []
        root_upright_steps, root_lin_steps, root_ang_steps = [], [], []
        last_root = last_upright = last_lin = last_ang = None

        for step_index in range(args_cli.max_steps):
            robot_data = env.unwrapped.scene["robot"].data
            root = robot_data.root_pos_w - env.unwrapped.scene.env_origins
            upright = -robot_data.projected_gravity_b[:, 2]
            lin_speed = torch.linalg.vector_norm(robot_data.root_lin_vel_w, dim=-1)
            ang_speed = torch.linalg.vector_norm(robot_data.root_ang_vel_w, dim=-1)
            if last_root is None:
                last_root = root.clone()
                last_upright = upright.clone()
                last_lin = lin_speed.clone()
                last_ang = ang_speed.clone()
            last_root[active] = root[active]
            last_upright[active] = upright[active]
            last_lin[active] = lin_speed[active]
            last_ang[active] = ang_speed[active]
            root_x_steps.append(last_root[:, 0].detach().cpu().numpy().copy())
            root_y_steps.append(last_root[:, 1].detach().cpu().numpy().copy())
            root_z_steps.append(last_root[:, 2].detach().cpu().numpy().copy())
            root_upright_steps.append(last_upright.detach().cpu().numpy().copy())
            root_lin_steps.append(last_lin.detach().cpu().numpy().copy())
            root_ang_steps.append(last_ang.detach().cpu().numpy().copy())

            with torch.inference_mode():
                actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            newly_done = active & dones.to(device=active.device, dtype=torch.bool).flatten()
            termination_step[newly_done] = step_index
            term_manager = env.unwrapped.termination_manager
            for term_name in term_manager.active_terms:
                term_done = term_manager.get_term(term_name).to(device=active.device, dtype=torch.bool)
                for env_index in torch.where(newly_done & term_done)[0].tolist():
                    termination_terms[env_index].append(term_name)
            hard_violation |= newly_done
            active &= ~newly_done
            if not bool(active.any()):
                break

        batch_result = evaluate_vault_batch(
            np.stack(root_x_steps, axis=0)[:, :batch],
            np.stack(root_y_steps, axis=0)[:, :batch],
            np.stack(root_z_steps, axis=0)[:, :batch],
            np.stack(root_upright_steps, axis=0)[:, :batch],
            np.stack(root_lin_steps, axis=0)[:, :batch],
            np.stack(root_ang_steps, axis=0)[:, :batch],
            layout=layout,
            hard_violation=hard_violation[:batch].detach().cpu().numpy(),
            termination_steps=termination_step[:batch].detach().cpu().numpy(),
            termination_terms=tuple(tuple(names) for names in termination_terms[:batch]),
        )
        episode_results.extend(batch_result.episodes)
        completed += batch

    result = VaultBatchResult(
        total=len(episode_results),
        successes=sum(ep.success for ep in episode_results),
        success_rate=sum(ep.success for ep in episode_results) / max(1, len(episode_results)),
        episodes=tuple(episode_results),
    )
    git_head, git_dirty, git_dirty_diff_sha256 = _git_lineage(REPO_ROOT)
    report = build_evaluation_report(
        result,
        checkpoint=checkpoint,
        motion=str(T4_VAULT_MOTION_FILE),
        task=args_cli.task,
        seed=args_cli.seed,
        git_head=git_head,
        git_dirty=git_dirty,
        git_dirty_diff_sha256=git_dirty_diff_sha256,
        bucket=args_cli.bucket,
        policy=args_cli.policy,
        action_scale=T4_VAULT_ACTION_SCALE,
        layout=layout,
    )
    output = Path(args_cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    env.close()
    return report


if __name__ == "__main__":
    exit_code = 0
    try:
        evaluate()
    except BaseException:
        import traceback

        traceback.print_exc()
        exit_code = 1
    finally:
        import threading

        threading.Timer(90.0, os._exit, args=(exit_code,)).start()
        simulation_app.close()
        os._exit(exit_code)

"""MuJoCo behavioral probe on sparse foothold terrains (G1).

Runs fixed [vx, 0, 0] episodes on stepping_stones / raised_pillars at a chosen
difficulty and reports per-episode forward progress, fall events, and pit
falls, mirroring the Isaac evaluator metrics (strict 3 m progress, reach_2m,
fall/pit counts) so the sim-to-sim capability gap is quantified, not guessed.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from legged_lab.assets.t4.schemas import PROPRIO_HISTORY_LENGTH  # noqa: E402
from legged_lab.scripts.play_t4_sparse_teacher_mujoco import (  # noqa: E402
    ACTION_SCALE,
    CLIP_ACTIONS,
    DECIMATION,
    GAIT_CYCLE,
    PHASE_OFFSET,
    PHYSICS_DT,
    STANDING_COMMAND_THRESHOLD,
    T4SparseTeacherMujocoRunner,
    PIT_FLOOR_TOP_Z,
)
from legged_lab.scripts.play_t4_teacher_viser import quat_rotate_inverse_wxyz  # noqa: E402


def gravity_z(quat):
    return quat_rotate_inverse_wxyz(quat, np.array([0.0, 0.0, -1.0]))[2]


def run_episode(runner: T4SparseTeacherMujocoRunner, vx: float, steps: int) -> dict:
    runner.reset()
    runner.command[:] = [vx, 0.0, 0.0]
    # Isaac training reset convention: pre-filled history; first frame has
    # gait phase [0, 0] (the offset only enters after the first step()).
    runner.gait_time = 0.0
    runner.gait_phase = np.zeros_like(runner.gait_phase)
    runner.action[:] = 0.0
    frame0 = runner._proprio_frame()
    runner.history = deque(
        [frame0.copy() for _ in range(PROPRIO_HISTORY_LENGTH)], maxlen=PROPRIO_HISTORY_LENGTH
    )
    scan0 = runner._height_scan()
    runner.scan_history = deque(
        [scan0.copy() for _ in range(runner.scan_history_length)], maxlen=runner.scan_history_length
    )

    max_x = float(runner.data.qpos[0])
    min_z = float(runner.data.qpos[2])
    fall_step = None
    pit_step = None
    for t in range(steps):
        obs = runner.observe()
        with torch.no_grad():
            action = runner.actor(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
        clipped = np.clip(action, -CLIP_ACTIONS, CLIP_ACTIONS).astype(np.float32)
        runner.action = clipped
        runner.data.ctrl[runner.ctrl_ids] = runner.default_pos + ACTION_SCALE * clipped
        for _ in range(DECIMATION):
            mujoco.mj_step(runner.model, runner.data)
        if np.linalg.norm(runner.command[:2]) > STANDING_COMMAND_THRESHOLD:
            runner.gait_time += (PHYSICS_DT * DECIMATION) / GAIT_CYCLE
        runner.gait_phase = (runner.gait_time + PHASE_OFFSET) % 1.0
        max_x = max(max_x, float(runner.data.qpos[0]))
        min_z = min(min_z, float(runner.data.qpos[2]))
        if pit_step is None and runner.data.qpos[2] < PIT_FLOOR_TOP_Z + 0.3:
            pit_step = t
        if fall_step is None and runner.fallen:
            fall_step = t
            break
    return {
        "max_forward_x_m": max_x,
        "min_root_z_m": min_z,
        "final_x_m": float(runner.data.qpos[0]),
        "final_z_m": float(runner.data.qpos[2]),
        "fall_step": fall_step,
        "pit_step": pit_step,
        "fell": fall_step is not None,
        "reached_2m": max_x >= 2.0,
        "reached_3m": max_x >= 3.0,
        "steps_run": t + 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--terrain", choices=("stepping_stones", "raised_pillars"), required=True)
    parser.add_argument("--difficulty", type=float, default=0.0)
    parser.add_argument("--vx", type=float, default=0.7)
    parser.add_argument("--steps", type=int, default=800, help="policy steps per episode (16 s at 50 Hz)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    runner = T4SparseTeacherMujocoRunner(
        args.checkpoint,
        difficulty=args.difficulty,
        terrain=args.terrain,
    )
    episodes = [run_episode(runner, args.vx, args.steps) for _ in range(args.episodes)]
    summary = {
        "sim": "mujoco",
        "mujoco_version": mujoco.__version__,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "terrain": args.terrain,
        "difficulty": args.difficulty,
        "vx": args.vx,
        "steps_per_episode": args.steps,
        "episodes": args.episodes,
        "reached_2m": sum(e["reached_2m"] for e in episodes),
        "reached_3m": sum(e["reached_3m"] for e in episodes),
        "falls": sum(e["fell"] for e in episodes),
        "pit_entries": sum(e["pit_step"] is not None for e in episodes),
        "mean_max_forward_x_m": float(np.mean([e["max_forward_x_m"] for e in episodes])),
        "min_root_z_m": float(np.min([e["min_root_z_m"] for e in episodes])),
        "episode_details": episodes,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[SUMMARY] {args.terrain} d={args.difficulty} vx={args.vx}: "
        f"reach2m {summary['reached_2m']}/{args.episodes}, reach3m {summary['reached_3m']}/{args.episodes}, "
        f"falls {summary['falls']}, pit {summary['pit_entries']}, "
        f"mean max x {summary['mean_max_forward_x_m']:.2f} m"
    )
    for i, e in enumerate(episodes):
        print(
            f"  ep{i}: max_x={e['max_forward_x_m']:.2f} final_z={e['final_z_m']:.3f} "
            f"fall_step={e['fall_step']} pit_step={e['pit_step']}"
        )
    print(f"[OK] wrote {output}")


if __name__ == "__main__":
    main()

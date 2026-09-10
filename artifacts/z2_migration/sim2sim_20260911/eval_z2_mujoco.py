"""Headless MuJoCo sim2sim acceptance probe for the Z2 reset_aligned_v1 teacher.

Mirrors the Isaac evaluator protocol (difficulty 0.0, vx=0.7, deterministic
actor, 20 s episodes) on the local MuJoCo plant from
``legged_lab/assets/z2/mjcf/assembly.xml`` + ``Z2_29DOF_WALK_POSE_DAMPED_PD_CFG``
PD numbers. Writes one JSON per terrain plus optional follow-cam MP4s.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from legged_lab.scripts.play_t4_sparse_teacher_mujoco import T4SparseTeacherMujocoRunner  # noqa: E402

POLICY_HZ = 50.0
EPISODE_STEPS = 1000  # 20 s


def run_episode(
    runner: T4SparseTeacherMujocoRunner,
    vx: float,
    spawn_y: float = 0.0,
    spawn_yaw_deg: float = 0.0,
) -> dict:
    runner.reset()
    runner.command[:] = [vx, 0.0, 0.0]
    if spawn_y or spawn_yaw_deg:
        # Seeded-random-reset analogue of the Isaac evaluator: keep the nominal
        # stance, perturb only the base placement before the first step.
        runner.data.qpos[1] += spawn_y
        yaw = float(np.deg2rad(spawn_yaw_deg))
        runner.data.qpos[3:7] = [np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)]
        import mujoco

        mujoco.mj_forward(runner.model, runner.data)
    root_x0, root_y0 = float(runner.data.qpos[0]), float(runner.data.qpos[1])
    max_forward = 0.0
    fall_step = None
    z_min = 1e9
    speeds = []
    for step in range(EPISODE_STEPS):
        runner.step()
        z_min = min(z_min, float(runner.data.qpos[2]))
        forward = float(runner.data.qpos[0]) - root_x0
        max_forward = max(max_forward, forward)
        speeds.append(float(runner.data.qvel[0]))
        if runner.fallen:
            fall_step = step
            break
    lateral = float(runner.data.qpos[1]) - root_y0
    return {
        "fall_step": fall_step,
        "fallen": fall_step is not None,
        "max_forward_m": max_forward,
        "final_forward_m": float(runner.data.qpos[0]) - root_x0,
        "final_lateral_m": lateral,
        "mean_vx_mps": float(np.mean(speeds)) if speeds else 0.0,
        "root_z_min_m": z_min,
        "steps": step + 1,
    }


def record_mp4(
    runner: T4SparseTeacherMujocoRunner,
    vx: float,
    out_path: Path,
    seconds: float = 20.0,
) -> None:
    import mujoco

    runner.reset()
    runner.command[:] = [vx, 0.0, 0.0]
    runner.model.vis.global_.offwidth = 1280
    runner.model.vis.global_.offheight = 720
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.6, 0.0, 0.55]
    camera.distance = 3.2
    camera.elevation = -15
    camera.azimuth = 135.0
    renderer = mujoco.Renderer(runner.model, height=720, width=1280)
    frames = []
    total = int(seconds * POLICY_HZ)
    fall_steps = []
    for step in range(total):
        runner.step()
        if runner.fallen:
            fall_steps.append(step)
            runner.reset()
            runner.command[:] = [vx, 0.0, 0.0]
        camera.lookat[:] = [runner.data.qpos[0] + 0.6, runner.data.qpos[1], 0.55]
        renderer.update_scene(runner.data, camera=camera)
        frames.append(renderer.render())
    renderer.close()
    import imageio.v2 as imageio

    writer = imageio.get_writer(out_path, fps=int(POLICY_HZ), quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"mp4 {out_path.name}: {len(frames)} frames, auto-resets at steps {fall_steps[:10]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(REPO / "artifacts/checkpoints/nubot/z2_reset_aligned_v1/model_29999.pt"))
    parser.add_argument("--terrain", choices=("flat", "stepping_stones", "raised_pillars"), default="flat")
    parser.add_argument("--difficulty", type=float, default=0.0)
    parser.add_argument("--vx", type=float, default=0.7)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--spawn-jitter",
        type=int,
        default=None,
        help="Seed for per-episode reset jitter: y in [-0.06, 0.06] m, yaw in [-6, 6] deg.",
    )
    parser.add_argument("--mp4", default=None, help="Optional MP4 output path.")
    args = parser.parse_args()

    runner = T4SparseTeacherMujocoRunner(
        args.checkpoint,
        difficulty=args.difficulty,
        terrain=args.terrain,
        robot="z2",
    )
    rng = np.random.default_rng(args.spawn_jitter)
    episodes = []
    for _ in range(args.episodes):
        if args.spawn_jitter is None:
            episodes.append(run_episode(runner, args.vx))
        else:
            spawn_y = float(rng.uniform(-0.06, 0.06))
            spawn_yaw = float(rng.uniform(-6.0, 6.0))
            episode = run_episode(runner, args.vx, spawn_y=spawn_y, spawn_yaw_deg=spawn_yaw)
            episode["spawn_y"] = round(spawn_y, 4)
            episode["spawn_yaw_deg"] = round(spawn_yaw, 2)
            episodes.append(episode)
    summary = {
        "simulator": "mujoco",
        "mujoco_version": __import__("mujoco").__version__,
        "checkpoint": args.checkpoint,
        "terrain": args.terrain,
        "difficulty": args.difficulty,
        "vx_command": args.vx,
        "episodes": args.episodes,
        "episode_steps": EPISODE_STEPS,
        "robot": "z2",
        "pd_contract": "Z2_29DOF_WALK_POSE_DAMPED_PD_CFG",
        "spawn_jitter_seed": args.spawn_jitter,
        "fall_episodes": sum(1 for e in episodes if e["fallen"]),
        "reach_2m_episodes": sum(1 for e in episodes if e["max_forward_m"] >= 2.0),
        "mean_forward_m": float(np.mean([e["max_forward_m"] for e in episodes])),
        "mean_vx_mps": float(np.mean([e["mean_vx_mps"] for e in episodes])),
        "episodes_detail": episodes,
    }
    out_dir = Path(__file__).resolve().parent
    jitter_tag = "nominal" if args.spawn_jitter is None else f"jitter{args.spawn_jitter}"
    out_path = out_dir / f"mujoco_{args.terrain}_d{args.difficulty:g}_vx{args.vx:g}_{jitter_tag}.json"
    out_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes_detail"}, indent=1))
    print(f"wrote {out_path}")
    if args.mp4:
        record_mp4(runner, args.vx, out_dir / args.mp4)


if __name__ == "__main__":
    main()

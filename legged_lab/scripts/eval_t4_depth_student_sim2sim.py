"""Headless MuJoCo sim2sim evaluator for the T4 depth student policy.

Runs the trained Stage S depth student (proprio history + 3-frame head-height
depth) in the migrated T4 MJCF across the deployment courses, records a
follow-camera MP4 per course, and emits a fixed-command / goal-nav progress
JSON. Pure MuJoCo + torch, no IsaacLab runtime.

Unlike the interactive ``sim2sim_t4_depth_student`` entrypoint, this is a
non-interactive fixed evaluator: it pins the command / navigator, records
forward progress, survival and fall facts, and writes machine-checkable
evidence.

Usage (from the repo root):

    python -m legged_lab.scripts.eval_t4_depth_student_sim2sim \
        --checkpoint artifacts/checkpoints/nubot/t4_loco_depth_student_stage_s_head35/model_24999.pt \
        --output-dir artifacts/eval/t4_depth_student_sim2sim
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from legged_lab.scripts.sim2sim_t4_depth_student import (
    STEP_DT,
    CourseNavigator,
    DepthStudentSim,
    course_waypoints_from_model,
    root_yaw_wxyz,
)

FALL_TRUNK_HEIGHT = 0.4

# Course -> (duration, control). Control is "nav" for goal-commanded obstacle
# courses, or a named fixed-command script for flat control tracking.
DEFAULT_RUNS = (
    ("flat", 20.0, "forward"),
    ("flat", 16.0, "turn"),
    ("flat", 20.0, "slalom"),
    ("loco", 120.0, "nav"),
    ("hurdles", 60.0, "nav"),
    ("rule", 180.0, "nav"),
)


def _command_script(name: str):
    """Fixed command scripts matching ``sim2sim_t4_depth_student``."""
    scripts = {
        "forward": lambda t: (0.6, 0.0, 0.0),
        "turn": lambda t: (0.4 if t < 8 else 0.0, 0.0, 0.6 if t >= 8 else 0.0),
        "slalom": lambda t: (0.5, 0.4 * math.sin(2 * math.pi * t / 8.0), 0.0),
    }
    if name not in scripts:
        raise KeyError(f"unknown command script {name!r}")
    return scripts[name]


def _follow_camera(sim: DepthStudentSim):
    """Reuse the follow camera mounted two metres behind the trunk."""
    rgb_renderer = mujoco.Renderer(sim.model, height=368, width=640)
    view_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, "view_cam")
    mount_body_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "view_cam_mount")
    mount_mocap_id = sim.model.body_mocapid[mount_body_id]
    if mount_mocap_id < 0:
        raise RuntimeError("view_cam_mount is not a mocap body")
    return rgb_renderer, view_id, mount_mocap_id


def run_course(
    sim: DepthStudentSim,
    *,
    duration: float,
    control: str,
    output_mp4: Path,
    navigator: CourseNavigator | None = None,
) -> dict:
    """Run one course, write an MP4, and return progress/survival facts."""
    import imageio.v2 as imageio

    rgb_renderer, view_id, mount_mocap_id = _follow_camera(sim)
    command_fn = None if control == "nav" else _command_script(control)

    def apply_command(t: float) -> None:
        if control == "nav" and navigator is not None:
            sim.command[:] = navigator.command(sim.qpos[:2], root_yaw_wxyz(sim.qpos[3:7]))
        elif command_fn is not None:
            sim.command[:] = command_fn(t)

    sim.reset_episode()
    if navigator is not None:
        navigator.reset()

    goal_x = float(navigator.waypoints[-1, 0]) if navigator is not None else None
    steps = int(round(duration / STEP_DT))

    writer = imageio.get_writer(str(output_mp4), fps=int(round(1 / STEP_DT)))
    trunk_heights: list[float] = []
    forward_vels: list[float] = []
    xs: list[float] = []
    fall_step: int | None = None

    apply_command(0.0)
    obs = sim.observe()
    sim.act(obs)
    try:
        for step in range(steps):
            t = step * STEP_DT
            apply_command(t)
            obs, info = sim.step()
            sim.act(obs)
            trunk_heights.append(float(info["trunk_height"]))
            forward_vels.append(float(info["root_lin_vel"][0]))
            xs.append(float(sim.qpos[0]))
            if fall_step is None and info["trunk_height"] < FALL_TRUNK_HEIGHT:
                fall_step = step
            pos = sim.data.xpos[sim.trunk_id]
            sim.data.mocap_pos[mount_mocap_id] = [pos[0] - 2.0, pos[1], 0.0]
            rgb_renderer.update_scene(sim.data, camera=view_id)
            writer.append_data(rgb_renderer.render())
    finally:
        writer.close()
        rgb_renderer.close()

    heights = np.asarray(trunk_heights)
    xs_arr = np.asarray(xs)
    result = {
        "control": control,
        "duration_s": duration,
        "steps": steps,
        "goal_x": goal_x,
        "final_x": float(xs_arr[-1]),
        "max_x": float(xs_arr.max()),
        "goal_reached": bool(goal_x is not None and xs_arr[-1] >= goal_x - 0.5),
        "fell": fall_step is not None,
        "fall_step": fall_step,
        "trunk_height_mean_m": float(heights.mean()),
        "trunk_height_min_m": float(heights.min()),
        "mean_forward_velocity_mps": float(np.mean(forward_vels)),
        "actual_forward_mean_mps": float((xs_arr[-1] - xs_arr[0]) / duration),
        "final_lateral_y": float(sim.qpos[1]),
        "mp4": str(output_mp4),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default="artifacts/checkpoints/nubot/t4_loco_depth_student_stage_s_head35/model_24999.pt",
    )
    parser.add_argument("--output-dir", type=Path, default="artifacts/eval/t4_depth_student_sim2sim")
    parser.add_argument("--cruise", type=float, default=0.55, help="Navigator forward speed in m/s")
    parser.add_argument(
        "--courses",
        nargs="*",
        default=None,
        help="Override the default run list as 'course:duration:control' entries.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    runs: list[tuple[str, float, str]] = []
    if args.courses:
        for spec in args.courses:
            course, duration, control = spec.split(":")
            runs.append((course, float(duration), control))
    else:
        runs = list(DEFAULT_RUNS)

    checkpoint = args.checkpoint
    print(f"[INFO] checkpoint {checkpoint}")
    results: list[dict] = []
    for index, (course, duration, control) in enumerate(runs, start=1):
        print(f"[INFO] [{index}/{len(runs)}] course={course} control={control} duration={duration:.0f}s")
        sim = DepthStudentSim(str(checkpoint), course=course)
        navigator = None
        if control == "nav":
            navigator = CourseNavigator(course_waypoints_from_model(sim.model), cruise_vx=args.cruise)
            goal = navigator.waypoints[-1]
            print(f"[INFO]   goal=({goal[0]:.1f},{goal[1]:.1f}) cruise={args.cruise:.2f}")
        slug = f"{course}_{control}".replace("/", "_")
        mp4 = output_dir / f"{slug}.mp4"
        result = run_course(sim, duration=duration, control=control, output_mp4=mp4, navigator=navigator)
        result["course"] = course
        result["checkpoint"] = str(checkpoint)
        results.append(result)
        print(f"[INFO]   {json.dumps(result, ensure_ascii=False)}")

    summary = {
        "evaluator": "t4_depth_student_sim2sim_v1",
        "checkpoint": str(checkpoint),
        "courses": results,
    }
    summary_path = output_dir / "sim2sim_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"[DONE] wrote {summary_path}")


if __name__ == "__main__":
    main()

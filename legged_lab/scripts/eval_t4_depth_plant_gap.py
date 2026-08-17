"""对比 T4 depth student 在训练侧与 ZL MuJoCo plant 上的闭环行为。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile

import numpy as np

import legged_lab.scripts.sim2sim_t4_depth_student as sim2sim


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "artifacts/checkpoints/t4_depth_student_hurdle030_seqref05/model_25746.pt"
ZL_MJCF = ROOT / "zl_deploy/install/mj_sim/share/mj_sim/description/T4_std_add_head/xml/t4_std_add_head.xml"


def build_zl_model_text(course: str) -> str:
    """Return ZL's plant with the same diagnostic course as direct MuJoCo."""
    xml = ZL_MJCF.read_text()
    mesh_dir = (ZL_MJCF.parent / ".." / "meshes").resolve().as_posix()
    xml = xml.replace('meshdir="../meshes/"', f'meshdir="{mesh_dir}/"')
    if course == "stairs":
        extras = sim2sim.build_stair_probe_course(include_goal=False)
    elif course == "flat":
        extras = ""
    else:
        raise ValueError(f"unsupported ZL diagnostic course: {course}")
    world_end = xml.index("</worldbody>")
    return xml[:world_end] + extras + xml[world_end:]


def build_zl_model_xml(course: str) -> str:
    """Write a temporary ZL diagnostic plant and return its path."""
    xml = build_zl_model_text(course)
    path = Path(tempfile.gettempdir()) / f"t4_zl_depth_{course}.xml"
    path.write_text(xml)
    return str(path)


def longest_stair_top_run(samples: list[dict[str, float | int]]) -> int:
    """Return consecutive 10 Hz samples jointly inside the stair-top region."""
    stair_end = sim2sim.STAIR_PROBE_START_X + sim2sim.STAIR_PROBE_TREAD * sim2sim.LOCO_STAIR_STEPS
    longest = current = 0
    for row in samples:
        on_top = (
            row["x_m"] >= stair_end + 0.15
            and row["height_m"] >= sim2sim.SPAWN_Z + 0.70
            and abs(row["y_m"]) <= 0.5 * sim2sim.STAIR_PROBE_WIDTH
        )
        current = current + 1 if on_top else 0
        longest = max(longest, current)
    return longest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plant", choices=("direct", "zl"), required=True)
    parser.add_argument("--vx", type=float, required=True)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--course", choices=("flat", "stairs"), default="flat")
    parser.add_argument("--depth-source", choices=("native", "zl64x36"), default="native")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.plant == "zl":
        if not ZL_MJCF.is_file():
            raise FileNotFoundError(ZL_MJCF)
        sim2sim.build_model_xml = lambda *unused_args, **unused_kwargs: build_zl_model_xml(args.course)

    depth_source_size = (36, 64) if args.depth_source == "zl64x36" else None
    sim = sim2sim.DepthStudentSim(str(args.checkpoint), course=args.course, depth_source_size=depth_source_size)
    sim.command[:] = (args.vx, 0.0, 0.0)
    samples: list[dict[str, float | int]] = []
    try:
        obs = sim.observe()
        sim.act(obs)
        for step in range(round(args.duration / sim2sim.STEP_DT)):
            obs, info = sim.step()
            sim.act(obs)
            if step % 5 == 0:
                samples.append(
                    {
                        "step": step,
                        "x_m": float(sim.qpos[0]),
                        "y_m": float(sim.qpos[1]),
                        "height_m": float(info["trunk_height"]),
                        "yaw_deg": math.degrees(sim2sim.root_yaw_wxyz(sim.qpos[3:7])),
                        "max_abs_action": float(np.max(np.abs(sim.previous_action))),
                        "max_abs_target": float(np.max(np.abs(sim.targets))),
                    }
                )
    finally:
        sim.renderer.close()

    top_run_samples = longest_stair_top_run(samples) if args.course == "stairs" else 0
    payload = {
        "plant": args.plant,
        "course": args.course,
        "depth_source": args.depth_source,
        "checkpoint": str(args.checkpoint),
        "command": [args.vx, 0.0, 0.0],
        "duration_s": args.duration,
        "min_height_m": min(row["height_m"] for row in samples),
        "max_abs_yaw_deg": max(abs(row["yaw_deg"]) for row in samples),
        "max_abs_action": max(row["max_abs_action"] for row in samples),
        "max_abs_target": max(row["max_abs_target"] for row in samples),
        "delta_x_m": samples[-1]["x_m"] - samples[0]["x_m"],
        "delta_y_m": samples[-1]["y_m"] - samples[0]["y_m"],
        "max_height_m": max(row["height_m"] for row in samples),
        "reached_stair_m": (
            max(row["x_m"] for row in samples) - sim2sim.STAIR_PROBE_START_X if args.course == "stairs" else None
        ),
        "stair_top_consecutive_samples": top_run_samples,
        "stair_top_reached": top_run_samples >= 5,
        "fell": min(row["height_m"] for row in samples) < 0.55,
        "last": samples[-1],
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "samples"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Headless IsaacLab playback audit for migrated T4 motion clips.

This is the M0 simulator gate that follows the offline CSV/MJCF audit. It
replays each raw T4 clip on the migrated IsaacLab articulation, records measured
body facts, and leaves human visual approval as an explicit pending field.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

from legged_lab.scripts.isaaclab_runtime_compat import patch_physx_backward_compatibility_setting


parser = argparse.ArgumentParser(description="Replay migrated T4 motions in IsaacLab.")
parser.add_argument("--task", default="t4_loco_teacher")
parser.add_argument(
    "--motion-dir",
    type=Path,
    default=Path("legged_lab/envs/t4/datasets/motion_source"),
    help="Directory containing root_xyz + root_quat_xyzw + q27 CSV clips.",
)
parser.add_argument("--motion", action="append", help="Optional motion stem to replay; can be repeated.")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--fps", type=float, default=30.0)
parser.add_argument("--sim-dt", type=float, default=1.0 / 200.0)
parser.add_argument("--sim-device", default="cuda:0")
parser.add_argument("--max-frames", type=int, default=None)
patch_physx_backward_compatibility_setting(AppLauncher)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app = AppLauncher(args_cli).app

import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

from legged_lab.assets.t4.constants import T4_JOINT_NAMES  # noqa: E402
from legged_lab.assets.t4.t4 import T4_CFG  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "t4_motion_playback.v1"
RAW_WIDTH = 34
REQUIRED_BODY_HINTS = {
    "left_foot": ("left_foot", "foot_left", "ankle_l"),
    "right_foot": ("right_foot", "foot_right", "ankle_r"),
    "left_hand": ("left_palm", "lhand", "hand_l", "AL7"),
    "right_hand": ("right_palm", "rhand", "hand_r", "AR7"),
}


def _read_motion(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.open(newline="") as stream:
        for row in csv.reader(stream):
            rows.append([float(value) for value in row])
    return rows


def _xyzw_to_wxyz(quat_xyzw: list[float]) -> list[float]:
    x, y, z, w = quat_xyzw
    return [w, x, y, z]


def _finite_rows(rows: list[list[float]]) -> bool:
    return all(math.isfinite(value) for row in rows for value in row)


def _resolve_body_ids(body_names: list[str]) -> dict[str, int | None]:
    lowered = {name.lower(): index for index, name in enumerate(body_names)}
    resolved: dict[str, int | None] = {}
    for role, hints in REQUIRED_BODY_HINTS.items():
        resolved[role] = None
        for hint in hints:
            hint_lower = hint.lower()
            for body_lower, index in lowered.items():
                if hint_lower in body_lower:
                    resolved[role] = index
                    break
            if resolved[role] is not None:
                break
    return resolved


def _motion_files(motion_dir: Path, selected: list[str] | None) -> list[Path]:
    motion_dir = motion_dir if motion_dir.is_absolute() else ROOT / motion_dir
    selected_set = set(selected or [])
    paths = sorted(motion_dir.glob("*.csv"))
    if selected_set:
        paths = [path for path in paths if path.stem in selected_set]
    missing = sorted(selected_set - {path.stem for path in paths})
    if missing:
        raise FileNotFoundError(f"missing requested motions: {missing}")
    return paths


def _step_to_frame(
    robot: Articulation,
    sim: sim_utils.SimulationContext,
    row: list[float],
    joint_vel: torch.Tensor,
    device: torch.device,
) -> None:
    root_pose = torch.tensor([row[:3] + _xyzw_to_wxyz(row[3:7])], dtype=torch.float32, device=device)
    root_velocity = torch.zeros((1, 6), dtype=torch.float32, device=device)
    joint_pos = torch.tensor([row[7:]], dtype=torch.float32, device=device)

    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(root_velocity)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()
    sim.step()
    robot.update(args_cli.sim_dt)


def _audit_motion(
    path: Path,
    robot: Articulation,
    sim: sim_utils.SimulationContext,
    body_ids: dict[str, int | None],
    device: torch.device,
) -> dict:
    rows = _read_motion(path)
    reject_reasons: list[str] = []
    warnings: list[str] = []
    widths = sorted({len(row) for row in rows})
    if widths != [RAW_WIDTH]:
        reject_reasons.append(f"raw_width_mismatch:{widths}")
    if len(rows) < 2:
        reject_reasons.append("too_few_frames")
    if not _finite_rows(rows):
        reject_reasons.append("nonfinite_raw_values")

    frames = rows if args_cli.max_frames is None else rows[: args_cli.max_frames]
    joint_vel = torch.zeros((1, len(T4_JOINT_NAMES)), dtype=torch.float32, device=device)
    root_z_values: list[float] = []
    body_z_values: dict[str, list[float]] = {role: [] for role in body_ids}
    nonfinite_sim_frames = 0

    for row in frames:
        _step_to_frame(robot, sim, row, joint_vel, device)
        root_state = robot.data.root_state_w[0].detach().cpu()
        if not torch.isfinite(robot.data.root_state_w).all():
            nonfinite_sim_frames += 1
        root_z_values.append(float(root_state[2]))
        for role, body_id in body_ids.items():
            if body_id is None:
                continue
            body_z = robot.data.body_state_w[0, body_id, 2].detach().cpu()
            body_z_values[role].append(float(body_z))

    if nonfinite_sim_frames:
        reject_reasons.append(f"nonfinite_sim_frames:{nonfinite_sim_frames}")
    unresolved = sorted(role for role, body_id in body_ids.items() if body_id is None)
    if unresolved:
        warnings.append(f"unresolved_body_roles:{','.join(unresolved)}")

    return {
        "name": path.stem,
        "path": str(path.relative_to(ROOT)),
        "machine_playback_status": "reject" if reject_reasons else "simulated",
        "machine_reject_reasons": reject_reasons,
        "machine_warnings": warnings,
        "human_playback_status": "pending",
        "human_playback_notes": "Review continuous playback/video before AMP expert generation.",
        "frames_requested": len(rows),
        "frames_played": len(frames),
        "raw_width": widths[0] if len(widths) == 1 else widths,
        "root_z_min": min(root_z_values) if root_z_values else None,
        "root_z_max": max(root_z_values) if root_z_values else None,
        "body_z_min": {
            role: min(values) if values else None
            for role, values in body_z_values.items()
        },
        "body_z_max": {
            role: max(values) if values else None
            for role, values in body_z_values.items()
        },
        "nonfinite_sim_frames": nonfinite_sim_frames,
    }


def main() -> None:
    device = torch.device(args_cli.sim_device)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=args_cli.sim_dt, device=args_cli.sim_device))
    sim_utils.spawn_ground_plane("/World/ground", sim_utils.GroundPlaneCfg())
    robot = Articulation(T4_CFG.replace(prim_path="/World/T4"))
    sim.reset()

    joint_order_ok = tuple(robot.joint_names) == T4_JOINT_NAMES
    body_ids = _resolve_body_ids(list(robot.body_names))
    motions = {}
    for path in _motion_files(args_cli.motion_dir, args_cli.motion):
        motions[path.stem] = _audit_motion(path, robot, sim, body_ids, device)

    result = {
        "schema_version": SCHEMA_VERSION,
        "task": args_cli.task,
        "simulator": "isaaclab",
        "fps": args_cli.fps,
        "sim_dt": args_cli.sim_dt,
        "sim_device": args_cli.sim_device,
        "asset": {
            "num_joints": robot.num_joints,
            "num_bodies": robot.num_bodies,
            "joint_names": list(robot.joint_names),
            "body_names": list(robot.body_names),
            "joint_order_matches_t4_constants": joint_order_ok,
            "resolved_body_ids": body_ids,
        },
        "motions": motions,
        "summary": {
            "motion_count": len(motions),
            "simulated_count": sum(
                1 for motion in motions.values() if motion["machine_playback_status"] == "simulated"
            ),
            "rejected_count": sum(
                1 for motion in motions.values() if motion["machine_playback_status"] == "reject"
            ),
            "human_playback_pending_count": len(motions),
        },
        "limitations": [
            "Headless playback is a simulator machine check, not human visual approval.",
            "Continuous motion quality, sliding, foot penetration, and early foot lift need replay/video review.",
        ],
    }
    if not joint_order_ok:
        result["summary"]["rejected_count"] = len(motions)
        result["summary"]["global_reject_reason"] = "joint_order_mismatch"

    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        "T4_MOTION_PLAYBACK "
        f"task={args_cli.task} motions={result['summary']['motion_count']} "
        f"simulated={result['summary']['simulated_count']} "
        f"rejected={result['summary']['rejected_count']} "
        f"output={args_cli.output}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()

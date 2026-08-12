"""Generate the M0 machine audit for migrated T4 assets and motion clips.

This script is intentionally offline: it checks static asset contracts and raw
motion facts without importing IsaacLab. IsaacLab playback and human visual
approval remain separate M0 gates.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
T4_ASSET = ROOT / "legged_lab" / "assets" / "t4"
MOTION_SOURCE = ROOT / "legged_lab" / "envs" / "t4" / "datasets" / "motion_source"
MOTION_VISUALIZATION = ROOT / "legged_lab" / "envs" / "t4" / "datasets" / "motion_visualization"

SCHEMA_VERSION = "t4_motion_audit.v1"
RAW_WIDTH = 34
VISUALIZATION_WIDTH = 66
FLOAT_LIMIT_EPS = 1.0e-5
NEAR_LIMIT_MARGIN = 0.02
T4_RUN_HOLDOUT = "t4_run"

REQUIRED_SITES = (
    "forward_camera",
    "left_foot",
    "right_foot",
    "left_palm",
    "right_palm",
)


def _literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == name:
                return ast.literal_eval(node.value)
    raise ValueError(f"missing assignment {name} in {path}")


def _parse_floats(value: str | None) -> list[float] | None:
    if value is None:
        return None
    return [float(part) for part in value.split()]


def _mjcf_facts(mjcf_path: Path, joint_names: tuple[str, ...]) -> dict:
    root = ET.parse(mjcf_path).getroot()
    joints = [joint for joint in root.findall(".//joint") if "name" in joint.attrib]
    joint_limits: dict[str, tuple[float, float]] = {}
    for joint in joints:
        name = joint.attrib["name"]
        if "range" in joint.attrib:
            lower, upper = _parse_floats(joint.attrib["range"]) or []
            joint_limits[name] = (lower, upper)

    sites = {}
    for site in root.findall(".//site"):
        name = site.attrib.get("name")
        if name:
            sites[name] = {
                "pos": _parse_floats(site.attrib.get("pos")),
                "euler": _parse_floats(site.attrib.get("euler")),
                "quat": _parse_floats(site.attrib.get("quat")),
            }

    bodies = [body.attrib["name"] for body in root.findall(".//body") if "name" in body.attrib]
    motors = [motor.attrib["joint"] for motor in root.findall(".//motor") if "joint" in motor.attrib]
    return {
        "mjcf_path": str(mjcf_path.relative_to(ROOT)),
        "body_names": bodies,
        "num_bodies": len(bodies),
        "joint_names": [joint.attrib["name"] for joint in joints],
        "joint_order_matches_t4_constants": tuple(joint.attrib["name"] for joint in joints) == joint_names,
        "joint_limits": {name: list(limits) for name, limits in joint_limits.items()},
        "motor_joints": motors,
        "motor_order_matches_t4_constants": tuple(motors) == joint_names,
        "mjcf_sites": sites,
        "required_sites": list(REQUIRED_SITES),
        "required_sites_present": all(name in sites for name in REQUIRED_SITES),
    }


def _urdf_facts(urdf_path: Path, joint_names: tuple[str, ...]) -> dict:
    root = ET.parse(urdf_path).getroot()
    joints = [joint for joint in root.findall("joint") if joint.attrib.get("type") != "fixed"]
    links = root.findall("link")
    return {
        "urdf_path": str(urdf_path.relative_to(ROOT)),
        "robot_name": root.attrib.get("name"),
        "num_links": len(links),
        "num_non_fixed_joints": len(joints),
        "joint_names": [joint.attrib["name"] for joint in joints],
        "joint_order_matches_t4_constants": tuple(joint.attrib["name"] for joint in joints) == joint_names,
    }


def _read_csv_floats(path: Path) -> list[list[float]]:
    rows = []
    with path.open(newline="") as stream:
        for row in csv.reader(stream):
            rows.append([float(value) for value in row])
    return rows


def _read_motion_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _classify_motion(name: str) -> str:
    lower = name.lower()
    if "stand" in lower:
        return "stand"
    if "run" in lower and "jog" not in lower:
        return "run_holdout"
    if "jog" in lower:
        return "jog"
    if "side" in lower:
        return "side"
    if "rotate" in lower or "turn" in lower:
        return "turn"
    if "back" in lower:
        return "backward"
    if "walk" in lower:
        return "walk"
    return "unknown"


def _motion_audit(
    raw_path: Path,
    visualization_path: Path,
    joint_names: tuple[str, ...],
    joint_limits: dict[str, list[float]],
) -> dict:
    raw_rows = _read_csv_floats(raw_path)
    widths = sorted({len(row) for row in raw_rows})
    reject_reasons = []
    warnings = []
    if widths != [RAW_WIDTH]:
        reject_reasons.append(f"raw_width_mismatch:{widths}")
    if len(raw_rows) < 2:
        reject_reasons.append("too_few_raw_frames")

    root_xyz = [row[:3] for row in raw_rows]
    joint_rows = [row[7:] for row in raw_rows]
    quat_rows = [row[3:7] for row in raw_rows]
    quat_norms = [math.sqrt(sum(value * value for value in quat)) for quat in quat_rows]
    nonfinite_count = sum(
        1
        for row in raw_rows
        for value in row
        if not math.isfinite(value)
    )
    if nonfinite_count:
        reject_reasons.append(f"nonfinite_values:{nonfinite_count}")

    hard_limit_violations = []
    near_limit_hits = []
    min_limit_margin = None
    for frame_idx, joints in enumerate(joint_rows):
        for joint_idx, joint_name in enumerate(joint_names):
            if joint_name not in joint_limits:
                continue
            lower, upper = joint_limits[joint_name]
            value = joints[joint_idx]
            margin = min(value - lower, upper - value)
            min_limit_margin = margin if min_limit_margin is None else min(min_limit_margin, margin)
            if value < lower - FLOAT_LIMIT_EPS or value > upper + FLOAT_LIMIT_EPS:
                hard_limit_violations.append(
                    {
                        "frame": frame_idx,
                        "joint": joint_name,
                        "value": value,
                        "lower": lower,
                        "upper": upper,
                    }
                )
            elif margin <= NEAR_LIMIT_MARGIN:
                near_limit_hits.append(
                    {
                        "frame": frame_idx,
                        "joint": joint_name,
                        "value": value,
                        "lower": lower,
                        "upper": upper,
                        "margin": margin,
                    }
                )

    if hard_limit_violations:
        reject_reasons.append(f"hard_joint_limit_violations:{len(hard_limit_violations)}")
    if near_limit_hits:
        warnings.append(f"near_joint_limit_hits:{len(near_limit_hits)}")

    if raw_path.stem == T4_RUN_HOLDOUT:
        reject_reasons.append("holdout:t4_run_requires_independent_playback_and_limit_review")

    visualization = _read_motion_json(visualization_path)
    frames = visualization.get("Frames", [])
    frame_widths = sorted({len(frame) for frame in frames})
    if frame_widths != [VISUALIZATION_WIDTH]:
        reject_reasons.append(f"visualization_width_mismatch:{frame_widths}")
    if tuple(visualization.get("JointOrder", ())) != joint_names:
        reject_reasons.append("visualization_joint_order_mismatch")

    dt = float(visualization.get("FrameDuration", 0.0) or 0.0)
    duration = dt * len(frames)
    xy_displacement = math.dist(root_xyz[0][:2], root_xyz[-1][:2]) if len(root_xyz) >= 2 else 0.0
    mean_xy_speed = xy_displacement / ((len(root_xyz) - 1) * dt) if len(root_xyz) >= 2 and dt > 0 else 0.0

    return {
        "name": raw_path.stem,
        "category": _classify_motion(raw_path.stem),
        "machine_status": "reject" if reject_reasons else "accept",
        "machine_reject_reasons": reject_reasons,
        "machine_warnings": warnings,
        "human_playback_status": "pending",
        "human_playback_notes": "Must be checked in IsaacLab playback before AMP expert generation.",
        "raw": {
            "path": str(raw_path.relative_to(ROOT)),
            "frames": len(raw_rows),
            "width": widths[0] if len(widths) == 1 else widths,
            "schema": "root_xyz(3)+root_quat_xyzw(4)+q27",
            "root_z_min": min(row[2] for row in root_xyz),
            "root_z_max": max(row[2] for row in root_xyz),
            "xy_displacement": xy_displacement,
            "mean_xy_speed": mean_xy_speed,
            "quat_norm_min": min(quat_norms),
            "quat_norm_max": max(quat_norms),
            "nonfinite_count": nonfinite_count,
            "min_joint_limit_margin": min_limit_margin,
            "hard_joint_limit_violation_count": len(hard_limit_violations),
            "near_joint_limit_hit_count": len(near_limit_hits),
            "hard_joint_limit_violations_sample": hard_limit_violations[:10],
            "near_joint_limit_hits_sample": near_limit_hits[:10],
        },
        "visualization": {
            "path": str(visualization_path.relative_to(ROOT)),
            "frames": len(frames),
            "frame_width": frame_widths[0] if len(frame_widths) == 1 else frame_widths,
            "frame_duration": dt,
            "duration_s": duration,
            "joint_order_matches_t4_constants": tuple(visualization.get("JointOrder", ())) == joint_names,
            "source_schema": visualization.get("SourceSchema"),
        },
    }


def build_audit(task: str) -> dict:
    joint_names = tuple(_literal_assignment(T4_ASSET / "constants.py", "T4_JOINT_NAMES"))
    mjcf = _mjcf_facts(T4_ASSET / "mjcf" / "t4_std.xml", joint_names)
    urdf = _urdf_facts(T4_ASSET / "urdf" / "t4_std.urdf", joint_names)
    joint_limits = mjcf["joint_limits"]

    motions = {}
    for raw_path in sorted(MOTION_SOURCE.glob("*.csv")):
        visualization_path = MOTION_VISUALIZATION / f"{raw_path.stem}.txt"
        if not visualization_path.is_file():
            motions[raw_path.stem] = {
                "name": raw_path.stem,
                "machine_status": "reject",
                "machine_reject_reasons": ["missing_visualization_file"],
                "human_playback_status": "pending",
            }
            continue
        motions[raw_path.stem] = _motion_audit(raw_path, visualization_path, joint_names, joint_limits)

    accepted = sorted(name for name, motion in motions.items() if motion["machine_status"] == "accept")
    rejected = sorted(name for name, motion in motions.items() if motion["machine_status"] == "reject")
    return {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "asset": {
            "joint_names": list(joint_names),
            "joint_count": len(joint_names),
            "mjcf": mjcf,
            "urdf": urdf,
            "mjcf_sites": mjcf["mjcf_sites"],
            "required_sites_present": mjcf["required_sites_present"],
        },
        "motions": motions,
        "summary": {
            "motion_count": len(motions),
            "machine_accepted_count": len(accepted),
            "machine_rejected_count": len(rejected),
            "machine_accepted": accepted,
            "machine_rejected": rejected,
            "human_playback_pending_count": len(motions),
        },
        "limitations": [
            "Offline audit does not prove IsaacLab playback quality.",
            "Foot penetration, sliding, discontinuity, and early foot lift require visual playback review.",
            "AMP expert generation must wait for accepted machine audit and human playback approval.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = build_audit(args.task)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(
        "T4_MOTION_AUDIT "
        f"task={args.task} motions={audit['summary']['motion_count']} "
        f"accepted={audit['summary']['machine_accepted_count']} "
        f"rejected={audit['summary']['machine_rejected_count']} "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()

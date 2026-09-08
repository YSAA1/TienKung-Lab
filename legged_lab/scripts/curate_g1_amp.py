# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

"""Select continuous G1 locomotion cycles and generate a separate 70D AMP dataset.

CPU preparation uses the official URDF. Run validate_g1_curated_amp.py in Isaac
before training; kinematic checks do not establish dynamic trackability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from legged_lab.assets.unitree_g1.constants import G1_29DOF_JOINT_NAMES
from legged_lab.assets.unitree_g1.schemas import AMP_MOTION_CLASSES

ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / "legged_lab/assets/unitree_g1/urdf/g1_29dof_rev_1_0.urdf"
FPS = 30


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model():
    tree = ET.parse(URDF)
    root = tree.getroot()
    root.find("mujoco/compiler").set("meshdir", str(URDF.parent / "meshes"))
    ET.SubElement(root, "link", name="world")
    joint = ET.SubElement(root, "joint", name="curation_root", type="floating")
    ET.SubElement(joint, "parent", link="world")
    ET.SubElement(joint, "child", link="pelvis")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    joint_ids = np.array([model.joint(name).id for name in G1_29DOF_JOINT_NAMES])
    assert model.nv == 35 and len(joint_ids) == 29
    velocity_limits = np.array(
        [float(root.find(f"joint[@name='{name}']/limit").get("velocity")) for name in G1_29DOF_JOINT_NAMES]
    )
    return model, joint_ids, velocity_limits


def kinematics(model, joint_ids, rows):
    data = mujoco.MjData(model)
    bodies = [
        model.body(name).id
        for name in ("left_wrist_yaw_link", "right_wrist_yaw_link", "left_ankle_roll_link", "right_ankle_roll_link")
    ]
    spheres = [
        [
            g
            for g in range(model.ngeom)
            if model.geom_bodyid[g] == body and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
        ]
        for body in bodies[2:]
    ]
    assert [len(g) for g in spheres] == [4, 4]
    endpoints, feet, soles, points = [], [], [], []
    for row in rows:
        data.qpos[:3] = row[:3]
        data.qpos[3:7] = row[[6, 3, 4, 5]]
        data.qpos[model.jnt_qposadr[joint_ids]] = row[7:]
        mujoco.mj_forward(model, data)
        rotation = data.xmat[model.body("pelvis").id].reshape(3, 3)
        endpoints.append(((data.xpos[bodies] - row[:3]) @ rotation).reshape(-1))
        feet.append(data.xpos[bodies[2:]].copy())
        positions = data.geom_xpos[np.array(spheres)].copy()
        points.append(positions)
        soles.append((positions[:, :, 2] - model.geom_size[np.array(spheres), 0]).min(1))
    return {
        "endpoints": np.array(endpoints),
        "feet": np.array(feet),
        "sole": np.array(soles),
        "points": np.array(points),
    }


def measure(rows, kin, limits, velocity_limits):
    velocity = np.gradient(rows[:, :3], 1 / FPS, axis=0)
    x, y, z, w = rows[:, 3:7].T
    yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    cosine, sine = np.cos(yaw), np.sin(yaw)
    vx = cosine * velocity[:, 0] + sine * velocity[:, 1]
    vy = -sine * velocity[:, 0] + cosine * velocity[:, 1]
    dq = np.diff(rows[:, 7:], axis=0) * FPS
    dq = np.vstack([dq, dq[-1]])
    tilt = np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))
    point_v = np.gradient(kin["points"], 1 / FPS, axis=0)
    # Contact is a geometric proxy only. Exclude swing points and allow rolling
    # about one of the four official sole spheres; no force data is available.
    near = (kin["points"][:, :, :, 2] - 0.005 < 0.015) & (np.abs(point_v[:, :, :, 2]) < 0.15)
    point_speed = np.linalg.norm(point_v[:, :, :, :2], axis=3)
    support_speed = np.min(np.where(near, point_speed, np.inf), axis=2)
    relative = kin["feet"][:, 0] - kin["feet"][:, 1]
    return dict(
        vx=vx,
        vy=vy,
        speed=np.linalg.norm(velocity[:, :2], axis=1),
        dq=dq,
        tilt=tilt,
        smooth_vx=uniform_filter1d(vx, 15),
        turn=uniform_filter1d(np.gradient(yaw) * FPS, 15),
        lateral=uniform_filter1d(np.abs(vy), 15),
        support_speed=support_speed,
        phase=relative[:, 0] * cosine + relative[:, 1] * sine,
        limit_excess=np.maximum(limits[:, 0] - rows[:, 7:], rows[:, 7:] - limits[:, 1]).clip(0).max(1),
        velocity_ratio=np.max(np.abs(dq) / velocity_limits, axis=1),
    )


def select_segments(metrics, kin, motion_class):
    lower = 0.85 if motion_class == "run" else 0.3
    good = (
        (metrics["smooth_vx"] > lower)
        & (np.abs(metrics["turn"]) < 1.2)
        & (metrics["lateral"] < np.maximum(0.15, 0.35 * metrics["smooth_vx"]))
        & (kin["sole"].min(1) > -0.025)
        & (metrics["limit_excess"] < 1e-6)
        & (metrics["velocity_ratio"] < 1)
        & (metrics["tilt"] < 0.45)
    )
    edges = np.flatnonzero(np.diff(np.r_[False, good, False])).reshape(-1, 2)
    peaks, _ = find_peaks(metrics["phase"], prominence=0.10, distance=12)
    result = []
    for begin, end in edges:
        cycles = peaks[(peaks >= begin + 8) & (peaks < end - 8)]
        if len(cycles) < 4:
            continue
        start, stop = int(cycles[0]), int(cycles[-1])
        if stop - start < 2.5 * FPS:
            continue
        if (metrics["speed"][start:stop] < 0.1).mean() > 0.01:
            continue
        # Separate files preserve source transitions; never concatenate gaps.
        result.append((start, stop, len(cycles) - 1))
    return result


def summarize(metrics, kin, start, stop):
    sl = slice(start, stop)
    support = metrics["support_speed"][sl]
    support = support[np.isfinite(support)]
    return {
        "duration_s": (stop - start) / FPS,
        "speed_mean_m_s": float(metrics["speed"][sl].mean()),
        "forward_speed_mean_m_s": float(metrics["vx"][sl].mean()),
        "speed_p10_p50_p90_m_s": np.quantile(metrics["speed"][sl], [0.1, 0.5, 0.9]).tolist(),
        "fraction_speed_below_0_1": float((metrics["speed"][sl] < 0.1).mean()),
        "hard_joint_limit_excess_rad": float(metrics["limit_excess"][sl].max()),
        "max_joint_velocity_limit_ratio": float(metrics["velocity_ratio"][sl].max()),
        "lowest_sole_m": float(kin["sole"][sl].min()),
        "highest_sole_m": float(kin["sole"][sl].max()),
        "max_root_tilt_rad": float(metrics["tilt"][sl].max()),
        "support_point_speed_proxy_p50_p95_m_s": np.quantile(support, [0.5, 0.95]).tolist() if len(support) else [],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite a dataset: {out}")
    model, joint_ids, velocity_limits = load_model()
    old = ROOT / "legged_lab/envs/g1/datasets/motion_amp_expert_unitree_v5"
    manifest = {
        "source_dataset": "lvhaidong/LAFAN1_Retargeting_Dataset",
        "urdf_sha256": digest(URDF),
        "generator_sha256": digest(Path(__file__)),
        "fps": FPS,
        "frame_dim": 70,
        "status": "kinematic_candidate_pending_isaac_and_visual_validation",
        "limits": "Kinematic selection only; not a dynamics or learned-policy acceptance test.",
        "selection": {
            "walk_min_smoothed_forward_m_s": 0.3,
            "run_min_smoothed_forward_m_s": 0.85,
            "max_smoothed_yaw_rate_rad_s": 1.2,
            "min_duration_s": 2.5,
            "min_complete_stride_cycles": 3,
            "maximum_low_speed_fraction": 0.01,
            "max_root_tilt_rad": 0.45,
            "minimum_sole_height_m": -0.025,
        },
        "class_probabilities": {"walk_forward": 0.625, "run": 0.375},
        "sources": {},
        "clips": {},
        "rejected_segments": {},
    }
    manifest["selection"].update(max_support_speed_proxy_p95_m_s=0.5, max_walk_sole_height_m=0.15)
    prepared = []
    for stem, motion_class in AMP_MOTION_CLASSES.items():
        source = ROOT / "legged_lab/envs/g1/datasets/motion_source" / f"{stem}.csv"
        rows = np.loadtxt(source, delimiter=",")
        assert rows.shape[1] == 36 and np.isfinite(rows).all()
        assert np.max(np.abs(np.linalg.norm(rows[:, 3:7], axis=1) - 1)) < 1e-4
        kin = kinematics(model, joint_ids, rows)
        metrics = measure(rows, kin, model.jnt_range[joint_ids], velocity_limits)
        reference = json.loads((old / f"{stem}.txt").read_text())
        expected = np.array(reference["Frames"])
        old_error = float(np.max(np.abs(kin["endpoints"][: len(expected)] - expected[:, 58:])))
        assert old_error < 2e-4, (stem, old_error)
        manifest["sources"][stem] = {
            "path": str(source.relative_to(ROOT)).replace("\\", "/"),
            "sha256": digest(source),
            "frames": len(rows),
            "old_isaac_endpoint_max_error_m": old_error,
            "old_first_30s": summarize(metrics, kin, 0, len(expected)),
        }
        segments = select_segments(metrics, kin, motion_class)
        for start, stop, cycles in segments:
            name = f"{stem}_{start:05d}_{stop:05d}"
            stats = summarize(metrics, kin, start, stop)
            stats.update(
                source=stem, source_start=start, source_stop=stop, full_stride_cycles=cycles, motion_class=motion_class
            )
            support = stats["support_point_speed_proxy_p50_p95_m_s"]
            reasons = []
            if not support or support[1] > 0.5:
                reasons.append("near-ground foot motion proxy above 0.5 m/s at p95")
            if motion_class == "walk_forward" and stats["highest_sole_m"] > 0.15:
                reasons.append("high-step walking outside the plain locomotion subset")
            if reasons:
                manifest["rejected_segments"][name] = dict(stats, reasons=reasons)
                continue
            frames = np.hstack([rows[start:stop, 7:], metrics["dq"][start:stop], kin["endpoints"][start:stop]])
            prepared.append((name, frames.astype(np.float32), stats))
        print(stem, "selected", len(segments), "FK error", old_error, flush=True)
    durations = {
        key: sum(s["duration_s"] for _, _, s in prepared if s["motion_class"] == key)
        for key in manifest["class_probabilities"]
    }
    assert durations["walk_forward"] >= 20 and durations["run"] >= 8, durations
    out.mkdir(parents=True)
    for name, frames, stats in prepared:
        cls = stats["motion_class"]
        weight = manifest["class_probabilities"][cls] * stats["duration_s"] / durations[cls]
        payload = {
            "LoopMode": "Clamp",
            "FrameDuration": 1 / FPS,
            "MotionWeight": weight,
            "MotionClass": cls,
            "JointOrder": list(G1_29DOF_JOINT_NAMES),
            "AmpSchemaVersion": "g1_amp.v1",
            "SourceMotion": manifest["sources"][stats["source"]]["path"],
            "SourceDataset": manifest["source_dataset"],
            "SourceFrameStart": stats["source_start"],
            "SourceFrameStop": stats["source_stop"],
            "Frames": frames.tolist(),
        }
        path = out / f"{name}.txt"
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        manifest["clips"][name] = dict(stats, motion_weight=weight, sha256=digest(path), frames=len(frames))
    manifest["duration_by_class_s"] = durations
    (out / "_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"clips": len(prepared), "duration_by_class_s": durations, "output": str(out)}))


if __name__ == "__main__":
    main()

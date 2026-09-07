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

"""Retarget the complete T4 training motion set to the official G1 29DoF.

Bounded robot-to-robot inverse kinematics preserves scaled sole poses and wrist
trajectories. This creates kinematic experts, not a dynamics-tracking policy.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation

from legged_lab.assets.t4.constants import T4_JOINT_NAMES
from legged_lab.assets.t4.schemas import (
    AMP_MOTION_CLASS_WEIGHTS,
    AMP_MOTION_CLASSES,
    amp_motion_weight,
)
from legged_lab.assets.unitree_g1.constants import G1_29DOF_JOINT_NAMES
from legged_lab.scripts.curate_g1_amp import (
    ROOT,
    URDF,
    digest,
    kinematics,
    load_model,
    measure,
    summarize,
)


class RobotRetargeter:
    def __init__(self):
        self.model, self.joint_ids, self.velocity_limits = load_model()
        self.data = mujoco.MjData(self.model)
        self.qids = self.model.jnt_qposadr[self.joint_ids]
        self.vids = self.model.jnt_dofadr[self.joint_ids]
        self.bounds = self.model.jnt_range[self.joint_ids]
        # T4's flexing knees map to the positive-flexion G1 IK branch. G1's
        # small legal hyperextension must not become an alternate warm-start
        # branch when a source knee passes near straight.
        self.solve_bounds = self.bounds.copy()
        self.solve_bounds[[3, 9], 0] = 0
        self.source_path = ROOT / "legged_lab/assets/t4/mjcf/t4_std.xml"
        self.source_model = mujoco.MjModel.from_xml_path(str(self.source_path))
        self.source_data = mujoco.MjData(self.source_model)
        self.source_joints = np.array([self.source_model.joint(n).id for n in T4_JOINT_NAMES])
        self.source_bodies = [self.source_model.body(n).id for n in ("left_foot_link", "right_foot_link", "AL7", "AR7")]
        self.bodies = [
            self.model.body(n).id
            for n in ("left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link")
        ]
        self.source_offsets = [np.array([0.047, 0, -0.0345]), np.array([0.047, 0, -0.0345]), np.zeros(3), np.zeros(3)]
        self.offsets = []
        for body in self.bodies:
            spheres = [
                g
                for g in range(self.model.ngeom)
                if self.model.geom_bodyid[g] == body and self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
            ]
            self.offsets.append(
                np.mean(self.model.geom_pos[spheres], axis=0) - [0, 0, 0.005] if spheres else np.zeros(3)
            )
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_forward(self.source_model, self.source_data)
        source_feet = np.array(
            [
                self.point(self.source_data, b, off) - self.source_data.xpos[self.source_model.body("Trunk").id]
                for b, off in zip(self.source_bodies[:2], self.source_offsets[:2])
            ]
        )
        feet = np.array([self.point(self.data, b, off) for b, off in zip(self.bodies[:2], self.offsets[:2])])
        leg_scale = feet[:, 2].mean() / source_feet[:, 2].mean()
        self.scale = np.array([leg_scale, abs(feet[0, 1] / source_feet[0, 1]), leg_scale])
        self.source_shoulders = [self.source_model.body(n).id for n in ("AL1", "AR1")]
        self.shoulders = [self.model.body(n).id for n in ("left_shoulder_pitch_link", "right_shoulder_pitch_link")]
        self.arm_scale = [
            np.linalg.norm(self.data.xpos[self.bodies[i + 2]] - self.data.xpos[self.shoulders[i]])
            / np.linalg.norm(
                self.source_data.xpos[self.source_bodies[i + 2]] - self.source_data.xpos[self.source_shoulders[i]]
            )
            for i in range(2)
        ]
        # T4 has only waist yaw. G1 waist roll/pitch remain neutral; its other
        # 26 joints are solved within the actual G1 hard limits.
        self.active = np.array([i for i in range(29) if i not in (12, 13, 14)])

    @staticmethod
    def point(data, body, offset):
        return data.xpos[body] + data.xmat[body].reshape(3, 3) @ offset

    def pose(self, q):
        self.data.qpos[:3] = 0
        self.data.qpos[3:7] = [1, 0, 0, 0]
        self.data.qpos[self.qids] = q
        mujoco.mj_forward(self.model, self.data)

    def convert(self, rows):
        q = np.zeros(29)
        q[[0, 6]], q[[3, 9]], q[[4, 10]], q[[18, 25]] = -0.2, 0.4, -0.2, 0.5
        output, diagnostics = [], []
        for row in rows:
            self.source_data.qpos[:3] = 0
            self.source_data.qpos[3:7] = [1, 0, 0, 0]
            self.source_data.qpos[self.source_model.jnt_qposadr[self.source_joints]] = row[7:]
            mujoco.mj_forward(self.source_model, self.source_data)
            targets = np.array(
                [
                    self.point(self.source_data, b, off) * self.scale
                    for b, off in zip(self.source_bodies, self.source_offsets)
                ]
            )
            rotations = [self.source_data.xmat[b].reshape(3, 3).copy() for b in self.source_bodies[:2]]
            q[12] = row[7 + T4_JOINT_NAMES.index("J_waist_yaw")]
            self.pose(q)
            for i in range(2):
                targets[i + 2] = self.data.xpos[self.shoulders[i]] + self.arm_scale[i] * (
                    self.source_data.xpos[self.source_bodies[i + 2]] - self.source_data.xpos[self.source_shoulders[i]]
                )
            prior = q.copy()
            prior[15:22], prior[22:29] = row[7:14], row[14:21]
            prior[[18, 25]] = abs(prior[[18, 25]])
            prior[[19, 20, 21, 26, 27, 28]] = 0
            anchor = prior[self.active]
            regularization = np.where(self.active >= 15, 0.3, 0.01)

            def residual(value, jacobian=False):
                q[self.active] = value
                self.pose(q)
                errors, jacobians = [], []
                for i, (body, offset) in enumerate(zip(self.bodies, self.offsets)):
                    pos = self.point(self.data, body, offset)
                    weight = 40 if i < 2 else 6
                    errors.extend((pos - targets[i]) * weight)
                    jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
                    mujoco.mj_jac(self.model, self.data, jp, jr, pos, body)
                    jacobians.append(jp[:, self.vids[self.active]] * weight)
                    if i < 2:
                        errors.extend(
                            Rotation.from_matrix(rotations[i].T @ self.data.xmat[body].reshape(3, 3)).as_rotvec()
                        )
                        jacobians.append(rotations[i].T @ jr[:, self.vids[self.active]])
                errors.extend((value - anchor) * regularization)
                jacobians.append(np.diag(regularization))
                return np.vstack(jacobians) if jacobian else np.asarray(errors)

            solution = least_squares(
                residual,
                q[self.active],
                jac=lambda x: residual(x, True),
                bounds=(self.solve_bounds[self.active, 0] + 1e-5, self.solve_bounds[self.active, 1] - 1e-5),
                max_nfev=60,
                gtol=1e-7,
                ftol=1e-7,
                xtol=1e-7,
            )
            # A nearly straight knee can warm-start onto G1's permitted small
            # negative-knee branch. Retry from the source anatomical pose when
            # sole tracking is poor; bounds apply to the solve, never clip output.
            residual(solution.x)
            sole_error = max(
                np.linalg.norm(self.point(self.data, b, off) - targets[i])
                for i, (b, off) in enumerate(zip(self.bodies[:2], self.offsets[:2]))
            )
            if sole_error > 0.01:
                anatomical = q.copy()
                anatomical[:12] = row[22:34]
                retry = least_squares(
                    residual,
                    np.clip(
                        anatomical[self.active],
                        self.solve_bounds[self.active, 0] + 1e-4,
                        self.solve_bounds[self.active, 1] - 1e-4,
                    ),
                    jac=lambda x: residual(x, True),
                    bounds=(self.solve_bounds[self.active, 0] + 1e-5, self.solve_bounds[self.active, 1] - 1e-5),
                    max_nfev=120,
                    gtol=1e-7,
                    ftol=1e-7,
                    xtol=1e-7,
                )
                if retry.cost < solution.cost:
                    solution = retry
            residual(solution.x)
            position_errors = [
                np.linalg.norm(self.point(self.data, b, off) - targets[i])
                for i, (b, off) in enumerate(zip(self.bodies, self.offsets))
            ]
            angle_errors = [
                np.linalg.norm(Rotation.from_matrix(rotations[i].T @ self.data.xmat[b].reshape(3, 3)).as_rotvec())
                for i, b in enumerate(self.bodies[:2])
            ]
            root = row[:7].copy()
            root[:3] *= self.scale[0]  # isotropic world translation; hip width is a local-body adaptation
            output.append(np.r_[root, q.copy()])
            diagnostics.append(position_errors + angle_errors)
        return np.asarray(output), np.asarray(diagnostics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-output-dir", type=Path, required=True)
    parser.add_argument("--legacy-forward-only", action="store_true", help="Reproduce the earlier two-clip selection")
    parser.add_argument("--t4-runtime-manifest", type=Path, help="Verified T4 saved-run file list and weights")
    args = parser.parse_args()
    out, sources = args.output_dir.resolve(), args.source_output_dir.resolve()
    if out.exists() or sources.exists():
        raise FileExistsError("Refusing to overwrite previous data")
    retargeter = RobotRetargeter()
    runtime = json.loads(args.t4_runtime_manifest.read_text()) if args.t4_runtime_manifest else None
    if not args.legacy_forward_only:
        if runtime is None or set(runtime["clips"]) != set(AMP_MOTION_CLASSES):
            raise ValueError("Full conversion requires the actual T4 runtime manifest with all 17 motions")
        for stem, clip in runtime["clips"].items():
            assert clip["weight"] == amp_motion_weight(stem)
            assert clip["metadata"]["MotionClass"] == AMP_MOTION_CLASSES[stem]
    classes = (
        {"t4_walk_forward": "walk_forward", "t4_jog_forward": "run"} if args.legacy_forward_only else AMP_MOTION_CLASSES
    )
    probabilities = (
        {"walk_forward": 0.625, "run": 0.375}
        if args.legacy_forward_only
        else {
            name: weight / sum(AMP_MOTION_CLASS_WEIGHTS.values()) for name, weight in AMP_MOTION_CLASS_WEIGHTS.items()
        }
    )
    prepared = []
    manifest = {
        "source_dataset": "T4 runtime CSV robot-to-robot retarget",
        "status": "kinematic_candidate_pending_isaac_and_visual_validation",
        "fps": 30,
        "frame_dim": 70,
        "urdf_sha256": digest(URDF),
        "t4_mjcf_sha256": digest(retargeter.source_path),
        "generator_sha256": digest(Path(__file__)),
        "leg_and_hip_scale": retargeter.scale.tolist(),
        "arm_scale": retargeter.arm_scale,
        "class_probabilities": probabilities,
        "t4_runtime_manifest": runtime,
        "selection": (
            "legacy forward stride crop"
            if args.legacy_forward_only
            else "Every source frame; last pose retained for forward difference; AMP has N-1 frames"
        ),
        "method": "Bounded sole pose and shoulder-relative wrist IK; waist yaw mapped; G1 waist roll/pitch neutral.",
        "limitations": "Kinematic mapping; no claim of dynamic tracking or trained locomotion.",
        "sources": {},
        "clips": {},
    }
    for stem, cls in sorted(classes.items()):
        source = ROOT / "legged_lab/envs/t4/datasets/motion_source" / f"{stem}.csv"
        raw = np.loadtxt(source, delimiter=",")
        assert raw.shape[1] == 34 and np.isfinite(raw).all()
        assert np.max(np.abs(np.linalg.norm(raw[:, 3:7], axis=1) - 1)) < 1e-4
        if not args.legacy_forward_only:
            assert len(raw) - 1 == runtime["clips"][stem]["frames"], stem
        rows, errors = retargeter.convert(raw)
        kin = kinematics(retargeter.model, retargeter.joint_ids, rows)
        # One rigid vertical translation places the lower sole near the plane;
        # no frame-wise root lifting, q editing, smoothing or time warping.
        shift = 0.003 - float(np.quantile(kin["sole"].min(1), 0.05))
        rows[:, 2] += shift
        kin = kinematics(retargeter.model, retargeter.joint_ids, rows)
        metrics = measure(rows, kin, retargeter.bounds, retargeter.velocity_limits)
        peaks = []
        start, stop = 0, len(rows) - 1
        if args.legacy_forward_only:
            moving = uniform_filter1d(metrics["vx"], 9) > (0.7 if cls == "run" else 0.3)
            edges = np.flatnonzero(np.diff(np.r_[False, moving, False])).reshape(-1, 2)
            begin, end = max(edges, key=lambda edge: edge[1] - edge[0])
            peaks, _ = find_peaks(metrics["phase"], prominence=0.10, distance=10)
            peaks = peaks[(peaks >= begin + 3) & (peaks < end - 3)]
            assert len(peaks) >= 3, (stem, peaks.tolist())
            start, stop = int(peaks[0]), int(peaks[-1])
        stats = summarize(metrics, kin, start, stop)
        selected_errors = errors[start:stop]
        stats.update(
            max_sole_position_error_m=float(selected_errors[:, :2].max()),
            max_wrist_position_error_m=float(selected_errors[:, 2:4].max()),
            max_sole_orientation_error_rad=float(selected_errors[:, 4:].max()),
            motion_class=cls,
            source_start=start,
            source_stop=stop,
            source=stem,
            full_stride_cycles=len(peaks) - 1 if args.legacy_forward_only else None,
            lateral_speed_mean_m_s=float(metrics["vy"][start:stop].mean()),
            yaw_rate_mean_rad_s=float(metrics["turn"][start:stop].mean()),
            root_vertical_shift_m=shift,
        )
        if args.legacy_forward_only:
            assert stats["fraction_speed_below_0_1"] == 0
        assert stats["max_sole_position_error_m"] < 0.025, stats
        assert stats["max_joint_velocity_limit_ratio"] < 1, stats
        assert stats["hard_joint_limit_excess_rad"] < 1e-6
        prepared.append((stem, rows, kin, metrics, stats))
        manifest["sources"][stem] = {
            "original_path": str(source.relative_to(ROOT)).replace("\\", "/"),
            "original_sha256": digest(source),
            "source_frames": len(raw),
        }
        print(stem, json.dumps(stats), flush=True)
    out.mkdir(parents=True)
    sources.mkdir(parents=True)
    for stem, rows, kin, metrics, stats in prepared:
        csv = sources / f"g1_from_{stem}.csv"
        np.savetxt(csv, rows, delimiter=",", fmt="%.12g")
        start, stop = stats["source_start"], stats["source_stop"]
        frames = np.hstack([rows[start:stop, 7:], metrics["dq"][start:stop], kin["endpoints"][start:stop]]).astype(
            np.float32
        )
        name = f"g1_from_{stem}_{start:04d}_{stop:04d}"
        path = out / f"{name}.txt"
        weight = probabilities[stats["motion_class"]] if args.legacy_forward_only else runtime["clips"][stem]["weight"]
        payload = {
            "LoopMode": "Clamp",
            "FrameDuration": 1 / 30,
            "MotionWeight": weight,
            "MotionClass": stats["motion_class"],
            "JointOrder": list(G1_29DOF_JOINT_NAMES),
            "AmpSchemaVersion": "g1_amp.v1",
            "SourceDataset": manifest["source_dataset"],
            "SourceMotion": str(csv.relative_to(ROOT)).replace("\\", "/"),
            "SourceFrameStart": start,
            "SourceFrameStop": stop,
            "Frames": frames.tolist(),
        }
        path.write_text(json.dumps(payload) + "\n")
        manifest["sources"][stem].update(path=payload["SourceMotion"], sha256=digest(csv))
        manifest["clips"][name] = dict(stats, sha256=digest(path), motion_weight=weight, frames=len(frames))
    manifest["duration_by_class_s"] = {
        cls: sum(s["duration_s"] for _, _, _, _, s in prepared if s["motion_class"] == cls) for cls in probabilities
    }
    (out / "_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

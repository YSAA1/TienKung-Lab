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

"""Generate 70D Z2 AMP experts from policy-order CSVs through AmpFeatureBuilder.

End-effector features are recomputed on the Z2 Isaac Lab articulation. This is
not a copy of upstream 64D no-feet experts and not a G1 tensor.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

from legged_lab.scripts.isaaclab_runtime_compat import (
    patch_missing_physx_material_attributes,
    patch_physx_backward_compatibility_setting,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--motion-dir",
    type=Path,
    default=Path("legged_lab/envs/z2/datasets/motion_source"),
)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--fps", type=float, default=None, help="Override per-clip CSV fps from the source manifest.")
parser.add_argument("--sim-dt", type=float, default=1.0 / 200.0)
parser.add_argument("--sim-device", default="cuda:0")
parser.add_argument("--max-frames", type=int, default=0, help="0 keeps every source frame.")
patch_physx_backward_compatibility_setting(AppLauncher)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app = AppLauncher(args_cli).app

import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

from legged_lab.assets.z2.amp_manifest import (  # noqa: E402
    amp_training_manifest,
    code_provenance,
    training_clip_record,
    validate_z2_source_manifest,
)
from legged_lab.assets.z2.constants import Z2_29DOF_JOINT_NAMES  # noqa: E402
from legged_lab.assets.z2.locomotion import Z2_LOCOMOTION  # noqa: E402
from legged_lab.assets.z2.schemas import (  # noqa: E402
    AMP_FRAME_DIM,
    AMP_HELD_OUT_MOTIONS,
    AMP_SCHEMA_VERSION,
    Z2_SOURCE_CSV_WIDTH,
    Z2_SOURCE_DATASET,
    amp_motion_class,
    amp_motion_weight,
)
from legged_lab.assets.z2.z2 import Z2_29DOF_WALK_POSE_DAMPED_PD_CFG  # noqa: E402
from legged_lab.locomotion.amp_features import AmpFeatureBuilder  # noqa: E402

patch_missing_physx_material_attributes()

ROOT = Path(__file__).resolve().parents[2]
MIN_KINEMATICS_RESPONSE = 1.0e-3


def _read_motion(path: Path) -> list[list[float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [[float(value) for value in row] for row in csv.reader(stream) if row]
    widths = {len(row) for row in rows}
    if widths != {Z2_SOURCE_CSV_WIDTH}:
        raise ValueError(f"{path} must have width {Z2_SOURCE_CSV_WIDTH}, got {sorted(widths)}")
    if args_cli.max_frames > 0:
        rows = rows[: args_cli.max_frames]
    if len(rows) < 3:
        raise ValueError(f"{path} needs at least three frames to form velocities and a transition")
    if not all(math.isfinite(value) for row in rows for value in row):
        raise ValueError(f"{path} contains non-finite values")
    return rows


def _selected_motions(motion_dir: Path, stems: list[str]) -> list[Path]:
    resolved = motion_dir if motion_dir.is_absolute() else ROOT / motion_dir
    paths = []
    for stem in sorted(stems):
        if stem in AMP_HELD_OUT_MOTIONS:
            raise ValueError(f"manifest motion {stem!r} is held out")
        path = resolved / f"{stem}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"declared AMP motion {stem!r} is missing at {path}")
        paths.append(path)
    return paths


def _write_state(robot, sim, row, joint_vel_row, motion_to_sim, device) -> None:
    x, y, z, qx, qy, qz, qw = row[:7]
    root_pose = torch.tensor([[x, y, z, qw, qx, qy, qz]], dtype=torch.float32, device=device)
    joint_pos = torch.tensor(
        [[row[7:][index] if index is not None else 0.0 for index in motion_to_sim]],
        dtype=torch.float32,
        device=device,
    )
    joint_vel = torch.tensor(
        [[joint_vel_row[index] if index is not None else 0.0 for index in motion_to_sim]],
        dtype=torch.float32,
        device=device,
    )
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=device))
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.write_data_to_sim()
    sim.forward()
    robot.update(args_cli.sim_dt)


def _kinematics_response(robot, sim, builder) -> float:
    neutral = robot.data.default_joint_pos.clone()
    zero_velocity = torch.zeros_like(neutral)
    knee_ids, _ = robot.find_joints(name_keys=["L_knee_joint", "R_knee_joint"], preserve_order=True)
    bent = neutral.clone()
    bent[:, knee_ids] += 0.4
    foot_features = []
    for joint_pos in (neutral, bent):
        robot.write_joint_state_to_sim(joint_pos, zero_velocity)
        robot.write_data_to_sim()
        sim.forward()
        robot.update(args_cli.sim_dt)
        foot_features.append(builder.compute()[0, -6:].clone())
    return (foot_features[1] - foot_features[0]).abs().max().item()


def _generate_motion(path, robot, sim, builder, motion_to_sim, device, fps, weight_stems):
    rows = _read_motion(path)
    dt = 1.0 / fps
    num_joints = len(Z2_29DOF_JOINT_NAMES)
    frames = []
    foot_travel = 0.0
    previous_foot = None
    for index in range(len(rows) - 1):
        row = rows[index]
        next_row = rows[index + 1]
        joint_vel_row = [(next_row[7 + j] - row[7 + j]) / dt for j in range(num_joints)]
        _write_state(robot, sim, row, joint_vel_row, motion_to_sim, device)
        state = builder.compute()
        if not torch.isfinite(state).all():
            raise ValueError(f"{path} produced non-finite AMP features at frame {index}")
        if state.shape[-1] != AMP_FRAME_DIM:
            raise RuntimeError(f"AMP frame width {state.shape[-1]} != {AMP_FRAME_DIM}")
        frame = state[0].detach().cpu().tolist()
        frames.append(frame)
        current_foot = frame[-6:]
        if previous_foot is not None:
            foot_travel += sum(abs(a - b) for a, b in zip(current_foot, previous_foot))
        previous_foot = current_foot
    return frames, {
        "motion_class": amp_motion_class(path.stem),
        "motion_weight": amp_motion_weight(path.stem, stems=weight_stems),
        "frames": len(frames),
        "frame_duration": dt,
        "fps": fps,
        "foot_feature_travel": foot_travel,
    }


def main() -> None:
    device = torch.device(args_cli.sim_device)
    source_dir = args_cli.motion_dir if args_cli.motion_dir.is_absolute() else ROOT / args_cli.motion_dir
    source_manifest_path = source_dir / "_manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"Z2 source manifest missing: {source_manifest_path}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    # The source manifest, not AMP_MOTION_CLASSES, defines this run's clip set.
    motions = _selected_motions(args_cli.motion_dir, [item["stem"] for item in source_manifest["motions"]])
    checked = validate_z2_source_manifest(source_manifest, source_dir)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=args_cli.sim_dt, device=args_cli.sim_device))
    robot = Articulation(Z2_29DOF_WALK_POSE_DAMPED_PD_CFG.replace(prim_path="/World/Z2"))
    sim.reset()
    sim_joint_names = list(robot.joint_names)
    motion_index = {name: index for index, name in enumerate(Z2_29DOF_JOINT_NAMES)}
    Z2_LOCOMOTION.validate_articulation(robot)
    extra = sorted(set(sim_joint_names) - set(Z2_29DOF_JOINT_NAMES) - set(Z2_LOCOMOTION.auxiliary_joint_names))
    if extra:
        raise RuntimeError(f"articulation has unexpected joints {extra}")
    motion_to_sim = [motion_index.get(name) for name in sim_joint_names]
    builder = AmpFeatureBuilder(robot, args_cli.sim_device, Z2_LOCOMOTION)
    kinematics_response = _kinematics_response(robot, sim, builder)
    if kinematics_response < MIN_KINEMATICS_RESPONSE:
        raise RuntimeError(
            f"foot features moved only {kinematics_response:.3e} when the knees were bent; "
            "forward kinematics were not refreshed before the AMP features were read"
        )
    output_dir = args_cli.output_dir if args_cli.output_dir.is_absolute() else ROOT / args_cli.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = {}
    clips = {}
    spawn = Z2_29DOF_WALK_POSE_DAMPED_PD_CFG.spawn
    usd_path = str(getattr(spawn, "usd_path", ""))
    if "assembly.usd" not in usd_path.replace("\\", "/"):
        raise RuntimeError(f"Z2 training plant must be original USD, got {usd_path}")
    converter = {
        "asset_mode": "upstream_usd",
        "usd_path": usd_path,
        "spawn_class": type(spawn).__name__,
        "urdf_converter_applied": False,
    }
    for path in motions:
        fps = float(args_cli.fps) if args_cli.fps is not None else checked[path.stem]["fps"]
        frames, stats = _generate_motion(
            path,
            robot,
            sim,
            builder,
            motion_to_sim,
            device,
            fps,
            weight_stems=tuple(item["stem"] for item in source_manifest["motions"]),
        )
        payload = {
            "LoopMode": "Wrap",
            "FrameDuration": stats["frame_duration"],
            "EnableCycleOffsetPosition": True,
            "EnableCycleOffsetRotation": True,
            "MotionWeight": stats["motion_weight"],
            "MotionClass": stats["motion_class"],
            "JointOrder": list(Z2_29DOF_JOINT_NAMES),
            "AmpSchemaVersion": AMP_SCHEMA_VERSION,
            "SourceMotion": str(path.relative_to(ROOT)),
            "SourceDataset": Z2_SOURCE_DATASET,
            "Frames": frames,
        }
        txt_path = output_dir / f"{path.stem}.txt"
        txt_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        generated[path.stem] = stats
        clips[path.stem] = training_clip_record(
            txt_path=txt_path,
            source_csv=path,
            source_csv_sha256=checked[path.stem]["csv_sha256"],
            source_raw_sha256=checked[path.stem]["source_sha256"],
            frames=stats["frames"],
            fps=fps,
            motion_class=stats["motion_class"],
            motion_weight=stats["motion_weight"],
            root=ROOT,
        )
        print(f"wrote {path.stem}: {stats['frames']} frames, fps={fps:.6f}", flush=True)
    manifest = amp_training_manifest(
        clips=clips,
        motions=generated,
        kinematics_response=kinematics_response,
        sim_dt=args_cli.sim_dt,
        max_frames=args_cli.max_frames,
        converter=converter,
        plant="Z2_29DOF_WALK_POSE_DAMPED_PD_CFG",
        joint_order=list(Z2_29DOF_JOINT_NAMES),
        held_out=list(AMP_HELD_OUT_MOTIONS),
        frame_dim=AMP_FRAME_DIM,
        provenance=code_provenance(ROOT),
    )
    (output_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Z2_AMP_EXPERT motions={len(generated)} frame_dim={AMP_FRAME_DIM} output={output_dir}", flush=True)


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        exit_code = 1
    import os
    import threading

    threading.Timer(30.0, os._exit, args=(exit_code,)).start()
    app.close()
    sys.exit(exit_code)

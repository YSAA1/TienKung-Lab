"""Generate the 66D T4 AMP expert set from the migrated motion CSVs.

The end-effector features are recomputed on the migrated IsaacLab articulation
through :class:`T4AmpFeatureBuilder`, the same builder the training env uses, so
expert and runtime transitions cannot drift apart.

The M0 human playback review is a hard gate: without ``--allow-pending-human-review``
the script refuses to write, and when the flag is used the manifest records the
expert set as provisional so it cannot be mistaken for a formal Stage E input.
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

parser = argparse.ArgumentParser(description="Generate T4 66D AMP expert files.")
parser.add_argument(
    "--motion-dir",
    type=Path,
    default=Path("legged_lab/envs/t4/datasets/motion_source"),
    help="Directory of root_xyz + root_quat_xyzw + q27 CSV clips.",
)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument(
    "--playback-audit",
    type=Path,
    default=Path("artifacts/eval/t4_motion_playback.json"),
    help="M0 playback audit that carries the human review verdicts.",
)
parser.add_argument("--fps", type=float, default=30.0)
parser.add_argument("--sim-dt", type=float, default=1.0 / 200.0)
parser.add_argument("--sim-device", default="cuda:0")
parser.add_argument(
    "--allow-pending-human-review",
    action="store_true",
    help="Write a provisional expert set while the M0 human playback review is pending.",
)
patch_physx_backward_compatibility_setting(AppLauncher)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app = AppLauncher(args_cli).app

import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

from legged_lab.assets.t4.constants import T4_JOINT_NAMES  # noqa: E402
from legged_lab.assets.t4.schemas import (  # noqa: E402
    AMP_FRAME_DIM,
    AMP_HELD_OUT_MOTIONS,
    AMP_MOTION_CLASSES,
    AMP_SCHEMA_VERSION,
    amp_motion_class,
    amp_motion_weight,
)
from legged_lab.assets.t4.t4 import T4_CFG  # noqa: E402
from legged_lab.envs.t4.amp_features import T4AmpFeatureBuilder  # noqa: E402

patch_missing_physx_material_attributes()

ROOT = Path(__file__).resolve().parents[2]
RAW_WIDTH = 7 + len(T4_JOINT_NAMES)
# Minimum foot-feature response to a deliberate knee change, used once to prove the
# kinematics read is live. A per-clip travel threshold cannot do this job: a static
# clip such as `t4_stand` legitimately produces almost no foot travel.
MIN_KINEMATICS_RESPONSE = 1.0e-3


def _read_motion(path: Path) -> list[list[float]]:
    with path.open(newline="") as stream:
        rows = [[float(value) for value in row] for row in csv.reader(stream)]
    widths = {len(row) for row in rows}
    if widths != {RAW_WIDTH}:
        raise ValueError(f"{path} must have width {RAW_WIDTH}, got {sorted(widths)}")
    if len(rows) < 3:
        raise ValueError(f"{path} needs at least three frames to form velocities and a transition")
    if not all(math.isfinite(value) for row in rows for value in row):
        raise ValueError(f"{path} contains non-finite values")
    return rows


def _human_review_status(audit_path: Path) -> dict[str, str]:
    resolved = audit_path if audit_path.is_absolute() else ROOT / audit_path
    if not resolved.is_file():
        return {}
    audit = json.loads(resolved.read_text())
    return {name: motion.get("human_playback_status", "missing") for name, motion in audit["motions"].items()}


def _selected_motions(motion_dir: Path) -> list[Path]:
    resolved = motion_dir if motion_dir.is_absolute() else ROOT / motion_dir
    paths = []
    for stem in sorted(AMP_MOTION_CLASSES):
        if stem in AMP_HELD_OUT_MOTIONS:
            continue
        path = resolved / f"{stem}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"declared AMP motion {stem!r} is missing at {path}")
        paths.append(path)
    return paths


def _write_state(
    robot: Articulation,
    sim: sim_utils.SimulationContext,
    row: list[float],
    joint_vel_row: list[float],
    motion_to_sim: list[int],
    device: torch.device,
) -> None:
    x, y, z, qx, qy, qz, qw = row[:7]
    root_pose = torch.tensor([[x, y, z, qw, qx, qy, qz]], dtype=torch.float32, device=device)
    joint_pos = torch.tensor([[row[7:][index] for index in motion_to_sim]], dtype=torch.float32, device=device)
    joint_vel = torch.tensor([[joint_vel_row[index] for index in motion_to_sim]], dtype=torch.float32, device=device)

    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=device))
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.write_data_to_sim()
    sim.forward()
    robot.update(args_cli.sim_dt)


def _kinematics_response(
    robot: Articulation,
    sim: sim_utils.SimulationContext,
    builder: T4AmpFeatureBuilder,
) -> float:
    """Return how far the foot features move when the knees are deliberately bent.

    A stale forward-kinematics read would silently freeze the end-effector part of
    every AMP frame, so this runs once before any motion is written.
    """
    neutral = robot.data.default_joint_pos.clone()
    zero_velocity = torch.zeros_like(neutral)
    knee_ids, _ = robot.find_joints(name_keys=["J_knee_l_pitch", "J_knee_r_pitch"], preserve_order=True)
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


def _generate_motion(
    path: Path,
    robot: Articulation,
    sim: sim_utils.SimulationContext,
    builder: T4AmpFeatureBuilder,
    motion_to_sim: list[int],
    device: torch.device,
) -> tuple[list[list[float]], dict]:
    rows = _read_motion(path)
    dt = 1.0 / args_cli.fps
    num_joints = len(T4_JOINT_NAMES)
    frames: list[list[float]] = []
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
        written = torch.tensor([[row[7:][i] for i in motion_to_sim]], dtype=torch.float32, device=device)
        applied = robot.data.joint_pos[:, :]
        if not torch.allclose(applied, written, atol=1.0e-4):
            raise RuntimeError(
                f"{path} frame {index}: simulator joint state does not match the written motion frame; "
                "kinematics were not refreshed before reading AMP features"
            )
        frame = state[0].detach().cpu().tolist()
        if len(frame) != AMP_FRAME_DIM:
            raise RuntimeError(f"AMP frame width {len(frame)} does not match schema width {AMP_FRAME_DIM}")
        frames.append(frame)

        current_foot = frame[-6:]
        if previous_foot is not None:
            foot_travel += sum(abs(a - b) for a, b in zip(current_foot, previous_foot))
        previous_foot = current_foot

    stats = {
        "motion_class": amp_motion_class(path.stem),
        "motion_weight": amp_motion_weight(path.stem),
        "frames": len(frames),
        "frame_duration": dt,
        "foot_feature_travel": foot_travel,
    }
    return frames, stats


def main() -> None:
    device = torch.device(args_cli.sim_device)
    review = _human_review_status(args_cli.playback_audit)
    motions = _selected_motions(args_cli.motion_dir)
    pending = sorted(path.stem for path in motions if review.get(path.stem, "missing") != "approved")
    if pending and not args_cli.allow_pending_human_review:
        raise SystemExit(
            "M0 human playback review is not approved for: "
            f"{pending}. Re-run with --allow-pending-human-review only to produce a provisional set."
        )

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=args_cli.sim_dt, device=args_cli.sim_device))
    robot = Articulation(T4_CFG.replace(prim_path="/World/T4"))
    sim.reset()

    sim_joint_names = list(robot.joint_names)
    motion_index = {name: index for index, name in enumerate(T4_JOINT_NAMES)}
    if set(sim_joint_names) != set(T4_JOINT_NAMES):
        raise RuntimeError(f"simulator joints {sorted(sim_joint_names)} do not match T4_JOINT_NAMES")
    motion_to_sim = [motion_index[name] for name in sim_joint_names]
    builder = T4AmpFeatureBuilder(robot, args_cli.sim_device)

    kinematics_response = _kinematics_response(robot, sim, builder)
    if kinematics_response < MIN_KINEMATICS_RESPONSE:
        raise RuntimeError(
            f"foot features moved only {kinematics_response:.3e} when the knees were bent; "
            "forward kinematics were not refreshed before the AMP features were read"
        )
    print(f"kinematics response {kinematics_response:.4f} m", flush=True)

    output_dir = args_cli.output_dir if args_cli.output_dir.is_absolute() else ROOT / args_cli.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = {}
    for path in motions:
        frames, stats = _generate_motion(path, robot, sim, builder, motion_to_sim, device)
        payload = {
            "LoopMode": "Wrap",
            "FrameDuration": stats["frame_duration"],
            "EnableCycleOffsetPosition": True,
            "EnableCycleOffsetRotation": True,
            "MotionWeight": stats["motion_weight"],
            "MotionClass": stats["motion_class"],
            "JointOrder": list(T4_JOINT_NAMES),
            "AmpSchemaVersion": AMP_SCHEMA_VERSION,
            "SourceMotion": str(path.relative_to(ROOT)),
            "Frames": frames,
        }
        (output_dir / f"{path.stem}.txt").write_text(json.dumps(payload) + "\n")
        generated[path.stem] = stats
        print(f"wrote {path.stem}: {stats['frames']} frames, weight {stats['motion_weight']:.4f}", flush=True)

    manifest = {
        "amp_schema_version": AMP_SCHEMA_VERSION,
        "frame_dim": AMP_FRAME_DIM,
        "transition_dim": 2 * AMP_FRAME_DIM,
        "joint_order": list(T4_JOINT_NAMES),
        "held_out_motions": list(AMP_HELD_OUT_MOTIONS),
        "human_playback_review": "approved" if not pending else "pending",
        "human_playback_pending": pending,
        "status": "formal" if not pending else "provisional",
        "kinematics_response": kinematics_response,
        "simulator": "isaaclab",
        "fps": args_cli.fps,
        "sim_dt": args_cli.sim_dt,
        "motions": generated,
    }
    (output_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"T4_AMP_EXPERT status={manifest['status']} motions={len(generated)} "
        f"frame_dim={AMP_FRAME_DIM} output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    # Isaac Sim shutdown can stall while an exception is unwinding, which would
    # otherwise swallow the failure; report it on stdout before closing the app.
    exit_code = 0
    try:
        main()
    except BaseException:
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        exit_code = 1
    app.close()
    sys.exit(exit_code)

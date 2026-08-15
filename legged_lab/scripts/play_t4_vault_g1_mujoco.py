"""Local MuJoCo sim2sim of the G1 vault-mimic teacher (150D tracking actor).

Applies the Isaac PD / contact contract from
``legged_lab.assets.t4.mujoco_sim2sim`` (position servo, kd assign, mu=1
friction, URDF foot boxes + sphere hands). Diagnostic only; not a G1 gate.

Default: open the MuJoCo viewer and keep replaying the policy.
Fallen or end-of-motion auto-resets so you can watch many attempts.

    PYTHONPATH=. python legged_lab/scripts/play_t4_vault_g1_mujoco.py

Optional video dump (one-shot):

    PYTHONPATH=. python legged_lab/scripts/play_t4_vault_g1_mujoco.py --record
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

import mujoco

from legged_lab.assets.t4.constants import T4_JOINT_NAMES
from legged_lab.assets.t4.mujoco_sim2sim import (
    apply_isaac_contact_friction,
    apply_isaac_pd,
    isaac_pd_gains,
    rewrite_vault_mjcf,
)

# IsaacLab URDF import order from artifacts/eval/t4_motion_playback_smoke.json.
# G1 obs/action are in this order, not T4_JOINT_NAMES.
ISAAC_RUNTIME_JOINT_NAMES: tuple[str, ...] = (
    "J_arm_l_01",
    "J_arm_r_01",
    "J_arm_l_02",
    "J_arm_r_02",
    "J_arm_l_03",
    "J_arm_r_03",
    "J_waist_yaw",
    "J_arm_l_04",
    "J_arm_r_04",
    "J_hip_l_pitch",
    "J_hip_r_pitch",
    "J_arm_l_05",
    "J_arm_r_05",
    "J_hip_l_roll",
    "J_hip_r_roll",
    "J_arm_l_06",
    "J_arm_r_06",
    "J_hip_l_yaw",
    "J_hip_r_yaw",
    "J_arm_l_07",
    "J_arm_r_07",
    "J_knee_l_pitch",
    "J_knee_r_pitch",
    "J_ankle_l_pitch",
    "J_ankle_r_pitch",
    "J_ankle_l_roll",
    "J_ankle_r_roll",
)
from legged_lab.assets.t4.tracking_motion import body_indices, load_t4_tracking_motion
from legged_lab.assets.t4.vault_contract import (
    T4_VAULT_ACTION_SCALE,
    T4_VAULT_BOX_POS,
    T4_VAULT_BOX_SIZE,
    T4_VAULT_MOTION_FILE,
    T4_VAULT_WRIST_TERMINATION_THRESHOLD,
)

# Standing pose copied from t4.py so this file stays Isaac-free.
T4_STANDING_JOINT_POS = dict.fromkeys(T4_JOINT_NAMES, 0.0)
T4_STANDING_JOINT_POS.update(
    {
        "J_arm_l_01": 0.20,
        "J_arm_l_02": 0.13,
        "J_arm_l_04": -0.43,
        "J_arm_r_01": 0.20,
        "J_arm_r_02": -0.13,
        "J_arm_r_04": -0.43,
        "J_hip_l_pitch": -0.20,
        "J_knee_l_pitch": 0.42,
        "J_ankle_l_pitch": -0.24,
        "J_hip_r_pitch": -0.20,
        "J_knee_r_pitch": 0.42,
        "J_ankle_r_pitch": -0.24,
    }
)

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "t4"
MJCF_PATH = ASSET_DIR / "mjcf" / "t4_std.xml"
PHYSICS_DT = 0.005
DECIMATION = 4
CLIP_OBS = 100.0
CLIP_ACTIONS = 100.0
G1_OBS_DIM = 150


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_inv(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    q_vec = np.array([x, y, z])
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(q_vec, v) * w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def quat_to_mat_first_two_cols(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    r00 = 1 - 2 * (y * y + z * z)
    r10 = 2 * (x * y + z * w)
    r20 = 2 * (x * z - y * w)
    r01 = 2 * (x * y - z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r21 = 2 * (y * z + x * w)
    # IsaacLab does mat[..., :2].reshape(..., 6) on a (3, 2) slice, which is
    # row-major: R00, R01, R10, R11, R20, R21.
    return np.array([r00, r01, r10, r11, r20, r21], dtype=np.float32)


def load_g1_actor(checkpoint: Path) -> tuple[torch.nn.Module, torch.nn.Module | None]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rsl_rl"))
    from rsl_rl.modules.normalizer import EmpiricalNormalization

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    actor_state = {k[len("actor.") :]: v for k, v in state.items() if k.startswith("actor.")}
    num_obs = int(actor_state["0.weight"].shape[1])
    if num_obs != G1_OBS_DIM:
        raise RuntimeError(f"G1 actor expects {num_obs}D obs, contract is {G1_OBS_DIM}D")
    dims = [num_obs, 512, 256, 128, len(T4_JOINT_NAMES)]
    layers: list[torch.nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*layers)
    actor.load_state_dict(actor_state)
    actor.eval()
    normalizer = None
    if "obs_norm_state_dict" in ckpt:
        normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8)
        normalizer.load_state_dict(ckpt["obs_norm_state_dict"])
        normalizer.eval()
    return actor, normalizer


class VaultG1MujocoRunner:
    def __init__(self, checkpoint: Path):
        self.actor, self.normalizer = load_g1_actor(checkpoint)
        self.motion = load_t4_tracking_motion(T4_VAULT_MOTION_FILE)
        self.trunk_i = body_indices(self.motion, ("Trunk",))[0]
        self.wrist_i = body_indices(self.motion, ("AL7", "AR7"))
        xml = rewrite_vault_mjcf(
            MJCF_PATH.read_text(),
            meshdir=ASSET_DIR / "meshes",
            box_pos=T4_VAULT_BOX_POS,
            box_size=T4_VAULT_BOX_SIZE,
        )
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.model.opt.timestep = PHYSICS_DT
        apply_isaac_pd(self.model, ISAAC_RUNTIME_JOINT_NAMES)
        apply_isaac_contact_friction(self.model)
        self.data = mujoco.MjData(self.model)
        self.qpos_adr = np.array([self.model.jnt_qposadr[self.model.joint(name).id] for name in ISAAC_RUNTIME_JOINT_NAMES])
        self.qvel_adr = np.array([self.model.jnt_dofadr[self.model.joint(name).id] for name in ISAAC_RUNTIME_JOINT_NAMES])
        self.ctrl_ids = np.array([self.model.actuator(f"M{name[1:]}").id for name in ISAAC_RUNTIME_JOINT_NAMES])
        self.trunk_body = self.model.body("Trunk").id
        self.wrist_bodies = (self.model.body("AL7").id, self.model.body("AR7").id)
        gains = np.array([isaac_pd_gains(name) for name in ISAAC_RUNTIME_JOINT_NAMES])
        self.kp, self.kd, self.effort_limit = gains[:, 0], gains[:, 1], gains[:, 2]
        self.default_pos = np.array([T4_STANDING_JOINT_POS[name] for name in ISAAC_RUNTIME_JOINT_NAMES])
        self._t4_to_isaac = np.array([T4_JOINT_NAMES.index(name) for name in ISAAC_RUNTIME_JOINT_NAMES])
        ground = self.model.geom("ground")
        hip = int(self.model.jnt_dofadr[self.model.joint("J_hip_l_pitch").id])
        print(
            "[sim2sim] Isaac PD/contact applied: "
            f"hip_pitch_kd={getattr(self.model, 'dof_damping')[hip]:.3f} (must be 4.000) "
            f"frictionloss={self.model.dof_frictionloss[hip]:.3f} "
            f"ground_condim={int(self.model.geom_condim[int(ground.id)])} "
            f"mu={self.model.geom_friction[int(ground.id), 0]:.3f} "
            "hands=half_sphere feet=urdf_boxes",
            flush=True,
        )
        self.reset()

    def reset(self, rng: np.random.Generator | None = None) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.time_step = 0
        self._apply_reference_state(0)
        if rng is not None:
            # Same residual noise Play/eval still applies after sampling_strategy="zero".
            self.data.qpos[0] += rng.uniform(-0.05, 0.05)
            self.data.qpos[1] += rng.uniform(-0.05, 0.05)
            self.data.qpos[2] += rng.uniform(-0.01, 0.01)
            delta = quat_from_euler_xyz(
                rng.uniform(-0.1, 0.1),
                rng.uniform(-0.1, 0.1),
                rng.uniform(-0.2, 0.2),
            )
            self.data.qpos[3:7] = quat_mul(delta, self.data.qpos[3:7])
            self.data.qpos[self.qpos_adr] += rng.uniform(-0.1, 0.1, size=len(ISAAC_RUNTIME_JOINT_NAMES))
            self.data.qvel[:3] += rng.uniform(-0.5, 0.5, size=3)
            self.data.qvel[3:6] += rng.uniform(-0.52, 0.52, size=3)
        mujoco.mj_forward(self.model, self.data)
        self.action = np.zeros(len(ISAAC_RUNTIME_JOINT_NAMES), dtype=np.float32)

    def _apply_reference_state(self, frame: int) -> None:
        trunk = self.motion["body_pos_w"][frame, self.trunk_i]
        quat = self.motion["body_quat_w"][frame, self.trunk_i]
        self.data.qpos[:3] = trunk
        self.data.qpos[3:7] = quat
        self.data.qpos[self.qpos_adr] = self.motion["joint_pos"][frame, self._t4_to_isaac]
        self.data.qvel[:] = 0.0
        self.data.qvel[self.qvel_adr] = self.motion["joint_vel"][frame, self._t4_to_isaac]
        self.data.qvel[:3] = self.motion["body_lin_vel_w"][frame, self.trunk_i]
        self.data.qvel[3:6] = self.motion["body_ang_vel_w"][frame, self.trunk_i]

    def wrist_z_error(self) -> float:
        frame = min(self.time_step, self.motion["num_frames"] - 1)
        errors = []
        for local_i, body_id in zip(self.wrist_i, self.wrist_bodies):
            ref_z = float(self.motion["body_pos_w"][frame, local_i, 2])
            robot_z = float(self.data.xpos[body_id, 2])
            errors.append(abs(ref_z - robot_z))
        return max(errors)

    def observe(self) -> np.ndarray:
        frame = min(self.time_step, self.motion["num_frames"] - 1)
        ref_q = self.motion["joint_pos"][frame, self._t4_to_isaac]
        ref_dq = self.motion["joint_vel"][frame, self._t4_to_isaac]
        ref_anchor_p = self.motion["body_pos_w"][frame, self.trunk_i]
        ref_anchor_q = self.motion["body_quat_w"][frame, self.trunk_i]
        robot_p = self.data.xpos[self.trunk_body].copy()
        robot_q = self.data.xquat[self.trunk_body].copy()
        pos_b = quat_rotate_inverse(robot_q, ref_anchor_p - robot_p)
        rel_q = quat_mul(quat_inv(robot_q), ref_anchor_q)
        ori_b = quat_to_mat_first_two_cols(rel_q)
        lin_vel_b = quat_rotate_inverse(robot_q, self.data.qvel[:3])
        ang_vel_b = self.data.qvel[3:6]
        joint_pos = self.data.qpos[self.qpos_adr] - self.default_pos
        joint_vel = self.data.qvel[self.qvel_adr]
        obs = np.concatenate(
            [ref_q, ref_dq, pos_b, ori_b, lin_vel_b, ang_vel_b, joint_pos, joint_vel, self.action]
        ).astype(np.float32)
        if obs.shape[0] != G1_OBS_DIM:
            raise RuntimeError(f"built {obs.shape[0]}D obs, expected {G1_OBS_DIM}")
        return np.clip(obs, -CLIP_OBS, CLIP_OBS)

    def step(self) -> None:
        obs = torch.from_numpy(self.observe()).unsqueeze(0)
        if self.normalizer is not None:
            with torch.no_grad():
                obs = self.normalizer(obs)
        with torch.no_grad():
            action = self.actor(obs).squeeze(0).numpy()
        self.action = np.clip(action, -CLIP_ACTIONS, CLIP_ACTIONS).astype(np.float32)
        self.data.ctrl[self.ctrl_ids] = self.default_pos + T4_VAULT_ACTION_SCALE * self.action
        for _ in range(DECIMATION):
            mujoco.mj_step(self.model, self.data)
        self.time_step += 1


def _write_mp4(output: Path, frames: list[np.ndarray], fps: float) -> None:
    import subprocess

    height, width = frames[0].shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert proc.stdin is not None
    for frame in frames:
        proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {output}")


def record(runner: VaultG1MujocoRunner, output: Path, steps: int, fps: float) -> dict:
    step_dt = PHYSICS_DT * DECIMATION
    record_every = max(1, int(round((1.0 / fps) / step_dt)))
    renderer = mujoco.Renderer(runner.model, height=480, width=640)
    cameras = []
    for elevation, yaw_offset, distance in ((-12.0, 140.0, 3.0), (-6.0, -90.0, 3.0)):
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.elevation = elevation
        cam.distance = distance
        cameras.append((cam, yaw_offset))

    xs, wrist_err = [], []
    frames = []
    for i in range(steps):
        runner.step()
        xs.append(float(runner.data.qpos[0]))
        wrist_err.append(runner.wrist_z_error())
        if i % record_every == 0:
            qpos = runner.data.qpos
            w, x, y, z = qpos[3:7]
            yaw = math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
            views = []
            for cam, yaw_offset in cameras:
                cam.azimuth = yaw + yaw_offset
                cam.lookat[:] = [qpos[0], qpos[1], 0.7]
                renderer.update_scene(runner.data, camera=cam)
                views.append(renderer.render().copy())
            frames.append(np.hstack(views))

    _write_mp4(output, frames, fps)
    start_wrist = wrist_err[0] if wrist_err else None
    return {
        "video": str(output),
        "steps": steps,
        "start_x": xs[0] if xs else None,
        "max_x": max(xs) if xs else None,
        "final_x": xs[-1] if xs else None,
        "start_wrist_z_error": start_wrist,
        "max_wrist_z_error": max(wrist_err) if wrist_err else None,
        "wrist_fail_at_reset": bool(start_wrist is not None and start_wrist > T4_VAULT_WRIST_TERMINATION_THRESHOLD),
        "crossed_box": bool(xs and max(xs) >= 0.88),
    }


def _fallen(runner: VaultG1MujocoRunner) -> bool:
    height = float(runner.data.qpos[2])
    gravity_z = quat_rotate_inverse(runner.data.qpos[3:7], np.array([0.0, 0.0, -1.0]))[2]
    return height < 0.35 or gravity_z > -0.4


def watch(runner: VaultG1MujocoRunner, auto_reset: bool, seconds: float, jitter: bool, seed: int) -> None:
    import mujoco.viewer

    step_dt = PHYSICS_DT * DECIMATION
    max_steps = max(1, int(round(seconds / step_dt)))
    attempt = 1
    rng = np.random.default_rng(seed) if jitter else None
    print(
        "MuJoCo viewer: close the window to quit. Auto-reset on fall / timeout. "
        + ("Each reset jitters like Isaac eval, so attempts differ." if jitter else "Deterministic: every reset is the same clip."),
        flush=True,
    )
    runner.reset(rng)
    with mujoco.viewer.launch_passive(runner.model, runner.data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.distance = 3.2
        viewer.cam.elevation = -12.0
        viewer.cam.azimuth = 140.0
        while viewer.is_running():
            t0 = time.time()
            runner.step()
            viewer.cam.lookat[:] = [float(runner.data.qpos[0]), float(runner.data.qpos[1]), 0.7]
            viewer.sync()
            done = runner.time_step >= max_steps or _fallen(runner)
            if done:
                print(
                    f"attempt {attempt}: x={runner.data.qpos[0]:+.2f} "
                    f"z={runner.data.qpos[2]:.2f} steps={runner.time_step} "
                    f"fallen={_fallen(runner)} wrist_z={runner.wrist_z_error():.2f}",
                    flush=True,
                )
                if not auto_reset:
                    time.sleep(1.5)
                    break
                attempt += 1
                runner.reset(rng)
            leftover = step_dt - (time.time() - t0)
            if leftover > 0:
                time.sleep(leftover)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="logs/t4_vault_mimic/2026-08-13_09-30-13/model_28500.pt",
    )
    parser.add_argument("--record", action="store_true", help="Write one MP4 and exit instead of opening the viewer.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/eval/g1_sim2sim/t4_vault_g1_m28500.mp4"))
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/eval/g1_sim2sim/t4_vault_g1_m28500_mujoco.json"))
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--no-auto-reset", action="store_true", help="Stop after the first fall/timeout.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Exact frame-0 reset every time. Default jitters like Isaac eval, so repeats differ.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(f"missing G1 checkpoint: {ckpt}")
    runner = VaultG1MujocoRunner(ckpt)
    if args.record:
        reset_wrist = runner.wrist_z_error()
        steps = max(1, int(round(args.seconds / (PHYSICS_DT * DECIMATION))))
        metrics = record(runner, args.output, steps, args.fps)
        metrics.update(
            {
                "evaluator": "t4_vault_g1_mujoco_sim2sim_v1",
                "checkpoint": str(ckpt.resolve()),
                "reset_wrist_z_error": reset_wrist,
                "wrist_threshold": T4_VAULT_WRIST_TERMINATION_THRESHOLD,
                "caveat": "MuJoCo diagnostic. G1 is 150D tracking; not the 1155D loco teacher.",
            }
        )
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics, indent=2), flush=True)
        return
    watch(
        runner,
        auto_reset=not args.no_auto_reset,
        seconds=args.seconds,
        jitter=not args.deterministic,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

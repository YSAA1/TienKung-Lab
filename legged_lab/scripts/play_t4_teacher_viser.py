# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""MuJoCo sim2sim playback of the T4 teacher policy.

Loads an rsl_rl AMP checkpoint (``model_*.pt``), rebuilds the actor MLP, and
runs it in MuJoCo on flat ground. The observation contract mirrors
:mod:`legged_lab.envs.t4.t4_env` exactly: 10-frame proprio history (oldest
first) + 195-dim flat-ground height scan, all obs scales 1.0.

Usage:
    python legged_lab/scripts/play_t4_teacher_viser.py \\
        --checkpoint logs/t4_loco_teacher/<run>/model_3500.pt
    python legged_lab/scripts/play_t4_teacher_viser.py \\
        --checkpoint logs/t4_loco_teacher/<run>/model_3500.pt \\
        --record artifacts/eval/t4_play.mp4 --duration 12
    python legged_lab/scripts/play_t4_teacher_viser.py \
        --checkpoint logs/t4_loco_teacher/<run>/model_24999.pt \
        --terrain loco --viewer mujoco
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import torch

from legged_lab.assets.t4.constants import T4_JOINT_NAMES
from legged_lab.assets.t4.mujoco_sim2sim import apply_isaac_contact_friction, apply_isaac_pd, isaac_pd_gains
from legged_lab.assets.t4.schemas import (
    PROPRIO_FRAME_DIM,
    PROPRIO_HISTORY_LENGTH,
    TEACHER_SCAN_CLIP,
    TEACHER_SCAN_DIM,
    TEACHER_SCAN_HEIGHT_OFFSET,
    TEACHER_SCAN_FORWARD_RANGE,
    TEACHER_SCAN_LATERAL_RANGE,
    TEACHER_SCAN_SHAPE,
)
# Keep this MuJoCo-only entrypoint independent of IsaacLab.  Importing the
# ``legged_lab.terrains`` package would execute its IsaacLab-heavy __init__;
# load the pure layout truth module directly instead.
_LAYOUT_PATH = Path(__file__).resolve().parents[1] / "terrains" / "hurdle_layout.py"
_LAYOUT_SPEC = importlib.util.spec_from_file_location("t4_hurdle_layout", _LAYOUT_PATH)
assert _LAYOUT_SPEC and _LAYOUT_SPEC.loader
_LAYOUT = importlib.util.module_from_spec(_LAYOUT_SPEC)
_LAYOUT_SPEC.loader.exec_module(_LAYOUT)
hurdle_bar_aabbs = _LAYOUT.hurdle_bar_aabbs
hurdle_bar_height = _LAYOUT.hurdle_bar_height
hurdle_ring_half_widths = _LAYOUT.hurdle_ring_half_widths
# Standing pose copied from legged_lab/assets/t4/t4.py::T4_STANDING_JOINT_POS;
# importing t4.py directly would pull in isaaclab, which this script must not need.
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
URDF_PATH = ASSET_DIR / "urdf" / "t4_std.urdf"

PHYSICS_DT = 0.005
DECIMATION = 4
ACTION_SCALE = 0.25
CLIP_ACTIONS = 100.0
CLIP_OBS = 100.0

GAIT_CYCLE = 0.85
PHASE_OFFSET = np.array([0.38, 0.88])
PHASE_RATIO = np.array([0.38, 0.38])
STANDING_COMMAND_THRESHOLD = 0.1

INIT_ROOT_Z = 0.85


def teacher_scan_local_points() -> np.ndarray:
    """Return scan points in IsaacLab GridPatternCfg(ordering="xy") order."""
    xs = np.linspace(*TEACHER_SCAN_FORWARD_RANGE, TEACHER_SCAN_SHAPE[0])
    ys = np.linspace(*TEACHER_SCAN_LATERAL_RANGE, TEACHER_SCAN_SHAPE[1])
    return np.asarray([(x, y) for y in ys for x in xs], dtype=np.float64)


def load_actor(checkpoint_path: str) -> torch.nn.Module:
    # The checkpoint pickle references classes from the in-repo rsl_rl package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rsl_rl"))
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    actor_state = {k[len("actor.") :]: v for k, v in state.items() if k.startswith("actor.")}
    num_obs = actor_state["0.weight"].shape[1]
    expected = PROPRIO_FRAME_DIM * PROPRIO_HISTORY_LENGTH + TEACHER_SCAN_DIM
    if num_obs != expected:
        raise RuntimeError(f"checkpoint actor expects {num_obs} obs, schema says {expected}")
    dims = [num_obs, 512, 256, 128, len(T4_JOINT_NAMES)]
    layers: list[torch.nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*layers)
    actor.load_state_dict(actor_state)
    actor.eval()
    return actor


def quat_rotate_inverse_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    q_vec = np.array([x, y, z])
    a = v * (2.0 * w**2 - 1.0)
    b = np.cross(q_vec, v) * w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


class T4MujocoRunner:
    def __init__(self, checkpoint: str, terrain: str = "flat", difficulty: float = 0.85):
        self.actor = load_actor(checkpoint)
        self.terrain = terrain
        self.bar_aabbs = []
        if terrain == "loco":
            # Keep teacher and depth-student comparisons on the exact same
            # Stage-E course, including the 25/28/30 cm hurdle sequence.
            from legged_lab.scripts.sim2sim_t4_depth_student import (
                CourseNavigator,
                build_model_xml,
                course_waypoints_from_model,
            )

            self.model = mujoco.MjModel.from_xml_path(build_model_xml(course="loco"))
            self.navigator = CourseNavigator(course_waypoints_from_model(self.model), cruise_vx=0.6)
        elif terrain == "hurdles":
            height = hurdle_bar_height(difficulty)
            widths = hurdle_ring_half_widths(1.1)
            # hurdle_layout uses tile-local coordinates centered at (4, 4).
            # MuJoCo uses the spawn point as world origin, so translate AABBs.
            for lo, hi in hurdle_bar_aabbs(widths, height):
                self.bar_aabbs.append(
                    ((lo[0] - 4.0, lo[1] - 4.0, lo[2]), (hi[0] - 4.0, hi[1] - 4.0, hi[2]))
                )

            xml = MJCF_PATH.read_text()
            xml = xml.replace('meshdir="../meshes/"', f'meshdir="{ASSET_DIR / "meshes"}"')
            geoms = []
            for i, (lo, hi) in enumerate(self.bar_aabbs):
                center = [(lo[j] + hi[j]) / 2.0 for j in range(3)]
                half = [(hi[j] - lo[j]) / 2.0 for j in range(3)]
                geoms.append(
                    f'<geom name="hurdle_{i}" type="box" pos="{center[0]} {center[1]} {center[2]}" '
                    f'size="{half[0]} {half[1]} {half[2]}" rgba="0.85 0.25 0.12 1"/>'
                )
            xml = xml.replace("</worldbody>", "".join(geoms) + "</worldbody>")
            self.model = mujoco.MjModel.from_xml_string(xml)
        else:
            self.model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
        if terrain != "loco":
            self.navigator = None
        self.model.opt.timestep = PHYSICS_DT
        apply_isaac_pd(self.model)
        apply_isaac_contact_friction(self.model)
        self.data = mujoco.MjData(self.model)

        # T4-order <-> MuJoCo mapping, resolved by name so XML ordering is irrelevant.
        self.qpos_adr = np.array(
            [self.model.jnt_qposadr[self.model.joint(name).id] for name in T4_JOINT_NAMES]
        )
        self.qvel_adr = np.array(
            [self.model.jnt_dofadr[self.model.joint(name).id] for name in T4_JOINT_NAMES]
        )
        self.ctrl_ids = np.array([self.model.actuator(f"M{name[1:]}").id for name in T4_JOINT_NAMES])

        gains = np.array([isaac_pd_gains(name) for name in T4_JOINT_NAMES])
        self.kp, self.kd, self.effort_limit = gains[:, 0], gains[:, 1], gains[:, 2]
        self.default_pos = np.array([T4_STANDING_JOINT_POS[name] for name in T4_JOINT_NAMES])
        self._ray_geomgroup = np.ones(6, dtype=np.uint8)
        self._robot_body_id = self.model.body("Trunk").id

        self.command = np.zeros(3)
        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = [0.0, 0.0, INIT_ROOT_Z]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[self.qpos_adr] = self.default_pos
        mujoco.mj_forward(self.model, self.data)

        self.action = np.zeros(len(T4_JOINT_NAMES), dtype=np.float32)
        self.gait_time = 0.0
        self.gait_phase = PHASE_OFFSET % 1.0
        if self.navigator is not None:
            self.navigator.reset()
        frame = self._proprio_frame()
        self.history: deque[np.ndarray] = deque(
            [frame.copy() for _ in range(PROPRIO_HISTORY_LENGTH)], maxlen=PROPRIO_HISTORY_LENGTH
        )

    def _proprio_frame(self) -> np.ndarray:
        quat = self.data.qpos[3:7]  # wxyz, world orientation of the floating base
        ang_vel = self.data.qvel[3:6]  # free-joint angular velocity is body-frame
        projected_gravity = quat_rotate_inverse_wxyz(quat, np.array([0.0, 0.0, -1.0]))
        joint_pos = self.data.qpos[self.qpos_adr] - self.default_pos
        joint_vel = self.data.qvel[self.qvel_adr]
        frame = np.concatenate(
            [
                ang_vel,
                projected_gravity,
                self.command,
                joint_pos,
                joint_vel,
                self.action,
                np.sin(2 * np.pi * self.gait_phase),
                np.cos(2 * np.pi * self.gait_phase),
                PHASE_RATIO,
            ]
        ).astype(np.float32)
        assert frame.shape[0] == PROPRIO_FRAME_DIM
        return frame

    def _height_scan(self) -> np.ndarray:
        baseline = self.data.qpos[2] - TEACHER_SCAN_HEIGHT_OFFSET
        if self.terrain == "loco":
            yaw = _root_yaw_wxyz(self.data.qpos[3:7])
            c, s = np.cos(yaw), np.sin(yaw)
            values = []
            for x, y in teacher_scan_local_points():
                wx = self.data.qpos[0] + c * x - s * y
                wy = self.data.qpos[1] + s * x + c * y
                geomid = np.zeros(1, dtype=np.int32)
                distance = mujoco.mj_ray(
                    self.model,
                    self.data,
                    np.array([wx, wy, 5.0]),
                    np.array([0.0, 0.0, -1.0]),
                    self._ray_geomgroup,
                    1,
                    self._robot_body_id,
                    geomid,
                )
                terrain_z = 0.0 if distance < 0.0 else 5.0 - distance
                values.append(np.clip(baseline - terrain_z, *TEACHER_SCAN_CLIP))
            return np.asarray(values, dtype=np.float32)
        if self.terrain != "hurdles":
            return np.full(TEACHER_SCAN_DIM, np.clip(baseline, *TEACHER_SCAN_CLIP), dtype=np.float32)
        yaw = _root_yaw_wxyz(self.data.qpos[3:7])
        c, s = np.cos(yaw), np.sin(yaw)
        values = []
        for x, y in teacher_scan_local_points():
            wx = self.data.qpos[0] + c * x - s * y
            wy = self.data.qpos[1] + s * x + c * y
            bar_z = 0.0
            for (x0, y0, z0), (x1, y1, z1) in self.bar_aabbs:
                if x0 <= wx <= x1 and y0 <= wy <= y1:
                    bar_z = max(bar_z, z1)
            values.append(np.clip(baseline - bar_z, *TEACHER_SCAN_CLIP))
        return np.asarray(values, dtype=np.float32)

    def hurdle_contact_count(self) -> int:
        if self.terrain != "hurdles":
            return 0
        count = 0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            a = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
            b = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
            count += int(a.startswith("hurdle_") or b.startswith("hurdle_"))
        return count

    def observe(self) -> np.ndarray:
        self.history.append(self._proprio_frame())
        obs = np.concatenate(list(self.history) + [self._height_scan()])
        return np.clip(obs, -CLIP_OBS, CLIP_OBS)

    def step(self) -> None:
        if self.navigator is not None:
            self.command[:] = self.navigator.command(
                self.data.qpos[:2], _root_yaw_wxyz(self.data.qpos[3:7])
            )
        obs = self.observe()
        with torch.no_grad():
            action = self.actor(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
        self.action = np.clip(action, -CLIP_ACTIONS, CLIP_ACTIONS).astype(np.float32)
        target = self.default_pos + ACTION_SCALE * self.action

        self.data.ctrl[self.ctrl_ids] = target
        for _ in range(DECIMATION):
            mujoco.mj_step(self.model, self.data)

        # Fixed clock, frozen on standing commands (matches T4GaitCfg mode="fixed_clock").
        if np.linalg.norm(self.command[:2]) > STANDING_COMMAND_THRESHOLD:
            self.gait_time += (PHYSICS_DT * DECIMATION) / GAIT_CYCLE
        self.gait_phase = (self.gait_time + PHASE_OFFSET) % 1.0

    @property
    def fallen(self) -> bool:
        gravity_z = quat_rotate_inverse_wxyz(self.data.qpos[3:7], np.array([0.0, 0.0, -1.0]))[2]
        return self.data.qpos[2] < 0.35 or gravity_z > -0.5


def _root_yaw_wxyz(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = quat_wxyz
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def record_video(
    runner: T4MujocoRunner,
    output: Path,
    duration_s: float,
    fps: float,
    vx: float,
    vy: float,
    wz: float,
) -> None:
    import imageio.v2 as imageio

    runner.command[:] = [vx, vy, wz]
    step_dt = PHYSICS_DT * DECIMATION
    n_steps = max(1, int(round(duration_s / step_dt)))
    record_every = max(1, int(round((1.0 / fps) / step_dt)))

    width, height = 640, 480
    renderer = mujoco.Renderer(runner.model, height=height, width=width)
    cameras = []
    for elevation, yaw_offset, distance in ((-10.0, 145.0, 2.6), (-4.0, -90.0, 2.6)):
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.elevation = elevation
        camera.distance = distance
        cameras.append((camera, yaw_offset))

    frames: list[np.ndarray] = []
    for i in range(n_steps):
        runner.step()
        if i % record_every == 0:
            qpos = runner.data.qpos
            yaw = math.degrees(_root_yaw_wxyz(qpos[3:7]))
            views = []
            for camera, yaw_offset in cameras:
                camera.azimuth = yaw + yaw_offset
                camera.lookat[:] = [qpos[0], qpos[1], 0.55]
                renderer.update_scene(runner.data, camera=camera)
                views.append(renderer.render().copy())
            frames.append(np.hstack(views))
        if runner.fallen:
            break

    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(output, frames, fps=fps, quality=8, macro_block_size=1)
    print(f"wrote {output} ({len(frames)} frames, fallen={runner.fallen}, t={runner.data.time:.2f}s)")


def run_native_viewer(runner: T4MujocoRunner, vx: float, vy: float, wz: float) -> None:
    """Run the same sim2sim loop with MuJoCo's built-in viewer.

    This keeps the teacher entrypoint usable in lightweight environments where
    the optional Viser/yourdfpy visualization packages are not installed.
    """
    import mujoco.viewer

    runner.command[:] = [vx, vy, wz]
    step_dt = PHYSICS_DT * DECIMATION
    with mujoco.viewer.launch_passive(runner.model, runner.data) as viewer:
        while viewer.is_running():
            t0 = time.time()
            runner.step()
            if runner.fallen:
                runner.reset()
                runner.command[:] = [vx, vy, wz]
            viewer.sync()
            remain = step_dt - (time.time() - t0)
            if remain > 0:
                time.sleep(remain)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="rsl_rl model_*.pt checkpoint path")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--viewer",
        choices=["auto", "mujoco", "viser"],
        default="auto",
        help="Viewer backend. MuJoCo renders the complete physics scene, including terrain and obstacles.",
    )
    parser.add_argument(
        "--terrain",
        choices=["flat", "hurdles", "loco"],
        default="flat",
        help="MuJoCo scene; loco reuses the student Stage-E course (25/28/30 cm hurdles).",
    )
    parser.add_argument(
        "--difficulty",
        type=float,
        default=1.0,
        help="Difficulty for the legacy hurdle-ring scene; 1.0 produces the 30 cm bar.",
    )
    parser.add_argument("--record", type=Path, default=None, help="Write an MP4 instead of opening the viser GUI.")
    parser.add_argument("--duration", type=float, default=12.0, help="Recorded seconds.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--vx", type=float, default=0.6)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.0)
    args = parser.parse_args()

    runner = T4MujocoRunner(args.checkpoint, terrain=args.terrain, difficulty=args.difficulty)
    if args.record is not None:
        record_video(runner, args.record, args.duration, args.fps, args.vx, args.vy, args.wz)
        return
    if args.viewer == "mujoco":
        run_native_viewer(runner, args.vx, args.vy, args.wz)
        return

    try:
        import viser
        import yourdfpy
        from viser.extras import ViserUrdf
    except ModuleNotFoundError as exc:
        if args.viewer == "viser":
            raise
        missing = exc.name or "optional viewer dependency"
        print(f"[INFO] {missing} is not installed; using MuJoCo native viewer.")
        run_native_viewer(runner, args.vx, args.vy, args.wz)
        return

    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/ground", width=50.0, height=20.0, cell_size=0.5)
    root_frame = server.scene.add_frame("/root", show_axes=False)
    urdf_model = yourdfpy.URDF.load(str(URDF_PATH))
    viser_urdf = ViserUrdf(server, urdf_model, root_node_name="/root")
    urdf_joint_names = list(urdf_model.actuated_joint_names)
    urdf_from_t4 = np.array([urdf_joint_names.index(name) for name in T4_JOINT_NAMES])

    with server.gui.add_folder("Command"):
        vx = server.gui.add_slider("vx (m/s)", min=-0.6, max=1.0, step=0.05, initial_value=0.6)
        vy = server.gui.add_slider("vy (m/s)", min=-0.5, max=0.5, step=0.05, initial_value=0.0)
        wz = server.gui.add_slider("wz (rad/s)", min=-1.57, max=1.57, step=0.05, initial_value=0.0)
    with server.gui.add_folder("Sim"):
        paused = server.gui.add_checkbox("Pause", initial_value=False)
        reset_btn = server.gui.add_button("Reset")
        status = server.gui.add_text("Status", initial_value="", disabled=True)

    reset_requested = {"flag": False}

    @reset_btn.on_click
    def _(_) -> None:
        reset_requested["flag"] = True

    step_dt = PHYSICS_DT * DECIMATION
    print(f"viser viewer: http://localhost:{args.port}")
    while True:
        t0 = time.time()
        if reset_requested["flag"]:
            reset_requested["flag"] = False
            runner.reset()
        if not paused.value:
            runner.command[:] = [vx.value, vy.value, wz.value]
            runner.step()
            if runner.fallen:
                status.value = "fell -> auto reset"
                runner.reset()

        qpos = runner.data.qpos
        root_frame.position = tuple(qpos[:3])
        root_frame.wxyz = tuple(qpos[3:7])
        cfg = np.zeros(len(urdf_joint_names))
        cfg[urdf_from_t4] = qpos[runner.qpos_adr]
        viser_urdf.update_cfg(cfg)

        speed = float(np.linalg.norm(runner.data.qvel[:2]))
        if not runner.fallen:
            status.value = f"t={runner.data.time:6.1f}s  base speed={speed:4.2f} m/s  z={qpos[2]:.2f}"

        remain = step_dt - (time.time() - t0)
        if remain > 0:
            time.sleep(remain)


if __name__ == "__main__":
    main()

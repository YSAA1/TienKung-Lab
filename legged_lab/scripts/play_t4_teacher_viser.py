# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""MuJoCo sim2sim playback of the T4 teacher policy with a viser web viewer.

Loads an rsl_rl AMP checkpoint (``model_*.pt``), rebuilds the actor MLP, and
runs it in MuJoCo on flat ground. The observation contract mirrors
:mod:`legged_lab.envs.t4.t4_env` exactly: 10-frame proprio history (oldest
first) + 195-dim flat-ground height scan, all obs scales 1.0. Commands are set
interactively from the viser GUI.

Usage:
    python legged_lab/scripts/play_t4_teacher_viser.py \
        --checkpoint artifacts/checkpoints/prov5_model_4500.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import torch
import viser
import yourdfpy
from viser.extras import ViserUrdf

from legged_lab.assets.t4.constants import T4_JOINT_NAMES
from legged_lab.assets.t4.schemas import (
    PROPRIO_FRAME_DIM,
    PROPRIO_HISTORY_LENGTH,
    TEACHER_SCAN_CLIP,
    TEACHER_SCAN_DIM,
    TEACHER_SCAN_HEIGHT_OFFSET,
)
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

# Per-joint PD gains and torque limits, copied from legged_lab/assets/t4/t4.py.
def _gains_for(joint: str) -> tuple[float, float, float]:
    if joint.startswith("J_arm"):
        idx = int(joint[-2:])
        kp = 20.0 if idx <= 5 else 10.0
        effort = 36.0 if idx <= 4 else 12.0
        return kp, 1.0, effort
    if joint == "J_waist_yaw":
        return 50.0, 2.0, 120.0
    if "hip" in joint:
        return (100.0, 4.0, 130.0) if joint.endswith("pitch") else (50.0, 2.0, 120.0)
    if "knee" in joint:
        return 100.0, 4.0, 130.0
    if "ankle" in joint:
        return (80.0, 4.0, 72.0) if joint.endswith("pitch") else (20.0, 1.0, 72.0)
    raise ValueError(f"unknown joint {joint}")


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
    def __init__(self, checkpoint: str):
        self.actor = load_actor(checkpoint)

        self.model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
        self.model.opt.timestep = PHYSICS_DT
        self.data = mujoco.MjData(self.model)

        # T4-order <-> MuJoCo mapping, resolved by name so XML ordering is irrelevant.
        self.qpos_adr = np.array(
            [self.model.jnt_qposadr[self.model.joint(name).id] for name in T4_JOINT_NAMES]
        )
        self.qvel_adr = np.array(
            [self.model.jnt_dofadr[self.model.joint(name).id] for name in T4_JOINT_NAMES]
        )
        self.ctrl_ids = np.array([self.model.actuator(f"M{name[1:]}").id for name in T4_JOINT_NAMES])

        gains = np.array([_gains_for(name) for name in T4_JOINT_NAMES])
        self.kp, self.kd, self.effort_limit = gains[:, 0], gains[:, 1], gains[:, 2]
        self.default_pos = np.array([T4_STANDING_JOINT_POS[name] for name in T4_JOINT_NAMES])

        # Mirror PhysX's implicit joint drive: turn each torque motor into a position
        # servo (gain kp) and fold kd into dof damping, which MuJoCo's Euler
        # integrator handles implicitly. Explicit PD torque at 200 Hz was unstable.
        for i, name in enumerate(T4_JOINT_NAMES):
            a = self.ctrl_ids[i]
            self.model.actuator_gaintype[a] = mujoco.mjtGain.mjGAIN_FIXED
            self.model.actuator_gainprm[a, :] = 0.0
            self.model.actuator_gainprm[a, 0] = self.kp[i]
            self.model.actuator_biastype[a] = mujoco.mjtBias.mjBIAS_AFFINE
            self.model.actuator_biasprm[a, :] = 0.0
            self.model.actuator_biasprm[a, 1] = -self.kp[i]
            self.model.actuator_ctrllimited[a] = 0
            self.model.actuator_forcelimited[a] = 1
            self.model.actuator_forcerange[a] = (-self.effort_limit[i], self.effort_limit[i])
            self.model.dof_damping[self.qvel_adr[i]] += self.kd[i]

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
        # Flat-ground playback: every scan cell sees ground z = 0.
        value = float(np.clip(self.data.qpos[2] - TEACHER_SCAN_HEIGHT_OFFSET, *TEACHER_SCAN_CLIP))
        return np.full(TEACHER_SCAN_DIM, value, dtype=np.float32)

    def observe(self) -> np.ndarray:
        self.history.append(self._proprio_frame())
        obs = np.concatenate(list(self.history) + [self._height_scan()])
        return np.clip(obs, -CLIP_OBS, CLIP_OBS)

    def step(self) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="rsl_rl model_*.pt checkpoint path")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    runner = T4MujocoRunner(args.checkpoint)

    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/ground", width=20.0, height=20.0, cell_size=0.5)
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

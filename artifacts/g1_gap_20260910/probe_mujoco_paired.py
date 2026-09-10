"""MuJoCo side of the G1 Isaac-vs-MuJoCo paired rollout probe.

Mirrors `probe_isaac_paired.py` exactly: flat ground, fixed [vx, 0, 0]
command, nominal reset, deterministic actor, and one 1997D obs vector per
policy step. Unlike the interactive play script, the proprio/scan histories
start zeroed to match the training-time CircularBuffer.reset() convention on
the Isaac side, so the first divergent channel is dynamics, not pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from legged_lab.assets.t4.schemas import PROPRIO_HISTORY_LENGTH  # noqa: E402
from legged_lab.scripts.play_t4_sparse_teacher_mujoco import (  # noqa: E402
    ACTION_SCALE,
    CLIP_ACTIONS,
    DECIMATION,
    GAIT_CYCLE,
    PHASE_OFFSET,
    PHYSICS_DT,
    STANDING_COMMAND_THRESHOLD,
    T4SparseTeacherMujocoRunner,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vx", type=float, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    runner = T4SparseTeacherMujocoRunner(args.checkpoint, difficulty=0.0, terrain="flat")
    runner.command[:] = [args.vx, 0.0, 0.0]

    # Isaac training reset convention: (a) observation histories are pre-filled
    # with the first frame, and (b) the FIRST frame carries gait phase [0, 0] --
    # the env's gait_phase tensor starts at zero and only picks up the phase
    # offset inside the first step(). runner.reset() prefills with phase
    # [0.38, 0.88], so rebuild the initial frame with zero phase.
    runner.gait_time = 0.0
    runner.gait_phase = np.zeros_like(runner.gait_phase)
    runner.action[:] = 0.0
    frame0 = runner._proprio_frame()
    runner.history = deque(
        [frame0.copy() for _ in range(PROPRIO_HISTORY_LENGTH)], maxlen=PROPRIO_HISTORY_LENGTH
    )
    scan0 = runner._height_scan()
    runner.scan_history = deque(
        [scan0.copy() for _ in range(runner.scan_history_length)], maxlen=runner.scan_history_length
    )

    n_joints = len(runner.robot.joint_names)
    obs_log = np.zeros((args.steps, runner.actor_obs_dim), dtype=np.float32)
    action_log = np.zeros((args.steps, n_joints), dtype=np.float32)
    ctrl_log = np.zeros((args.steps, n_joints), dtype=np.float32)
    root_log = np.zeros((args.steps, 13), dtype=np.float32)
    joint_log = np.zeros((args.steps, 2, n_joints), dtype=np.float32)
    contact_log = np.zeros((args.steps, 2), dtype=np.float32)

    for t in range(args.steps):
        obs = runner.observe()
        obs_log[t] = obs
        with torch.no_grad():
            action = runner.actor(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
        clipped = np.clip(action, -CLIP_ACTIONS, CLIP_ACTIONS).astype(np.float32)
        action_log[t] = clipped
        runner.action = clipped
        ctrl = runner.default_pos + ACTION_SCALE * clipped
        ctrl_log[t] = ctrl
        runner.data.ctrl[runner.ctrl_ids] = ctrl
        for _ in range(DECIMATION):
            mujoco.mj_step(runner.model, runner.data)
        if np.linalg.norm(runner.command[:2]) > STANDING_COMMAND_THRESHOLD:
            runner.gait_time += (PHYSICS_DT * DECIMATION) / GAIT_CYCLE
        runner.gait_phase = (runner.gait_time + PHASE_OFFSET) % 1.0

        qpos = runner.data.qpos
        qvel = runner.data.qvel
        root_log[t] = [*qpos[:7], *qvel[:6]]
        joint_log[t, 0] = qpos[runner.qpos_adr]
        joint_log[t, 1] = qvel[runner.qvel_adr]
        contact_log[t] = runner._feet_contact()

    meta = {
        "sim": "mujoco",
        "mujoco_version": mujoco.__version__,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "vx": args.vx,
        "steps": args.steps,
        "physics_dt": PHYSICS_DT,
        "decimation": DECIMATION,
        "action_scale": ACTION_SCALE,
        "joint_names": list(runner.robot.joint_names),
        "init_root_z": float(runner.robot.init_root_z),
        "history_zeroed_at_reset": True,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        obs=obs_log,
        action=action_log,
        ctrl=ctrl_log,
        root=root_log,
        joint=joint_log,
        contact=contact_log,
    )
    meta_path = output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] wrote {output} and {meta_path}")


if __name__ == "__main__":
    main()

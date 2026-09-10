"""Forensic trace of the MuJoCo stepping-stone transition failure.

Runs one deterministic episode on stepping stones and prints, around the
platform-to-first-stones transition, the root state, per-foot world position
and contact, against the known stone centers of the play lattice.
"""

from __future__ import annotations

import argparse
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
    LANE_COUNT,
    PHASE_OFFSET,
    PHYSICS_DT,
    STANDING_COMMAND_THRESHOLD,
    T4SparseTeacherMujocoRunner,
    sparse_course_layout,
)
from legged_lab.scripts.play_t4_teacher_viser import quat_rotate_inverse_wxyz  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--difficulty", type=float, default=1.0)
    parser.add_argument("--vx", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=260)
    parser.add_argument("--from-step", type=int, default=90)
    args = parser.parse_args()

    runner = T4SparseTeacherMujocoRunner(
        args.checkpoint, difficulty=args.difficulty, terrain="stepping_stones"
    )
    layout = runner.layout
    stones = [g for g in layout["geoms"] if g["kind"] == "stone"]
    first_row_x = min(g["pos"][0] for g in stones)
    print(
        f"[LAYOUT] stone w={layout['stone_width']:.3f} gap={layout['stone_gap']:.3f} "
        f"pitch={layout['stone_width']+layout['stone_gap']:.3f} h={layout['stone_height']:.3f} "
        f"first_row_x={first_row_x:.3f} platform_edge_x=1.0"
    )

    runner.reset()
    runner.command[:] = [args.vx, 0.0, 0.0]
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

    foot_bodies = [runner.model.body("left_ankle_roll_link").id, runner.model.body("right_ankle_roll_link").id]
    # Fall back to ankle roll bodies if the roll link is absent.
    try:
        foot_bodies
    except Exception:
        pass

    def foot_world(body_id):
        return runner.data.xpos[body_id]

    for t in range(args.steps):
        obs = runner.observe()
        with torch.no_grad():
            action = runner.actor(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
        clipped = np.clip(action, -CLIP_ACTIONS, CLIP_ACTIONS).astype(np.float32)
        runner.action = clipped
        runner.data.ctrl[runner.ctrl_ids] = runner.default_pos + ACTION_SCALE * clipped
        for _ in range(DECIMATION):
            mujoco.mj_step(runner.model, runner.data)
        if np.linalg.norm(runner.command[:2]) > STANDING_COMMAND_THRESHOLD:
            runner.gait_time += (PHYSICS_DT * DECIMATION) / GAIT_CYCLE
        runner.gait_phase = (runner.gait_time + PHASE_OFFSET) % 1.0

        if t >= args.from_step and (t % 4 == 0 or runner.fallen):
            q = runner.data.qpos
            yaw = np.arctan2(2 * (q[3] * q[6] + q[4] * q[5]), 1 - 2 * (q[5] ** 2 + q[6] ** 2))
            con = runner._feet_contact()
            lf, rf = foot_world(foot_bodies[0]), foot_world(foot_bodies[1])
            # nearest stone center for each foot
            def nearest_stone(fx, fy):
                best, bd = None, 1e9
                for g in stones:
                    dx, dy = fx - g["pos"][0], fy - g["pos"][1]
                    d = (dx * dx + dy * dy) ** 0.5
                    if d < bd:
                        best, bd = g, d
                return best, bd

            bl, dl = nearest_stone(lf[0], lf[1])
            br, dr = nearest_stone(rf[0], rf[1])
            print(
                f"t={t:3d} root=({q[0]:+.2f},{q[1]:+.2f},{q[2]:.3f}) yaw={np.degrees(yaw):+5.1f} "
                f"Lfoot=({lf[0]:+.2f},{lf[1]:+.2f},{lf[2]:.3f}) c={int(con[0])} dStone={dl:.3f} "
                f"Rfoot=({rf[0]:+.2f},{rf[1]:+.2f},{rf[2]:.3f}) c={int(con[1])} dStone={dr:.3f}"
            )
        if runner.fallen:
            print(f"[FALLEN] at t={t}")
            break


if __name__ == "__main__":
    main()

"""Full MuJoCo sim2sim diagnostic for the T4 hurdle terrain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from legged_lab.scripts.play_t4_teacher_viser import T4MujocoRunner, record_video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--difficulty", type=float, default=0.85)
    parser.add_argument("--vx", type=float, default=0.7)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runner = T4MujocoRunner(args.checkpoint, terrain="hurdles", difficulty=args.difficulty)
    runner.command[:] = [args.vx, 0.0, 0.0]
    record_video(runner, args.video, min(args.timeout, 20.0), 30.0, args.vx, 0.0, 0.0)

    max_steps = round(args.timeout / (0.005 * 4))
    progress_successes = 0
    strict_successes = 0
    rows = []
    for episode in range(args.episodes):
        runner.reset()
        runner.command[:] = [args.vx, 0.0, 0.0]
        contacts = 0
        for _ in range(max_steps):
            runner.step()
            contacts += runner.hurdle_contact_count()
            if runner.fallen:
                break
        progress = float(runner.data.qpos[0])
        fallen = bool(runner.fallen)
        progress_success = (not fallen) and progress >= 3.0
        strict_success = progress_success and contacts == 0
        progress_successes += int(progress_success)
        strict_successes += int(strict_success)
        rows.append(
            {
                "episode": episode,
                "progress_success": progress_success,
                "strict_zero_contact_success": strict_success,
                "progress_m": progress,
                "fallen": fallen,
                "hurdle_contact_events": contacts,
                "sim_time_s": float(runner.data.time),
            }
        )

    result = {
        "evaluator": "t4_hurdle_mujoco_sim2sim_v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "terrain": "hurdles",
        "difficulty": args.difficulty,
        "command_vx": args.vx,
        "requested_episodes": args.episodes,
        "completed_episodes": len(rows),
        "progress_successes": progress_successes,
        "progress_success_rate": progress_successes / max(1, len(rows)),
        "strict_zero_contact_successes": strict_successes,
        "strict_zero_contact_success_rate": strict_successes / max(1, len(rows)),
        "rows": rows,
        "caveat": "MuJoCo sim2sim diagnostic; not IsaacLab final gate. Hurdle geometry and teacher scan are reconstructed from hurdle_layout/schema.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

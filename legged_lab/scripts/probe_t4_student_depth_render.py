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

"""Headless probe: prove T4 sparse student tiled RTX depth refreshes.

Records raw camera depth checksums after each control step. Deploy distill
renders at 50 Hz (one tiled capture per control step). ``--skip_rtx`` is only
for debugging the scheduler; do not use it as a warp fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from isaaclab.app import AppLauncher

from legged_lab.scripts.isaaclab_runtime_compat import (
    patch_missing_physx_material_attributes,
    patch_physx_backward_compatibility_setting,
)

parser = argparse.ArgumentParser(description="Probe T4 student tiled RTX depth refresh phase.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=24)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--skip_rtx",
    action="store_true",
    help="Do not mark RTX ticks on last_rtx_sim_step.",
)
patch_physx_backward_compatibility_setting(AppLauncher)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = True
if not getattr(args_cli, "device", None):
    args_cli.device = "cuda:0"
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from legged_lab.envs.t4.depth_student_env import T4LocoSparseDepthStudentEnvCfg  # noqa: E402
from legged_lab.locomotion.depth_env import LightLPDepthDistillationEnv  # noqa: E402
from legged_lab.utils.env_utils.scene import rtx_render_interval_physics  # noqa: E402

patch_missing_physx_material_attributes()


def _frame_stats(depth: torch.Tensor) -> dict:
    flat = depth.detach().reshape(-1).float().cpu()
    payload = flat.numpy().tobytes()
    finite = torch.isfinite(flat)
    finite_vals = flat[finite]
    return {
        "checksum": hashlib.sha1(payload).hexdigest()[:16],
        "mean": float(finite_vals.mean()) if finite_vals.numel() else None,
        "finite_frac": float(finite.float().mean()),
        "min": float(finite_vals.min()) if finite_vals.numel() else None,
        "max": float(finite_vals.max()) if finite_vals.numel() else None,
    }


def main() -> dict:
    env_cfg = T4LocoSparseDepthStudentEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.device = args_cli.device
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.seed = 0
    # Keep the S12 curriculum mesh (default 10x20). Do not flatten to
    # 1 x num_envs: curriculum proportions break, and collection would not
    # match the 1024-env training terrain.

    env = LightLPDepthDistillationEnv(env_cfg, headless=True)
    env.schedule_rtx_render = not bool(args_cli.skip_rtx)
    period = env._depth_camera_update_period()
    interval = rtx_render_interval_physics(period, env.physics_dt)
    student_obs, initial_extras = env.get_observations()
    teacher_obs = initial_extras["observations"]["teacher"]
    observation_dims = {
        "student": int(student_obs.shape[-1]),
        "teacher": int(teacher_obs.shape[-1]),
        "proprio": int(env.observation_layout.proprio_dim),
        "actions": int(env.num_actions),
    }
    assert observation_dims == {"student": 3168, "teacher": 1937, "proprio": 96, "actions": 27}

    records = []
    step_times = []
    prev_last_rtx = int(env.last_rtx_sim_step)
    for step in range(int(args_cli.steps)):
        t0 = time.perf_counter()
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        env.step(actions)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
        raw = env.depth_camera.data.output["distance_to_image_plane"]
        stats = _frame_stats(raw)
        policy_depth_stats = _frame_stats(env.depth_history[:, -1])
        last_rtx = int(env.last_rtx_sim_step)
        records.append(
            {
                "control_step": step,
                "sim_step_counter": int(env.sim_step_counter),
                "last_rtx_sim_step": last_rtx,
                "rendered": last_rtx != prev_last_rtx,
                "depth_update_counter": int(env.depth_update_counter),
                "policy_depth": policy_depth_stats,
                **stats,
            }
        )
        prev_last_rtx = last_rtx

    warmup = max(2, int(round(period / max(float(env.step_dt), 1.0e-6))))
    rendered_rows = [row for row in records[warmup:] if row["rendered"]]
    idle_rows = [row for row in records[warmup:] if not row["rendered"]]
    checksum_changed_on_idle = any(
        records[i]["checksum"] != records[i - 1]["checksum"]
        for i, row in enumerate(records)
        if i > 0 and i >= warmup and not row["rendered"]
    )
    checksum_changed_on_render = any(
        records[i]["checksum"] != records[i - 1]["checksum"]
        for i, row in enumerate(records)
        if i > 0 and i >= warmup and row["rendered"]
    )
    policy_depth_changes = sum(
        records[i]["policy_depth"]["checksum"] != records[i - 1]["policy_depth"]["checksum"]
        for i in range(max(1, warmup), len(records))
    )
    finite_ok = all(row["finite_frac"] > 0.25 for row in records[warmup:]) if records[warmup:] else False
    expected_render_steps = [step for step in range(int(args_cli.steps)) if (step + 1) * env.cfg.sim.decimation % interval == 0]
    actual_render_steps = [row["control_step"] for row in records if row["rendered"]]
    every_control_step = interval <= int(env.cfg.sim.decimation)

    if args_cli.skip_rtx:
        ok = not any(row["rendered"] for row in records)
        reason = "skip_rtx: depth ticks not marked"
    else:
        schedule_ok = actual_render_steps == expected_render_steps and len(rendered_rows) > 0
        idle_ok = (not idle_rows) or (not checksum_changed_on_idle)
        render_change_ok = every_control_step or checksum_changed_on_render
        ok = (
            finite_ok
            and schedule_ok
            and idle_ok
            and render_change_ok
            and policy_depth_changes > 0
            and bool(env._has_rtx_sensors())
            and not bool(env._has_warp_depth_camera())
        )
        reason = (
            "live tiled RTX depth: finite frames, render ticks match update_period, "
            "not warp raycast"
        )

    collection_s = float(sum(step_times))
    collection_median_step_s = float(sorted(step_times)[len(step_times) // 2]) if step_times else None
    collection_target_s = 3.0
    collection_fallback_s = 4.0
    collection_meets_target = collection_s <= collection_target_s
    collection_meets_fallback = collection_s <= collection_fallback_s

    result = {
        "ok": bool(ok),
        "time": time.time(),
        "runtime_class": f"{type(env).__module__}.{type(env).__name__}",
        "reason": reason,
        "backend": "tiled_rtx",
        "skip_rtx": bool(args_cli.skip_rtx),
        "num_envs": int(args_cli.num_envs),
        "steps": int(args_cli.steps),
        "observation_dims": observation_dims,
        "policy_depth_changes": policy_depth_changes,
        "update_period": period,
        "physics_dt": float(env.physics_dt),
        "render_interval_physics": interval,
        "hold_steps": int(getattr(env, "student_depth_hold_steps", 1)),
        "expected_render_control_steps": expected_render_steps,
        "actual_render_control_steps": actual_render_steps,
        "checksum_changed_on_idle": bool(checksum_changed_on_idle),
        "checksum_changed_on_render": bool(checksum_changed_on_render),
        "finite_ok": bool(finite_ok),
        "idle_count": len(idle_rows),
        "render_count": len(rendered_rows),
        "collection_s": collection_s,
        "collection_median_step_s": collection_median_step_s,
        "collection_target_s": collection_target_s,
        "collection_fallback_s": collection_fallback_s,
        "collection_meets_target": bool(collection_meets_target),
        "collection_meets_fallback": bool(collection_meets_fallback),
        "has_rtx_sensors": bool(env._has_rtx_sensors()),
        "has_warp_depth_camera": bool(env._has_warp_depth_camera()),
        "records": records,
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in result if k != "records"}, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    import os
    import threading

    code = 0
    try:
        payload = main()
        code = 0 if payload["ok"] else 1
    except BaseException:
        import traceback

        traceback.print_exc()
        code = 1
    finally:
        # Match the existing Isaac portability probe: Kit shutdown can hang
        # after RTX capture; retain the result/exit code and bound its cleanup.
        threading.Timer(30.0, os._exit, args=(code,)).start()
        simulation_app.close()
        os._exit(code)

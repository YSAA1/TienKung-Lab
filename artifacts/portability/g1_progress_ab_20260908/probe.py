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

"""Small Isaac integration gate for the full-speed curriculum A/B."""

import argparse
import json
import os
import threading
from pathlib import Path

from isaaclab.app import AppLauncher

from legged_lab.scripts.isaaclab_runtime_compat import (
    patch_missing_physx_material_attributes,
    patch_physx_backward_compatibility_setting,
)

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
patch_physx_backward_compatibility_setting(AppLauncher)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

from legged_lab.envs import *  # noqa: F401,F403
from legged_lab.utils import task_registry


def main():
    patch_missing_physx_material_attributes()
    cfg, _ = task_registry.get_cfgs("g1_loco_teacher")
    cfg.scene.num_envs = 32
    cfg.scene.seed = 42
    cfg.device = cfg.sim.device = args.device
    cfg.progress_monitor_enabled = True
    env = task_registry.get_task_class("g1_loco_teacher")(cfg, True)
    ids = torch.arange(env.num_envs, device=env.device)
    env.reset(ids)
    assert env.progress_monitor.steps.sum() == 0
    original_levels = env.scene.terrain.terrain_levels.clone()
    samples = {}
    for level in (0, 4, 9):
        env.scene.terrain.terrain_levels[:] = level
        torch.manual_seed(742)
        speeds = []
        for _ in range(20):
            env._resample_terrain_aware_commands(ids)
            speeds.append(env._sparse_command[env._sparse_command_active, 0].clone())
        speeds = torch.cat(speeds)
        assert speeds.numel() and speeds.min() >= 0.6 and speeds.max() <= 2.0
        samples[str(level)] = speeds.tolist()
    assert samples["0"] == samples["4"] == samples["9"]
    env.scene.terrain.terrain_levels.copy_(original_levels)
    env.reset(ids)
    observed = set()
    for _ in range(100):
        _, reward, _, info = env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device))
        assert torch.isfinite(reward).all()
        observed.update(key for key in info.get("log", {}) if key.startswith("Progress/"))
    assert observed
    env.reset(ids[:2])
    assert env.progress_monitor.steps[:2].sum() == 0
    assert env.progress_monitor.delta_history[:, :2].abs().sum() == 0
    assert torch.equal(env.progress_monitor.spawn[:2], env.robot.data.root_pos_w[:2, :2])
    # Exercise actual environment wiring using synthetic history. Suppress only
    # random level revisits for this algebra gate; no policy capability claim.
    env.cfg.random_level_reset_fraction = 0.0
    env.command_generator.command[:, 0] = 0.7
    env.command_generator.command[:, 1] = 0
    env.episode_tracking_steps[:] = 100
    env.episode_tracking_sum[:] = 80
    env.episode_path_length[:] = 5
    env.episode_max_radial_dist[:] = 0.1
    env.pit_fall_buf[:] = False
    rates = {}
    for mode in ("path_length", "max_radial"):
        env.cfg.lightlp_promotion_distance = mode
        logs = env.update_terrain_levels(ids)
        rates[mode] = float(logs["Curriculum/promotion_rate"])
    assert rates == {"path_length": 1.0, "max_radial": 0.0}
    Path(args.output).write_text(
        json.dumps(
            {
                "ok": True,
                "num_envs": env.num_envs,
                "speed_scale": cfg.sparse_command_min_speed_scale,
                "command_samples": samples,
                "synthetic_loop_promotion": rates,
                "progress_tags": sorted(observed),
                "partial_reset_pass": True,
                "note": "100 zero-action steps and synthetic history validate wiring, not locomotion capability",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    code = 0
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        code = 1
    finally:
        threading.Timer(20, os._exit, args=(code,)).start()
        app.close()
        os._exit(code)

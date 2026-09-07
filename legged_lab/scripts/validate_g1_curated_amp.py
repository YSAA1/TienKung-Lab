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

"""Validate every prepared AMP frame on the official IsaacLab G1 articulation."""

import argparse
import hashlib
import json
import os
import threading
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

from legged_lab.scripts.isaaclab_runtime_compat import (
    patch_missing_physx_material_attributes,
    patch_physx_backward_compatibility_setting,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-dir", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
patch_physx_backward_compatibility_setting(AppLauncher)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation

from legged_lab.assets.unitree_g1.constants import G1_29DOF_JOINT_NAMES
from legged_lab.assets.unitree_g1.g1 import G1_29DOF_CFG
from legged_lab.assets.unitree_g1.locomotion import G1_LOCOMOTION
from legged_lab.envs.g1.amp_features import G1AmpFeatureBuilder


def main():
    patch_missing_physx_material_attributes()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device))
    robot = Articulation(G1_29DOF_CFG.replace(prim_path="/World/G1"))
    sim.reset()
    G1_LOCOMOTION.validate_articulation(robot)
    builder = G1AmpFeatureBuilder(robot, args.device)
    order = [G1_29DOF_JOINT_NAMES.index(name) for name in robot.joint_names]
    result = {"robot_joint_names": robot.joint_names, "frame_dim": 70, "clips": {}, "passed": False}
    manifest = json.loads((args.data_dir / "_manifest.json").read_text())
    for name, declared in manifest["clips"].items():
        path = args.data_dir / f"{name}.txt"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == declared["sha256"]
        raw = json.loads(path.read_text())
        assert raw["JointOrder"] == list(G1_29DOF_JOINT_NAMES)
        frames = torch.tensor(raw["Frames"], dtype=torch.float32, device=args.device)
        error = 0.0
        for frame in frames:
            robot.write_joint_state_to_sim(frame[:29][order].unsqueeze(0), frame[29:58][order].unsqueeze(0))
            robot.write_data_to_sim()
            sim.forward()
            robot.update(0.005)
            actual = builder.compute()[0]
            assert torch.isfinite(actual).all()
            error = max(error, float((actual - frame).abs().max()))
        assert error < 2e-4, (name, error)
        result["clips"][name] = {"sha256": digest, "frames": len(frames), "max_feature_error": error}
        print(name, len(frames), error, flush=True)
    result["passed"] = True
    result["total_frames"] = sum(item["frames"] for item in result["clips"].values())
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    code = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        code = 1
    finally:
        threading.Timer(30, os._exit, args=(code,)).start()
        app.close()
        os._exit(code)

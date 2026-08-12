"""Headless IsaacLab smoke test for the migrated T4 articulation."""

import json
from pathlib import Path

from isaaclab.app import AppLauncher

from legged_lab.scripts.isaaclab_runtime_compat import (
    patch_missing_physx_material_attributes,
    patch_physx_backward_compatibility_setting,
)


patch_physx_backward_compatibility_setting(AppLauncher)
app = AppLauncher(headless=True).app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

from legged_lab.assets.t4.constants import T4_JOINT_NAMES  # noqa: E402
from legged_lab.assets.t4.t4 import T4_CFG  # noqa: E402

patch_missing_physx_material_attributes()


def main() -> None:
    stage_path = Path("/tmp/t4_asset_smoke_stage.txt")
    stage_path.write_text("simulation_context\n")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.002, device="cpu"))
    stage_path.write_text("articulation_constructor\n")
    robot = Articulation(T4_CFG.replace(prim_path="/World/T4"))
    stage_path.write_text("simulation_reset\n")
    sim.reset()
    stage_path.write_text("validation\n")
    print(f"T4_SMOKE num_joints={robot.num_joints} num_bodies={robot.num_bodies}", flush=True)
    print(f"T4_SMOKE joint_names={robot.joint_names}", flush=True)
    if robot.num_joints != len(T4_JOINT_NAMES):
        raise RuntimeError(f"expected {len(T4_JOINT_NAMES)} joints, got {robot.num_joints}")
    if tuple(robot.joint_names) != T4_JOINT_NAMES:
        raise RuntimeError("IsaacLab joint order does not match T4_JOINT_NAMES")
    Path("/tmp/t4_asset_smoke_result.json").write_text(
        json.dumps(
            {
                "num_joints": robot.num_joints,
                "num_bodies": robot.num_bodies,
                "joint_names": robot.joint_names,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()

"""Headless IsaacLab smoke for the T4 articulation and required collision prims."""

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
from pxr import Usd, UsdPhysics  # noqa: E402

from legged_lab.assets.t4.constants import T4_JOINT_NAMES  # noqa: E402
from legged_lab.assets.t4.t4 import T4_CFG  # noqa: E402

patch_missing_physx_material_attributes()

REQUIRED_COLLISION_BODIES = ("Trunk", "Shank_Left", "Shank_Right")


def _collision_prim_paths(stage, root_path: str) -> list[str]:
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return []
    prim_range = Usd.PrimRange(root, Usd.TraverseInstanceProxies())
    return [str(prim.GetPath()) for prim in prim_range if prim.HasAPI(UsdPhysics.CollisionAPI)]


def main() -> None:
    stage_path = Path("/tmp/t4_asset_smoke_stage.txt")
    result_path = Path("/tmp/t4_asset_smoke_result.json")
    result_path.unlink(missing_ok=True)
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
    if set(robot.joint_names) != set(T4_JOINT_NAMES):
        raise RuntimeError("IsaacLab joint names do not match T4_JOINT_NAMES")
    if not set(REQUIRED_COLLISION_BODIES) <= set(robot.body_names):
        raise RuntimeError(f"missing required collision bodies in articulation: {robot.body_names}")

    collision_paths = _collision_prim_paths(sim.stage, "/World/T4")
    collision_paths_by_body = {
        body_name: [path for path in collision_paths if body_name.lower() in path.lower()]
        for body_name in REQUIRED_COLLISION_BODIES
    }
    missing_collision_bodies = [name for name, paths in collision_paths_by_body.items() if not paths]
    if missing_collision_bodies:
        raise RuntimeError(
            f"converted USD is missing collisions for {missing_collision_bodies}; "
            f"found {len(collision_paths)} collision prims"
        )
    result_path.write_text(
        json.dumps(
            {
                "num_joints": robot.num_joints,
                "num_bodies": robot.num_bodies,
                "joint_names": robot.joint_names,
                "motion_joint_names": T4_JOINT_NAMES,
                "joint_name_set_matches_t4_constants": True,
                "joint_order_matches_t4_constants": tuple(robot.joint_names) == T4_JOINT_NAMES,
                "required_collision_bodies": REQUIRED_COLLISION_BODIES,
                "collision_prim_paths_by_body": collision_paths_by_body,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        exit_code = 1
    finally:
        import os
        import threading

        threading.Timer(60.0, os._exit, args=(exit_code,)).start()
        app.close()
        os._exit(exit_code)

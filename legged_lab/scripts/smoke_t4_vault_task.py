"""Headless smoke test for the t4_vault_mimic task on the GPU runtime.

Asserts the frozen G1 observation contract (policy 150-D, critic 276-D,
27 actions), verifies the sphere-hand collision survived USD conversion, and
steps the env with zero actions to prove the MDP wiring is finite.
"""

import json
from pathlib import Path

from isaaclab.app import AppLauncher

from legged_lab.scripts.isaaclab_runtime_compat import (
    patch_missing_physx_material_attributes,
    patch_physx_backward_compatibility_setting,
)

patch_physx_backward_compatibility_setting(AppLauncher)
app = AppLauncher(headless=True).app

import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from pxr import Usd, UsdPhysics  # noqa: E402

import legged_lab.envs.t4.vault_mimic  # noqa: F401, E402
from legged_lab.envs.t4.vault_mimic.vault_env_cfg import (  # noqa: E402
    T4VaultMimicEnvCfg,
)

patch_missing_physx_material_attributes()

EXPECTED_POLICY_DIM = 150
EXPECTED_CRITIC_DIM = 276
EXPECTED_ACTION_DIM = 27


def _collision_prim_paths(stage, root_path: str) -> list[str]:
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return []
    return [str(prim.GetPath()) for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.CollisionAPI)]


def main() -> None:
    env_cfg = T4VaultMimicEnvCfg()
    env_cfg.scene.num_envs = 2
    env = ManagerBasedRLEnv(cfg=env_cfg)

    obs_dict, _ = env.reset()
    policy_obs = obs_dict["policy"]
    critic_obs = obs_dict["critic"]
    assert policy_obs.shape[1] == EXPECTED_POLICY_DIM, f"policy obs {policy_obs.shape} != {EXPECTED_POLICY_DIM}"
    assert critic_obs.shape[1] == EXPECTED_CRITIC_DIM, f"critic obs {critic_obs.shape} != {EXPECTED_CRITIC_DIM}"
    action_dim = env.action_manager.total_action_dim
    assert action_dim == EXPECTED_ACTION_DIM, f"action dim {action_dim} != {EXPECTED_ACTION_DIM}"

    robot = env.scene["robot"]
    import isaacsim.core.utils.stage as stage_utils

    stage = stage_utils.get_current_stage()
    collision_paths = _collision_prim_paths(stage, "/World/envs/env_0/Robot")
    hand_collision = {
        "left": [p for p in collision_paths if "sphere_hand" in p.lower() or "/AL7" in p],
        "right": [p for p in collision_paths if "sphere_hand" in p.lower() or "/AR7" in p],
    }
    assert hand_collision["left"], f"no left hand collision prim found in {len(collision_paths)} collision prims"
    assert hand_collision["right"], f"no right hand collision prim found in {len(collision_paths)} collision prims"

    for _ in range(50):
        actions = torch.zeros(env.num_envs, action_dim, device=env.device)
        obs_dict, rewards, terminated, truncated, _ = env.step(actions)
        assert torch.isfinite(obs_dict["policy"]).all(), "non-finite policy obs"
        assert torch.isfinite(rewards).all(), "non-finite rewards"

    command = env.command_manager.get_term("motion")
    result = {
        "policy_obs_dim": int(policy_obs.shape[1]),
        "critic_obs_dim": int(critic_obs.shape[1]),
        "action_dim": int(action_dim),
        "num_bodies": int(robot.num_bodies),
        "num_joints": int(robot.num_joints),
        "motion_frames": int(command.motion.time_step_total),
        "motion_fps": float(command.motion.fps),
        "left_hand_collision_prims": hand_collision["left"][:3],
        "right_hand_collision_prims": hand_collision["right"][:3],
        "error_anchor_pos_mean": float(command.metrics["error_anchor_pos"].mean()),
        "ok": True,
    }
    Path("/tmp/t4_vault_smoke_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"T4_VAULT_SMOKE {json.dumps(result)}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()

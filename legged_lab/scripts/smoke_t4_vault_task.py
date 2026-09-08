"""Headless smoke test for the t4_vault_mimic task on the GPU runtime.

Asserts the frozen G1 observation contract (policy 150-D, critic 276-D,
27 actions), verifies the sphere-hand collision survived USD conversion, and
steps the env with zero actions to prove the MDP wiring is finite.
"""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

from legged_lab.scripts.isaaclab_runtime_compat import (
    patch_missing_physx_material_attributes,
    patch_physx_backward_compatibility_setting,
)

patch_physx_backward_compatibility_setting(AppLauncher)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("/tmp/t4_vault_smoke_result.json"))
args = parser.parse_args()
app = AppLauncher(headless=True).app

import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from pxr import Usd, UsdPhysics  # noqa: E402

import legged_lab.envs.t4.vault_mimic  # noqa: F401, E402
from legged_lab.envs.t4.vault_mimic.vault_env_cfg import (  # noqa: E402
    T4VaultMimicEnvCfg,
)
from legged_lab.utils.rsl_rl_compat import RslRlVecEnvWrapper  # noqa: E402

patch_missing_physx_material_attributes()

EXPECTED_POLICY_DIM = 150
EXPECTED_CRITIC_DIM = 276
EXPECTED_ACTION_DIM = 27


def _collision_prim_paths(stage, root_path: str) -> list[str]:
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return []
    # The converted robot USD is instanceable; plain traversal skips instance
    # proxies and would miss every robot collision prim.
    prim_range = Usd.PrimRange(root, Usd.TraverseInstanceProxies())
    return [str(prim.GetPath()) for prim in prim_range if prim.HasAPI(UsdPhysics.CollisionAPI)]


def main() -> None:
    env_cfg = T4VaultMimicEnvCfg()
    env_cfg.scene.num_envs = 2
    print("[VAULT_SMOKE] creating env", flush=True)
    env = ManagerBasedRLEnv(cfg=env_cfg)
    print("[VAULT_SMOKE] env created", flush=True)

    obs_dict, _ = env.reset()
    print("[VAULT_SMOKE] reset done", flush=True)
    policy_obs = obs_dict["policy"]
    critic_obs = obs_dict["critic"]
    assert policy_obs.shape[1] == EXPECTED_POLICY_DIM, f"policy obs {policy_obs.shape} != {EXPECTED_POLICY_DIM}"
    assert critic_obs.shape[1] == EXPECTED_CRITIC_DIM, f"critic obs {critic_obs.shape} != {EXPECTED_CRITIC_DIM}"
    action_dim = env.action_manager.total_action_dim
    assert action_dim == EXPECTED_ACTION_DIM, f"action dim {action_dim} != {EXPECTED_ACTION_DIM}"

    robot = env.scene["robot"]
    stage = env.scene.stage
    collision_paths = _collision_prim_paths(stage, "/World/envs/env_0/Robot")
    if not collision_paths:
        env_prim = stage.GetPrimAtPath("/World/envs/env_0")
        children = [str(p.GetPath()) for p in env_prim.GetChildren()] if env_prim.IsValid() else "<invalid>"
        print(f"[VAULT_SMOKE] /World/envs/env_0 children: {children}", flush=True)
        collision_paths = [str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)]
        print(f"[VAULT_SMOKE] stage-wide collision prims: {len(collision_paths)}", flush=True)
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
    wrapper = RslRlVecEnvWrapper(env)
    wrapped_obs, extras = wrapper.get_observations()
    assert wrapped_obs.shape == policy_obs.shape
    assert extras["observations"]["critic"].shape == critic_obs.shape
    wrapped_obs, wrapped_rewards, wrapped_dones, _ = wrapper.step(
        torch.zeros(env.num_envs, action_dim, device=env.device)
    )
    assert torch.isfinite(wrapped_obs).all() and torch.isfinite(wrapped_rewards).all()
    assert wrapped_dones.shape == (env.num_envs,)
    result = {
        "motion_command_class": f"{type(command).__module__}.{type(command).__name__}",
        "rl_wrapper_class": f"{type(wrapper).__module__}.{type(wrapper).__name__}",
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"T4_VAULT_SMOKE {json.dumps(result)}", flush=True)


if __name__ == "__main__":
    # SimulationApp.close() can terminate the process with exit code 0 before a
    # pending traceback is printed, so surface the failure explicitly first.
    # It can also hang forever in the headless container, so a watchdog forces
    # the real exit code after a grace period.
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

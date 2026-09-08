"""Isaac runtime check of action units, observation layout and auto-reset AMP.

Run with the target Isaac launcher inside tmux. This is a plumbing/plant probe,
not a locomotion capability evaluation.
"""

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

from legged_lab.scripts.isaaclab_runtime_compat import (
    patch_missing_physx_material_attributes,
    patch_physx_backward_compatibility_setting,
)

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="g1_loco_teacher")
parser.add_argument("--output", required=True)
patch_physx_backward_compatibility_setting(AppLauncher)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app = AppLauncher(args).app

import torch  # noqa: E402

from legged_lab.envs import *  # noqa: E402,F401,F403
from legged_lab.utils import task_registry  # noqa: E402


def main():
    patch_missing_physx_material_attributes()
    cfg, agent_cfg = task_registry.get_cfgs(args.task)
    t4_cfg, _ = task_registry.get_cfgs("t4_loco_teacher_sparse")
    assert t4_cfg.robot.action_scale_effort_fraction is None, "G1 config leaked into T4"
    recipe = {
        "random_fraction": cfg.random_level_reset_fraction,
        "random_min": cfg.random_level_reset_min_level,
        "random_max": cfg.random_level_reset_max_level,
        "terrain_rows": cfg.scene.terrain_generator.num_rows,
        "amp_decay_start": cfg.amp_terrain_schedule.decay_start_difficulty,
        "amp_min_scale": cfg.amp_terrain_schedule.min_scale,
        "amp_reward_coef": agent_cfg.amp_reward_coef,
        "amp_task_reward_lerp": agent_cfg.amp_task_reward_lerp,
        "robot_spec": cfg.robot_spec.name,
        "teacher_base": type(cfg).__bases__[0].__module__ + "." + type(cfg).__bases__[0].__name__,
    }
    assert recipe["random_fraction"] == 0.10 and recipe["terrain_rows"] == 10
    assert recipe["random_min"] is None and recipe["random_max"] is None
    assert recipe["amp_decay_start"] == 0.3 and recipe["amp_min_scale"] == 0.3
    cfg.device = args.device
    cfg.sim.device = args.device
    cfg.scene.num_envs = 32
    cfg.scene.seed = 42
    cfg.scene.max_init_terrain_level = 0
    cfg.scene.terrain_generator.sub_terrains = {"flat": cfg.scene.terrain_generator.sub_terrains["flat"]}
    cfg.scene.terrain_generator.sub_terrains["flat"].proportion = 1.0
    cfg.scene.terrain_generator.num_rows = 10
    cfg.scene.terrain_generator.num_cols = 1
    cfg.scene.terrain_generator.curriculum = True
    cfg.random_level_reset_fraction = 0.0
    cfg.terrain_aware_commands = False
    cfg.noise.add_noise = False
    cfg.commands.ranges.lin_vel_x = (0.0, 0.0)
    cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
    cfg.domain_rand.events.push_robot = None
    reset_base = cfg.domain_rand.events.reset_base.params
    reset_base["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    reset_base["velocity_range"] = {}
    cfg.domain_rand.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    env = task_registry.get_task_class(args.task)(cfg, headless=True)
    env.validate_training_contract(agent_cfg.to_dict())
    from legged_lab.locomotion.symmetry import get_symmetric_states

    # Exercise actual random-reset code with 10 real terrain rows. The sentinel
    # distinguishes selected resets (including a draw of level zero) from others.
    cfg.random_level_reset_fraction = recipe["random_fraction"]
    terrain = env.scene.terrain
    seen, selected = set(), 0
    ids = torch.arange(env.num_envs, device=env.device)
    for _ in range(200):
        terrain.terrain_levels.fill_(-1)
        env._apply_random_level_resets(ids)
        levels = terrain.terrain_levels[terrain.terrain_levels >= 0].cpu().tolist()
        seen.update(levels)
        selected += len(levels)
    assert seen == set(range(10)), seen
    terrain.terrain_levels[:] = ids % 10
    scales = env.amp_reward_coef_scale()[:10].cpu().tolist()
    expected = torch.tensor([1.0 - max(0.0, (i / 9 - 0.3) / 0.7) * 0.7 for i in range(10)])
    assert torch.allclose(torch.tensor(scales), expected)
    terrain.terrain_levels.zero_()
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    cfg.random_level_reset_fraction = 0.0
    env.reset(ids)
    obs, extras = env.get_observations()
    mirrored, _ = get_symmetric_states(obs=obs, env=env)
    restored, _ = get_symmetric_states(obs=mirrored, env=env)
    assert torch.allclose(obs, restored)
    kp = env.robot.data.default_joint_stiffness[0]
    effort = env.robot.data.joint_effort_limits[0]
    scale = torch.ones_like(kp) * env.action_scale if isinstance(env.action_scale, float) else env.action_scale[0]
    result = {
        "recipe": recipe,
        "sampled_random_levels": sorted(seen),
        "realized_random_fraction": selected / (200 * env.num_envs),
        "amp_scales_by_level": scales,
        "runtime_mirror_involution": True,
        "task": args.task,
        "actor_width": obs.shape[-1],
        "critic_width": extras["observations"]["critic"].shape[-1],
        "amp_width": env.get_amp_obs_for_expert_trans().shape[-1],
        "policy_joint_names": list(env.policy_joint_names),
        "policy_to_sim_joint_ids": list(env.policy_joint_ids),
        "t4_action_contract_unchanged": True,
        "joints": {
            name: {
                "kp": float(kp[i]),
                "effort_limit": float(effort[i]),
                "action_scale_rad": float(scale[i]),
                "old_unit_action_torque_fraction": float(kp[i] * 0.25 / effort[i]),
                "new_unit_action_torque_fraction": float(kp[i] * scale[i] / effort[i]),
            }
            for i, name in enumerate(env.robot.joint_names)
        },
    }
    original_reset = env.reset
    snapshot = {}

    def capture_before_reset(ids):
        if len(ids):
            snapshot["amp"] = env.get_amp_obs_for_expert_trans()[ids].clone()
        return original_reset(ids)

    env.reset = capture_before_reset
    checked = reset_state_different = 0
    zero_action_min_clearance = float("inf")
    with torch.inference_mode():
        for step in range(240):
            actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
            if step >= 120:
                actions.uniform_(-1.0, 1.0)
            if step in (60, 180):
                env.episode_length_buf[::2] = int(env.max_episode_length) - 1
            obs, reward, dones, info = env.step(actions)
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
            ids = env.reset_env_ids
            if len(ids):
                assert torch.equal(info["terminal_amp_obs"], snapshot["amp"])
                reset_state = env.get_amp_obs_for_expert_trans()[ids]
                reset_state_different += int((info["terminal_amp_obs"] != reset_state).any(dim=1).sum())
                checked += len(ids)
            else:
                assert "terminal_amp_obs" not in info, "stale terminal snapshot"
            if step < 120:
                clearance = (
                    env.robot.data.root_pos_w[:, 2] - env.robot.data.body_pos_w[:, env.feet_body_ids, 2].min(1).values
                )
                zero_action_min_clearance = min(zero_action_min_clearance, float(clearance.min()))
    assert checked > 0 and reset_state_different > 0
    result.update(
        terminal_snapshots_checked=checked,
        terminal_distinct_from_reset=reset_state_different,
        zero_action_min_clearance_m=zero_action_min_clearance,
        finite_steps=240,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    code = 0
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        code = 1
    finally:
        import threading

        threading.Timer(30.0, os._exit, args=(code,)).start()
        app.close()
        os._exit(code)

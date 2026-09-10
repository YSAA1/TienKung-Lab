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

"""Isaac plant/wiring probe for z2_loco_teacher. Not a locomotion capability test.

Run on nubot inside tmux:
  bash scripts/nubot_run.sh legged_lab/scripts/probe_z2_isaac.py --output artifacts/z2_migration/isaac_probe.json --headless
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

from legged_lab.scripts.isaaclab_runtime_compat import (
    patch_missing_physx_material_attributes,
    patch_physx_backward_compatibility_setting,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="z2_loco_teacher")
parser.add_argument("--output", required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--pulse_amplitude", type=float, default=0.08)
patch_physx_backward_compatibility_setting(AppLauncher)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
if args.num_envs < 3 or args.steps < 8:
    parser.error("num_envs>=3 and steps>=8 required")
if not args.headless:
    parser.error("this diagnostic must run with --headless")
app = AppLauncher(args).app

import omni.usd  # noqa: E402
import torch  # noqa: E402
from pxr import PhysxSchema, Usd, UsdPhysics  # noqa: E402

from legged_lab.assets.z2.amp_manifest import usd_layer_shas  # noqa: E402
from legged_lab.assets.z2.constants import (  # noqa: E402
    NUM_Z2_29DOF_JOINTS,
    Z2_29DOF_JOINT_NAMES,
    Z2_FIXED_JOINT_NAMES,
    Z2_STANDING_PELVIS_Z,
)
from legged_lab.assets.z2.probe_contract import (  # noqa: E402
    G1_70D_FIXTURE,
    WIDTH64_FIXTURE,
    assert_command_target_vector,
    assert_pair_initial_match,
    assert_policy_maps_onto_sim,
    assert_signed_channel_response,
    physx_try_call,
    restore_level0_terrain,
)
from legged_lab.assets.z2.schemas import (  # noqa: E402
    AMP_FOOT_BODIES,
    AMP_FRAME_DIM,
    AMP_HAND_BODIES,
)
from legged_lab.envs import *  # noqa: E402,F401,F403
from legged_lab.locomotion.symmetry import get_symmetric_states  # noqa: E402
from legged_lab.utils import task_registry  # noqa: E402
from rsl_rl.utils.motion_loader import AMPLoader  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
result = {"ok": False, "task": args.task, "unknowns": [], "negatives": {}}


def _fail(message: str) -> None:
    result["error"] = message
    raise AssertionError(message)


def main() -> None:  # noqa: C901 - sequential diagnostic phases share one live Isaac env
    patch_missing_physx_material_attributes()
    if args.task != "z2_loco_teacher":
        _fail(f"this probe is Z2-only, got {args.task}")
    cfg, agent_cfg = task_registry.get_cfgs(args.task)
    t4_cfg, _ = task_registry.get_cfgs("t4_loco_teacher_sparse")
    g1_cfg, _ = task_registry.get_cfgs("g1_loco_teacher")
    if t4_cfg.robot.action_scale_effort_fraction is not None:
        _fail("Z2 config leaked into T4")
    if cfg.robot_spec.name != "z2":
        _fail(f"robot_spec is {cfg.robot_spec.name}")
    if tuple(cfg.robot_spec.joint_names) != Z2_29DOF_JOINT_NAMES:
        _fail("policy joints != Z2_29DOF_JOINT_NAMES")
    if agent_cfg.amp_frame_dim != AMP_FRAME_DIM:
        _fail(f"amp_frame_dim {agent_cfg.amp_frame_dim} != {AMP_FRAME_DIM}")
    if tuple(agent_cfg.amp_joint_order) != Z2_29DOF_JOINT_NAMES:
        _fail("amp_joint_order != policy joints")
    if getattr(cfg, "lightlp_promotion_distance", "path_length") != "max_radial":
        _fail("Z2 recipe must resolve max_radial promotion, not inherited path_length")
    if not getattr(cfg, "progress_monitor_enabled", False):
        _fail("Z2 recipe must enable progress monitoring")
    result["lightlp_promotion_distance"] = cfg.lightlp_promotion_distance
    result["progress_monitor_enabled"] = bool(cfg.progress_monitor_enabled)
    result["action_scale"] = cfg.robot.action_scale
    result["sim_dt"] = cfg.sim.dt
    result["decimation"] = cfg.sim.decimation
    spawn = cfg.scene.robot.spawn
    usd_path = str(getattr(spawn, "usd_path", ""))
    result["asset_mode"] = "upstream_usd"
    result["asset"] = {
        "class": type(spawn).__name__,
        "usd_path": usd_path,
        "urdf_converter_applied": False,
        "usd_layers": usd_layer_shas(ROOT / "legged_lab/assets/z2"),
    }
    if "UsdFileCfg" not in type(spawn).__name__:
        _fail(f"training spawn must be UsdFileCfg, got {type(spawn).__name__}")
    if "assembly.usd" not in usd_path.replace("\\", "/"):
        _fail(f"training plant must be original assembly.usd, got {usd_path}")
    result["converter_archival"] = {
        "source": "usd/assembly_29dof/config.yaml",
        "not_applied_at_runtime": True,
        "make_instanceable": False,
        "merge_fixed_joints": True,
        "collider_type": "convex_hull",
    }
    result["enabled_self_collisions"] = bool(spawn.articulation_props.enabled_self_collisions)
    result["actuator_cfg"] = {
        name: {
            "stiffness": act.stiffness,
            "damping": act.damping,
            "effort_limit_sim": act.effort_limit_sim,
            "velocity_limit_sim": act.velocity_limit_sim,
            "armature": act.armature,
        }
        for name, act in cfg.scene.robot.actuators.items()
    }

    cfg.device = args.device
    cfg.sim.device = args.device
    cfg.scene.num_envs = args.num_envs
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
    cfg.domain_rand.events.physics_material = None
    cfg.domain_rand.events.add_base_mass = None
    cfg.domain_rand.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    cfg.domain_rand.events.reset_base.params["velocity_range"] = {}
    cfg.domain_rand.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    cfg.domain_rand.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

    env = task_registry.get_task_class(args.task)(cfg, headless=True)
    env.validate_training_contract(agent_cfg.to_dict())

    sim_joints = list(env.robot.joint_names)
    sim_bodies = list(env.robot.body_names)
    policy_joints = list(env.policy_joint_names)
    result["sim_joint_names"] = sim_joints
    result["policy_joint_names"] = policy_joints
    result["policy_to_sim_joint_ids"] = list(env.policy_joint_ids)
    result["sim_body_names"] = sim_bodies
    result["num_actions"] = int(env.num_actions)
    result["num_sim_joints"] = len(sim_joints)
    result["num_sim_bodies"] = len(sim_bodies)
    result["has_neck_link_body"] = "neck_link" in sim_bodies
    if "neck_link" in sim_bodies:
        _fail("neck_link body present; training plant is not the original compiled USD")
    result["root_z"] = float(env.robot.data.root_pos_w[0, 2])
    result["standing_pelvis_z_declared"] = Z2_STANDING_PELVIS_Z

    if env.num_actions != NUM_Z2_29DOF_JOINTS:
        _fail(f"num_actions {env.num_actions} != {NUM_Z2_29DOF_JOINTS}")
    if tuple(policy_joints) != Z2_29DOF_JOINT_NAMES:
        _fail("runtime policy joints != declared Z2_29DOF_JOINT_NAMES")
    extra = sorted(set(sim_joints) - set(Z2_29DOF_JOINT_NAMES))
    missing = sorted(set(Z2_29DOF_JOINT_NAMES) - set(sim_joints))
    result["extra_sim_joints"] = extra
    result["missing_policy_joints"] = missing
    if missing:
        _fail(f"articulation missing policy joints {missing}")
    for name in Z2_FIXED_JOINT_NAMES:
        if name in sim_joints:
            _fail(f"fixed joint {name} appeared as a DOF; merge_fixed_joints may be off")
    if any("neck" in name and name.endswith("_joint") and name != "neck_fixed" for name in sim_joints):
        _fail(f"actuated neck in simulator joints: {sim_joints}")
    for body in AMP_HAND_BODIES + AMP_FOOT_BODIES + ("waist_roll_link", "base_link"):
        if body not in sim_bodies:
            _fail(f"missing body {body}")
    merged_hands = [name for name in ("L_sphere_hand", "R_sphere_hand") if name in sim_bodies]
    result["unmerged_sphere_hands"] = merged_hands
    if merged_hands:
        result["unknowns"].append("sphere hands did not merge; AMP hands are still wrist_yaw origins")

    try:
        assert_policy_maps_onto_sim(sim_joints, env.policy_joint_ids, policy_joints)
    except AssertionError as exc:
        _fail(str(exc))
    if list(env.policy_joint_ids) != list(env.amp_builder.joint_ids):
        _fail("AMP joint ids != policy joint ids")
    result["mapped_sim_joints"] = [sim_joints[int(i)] for i in env.policy_joint_ids]
    result["cached_articulation_data"] = {
        "label": "ArticulationData default_* / limit tensors; not a PhysX consistency assertion",
        "default_joint_stiffness": env.robot.data.default_joint_stiffness[0].detach().cpu().tolist(),
        "default_joint_damping": env.robot.data.default_joint_damping[0].detach().cpu().tolist(),
        "joint_effort_limits": env.robot.data.joint_effort_limits[0].detach().cpu().tolist(),
        "joint_vel_limits": env.robot.data.joint_vel_limits[0].detach().cpu().tolist(),
    }
    view = getattr(env.robot, "root_physx_view", None)
    result["physx_readback"] = {
        method: physx_try_call(view, method)
        for method in (
            "get_masses",
            "get_coms",
            "get_inertias",
            "get_dof_stiffnesses",
            "get_dof_dampings",
            "get_dof_max_forces",
            "get_dof_max_velocities",
            "get_dof_limits",
            "get_dof_armatures",
        )
    }
    result["physx_readback_layout"] = {
        "get_coms": "xyz + qx qy qz qw principal axes, relative to body prim",
        "get_inertias": "9 column-major entries about COM in body-prim frame; do not rotate again",
    }
    stage = omni.usd.get_context().get_stage()
    robot_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    if not robot_prim.IsValid():
        _fail("env0 robot prim /World/envs/env_0/Robot is missing")
    articulation_roots = []
    colliders = []
    for prim in Usd.PrimRange(robot_prim, Usd.TraverseInstanceProxies()):
        if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            api = PhysxSchema.PhysxArticulationAPI(prim)
            articulation_roots.append(
                {
                    "path": str(prim.GetPath()),
                    "self_collisions": api.GetEnabledSelfCollisionsAttr().Get(),
                    "position_iterations": api.GetSolverPositionIterationCountAttr().Get(),
                    "velocity_iterations": api.GetSolverVelocityIterationCountAttr().Get(),
                }
            )
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            entry = {
                "path": str(prim.GetPath()),
                "type": prim.GetTypeName(),
                "enabled": UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get(),
            }
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                entry["approximation"] = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            colliders.append(entry)
    result["usd_articulation_roots"] = articulation_roots
    result["usd_colliders"] = colliders
    result["usd_enabled_collider_count"] = sum(1 for item in colliders if item.get("enabled"))
    result["physx_collision_flags"] = {
        "enabled_self_collisions_cfg": bool(spawn.articulation_props.enabled_self_collisions),
        "enabled_self_collisions_usd": articulation_roots[0]["self_collisions"] if articulation_roots else None,
    }

    obs, extras = env.get_observations()
    amp = env.get_amp_obs_for_expert_trans()
    result["actor_width"] = int(obs.shape[-1])
    result["critic_width"] = int(extras["observations"]["critic"].shape[-1])
    result["amp_width"] = int(amp.shape[-1])
    if amp.shape[-1] != AMP_FRAME_DIM:
        _fail(f"runtime AMP width {amp.shape[-1]} != {AMP_FRAME_DIM}")
    if not torch.isfinite(obs).all() or not torch.isfinite(amp).all():
        _fail("non-finite observation at reset")

    mirrored, _ = get_symmetric_states(obs=obs, env=env)
    restored, _ = get_symmetric_states(obs=mirrored, env=env)
    if not torch.allclose(obs, restored, atol=1.0e-5):
        _fail("observation mirror is not an involution")
    result["runtime_mirror_involution"] = True

    feet = env.robot.data.body_pos_w[0, env.feet_body_ids, :]
    hands = env.robot.data.body_pos_w[0, env.amp_builder.hand_body_ids, :]
    result["standing_foot_origins"] = feet.detach().cpu().tolist()
    result["standing_hand_origins"] = hands.detach().cpu().tolist()
    result["standing_feet_y_distance"] = float((feet[0, 1] - feet[1, 1]).abs())
    clearance = float((env.robot.data.root_pos_w[:, 2] - feet[:, 2].min()).min())
    result["standing_root_above_foot_origin_m"] = clearance

    ids = torch.arange(env.num_envs, device=env.device)
    if env.num_envs < 3:
        _fail("paired signed pulse needs null/+/- envs")
    if abs(float(cfg.robot.action_scale) - 0.25) > 1.0e-12:
        _fail(f"action_scale {cfg.robot.action_scale} != 0.25")

    def _zero_velocities() -> None:
        env.robot.write_joint_velocity_to_sim(torch.zeros_like(env.robot.data.joint_vel))
        root_vel = torch.zeros(env.num_envs, 6, device=env.device)
        env.robot.write_root_velocity_to_sim(root_vel)
        env.robot.write_data_to_sim()
        env.sim.forward()

    def _view_row(method: str, env_id: int):
        payload = physx_try_call(getattr(env.robot, "root_physx_view", None), method)
        if "unavailable" in payload:
            return None
        value = getattr(env.robot.root_physx_view, method)()
        return value[env_id].detach().cpu().tolist()

    def _pair_snapshot(env_id: int) -> dict:
        origin = env.scene.env_origins[env_id]
        root = env.robot.data.root_pos_w[env_id]
        lin = env.robot.data.root_lin_vel_w[env_id].detach().cpu().tolist()
        ang = env.robot.data.root_ang_vel_w[env_id].detach().cpu().tolist()
        return {
            "q": env.robot.data.joint_pos[env_id].detach().cpu().tolist(),
            "qd": env.robot.data.joint_vel[env_id].detach().cpu().tolist(),
            "root_rel": (root - origin).detach().cpu().tolist(),
            "root_vel": lin + ang,
            "masses": _view_row("get_masses", env_id),
            "stiffness": _view_row("get_dof_stiffnesses", env_id),
            "damping": _view_row("get_dof_dampings", env_id),
        }

    pulse_records = []
    for i, name in enumerate(Z2_29DOF_JOINT_NAMES):
        env.reset(ids)
        _zero_velocities()
        try:
            assert_pair_initial_match(_pair_snapshot(0), _pair_snapshot(1))
            assert_pair_initial_match(_pair_snapshot(0), _pair_snapshot(2))
        except AssertionError as exc:
            _fail(f"{name} pair IC: {exc}")
        before = env.robot.data.joint_pos.clone()
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        actions[1, i] = args.pulse_amplitude
        actions[2, i] = -args.pulse_amplitude
        obs, reward, dones, info = env.step(actions)
        default_q = env.robot.data.default_joint_pos[0].detach().cpu().tolist()
        target = getattr(env.robot.data, "joint_pos_target", None)
        if target is None:
            _fail("joint_pos_target readback unavailable; cannot assert commanded mapping")
        plus_target = target[1].detach().cpu().tolist()
        minus_target = target[2].detach().cpu().tolist()
        try:
            assert_command_target_vector(
                readback_target=plus_target,
                default_joint_pos=default_q,
                policy_joint_ids=env.policy_joint_ids,
                policy_names=policy_joints,
                sim_names=sim_joints,
                channel=i,
                action=args.pulse_amplitude,
                action_scale=cfg.robot.action_scale,
            )
            assert_command_target_vector(
                readback_target=minus_target,
                default_joint_pos=default_q,
                policy_joint_ids=env.policy_joint_ids,
                policy_names=policy_joints,
                sim_names=sim_joints,
                channel=i,
                action=-args.pulse_amplitude,
                action_scale=cfg.robot.action_scale,
            )
        except AssertionError as exc:
            _fail(f"{name} command target: {exc}")
        for _ in range(3):
            env.step(actions)
        after = env.robot.data.joint_pos
        null_delta = (after[0] - before[0]).detach().cpu().tolist()
        plus_delta = (after[1] - before[1]).detach().cpu().tolist()
        minus_delta = (after[2] - before[2]).detach().cpu().tolist()
        sim_id = int(env.policy_joint_ids[i])
        try:
            plus_signed = assert_signed_channel_response(null_delta, plus_delta, sim_id, expected_sign=1)
            minus_signed = assert_signed_channel_response(null_delta, minus_delta, sim_id, expected_sign=-1)
        except AssertionError as exc:
            _fail(f"{name} signed response: {exc}")
        pulse_records.append(
            {
                "policy_name": name,
                "sim_name": sim_joints[sim_id],
                "sim_id": sim_id,
                "plus_action": args.pulse_amplitude,
                "minus_action": -args.pulse_amplitude,
                "plus_target_sim": plus_target[sim_id],
                "minus_target_sim": minus_target[sim_id],
                "plus_signed_dq": plus_signed,
                "minus_signed_dq": minus_signed,
            }
        )
    result["pulse_records"] = pulse_records

    # Random-level coverage on ten rows.
    cfg.random_level_reset_fraction = 0.10
    terrain = env.scene.terrain
    seen = set()
    ids = torch.arange(env.num_envs, device=env.device)
    for _ in range(200):
        terrain.terrain_levels.fill_(-1)
        env._apply_random_level_resets(ids)
        seen.update(terrain.terrain_levels[terrain.terrain_levels >= 0].cpu().tolist())
    result["sampled_random_levels"] = sorted(seen)
    if seen != set(range(10)):
        _fail(f"random level reset did not cover 0-9, got {sorted(seen)}")
    restore_level0_terrain(terrain, env)
    if int(terrain.terrain_levels.min()) != 0 or int(terrain.terrain_levels.max()) != 0:
        _fail("terrain levels were not restored to 0 before contact/reset checks")
    result["terrain_levels_after_restore"] = terrain.terrain_levels[: min(8, env.num_envs)].detach().cpu().tolist()
    cfg.random_level_reset_fraction = 0.0
    env.reset(ids)
    zeros = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    settle_steps = min(50, args.steps)
    landing = []
    cumulative_resets = 0
    for step in range(settle_steps):
        env.step(zeros)
        gravity = env.robot.data.projected_gravity_b[0].detach().cpu().tolist()
        foot_fz = env.contact_sensor.data.net_forces_w[0, env.feet_cfg.body_ids, 2].detach().cpu().tolist()
        reset = bool(env.reset_buf[0])
        timeout = bool(env.time_out_buf[0])
        terminated = bool(getattr(env, "terminated_buf", env.reset_buf)[0])
        cumulative_resets += int(env.reset_buf.sum().item())
        landing.append(
            {
                "step": step,
                "root_z": float(env.robot.data.root_pos_w[0, 2]),
                "projected_gravity_b": gravity,
                "foot_force_z": foot_fz,
                "reset": reset,
                "timeout": timeout,
                "terminated": terminated,
            }
        )
    result["landing_contact_series"] = {
        "label": "zero_action_pd_landing_timeseries",
        "not_claimed": "sustained_balance_or_locomotion",
        "steps": settle_steps,
        "samples": landing,
        "cumulative_resets_all_envs": cumulative_resets,
        "repeated_collapse": cumulative_resets >= env.num_envs,
        "note": "finite contact/height samples under fixed PD; not plant_support_proven",
    }

    env.reset(ids)
    original_reset = env.reset
    snapshot = {}

    def capture_before_reset(reset_ids):
        if len(reset_ids):
            snapshot["amp"] = env.get_amp_obs_for_expert_trans()[reset_ids].clone()
        return original_reset(reset_ids)

    env.reset = capture_before_reset
    checked = reset_state_different = 0
    with torch.inference_mode():
        for step in range(args.steps):
            actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
            if step >= args.steps // 2:
                actions.uniform_(-1.0, 1.0)
            if step in (args.steps // 3, 2 * args.steps // 3):
                env.episode_length_buf[::2] = int(env.max_episode_length) - 1
            obs, reward, dones, info = env.step(actions)
            if not torch.isfinite(obs).all() or not torch.isfinite(reward).all():
                _fail(f"non-finite at step {step}")
            reset_ids = env.reset_env_ids
            if len(reset_ids):
                if not torch.equal(info["terminal_amp_obs"], snapshot["amp"]):
                    _fail("terminal AMP snapshot is not the pre-reset state")
                reset_state = env.get_amp_obs_for_expert_trans()[reset_ids]
                reset_state_different += int((info["terminal_amp_obs"] != reset_state).any(dim=1).sum())
                checked += len(reset_ids)
    result["terminal_snapshots_checked"] = checked
    result["terminal_distinct_from_reset"] = reset_state_different
    result["finite_steps"] = args.steps
    if checked == 0:
        result["unknowns"].append("no auto-reset fired; terminal AMP snapshot untested this run")

    g1_expert = G1_70D_FIXTURE
    upstream_64 = WIDTH64_FIXTURE
    for label, path, expect in (
        ("g1_70d", g1_expert, "runtime contract expects"),
        ("upstream_64d", upstream_64, "does not match schema width"),
    ):
        try:
            if not path.is_file():
                raise FileNotFoundError(path)
            AMPLoader(
                device="cpu",
                time_between_frames=float(env.step_dt),
                frame_dim=AMP_FRAME_DIM,
                motion_files=[str(path)],
                expected_joint_order=list(Z2_29DOF_JOINT_NAMES),
            )
        except FileNotFoundError as exc:
            _fail(f"{label} fixture missing: {exc}")
        except ValueError as exc:
            if expect not in str(exc):
                _fail(f"{label} rejected for unexpected reason ({exc}); expected {expect!r}")
            result["negatives"][label] = {
                "path": str(path),
                "rejected": True,
                "error": f"ValueError:{exc}",
                "expect": expect,
            }
            continue
        _fail(f"{label} loaded as a Z2 expert: {path}")

    result["t4_action_scale"] = t4_cfg.robot.action_scale
    result["g1_action_scale"] = g1_cfg.robot.action_scale
    result["z2_action_scale"] = cfg.robot.action_scale
    result["ready"] = False
    result["ok"] = True


if __name__ == "__main__":
    code = 0
    try:
        main()
    except BaseException as exc:
        import traceback

        result.setdefault("error", str(exc))
        result["traceback"] = traceback.format_exc()
        traceback.print_exc()
        code = 1
    finally:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        import threading

        threading.Timer(30.0, os._exit, args=(code,)).start()
        app.close()
        os._exit(code)

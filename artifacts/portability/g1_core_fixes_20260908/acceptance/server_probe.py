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

"""Read-only-source Isaac wiring/plant probe. Run through the repo launcher in tmux.

Only diagnostic environment instances are created. Their formal reward, noise,
commands, randomization, termination and curriculum settings remain unchanged.
Zero actions and joint pulses are plumbing probes, not learned capability tests.
"""

import argparse
import copy
import hashlib
import importlib
import inspect
import json
import os
import platform
import subprocess
import sys
import threading
import traceback
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

from legged_lab.scripts.isaaclab_runtime_compat import (
    patch_missing_physx_material_attributes,
    patch_physx_backward_compatibility_setting,
)

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=("g1_loco_teacher", "t4_loco_teacher_sparse"), default="g1_loco_teacher")
parser.add_argument("--terrain", choices=("mixed", "flat"), default="mixed")
parser.add_argument("--g1_progress_ab", choices=("A", "B"), help="Apply the exact current train.py A/B recipe switches")
parser.add_argument(
    "--amp_expert_manifest", type=Path, help="Verify the formal clip hashes without loading a discriminator"
)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--pulse_steps", type=int, default=20)
parser.add_argument("--pulse_amplitude", type=float, default=0.1)
parser.add_argument("--output", required=True)
patch_physx_backward_compatibility_setting(AppLauncher)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 3 or args.steps < 1 or args.pulse_steps < 1 or not 0 < args.pulse_amplitude <= 0.2:
    parser.error("Require num_envs>=3, steps>=1, pulse_steps>=1, and 0<pulse_amplitude<=0.2")
if not args.headless:
    parser.error("This diagnostic must run with --headless")
if args.g1_progress_ab and args.task != "g1_loco_teacher":
    parser.error("--g1_progress_ab only applies to g1_loco_teacher")
app = AppLauncher(args).app

import torch

from legged_lab.envs import *  # noqa: F403,F401,E402
from legged_lab.utils import task_registry

result = {
    "ok": False,
    "task": args.task,
    "terrain_selection": args.terrain,
    "scope": "Isaac wiring and zero-action/pulse plant probe, not locomotion capability",
    "sections_completed": [],
}
output = Path(args.output)


def save():
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def stats(value):
    value = value.detach().float().cpu()
    finite = torch.isfinite(value)
    clean = value[finite]
    sample = value.reshape(-1)[:18].tolist()
    return {
        "shape": list(value.shape),
        "finite_fraction": float(finite.float().mean()),
        "min": float(clean.min()) if clean.numel() else None,
        "max": float(clean.max()) if clean.numel() else None,
        "mean": float(clean.mean()) if clean.numel() else None,
        "sample": [v if torch.isfinite(torch.tensor(v)) else None for v in sample],
    }


def delta_stats(before, after):
    finite = torch.isfinite(before) & torch.isfinite(after)
    delta = (after - before)[finite]
    return {
        "shared_finite_count": int(finite.sum()),
        "changed_finite_count": int((delta.abs() > 1e-7).sum()),
        "max_abs_delta": float(delta.abs().max()) if delta.numel() else None,
        "finite_pattern_changed_count": int((torch.isfinite(before) != torch.isfinite(after)).sum()),
    }


def source_info(module_name):
    module = importlib.import_module(module_name)
    filename = Path(module.__file__).resolve()
    return {"file": str(filename), "sha256": hashlib.sha256(filename.read_bytes()).hexdigest()}


def spawn_info(spawn):
    fields = (
        "asset_path",
        "usd_path",
        "usd_dir",
        "usd_file_name",
        "force_usd_conversion",
        "fix_base",
        "merge_fixed_joints",
        "replace_cylinders_with_capsules",
        "self_collision",
    )
    data = {name: getattr(spawn, name) for name in fields if hasattr(spawn, name)}
    data["class"] = type(spawn).__module__ + "." + type(spawn).__qualname__
    data["class_source"] = inspect.getsourcefile(type(spawn))
    for name in ("asset_path", "usd_path"):
        path = data.get(name)
        if path and Path(path).is_file():
            data[name + "_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return data


def mark(section):
    result["sections_completed"].append(section)
    save()
    print("PROBE_SECTION " + section, flush=True)


def main():
    patch_missing_physx_material_attributes()
    cfg, agent_cfg = task_registry.get_cfgs(args.task)
    if args.g1_progress_ab is not None:
        # These are formal train.py recipe switches, not diagnostic tuning.
        cfg.sparse_command_min_speed_scale = 1.0
        cfg.lightlp_promotion_distance = "path_length" if args.g1_progress_ab == "A" else "max_radial"
        cfg.progress_monitor_enabled = True
    if args.amp_expert_manifest is not None:
        manifest_path = args.amp_expert_manifest.resolve()
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get("clips"):
            raise RuntimeError("AMP manifest contains no clips")
        clips = []
        for name, metadata in manifest["clips"].items():
            clip_path = manifest_path.parent / (name + ".txt")
            actual_hash = hashlib.sha256(clip_path.read_bytes()).hexdigest()
            if actual_hash != metadata["sha256"]:
                raise RuntimeError("AMP hash mismatch: " + str(clip_path))
            clips.append({"file": str(clip_path), "sha256": actual_hash, "manifest_metadata": metadata})
        result["expert_manifest_verification"] = {
            "file": str(manifest_path),
            "count": len(clips),
            "clips": clips,
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        }
        agent_cfg.amp_expert_dir = str(manifest_path.parent)
        agent_cfg.amp_motion_files = [clip["file"] for clip in clips]
    assets_module = importlib.import_module("isaaclab.utils.assets")
    result["provenance"] = {
        "python": sys.executable,
        "cwd": os.getcwd(),
        "sys_path": sys.path,
        "argv": sys.argv,
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "modules": {
            name: source_info(name)
            for name in (
                "legged_lab",
                "legged_lab.envs",
                "legged_lab.locomotion.env",
                "legged_lab.locomotion.teacher_cfg",
                "legged_lab.locomotion.amp_features",
                "legged_lab.locomotion.schemas",
                "legged_lab.mdp.rewards",
                "legged_lab.envs.g1.teacher_cfg",
                "legged_lab.envs.t4.teacher_cfg",
                "isaaclab.envs.mdp.commands.velocity_command",
                "isaaclab.sensors.ray_caster.ray_caster",
                "isaaclab.utils.assets",
                "rsl_rl.algorithms.amp_ppo",
            )
        },
        "nucleus_constants": {
            name: getattr(assets_module, name, "ATTRIBUTE_ABSENT")
            for name in ("NUCLEUS_ASSET_ROOT_DIR", "ISAAC_NUCLEUS_DIR", "ISAACLAB_NUCLEUS_DIR")
        },
        "cfg_robot_spawn": spawn_info(cfg.scene.robot.spawn),
    }
    dependency = Path(assets_module.__file__).resolve()
    lab = Path("/home/nubot/IsaacLab")
    result["runtime_versions"] = dict(
        python=platform.python_version(),
        torch=torch.__version__,
        numpy=np.__version__,
        isaac_sim=Path("/home/nubot/isaac-sim-standalone-5.1.0-linux-x86_64/VERSION").read_text().strip(),
        isaaclab_commit=subprocess.check_output(["git", "-C", str(lab), "rev-parse", "HEAD"], text=True).strip(),
        assets_sha256=hashlib.sha256(dependency.read_bytes()).hexdigest(),
        assets_patch=subprocess.check_output(["git", "-C", str(lab), "diff", "--", str(dependency)], text=True),
    )
    assert result["runtime_versions"]["isaaclab_commit"] == "3d5ea25ddbcba05bef4c9acd1dacb9fa728b289b"
    result["formal_cfg_before_diagnostic_overrides"] = {
        "num_envs": cfg.scene.num_envs,
        "robot_spec": cfg.robot_spec.name,
        "scene_seed": getattr(cfg.scene, "seed", None),
        "agent_seed": agent_cfg.seed,
        "action_scale": cfg.robot.action_scale,
        "noise": cfg.noise.add_noise,
        "random_level_reset_fraction": cfg.random_level_reset_fraction,
        "max_init_terrain_level": cfg.scene.max_init_terrain_level,
        "terrain_columns": cfg.scene.terrain_generator.num_cols,
        "terrain_rows": cfg.scene.terrain_generator.num_rows,
        "g1_progress_ab": args.g1_progress_ab,
        "lightlp_promotion_distance": cfg.lightlp_promotion_distance,
        "progress_monitor_enabled": cfg.progress_monitor_enabled,
        "collapse_grace_s": cfg.collapse_reset_grace_s,
        "collapse_immunity": cfg.collapse_reset_respects_impact_immunity,
        "amp_reward_coef": agent_cfg.amp_reward_coef,
        "amp_task_reward_lerp": agent_cfg.amp_task_reward_lerp,
        "sparse_min_speed_scale": getattr(cfg, "sparse_command_min_speed_scale", 1.0),
        "commands": cfg.commands.to_dict(),
    }
    cfg.scene.num_envs = args.num_envs
    # train.py also fills this scene field from the runner's formal seed.
    cfg.scene.seed = agent_cfg.seed
    cfg.device = cfg.sim.device = args.device
    if args.terrain == "flat":
        # Terrain selection is the only recipe override; preserve ten rows,
        # random revisits, command distribution, pushes and sensor/noise settings.
        flat = cfg.scene.terrain_generator.sub_terrains["flat"]
        flat.proportion = 1.0
        cfg.scene.terrain_generator.sub_terrains = {"flat": flat}
        cfg.scene.terrain_generator.num_cols = 1
    result["diagnostic_overrides"] = {
        "num_envs": args.num_envs,
        "device": args.device,
        "scene_seed_from_agent": cfg.scene.seed,
        "headless": True,
        "terrain_selection": args.terrain,
    }
    mark("provenance_and_config")
    env = task_registry.get_task_class(args.task)(cfg, headless=True)
    env.validate_training_contract(agent_cfg.to_dict())
    layout = env.observation_layout
    n = env.num_actions
    ids = torch.arange(env.num_envs, device=env.device)
    zeros = torch.zeros(env.num_envs, n, device=env.device)
    robot = env.robot
    result["actual_articulation"] = {
        "spawn": spawn_info(robot.cfg.spawn),
        "joint_names_sim_order": robot.joint_names,
        "env0_default_stiffness": robot.data.default_joint_stiffness[0].cpu().tolist(),
        "env0_default_damping": robot.data.default_joint_damping[0].cpu().tolist(),
        "env0_effort_limits": robot.data.joint_effort_limits[0].cpu().tolist(),
        "env0_soft_joint_pos_limits": robot.data.soft_joint_pos_limits[0].cpu().tolist(),
        "env0_default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
    }
    required_sensors = ("height_scanner", "left_foot_scanner", "right_foot_scanner")
    for sensor_name in required_sensors:
        if sensor_name not in env.scene.sensors:
            raise RuntimeError("Required formal sensor missing: " + sensor_name)

    def scan_snapshot():
        return {
            name: {
                "hits": env.scene.sensors[name].data.ray_hits_w.clone(),
                "pos": env.scene.sensors[name].data.pos_w.clone(),
            }
            for name in required_sensors
        }

    def scan_report(snapshot):
        return {
            name: {
                "hits": stats(data["hits"]),
                "sensor_position": stats(data["pos"]),
                "first_env_hits": [
                    [float(v) if torch.isfinite(v) else None for v in point] for point in data["hits"][0, :10].cpu()
                ],
                "valid_ray_fraction": float(torch.isfinite(data["hits"]).all(-1).float().mean()),
            }
            for name, data in snapshot.items()
        }

    def observation_report(actor, critic):
        amp = env.get_amp_obs_for_expert_trans()
        expected = (1997, 2076, 70) if cfg.robot_spec.name == "g1" else (1937, 2016, 66)
        actual = (actor.shape[-1], critic.shape[-1], amp.shape[-1])
        if actual != expected:
            raise RuntimeError(f"Unexpected {cfg.robot_spec.name} widths: actual={actual}, expected={expected}")
        for name, value in (("actor", actor), ("critic", critic), ("amp", amp)):
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError("Nonfinite " + name)
        a_end = layout.proprio_dim * layout.actor_history
        c_end = layout.critic_frame_dim * layout.critic_history
        scan_width = layout.scan_dim * layout.scan_history
        cmd_start, cmd_end = env._proprio_field_slice("velocity_command")
        latest_actor = actor[:, a_end - layout.proprio_dim : a_end]
        latest_critic = critic[:, c_end - layout.critic_frame_dim : c_end]
        return {
            "actor": stats(actor),
            "critic": stats(critic),
            "amp": stats(amp),
            "returned_history_wiring": {
                "latest_actor_command": stats(latest_actor[:, cmd_start:cmd_end]),
                "latest_command_max_error": float(
                    (latest_actor[:, cmd_start:cmd_end] - env.command_generator.command * env.obs_scales.commands)
                    .abs()
                    .max()
                ),
                "latest_critic_root_velocity": stats(latest_critic[:, layout.proprio_dim : layout.proprio_dim + 3]),
                "latest_velocity_max_error": float(
                    (
                        latest_critic[:, layout.proprio_dim : layout.proprio_dim + 3]
                        - robot.data.root_lin_vel_b * env.obs_scales.lin_vel
                    )
                    .abs()
                    .max()
                ),
                "amp_q_max_error": float((amp[:, :n] - robot.data.joint_pos[:, env.amp_builder.joint_ids]).abs().max()),
                "amp_dq_max_error": float(
                    (amp[:, n : 2 * n] - robot.data.joint_vel[:, env.amp_builder.joint_ids]).abs().max()
                ),
            },
            "blocks": {
                "actor_proprio_history": stats(actor[:, :a_end]),
                "actor_scan_history": stats(actor[:, a_end : a_end + scan_width]),
                "actor_contact": stats(actor[:, a_end + scan_width :]),
                "critic_proprio_history": stats(critic[:, :c_end]),
                "critic_scan_history": stats(critic[:, c_end : c_end + scan_width]),
                "critic_foot_scan_and_immunity": stats(critic[:, c_end + scan_width :]),
                "amp_q": stats(amp[:, :n]),
                "amp_dq": stats(amp[:, n : 2 * n]),
                "amp_hands": stats(amp[:, 2 * n : 2 * n + 6]),
                "amp_feet": stats(amp[:, 2 * n + 6 :]),
            },
        }

    def command_velocity_evidence():
        current_actor, current_critic = env.compute_current_observations()
        start, end = env._proprio_field_slice("velocity_command")
        cmd = env.command_generator.command * env.obs_scales.commands
        vel = robot.data.root_lin_vel_b * env.obs_scales.lin_vel
        return {
            "command": stats(cmd),
            "actor_current_command": stats(current_actor[:, start:end]),
            "command_max_abs_error": float((current_actor[:, start:end] - cmd).abs().max()),
            "physical_root_velocity_body": stats(vel),
            "critic_current_root_velocity": stats(current_critic[:, layout.proprio_dim : layout.proprio_dim + 3]),
            "velocity_max_abs_error": float(
                (current_critic[:, layout.proprio_dim : layout.proprio_dim + 3] - vel).abs().max()
            ),
        }

    actor, extra = env.get_observations()
    result["initial_observations"] = observation_report(actor, extra["observations"]["critic"])
    result["order_and_functions"] = {
        "sim_joint_names": robot.joint_names,
        "policy_joint_names": list(env.policy_joint_names),
        "policy_to_sim_ids": list(env.policy_joint_ids),
        "amp_to_sim_ids": list(env.amp_builder.joint_ids),
        "amp_joint_names": [robot.joint_names[i] for i in env.amp_builder.joint_ids],
        "amp_hand_names": [robot.body_names[i] for i in env.amp_builder.hand_body_ids],
        "amp_foot_names": [robot.body_names[i] for i in env.amp_builder.foot_body_ids],
        "action_scale": stats(torch.as_tensor(env.action_scale, device=env.device)),
        "default_joint_pose": stats(robot.data.default_joint_pos),
        "env_step_file": inspect.getsourcefile(type(env).step),
        "current_obs_file": inspect.getsourcefile(type(env).compute_current_observations),
        "amp_compute_file": inspect.getsourcefile(type(env.amp_builder).compute),
        "sensor_class_files": {name: inspect.getsourcefile(type(env.scene.sensors[name])) for name in required_sensors},
    }
    if list(env.policy_joint_ids) != list(env.amp_builder.joint_ids):
        raise RuntimeError("Policy and AMP joint order differ")
    initial_scan = scan_snapshot()
    result["initial_scans"] = scan_report(initial_scan)
    result["initial_command_velocity"] = command_velocity_evidence()
    mark("initial_state")

    original_reset = env.reset
    pre_reset = {}
    stage = "zero_action"
    terminal_checked = 0
    terminal_different = 0
    terminal_examples = []
    timeout_mismatches = []
    reset_counts = {}

    critic_checked = 0
    history_checked = 0
    original_preview = env.preview_terminal_critic_observations

    def checked_preview(selected):
        nonlocal critic_checked, history_checked
        before = [
            (buf.buffer.clone(), buf._num_pushes.clone(), buf._pointer)
            for buf in (env.actor_obs_buffer, env.critic_obs_buffer, env.scan_obs_buffer)
        ]
        rng = torch.cuda.get_rng_state(env.device).clone()
        actual = original_preview(selected)
        assert torch.equal(rng, torch.cuda.get_rng_state(env.device)), "preview consumed actor RNG"
        for buf, (frames, pushes, pointer) in zip(
            (env.actor_obs_buffer, env.critic_obs_buffer, env.scan_obs_buffer), before
        ):
            assert torch.equal(buf.buffer, frames) and torch.equal(buf._num_pushes, pushes) and buf._pointer == pointer
        _, current = env.compute_current_observations()
        critic_copy = copy.deepcopy(env.critic_obs_buffer)
        scan_copy = copy.deepcopy(env.scan_obs_buffer)
        critic_copy.append(current)
        scan_copy.append(env.compute_teacher_terrain_privilege())
        parts = [critic_copy.buffer[selected].flatten(1), scan_copy.buffer[selected].flatten(1)]
        if env.append_critic_foot_scan:
            parts.append(env.compute_foot_scan_privilege()[selected])
        if env.append_critic_immunity:
            parts.append(env.impact_immunity[selected].float().unsqueeze(-1))
        expected = torch.cat(parts, -1).clamp(-env.clip_obs, env.clip_obs)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        critic_checked += len(selected)
        history_checked += 1
        return actual

    env.preview_terminal_critic_observations = checked_preview

    def capture_reset(reset_ids):
        pre_reset.clear()
        if len(reset_ids):
            pre_reset.update(
                ids=reset_ids.clone(),
                amp=env.get_amp_obs_for_expert_trans()[reset_ids].clone(),
                root=robot.data.root_pos_w[reset_ids].clone(),
            )
        history_before = env.prev_foot_lin_vel_w.clone()
        untouched = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        untouched[reset_ids] = False
        original_reset(reset_ids)
        torch.testing.assert_close(env.prev_foot_lin_vel_w[untouched], history_before[untouched], atol=0, rtol=0)
        if len(reset_ids):
            raw = robot.root_physx_view.get_link_velocities()[reset_ids][:, env.feet_body_ids, :3]
            torch.testing.assert_close(env.prev_foot_lin_vel_w[reset_ids], raw, atol=0, rtol=0)
            assert torch.all(env.foot_accel_ema[reset_ids] == 0)
        pre_reset["pushes_after_reset"] = env.critic_obs_buffer._num_pushes.clone()

    env.reset = capture_reset

    def take_step(actions, index):
        nonlocal terminal_checked, terminal_different
        actor, reward, dones, info = env.step(actions)
        if not bool(torch.isfinite(actor).all() and torch.isfinite(reward).all()):
            raise RuntimeError(f"Nonfinite actor/reward at {stage}:{index}")
        torch.testing.assert_close(env.critic_obs_buffer._num_pushes, pre_reset["pushes_after_reset"] + 1)
        assert torch.equal(info["bootstrap_mask"], info["time_outs"] & dones & ~info["terminated"])
        current_timeout = env.time_out_buf
        if "time_outs" not in info:
            raise RuntimeError("Runtime did not return time_outs")
        mismatch = info["time_outs"].bool() != current_timeout.bool()
        if bool(mismatch.any()) and len(timeout_mismatches) < 12:
            timeout_mismatches.append(
                {
                    "stage": stage,
                    "step": index,
                    "current": current_timeout.cpu().tolist(),
                    "returned": info["time_outs"].cpu().tolist(),
                    "dones": dones.cpu().tolist(),
                }
            )
        reset_counts[stage] = reset_counts.get(stage, 0) + int(dones.sum())
        if len(env.reset_env_ids):
            if "terminal_amp_obs" not in info or "amp" not in pre_reset:
                raise RuntimeError("Physical reset occurred without AMP terminal snapshot")
            terminal = info["terminal_amp_obs"]
            if not torch.equal(terminal, pre_reset["amp"]):
                raise RuntimeError("Terminal AMP differs from actual pre-reset physical snapshot")
            after = env.get_amp_obs_for_expert_trans()[env.reset_env_ids]
            terminal_checked += len(env.reset_env_ids)
            terminal_different += int((after != terminal).any(-1).sum())
            if len(terminal_examples) < 3 or stage == "forced_horizon":
                terminal_examples.append(
                    {
                        "stage": stage,
                        "step": index,
                        "ids": env.reset_env_ids.cpu().tolist(),
                        "terminal_amp": stats(terminal),
                        "post_reset_amp": stats(after),
                        "physical_terminal_root": stats(pre_reset["root"]),
                        "terminal_equals_physical_pre_reset": True,
                    }
                )
        elif "terminal_amp_obs" in info:
            raise RuntimeError("Stale terminal_amp_obs on non-reset step")
        return actor, reward, dones, info

    with torch.inference_mode():
        samples = []
        for step in range(args.steps):
            actor, reward, dones, info = take_step(zeros, step)
            if step % 100 == 0 or step == args.steps - 1:
                samples.append(
                    {
                        "step": step,
                        "reward": stats(reward),
                        "root_pos": stats(robot.data.root_pos_w),
                        "root_vel": stats(robot.data.root_lin_vel_b),
                        "foot_accel_ema": stats(env.foot_accel_ema),
                        "termination_flags": {k: int(v.sum()) for k, v in env.reset_reason_masks.items()},
                        "command_velocity": command_velocity_evidence(),
                    }
                )
                print(f"PROBE_ZERO_STEP {step}", flush=True)
        after_zero_scan = scan_snapshot()
        result["zero_action"] = {
            "steps": args.steps,
            "samples": samples,
            "final_observations": observation_report(actor, info["observations"]["critic"]),
            "final_scans": scan_report(after_zero_scan),
            "scan_changes": {
                name: delta_stats(initial_scan[name]["hits"], after_zero_scan[name]["hits"])
                for name in required_sensors
            },
        }
        mark("zero_action")

        stage = "joint_pulse"
        env.reset(ids)
        before_q = robot.data.joint_pos[:, env.policy_joint_ids].clone()
        pulses = zeros.clone()
        pulse_ids = ids[: min(n, env.num_envs - 1)]
        pulse_joint = pulse_ids % n
        pulses[pulse_ids, pulse_joint] = args.pulse_amplitude
        pulse_resets = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        for step in range(args.pulse_steps):
            actor, reward, dones, info = take_step(pulses, step)
            pulse_resets += dones.long()
        after_q = robot.data.joint_pos[:, env.policy_joint_ids].clone()
        expected_scatter = torch.zeros_like(env.action)
        expected_scatter[:, env.policy_joint_ids] = pulses
        result["joint_pulse"] = {
            "amplitude_policy_units": args.pulse_amplitude,
            "steps": args.pulse_steps,
            "actual_sim_action": stats(env.action),
            "actual_policy_action": stats(env.policy_action),
            "scatter_max_abs_error": float((env.action - expected_scatter).abs().max()),
            "per_stimulated_env": [
                {
                    "env_id": int(i),
                    "joint": env.policy_joint_names[int(j)],
                    "q_before": float(before_q[i, j]),
                    "q_after": float(after_q[i, j]),
                    "q_delta": float(after_q[i, j] - before_q[i, j]),
                    "qvel_after": float(robot.data.joint_vel[i, env.policy_joint_ids[int(j)]]),
                    "applied_torque_after": float(robot.data.applied_torque[i, env.policy_joint_ids[int(j)]]),
                    "episode_resets": int(pulse_resets[i]),
                }
                for i, j in zip(pulse_ids, pulse_joint)
            ],
            "zero_action_control_ids": ids[len(pulse_ids) :].cpu().tolist(),
            "control_q_deltas": stats((after_q - before_q)[len(pulse_ids) :]),
            "note": (
                "Different envs have different formal terrain/random states; this is response wiring, not a causal PD"
                " quality comparison"
            ),
        }
        mark("joint_pulse")

        stage = "partial_reset"
        subset = ids[:2]
        other_root = robot.data.root_state_w[2:].clone()
        partial = {
            "ids": subset.cpu().tolist(),
            "before_foot_velocity": stats(robot.data.body_lin_vel_w[subset][:, env.feet_body_ids]),
            "before_prev_foot_velocity": stats(env.prev_foot_lin_vel_w[subset]),
        }
        env.reset(subset)
        partial.update(
            after_foot_velocity=stats(robot.data.body_lin_vel_w[subset][:, env.feet_body_ids]),
            after_prev_foot_velocity=stats(env.prev_foot_lin_vel_w[subset]),
            prev_foot_cache_max_error=float(
                (env.prev_foot_lin_vel_w[subset] - robot.data.body_lin_vel_w[subset][:, env.feet_body_ids]).abs().max()
            ),
            ema_after_reset=stats(env.foot_accel_ema[subset]),
            other_env_root_max_delta=float((robot.data.root_state_w[2:] - other_root).abs().max()),
            foot_privilege_after_reset=stats(env.compute_foot_scan_privilege()[subset]),
            scans_after_reset=scan_report(scan_snapshot()),
        )
        actor, extra = env.get_observations()
        partial["post_reset_observations"] = observation_report(actor, extra["observations"]["critic"])
        actor, reward, dones, info = take_step(zeros, 0)
        partial["first_step_foot_accel_ema"] = stats(env.foot_accel_ema[subset])
        partial["first_step_dones"] = dones[subset].cpu().tolist()
        partial["command_velocity"] = command_velocity_evidence()
        result["partial_reset"] = partial
        mark("partial_reset")

        # Real articulation state write, no physics step or model mutation: check
        # the current production observation against two physically distinct root
        # velocities at the exact same pose/joints/history. Restore immediately.
        saved_velocity = robot.data.root_state_w[subset, 7:13].clone()
        try:
            test_velocity = saved_velocity.clone()
            test_velocity[:, :3] = 0.0
            robot.write_root_velocity_to_sim(test_velocity, env_ids=subset)
            actor_zero, critic_zero = env.compute_current_observations()
            actual_zero = robot.data.root_lin_vel_b[subset].clone()
            test_velocity[:, 0] = 0.7
            robot.write_root_velocity_to_sim(test_velocity, env_ids=subset)
            actor_moving, critic_moving = env.compute_current_observations()
            actual_moving = robot.data.root_lin_vel_b[subset].clone()
            result["real_articulation_velocity_counterexample"] = {
                "ids": subset.cpu().tolist(),
                "physics_steps": 0,
                "injection": "World x root velocity 0 versus 0.7m/s; other state held; saved velocity restored",
                "physical_body_velocity_zero": stats(actual_zero),
                "physical_body_velocity_moving": stats(actual_moving),
                "physical_velocity_change_norm": (actual_moving - actual_zero).norm(dim=-1).cpu().tolist(),
                "actor_current_max_delta": float((actor_moving[subset] - actor_zero[subset]).abs().max()),
                "critic_current_max_delta": float((critic_moving[subset] - critic_zero[subset]).abs().max()),
                "critic_velocity_delta": stats(
                    critic_moving[subset, layout.proprio_dim : layout.proprio_dim + 3]
                    - critic_zero[subset, layout.proprio_dim : layout.proprio_dim + 3]
                ),
            }
        finally:
            robot.write_root_velocity_to_sim(saved_velocity, env_ids=subset)
        mark("real_articulation_velocity_counterexample")

        stage = "forced_horizon"
        env.reset(ids)
        # Diagnostic buffer injection, explicitly separate from the natural rollout.
        # Guarantees the physical auto-reset/terminal path is exercised, even if all
        # zero-action episodes survived the preceding finite-duration probe.
        env.episode_length_buf[subset] = int(env.max_episode_length) - 1
        horizon_steps = []
        for step in range(4):
            actor, reward, dones, info = take_step(zeros, step)
            horizon_steps.append(
                {
                    "step": step,
                    "reset_ids": env.reset_env_ids.cpu().tolist(),
                    "current_timeout": env.time_out_buf.cpu().tolist(),
                    "returned_timeout": info["time_outs"].cpu().tolist(),
                }
            )
        if terminal_checked == 0:
            raise RuntimeError("Terminal AMP path was never exercised")
        result["terminal_and_timeout"] = {
            "terminal_snapshots_checked": terminal_checked,
            "terminal_distinct_from_reset": terminal_different,
            "terminal_examples": terminal_examples,
            "reset_counts": reset_counts,
            "timeout_mismatch_examples": timeout_mismatches,
            "all_observed_timeout_masks_consistent": not timeout_mismatches,
            "forced_horizon_steps": horizon_steps,
            "forced_horizon_injection_ids": subset.cpu().tolist(),
        }
        assert not timeout_mismatches
        assert not any(row["reset_ids"] for row in horizon_steps[1:]), "expected three no-reset steps after horizon"
        assert not any(any(row["returned_timeout"]) for row in horizon_steps[1:]), "stale timeout after horizon"
        # Inject root velocity before a physical step to produce a real acceleration failure at a horizon.
        # Preserve all termination thresholds and immunity rules.
        stage = "overlap"
        env.reset(ids)
        overlap_id = ids[:1]
        velocity = robot.data.root_state_w[overlap_id, 7:13].clone()
        velocity[:, 0] = 10.0
        robot.write_root_velocity_to_sim(velocity, env_ids=overlap_id)
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.impact_immunity[overlap_id] = False
        env.episode_length_buf[overlap_id] = env.max_episode_length - 1
        actor, reward, dones, info = take_step(zeros, 0)
        result["overlap_observed_reasons"] = {key: bool(value[0]) for key, value in env.reset_reason_masks.items()}
        save()
        assert info["time_outs"][0] and info["terminated"][0] and dones[0], result["overlap_observed_reasons"]
        assert not info["bootstrap_mask"][0], "physical failure must dominate timeout"
        result["physical_failure_timeout_overlap"] = {
            k: bool(info[k][0]) for k in ("time_outs", "terminated", "bootstrap_mask")
        }
        result["terminal_critic_checked"] = critic_checked
        result["history_preview_checked"] = history_checked
        assert critic_checked > 0 and history_checked > 0
        mark("terminal_and_timeout")
    result["ok"] = True
    result["ok_scope"] = (
        "F1/F2/F3 semantic assertions and fixed observation widths passed; diagnostic injections are not learned"
        " behavior"
    )
    save()


if __name__ == "__main__":
    code = 0
    try:
        main()
    except BaseException as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        save()
        traceback.print_exc()
        code = 1
    finally:
        threading.Timer(20, os._exit, args=(code,)).start()
        app.close()
        os._exit(code)

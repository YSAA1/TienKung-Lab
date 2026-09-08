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

"""Eight-env G1 flat reset-cache probe; run in tmux with --headless --output.

Imports the existing probe only for its AppLauncher/bootstrap/helpers. Its main
does not execute. No training instance or source file is modified.
"""

import json
import math
import os
import threading
import traceback
from pathlib import Path

import server_probe as p
import torch
from pxr import Usd

from legged_lab.utils import task_registry

result = {"ok": False, "scope": "Diagnostic instance: real foot velocity before/after reset, no training"}


def save():
    path = Path(p.args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def main():
    p.patch_missing_physx_material_attributes()
    cfg, agent = task_registry.get_cfgs(p.args.task)
    cfg.scene.num_envs = 8
    cfg.scene.seed = agent.seed
    cfg.device = cfg.sim.device = p.args.device
    flat = cfg.scene.terrain_generator.sub_terrains["flat"]
    flat.proportion = 1.0
    cfg.scene.terrain_generator.sub_terrains = {"flat": flat}
    cfg.scene.terrain_generator.num_cols = 1
    env = task_registry.get_task_class(p.args.task)(cfg, headless=True)
    robot = env.robot
    subset = torch.tensor([0, 1], device=env.device, dtype=torch.long)
    zeros = torch.zeros(8, env.num_actions, device=env.device)
    foot_ids = env.feet_body_ids
    # Deliberately require this API: cached/body velocity agreement cannot be
    # established by comparing two views of the same ArticulationData cache.
    view = robot.root_physx_view
    if not callable(getattr(view, "get_link_velocities", None)):
        raise RuntimeError("PhysX get_link_velocities unavailable: cannot validate cache independently")

    def snapshot():
        raw = view.get_link_velocities().clone()
        body = robot.data.body_lin_vel_w.clone()
        return {
            "root_state": robot.data.root_state_w[subset].cpu().tolist(),
            "joint_pos": robot.data.joint_pos[subset].cpu().tolist(),
            "joint_vel": robot.data.joint_vel[subset].cpu().tolist(),
            "body_velocity": body[subset].cpu().tolist(),
            "foot_velocity": body[subset][:, foot_ids].cpu().tolist(),
            "physx_link_velocities": raw[subset].cpu().tolist(),
            "physx_foot_velocities": raw[subset][:, foot_ids].cpu().tolist(),
            "prev_foot_velocity": env.prev_foot_lin_vel_w[subset].cpu().tolist(),
            "foot_accel_ema": env.foot_accel_ema[subset].cpu().tolist(),
        }

    result.update(
        num_envs=8,
        task=p.args.task,
        terrain="flat",
        seed=agent.seed,
        body_names=robot.body_names,
        foot_ids=foot_ids,
        foot_names=[robot.body_names[i] for i in foot_ids],
        dt=env.step_dt,
        physics_dt=env.physics_dt,
        source=p.source_info("legged_lab.locomotion.env"),
    )
    with torch.inference_mode():
        for _ in range(20):
            env.step(zeros)
        result["before_velocity_injection"] = snapshot()
        velocity = robot.data.root_state_w[subset, 7:13].clone()
        velocity[:, :3] = 0.0
        velocity[:, 0] = 2.0
        robot.write_root_velocity_to_sim(velocity, env_ids=subset)
        for raw_steps in range(1, 5):
            # Actual PhysX motion, deliberately no env.check_reset during injection.
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            excited_speed = robot.data.body_lin_vel_w[subset][:, foot_ids].norm(dim=-1)
            if bool((excited_speed.max(dim=-1).values > 0.8).all()):
                break
        result["raw_physics_substeps"] = raw_steps
        result["excited_foot_speed_mps"] = excited_speed.cpu().tolist()
        result["before_reset_after_raw_substeps"] = snapshot()
        torch.testing.assert_close(robot.data.body_lin_vel_w, view.get_link_velocities()[..., :3], atol=1e-5, rtol=1e-5)
        preserved = env.prev_foot_lin_vel_w[2:].clone()
        env.reset(subset)
        torch.testing.assert_close(env.prev_foot_lin_vel_w[2:], preserved, atol=0, rtol=0)
        torch.testing.assert_close(
            env.prev_foot_lin_vel_w[subset], view.get_link_velocities()[subset][:, foot_ids, :3], atol=0, rtol=0
        )
        result["immediately_after_reset"] = snapshot()
        old_prev = env.prev_foot_lin_vel_w[subset].clone()
        old_ema = env.foot_accel_ema[subset].clone()
        post_root_vel = robot.data.root_state_w[subset, 7:13].clone()
        post_joint_vel = robot.data.joint_vel[subset].clone()
        zero_kinematic_velocity = bool((post_root_vel == 0).all() and (post_joint_vel == 0).all())
        _, reward, dones, info = env.step(zeros)
        result["after_first_normal_step"] = snapshot()
        current_foot = robot.data.body_lin_vel_w[subset][:, foot_ids].clone()
        accel_old = (current_foot - old_prev) / env.step_dt
        accel_zero = current_foot / env.step_dt
        decay = math.exp(-env.step_dt / 0.06)
        theoretical_old = decay * old_ema + (accel_old.norm(dim=-1) - 30.0).clamp(min=0).sum(-1)
        theoretical_zero = decay * old_ema + (accel_zero.norm(dim=-1) - 30.0).clamp(min=0).sum(-1)
        result["first_step_comparison"] = {
            "dones": dones[subset].cpu().tolist(),
            "reward": reward[subset].cpu().tolist(),
            "post_reset_root_and_joint_velocities_all_zero": zero_kinematic_velocity,
            "raw_acceleration_using_stored_prev": accel_old.cpu().tolist(),
            "raw_acceleration_using_zero": accel_zero.cpu().tolist(),
            "predicted_ema_using_stored_prev": theoretical_old.cpu().tolist(),
            "predicted_ema_using_zero": theoretical_zero.cpu().tolist(),
            "observed_ema": env.foot_accel_ema[subset].cpu().tolist(),
            "interpretation_gate": (
                "If dones is true, post-step auto-reset invalidates direct EMA comparison; zero baseline applies only"
                " when reset root/joint velocities were all zero"
            ),
        }
        assert not dones[subset].any(), "auto-reset makes first-step comparison invalid"
        torch.testing.assert_close(env.foot_accel_ema[subset], theoretical_old, atol=1e-4, rtol=1e-5)
        if p.args.task == "g1_loco_teacher":
            assert zero_kinematic_velocity
            assert torch.all(env.foot_accel_ema[subset] == 0), "G1 gravity-only first step must have zero excess"
        else:
            assert old_prev.abs().max() > 0, "T4 nonzero reset velocity was not exercised"
        second_prev = env.prev_foot_lin_vel_w[subset].clone()
        second_ema = env.foot_accel_ema[subset].clone()
        second_velocity = robot.data.root_state_w[subset, 7:13].clone()
        second_velocity[:, 0] += 1.5
        robot.write_root_velocity_to_sim(second_velocity, env_ids=subset)
        _, _, second_done, _ = env.step(zeros)
        assert not second_done[subset].any(), "second-step auto-reset invalidates differential check"
        second_accel = (robot.data.body_lin_vel_w[subset][:, foot_ids] - second_prev) / env.step_dt
        expected_second = decay * second_ema + (second_accel.norm(dim=-1) - 30).clamp(min=0).sum(-1)
        torch.testing.assert_close(env.foot_accel_ema[subset], expected_second, atol=1e-4, rtol=1e-5)
        assert torch.all(
            expected_second > decay * second_ema
        ), "second-step acceleration contribution was not exercised"
        result["second_step"] = dict(
            expected=expected_second.cpu().tolist(), observed=env.foot_accel_ema[subset].cpu().tolist()
        )
        for _ in range(3):
            untouched = env.prev_foot_lin_vel_w[2:].clone()
            env.reset(subset)
            torch.testing.assert_close(env.prev_foot_lin_vel_w[2:], untouched, atol=0, rtol=0)
            torch.testing.assert_close(
                env.prev_foot_lin_vel_w[subset], view.get_link_velocities()[subset][:, foot_ids, :3], atol=0, rtol=0
            )
            assert torch.all(env.foot_accel_ema[subset] == 0)
        result["consecutive_partial_resets_checked"] = 3
    try:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        root_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        if not root_prim.IsValid():
            raise RuntimeError("env_0 Robot prim not found")
        attrs = []
        for prim in Usd.PrimRange(root_prim):
            attr = prim.GetAttribute("physxArticulation:enabledSelfCollisions")
            if attr.IsValid():
                attrs.append(
                    {
                        "prim_path": str(prim.GetPath()),
                        "value": attr.Get(),
                        "has_authored_value_opinion": attr.HasAuthoredValueOpinion(),
                    }
                )
        result["usd_self_collision_attributes"] = attrs
        if not attrs:
            result["usd_self_collision_note"] = "No matching attribute found; not interpreted as disabled"
    except Exception as exc:
        result["usd_self_collision_query_error"] = f"{type(exc).__name__}: {exc}"
    result["ok"] = True
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
        p.app.close()
        os._exit(code)

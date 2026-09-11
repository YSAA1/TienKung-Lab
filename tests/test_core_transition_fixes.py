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

"""Behavioral counterexamples for the core locomotion repair (CPU)."""

import ast
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rsl_rl"))
from rsl_rl.algorithms.amp_ppo import AMPPPO
from rsl_rl.modules import ActorCritic, Discriminator
from rsl_rl.utils import Normalizer
from rsl_rl.utils.motion_loader import AMPLoader


def production_method(path, name, class_name=None, **symbols):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = (
        tree.body
        if class_name is None
        else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name).body
    )
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    scope = {"torch": torch, "math": math, "SceneEntityCfg": lambda name: NS(name=name), **symbols}
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), scope)
    return scope[name]


def make_alg(std_type="scalar", initial=1.0, floor=None):
    policy = ActorCritic(
        2, 2, 2, actor_hidden_dims=[4], critic_hidden_dims=[4], noise_std_type=std_type, init_noise_std=initial
    )
    alg = AMPPPO(
        policy,
        Discriminator(4, 0.3, [4], "cpu", 0.7),
        None,
        None,
        min_std=floor,
        amp_replay_buffer_size=8,
        device="cpu",
        gamma=0.99,
    )
    alg.init_storage("rl", 2, 1, [2], [2], [2])
    return alg


def test_timeout_uses_terminal_value_and_failure_overlap_is_not_bootstrapped():
    alg = make_alg()
    obs = torch.ones(2, 2)
    alg.act(obs, obs, obs)
    alg.transition.values.fill_(10)
    infos = {
        "time_outs": torch.tensor([True, True]),
        "bootstrap_mask": torch.tensor([True, False]),
        "terminal_values": torch.tensor([2.0]),
    }
    alg.process_env_step(torch.ones(2), torch.ones(2, dtype=torch.bool), infos, obs)
    torch.testing.assert_close(alg.storage.rewards[0, :, 0], torch.tensor([2.98, 1.0]))
    alg.compute_returns(torch.full((2, 2), 1000.0))
    torch.testing.assert_close(alg.storage.returns[0, :, 0], torch.tensor([2.98, 1.0]))


@pytest.mark.parametrize("kind", ["scalar", "log"])
def test_floor_initialization_load_and_optimizer_projection(kind):
    alg = make_alg(kind, 0.01, torch.tensor([0.05, 0.05]))
    policy = alg.policy
    for operation in ("initial", "load", "update"):
        if operation == "load":
            state = policy.state_dict()
            state["std" if kind == "scalar" else "log_std"].fill_(0.01 if kind == "scalar" else math.log(0.01))
            policy.load_state_dict(state)
        elif operation == "update":
            parameter = policy.std if kind == "scalar" else policy.log_std
            with torch.no_grad():
                parameter.fill_(-10)
            alg.enforce_min_std()
        policy.update_distribution(torch.zeros(2, 2))
        torch.testing.assert_close(policy.action_std, torch.full((2, 2), 0.05))


def test_heading_ignores_standing_and_nonheading_targets():
    fn = production_method(
        "legged_lab/mdp/rewards.py", "heading_error", math_utils=NS(wrap_to_pi=lambda x: torch.atan2(x.sin(), x.cos()))
    )
    command = NS(
        heading_target=torch.tensor([0.0, math.pi, math.pi, 1.0]),
        is_heading_env=torch.tensor([True, True, False, True]),
        is_standing_env=torch.tensor([True, True, False, False]),
    )
    env = NS(
        num_envs=4, device="cpu", command_generator=command, scene={"robot": NS(data=NS(heading_w=torch.zeros(4)))}
    )
    torch.testing.assert_close(fn(env), torch.tensor([0.0, 0.0, 0.0, 1.0]))


def loader(tmp_path, step=0.2):
    path = tmp_path / "linear.json"
    path.write_text(json.dumps({"Frames": [[0.0], [1.0], [2.0], [3.0]], "FrameDuration": 0.1, "MotionWeight": 1.0}))
    return AMPLoader("cpu", step, 1, [str(path)])


def test_loader_frame_time_and_endpoint_scalar_batch(tmp_path):
    data = loader(tmp_path)
    for time, expected in ((0.0, 0.0), (0.1, 1.0), (0.15, 1.5), (0.3, 3.0)):
        for fn in (data.get_frame_at_time, data.get_full_frame_at_time):
            torch.testing.assert_close(fn(0, time), torch.tensor([expected]))
        for fn in (data.get_frame_at_time_batch, data.get_full_frame_at_time_batch):
            torch.testing.assert_close(fn(np.array([0]), np.array([time])), torch.tensor([[expected]]))


def test_loader_uniform_start_without_zero_atom(tmp_path):
    data = loader(tmp_path)
    np.random.seed(42)
    times = data.traj_time_sample_batch(np.zeros(10000, dtype=int))
    assert times.min() > 0 and times.max() <= 0.1 + 1e-12
    assert abs(times.mean() - 0.05) < 0.001
    np.random.seed(42)
    assert data.traj_time_sample(0) == pytest.approx(times[0])


def test_loader_rejects_too_short_clip(tmp_path):
    with pytest.raises(ValueError, match="transition"):
        loader(tmp_path, step=0.4)


def test_reset_seeds_physical_velocity_after_forward_and_preserves_other_envs():
    reset = production_method("legged_lab/locomotion/env.py", "reset", "LocomotionEnv")
    ids = torch.tensor([0, 2])
    physical = torch.zeros(3, 2, 6)
    physical[2, :, :3] = 0.4

    def noop(*args, **kwargs):
        return None

    buffer = NS(reset=noop)
    env = NS(
        num_envs=3,
        step_dt=0.02,
        sim_step_counter=10,
        feet_body_ids=[0, 1],
        progress_monitor=None,
        cfg=NS(scene=NS(terrain_generator=None), sim=NS(decimation=4)),
        extras={},
        scene=NS(reset=noop, write_data_to_sim=noop),
        event_manager=NS(available_modes=[]),
        reward_manager=NS(reset=lambda _: {}),
        command_generator=NS(reset=noop),
        command_provenance_log=lambda: {},
        _resample_terrain_aware_commands=noop,
        _enforce_terrain_aware_commands=noop,
        actor_obs_buffer=buffer,
        critic_obs_buffer=buffer,
        scan_obs_buffer=buffer,
        scan_occlusion_masks=None,
        action_buffer=buffer,
        robot=NS(
            data=NS(body_lin_vel_w=torch.full((3, 2, 3), 9.0), root_pos_w=torch.zeros(3, 3)),
            root_physx_view=NS(get_link_velocities=lambda: physical),
        ),
        sim=NS(forward=noop),
        prev_step_root_pos_w=torch.zeros(3, 3),
    )
    for name in (
        "collapse_low_steps",
        "avg_feet_force_per_step",
        "avg_feet_speed_per_step",
        "foot_accel_ema",
        "episode_length_buf",
        "episode_max_radial_dist",
        "episode_path_length",
        "episode_tracking_sum",
        "episode_tracking_steps",
        "gait_time",
        "time_out_buf",
    ):
        setattr(env, name, torch.ones(3))
    env.prev_foot_lin_vel_w = torch.full((3, 2, 3), 9.0)
    reset(env, ids)
    torch.testing.assert_close(env.prev_foot_lin_vel_w[ids], physical[ids, :, :3])
    assert torch.all(env.prev_foot_lin_vel_w[1] == 9)
    assert env.foot_accel_ema.tolist() == [0.0, 1.0, 0.0]
    penalty = production_method("legged_lab/locomotion/env.py", "update_foot_accel_penalty", "LocomotionEnv")
    env.robot.data.body_lin_vel_w = env.prev_foot_lin_vel_w.clone()
    assert penalty(env)[0] == 0
    env.robot.data.body_lin_vel_w[0, :, 0] += 1
    assert penalty(env)[0] > 0


@pytest.mark.parametrize("length", [1, 5])
@pytest.mark.parametrize("pushes", [0, 1, 3, 8])
def test_terminal_history_preview_order_empty_partial_full(length, pushes):
    fn = production_method("legged_lab/locomotion/env.py", "_preview_history", "LocomotionEnv")
    history = NS(
        buffer=torch.arange(3 * length * 2).reshape(3, length, 2).float(),
        current_length=torch.tensor([pushes, 1, pushes]),
    )
    before = history.buffer.clone()
    current = torch.tensor([[100.0, 101.0], [200.0, 201.0], [300.0, 301.0]])
    ids = torch.tensor([0, 2])
    actual = fn(history, current, ids).reshape(2, length, 2)
    expected = torch.cat((before[ids, 1:], current[ids, None]), dim=1)
    if pushes == 0:
        expected[:] = current[ids, None]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(history.buffer, before)
    assert history.current_length.tolist() == [pushes, 1, pushes]


def test_required_scanner_missing_is_not_valid_zero_observation():
    fn = production_method("legged_lab/locomotion/env.py", "compute_foot_scan_privilege", "LocomotionEnv")
    env = NS(
        scene=NS(sensors={"left_foot_scanner": object()}),
        append_critic_foot_scan=True,
        num_envs=2,
        device="cpu",
        observation_layout=NS(foot_scan_dim=4),
    )
    with pytest.raises(RuntimeError, match="foot scanners"):
        fn(env)
    env.append_critic_foot_scan = False
    assert fn(env).shape == (2, 8)


@pytest.mark.parametrize("kind", ["scalar", "log"])
def test_std_above_floor_is_unchanged(kind):
    alg = make_alg(kind, 0.7, torch.tensor([0.05, 0.05]))
    alg.policy.update_distribution(torch.zeros(2, 2))
    torch.testing.assert_close(alg.policy.action_std, torch.full((2, 2), 0.7))


@pytest.mark.parametrize("kind", ["scalar", "log"])
def test_actual_amp_optimizer_update_cannot_cross_std_floor(kind):
    alg = make_alg(kind, 0.06, torch.tensor([0.05, 0.05]))
    raw = torch.tensor([[1.0, 2.0], [2.0, 3.0]])
    alg.amp_data = NS(feed_forward_generator=lambda *args: iter([(raw, raw + 0.1)]))
    alg.amp_normalizer = Normalizer(2)
    # A diagnostic large negative entropy coefficient deliberately collapses std.
    alg.entropy_coef = -1000.0
    for group in alg.optimizer.param_groups:
        group["lr"] = 10.0
    alg.act(raw, raw, raw)
    alg.process_env_step(torch.ones(2), torch.zeros(2, dtype=torch.bool), {}, raw + 0.1)
    alg.compute_returns(raw)
    alg.update()
    std = alg.policy.std if kind == "scalar" else alg.policy.log_std.exp()
    torch.testing.assert_close(std, torch.full((2,), 0.05))


def test_timeout_is_replaced_on_three_no_reset_control_steps():
    step = production_method("legged_lab/locomotion/env.py", "step", "LocomotionEnv", STANDING_COMMAND_THRESHOLD=0.1)

    def noop(*args, **kwargs):
        return None

    zeros = torch.zeros(2)
    false = torch.zeros(2, dtype=torch.bool)
    contact = NS(data=NS(net_forces_w=torch.zeros(2, 2, 3)))
    robot = NS(
        data=NS(default_joint_pos=torch.zeros(2, 2), body_lin_vel_w=torch.zeros(2, 2, 3), root_pos_w=torch.zeros(2, 3)),
        set_joint_position_target=noop,
    )
    env = NS(
        progress_monitor=None,
        action_buffer=NS(compute=lambda x: x),
        clip_actions=1.0,
        device="cpu",
        action=torch.zeros(2, 2),
        policy_joint_ids=[0, 1],
        action_scale=0.25,
        robot=robot,
        avg_feet_force_per_step=torch.zeros(2, 2),
        avg_feet_speed_per_step=torch.zeros(2, 2),
        _has_rtx_sensors=lambda: False,
        _has_warp_depth_camera=lambda: False,
        cfg=NS(sim=NS(decimation=1)),
        step_dt=0.02,
        physics_dt=0.02,
        sim_step_counter=0,
        scene=NS(write_data_to_sim=noop, update=noop, env_origins=torch.zeros(2, 3)),
        sim=NS(step=noop),
        contact_sensor=contact,
        feet_cfg=NS(body_ids=[0, 1]),
        feet_body_ids=[0, 1],
        headless=True,
        episode_length_buf=zeros.clone(),
        use_lightlp_terminations=False,
        episode_max_radial_dist=zeros.clone(),
        prev_step_root_pos_w=torch.zeros(2, 3),
        episode_path_length=zeros.clone(),
        _gait_tracking_scale=lambda: zeros,
        command_generator=NS(command=torch.zeros(2, 3), compute=noop),
        episode_tracking_sum=zeros.clone(),
        episode_tracking_steps=zeros.clone(),
        _update_gait=noop,
        _enforce_terrain_aware_commands=noop,
        event_manager=NS(available_modes=[]),
        check_reset=lambda: (false.clone(), false.clone()),
        terminated_buf=false.clone(),
        reward_manager=NS(compute=lambda _: zeros.clone()),
        extras={"time_outs": ~false},
        reset=noop,
        _collapse_step_log={},
        compute_observations=lambda: (torch.zeros(2, 3), torch.zeros(2, 4)),
    )
    step.__globals__["rtx_render_due"] = lambda *args: False
    for _ in range(3):
        _, _, dones, info = step(env, torch.zeros(2, 2))
        assert not dones.any()
        assert not info["time_outs"].any()
        assert not info["bootstrap_mask"].any()

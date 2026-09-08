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

"""Exercise opt-in termination and gait behavior without importing Isaac."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
from test_core_transition_fixes import production_method

from legged_lab.locomotion.mdp import sparse_signals as sig

ROOT = Path(__file__).resolve().parents[1]


def test_profile_rejects_other_robots_before_mutation():
    spec = importlib.util.spec_from_file_location("motion_profile", ROOT / "legged_lab/envs/g1/motion_experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = NS(robot_spec=NS(name="t4"))
    with pytest.raises(ValueError, match="G1-specific"):
        module.apply_vital_motion_experiment(cfg, "vital_v1")
    assert list(vars(cfg)) == ["robot_spec"]


@pytest.mark.parametrize(
    ("profile", "action_rate", "accel_enabled", "fall_limits", "gait_gate"),
    [
        ("vital_v1", -0.01, False, (0.8, 1.0), False),
        ("vital_termination_only", -0.1, False, (0.8, 1.0), True),
        ("vital_gait_gate_off_only", -0.1, True, None, False),
        ("vital_action_rate_only", -0.01, True, None, True),
    ],
)
def test_motion_experiment_profiles_are_single_factor(profile, action_rate, accel_enabled, fall_limits, gait_gate):
    spec = importlib.util.spec_from_file_location("motion_profile", ROOT / "legged_lab/envs/g1/motion_experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    weights = {
        "action_rate_l2": -0.1,
        "upright_orientation": 1.0,
        "velocity_slack": 1.5,
        "track_ang_vel_z_exp": 2.0,
        "body_orientation_l2": 0.0,
        "gait_feet_frc_perio": 1.0,
        "gait_feet_frc_support_perio": 0.6,
        "termination_penalty": 0.0,
    }
    cfg = NS(
        robot_spec=NS(name="g1"),
        acceleration_termination_enabled=True,
        deterministic_fall_limits=None,
        gait=NS(tracking_gate_enabled=True),
        reward=NS(**{k: NS(weight=v) for k, v in weights.items()}),
    )
    module.apply_vital_motion_experiment(cfg, profile)
    actual = {k: v.weight for k, v in vars(cfg.reward).items()}
    assert actual == {**weights, "action_rate_l2": action_rate}
    assert cfg.acceleration_termination_enabled is accel_enabled
    assert cfg.deterministic_fall_limits == fall_limits
    assert cfg.gait.tracking_gate_enabled is gait_gate
    assert cfg.lightlp_promotion_distance == "max_radial"
    assert cfg.progress_monitor_enabled is True
    assert cfg.sparse_command_min_speed_scale == 1.0


@pytest.mark.parametrize("experimental", [False, True])
def test_accel_override_preserves_timeout_and_physical_failure(experimental):
    fn = production_method(
        "legged_lab/locomotion/env.py",
        "_check_reset_lightlp",
        "LocomotionEnv",
        lightlp_timeout_and_reset=sig.lightlp_timeout_and_reset,
        mask_recent_push_accel=sig.mask_recent_push_accel,
        tilt_from_upright_rad=sig.tilt_from_upright_rad,
        euler_xyz_from_quat=lambda q: (q[:, 0], q[:, 1], q[:, 2]),
    )
    zero = torch.zeros(4)
    false = torch.zeros(4, dtype=torch.bool)
    vel = torch.zeros(4, 3)
    vel[0, 0] = 1.0  # 50m/s2, upright: only baseline should terminate.
    quat = torch.zeros(4, 4)
    quat[1, 0] = 0.9  # stubbed Euler roll, above the experimental .8 limit.
    env = NS(
        cfg=NS(
            scene=NS(terrain_generator=NS(size=(8, 8))),
            sim=NS(decimation=4),
            acceleration_termination_enabled=not experimental,
            deterministic_fall_limits=(0.8, 1.0) if experimental else None,
        ),
        robot=NS(
            data=NS(
                root_pos_w=torch.zeros(4, 3),
                root_lin_vel_w=vel,
                root_quat_w=quat,
                projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(4, 1),
                joint_vel=torch.zeros(4, 2),
            )
        ),
        scene=NS(env_origins=torch.zeros(4, 3)),
        contact_sensor=NS(data=NS(net_forces_w_history=torch.zeros(4, 1, 1, 3))),
        termination_contact_cfg=NS(body_ids=[]),
        prev_root_lin_vel_w=torch.zeros(4, 3),
        step_dt=0.02,
        last_root_accel_mps2=zero.clone(),
        last_tilt_rad=zero.clone(),
        sim_step_counter=8,
        _push_step_marker=torch.full((4,), -1000),
        episode_length_buf=torch.tensor([60, 60, 1000, 60]),
        max_episode_length=1000,
        num_envs=4,
        device="cpu",
        impact_immunity=false.clone(),
        pit_fall_buf=false.clone(),
        sparse_foothold_type_ids=[],
        refresh_sparse_tile_mask=lambda: false,
        feet_body_ids=[],
    )
    done, timeout = fn(env)
    assert done.tolist() == ([False, True, True, False] if experimental else [True, False, True, False])
    assert timeout.tolist() == [False, False, True, False]
    assert env.terminated_buf.tolist() == ([False, True, False, False] if experimental else [True, False, False, False])
    assert float(env.last_root_accel_mps2[0]) == 50.0


@pytest.mark.parametrize("gate", [False, True])
def test_gait_experiment_preserves_standing_and_sparse_exclusions(gate):
    fn = production_method(
        "legged_lab/locomotion/env.py", "_update_gait", "LocomotionEnv", STANDING_COMMAND_THRESHOLD=0.1
    )
    gait = NS(tracking_gate_enabled=gate, mode="fixed_clock")
    env = NS(
        cfg=NS(gait=gait),
        command_generator=NS(command=torch.tensor([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [0.7, 0.0, 0.0]])),
        _gait_tracking_scale=lambda: torch.tensor([0.0, 0.001, 0.001]),
        gait_time=torch.zeros(3),
        step_dt=0.02,
        gait_cycle=torch.ones(3),
        gait_phase=torch.zeros(3, 2),
        phase_offset=torch.zeros(3, 2),
        num_envs=3,
        device="cpu",
        sparse_foothold_type_ids=[1],
        sparse_tile_mask=torch.tensor([False, False, True]),
        refresh_sparse_tile_mask=lambda: None,
    )
    fn(env)
    torch.testing.assert_close(env.gait_reward_scale, torch.tensor([0.0, 0.001 if gate else 1.0, 0.0]))

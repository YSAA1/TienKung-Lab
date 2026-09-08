# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""Isaac-free contracts for S12 sparse terrain-aware commands."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SIG_PATH = Path(__file__).resolve().parents[1] / "legged_lab" / "envs" / "t4" / "mdp" / "sparse_signals.py"
_spec = importlib.util.spec_from_file_location("t4_sparse_signals_cmd", _SIG_PATH)
sig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sig)

CFG = Path(__file__).resolve().parents[1] / "legged_lab" / "locomotion" / "teacher_cfg.py"
ENV = Path(__file__).resolve().parents[1] / "legged_lab" / "locomotion" / "env.py"
EVAL = Path(__file__).resolve().parents[1] / "legged_lab" / "scripts" / "eval_locomotion.py"
PLAY = Path(__file__).resolve().parents[1] / "legged_lab" / "scripts" / "play.py"


def test_sparse_foothold_command_is_forward_only_with_discrete_straight_yaw():
    vx, vy, wz = sig.sample_sparse_foothold_velocity(vx_u=0.0, yaw_mode_u=0.0, yaw_u=0.5)
    assert vx == pytest.approx(0.6)
    assert vy == 0.0
    assert wz == 0.0

    vx, vy, wz = sig.sample_sparse_foothold_velocity(vx_u=1.0, yaw_mode_u=0.59, yaw_u=1.0)
    assert vx == pytest.approx(2.0)
    assert vy == 0.0
    assert wz == 0.0

    vx, vy, wz = sig.sample_sparse_foothold_velocity(vx_u=0.5, yaw_mode_u=0.59, yaw_u=0.0)
    assert vx == pytest.approx(1.3)
    assert wz == 0.0

    vx, vy, wz = sig.sample_sparse_foothold_velocity(vx_u=0.5, yaw_mode_u=0.60, yaw_u=0.0)
    assert wz == pytest.approx(-0.3)

    vx, vy, wz = sig.sample_sparse_foothold_velocity(vx_u=0.5, yaw_mode_u=0.60, yaw_u=1.0)
    assert wz == pytest.approx(0.3)


def test_overlay_keeps_nonsparse_omni_commands():
    commands = np.array([[-0.4, 0.3, 1.0], [0.1, -0.2, -0.5]], dtype=float)
    out = sig.overlay_sparse_foothold_commands(
        commands,
        is_sparse=np.array([True, False]),
        vx_u=np.array([0.0, 0.0]),
        yaw_mode_u=np.array([0.0, 0.0]),
        yaw_u=np.array([0.0, 0.0]),
    )
    np.testing.assert_allclose(out[0], [0.6, 0.0, 0.0])
    np.testing.assert_allclose(out[1], [0.1, -0.2, -0.5])


def test_pinned_command_ranges_disable_terrain_aware_overlay():
    assert (
        sig.terrain_aware_commands_enabled(
            enabled=True, lin_vel_x=(0.7, 0.7), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
        )
        is False
    )
    assert (
        sig.terrain_aware_commands_enabled(
            enabled=True, lin_vel_x=(-0.6, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.57, 1.57)
        )
        is True
    )
    assert (
        sig.terrain_aware_commands_enabled(
            enabled=False, lin_vel_x=(-0.6, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.57, 1.57)
        )
        is False
    )


def test_sparse_teacher_cfg_enables_s12_command_and_full_level_reset():
    source = CFG.read_text()
    sparse = source.split("class LightLPLocomotionEnvCfg", 1)[1]
    teacher = source.split("class AmpLocomotionEnvCfg", 1)[1].split("class T4LocoTeacherAgentCfg", 1)[0]
    rewards = source.split("class LightLPRewardCfg", 1)[1].split("@configclass", 1)[0]

    assert "random_level_reset_max_level: int | None = None" in sparse
    assert "self.random_level_reset_max_level = None" in sparse
    assert "self.random_level_reset_min_level = None" in sparse
    assert "self.random_level_reset_max_level = 4" not in sparse
    assert "terrain_aware_commands: bool = True" in sparse
    assert "sparse_command_lin_vel_x" in sparse
    assert "(0.6, 2.0)" in sparse
    assert "self.commands.ranges.lin_vel_x = (-0.6, 2.0)" in sparse
    assert "sparse_command_straight_yaw_prob: float = 0.60" in sparse
    assert "self.sparse_command_straight_yaw_prob = 0.60" in sparse
    assert "self.sparse_command_gentle_ang_vel_z = (-0.3, 0.3)" in sparse
    assert "(-0.3, 0.3)" in sparse
    assert 'run_name = "t_sparse_lightlp_s12_rim_yaw40"' in (Path(__file__).resolve().parents[1] / "legged_lab/envs/t4/teacher_cfg.py").read_text()
    assert "weight=-2.0" in rewards
    ori_block = rewards.split("body_orientation_l2", 1)[1].split("upright_orientation", 1)[0]
    upright_block = rewards.split("upright_orientation", 1)[1].split("undesired_contacts", 1)[0]
    assert "weight=0.0" in ori_block
    assert "weight=1.0" in upright_block
    assert "lin_vel_x=(-0.6, 1.0)" in teacher
    assert "rel_standing_envs=0.2" in teacher
    assert "LIGHTLP_ACCEL_LIMIT" not in teacher


def test_env_enforces_sparse_commands_after_heading_compute():
    source = ENV.read_text()
    assert "_resample_terrain_aware_commands" in source
    assert "_enforce_terrain_aware_commands" in source
    assert "sample_sparse_foothold_velocity(" in source
    assert "terrain_aware_commands_enabled(" in source
    step = source.split("def step(", 1)[1].split("def _resample_impact_immunity", 1)[0]
    assert "self.command_generator.compute(self.step_dt)" in step
    assert "self._enforce_terrain_aware_commands()" in step
    assert step.find("self.command_generator.compute(self.step_dt)") < step.find(
        "self._enforce_terrain_aware_commands()"
    )


def test_sparse_yaw_mode_has_sixty_percent_straight_mass():
    rng = np.random.default_rng(0)
    n = 20000
    vx, vy, wz = sig.sample_sparse_foothold_velocity(rng.random(n), rng.random(n), rng.random(n))
    assert np.all(vy == 0.0)
    assert vx.min() >= 0.6 - 1.0e-9
    assert vx.max() <= 2.0 + 1.0e-9
    straight = np.abs(wz) < 1.0e-12
    assert straight.mean() == pytest.approx(0.60, abs=0.01)
    gentle = wz[~straight]
    assert gentle.min() >= -0.3 - 1.0e-9
    assert gentle.max() <= 0.3 + 1.0e-9


def test_env_clears_heading_and_standing_on_sparse_overlay():
    source = ENV.read_text()
    enforce = source.split("def _enforce_terrain_aware_commands", 1)[1].split("def _batch_terrain_outcome_log", 1)[0]
    assert "is_standing_env" in enforce
    assert "is_heading_env" in enforce
    assert "heading_target" in enforce
    assert "standing[mask] = False" in enforce
    assert "heading_env[mask] = False" in enforce
    reset = source.split("def reset(", 1)[1].split("def update_terrain_levels", 1)[0]
    assert "self.command_generator.reset(env_ids)" in reset
    assert reset.find("self.command_generator.reset(env_ids)") < reset.find("self._resample_terrain_aware_commands")
    assert reset.find("self._resample_terrain_aware_commands") < reset.find("self._enforce_terrain_aware_commands")
    assert 'self.extras.pop("log", None)' in source


def test_eval_and_play_disable_terrain_aware_commands():
    eval_src = EVAL.read_text()
    play_src = PLAY.read_text()
    assert "terrain_aware_commands = False" in eval_src
    assert "terrain_aware_commands = False" in play_src
    assert "heading_command = False" in play_src
    assert "ranges.ang_vel_z = (0.0, 0.0)" in play_src
    assert "ranges.heading = (0.0, 0.0)" in play_src

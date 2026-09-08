# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""Isaac-free contracts for LightLP sparse rewards and observation dims."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SIG_PATH = Path(__file__).resolve().parents[1] / "legged_lab" / "envs" / "t4" / "mdp" / "sparse_signals.py"
_spec = importlib.util.spec_from_file_location("t4_sparse_signals", _SIG_PATH)
sig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sig)

from legged_lab.assets.t4 import schemas


def test_velocity_slack_band_and_standstill():
    assert sig.velocity_slack(0.8, 0.8) == 1.0
    assert sig.velocity_slack(0.8, 0.23) == 0.0
    assert sig.velocity_slack(0.8, 0.24) == 1.0
    assert sig.velocity_slack(0.8, 1.2) == 1.0
    assert sig.velocity_slack(0.8, 1.21) == 0.0
    assert sig.velocity_slack(0.0, 0.4) == 0.0


def test_illegal_footstep_fraction():
    assert sig.illegal_footstep_fraction(0.12, [0.12, 0.11, 0.12], in_contact=True) == 0.0
    assert sig.illegal_footstep_fraction(0.12, [-2.0, -2.0, 0.12], in_contact=True) == 2.0 / 3.0
    assert sig.illegal_footstep_fraction(0.12, [-2.0, -2.0], in_contact=False) == 0.0
    assert sig.illegal_footstep_fraction(0.12, [], in_contact=True) == 0.0
    assert sig.illegal_footstep_fraction(0.12, [float("nan"), 0.12], in_contact=True) == 0.5


def test_opposite_direction_and_foot_accel_ema():
    assert sig.opposite_direction((0.8, 0.0), (-0.2, 0.0)) == pytest.approx(0.2)
    assert sig.opposite_direction((0.8, 0.0), (0.2, 0.0)) == 0.0
    assert sig.opposite_direction((0.0, 0.0), (-1.0, 0.0)) == 0.0
    first = sig.foot_accel_ema_step(0.0, (40.0, 40.0), tau_s=0.06, dt_s=0.02, threshold_mps2=30.0)
    assert first == pytest.approx(20.0)
    second = sig.foot_accel_ema_step(first, (40.0, 40.0), tau_s=0.06, dt_s=0.02, threshold_mps2=30.0)
    decay = __import__("math").exp(-0.02 / 0.06)
    assert second == pytest.approx(decay * 20.0 + 20.0)
    assert sig.illegal_from_foot_fractions(0.5, 0.0) == 0.5
    assert sig.illegal_from_foot_fractions(0.5, 0.5) == 1.0
    flags = sig.random_level_reset_mask([0.05, 0.2, 0.09], fraction=0.10)
    assert flags == [True, False, True]
    assert sig.random_level_reset_high(10, 4) == 4
    assert sig.random_level_reset_high(3, 4) == 3
    assert sig.random_level_reset_high(10, None) == 10
    assert sig.random_level_reset_high(0, 4) == 1
    assert sig.random_level_reset_low(None, 10) == 0
    assert sig.random_level_reset_low(6, 10) == 6
    assert sig.random_level_reset_low(10, 10) == 9
    assert sig.random_level_reset_low(-1, 10) == 0
    promotion, timeout_success, fall = sig.monitor_outcome_flags(
        move_up=np.array([True, True, False]),
        timed_out=np.array([True, False, False]),
    )
    np.testing.assert_array_equal(promotion, np.array([True, True, False]))
    np.testing.assert_array_equal(timeout_success, np.array([True, False, False]))
    np.testing.assert_array_equal(fall, np.array([False, True, True]))


def test_sparse_pit_fall_only_applies_to_sparse_terrains():
    fallen = sig.sparse_pit_fall_mask(
        root_z=np.array([-0.6, -0.6, -0.4]),
        origin_z=np.zeros(3),
        is_sparse=np.array([True, False, True]),
        drop_threshold=0.5,
    )
    np.testing.assert_array_equal(fallen, np.array([True, False, False]))


def test_soft_terrain_disables_pit_fall():
    fallen = sig.sparse_pit_fall_mask(
        root_z=np.array([-0.6, -0.6]),
        origin_z=np.zeros(2),
        is_sparse=np.array([True, True]),
        drop_threshold=0.5,
        soft_terrain=True,
    )
    np.testing.assert_array_equal(fallen, np.array([False, False]))


def test_sparse_curriculum_requires_timeout_success_and_demotes_early_falls():
    move_up, move_down = sig.sparse_curriculum_moves(
        move_up=np.array([True, True, False, False]),
        move_down=np.array([False, False, False, True]),
        is_sparse=np.array([True, True, True, False]),
        moving=np.array([True, True, True, True]),
        timed_out=np.array([True, False, False, False]),
        pit_fall=np.array([False, True, True, False]),
    )
    np.testing.assert_array_equal(move_up, np.array([True, False, False, False]))
    np.testing.assert_array_equal(move_down, np.array([False, True, True, True]))


def test_distill_can_keep_failed_sparse_rows_in_the_distribution():
    move_down = np.array([False, True, True, True])
    is_sparse = np.array([True, True, True, False])
    kept = sig.mask_sparse_curriculum_demote(move_down, is_sparse, demote_sparse=False)
    np.testing.assert_array_equal(kept, np.array([False, False, False, True]))
    same = sig.mask_sparse_curriculum_demote(move_down, is_sparse, demote_sparse=True)
    np.testing.assert_array_equal(same, move_down)


def test_sparse_difficulty_bands_cover_all_ten_rows():
    assert [sig.terrain_difficulty_band(level, 9) for level in range(10)] == [
        "easy",
        "easy",
        "easy",
        "mid",
        "mid",
        "mid",
        "hard",
        "hard",
        "hard",
        "hard",
    ]


def test_lightlp_termination_helpers_match_section_iv_c2():
    import math

    assert abs(sig.tilt_from_upright_rad(0.0, 0.0, -1.0)) < 1.0e-6
    assert sig.tilt_from_upright_rad(1.0, 0.0, 0.0) > 1.5
    assert sig.tilt_from_upright_rad(0.0, 0.0, 1.0) == pytest.approx(math.pi, abs=1.0e-5)
    assert sig.stochastic_fall_over(1.2, 0.0) is True
    assert sig.stochastic_fall_over(1.2, 0.5) is False
    assert sig.stochastic_fall_over(0.2, 0.0) is False
    flags = sig.resample_impact_immunity(10, fraction=0.1, draws=[0.05] + [0.9] * 9)
    assert flags[0] is True
    assert flags.count(True) == 1
    assert sig.joint_velocity_timeout(50.1) is True
    assert sig.joint_velocity_timeout(49.0) is False
    assert sig.excessive_base_accel(50.0, 0.5) is False
    assert sig.excessive_base_accel(50.0, 1.1) is True
    assert sig.collapsed_pelvis_above_feet_mask(0.76, 0.05, 0.40) is False
    assert sig.collapsed_pelvis_above_feet_mask(0.16, 0.05, 0.40) is True
    assert sig.collapsed_pelvis_above_feet_mask(0.20, -0.50, 0.40) is False
    assert sig.wrap_heading_error(0.0, 0.2) == pytest.approx(0.2)
    upright = sig.upright_orientation_reward(0.0, 0.0)
    tilted = sig.upright_orientation_reward(0.5, 0.0)
    assert upright > tilted
    assert upright == pytest.approx(1.1)


def test_oob_linf_sits_past_l2_promote_bar():
    """Axis-aligned walk must be able to promote (L2>half) before L-inf OOB."""
    import importlib.util

    cur_path = Path(__file__).resolve().parents[1] / "legged_lab" / "locomotion" / "curriculum.py"
    spec = importlib.util.spec_from_file_location("t4_curriculum_oob", cur_path)
    cur = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cur)

    tile = 8.0
    promote = sig.promote_radius_m(tile)
    oob_lim = sig.oob_linf_limit_m(tile)
    assert oob_lim > promote
    just_promoted = promote + 0.01
    xy = np.array([[just_promoted, 0.0], [promote - 0.01, 0.0], [oob_lim + 0.01, 0.0]])
    oob = sig.out_of_bounds_linf(xy, tile)
    np.testing.assert_array_equal(oob, np.array([False, False, True]))
    radial = np.linalg.norm(xy, axis=1)
    up, _down = cur.terrain_level_moves(radial, np.ones(3), episode_length_s=20.0, tile_size=tile)
    assert bool(up[0]) and not bool(oob[0])
    assert not bool(up[1])
    assert bool(oob[2])


def test_lightlp_timeout_and_reset_is_what_the_env_calls():
    env_src = (Path(__file__).resolve().parents[1] / "legged_lab" / "locomotion" / "env.py").read_text()
    assert "lightlp_timeout_and_reset(" in env_src
    assert "impact_immunity_from_draws(" in env_src
    n = 4
    reset, timeout, reasons = sig.lightlp_timeout_and_reset(
        episode_timeout=np.zeros(n, dtype=bool),
        offset_xy=np.array([[4.01, 0.0], [4.30, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        tile_size=8.0,
        max_abs_joint_vel=np.zeros(n),
        torso_hit=np.zeros(n, dtype=bool),
        accel_mps2=np.zeros(n),
        elapsed_s=np.full(n, 2.0),
        gravity_gx=np.zeros(n),
        gravity_gy=np.zeros(n),
        gravity_gz=np.array([-1.0, -1.0, 1.0, -1.0]),
        fall_draws=np.array([0.5, 0.5, 0.0, 0.5]),
        immunity=np.zeros(n, dtype=bool),
    )
    # 4.01 m axis-aligned: past promote, before OOB 4.25 — not a timeout.
    assert not bool(timeout[0]) and not bool(reset[0])
    # 4.30 m: past OOB — timeout.
    assert bool(timeout[1]) and bool(reset[1])
    assert bool(reasons["oob"][1]) and not bool(reasons["horizon"][1])
    # inverted gravity + draw 0: fall-over reset, not timeout.
    assert bool(reset[2]) and not bool(timeout[2])
    assert bool(reasons["fall_over"][2])
    assert not bool(reset[3])


def test_stage_e_task_has_no_soft_hard_switch():
    from pathlib import Path

    train = Path(__file__).resolve().parents[1] / "legged_lab" / "scripts" / "train.py"
    cfg = Path(__file__).resolve().parents[1] / "legged_lab" / "locomotion" / "teacher_cfg.py"
    train_src = train.read_text()
    cfg_src = cfg.read_text()
    assert "--hard_sparse_pits" not in train_src
    assert "class AmpLocomotionEnvCfg" in cfg_src
    # Stage E class body must not set a two-stage pit switch.
    stage_e = cfg_src.split("class LightLPLocomotionEnvCfg")[0]
    assert "soft_sparse_terrain" not in stage_e
    assert "apply_soft_sparse_stage" not in stage_e
    assert schemas.TEACHER_ACTOR_OBS_DIM == 1155


def test_sparse_teacher_keeps_lightlp_terminal_cost_disabled():
    root = Path(__file__).resolve().parents[1]
    cfg_src = (root / "legged_lab" / "locomotion" / "teacher_cfg.py").read_text()
    sparse_rewards = cfg_src.split("class LightLPRewardCfg", 1)[1].split("@configclass", 1)[0]

    assert "termination_penalty = RewTerm(func=mdp.is_terminated, weight=0.0)" in sparse_rewards


def test_sparse_teacher_keeps_s10_upright_bonus_without_ori_penalty():
    root = Path(__file__).resolve().parents[1]
    cfg_src = (root / "legged_lab" / "locomotion" / "teacher_cfg.py").read_text()
    sparse_rewards = cfg_src.split("class LightLPRewardCfg", 1)[1].split("@configclass", 1)[0]
    teacher_rewards = cfg_src.split("class AmpLocomotionRewardCfg", 1)[1].split("class AmpLocomotionEnvCfg", 1)[0]

    assert "body_orientation_l2" in teacher_rewards
    assert "weight=-2.0" in teacher_rewards
    ori_block = sparse_rewards.split("body_orientation_l2", 1)[1].split("upright_orientation", 1)[0]
    upright_block = sparse_rewards.split("upright_orientation", 1)[1].split("undesired_contacts", 1)[0]
    assert "weight=0.0" in ori_block
    assert "weight=-2.0" not in ori_block
    assert "weight=1.0" in upright_block
    assert "weight=-5.0" not in sparse_rewards


def test_sparse_teacher_hard_contact_termination_is_trunk_only():
    root = Path(__file__).resolve().parents[1]
    cfg_src = (root / "legged_lab" / "locomotion" / "teacher_cfg.py").read_text()
    teacher_env = cfg_src.split("class AmpLocomotionEnvCfg", 1)[1].split("class T4LocoTeacherAgentCfg", 1)[0]
    sparse_rewards = cfg_src.split("class LightLPRewardCfg", 1)[1].split("@configclass", 1)[0]

    from legged_lab.assets.t4.locomotion import T4_LOCOMOTION
    binding = (root / "legged_lab/locomotion/config_binding.py").read_text()
    assert "cfg.robot.terminate_contacts_body_names = [spec.torso]" in binding
    assert T4_LOCOMOTION.torso == "Trunk"
    assert T4_LOCOMOTION.reward_bodies["sparse_undesired"] == ("A[LR]2", "A[LR]4", "Trunk", "Shank_.*")
    assert 'body_names="$sparse_undesired"' in sparse_rewards
    assert "weight=-2.0" in sparse_rewards


def test_train_can_reset_optimizer_for_a_changed_mdp_warm_start():
    root = Path(__file__).resolve().parents[1]
    cli_src = (root / "legged_lab" / "utils" / "cli_args.py").read_text()
    train_src = (root / "legged_lab" / "scripts" / "train.py").read_text()

    assert '"--reset_optimizer"' in cli_src
    assert "Loading model state with a fresh optimizer." in train_src
    assert "runner.load(resume_path, load_optimizer=not args_cli.reset_optimizer)" in train_src


def test_default_teacher_stays_1155_sparse_is_new_dim():
    assert schemas.TEACHER_ACTOR_OBS_DIM == 1155
    assert schemas.TEACHER_SPARSE_SCAN_HISTORY_LENGTH == 5
    assert schemas.TEACHER_SPARSE_ACTOR_OBS_DIM == 96 * 10 + 195 * 5 + 2
    assert schemas.TEACHER_SPARSE_ACTOR_OBS_DIM == 1937
    assert schemas.TEACHER_SPARSE_CRITIC_OBS_DIM == 101 * 10 + 195 * 5 + 30 + 1
    assert schemas.FOOT_SCAN_BOTH_DIM == 30
    assert schemas.TEACHER_SPARSE_IMMUNITY_DIM == 1
    assert "contact_truth" in schemas.TEACHER_FORBIDDEN_PRIVILEGE_FIELDS


def test_recent_push_mask_window_boundaries():
    dec = 4
    assert sig.recent_push_mask(4, 4, dec) is True
    assert sig.recent_push_mask(8, 4, dec) is True
    assert sig.recent_push_mask(9, 4, dec) is False
    assert sig.recent_push_mask(4, 5, dec) is False
    assert sig.recent_push_mask(4, -(10**9), dec) is False
    last = np.array([4, 4, 4, 4, -(10**9)])
    sim = np.array([4, 8, 9, 3, 4])
    np.testing.assert_array_equal(
        sig.recent_push_mask(sim, last, dec),
        np.array([True, True, False, False, False]),
    )


def test_push_velocity_jump_trips_accel_gate_until_masked():
    rng = np.random.default_rng(0)
    n = 20_000
    dt = 0.02
    v0 = np.broadcast_to(np.array([0.7, 0.0, 0.0]), (n, 3)).copy()
    add_xy = rng.uniform(-1.0, 1.0, size=(n, 2))
    v_add = v0.copy()
    v_add[:, :2] += add_xy
    accel_add = np.linalg.norm(v_add - v0, axis=1) / dt
    set_xy = rng.uniform(-1.0, 1.0, size=(n, 2))
    v_set = v0.copy()
    v_set[:, :2] = set_xy
    accel_set = np.linalg.norm(v_set - v0, axis=1) / dt
    assert (accel_add > 40.0).mean() > 0.3
    assert (accel_set > 40.0).mean() > 0.3
    tagged = np.full(n, 4)
    masked = sig.mask_recent_push_accel(accel_add, 4, tagged, decimation=4)
    assert not (masked > 40.0).any()
    later = sig.mask_recent_push_accel(accel_add, 12, tagged, decimation=4)
    np.testing.assert_array_equal(later, accel_add)


def test_lightlp_sparse_promotion_guard_blocks_pit_fall_only_on_sparse():
    move_up = np.array([True, True, True])
    is_sparse = np.array([True, True, False])
    pit_fall = np.array([True, False, True])
    guarded = sig.lightlp_sparse_promotion_guard(move_up, is_sparse, pit_fall)
    np.testing.assert_array_equal(guarded, np.array([False, True, True]))
    assert sig.lightlp_sparse_promotion_guard(True, True, True) is False
    assert sig.lightlp_sparse_promotion_guard(True, False, True) is True

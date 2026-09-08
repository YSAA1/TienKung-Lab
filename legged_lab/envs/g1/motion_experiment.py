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

"""Opt-in G1 training experiments inspired by the public VITAL G1 reward recipe.

This tests a recipe package, not attribution to one coefficient. It preserves
the robot, observations, PPO, AMP data and sparse-terrain task contracts.
"""

REFERENCE_COMMIT = "8b1a371b523a7d5024f7743c8581a7c4e8ae58a3"

G1_MOTION_EXPERIMENTS = (
    "vital_v1",
    "vital_termination_only",
    "vital_gait_gate_off_only",
    "vital_action_rate_only",
    "vital_no_action_rate",
    "vital_no_gait_gate_off",
)


def _require_g1(env_cfg, profile: str):
    if env_cfg.robot_spec.name != "g1":
        raise ValueError(f"{profile} is a G1-specific experimental recipe")


def _apply_vital_termination(env_cfg):
    env_cfg.acceleration_termination_enabled = False
    # VITAL uses |roll|>.8 or |pitch|>1.0. Keep geometric collapse as well;
    # do not restore torso net-force termination with G1 self-collision.
    env_cfg.deterministic_fall_limits = (0.8, 1.0)


def _apply_gait_gate_off(env_cfg):
    env_cfg.gait.tracking_gate_enabled = False


def _apply_action_rate(env_cfg):
    env_cfg.reward.action_rate_l2.weight = -0.01


def _apply_shared_g1_training_contract(env_cfg):
    # Keep the paper's upright/slack/yaw terms and all other reward weights.
    # The user explicitly retained the successful T4/LightLP reward design.
    # Full17 AMP and sparse terrain masks are also unchanged.
    env_cfg.lightlp_promotion_distance = "max_radial"
    env_cfg.progress_monitor_enabled = True
    env_cfg.sparse_command_min_speed_scale = 1.0


def apply_vital_motion_experiment(env_cfg, profile: str):
    """Apply an opt-in fresh-start G1 profile before environment construction."""
    _require_g1(env_cfg, profile)
    if profile == "vital_v1":
        _apply_vital_termination(env_cfg)
        _apply_gait_gate_off(env_cfg)
        _apply_action_rate(env_cfg)
    elif profile == "vital_no_action_rate":
        _apply_vital_termination(env_cfg)
        _apply_gait_gate_off(env_cfg)
    elif profile == "vital_no_gait_gate_off":
        _apply_vital_termination(env_cfg)
        _apply_action_rate(env_cfg)
    elif profile == "vital_termination_only":
        _apply_vital_termination(env_cfg)
    elif profile == "vital_gait_gate_off_only":
        _apply_gait_gate_off(env_cfg)
    elif profile == "vital_action_rate_only":
        _apply_action_rate(env_cfg)
    else:
        raise ValueError(f"unknown G1 motion experiment: {profile}")
    _apply_shared_g1_training_contract(env_cfg)


def apply_vital_motion_v1(env_cfg):
    """Compatibility wrapper for the original full VITAL-inspired package."""
    apply_vital_motion_experiment(env_cfg, "vital_v1")

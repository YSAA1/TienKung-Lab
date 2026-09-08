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

"""Opt-in G1 training experiment, inspired by the public VITAL G1 reward recipe.

This tests a recipe package, not attribution to one coefficient. It preserves
the robot, observations, PPO, AMP data and sparse-terrain task contracts.
"""

REFERENCE_COMMIT = "8b1a371b523a7d5024f7743c8581a7c4e8ae58a3"


def apply_vital_motion_v1(env_cfg):
    """Apply only to a fresh G1 config, before environment construction."""
    if env_cfg.robot_spec.name != "g1":
        raise ValueError("vital_motion_v1 is a G1-specific experimental recipe")
    env_cfg.acceleration_termination_enabled = False
    # VITAL uses |roll|>.8 or |pitch|>1.0. Keep geometric collapse as well;
    # do not restore torso net-force termination with G1 self-collision.
    env_cfg.deterministic_fall_limits = (0.8, 1.0)
    env_cfg.gait.tracking_gate_enabled = False
    env_cfg.reward.action_rate_l2.weight = -0.01
    # Keep the paper's upright/slack/yaw terms and all other reward weights.
    # The user explicitly retained the successful T4/LightLP reward design.
    # Full17 AMP and sparse terrain masks are also unchanged.
    env_cfg.lightlp_promotion_distance = "max_radial"
    env_cfg.progress_monitor_enabled = True
    env_cfg.sparse_command_min_speed_scale = 1.0

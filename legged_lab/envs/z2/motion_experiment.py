# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""Opt-in Z2 training experiments.

The Z2 teacher task (``z2_loco_teacher``) already carries the reset-aligned
G1-parity termination/contact recipe in code, so unlike the G1 vital profiles
these experiments only layer the shared ramped plant DR on top of the current
configuration. Rewards and terminations are untouched.
"""

from __future__ import annotations

from legged_lab.locomotion.plant_dr import apply_v31_plant_dr

Z2_MOTION_EXPERIMENTS = ("z2_vital_v31",)


def _require_z2(env_cfg, profile: str):
    if env_cfg.robot_spec.name != "z2":
        raise ValueError(f"{profile} is a Z2-specific experimental recipe")


def apply_z2_motion_experiment(env_cfg, profile: str):
    """Apply an opt-in fresh-start Z2 profile before environment construction."""
    _require_z2(env_cfg, profile)
    if profile == "z2_vital_v31":
        apply_v31_plant_dr(env_cfg)
    else:
        raise ValueError(f"unknown Z2 motion experiment: {profile}")

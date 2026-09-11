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
the robot, observations, PPO and sparse-terrain task contracts. ``vital_v2``
also changes the training plant DR and is meant to be paired with LAFAN AMP.
"""

from types import SimpleNamespace as NS

REFERENCE_COMMIT = "8b1a371b523a7d5024f7743c8581a7c4e8ae58a3"

G1_MOTION_EXPERIMENTS = (
    "vital_v1",
    "vital_v2",
    "vital_v3",
    "vital_v31",
    "vital_termination_only",
    "vital_gait_gate_off_only",
    "vital_action_rate_only",
    "vital_no_action_rate",
    "vital_no_gait_gate_off",
)

VITAL_V2_ACTION_DELAY = {"min_delay": 0, "max_delay": 2}
VITAL_V2_STATIC_FRICTION = (0.4, 1.2)
VITAL_V2_DYNAMIC_FRICTION = (0.3, 1.0)
VITAL_V2_RESET_VELOCITY = {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-0.3, 0.3)}
VITAL_V2_ACTUATOR_GAIN_SCALE = (0.9, 1.1)

# vital_v3 keeps the plant DR active from step 0 but ramps every range from the
# proven vital_v1 condition to its full strength over VITAL_V3_RAMP_STEPS policy
# steps (~1500 PPO iterations at 24 steps/env), so the terrain promotion gate
# faces the same plant it did in vital_v1 while the policy is still locking in
# its gait. Ranges follow the Isaac<->MuJoCo paired-rollout findings and mature
# open recipes (BeamDojo-style gain factors; no reset-velocity DR anywhere).
# The numbers and event wiring live in the shared robot-neutral
# ``legged_lab.locomotion.plant_dr``; the aliases below keep this module's
# historical names importable.
from legged_lab.locomotion.plant_dr import (  # noqa: F401
    VITAL_V31_COM_OFFSET_RANGE,
    VITAL_V31_ENCODER_BIAS,
    VITAL_V31_SCAN_OCCLUSION,
    VITAL_V3_ACTION_DELAY,
    VITAL_V3_ACTUATOR_GAIN_SCALE,
    VITAL_V3_DYNAMIC_FRICTION_SOURCE,
    VITAL_V3_DYNAMIC_FRICTION_TARGET,
    VITAL_V3_MATERIAL_REFRESH_S,
    VITAL_V3_RAMP_STEPS,
    VITAL_V3_STATIC_FRICTION_SOURCE,
    VITAL_V3_STATIC_FRICTION_TARGET,
    apply_v31_plant_dr as _apply_v31_plant_dr_impl,
    apply_v3_plant_dr as _apply_plant_dr_v3_impl,
    attach_ramped_dr_events as _attach_ramped_dr_events_impl,
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
    # Sparse terrain AMP masking is unchanged. AMP files are chosen by train.py.
    env_cfg.lightlp_promotion_distance = "max_radial"
    env_cfg.progress_monitor_enabled = True
    env_cfg.sparse_command_min_speed_scale = 1.0


def _attach_actuator_gain_event(events):
    params = {
        "stiffness_distribution_params": VITAL_V2_ACTUATOR_GAIN_SCALE,
        "damping_distribution_params": VITAL_V2_ACTUATOR_GAIN_SCALE,
        "operation": "scale",
        "distribution": "uniform",
    }
    try:
        from isaaclab.managers import EventTermCfg as EventTerm
        from isaaclab.managers import SceneEntityCfg

        import legged_lab.mdp as mdp
    except ImportError:
        events.actuator_gains = NS(params=params)
        return
    params = {
        **params,
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    }
    events.actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params=params,
    )


def _apply_plant_dr(env_cfg):
    """Training-only contact/latency/PD jitter. Do not use this helper at eval."""
    delay = env_cfg.domain_rand.action_delay
    delay.enable = True
    delay.params = dict(VITAL_V2_ACTION_DELAY)
    material = env_cfg.domain_rand.events.physics_material.params
    material["static_friction_range"] = VITAL_V2_STATIC_FRICTION
    material["dynamic_friction_range"] = VITAL_V2_DYNAMIC_FRICTION
    env_cfg.domain_rand.events.reset_base.params["velocity_range"] = dict(VITAL_V2_RESET_VELOCITY)
    _attach_actuator_gain_event(env_cfg.domain_rand.events)


def _attach_ramped_dr_events(events):
    """Delegate to the shared robot-neutral ramped plant DR wiring."""
    _attach_ramped_dr_events_impl(events)


def _apply_plant_dr_v3(env_cfg):
    """End-to-end plant DR: active from step 0, ramped to full strength."""
    _apply_plant_dr_v3_impl(env_cfg)


def _apply_plant_dr_v31(env_cfg):
    """v3 ramped plant DR plus encoder-bias / torso-COM / scan-occlusion DR."""
    _apply_v31_plant_dr_impl(env_cfg)


def apply_vital_motion_experiment(env_cfg, profile: str):
    """Apply an opt-in fresh-start G1 profile before environment construction."""
    _require_g1(env_cfg, profile)
    if profile == "vital_v1":
        _apply_vital_termination(env_cfg)
        _apply_gait_gate_off(env_cfg)
        _apply_action_rate(env_cfg)
    elif profile == "vital_v2":
        _apply_vital_termination(env_cfg)
        _apply_gait_gate_off(env_cfg)
        _apply_action_rate(env_cfg)
        _apply_plant_dr(env_cfg)
    elif profile == "vital_v3":
        _apply_vital_termination(env_cfg)
        _apply_gait_gate_off(env_cfg)
        _apply_action_rate(env_cfg)
        _apply_plant_dr_v3(env_cfg)
    elif profile == "vital_v31":
        _apply_vital_termination(env_cfg)
        _apply_gait_gate_off(env_cfg)
        _apply_action_rate(env_cfg)
        _apply_plant_dr_v31(env_cfg)
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

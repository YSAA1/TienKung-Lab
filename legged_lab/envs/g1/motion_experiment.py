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
VITAL_V3_RAMP_STEPS = 36000
VITAL_V3_STATIC_FRICTION_SOURCE = (0.6, 1.0)
VITAL_V3_STATIC_FRICTION_TARGET = (0.6, 1.2)
VITAL_V3_DYNAMIC_FRICTION_SOURCE = (0.4, 0.8)
VITAL_V3_DYNAMIC_FRICTION_TARGET = (0.5, 1.0)
VITAL_V3_ACTUATOR_GAIN_SCALE = (0.9, 1.1)
VITAL_V3_ACTION_DELAY = {"min_delay": 0, "max_delay": 1}
VITAL_V3_MATERIAL_REFRESH_S = (45.0, 60.0)
# vital_v31 additions: constant encoder bias on joint-position observations,
# torso center-of-mass offsets, and lateral band occlusion of the privileged
# scan, all ramped in over the same window as the v3 plant DR.
VITAL_V31_ENCODER_BIAS = {"bias_range": (-0.015, 0.015), "ramp_steps": VITAL_V3_RAMP_STEPS}
VITAL_V31_COM_OFFSET_RANGE = (-0.05, 0.05)
VITAL_V31_SCAN_OCCLUSION = {
    "occlusion_probability": 0.10,
    "occlusion_band_fraction": (0.1, 0.3),
    "occlusion_ramp_steps": VITAL_V3_RAMP_STEPS,
}


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
    """Training-only ramped plant DR for vital_v3. Do not use this helper at eval.

    The startup friction term keeps its vital_v1 ranges (interval terms do not fire
    at startup); the interval term re-randomizes friction with ramped ranges, the
    reset terms resample gains and action latency per episode.
    """
    material_params = {
        "static_friction_source": VITAL_V3_STATIC_FRICTION_SOURCE,
        "static_friction_target": VITAL_V3_STATIC_FRICTION_TARGET,
        "dynamic_friction_source": VITAL_V3_DYNAMIC_FRICTION_SOURCE,
        "dynamic_friction_target": VITAL_V3_DYNAMIC_FRICTION_TARGET,
        "restitution_range": (0.0, 0.005),
        "ramp_steps": VITAL_V3_RAMP_STEPS,
    }
    gains_params = {
        "stiffness_distribution_params": VITAL_V3_ACTUATOR_GAIN_SCALE,
        "damping_distribution_params": VITAL_V3_ACTUATOR_GAIN_SCALE,
        "ramp_steps": VITAL_V3_RAMP_STEPS,
        "operation": "scale",
        "distribution": "uniform",
    }
    delay_params = {**VITAL_V3_ACTION_DELAY, "ramp_steps": VITAL_V3_RAMP_STEPS}
    try:
        from isaaclab.managers import EventTermCfg as EventTerm
        from isaaclab.managers import SceneEntityCfg

        import legged_lab.mdp as mdp
    except ImportError:
        events.physics_material_ramp = NS(params=material_params, mode="interval")
        events.actuator_gains = NS(params=gains_params, mode="reset")
        events.action_delay_reset = NS(params=delay_params, mode="reset")
        return
    events.physics_material_ramp = EventTerm(
        func=mdp.randomize_rigid_body_material_ramped,
        mode="interval",
        interval_range_s=VITAL_V3_MATERIAL_REFRESH_S,
        params={**material_params, "asset_cfg": SceneEntityCfg("robot", body_names=".*")},
    )
    events.actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains_ramped,
        mode="reset",
        params={**gains_params, "asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    events.action_delay_reset = EventTerm(
        func=mdp.randomize_action_delay_ramped,
        mode="reset",
        params=delay_params,
    )


def _apply_plant_dr_v3(env_cfg):
    """End-to-end plant DR: active from step 0, ramped to full strength."""
    delay = env_cfg.domain_rand.action_delay
    delay.enable = True
    delay.params = dict(VITAL_V3_ACTION_DELAY)
    _attach_ramped_dr_events(env_cfg.domain_rand.events)


def _apply_plant_dr_v31(env_cfg):
    """v3 ramped plant DR plus encoder-bias / torso-COM / scan-occlusion DR."""
    _apply_plant_dr_v3(env_cfg)
    bias = env_cfg.domain_rand.encoder_bias
    bias.enable = True
    bias.params = dict(VITAL_V31_ENCODER_BIAS)
    scanner = env_cfg.scene.height_scanner
    scanner.occlusion_probability = VITAL_V31_SCAN_OCCLUSION["occlusion_probability"]
    scanner.occlusion_band_fraction = VITAL_V31_SCAN_OCCLUSION["occlusion_band_fraction"]
    scanner.occlusion_ramp_steps = VITAL_V31_SCAN_OCCLUSION["occlusion_ramp_steps"]
    com_params = {
        "asset_cfg": env_cfg.robot_spec.torso,
        "com_range": {
            "x": VITAL_V31_COM_OFFSET_RANGE,
            "y": VITAL_V31_COM_OFFSET_RANGE,
            "z": VITAL_V31_COM_OFFSET_RANGE,
        },
    }
    try:
        from isaaclab.managers import EventTermCfg as EventTerm
        from isaaclab.managers import SceneEntityCfg

        import legged_lab.mdp as mdp
    except ImportError:
        env_cfg.domain_rand.events.randomize_com = NS(params=com_params, mode="startup")
        return
    com_params["asset_cfg"] = SceneEntityCfg("robot", body_names=env_cfg.robot_spec.torso)
    env_cfg.domain_rand.events.randomize_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params=com_params,
    )


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

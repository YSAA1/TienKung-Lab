# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""Shared, robot-neutral ramped plant DR for locomotion teachers (vital v3/v31).

The vital recipe ramps every plant randomization from the proven v1 condition
to full strength over ``VITAL_V3_RAMP_STEPS`` policy steps so the terrain
curriculum gates face a v1-equivalent plant while the gait locks in. This
module holds the numbers and the event wiring only; reward and termination
packaging stays per-robot (``g1/motion_experiment.py``, ``z2/motion_experiment.py``).
"""

from __future__ import annotations

from types import SimpleNamespace as NS

VITAL_V3_RAMP_STEPS = 36000
VITAL_V3_STATIC_FRICTION_SOURCE = (0.6, 1.0)
VITAL_V3_STATIC_FRICTION_TARGET = (0.6, 1.2)
VITAL_V3_DYNAMIC_FRICTION_SOURCE = (0.4, 0.8)
VITAL_V3_DYNAMIC_FRICTION_TARGET = (0.5, 1.0)
VITAL_V3_ACTUATOR_GAIN_SCALE = (0.9, 1.1)
VITAL_V3_ACTION_DELAY = {"min_delay": 0, "max_delay": 1}
VITAL_V3_MATERIAL_REFRESH_S = (45.0, 60.0)
# v31 additions: constant encoder bias on joint-position observations, torso
# center-of-mass offsets, and lateral band occlusion of the privileged scan,
# all ramped in over the same window as the v3 plant DR.
VITAL_V31_ENCODER_BIAS = {"bias_range": (-0.015, 0.015), "ramp_steps": VITAL_V3_RAMP_STEPS}
VITAL_V31_COM_OFFSET_RANGE = (-0.05, 0.05)
VITAL_V31_SCAN_OCCLUSION = {
    "occlusion_probability": 0.10,
    "occlusion_band_fraction": (0.1, 0.3),
    "occlusion_ramp_steps": VITAL_V3_RAMP_STEPS,
}


def attach_ramped_dr_events(events):
    """Training-only ramped plant DR. Do not use this helper at eval.

    The startup friction term keeps its vital_v1 ranges (interval terms do not
    fire at startup); the interval term re-randomizes friction with ramped
    ranges, the reset terms resample gains and action latency per episode.
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


def apply_v3_plant_dr(env_cfg):
    """End-to-end plant DR: active from step 0, ramped to full strength."""
    delay = env_cfg.domain_rand.action_delay
    delay.enable = True
    delay.params = dict(VITAL_V3_ACTION_DELAY)
    attach_ramped_dr_events(env_cfg.domain_rand.events)


def apply_v31_dr_extras(env_cfg):
    """v31 additions on top of v3: encoder bias, scan occlusion, torso COM."""
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


def apply_v31_plant_dr(env_cfg):
    """v3 ramped plant DR plus encoder-bias / torso-COM / scan-occlusion DR."""
    apply_v3_plant_dr(env_cfg)
    apply_v31_dr_extras(env_cfg)

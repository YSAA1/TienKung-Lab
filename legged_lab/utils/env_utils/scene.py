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

from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, patterns
from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from legged_lab.terrains.ray_caster_cfg import RayCasterCfg

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env_config import BaseSceneCfg


def resolve_depth_camera_update_period(configured_period, step_dt):
    """Keep a task-set camera period.

    Isaac uses ``update_period=0`` to mean every physics substep. Scene construction
    historically mapped that to control-rate ``step_dt`` so tiled depth does not
    render at 200 Hz. A positive configured period (for example 0.06 s) must be
    preserved; unconditionally writing ``step_dt`` made 16.7 Hz student configs a
    no-op.
    """
    if configured_period is None or configured_period <= 0:
        return step_dt
    return configured_period


def rtx_render_interval_physics(update_period, physics_dt) -> int:
    """Physics substeps between RTX ticks. 0.06 s / 0.005 s = 12."""
    if update_period is None or physics_dt is None:
        return 1
    period = float(update_period)
    dt = float(physics_dt)
    if period <= 0.0 or dt <= 0.0:
        return 1
    return max(1, int(round(period / dt)))


def rtx_render_due(sim_step_counter, physics_dt, update_period) -> bool:
    """True on physics steps that should call ``sim.render()`` for RTX sensors."""
    interval = rtx_render_interval_physics(update_period, physics_dt)
    return int(sim_step_counter) % interval == 0


def pending_post_reset_rtx(reset_sim_step, last_rtx_sim_step) -> bool:
    """True when this env reset at or after the last RTX tick (image is pre-teleport)."""
    return int(reset_sim_step) >= int(last_rtx_sim_step)


def sensor_is_warp_raycast(sensor) -> bool:
    """True for Isaac Lab ``RayCasterCamera`` / ``RayCaster`` (warp, not RTX)."""
    if sensor is None:
        return False
    names = [type(sensor).__name__]
    cfg = getattr(sensor, "cfg", None)
    if cfg is not None:
        names.append(type(cfg).__name__)
        class_type = getattr(cfg, "class_type", None)
        if class_type is not None:
            names.append(getattr(class_type, "__name__", ""))
    return any("RayCaster" in name for name in names)


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    def __init__(self, config: "BaseSceneCfg", physics_dt, step_dt):
        super().__init__(num_envs=config.num_envs, env_spacing=config.env_spacing)

        self.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type=config.terrain_type,
            terrain_generator=config.terrain_generator,
            max_init_terrain_level=config.max_init_terrain_level,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=None
            if config.disable_visual_assets
            else sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=False,
        )

        self.robot: ArticulationCfg = config.robot.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, update_period=physics_dt
        )

        self.light = None
        self.sky_light = None
        if not config.disable_visual_assets:
            self.light = AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
            )
            self.sky_light = AssetBaseCfg(
                prim_path="/World/skyLight",
                spawn=sim_utils.DomeLightCfg(
                    intensity=750.0,
                    texture_file=(
                        f"{ISAACLAB_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"
                    ),
                ),
            )

        if config.height_scanner.enable_height_scan:
            self.height_scanner = RayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + config.height_scanner.prim_body_name,
                offset=RayCasterCfg.OffsetCfg(
                    pos=(config.height_scanner.offset[0], config.height_scanner.offset[1], 20.0)
                ),
                attach_yaw_only=True,
                pattern_cfg=patterns.GridPatternCfg(
                    resolution=config.height_scanner.resolution, size=config.height_scanner.size
                ),
                debug_vis=config.height_scanner.debug_vis,
                mesh_prim_paths=["/World/ground"],
                update_period=step_dt,
                drift_range=config.height_scanner.drift_range,
            )

        if getattr(config, "foot_scanner", None) is not None and config.foot_scanner.enable:
            for body_name, attr in zip(
                config.foot_scanner.body_names, ("left_foot_scanner", "right_foot_scanner"), strict=False
            ):
                setattr(
                    self,
                    attr,
                    RayCasterCfg(
                        prim_path="{ENV_REGEX_NS}/Robot/" + body_name,
                        offset=RayCasterCfg.OffsetCfg(pos=(0.04, 0.0, 0.02)),
                        attach_yaw_only=True,
                        pattern_cfg=patterns.GridPatternCfg(
                            resolution=config.foot_scanner.resolution, size=config.foot_scanner.size
                        ),
                        debug_vis=config.foot_scanner.debug_vis,
                        mesh_prim_paths=["/World/ground"],
                        update_period=step_dt,
                    ),
                )

        if config.lidar.enable_lidar:
            self.lidar = RayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + config.lidar.prim_body_name,
                offset=RayCasterCfg.OffsetCfg(pos=config.lidar.offset, rot=config.lidar.rotation),
                attach_yaw_only=True,
                pattern_cfg=config.lidar.pattern_cfg,
                debug_vis=config.lidar.debug_vis,
                mesh_prim_paths=config.lidar.mesh_prim_paths,
                max_distance=config.lidar.max_distance,
            )

        if config.depth_camera.enable_depth_camera:
            # Preserve the configured sensor class (regular Camera vs tiled
            # Camera). Rebuilding every sensor as TiledCamera silently breaks
            # depth-only tasks that intentionally use one render product per env.
            camera_update_period = resolve_depth_camera_update_period(
                getattr(config.depth_camera, "update_period", 0.0),
                step_dt,
            )
            self.depth_camera = config.depth_camera.replace(
                prim_path="{ENV_REGEX_NS}/Robot/" + config.depth_camera.prim_body_name,
                update_period=camera_update_period,
            )

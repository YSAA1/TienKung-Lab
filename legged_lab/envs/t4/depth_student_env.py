"""T4 online depth distillation environment.

The physics, rewards and reset semantics are inherited from the Stage E teacher.
The environment exposes two observation streams: a deployable depth/proprio
student input and the frozen teacher's proprio/HeightScan input for supervision.
"""

from __future__ import annotations

import copy

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp

from legged_lab.assets.t4.schemas import DEPTH_CAMERA_SITE_POS, DEPTH_POLICY_SIZE, depth_camera_ros_quat_wxyz
from legged_lab.config import EventCfg, FootScannerCfg
from legged_lab.locomotion.mdp.camera_extrinsic import LIGHTLP_CAMERA_ORI_JITTER_RAD, LIGHTLP_CAMERA_POS_JITTER_M
from legged_lab.locomotion.mdp.depth_noise import LIGHTLP_DEPTH_DELAY_STEPS, LIGHTLP_DEPTH_HOLD_STEPS
from legged_lab.envs.t4.teacher_cfg import T4LocoSparseTeacherEnvCfg, T4LocoTeacherEnvCfg, T4SparseTeacherRewardCfg
from legged_lab.sensors.camera.camera_cfg import CameraCfg
from legged_lab.sensors.camera.camera_cfgs import D455CameraCfg, TiledD455CameraCfg
from legged_lab.terrains import LIGHTLP_TERRAINS_CFG

# Native policy resolution. Tiled RTX renders 48x64; no 72x128 capture.
SPARSE_STUDENT_RENDER_SIZE = DEPTH_POLICY_SIZE


@configclass
class T4LocoDepthStudentEnvCfg(T4LocoTeacherEnvCfg):
    """Stage E physics with the schema head-height depth camera."""

    policy_role: str = "student"

    def __post_init__(self):
        # Keep the same local HeightScan for teacher supervision, while the
        # student receives only the camera stream through the env below.
        # Head-height Trunk site + 35 deg down. Do not inherit the stock D455
        # pelvis offset: that rolls the image 90 deg and is off-contract.
        self.scene.depth_camera = D455CameraCfg(
            prim_body_name="Trunk/depth_camera",
            width=480,
            height=270,
            debug_vis=False,
            data_types=["distance_to_image_plane"],
            offset=CameraCfg.OffsetCfg(
                pos=DEPTH_CAMERA_SITE_POS,
                rot=depth_camera_ros_quat_wxyz(),
                convention="ros",
            ),
        )
        # The nubot headless image does not ship the Isaac Nucleus visual
        # materials referenced by the generic SceneCfg. Camera sensors only
        # need the collision terrain, so keep rendering self-contained.
        self.scene.disable_visual_assets = True
        self.noise.add_noise = False


@configclass
class T4LocoDepthStudentFtEnvCfg(T4LocoDepthStudentEnvCfg):
    """Sparse-mix student FT: LightLP rewards, existing depth student obs."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain_generator = LIGHTLP_TERRAINS_CFG
        self.scene.max_init_terrain_level = 5
        self.scene.foot_scanner = FootScannerCfg(enable=True)
        self.reward = T4SparseTeacherRewardCfg()
        self.append_actor_feet_contact = False


def _t4_student_depth_camera(
    *,
    update_period: float = 0.02,
    sensor_noise_enable: bool = False,
) -> TiledD455CameraCfg:
    # Tiled RTX D455 at native policy resolution. LightLP §VI noise is applied
    # in Python; keep Isaac SensorNoiseCfg off so the two models do not stack.
    height, width = SPARSE_STUDENT_RENDER_SIZE
    cfg = TiledD455CameraCfg(
        prim_body_name="Trunk/depth_camera",
        width=width,
        height=height,
        debug_vis=False,
        data_types=["distance_to_image_plane"],
        offset=CameraCfg.OffsetCfg(
            pos=DEPTH_CAMERA_SITE_POS,
            rot=depth_camera_ros_quat_wxyz(),
            convention="ros",
        ),
    )
    cfg.sensor_noise.enable = bool(sensor_noise_enable)
    cfg.update_period = update_period
    cfg.enable_depth_camera = True
    return cfg


@configclass
class T4LocoSparseDepthStudentEnvCfg(T4LocoSparseTeacherEnvCfg):
    """S12 sparse teacher MDP with a deployable depth/proprio student stream."""

    policy_role: str = "student"
    student_depth_noise: bool = True
    student_depth_hold_steps: int = LIGHTLP_DEPTH_HOLD_STEPS
    student_depth_delay_steps: tuple[int, int] = LIGHTLP_DEPTH_DELAY_STEPS
    # 50 Hz tiled RTX so the 30–60 ms delay model is not quantized to 16.7 Hz.
    student_depth_camera_update_period: float = 0.02
    student_depth_d455_sensor_noise: bool = False
    student_depth_dropout_after_resize: bool = True
    student_camera_pos_jitter_m: float = LIGHTLP_CAMERA_POS_JITTER_M
    student_camera_ori_jitter_rad: float = LIGHTLP_CAMERA_ORI_JITTER_RAD
    student_depth_boundary_corruption: bool = False
    student_depth_boundary_probability: float = 0.08
    student_depth_boundary_threshold_m: float = 0.08
    sparse_curriculum_demote: bool = False
    random_level_reset_fraction: float = 0.10
    random_level_reset_min_level: int | None = None
    random_level_reset_max_level: int | None = None

    def __post_init__(self):
        super().__post_init__()
        self.scene.depth_camera = _t4_student_depth_camera(
            update_period=self.student_depth_camera_update_period,
            sensor_noise_enable=self.student_depth_d455_sensor_noise,
        )
        # Skip Nucleus marble/sky assets; keep local robot USD + terrain mesh
        # so RTX depth can self-occlude.
        self.scene.disable_visual_assets = True
        self.noise.add_noise = True
        self.noise.noise_scales.height_scan = 0.0
        self.student_depth_noise = True
        self.sparse_curriculum_demote = False
        self.random_level_reset_fraction = 0.10
        self.random_level_reset_min_level = None
        self.random_level_reset_max_level = None


@configclass
class T4LocoSparseDepthStudentReprFirstEnvCfg(T4LocoSparseDepthStudentEnvCfg):
    """Representation stage starts on hard sparse rows; action stage snaps back to teacher 0.10/None."""

    random_level_reset_fraction: float = 0.50
    random_level_reset_min_level: int | None = 6
    random_level_reset_max_level: int | None = None

    def __post_init__(self):
        super().__post_init__()
        self.random_level_reset_fraction = 0.50
        self.random_level_reset_min_level = 6
        self.random_level_reset_max_level = None


@configclass
class T4LocoSparseDepthStudentFtEnvCfg(T4LocoSparseDepthStudentEnvCfg):
    """Same deploy-domain env for gated D4c task RL. Not launched in this slice."""

    student_depth_camera_update_period: float = 0.02

    def __post_init__(self):
        super().__post_init__()
        self.student_depth_noise = True


@configclass
class T4LocoSparseDepthStudentResidualFtEnvCfg(T4LocoSparseDepthStudentFtEnvCfg):
    """Residual FT: same depth noise, more hard sparse exposure, no plant/manufacturing DR."""

    random_level_reset_fraction: float = 0.50
    random_level_reset_min_level: int | None = 6
    random_level_reset_max_level: int | None = None

    def __post_init__(self):
        super().__post_init__()
        self.student_depth_noise = True
        self.student_depth_boundary_corruption = False
        self.random_level_reset_fraction = 0.50
        self.random_level_reset_min_level = 6
        self.random_level_reset_max_level = None
        self.domain_rand.action_delay.enable = False
        generator = copy.deepcopy(LIGHTLP_TERRAINS_CFG)
        for name, sub in generator.sub_terrains.items():
            if name in ("stepping_stones", "raised_pillars"):
                sub.proportion = 0.30
            else:
                sub.proportion = float(sub.proportion) * (2.0 / 3.0)
            if hasattr(sub, "targeted_layout_seed"):
                sub.targeted_layout_seed = False
            if hasattr(sub, "targeted_manufacturing_variation"):
                sub.targeted_manufacturing_variation = False
        self.scene.terrain_generator = generator


@configclass
class T4TargetedFtEventCfg(EventCfg):

    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    joint_effort = EventTerm(
        func=mdp.randomize_joint_effort_limits,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "effort_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class T4LocoSparseDepthStudentTargetedFtEnvCfg(T4LocoSparseDepthStudentFtEnvCfg):
    """Targeted-only sim-to-sim robustness domain; default S12/evaluator stays frozen."""

    student_depth_boundary_corruption: bool = True

    def __post_init__(self):
        super().__post_init__()
        base_events = self.domain_rand.events
        targeted_events = T4TargetedFtEventCfg()
        for name in ("physics_material", "add_base_mass", "reset_base", "reset_robot_joints", "push_robot"):
            setattr(targeted_events, name, getattr(base_events, name))
        self.domain_rand.events = targeted_events
        self.domain_rand.action_delay.enable = True
        self.domain_rand.action_delay.params = {"min_delay": 0, "max_delay": 2}
        self.student_depth_boundary_corruption = True
        generator = copy.deepcopy(LIGHTLP_TERRAINS_CFG)
        stones = generator.sub_terrains["stepping_stones"]
        stones.targeted_layout_seed = True
        pillars = generator.sub_terrains["raised_pillars"]
        pillars.targeted_manufacturing_variation = True
        pillars.xy_jitter_m = 0.015
        pillars.height_jitter_m = 0.015
        pillars.top_tilt_rad = 0.035
        pillars.diameter_scale_jitter = 0.04
        pillars.pitch_scale_jitter = 0.03
        self.scene.terrain_generator = generator


@configclass
class T4LocoSparseDepthStudentPlantFtEnvCfg(T4LocoSparseDepthStudentFtEnvCfg):
    """Plant FT: same depth noise, teacher curriculum, delay+actuator, no manufacturing."""

    def __post_init__(self):
        super().__post_init__()
        self.student_depth_noise = True
        self.student_depth_boundary_corruption = False
        self.random_level_reset_fraction = 0.10
        self.random_level_reset_min_level = None
        self.random_level_reset_max_level = None
        base_events = self.domain_rand.events
        plant_events = T4TargetedFtEventCfg()
        for name in ("physics_material", "add_base_mass", "reset_base", "reset_robot_joints", "push_robot"):
            setattr(plant_events, name, getattr(base_events, name))
        self.domain_rand.events = plant_events
        self.domain_rand.action_delay.enable = True
        self.domain_rand.action_delay.params = {"min_delay": 0, "max_delay": 2}


# Compatibility exports for existing scripts and saved configurations.
from legged_lab.locomotion.depth_env import (  # noqa: E402,F401
    DepthDistillationEnv as T4LocoDepthDistillEnv,
    LightLPDepthDistillationEnv as T4LocoSparseDepthDistillEnv,
)

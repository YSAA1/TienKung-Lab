# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""Stage E configuration for the AMP privileged locomotion teacher `pi_teacher`.

Everything that the plan freezes before formal training lives here: the
forward-asymmetric terrain privilege, the AMP style-weight schedule, the gait
representation and the adaptive terrain curriculum. Changing any of them is an
MDP change and therefore starts a new lineage.
"""

from __future__ import annotations

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

import legged_lab.mdp as mdp
from dataclasses import MISSING
from legged_lab.locomotion.robot_spec import LocomotionRobotSpec
from legged_lab.locomotion.config_binding import bind_robot_spec
from legged_lab.locomotion.schemas import (
    FOOT_SCAN_RESOLUTION, FOOT_SCAN_SIZE, PROPRIO_HISTORY_LENGTH,
    TEACHER_SCAN_HEIGHT_OFFSET, TEACHER_SCAN_OFFSET, TEACHER_SCAN_RESOLUTION,
    TEACHER_SCAN_SIZE, TEACHER_SPARSE_SCAN_HISTORY_LENGTH,
)
from legged_lab.config import (
    ActionDelayCfg,
    BaseSceneCfg,
    CommandRangesCfg,
    CommandsCfg,
    DomainRandCfg,
    EventCfg,
    FootScannerCfg,
    HeightScannerCfg,
    NoiseCfg,
    NoiseScalesCfg,
    NormalizationCfg,
    ObsScalesCfg,
    PhysxCfg,
    RobotCfg,
    SimCfg,
)
from legged_lab.terrains import LIGHTLP_TERRAINS_CFG, AMP_LOCOMOTION_TERRAINS_CFG


@configclass
class GaitCfg:
    """Frozen gait representation.

    ``mode`` selects one of the three representations the plan allows:

    - ``fixed_clock``: one global cycle, identical to the TienKung baseline.
    - ``command_conditioned``: cycle interpolated by commanded speed.
    - ``difficulty_relaxed``: fixed clock, but the periodic gait rewards fade out
      as the per-env terrain level rises, so stairs are not forced onto the
      flat-ground rhythm.

    In every mode the periodic gait terms are multiplied by planar velocity
    tracking accuracy (zero when standing) so marching in place cannot farm them.
    """

    mode: str = "fixed_clock"
    gait_air_ratio_l: float = 0.38
    gait_air_ratio_r: float = 0.38
    gait_phase_offset_l: float = 0.38
    gait_phase_offset_r: float = 0.88
    gait_cycle: float = 0.85
    # command_conditioned only: cycle at zero command and at max commanded speed.
    slow_gait_cycle: float = 0.95
    fast_gait_cycle: float = 0.70
    reference_max_speed: float = 1.0
    # Same kernel std as track_lin_vel_xy_exp; gait pays only when this tracking is good.
    tracking_std: float = 0.5
    # difficulty_relaxed only.
    min_gait_reward_scale: float = 0.3
    gait_relax_start_difficulty: float = 0.4


@configclass
class AmpTerrainScheduleCfg:
    """Terrain-difficulty schedule for the AMP style weight.

    The AMP expert set is entirely flat-ground, so on stairs and strong rough the
    discriminator would otherwise punish the correct terrain gait. ``min_scale``
    is the floor the style weight decays to at the hardest terrain level.
    """

    enable: bool = True
    mode: str = "linear_decay"
    min_scale: float = 0.3
    decay_start_difficulty: float = 0.3


@configclass
class AmpLocomotionRewardCfg:
    # Task/penalty weights follow the validated humanoid task that trained this robot on
    # rough/stair terrain: stronger velocity tracking, a mild vertical-velocity
    # penalty (stairs require root z motion), and no hip roll/yaw action penalty.
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=2.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"std": 0.5})
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.15)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    # Keep the calibrated 0.25-radian reward units when actuator scales differ.
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01, params={"reference_scale": 0.25})
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="$undesired"),
            "threshold": 1.0,
        },
    )
    # A knee brushing a stair riser is expected on the up-stairs curriculum, so
    # shank contact is penalized more mildly than arm/trunk contact.
    shank_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="$shanks"),
            "threshold": 1.0,
        },
    )
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names="$torso")}, weight=-2.0
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="$feet"),
            "asset_cfg": SceneEntityCfg("robot", body_names="$feet"),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="$feet"),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="$feet"), "threshold": 0.2},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="$feet")},
    )
    foot_touchdown_impact = RewTerm(
        func=mdp.foot_touchdown_impact_penalty,
        weight=-0.08,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="$feet"),
            "asset_cfg": SceneEntityCfg("robot", body_names="$feet"),
            "force_threshold": 20.0,
            "safe_downward_speed": 0.20,
            "contact_time_window": 0.04,
        },
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names="$hip",
            )
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="$arms")},
    )
    # Robot adapters declare which uncommanded joints need a standing anchor.
    joint_deviation_wrist_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="$wrist_waist")},
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names="$legs",
            )
        },
    )

    gait_feet_frc_perio = RewTerm(func=mdp.gait_feet_frc_perio, weight=1.0, params={"delta_t": 0.02})
    gait_feet_spd_perio = RewTerm(func=mdp.gait_feet_spd_perio, weight=1.0, params={"delta_t": 0.02})
    gait_feet_frc_support_perio = RewTerm(func=mdp.gait_feet_frc_support_perio, weight=0.6, params={"delta_t": 0.02})

    ankle_torque = RewTerm(func=mdp.ankle_torque, weight=-0.0005)
    ankle_action = RewTerm(func=mdp.ankle_action, weight=-0.001, params={"reference_scale": 0.25})
    feet_y_distance = RewTerm(func=mdp.feet_y_distance, weight=-2.0, params={"target": 0.0})


@configclass
class AmpLocomotionEnvCfg:
    robot_spec: LocomotionRobotSpec = MISSING
    enable_amp: bool = True
    policy_role: str = "teacher"
    device: str = "cuda:0"
    random_level_reset_fraction: float = 0.0
    random_level_reset_max_level: int | None = None
    random_level_reset_min_level: int | None = None
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=20.0,
        # Formal starting point for the first Stage E lineage; the capacity probe
        # decides whether to scale to 2048/4096.
        num_envs=1024,
        env_spacing=2.5,
        robot=MISSING,
        terrain_type="generator",
        terrain_generator=AMP_LOCOMOTION_TERRAINS_CFG,
        max_init_terrain_level=3,
        height_scanner=HeightScannerCfg(
            enable_height_scan=True,
            prim_body_name=MISSING,
            resolution=TEACHER_SCAN_RESOLUTION,
            size=TEACHER_SCAN_SIZE,
            offset=TEACHER_SCAN_OFFSET,
            debug_vis=False,
            drift_range=(0.0, 0.0),
        ),
    )
    robot: RobotCfg = RobotCfg(
        actor_obs_history_length=PROPRIO_HISTORY_LENGTH,
        critic_obs_history_length=PROPRIO_HISTORY_LENGTH,
        action_scale=0.25,
        # Arm and shank contacts stay penalties rather than terminations: brushing an
        # obstacle must not end the episode, while a trunk collision remains a hard fall.
        terminate_contacts_body_names=[],
        feet_body_names=[],
    )
    reward = AmpLocomotionRewardCfg()
    append_actor_feet_contact: bool = False
    gait = GaitCfg()
    amp_terrain_schedule = AmpTerrainScheduleCfg()
    normalization: NormalizationCfg = NormalizationCfg(
        obs_scales=ObsScalesCfg(
            lin_vel=1.0,
            ang_vel=1.0,
            projected_gravity=1.0,
            commands=1.0,
            joint_pos=1.0,
            joint_vel=1.0,
            actions=1.0,
            height_scan=1.0,
        ),
        clip_observations=100.0,
        clip_actions=100.0,
        height_scan_offset=TEACHER_SCAN_HEIGHT_OFFSET,
    )
    commands: CommandsCfg = CommandsCfg(
        # One command per 20s episode: the terrain curriculum judges promotion on a
        # completed tile traversal, so the episode must be a single traversal attempt
        # instead of two random-heading legs that cancel each other's displacement.
        resampling_time_range=(20.0, 20.0),
        rel_standing_envs=0.2,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=CommandRangesCfg(
            lin_vel_x=(-0.6, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.57, 1.57), heading=(-math.pi, math.pi)
        ),
    )
    noise: NoiseCfg = NoiseCfg(
        add_noise=True,
        noise_scales=NoiseScalesCfg(
            ang_vel=0.2,
            projected_gravity=0.05,
            joint_pos=0.01,
            joint_vel=1.5,
            height_scan=0.1,
        ),
    )
    domain_rand: DomainRandCfg = DomainRandCfg(
        events=EventCfg(
            physics_material=EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "static_friction_range": (0.6, 1.0),
                    "dynamic_friction_range": (0.4, 0.8),
                    "restitution_range": (0.0, 0.005),
                    "num_buckets": 64,
                },
            ),
            add_base_mass=EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names="$torso"),
                    "mass_distribution_params": (-3.0, 3.0),
                    "operation": "add",
                },
            ),
            reset_base=EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                    "velocity_range": {
                        "x": (-0.5, 0.5),
                        "y": (-0.5, 0.5),
                        "z": (-0.5, 0.5),
                        "roll": (-0.5, 0.5),
                        "pitch": (-0.5, 0.5),
                        "yaw": (-0.5, 0.5),
                    },
                },
            ),
            reset_robot_joints=EventTerm(
                func=mdp.reset_joints_by_scale,
                mode="reset",
                params={
                    "position_range": (0.5, 1.5),
                    "velocity_range": (0.0, 0.0),
                },
            ),
            push_robot=EventTerm(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(10.0, 15.0),
                params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
            ),
        ),
        action_delay=ActionDelayCfg(enable=False, params={"max_delay": 5, "min_delay": 0}),
    )
    sim: SimCfg = SimCfg(dt=0.005, decimation=4, physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15))


    def __post_init__(self):
        bind_robot_spec(self)


@configclass
class AmpLocomotionAgentCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    # Stage E resource budget for this lineage; not a pass criterion.
    max_iterations = 40000
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="AMPPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        # Sagittal mirror augmentation + mirror loss, following the validated humanoid
        # recipe (mirror_loss_coeff=5.0). The mirror plan is frozen with the
        # observation schema in legged_lab/locomotion/symmetry.py.
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            data_augmentation_func="legged_lab.locomotion.symmetry:get_symmetric_states",
            mirror_loss_coeff=5.0,
        ),
        rnd_cfg=None,
    )
    clip_actions = None
    save_interval = 500
    runner_class_name = "AmpOnPolicyRunner"
    experiment_name = "amp_locomotion"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "amp_locomotion"
    wandb_project = "amp_locomotion"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"
    amp_reward_coef = 0.3
    amp_frame_dim: int = MISSING
    amp_joint_order: list[str] = MISSING
    amp_expert_dir: str = MISSING
    amp_motion_files: list[str] = MISSING
    amp_num_preload_transitions = 200000
    amp_task_reward_lerp = 0.7
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std: list[float] = MISSING


@configclass
class LightLPRewardCfg(AmpLocomotionRewardCfg):
    """LightLP §IV Table I on the mixed sparse teacher.

    Stage E extras (periodic gait, AMP, stumble) stay in the base class so
    continuous tiles keep the successful walk recipe; the env zeros them on
    sparse tiles. Table I terms that Stage E lacked are added here.
    Instant pit-fall −200 is not a sparse failure mode.
    """

    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"std": 0.5})
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names="$torso")}, weight=0.0
    )
    upright_orientation = RewTerm(
        func=mdp.upright_orientation, params={"asset_cfg": SceneEntityCfg("robot", body_names="$torso")}, weight=1.0
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="$sparse_undesired"),
            "threshold": 1.0,
        },
    )
    shank_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="$shanks"), "threshold": 1.0},
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1, params={"reference_scale": 0.25})
    # LightLP sparse locomotion uses the paper reward table without an extra
    # terminal cost. Physical falls remain terminations and lose future return.
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=0.0)
    heading_error = RewTerm(func=mdp.heading_error, weight=-1.0)
    velocity_slack = RewTerm(func=mdp.velocity_slack, weight=1.5)
    illegal_footstep = RewTerm(func=mdp.illegal_footstep, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="$feet")})
    opposite_direction = RewTerm(func=mdp.opposite_direction, weight=-1.0)
    hurdle_bar_contact = RewTerm(func=mdp.hurdle_bar_contact, weight=-2.0)
    foot_touchdown_impact = RewTerm(
        func=mdp.foot_acceleration_penalty,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="$feet"),
            "tau_s": 0.06,
            "threshold_mps2": 30.0,
        },
    )


@configclass
class LightLPLocomotionEnvCfg(AmpLocomotionEnvCfg):
    """One-stage LightLP §IV sparse teacher: real holes, no soft/hard switch."""

    teacher_scan_history_length: int = TEACHER_SPARSE_SCAN_HISTORY_LENGTH
    append_actor_feet_contact: bool = True
    append_critic_foot_scan: bool = True
    append_critic_immunity: bool = True
    use_lightlp_terminations: bool = True
    use_algebraic_sparse_scan: bool = False
    terminate_on_pit_fall: bool = False
    random_level_reset_fraction: float = 0.10
    random_level_reset_max_level: int | None = None
    terrain_aware_commands: bool = True
    sparse_command_lin_vel_x: tuple[float, float] = (0.6, 2.0)
    sparse_command_straight_yaw_prob: float = 0.60
    sparse_command_gentle_ang_vel_z: tuple[float, float] = (-0.3, 0.3)

    def __post_init__(self):
        self.scene.terrain_generator = LIGHTLP_TERRAINS_CFG
        self.scene.max_init_terrain_level = 2
        self.scene.foot_scanner = FootScannerCfg(
            enable=True,
            resolution=FOOT_SCAN_RESOLUTION,
            size=FOOT_SCAN_SIZE,
        )
        self.reward = LightLPRewardCfg()
        self.append_actor_feet_contact = True
        self.append_critic_foot_scan = True
        self.append_critic_immunity = True
        self.use_lightlp_terminations = True
        self.use_algebraic_sparse_scan = False
        self.random_level_reset_fraction = 0.10
        self.random_level_reset_max_level = None
        self.random_level_reset_min_level = None
        self.terrain_aware_commands = True
        self.commands.ranges.lin_vel_x = (-0.6, 2.0)
        self.sparse_command_lin_vel_x = (0.6, 2.0)
        self.sparse_command_straight_yaw_prob = 0.60
        self.sparse_command_gentle_ang_vel_z = (-0.3, 0.3)
        self.amp_terrain_schedule.enable = True
        self.noise.noise_scales.height_scan = 0.0
        self.domain_rand.events.push_robot.func = mdp.push_by_setting_velocity_tagged
        generator = self.scene.terrain_generator
        sub = getattr(generator, "sub_terrains", None) or {}
        for name in ("stepping_stones", "raised_pillars"):
            cfg = sub.get(name)
            if cfg is not None and hasattr(cfg, "soft_fill"):
                cfg.soft_fill = False

        bind_robot_spec(self)


@configclass
class LightLPLocomotionAgentCfg(AmpLocomotionAgentCfg):
    experiment_name = "lightlp_locomotion"
    run_name = "lightlp"
    neptune_project = "lightlp_locomotion"
    wandb_project = "lightlp_locomotion"
    max_iterations = 40000

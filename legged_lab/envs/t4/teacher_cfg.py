# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""Stage E configuration for the T4 privileged locomotion teacher `pi_teacher`.

Everything that the plan freezes before formal training lives here: the
forward-asymmetric terrain privilege, the AMP style-weight schedule, the gait
representation and the adaptive terrain curriculum. Changing any of them is an
MDP change and therefore starts a new lineage.
"""

from __future__ import annotations

import math
import os

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
from legged_lab.assets.t4.constants import T4_JOINT_NAMES, T4_NOMINAL_FEET_Y_DISTANCE
from legged_lab.assets.t4.schemas import (
    AMP_FORMAL_EXPERT_DIR,
    AMP_FRAME_DIM,
    FOOT_SCAN_RESOLUTION,
    FOOT_SCAN_SIZE,
    NUM_T4_JOINTS,
    PROPRIO_HISTORY_LENGTH,
    TEACHER_SCAN_BODY,
    TEACHER_SCAN_HEIGHT_OFFSET,
    TEACHER_SCAN_OFFSET,
    TEACHER_SCAN_RESOLUTION,
    TEACHER_SCAN_SIZE,
    TEACHER_SPARSE_SCAN_HISTORY_LENGTH,
    amp_expert_files,
)
from legged_lab.assets.t4.t4 import T4_CFG
from legged_lab.envs.base.base_config import (
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
from legged_lab.terrains import T4_STAGE_E_SPARSE_TERRAINS_CFG, T4_STAGE_E_TERRAINS_CFG


@configclass
class T4GaitCfg:
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
class T4AmpTerrainScheduleCfg:
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
class T4TeacherRewardCfg:
    # Task/penalty weights follow the VITAL T4_27 task that trained this robot on
    # rough/stair terrain: stronger velocity tracking, a mild vertical-velocity
    # penalty (stairs require root z motion), and no hip roll/yaw action penalty.
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=2.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"std": 0.5})
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.15)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["A[LR]2", "A[LR]4", "Trunk"]),
            "threshold": 1.0,
        },
    )
    # A knee brushing a stair riser is expected on the up-stairs curriculum, so
    # shank contact is penalized more mildly than arm/trunk contact.
    shank_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["Shank_.*"]),
            "threshold": 1.0,
        },
    )
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names="Trunk")}, weight=-2.0
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_foot_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_foot_link"),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_foot_link"]), "threshold": 0.2},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*_foot_link"])},
    )
    foot_touchdown_impact = RewTerm(
        func=mdp.foot_touchdown_impact_penalty,
        weight=-0.08,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*_foot_link"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_foot_link"]),
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
                joint_names=["J_hip_[lr]_yaw", "J_hip_[lr]_roll", "J_arm_[lr]_01", "J_arm_[lr]_04"],
            )
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["J_arm_[lr]_02", "J_arm_[lr]_03"])},
    )
    # The wrists and the waist yaw only exist on T4; without a standing anchor they
    # drift, since the task reward never observes them.
    joint_deviation_wrist_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["J_arm_[lr]_0[567]", "J_waist_yaw"])},
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["J_hip_[lr]_pitch", "J_knee_[lr]_pitch", "J_ankle_[lr]_pitch", "J_ankle_[lr]_roll"],
            )
        },
    )

    gait_feet_frc_perio = RewTerm(func=mdp.gait_feet_frc_perio, weight=1.0, params={"delta_t": 0.02})
    gait_feet_spd_perio = RewTerm(func=mdp.gait_feet_spd_perio, weight=1.0, params={"delta_t": 0.02})
    gait_feet_frc_support_perio = RewTerm(func=mdp.gait_feet_frc_support_perio, weight=0.6, params={"delta_t": 0.02})

    ankle_torque = RewTerm(func=mdp.ankle_torque, weight=-0.0005)
    ankle_action = RewTerm(func=mdp.ankle_action, weight=-0.001)
    feet_y_distance = RewTerm(func=mdp.feet_y_distance, weight=-2.0, params={"target": T4_NOMINAL_FEET_Y_DISTANCE})


@configclass
class T4LocoTeacherEnvCfg:
    policy_role: str = "teacher"
    device: str = "cuda:0"
    random_level_reset_fraction: float = 0.0
    random_level_reset_max_level: int | None = None
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=20.0,
        # Formal starting point for the first Stage E lineage; the capacity probe
        # decides whether to scale to 2048/4096.
        num_envs=1024,
        env_spacing=2.5,
        robot=T4_CFG,
        terrain_type="generator",
        terrain_generator=T4_STAGE_E_TERRAINS_CFG,
        max_init_terrain_level=3,
        height_scanner=HeightScannerCfg(
            enable_height_scan=True,
            prim_body_name=TEACHER_SCAN_BODY,
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
        # Shank contact stays a penalty rather than a termination: a knee brushing a
        # stair riser is common on the up-stairs curriculum and must not end the episode.
        terminate_contacts_body_names=["Trunk", "A[LR]2", "A[LR]4"],
        feet_body_names=[".*_foot_link"],
    )
    reward = T4TeacherRewardCfg()
    append_actor_feet_contact: bool = False
    gait = T4GaitCfg()
    amp_terrain_schedule = T4AmpTerrainScheduleCfg()
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
                    "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
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


@configclass
class T4LocoTeacherAgentCfg(RslRlOnPolicyRunnerCfg):
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
        # Sagittal mirror augmentation + mirror loss, following the VITAL T4_27
        # recipe (mirror_loss_coeff=5.0). The mirror plan is frozen with the
        # observation schema in legged_lab/envs/t4/symmetry.py.
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            data_augmentation_func="legged_lab.envs.t4.symmetry:get_symmetric_states",
            mirror_loss_coeff=5.0,
        ),
        rnd_cfg=None,
    )
    clip_actions = None
    save_interval = 500
    runner_class_name = "AmpOnPolicyRunner"
    experiment_name = "t4_loco_teacher"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "t4_loco_teacher"
    wandb_project = "t4_loco_teacher"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"
    amp_reward_coef = 0.3
    amp_frame_dim = AMP_FRAME_DIM
    amp_joint_order = list(T4_JOINT_NAMES)
    amp_expert_dir = os.environ.get("T4_AMP_EXPERT_DIR", AMP_FORMAL_EXPERT_DIR)
    amp_motion_files = amp_expert_files(os.environ.get("T4_AMP_EXPERT_DIR", AMP_FORMAL_EXPERT_DIR))
    amp_num_preload_transitions = 200000
    amp_task_reward_lerp = 0.7
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * NUM_T4_JOINTS


@configclass
class T4SparseTeacherRewardCfg(T4TeacherRewardCfg):
    """LightLP §IV Table I on the mixed sparse teacher.

    Stage E extras (periodic gait, AMP, stumble) stay in the base class so
    continuous tiles keep the successful walk recipe; the env zeros them on
    sparse tiles. Table I terms that Stage E lacked are added here.
    Instant pit-fall −200 is not a sparse failure mode.
    """

    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"std": 0.5})
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names="Trunk")}, weight=0.0
    )
    upright_orientation = RewTerm(
        func=mdp.upright_orientation, params={"asset_cfg": SceneEntityCfg("robot", body_names="Trunk")}, weight=1.0
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["A[LR]2", "A[LR]4", "Trunk", "Shank_.*"]),
            "threshold": 1.0,
        },
    )
    shank_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["Shank_.*"]), "threshold": 1.0},
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=0.0)
    heading_error = RewTerm(func=mdp.heading_error, weight=-1.0)
    velocity_slack = RewTerm(func=mdp.velocity_slack, weight=1.5)
    illegal_footstep = RewTerm(func=mdp.illegal_footstep, weight=-1.0)
    opposite_direction = RewTerm(func=mdp.opposite_direction, weight=-1.0)
    hurdle_bar_contact = RewTerm(func=mdp.hurdle_bar_contact, weight=-2.0)
    foot_touchdown_impact = RewTerm(
        func=mdp.foot_acceleration_penalty,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_foot_link"]),
            "tau_s": 0.06,
            "threshold_mps2": 30.0,
        },
    )


@configclass
class T4LocoSparseTeacherEnvCfg(T4LocoTeacherEnvCfg):
    """One-stage LightLP §IV sparse teacher: real holes, no soft/hard switch."""

    teacher_scan_history_length: int = TEACHER_SPARSE_SCAN_HISTORY_LENGTH
    append_actor_feet_contact: bool = True
    append_critic_foot_scan: bool = True
    append_critic_immunity: bool = True
    use_lightlp_terminations: bool = True
    use_algebraic_sparse_scan: bool = False
    terminate_on_pit_fall: bool = False
    random_level_reset_fraction: float = 0.10
    # Exclusive. 4 → rows 0–3 (d ≤ 1/3) until easy stones actually get used.
    random_level_reset_max_level: int | None = 4

    def __post_init__(self):
        self.scene.terrain_generator = T4_STAGE_E_SPARSE_TERRAINS_CFG
        self.scene.max_init_terrain_level = 2
        self.scene.foot_scanner = FootScannerCfg(
            enable=True,
            resolution=FOOT_SCAN_RESOLUTION,
            size=FOOT_SCAN_SIZE,
        )
        self.reward = T4SparseTeacherRewardCfg()
        self.append_actor_feet_contact = True
        self.append_critic_foot_scan = True
        self.append_critic_immunity = True
        self.use_lightlp_terminations = True
        self.use_algebraic_sparse_scan = False
        self.random_level_reset_fraction = 0.10
        self.random_level_reset_max_level = 4
        self.amp_terrain_schedule.enable = True
        self.noise.noise_scales.height_scan = 0.0
        generator = self.scene.terrain_generator
        sub = getattr(generator, "sub_terrains", None) or {}
        for name in ("stepping_stones", "raised_pillars"):
            cfg = sub.get(name)
            if cfg is not None and hasattr(cfg, "soft_fill"):
                cfg.soft_fill = False


@configclass
class T4LocoSparseTeacherAgentCfg(T4LocoTeacherAgentCfg):
    experiment_name = "t4_loco_teacher_sparse"
    run_name = "t_sparse_lightlp_s6"
    neptune_project = "t4_loco_teacher_sparse"
    wandb_project = "t4_loco_teacher_sparse"
    max_iterations = 40000

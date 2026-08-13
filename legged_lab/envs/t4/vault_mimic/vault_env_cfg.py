"""T4 1 m box-vault mimic teacher environment (G1).

Flattened port of the proven PHP ``T4ClimbEnvCfg`` recipe onto the Stage E
plant: URDF ``t4_std`` (with sphere-hand collision), Stage E actuator gains,
uniform 0.25 rad action scale, and the repo-local reference motion instead of
a wandb registry.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import legged_lab.envs.t4.vault_mimic.mdp as mdp
from legged_lab.assets.t4.t4 import T4_CFG
from legged_lab.assets.t4.vault_contract import (
    T4_VAULT_ACTION_SCALE,
    T4_VAULT_ADAPTIVE_KERNEL_LAMBDA,
    T4_VAULT_ADAPTIVE_KERNEL_SIZE,
    T4_VAULT_ANCHOR_BODY_NAME,
    T4_VAULT_ANCHOR_ORI_TERMINATION_THRESHOLD,
    T4_VAULT_ANCHOR_TERMINATION_THRESHOLD,
    T4_VAULT_BOX_POS,
    T4_VAULT_BOX_ROT,
    T4_VAULT_BOX_SIZE,
    T4_VAULT_END_EFFECTOR_BODY_NAMES,
    T4_VAULT_FOOT_BODY_NAMES,
    T4_VAULT_FOOT_TERMINATION_THRESHOLD,
    T4_VAULT_MOTION_FILE,
    T4_VAULT_TRACKING_BODY_NAMES,
    T4_VAULT_WRIST_BODY_NAMES,
    T4_VAULT_WRIST_TERMINATION_THRESHOLD,
)

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

# Penalize contact on every body except the vault support surfaces (feet, wrists).
UNDESIRED_CONTACT_REGEX = r"^(?!(?:" + "|".join(T4_VAULT_END_EFFECTOR_BODY_NAMES) + r")$).+$"

# Procedural ground material; also installed as the sim default material.
GROUND_PHYSICS_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    friction_combine_mode="multiply",
    restitution_combine_mode="multiply",
    static_friction=1.0,
    dynamic_friction=1.0,
)


@configclass
class VaultSceneCfg(InteractiveSceneCfg):
    """Flat ground, the 1 m box from the reference scene, and the T4 robot."""

    # A procedural cuboid slab instead of TerrainImporter's ground plane: the
    # default GroundPlaneCfg references a Nucleus-hosted USD that is not
    # reachable from the training hosts. Top surface sits at z=0 and the
    # 800 m extent covers the env-origin grid at 4096 envs x 8 m spacing.
    # visual_material is intentionally None: PreviewSurface material creation
    # needs kit material extensions that fail in the headless container.
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.CuboidCfg(
            size=(800.0, 800.0, 2.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=GROUND_PHYSICS_MATERIAL,
            visual_material=None,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.0)),
    )
    robot: ArticulationCfg = T4_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    box = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ObstacleBox",
        spawn=sim_utils.CuboidCfg(
            size=T4_VAULT_BOX_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=None,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=T4_VAULT_BOX_POS, rot=T4_VAULT_BOX_ROT),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=10.0,
        debug_vis=False,
    )


@configclass
class CommandsCfg:
    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        motion_file=str(T4_VAULT_MOTION_FILE),
        anchor_body_name=T4_VAULT_ANCHOR_BODY_NAME,
        body_names=list(T4_VAULT_TRACKING_BODY_NAMES),
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
        adaptive_kernel_size=T4_VAULT_ADAPTIVE_KERNEL_SIZE,
        adaptive_lambda=T4_VAULT_ADAPTIVE_KERNEL_LAMBDA,
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        use_default_offset=True,
        scale=T4_VAULT_ACTION_SCALE,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(
            func=mdp.motion_anchor_pos_b,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.25, n_max=0.25),
        )
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=T4_VAULT_ANCHOR_BODY_NAME),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
    )


@configclass
class RewardsCfg:
    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14},
    )
    motion_wrist_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.5,
        params={
            "command_name": "motion",
            "std": 0.15,
            "body_names": list(T4_VAULT_WRIST_BODY_NAMES),
        },
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[UNDESIRED_CONTACT_REGEX]),
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": T4_VAULT_ANCHOR_TERMINATION_THRESHOLD,
        },
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "motion",
            "threshold": T4_VAULT_ANCHOR_ORI_TERMINATION_THRESHOLD,
        },
    )
    foot_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": T4_VAULT_FOOT_TERMINATION_THRESHOLD,
            "body_names": list(T4_VAULT_FOOT_BODY_NAMES),
        },
    )
    wrist_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": T4_VAULT_WRIST_TERMINATION_THRESHOLD,
            "body_names": list(T4_VAULT_WRIST_BODY_NAMES),
        },
    )


@configclass
class T4VaultMimicEnvCfg(ManagerBasedRLEnvCfg):
    """G1 vault mimic teacher on the Stage E plant."""

    scene: VaultSceneCfg = VaultSceneCfg(num_envs=2048, env_spacing=8)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = GROUND_PHYSICS_MATERIAL
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15


@configclass
class T4VaultMimicPlayEnvCfg(T4VaultMimicEnvCfg):
    """Deterministic playback from the first reference frame."""

    def __post_init__(self):
        super().__post_init__()
        self.events.physics_material = None
        self.events.add_joint_default_pos = None
        self.events.base_com = None
        self.events.push_robot = None
        self.commands.motion.sampling_strategy = "zero"
        self.observations.policy.enable_corruption = False

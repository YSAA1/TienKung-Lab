"""G2 heightscan vault skill environment.

Student policy obs is the Stage E 1155D teacher-actor contract. The G1 150D
tracking observation is exposed only as the ``teacher`` group so Distillation
can query the frozen mimic expert. Motion / RSI stay in the scene for that
query; they are not part of the student observation.
"""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

import legged_lab.envs.t4.vault_mimic.mdp as g1_mdp
import legged_lab.envs.t4.vault_skill.mdp as mdp
from legged_lab.assets.t4.vault_skill_contract import G2_ACTION_SCALE
from legged_lab.envs.t4.vault_mimic.vault_env_cfg import (
    ActionsCfg,
    CommandsCfg,
    EventCfg,
    RewardsCfg,
    T4VaultMimicEnvCfg,
    TerminationsCfg,
    VaultSceneCfg,
)


@configclass
class VaultSkillSceneCfg(VaultSceneCfg):
    """Same G1 scene; scan is analytic, so no RayCaster mesh is required."""


@configclass
class SkillObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actor_obs = ObsTerm(func=mdp.g2_actor_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class TeacherCfg(ObsGroup):
        """G1 policy observation, noise-free, for the frozen mimic expert."""

        command = ObsTerm(func=g1_mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=g1_mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=g1_mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=g1_mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=g1_mdp.base_ang_vel)
        joint_pos = ObsTerm(func=g1_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=g1_mdp.joint_vel_rel)
        actions = ObsTerm(func=g1_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    teacher: TeacherCfg = TeacherCfg()
    critic: TeacherCfg = TeacherCfg()


@configclass
class T4VaultSkillEnvCfg(T4VaultMimicEnvCfg):
    """G2 distillation env: 1155D student + 150D G1 teacher query."""

    scene: VaultSkillSceneCfg = VaultSkillSceneCfg(num_envs=2048, env_spacing=8)
    observations: SkillObservationsCfg = SkillObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.scale = G2_ACTION_SCALE
        self.observations.policy.enable_corruption = False
        self.observations.teacher.enable_corruption = False
        self.events.push_robot = None

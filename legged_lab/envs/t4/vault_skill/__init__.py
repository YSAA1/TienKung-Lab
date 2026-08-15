"""Gym registrations for the T4 G2 heightscan vault skill."""

import gymnasium as gym

from .agents import T4VaultSkillDistillRunnerCfg
from .skill_env_cfg import T4VaultSkillEnvCfg

gym.register(
    id="t4_vault_skill",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": T4VaultSkillEnvCfg,
        "rsl_rl_cfg_entry_point": T4VaultSkillDistillRunnerCfg,
    },
)

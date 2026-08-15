"""Gym registrations for the T4 vault mimic teacher (G1)."""

import gymnasium as gym

from .agents import T4VaultMimicPPORunnerCfg
from .vault_env_cfg import T4VaultMimicEnvCfg, T4VaultMimicEvalEnvCfg, T4VaultMimicPlayEnvCfg

gym.register(
    id="t4_vault_mimic",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": T4VaultMimicEnvCfg,
        "rsl_rl_cfg_entry_point": T4VaultMimicPPORunnerCfg,
    },
)

gym.register(
    id="t4_vault_mimic_play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": T4VaultMimicPlayEnvCfg,
        "rsl_rl_cfg_entry_point": T4VaultMimicPPORunnerCfg,
    },
)

gym.register(
    id="t4_vault_mimic_eval",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": T4VaultMimicEvalEnvCfg,
        "rsl_rl_cfg_entry_point": T4VaultMimicPPORunnerCfg,
    },
)

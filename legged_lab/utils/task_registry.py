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

from copy import deepcopy
from typing import TYPE_CHECKING, Tuple

from rsl_rl.env import VecEnv

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env import BaseEnvConfig
    from legged_lab.envs.base.base_env_config import BaseAgentConfig


class TaskRegistry:
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}

    def register(self, name: str, task_class: VecEnv, env_cfg: "BaseEnvConfig", train_cfg: "BaseAgentConfig"):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg

    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]

    def get_cfgs(self, name) -> Tuple["BaseEnvConfig", "BaseAgentConfig"]:
        """Return per-call deep copies so caller mutations never leak into the registry.

        Scripts routinely tune ``num_envs``/seeds/manifests on the returned cfgs;
        sharing the stored object let one task's edits contaminate every later
        ``get_cfgs`` call in the same process (seen across eval/probe scripts).
        Note: dataclass ``MISSING`` sentinels do not survive deepcopy by identity
        (``deepcopy(MISSING) is not MISSING``); no consumer in this repo or the
        vendored rsl_rl compares instance values with ``is``, so this is safe.
        """
        return deepcopy(self.env_cfgs[name]), deepcopy(self.train_cfgs[name])


# make global task registry
task_registry = TaskRegistry()

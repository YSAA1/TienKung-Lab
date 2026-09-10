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

from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg
from legged_lab.envs.g1.teacher_cfg import G1LocoTeacherAgentCfg, G1LocoTeacherEnvCfg
from legged_lab.envs.t4.depth_student_cfg import T4SparseDepthStudentAgentCfg
from legged_lab.envs.t4.depth_student_env import (  # noqa: E402
    T4LocoSparseDepthStudentEnvCfg,
)
from legged_lab.envs.t4.teacher_cfg import (
    T4LocoSparseTeacherAgentCfg,
    T4LocoSparseTeacherEnvCfg,
    T4LocoTeacherAgentCfg,
    T4LocoTeacherEnvCfg,
)
from legged_lab.envs.tienkung.run_cfg import TienKungRunAgentCfg, TienKungRunFlatEnvCfg
from legged_lab.envs.tienkung.run_with_sensor_cfg import (
    TienKungRunWithSensorAgentCfg,
    TienKungRunWithSensorFlatEnvCfg,
)
from legged_lab.envs.tienkung.tienkung_env import TienKungEnv
from legged_lab.envs.tienkung.walk_cfg import (
    TienKungWalkAgentCfg,
    TienKungWalkFlatEnvCfg,
)
from legged_lab.envs.tienkung.walk_with_sensor_cfg import (
    TienKungWalkWithSensorAgentCfg,
    TienKungWalkWithSensorFlatEnvCfg,
)
from legged_lab.envs.z2.teacher_cfg import Z2LocoTeacherAgentCfg, Z2LocoTeacherEnvCfg
from legged_lab.locomotion.depth_env import LightLPDepthDistillationEnv  # noqa: E402
from legged_lab.locomotion.env import LocomotionEnv
from legged_lab.utils.task_registry import task_registry

task_registry.register("walk", TienKungEnv, TienKungWalkFlatEnvCfg(), TienKungWalkAgentCfg())
task_registry.register("run", TienKungEnv, TienKungRunFlatEnvCfg(), TienKungRunAgentCfg())
task_registry.register(
    "walk_with_sensor", TienKungEnv, TienKungWalkWithSensorFlatEnvCfg(), TienKungWalkWithSensorAgentCfg()
)
task_registry.register(
    "run_with_sensor", TienKungEnv, TienKungRunWithSensorFlatEnvCfg(), TienKungRunWithSensorAgentCfg()
)
task_registry.register("g1_loco_teacher", LocomotionEnv, G1LocoTeacherEnvCfg(), G1LocoTeacherAgentCfg())
task_registry.register("z2_loco_teacher", LocomotionEnv, Z2LocoTeacherEnvCfg(), Z2LocoTeacherAgentCfg())
task_registry.register("t4_loco_teacher", LocomotionEnv, T4LocoTeacherEnvCfg(), T4LocoTeacherAgentCfg())
task_registry.register(
    "t4_loco_teacher_sparse", LocomotionEnv, T4LocoSparseTeacherEnvCfg(), T4LocoSparseTeacherAgentCfg()
)
task_registry.register(
    "t4_loco_sparse_depth_student",
    LightLPDepthDistillationEnv,
    T4LocoSparseDepthStudentEnvCfg(),
    T4SparseDepthStudentAgentCfg(),
)

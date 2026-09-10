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

"""Registry isolation contract: ``get_cfgs`` returns per-call copies.

Runs only where the full IsaacLab runtime (incl. ``warp``) is importable,
e.g. an Isaac Sim python with the AppLauncher environment set up. Machines
without it skip; the same assertions also execute implicitly at every
training launch, which fails fast if registry isolation regresses.
"""

import pytest

pytest.importorskip("isaaclab")
pytest.importorskip("warp")


def test_get_cfgs_returns_mutation_isolated_copies():
    import legged_lab.envs  # noqa: F401 - import side effect registers all tasks
    from legged_lab.utils import task_registry

    for task in ("t4_loco_teacher", "g1_loco_teacher", "z2_loco_teacher"):
        env_a, _ = task_registry.get_cfgs(task)
        env_b, _ = task_registry.get_cfgs(task)
        assert env_a is not env_b, task
        nominal = env_a.scene.num_envs
        env_a.scene.num_envs = 12345
        fresh, _ = task_registry.get_cfgs(task)
        assert fresh.scene.num_envs == nominal, task


def test_motion_experiment_profile_stays_caller_local():
    import legged_lab.envs  # noqa: F401 - import side effect registers all tasks
    from legged_lab.envs.g1.motion_experiment import apply_vital_motion_experiment
    from legged_lab.utils import task_registry

    env_cfg, _ = task_registry.get_cfgs("g1_loco_teacher")
    apply_vital_motion_experiment(env_cfg, "vital_v3")
    pristine, _ = task_registry.get_cfgs("g1_loco_teacher")
    assert pristine.domain_rand.action_delay.enable is False
    assert not hasattr(pristine.domain_rand.events, "physics_material_ramp")

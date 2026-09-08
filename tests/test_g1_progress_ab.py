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

# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

import importlib.util
from pathlib import Path

import numpy as np
import torch


def load(name):
    path = Path(__file__).resolve().parents[1] / "legged_lab" / "locomotion" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


curriculum = load("curriculum")
ProgressMonitor = load("progress").ProgressMonitor


def test_local_loop_promotes_only_legacy_even_with_good_tracking():
    t = np.arange(1001) * 0.02
    xy = np.column_stack([0.06 * np.sin(2 * np.pi * t), np.zeros_like(t)])
    path = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
    peak = np.linalg.norm(xy, axis=1).max()
    assert path > 4 and peak < 0.1
    args = (np.array([path]), np.array([0.8]), np.array([0.6]), 8)
    legacy, _ = curriculum.lightlp_terrain_level_moves(*args)
    traversal, down = curriculum.lightlp_terrain_level_moves(*args, max_radial_dist=np.array([peak]))
    assert legacy[0] and not traversal[0] and not down[0]


def test_traversal_handles_spawn_offset_diagonal_and_standing():
    # Offset spawn at x=.5 reaches x=4.26 after only 3.76 m; diagonal
    # traversal uses the documented radial criterion, independently of OOB.
    peaks = np.array([4.26, np.hypot(3, 3), 4.0, 0.71, 5, 5])
    tracking = np.array([0.8, 0.8, 0.8, 0.8, 1, 0.1])
    cmd = np.array([0.6, 0.6, 0.6, 0.6, 0, 0.6])
    up, down = curriculum.lightlp_terrain_level_moves(np.full(6, 6.0), tracking, cmd, 8, max_radial_dist=peaks)
    np.testing.assert_array_equal(up, [True, True, False, False, False, False])
    np.testing.assert_array_equal(down, [False, False, False, False, False, True])


def test_progress_constant_walk_and_partial_reset_excludes_teleport():
    m = ProgressMonitor(2, "cpu", 0.02)
    ids = torch.arange(2)
    xy = torch.tensor([[100.0, 50.0], [0.0, 0.0]])
    m.reset(ids, xy)
    cmd = torch.tensor([[0.7, 0.0], [0.0, 0.0]])
    for _ in range(50):
        xy += cmd * 0.02
        m.update(xy, cmd)
    values = m.metrics(ids)
    assert abs(values["net_displacement_m"][0] - 0.7) < 2e-4
    assert abs(values["command_completion"][0] - 1) < 2e-4
    assert abs(values["window_net_speed_mps"][0] - 0.7) < 2e-4
    xy[0] = torch.tensor([-100.0, -50.0])
    m.reset(torch.tensor([0]), xy[:1])
    m.update(xy, cmd)
    assert m.metrics(ids)["path_length_m"][0] == 0
    assert m.steps.tolist() == [1, 51]
    assert torch.isfinite(torch.stack(list(m.metrics(ids).values()))).all()


def test_progress_loop_and_command_turn_use_signed_world_displacement():
    m = ProgressMonitor(1, "cpu", 0.02)
    ids = torch.tensor([0])
    m.reset(ids, torch.zeros(1, 2))
    for step in range(1, 201):
        xy = torch.tensor([[0.04 * np.sin(step * np.pi / 10), 0]], dtype=torch.float32)
        m.update(xy, torch.tensor([[0.7, 0.0]]))
    values = m.metrics(ids)
    assert values["path_length_m"][0] > 1
    assert values["net_displacement_m"][0] < 1e-5
    assert abs(values["command_completion"][0]) < 1e-5
    m.reset(ids, torch.zeros(1, 2))
    m.update(torch.tensor([[0.014, 0.0]]), torch.tensor([[0.7, 0.0]]))
    m.update(torch.tensor([[0.014, 0.014]]), torch.tensor([[0.0, 0.7]]))
    assert abs(m.metrics(ids)["command_completion"][0] - 1) < 1e-5

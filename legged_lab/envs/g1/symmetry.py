# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""Sagittal mirror symmetry for the G1 sparse teacher (29-DoF).

Layout matches :meth:`T4LocoEnv._proprio_field_slice` when ``num_actions != 27``.
Scan / foot-scan mirroring reuses the same LightLP sparse contracts as T4.
"""

from __future__ import annotations

from functools import lru_cache

from legged_lab.assets.t4.schemas import (
    FOOT_SCAN_BOTH_DIM,
    PROPRIO_HISTORY_LENGTH,
    TEACHER_SCAN_DIM,
    TEACHER_SCAN_SHAPE,
    TEACHER_SPARSE_CONTACT_DIM,
    TEACHER_SPARSE_IMMUNITY_DIM,
    TEACHER_SPARSE_SCAN_HISTORY_LENGTH,
)
from legged_lab.assets.unitree_g1.constants import G1_29DOF_JOINT_NAMES, NUM_G1_29DOF_JOINTS

G1_PROPRIO_FRAME_DIM = 9 + 3 * NUM_G1_29DOF_JOINTS + 6
G1_CRITIC_FRAME_DIM = G1_PROPRIO_FRAME_DIM + 5
G1_SPARSE_ACTOR_OBS_DIM = (
    G1_PROPRIO_FRAME_DIM * PROPRIO_HISTORY_LENGTH
    + TEACHER_SCAN_DIM * TEACHER_SPARSE_SCAN_HISTORY_LENGTH
    + TEACHER_SPARSE_CONTACT_DIM
)
G1_SPARSE_CRITIC_OBS_DIM = (
    G1_CRITIC_FRAME_DIM * PROPRIO_HISTORY_LENGTH
    + TEACHER_SCAN_DIM * TEACHER_SPARSE_SCAN_HISTORY_LENGTH
    + FOOT_SCAN_BOTH_DIM
    + TEACHER_SPARSE_IMMUNITY_DIM
)

_DEVICE_PLAN_CACHE: dict = {}


def _grid_plan(start: int, shape: tuple[int, int]) -> list[int]:
    num_x, num_y = shape
    return [start + (num_y - 1 - iy) * num_x + ix for iy in range(num_y) for ix in range(num_x)]


def _apply_plan(values, indices, signs):
    if type(values).__module__.split(".")[0] == "torch":
        import torch

        key = (indices, signs, values.device, values.dtype)
        cached = _DEVICE_PLAN_CACHE.get(key)
        if cached is None:
            cached = (
                torch.tensor(indices, dtype=torch.long, device=values.device),
                torch.tensor(signs, dtype=values.dtype, device=values.device),
            )
            _DEVICE_PLAN_CACHE[key] = cached
        index_tensor, sign_tensor = cached
        return values.index_select(-1, index_tensor) * sign_tensor
    import numpy as np

    return values[..., np.asarray(indices)] * np.asarray(signs, dtype=values.dtype)


def _concat(first, second):
    if type(first).__module__.split(".")[0] == "torch":
        import torch

        return torch.cat((first, second), dim=0)
    import numpy as np

    return np.concatenate((first, second), axis=0)


def _concat_last(first, second):
    if type(first).__module__.split(".", 1)[0] == "torch":
        import torch

        return torch.cat((first, second), dim=-1)
    import numpy as np

    return np.concatenate((first, second), axis=-1)


def _joint_mirror_sign(name: str) -> float:
    if name.endswith(("_pitch_joint", "_knee_joint", "_elbow_joint")):
        return 1.0
    if name.endswith(("_roll_joint", "_yaw_joint")):
        return -1.0
    raise ValueError(f"no mirror sign rule for G1 joint {name!r}")


def _joint_counterpart(name: str) -> str:
    if name.startswith("left_"):
        return "right_" + name[len("left_") :]
    if name.startswith("right_"):
        return "left_" + name[len("right_") :]
    return name


@lru_cache(maxsize=1)
def joint_mirror_map() -> tuple[tuple[int, ...], tuple[float, ...]]:
    index_by_name = {name: index for index, name in enumerate(G1_29DOF_JOINT_NAMES)}
    sources = []
    signs = []
    for name in G1_29DOF_JOINT_NAMES:
        counterpart = _joint_counterpart(name)
        if counterpart not in index_by_name:
            raise ValueError(f"missing mirror counterpart {counterpart!r} for {name!r}")
        sources.append(index_by_name[counterpart])
        signs.append(_joint_mirror_sign(name))
    return tuple(sources), tuple(signs)


def _proprio_offsets(num_joints: int) -> dict[str, int]:
    joint_pos = 9
    joint_vel = joint_pos + num_joints
    previous_action = joint_vel + num_joints
    gait_sin = previous_action + num_joints
    return {
        "base_ang_vel": 0,
        "projected_gravity": 3,
        "velocity_command": 6,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "previous_action": previous_action,
        "gait_phase_sin": gait_sin,
        "gait_phase_cos": gait_sin + 2,
        "gait_air_ratio": gait_sin + 4,
    }


def _frame_plan(is_critic: bool) -> tuple[list[int], list[float]]:
    frame_dim = G1_CRITIC_FRAME_DIM if is_critic else G1_PROPRIO_FRAME_DIM
    indices = list(range(frame_dim))
    signs = [1.0] * frame_dim
    offsets = _proprio_offsets(NUM_G1_29DOF_JOINTS)
    joint_sources, joint_signs = joint_mirror_map()

    def negate(index: int) -> None:
        signs[index] = -1.0

    def swap_pair(start: int) -> None:
        indices[start], indices[start + 1] = indices[start + 1], indices[start]

    def joint_block(start: int) -> None:
        for offset, (source, sign) in enumerate(zip(joint_sources, joint_signs)):
            indices[start + offset] = start + source
            signs[start + offset] = sign

    negate(offsets["base_ang_vel"] + 0)
    negate(offsets["base_ang_vel"] + 2)
    negate(offsets["projected_gravity"] + 1)
    negate(offsets["velocity_command"] + 1)
    negate(offsets["velocity_command"] + 2)
    for field in ("joint_pos", "joint_vel", "previous_action"):
        joint_block(offsets[field])
    for field in ("gait_phase_sin", "gait_phase_cos", "gait_air_ratio"):
        swap_pair(offsets[field])

    if is_critic:
        base_lin_vel = offsets["gait_air_ratio"] + 2
        feet_contact = base_lin_vel + 3
        negate(base_lin_vel + 1)
        swap_pair(feet_contact)
    return indices, signs


def _scan_plan(start: int) -> list[int]:
    num_x, num_y = TEACHER_SCAN_SHAPE
    return [start + (num_y - 1 - iy) * num_x + ix for iy in range(num_y) for ix in range(num_x)]


def _mirror_scan_stack(start: int, history: int) -> list[int]:
    indices: list[int] = []
    for h in range(history):
        indices.extend(_scan_plan(start + h * TEACHER_SCAN_DIM))
    return indices


def _mirror_foot_scan_stack(start: int) -> list[int]:
    from legged_lab.assets.t4.schemas import FOOT_SCAN_DIM, FOOT_SCAN_SHAPE

    left = _grid_plan(start, FOOT_SCAN_SHAPE)
    right = _grid_plan(start + FOOT_SCAN_DIM, FOOT_SCAN_SHAPE)
    return right + left


@lru_cache(maxsize=2)
def sparse_observation_mirror_plan(is_critic: bool) -> tuple[tuple[int, ...], tuple[float, ...]]:
    frame_indices, frame_signs = _frame_plan(is_critic)
    frame_dim = len(frame_indices)
    history = PROPRIO_HISTORY_LENGTH
    indices: list[int] = []
    signs: list[float] = []
    for history_index in range(history):
        offset = history_index * frame_dim
        indices.extend(offset + index for index in frame_indices)
        signs.extend(frame_signs)
    scan_start = history * frame_dim
    indices.extend(_mirror_scan_stack(scan_start, TEACHER_SPARSE_SCAN_HISTORY_LENGTH))
    signs.extend([1.0] * (TEACHER_SCAN_DIM * TEACHER_SPARSE_SCAN_HISTORY_LENGTH))
    if is_critic:
        foot_start = scan_start + TEACHER_SCAN_DIM * TEACHER_SPARSE_SCAN_HISTORY_LENGTH
        indices.extend(_mirror_foot_scan_stack(foot_start))
        signs.extend([1.0] * FOOT_SCAN_BOTH_DIM)
        if TEACHER_SPARSE_IMMUNITY_DIM:
            immune_i = foot_start + FOOT_SCAN_BOTH_DIM
            indices.append(immune_i)
            signs.append(1.0)
    return tuple(indices), tuple(signs)


def mirror_actions(actions):
    sources, signs = joint_mirror_map()
    return _apply_plan(actions, sources, signs)


def mirror_observations(obs, is_critic: bool):
    width = obs.shape[-1]
    if not is_critic and width == G1_SPARSE_ACTOR_OBS_DIM:
        body = G1_PROPRIO_FRAME_DIM * PROPRIO_HISTORY_LENGTH
        indices, signs = sparse_observation_mirror_plan(False)
        core_w = body + TEACHER_SCAN_DIM * TEACHER_SPARSE_SCAN_HISTORY_LENGTH
        mirrored = _apply_plan(obs[..., :core_w], indices, signs)
        contact = obs[..., -TEACHER_SPARSE_CONTACT_DIM:]
        return _concat_last(mirrored, contact[..., [1, 0]])
    if is_critic and width == G1_SPARSE_CRITIC_OBS_DIM:
        indices, signs = sparse_observation_mirror_plan(True)
        return _apply_plan(obs, indices, signs)
    raise ValueError(
        f"{'critic' if is_critic else 'actor'} observation width {width} does not match "
        f"G1 sparse contract ({G1_SPARSE_CRITIC_OBS_DIM if is_critic else G1_SPARSE_ACTOR_OBS_DIM})"
    )


def get_symmetric_states(obs=None, actions=None, env=None, obs_type=None, is_critic=None):
    if is_critic is None:
        is_critic = obs_type == "critic"
    return_augmented_batch = obs_type is not None

    mirrored_obs = mirror_observations(obs, bool(is_critic)) if obs is not None else None
    mirrored_actions = mirror_actions(actions) if actions is not None else None

    if return_augmented_batch:
        if mirrored_obs is not None:
            mirrored_obs = _concat(obs, mirrored_obs)
        if mirrored_actions is not None:
            mirrored_actions = _concat(actions, mirrored_actions)
    return mirrored_obs, mirrored_actions

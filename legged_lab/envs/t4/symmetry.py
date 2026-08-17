# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""Sagittal (left/right) mirror symmetry for the T4 Stage E teacher.

The mirror is a fixed index permutation plus a sign vector derived entirely
from the frozen schemas in :mod:`legged_lab.assets.t4.schemas`, so it can be
contract-tested without Isaac Sim (see ``tests/test_t4_observation_contracts.py``).
Joint mirror signs come from the T4 MJCF joint axes: reflection through the
sagittal plane keeps y-axis (pitch-like) joints and negates x/z-axis
(roll/yaw-like) joints. The arm joints are numbered rather than named by axis,
so their signs are pinned explicitly:

- ``J_arm_*_01`` shoulder pitch-like (axis ~y): +1
- ``J_arm_*_02`` shoulder roll (axis x): -1
- ``J_arm_*_03`` shoulder yaw (axis z): -1
- ``J_arm_*_04`` elbow pitch (axis y): +1
- ``J_arm_*_05`` wrist yaw (axis z): -1
- ``J_arm_*_06`` wrist pitch (axis y): +1
- ``J_arm_*_07`` wrist roll (axis x): -1

The entry point :func:`get_symmetric_states` implements the RSL-RL symmetry
``data_augmentation_func`` interface used by ``AMPPPO``.
"""

from __future__ import annotations

from functools import lru_cache

from legged_lab.assets.t4.constants import T4_JOINT_NAMES
from legged_lab.assets.t4.schemas import (
    CRITIC_EXTRA_FIELDS,
    CRITIC_FRAME_DIM,
    PROPRIO_FIELDS,
    PROPRIO_FRAME_DIM,
    TEACHER_PAPER_ACTOR_OBS_DIM,
    TEACHER_PAPER_CONTACT_DIM,
    TEACHER_SCAN_DIM,
    TEACHER_SCAN_SHAPE,
)

_ARM_NUMBER_SIGNS = {
    "01": 1.0,
    "02": -1.0,
    "03": -1.0,
    "04": 1.0,
    "05": -1.0,
    "06": 1.0,
    "07": -1.0,
}


def _joint_mirror_sign(name: str) -> float:
    if name.startswith("J_arm_"):
        return _ARM_NUMBER_SIGNS[name.rsplit("_", 1)[-1]]
    if name.endswith("_pitch"):
        return 1.0
    if name.endswith(("_roll", "_yaw")):
        return -1.0
    raise ValueError(f"no mirror sign rule for T4 joint {name!r}")


def _joint_counterpart(name: str) -> str:
    if "_l_" in name:
        return name.replace("_l_", "_r_")
    if "_r_" in name:
        return name.replace("_r_", "_l_")
    return name


@lru_cache(maxsize=1)
def joint_mirror_map() -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Source index and sign per joint, in ``T4_JOINT_NAMES`` order."""
    index_by_name = {name: index for index, name in enumerate(T4_JOINT_NAMES)}
    sources = []
    signs = []
    for name in T4_JOINT_NAMES:
        counterpart = _joint_counterpart(name)
        if counterpart not in index_by_name:
            raise ValueError(f"missing mirror counterpart {counterpart!r} for {name!r}")
        sources.append(index_by_name[counterpart])
        signs.append(_joint_mirror_sign(name))
    return tuple(sources), tuple(signs)


def _field_offsets(fields) -> dict[str, int]:
    offsets = {}
    cursor = 0
    for name, width in fields:
        offsets[name] = cursor
        cursor += width
    return offsets


def _frame_plan(is_critic: bool) -> tuple[list[int], list[float]]:
    """Mirror permutation and signs for one proprio (or critic) frame."""
    frame_dim = CRITIC_FRAME_DIM if is_critic else PROPRIO_FRAME_DIM
    indices = list(range(frame_dim))
    signs = [1.0] * frame_dim

    offsets = _field_offsets(PROPRIO_FIELDS + (CRITIC_EXTRA_FIELDS if is_critic else ()))
    joint_sources, joint_signs = joint_mirror_map()

    def negate(index: int) -> None:
        signs[index] = -1.0

    def swap_pair(start: int) -> None:
        indices[start], indices[start + 1] = indices[start + 1], indices[start]

    def joint_block(start: int) -> None:
        for offset, (source, sign) in enumerate(zip(joint_sources, joint_signs)):
            indices[start + offset] = start + source
            signs[start + offset] = sign

    # Angular velocity is an axial vector: negate roll (x) and yaw (z) rates.
    negate(offsets["base_ang_vel"] + 0)
    negate(offsets["base_ang_vel"] + 2)
    # Gravity is a polar vector: negate the lateral component.
    negate(offsets["projected_gravity"] + 1)
    # Command (vx, vy, wz): negate vy and wz.
    negate(offsets["velocity_command"] + 1)
    negate(offsets["velocity_command"] + 2)
    for field in ("joint_pos", "joint_vel", "previous_action"):
        joint_block(offsets[field])
    # The gait clock is (left, right) per field; mirroring swaps the legs.
    for field in ("gait_phase_sin", "gait_phase_cos", "gait_air_ratio"):
        swap_pair(offsets[field])

    if is_critic:
        negate(offsets["base_lin_vel"] + 1)
        swap_pair(offsets["feet_contact"])
    return indices, signs


def _grid_plan(start: int, shape: tuple[int, int]) -> list[int]:
    """Sagittal mirror of an ``ordering='xy'`` grid: reverse the outer y axis."""
    num_x, num_y = shape
    return [start + (num_y - 1 - iy) * num_x + ix for iy in range(num_y) for ix in range(num_x)]


def _scan_plan(start: int) -> list[int]:
    """Mirror permutation for the flattened teacher height scan."""
    return _grid_plan(start, TEACHER_SCAN_SHAPE)


@lru_cache(maxsize=8)
def observation_mirror_plan(is_critic: bool, history_length: int) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Full-observation mirror plan: history-stacked frames plus the scan tail."""
    frame_indices, frame_signs = _frame_plan(is_critic)
    frame_dim = len(frame_indices)
    indices: list[int] = []
    signs: list[float] = []
    for history_index in range(history_length):
        offset = history_index * frame_dim
        indices.extend(offset + index for index in frame_indices)
        signs.extend(frame_signs)
    indices.extend(_scan_plan(start=history_length * frame_dim))
    signs.extend([1.0] * TEACHER_SCAN_DIM)
    return tuple(indices), tuple(signs)


_DEVICE_PLAN_CACHE: dict = {}


def _apply_plan(values, indices, signs):
    """Gather + sign-multiply on the last axis; works on torch and numpy."""
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


def mirror_actions(actions):
    """Mirror an action batch in ``T4_JOINT_NAMES`` order."""
    sources, signs = joint_mirror_map()
    return _apply_plan(actions, sources, signs)


def mirror_observations(obs, is_critic: bool):
    """Mirror an actor/critic observation batch (history frames + scan tail)."""
    frame_dim = CRITIC_FRAME_DIM if is_critic else PROPRIO_FRAME_DIM
    extra = 0
    width = obs.shape[-1]
    if not is_critic and width == TEACHER_PAPER_ACTOR_OBS_DIM:
        extra = TEACHER_PAPER_CONTACT_DIM
        width -= extra
    body_width = width - TEACHER_SCAN_DIM
    if body_width <= 0 or body_width % frame_dim:
        raise ValueError(
            f"{'critic' if is_critic else 'actor'} observation width {obs.shape[-1]} does not match "
            f"{frame_dim}-D frames plus a {TEACHER_SCAN_DIM}-D scan"
        )
    indices, signs = observation_mirror_plan(bool(is_critic), body_width // frame_dim)
    core = obs[..., :width] if extra else obs
    mirrored = _apply_plan(core, indices, signs)
    if extra:
        contact = obs[..., -extra:]
        swapped = contact[..., [1, 0]] if extra == 2 else contact
        return _concat_last(mirrored, swapped)
    return mirrored


def _concat_last(first, second):
    if type(first).__module__.split(".", 1)[0] == "torch":
        import torch

        return torch.cat((first, second), dim=-1)
    import numpy as np

    return np.concatenate((first, second), axis=-1)


def get_symmetric_states(obs=None, actions=None, env=None, obs_type=None, is_critic=None):
    """RSL-RL symmetry hook: return ``[original; mirrored]`` batches.

    ``env`` is accepted for interface compatibility but unused; the T4 layout
    is frozen in the schemas module.
    """
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

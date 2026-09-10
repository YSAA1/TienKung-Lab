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

"""Isaac-free Z2 probe contracts: negative AMP, terrain restore, pulse mapping."""

from __future__ import annotations

from pathlib import Path

from legged_lab.assets.z2.constants import Z2_29DOF_JOINT_NAMES
from legged_lab.assets.z2.schemas import AMP_FRAME_DIM

FIXTURE_DIR = Path(__file__).resolve().parent / "probe_negatives"
G1_70D_FIXTURE = FIXTURE_DIR / "g1_joint_order_70d.txt"
WIDTH64_FIXTURE = FIXTURE_DIR / "upstream_width_64d.txt"

NEGATIVE_AMP_CASES: tuple[tuple[str, Path, str], ...] = (
    ("g1_70d", G1_70D_FIXTURE, "runtime contract expects"),
    ("upstream_64d", WIDTH64_FIXTURE, "does not match schema width"),
)


def mapped_policy_joints(sim_joint_names, policy_joint_ids) -> list[str]:
    return [sim_joint_names[int(index)] for index in policy_joint_ids]


def assert_policy_maps_onto_sim(sim_joint_names, policy_joint_ids, policy_joint_names) -> None:
    mapped = mapped_policy_joints(sim_joint_names, policy_joint_ids)
    if mapped != list(policy_joint_names):
        raise AssertionError(f"policy-to-sim mapping {mapped} != {list(policy_joint_names)}")


def signed_pair_delta(null_delta, pulsed_delta, channel: int) -> float:
    return float(pulsed_delta[channel]) - float(null_delta[channel])


def assert_signed_channel_response(
    null_delta, pulsed_delta, channel: int, expected_sign: int, min_abs: float = 1.0e-5
) -> float:
    if expected_sign not in (-1, 1):
        raise ValueError("expected_sign must be +1 or -1")
    signed = signed_pair_delta(null_delta, pulsed_delta, channel)
    if signed * expected_sign <= min_abs:
        raise AssertionError(
            f"channel {channel} signed delta {signed} does not match expected sign {expected_sign} (min_abs={min_abs})"
        )
    return signed


def scatter_policy_action(num_sim_joints: int, policy_joint_ids, policy_action) -> list[float]:
    scattered = [0.0] * int(num_sim_joints)
    for policy_i, sim_i in enumerate(policy_joint_ids):
        scattered[int(sim_i)] = float(policy_action[policy_i])
    return scattered


def processed_position_targets(default_joint_pos, scattered_action, action_scale: float) -> list[float]:
    return [
        float(default) + float(action) * float(action_scale)
        for default, action in zip(default_joint_pos, scattered_action)
    ]


def assert_command_target_vector(
    *,
    readback_target,
    default_joint_pos,
    policy_joint_ids,
    policy_names,
    sim_names,
    channel: int,
    action: float,
    action_scale: float,
    atol: float = 1.0e-6,
) -> None:
    if abs(float(action_scale) - 0.25) > 1.0e-12:
        raise AssertionError(f"action_scale {action_scale} != 0.25")
    sim_id = int(policy_joint_ids[channel])
    if list(sim_names)[sim_id] != list(policy_names)[channel]:
        raise AssertionError(f"policy {policy_names[channel]!r} maps to sim {sim_names[sim_id]!r} at id {sim_id}")
    policy_action = [0.0] * len(policy_names)
    policy_action[channel] = float(action)
    expected = processed_position_targets(
        default_joint_pos,
        scatter_policy_action(len(sim_names), policy_joint_ids, policy_action),
        action_scale,
    )
    if len(readback_target) != len(expected):
        raise AssertionError(f"target width {len(readback_target)} != {len(expected)}")
    for i, (got, exp, default) in enumerate(zip(readback_target, expected, default_joint_pos)):
        if i != sim_id and abs(float(got) - float(default)) > atol:
            raise AssertionError(f"uncommanded sim target[{i}] changed from {default} to {got}")
        if abs(float(got) - float(exp)) > atol:
            raise AssertionError(f"sim target[{i}] {got} != {exp} (default {default}, channel {channel})")


def assert_pair_initial_match(left: dict, right: dict, atol: float = 1.0e-5) -> None:
    for key in ("q", "qd", "root_rel", "root_vel"):
        a, b = left[key], right[key]
        if len(a) != len(b):
            raise AssertionError(f"pair {key} width {len(a)} != {len(b)}")
        err = max(abs(float(x) - float(y)) for x, y in zip(a, b))
        if err > atol:
            raise AssertionError(f"pair {key} mismatch max_abs={err} > {atol}")
    for key in ("masses", "stiffness", "damping"):
        if left.get(key) is None or right.get(key) is None:
            continue
        err = max(abs(float(x) - float(y)) for x, y in zip(left[key], right[key]))
        if err > atol:
            raise AssertionError(f"pair {key} mismatch max_abs={err} > {atol}")


def physx_try_call(view, method: str):
    fn = getattr(view, method, None)
    if not callable(fn):
        return {"unavailable": method}
    try:
        value = fn()
    except Exception as exc:  # runtime API surface differs across Isaac builds
        return {"unavailable": method, "error": f"{type(exc).__name__}:{exc}"}
    if hasattr(value, "detach"):
        row = value[0].detach().cpu()
        return {"source": f"root_physx_view.{method}", "value": row.tolist()}
    return {"source": f"root_physx_view.{method}", "value": value}


def restore_level0_terrain(terrain, env) -> None:
    """Undo sentinel -1 sampling before later contact/reset checks.

    Matches ``probe_locomotion_portability.py``: level 0 then
    ``terrain.env_origins = terrain.terrain_origins[levels, types]``. Also copies
    ``scene.env_origins`` / ``env.env_origins`` when those are distinct tensors.
    """
    terrain.terrain_levels.zero_()
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    scene = getattr(env, "scene", None)
    scene_origins = getattr(scene, "env_origins", None) if scene is not None else None
    if scene_origins is not None:
        scene_origins[:] = terrain.env_origins
    env_origins = getattr(env, "env_origins", None)
    if env_origins is not None:
        env_origins[:] = terrain.env_origins


def reject_negative_amp_file(*, label: str, path: Path, expect: str, loader_cls, frame_dim, joint_order, step_dt, fail):
    if not path.is_file():
        fail(f"{label} fixture missing: {path}")
    try:
        loader_cls(
            device="cpu",
            time_between_frames=float(step_dt),
            frame_dim=int(frame_dim),
            motion_files=[str(path)],
            expected_joint_order=list(joint_order),
        )
    except FileNotFoundError as exc:
        fail(f"{label} fixture missing during load: {exc}")
    except ValueError as exc:
        message = str(exc)
        if expect not in message:
            fail(f"{label} rejected for unexpected reason ({exc}); expected {expect!r}")
        return {
            "path": str(path),
            "rejected": True,
            "error": f"ValueError:{exc}",
            "expect": expect,
        }
    fail(f"{label} loaded as a Z2 expert: {path}")
    raise AssertionError("unreachable")


def reject_packaged_negative_amp(
    *, loader_cls, step_dt, fail, frame_dim=AMP_FRAME_DIM, joint_order=Z2_29DOF_JOINT_NAMES
):
    result = {}
    for label, path, expect in NEGATIVE_AMP_CASES:
        result[label] = reject_negative_amp_file(
            label=label,
            path=path,
            expect=expect,
            loader_cls=loader_cls,
            frame_dim=frame_dim,
            joint_order=joint_order,
            step_dt=step_dt,
            fail=fail,
        )
    return result

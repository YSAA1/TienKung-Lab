"""Sagittal mirror plans from robot semantics and the actual observation layout."""

from functools import lru_cache

from legged_lab.locomotion.schemas import CRITIC_EXTRA_FIELDS, ObservationLayout, field_slice, proprio_fields

_DEVICE_PLAN_CACHE = {}


def apply_plan(values, indices, signs):
    if type(values).__module__.split(".")[0] == "torch":
        import torch
        key = (indices, signs, values.device, values.dtype)
        if key not in _DEVICE_PLAN_CACHE:
            _DEVICE_PLAN_CACHE[key] = (
                torch.tensor(indices, dtype=torch.long, device=values.device),
                torch.tensor(signs, dtype=values.dtype, device=values.device),
            )
        ids, scale = _DEVICE_PLAN_CACHE[key]
        return values.index_select(-1, ids) * scale
    import numpy as np
    return values[..., np.asarray(indices)] * np.asarray(signs, dtype=values.dtype)


def _grid(start, shape):
    nx, ny = shape
    return [start + (ny - 1 - iy) * nx + ix for iy in range(ny) for ix in range(nx)]


@lru_cache(maxsize=64)
def observation_mirror_plan(layout: ObservationLayout, joint_sources, joint_signs, is_critic: bool):
    fields = proprio_fields(layout.num_joints) + (CRITIC_EXTRA_FIELDS if is_critic else ())
    width = sum(n for _, n in fields)
    ids, signs = list(range(width)), [1.0] * width
    for name, offsets in (("base_ang_vel", (0, 2)), ("projected_gravity", (1,)),
                          ("velocity_command", (1, 2))):
        start, _ = field_slice(fields, name)
        for offset in offsets:
            signs[start + offset] = -1.0
    for name in ("joint_pos", "joint_vel", "previous_action"):
        start, end = field_slice(fields, name)
        ids[start:end] = [start + i for i in joint_sources]
        signs[start:end] = joint_signs
    for name in ("gait_phase_sin", "gait_phase_cos", "gait_air_ratio") + (("feet_contact",) if is_critic else ()):
        start, _ = field_slice(fields, name)
        ids[start:start + 2] = [start + 1, start]
    if is_critic:
        start, _ = field_slice(fields, "base_lin_vel")
        signs[start + 1] = -1.0
    history = layout.critic_history if is_critic else layout.actor_history
    all_ids = [h * width + i for h in range(history) for i in ids]
    all_signs = signs * history
    for _ in range(layout.scan_history):
        all_ids.extend(_grid(len(all_ids), layout.scan_shape))
        all_signs.extend([1.0] * layout.scan_dim)
    if is_critic and layout.critic_foot_scan:
        start = len(all_ids)
        all_ids.extend(_grid(start + layout.foot_scan_dim, layout.foot_scan_shape)
                       + _grid(start, layout.foot_scan_shape))
        all_signs.extend([1.0] * (2 * layout.foot_scan_dim))
    if is_critic and layout.critic_immunity:
        all_ids.append(len(all_ids))
        all_signs.append(1.0)
    if not is_critic and layout.actor_contact:
        start = len(all_ids)
        all_ids.extend((start + 1, start))
        all_signs.extend((1.0, 1.0))
    return tuple(all_ids), tuple(all_signs)


def mirror_actions(actions, spec):
    if actions.shape[-1] != len(spec.joint_names):
        raise ValueError(f"action width {actions.shape[-1]} != {spec.name} joint count {len(spec.joint_names)}")
    return apply_plan(actions, spec.mirror_indices, spec.mirror_signs)


def mirror_observations(obs, is_critic, spec, layout):
    expected = layout.critic_dim if is_critic else layout.actor_dim
    if obs.shape[-1] != expected:
        raise ValueError(f"{'critic' if is_critic else 'actor'} width {obs.shape[-1]} != configured {expected}")
    ids, signs = observation_mirror_plan(layout, spec.mirror_indices, spec.mirror_signs, bool(is_critic))
    return apply_plan(obs, ids, signs)


def _concat(first, second):
    if type(first).__module__.split(".")[0] == "torch":
        import torch
        return torch.cat((first, second), dim=0)
    import numpy as np
    return np.concatenate((first, second), axis=0)


def get_symmetric_states(obs=None, actions=None, env=None, obs_type=None, is_critic=None):
    if env is None:
        raise ValueError("symmetry requires the environment's robot spec and observation layout")
    spec, layout = env.cfg.robot_spec, env.observation_layout
    critic = obs_type == "critic" if is_critic is None else bool(is_critic)
    mirrored_obs = mirror_observations(obs, critic, spec, layout) if obs is not None else None
    mirrored_actions = mirror_actions(actions, spec) if actions is not None else None
    if obs_type is not None:
        if obs is not None:
            mirrored_obs = _concat(obs, mirrored_obs)
        if actions is not None:
            mirrored_actions = _concat(actions, mirrored_actions)
    return mirrored_obs, mirrored_actions

"""State-safe counterfactual policy evaluation helpers."""

from __future__ import annotations

import torch


def _clone_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_clone_state(item) for item in value)
    if isinstance(value, list):
        return [_clone_state(item) for item in value]
    return value


def evaluate_counterfactual_actions(policy, inference_fn, observations_by_mode, *, selected_mode):
    """Evaluate observation interventions without corrupting a recurrent policy's live state.

    Every counterfactual starts from the same pre-step hidden state. Afterward the
    policy retains only the state transition caused by ``selected_mode``.
    """
    if selected_mode not in observations_by_mode:
        raise KeyError(f"selected mode {selected_mode!r} is unavailable")
    if not getattr(policy, "is_recurrent", False):
        return {mode: inference_fn(observations) for mode, observations in observations_by_mode.items()}

    state_before = _clone_state(policy.get_hidden_states())
    selected_action = inference_fn(observations_by_mode[selected_mode])
    state_after_selected = _clone_state(policy.get_hidden_states())
    actions = {selected_mode: selected_action}
    try:
        for mode, observations in observations_by_mode.items():
            if mode == selected_mode:
                continue
            policy.reset(hidden_states=_clone_state(state_before))
            actions[mode] = inference_fn(observations)
    finally:
        policy.reset(hidden_states=_clone_state(state_after_selected))
    return actions


def reset_recurrent_policy(policy, dones):
    """Reset inference-created recurrent state inside the matching torch mode."""
    with torch.inference_mode():
        policy.reset(dones)

"""Adapter between the current IsaacLab RL wrapper and the vendored RSL-RL.

Vendored from the PHP vault recipe (``whole_body_tracking.utils.rsl_rl_compat``).
The in-repo ``rsl_rl`` consumes the legacy ``(policy_obs, extras)`` contract
with observation groups under ``extras["observations"]``, while newer IsaacLab
wrappers return observation-group dicts directly.
"""

from __future__ import annotations

from collections.abc import Mapping

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper as IsaacLabRslRlVecEnvWrapper


def coerce_rsl_rl_observations(observations):
    """Turn Isaac Lab 2.1 dicts or 2.3 TensorDicts into ``(policy, extras)``."""
    if isinstance(observations, tuple) and len(observations) == 2:
        raw_observations, extras = observations
    else:
        raw_observations, extras = observations, {}
    extras = dict(extras)
    groups = _as_obs_groups(raw_observations)
    if groups is None:
        groups = _as_obs_groups(extras.get("observations"))
    if groups is None:
        policy = raw_observations
        groups = {"policy": policy}
    if "policy" not in groups:
        raise KeyError("Isaac Lab observation groups must contain 'policy'")
    extras["observations"] = {key: _batch_obs(value) for key, value in groups.items()}
    return extras["observations"]["policy"], extras


def _as_obs_groups(raw):
    if raw is None:
        return None
    if isinstance(raw, Mapping) or (hasattr(raw, "keys") and hasattr(raw, "__getitem__")):
        try:
            if "policy" in raw:
                return {key: raw[key] for key in raw.keys()}
        except Exception:
            return None
    return None


def _batch_obs(value):
    if hasattr(value, "dim") and value.dim() == 1:
        return value.unsqueeze(0)
    return value


class RslRlVecEnvWrapper(IsaacLabRslRlVecEnvWrapper):
    """Expose the legacy ``(policy_obs, extras)`` contract used by the vendored RSL-RL."""

    @staticmethod
    def _legacy_observations(observations):
        return coerce_rsl_rl_observations(observations)

    def reset(self):
        return self._legacy_observations(super().reset())

    def get_observations(self):
        return self._legacy_observations(super().get_observations())

    def step(self, actions):
        observations, rewards, dones, extras = super().step(actions)
        policy, extras = self._legacy_observations((observations, extras))
        return policy, rewards, dones, extras


__all__ = ["RslRlVecEnvWrapper"]

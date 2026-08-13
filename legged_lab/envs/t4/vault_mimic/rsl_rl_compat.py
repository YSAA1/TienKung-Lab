"""Adapter between the current IsaacLab RL wrapper and the vendored RSL-RL.

Vendored from the PHP vault recipe (``whole_body_tracking.utils.rsl_rl_compat``).
The in-repo ``rsl_rl`` consumes the legacy ``(policy_obs, extras)`` contract
with observation groups under ``extras["observations"]``, while newer IsaacLab
wrappers return observation-group dicts directly.
"""

from __future__ import annotations

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper as IsaacLabRslRlVecEnvWrapper


class RslRlVecEnvWrapper(IsaacLabRslRlVecEnvWrapper):
    """Expose the legacy ``(policy_obs, extras)`` contract used by the vendored RSL-RL."""

    @staticmethod
    def _legacy_observations(observations):
        if isinstance(observations, tuple) and len(observations) == 2:
            raw_observations, extras = observations
        else:
            raw_observations, extras = observations, {}
        extras = dict(extras)
        if isinstance(raw_observations, dict):
            policy = raw_observations["policy"]
            observation_groups = raw_observations
        else:
            policy = raw_observations
            observation_groups = extras.get("observations", {"policy": policy})
        if not isinstance(observation_groups, dict) or "policy" not in observation_groups:
            raise KeyError("Isaac Lab observation groups must contain 'policy'")
        extras["observations"] = observation_groups
        return policy, extras

    def reset(self):
        return self._legacy_observations(super().reset())

    def get_observations(self):
        return self._legacy_observations(super().get_observations())

    def step(self, actions):
        observations, rewards, dones, extras = super().step(actions)
        policy, extras = self._legacy_observations((observations, extras))
        return policy, rewards, dones, extras


__all__ = ["RslRlVecEnvWrapper"]

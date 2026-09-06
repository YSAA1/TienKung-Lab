"""Isaac-free tests for the G1/G2 RSL-RL observation adapter."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch

if "isaaclab_rl" not in sys.modules:
    isaaclab_rl = types.ModuleType("isaaclab_rl")
    rsl_rl_mod = types.ModuleType("isaaclab_rl.rsl_rl")
    rsl_rl_mod.RslRlVecEnvWrapper = object
    isaaclab_rl.rsl_rl = rsl_rl_mod
    sys.modules["isaaclab_rl"] = isaaclab_rl
    sys.modules["isaaclab_rl.rsl_rl"] = rsl_rl_mod

_COMPAT = Path(__file__).resolve().parents[1] / "legged_lab/utils/rsl_rl_compat.py"
_SPEC = importlib.util.spec_from_file_location("t4_vault_rsl_rl_compat", _COMPAT)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)
coerce_rsl_rl_observations = _MOD.coerce_rsl_rl_observations


class _FakeTensorDict:
    def __init__(self, data, batch_shape):
        self._data = data
        self.shape = torch.Size(batch_shape)

    def keys(self):
        return self._data.keys()

    def __contains__(self, key):
        return key in self._data

    def __getitem__(self, key):
        return self._data[key]


def test_tensordict_obs_keeps_batch_dim():
    policy = torch.zeros(1, 150)
    critic = torch.zeros(1, 276)
    td = _FakeTensorDict({"policy": policy, "critic": critic}, (1,))
    obs, extras = coerce_rsl_rl_observations(td)
    assert obs.shape == (1, 150)
    assert extras["observations"]["critic"].shape == (1, 276)


def test_reset_tuple_tensordict():
    td = _FakeTensorDict({"policy": torch.zeros(1, 150)}, (1,))
    obs, extras = coerce_rsl_rl_observations((td, {"time_outs": torch.zeros(1)}))
    assert obs.shape == (1, 150)
    assert "time_outs" in extras


def test_plain_dict_still_works():
    obs, extras = coerce_rsl_rl_observations({"policy": torch.zeros(4, 150)})
    assert obs.shape == (4, 150)
    assert extras["observations"]["policy"].shape == (4, 150)

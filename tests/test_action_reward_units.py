"""Exercise production reward functions across action coordinates and joint orders."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _reward(name):
    source = Path(__file__).resolve().parents[1] / "legged_lab/mdp/rewards.py"
    node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0), node],
                        type_ignores=[])
    scope = {"torch": torch}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), scope)
    return scope[name]


def _env(targets, scale, order):
    # targets and action are simulator-ordered; the buffer is policy-ordered.
    actions = targets / scale
    return SimpleNamespace(
        action_scale=scale, policy_joint_ids=order, action=actions[:, -1], ankle_joint_ids=[1, 3],
        action_buffer=SimpleNamespace(_circular_buffer=SimpleNamespace(buffer=actions[:, :, order])),
    )


@pytest.mark.parametrize("joints", [21, 27, 29])
def test_same_physical_targets_have_same_cost_despite_per_joint_scale_and_order(joints):
    generator = torch.Generator().manual_seed(42)
    targets = torch.randn(2, 2, joints, generator=generator) * 0.1
    order = torch.randperm(joints, generator=generator).tolist()
    old = _env(targets, 0.25, order)
    # Include the G1 ankle and waist extremes, with different scale on each env.
    scale = torch.linspace(0.0583333333, 0.4375, joints).repeat(2, 1)
    scale[1] *= 0.8
    current = _env(targets, scale[:, None, :], order)
    current.action_scale = scale
    for name in ("action_rate_l2", "ankle_action"):
        reward = _reward(name)
        expected = reward(old)
        torch.testing.assert_close(reward(current, reference_scale=0.25), expected)
        torch.testing.assert_close(reward(old, reference_scale=0.25), expected, rtol=0, atol=0)


def test_legacy_call_keeps_raw_action_penalty():
    targets = torch.tensor([[[0.1, 0.2, 0.3, 0.4], [-0.2, 0.1, 0.2, 0.3]]])
    env = _env(targets, 0.25, [3, 1, 0, 2])
    raw = env.action_buffer._circular_buffer.buffer
    torch.testing.assert_close(_reward("action_rate_l2")(env), ((raw[:, -1] - raw[:, -2]) ** 2).sum(1))
    torch.testing.assert_close(_reward("ankle_action")(env), env.action[:, [1, 3]].abs().sum(1))

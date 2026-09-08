"""Exercise the runtime call sites without booting Isaac Sim."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _action_scale():
    spec = importlib.util.spec_from_file_location("action_scale", ROOT / "legged_lab/utils/action_scale.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.effort_scaled_action_scale


def _nodes(path, first, last=None):
    source = (ROOT / path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    start = source[: source.index(first)].count("\n") + 1
    end = source[: source.index(last)].count("\n") + 1 if last else start + 1
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.stmt) and start <= node.lineno < end]
    # Keep only top-level statements of the selected fragment.
    selected = [node for node in nodes if not any(node in list(ast.walk(other))[1:] for other in nodes)]
    return compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), str(path), "exec")


@pytest.mark.parametrize("width", [66, 70])
def test_amp_runner_uses_pre_reset_terminal_state(width):
    reset_state = torch.zeros(3, width)
    terminal = torch.full((1, width), 9.0)
    env = SimpleNamespace(reset_env_ids=torch.tensor([1]), get_amp_obs_for_expert_trans=lambda: reset_state.clone())
    scope = dict(
        torch=torch,
        self=SimpleNamespace(env=env, device="cpu"),
        next_amp_obs=reset_state,
        infos={"terminal_amp_obs": terminal},
    )
    exec(
        _nodes(
            "rsl_rl/rsl_rl/runners/amp_on_policy_runner.py",
            "next_amp_obs_with_term =",
            "rewards = self.alg.discriminator.predict_amp_reward(",
        ),
        scope,
    )
    expected = reset_state.clone()
    expected[1] = terminal[0]
    assert torch.equal(scope["next_amp_obs_with_term"], expected)
    assert torch.equal(reset_state, torch.zeros_like(reset_state))


@pytest.mark.parametrize("joints, expected", [(27, 960), (29, 1020)])
def test_evaluator_scan_starts_after_actual_proprio_history(joints, expected):
    frame = 15 + 3 * joints
    env = SimpleNamespace(
        actor_obs_buffer=SimpleNamespace(buffer=torch.zeros(2, 10, frame)),
        cfg=SimpleNamespace(robot=SimpleNamespace(actor_obs_history_length=10)),
    )
    scope = dict(env=env, PROPRIO_FRAME_DIM=96, PROPRIO_HISTORY_LENGTH=10)
    exec(_nodes("legged_lab/scripts/eval_locomotion.py", "scan_start ="), scope)
    assert scope["scan_start"] == expected


def test_evaluator_diagnostic_joints_follow_policy_and_simulator_orders():
    names = ["right_knee_joint", "left_hip_pitch_joint", "right_hip_pitch_joint", "left_knee_joint"]
    env = SimpleNamespace(
        robot=SimpleNamespace(joint_names=names),
        left_leg_ids=[9, 1, 8, 3],
        right_leg_ids=[7, 2, 6, 0],
        policy_joint_names=tuple(reversed(names)),
    )
    from legged_lab.assets.t4.constants import T4_JOINT_NAMES

    scope = dict(env=env, T4_JOINT_NAMES=T4_JOINT_NAMES)
    exec(_nodes("legged_lab/scripts/eval_locomotion.py", "diagnostic_joint_names =", "joint_action_abs_sum ="), scope)
    assert list(scope["diagnostic_joint_names"]) == [names[1], names[3], names[2], names[0]]
    assert scope["diagnostic_joint_indices"] == [2, 0, 1, 3]


def test_effort_scaled_actions_preserve_fraction_under_joint_permutation():
    effort_scaled_action_scale = _action_scale()

    kp = torch.tensor([[20.0, 150.0, 200.0]])  # G1 ankle, waist, hip
    limits = torch.tensor([[35.0, 35.0, 139.0]])
    scale = effort_scaled_action_scale(kp, limits, 0.25)
    assert torch.allclose(kp * scale / limits, torch.full_like(kp, 0.25))
    order = [2, 0, 1]
    assert torch.allclose(effort_scaled_action_scale(kp[:, order], limits[:, order], 0.25), scale[:, order])
    assert scale[0, 0].item() == pytest.approx(0.4375)
    assert scale[0, 1].item() == pytest.approx(0.058333333)


@pytest.mark.parametrize("kp,limit,fraction", [(0, 35, 0.25), (20, float("inf"), 0.25), (20, 35, 0)])
def test_effort_scaled_actions_reject_invalid_actuators(kp, limit, fraction):
    effort_scaled_action_scale = _action_scale()

    with pytest.raises(ValueError):
        effort_scaled_action_scale(torch.tensor([kp]), torch.tensor([limit]), fraction)


@pytest.mark.parametrize("min_scale", [1.0, 0.5])
def test_sparse_speed_ramp_keeps_full_target_and_t4_default(min_scale):
    difficulty = torch.tensor([0.0, 0.5, 1.0])
    env = SimpleNamespace(
        cfg=SimpleNamespace(sparse_command_min_speed_scale=min_scale), terrain_difficulty=lambda: difficulty
    )
    scope = dict(self=env, vx=torch.tensor([2.0, 2.0, 2.0]), chosen=torch.arange(3))
    exec(_nodes("legged_lab/locomotion/env.py", "min_speed_scale =", "self._sparse_command[chosen, 0] ="), scope)
    assert scope["vx"].tolist() == pytest.approx([2 * min_scale, 1 + min_scale, 2])

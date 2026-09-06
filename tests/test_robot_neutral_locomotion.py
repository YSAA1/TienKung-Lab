"""Shared interfaces work for both deployed robots and a differently named biped."""

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from legged_lab.assets.t4.locomotion import T4_LOCOMOTION
from legged_lab.assets.unitree_g1.locomotion import G1_LOCOMOTION
from legged_lab.locomotion.amp_features import AmpFeatureBuilder, site_pos_in_root_frame
from legged_lab.locomotion.robot_spec import LocomotionRobotSpec
from legged_lab.locomotion.schemas import ObservationLayout
from legged_lab.locomotion.symmetry import mirror_actions, mirror_observations
from legged_lab.locomotion.validation import validate_training_contract


def _third_robot():
    joints = tuple(f"motor_{i}" for i in range(21))
    return LocomotionRobotSpec(
        name="test_biped_21", joint_names=joints, feet=("sole_a", "sole_b"), hands=("tip_a", "tip_b"),
        torso="chassis", diagnostic_bodies=("chassis",), left_leg=joints[:6], right_leg=joints[6:12],
        ankles=(joints[4], joints[10], joints[5], joints[11]), hand_site_offset=(0, 0, 0),
        foot_site_offset=(0, 0, 0), mirror_indices=tuple(range(6, 12)) + tuple(range(6)) + tuple(range(12, 21)),
        mirror_signs=(-1, 1, -1, 1, 1, -1) * 2 + (1,) * 9, nominal_feet_distance=0.18,
        reward_bodies={}, reward_joints={},
    )


def _articulation(spec):
    names = tuple(reversed(spec.joint_names))
    bodies = tuple(dict.fromkeys(spec.feet + spec.hands + spec.diagnostic_bodies + (spec.torso,)))
    q = torch.arange(len(names)).float().unsqueeze(0)
    b = len(bodies)
    data = SimpleNamespace(
        root_state_w=torch.tensor([[0., 0., 0., 1., 0., 0., 0.]]),
        joint_pos=q, joint_vel=-q,
        body_pos_w=torch.arange(3 * b).float().reshape(1, b, 3),
        body_quat_w=torch.tensor([1., 0., 0., 0.]).repeat(1, b, 1),
    )
    return SimpleNamespace(
        joint_names=names, body_names=bodies, data=data,
        find_joints=lambda name_keys, preserve_order: ([names.index(n) for n in name_keys], name_keys),
        find_bodies=lambda name_keys, preserve_order: ([bodies.index(n) for n in name_keys], name_keys),
    )


@pytest.mark.parametrize("spec,actor,critic,amp", [
    (T4_LOCOMOTION, 1937, 2016, 66), (G1_LOCOMOTION, 1997, 2076, 70), (_third_robot(), 1757, 1836, 54),
])
def test_layout_amp_and_mirror_use_robot_contract(spec, actor, critic, amp):
    spec.validate()
    layout = ObservationLayout(len(spec.joint_names), scan_history=5, actor_contact=True,
                               critic_foot_scan=True, critic_immunity=True)
    assert (layout.actor_dim, layout.critic_dim, spec.amp_frame_dim) == (actor, critic, amp)
    robot = _articulation(spec)
    state = AmpFeatureBuilder(robot, "cpu", spec).compute()
    assert state.shape == (1, amp)
    n = len(spec.joint_names)
    assert torch.equal(state[0, :n], torch.arange(n).flip(0).float())
    assert torch.equal(state[0, n:2*n], -state[0, :n])
    for is_critic, width in ((False, actor), (True, critic)):
        obs = torch.randn(2, width)
        mirrored = mirror_observations(obs, is_critic, spec, layout)
        assert torch.equal(mirror_observations(mirrored, is_critic, spec, layout), obs)
    actions = torch.randn(2, n)
    assert torch.equal(mirror_actions(mirror_actions(actions, spec), spec), actions)


@pytest.mark.parametrize("change", [
    {"joint_names": ("duplicate",) * 27}, {"mirror_indices": (0,) * 27},
    {"mirror_signs": (0.,) * 27}, {"feet": ("same", "same")}, {"left_leg": ("absent",) * 6},
])
def test_incomplete_robot_specs_fail_before_simulation(change):
    with pytest.raises(ValueError):
        replace(T4_LOCOMOTION, **change).validate()


def test_runtime_rejects_wrong_robot_and_training_contract():
    spec = G1_LOCOMOTION
    with pytest.raises(ValueError, match="joint mismatch"):
        spec.validate_articulation(_articulation(T4_LOCOMOTION))
    layout = ObservationLayout(29, scan_history=5, actor_contact=True, critic_foot_scan=True, critic_immunity=True)
    env = SimpleNamespace(cfg=SimpleNamespace(robot_spec=spec), robot=_articulation(spec), observation_layout=layout,
                          get_observations=lambda: (torch.zeros(1, layout.actor_dim),
                                                   {"observations": {"critic": torch.zeros(1, layout.critic_dim)}}))
    with pytest.raises(ValueError, match="AMP config width"):
        validate_training_contract(env, {"amp_frame_dim": 66})


def test_amp_sites_rotate_in_body_then_root_frame():
    yaw90 = torch.tensor([[2**-0.5, 0., 0., 2**-0.5]])
    result = site_pos_in_root_frame(torch.tensor([[1., 2., 3.]]), yaw90,
                                    torch.tensor([[[1., 3., 3.]]]), yaw90.unsqueeze(1),
                                    torch.tensor([1., 0., 0.]))
    assert torch.allclose(result, torch.tensor([[2., 0., 0.]]), atol=1e-6)


def test_algorithm_modules_do_not_import_robot_implementations():
    root = Path(__file__).resolve().parents[1]
    shared_files = [
        *(root / "legged_lab/locomotion").rglob("*.py"),
        *(root / "legged_lab/motion_tracking").rglob("*.py"),
        root / "legged_lab/config.py",
        root / "legged_lab/utils/rsl_rl_compat.py",
    ]
    for path in shared_files:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                statement = ast.unparse(node)
                assert not any(name in statement for name in ("legged_lab.assets", "legged_lab.envs")), path

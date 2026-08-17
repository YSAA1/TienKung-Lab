"""Layer-0 contracts for the T4 1 m vault corridor evaluator.

These tests run without Isaac Sim or a GPU. They freeze the corridor + ordered
gate layout and the zip-ported strict success semantics (sustained crossing +
stable landing) that G1/G2/G3 gates consume. The IsaacLab eval scene must
instantiate the same wall / box AABBs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from legged_lab.assets.t4 import vault_contract
from legged_lab.assets.t4.tracking_motion import body_indices, load_t4_tracking_motion

ROOT = Path(__file__).resolve().parents[1]
_EVAL_PATH = ROOT / "legged_lab" / "envs" / "t4" / "vault_eval.py"
_spec = importlib.util.spec_from_file_location("t4_vault_eval", _EVAL_PATH)
vault_eval = importlib.util.module_from_spec(_spec)
sys.modules["t4_vault_eval"] = vault_eval
_spec.loader.exec_module(vault_eval)


def _traj(*, xs, ys=None, zs=None, upright=None, lin=None, ang=None):
    n = len(xs)
    if ys is None:
        ys = [vault_contract.T4_VAULT_BOX_POS[1]] * n
    if zs is None:
        zs = [0.85] * n
    if upright is None:
        upright = [1.0] * n
    if lin is None:
        lin = [0.1] * n
    if ang is None:
        ang = [0.1] * n
    return xs, ys, zs, upright, lin, ang


def _judge(xs, ys, zs, upright, lin, ang, **kwargs):
    layout = vault_eval.canonical_vault_corridor_layout()
    return vault_eval.evaluate_vault_trajectory(
        root_x=xs,
        root_y=ys,
        root_z=zs,
        root_upright=upright,
        root_lin_speed=lin,
        root_ang_speed=ang,
        layout=layout,
        **kwargs,
    )


def _success_path(n=80):
    """Walk from in front of the box, over the far face, and settle."""
    layout = vault_eval.canonical_vault_corridor_layout()
    y = 0.5 * (layout.corridor_y_min + layout.corridor_y_max)
    xs, ys, zs, up, lin, ang = [], [], [], [], [], []
    for i in range(n):
        if i < 20:
            x = layout.approach_gate_x - 0.25 + i * 0.03
            z = 0.86
        elif i < 45:
            x = layout.box_front_x + (i - 20) * 0.05
            z = 1.20
        else:
            x = layout.box_back_x + 0.05 + (i - 45) * 0.03
            z = 0.85
        xs.append(x)
        ys.append(y)
        zs.append(z)
        up.append(0.99)
        lin.append(0.15 if i >= 50 else 0.35)
        ang.append(0.2)
    return xs, ys, zs, up, lin, ang


def test_layout_matches_fixed_1m_box_and_blocks_side_bypass():
    layout = vault_eval.canonical_vault_corridor_layout()
    hx, hy, hz = [s / 2.0 for s in vault_contract.T4_VAULT_BOX_SIZE]
    cx, cy, cz = vault_contract.T4_VAULT_BOX_POS
    assert layout.box_size == tuple(vault_contract.T4_VAULT_BOX_SIZE)
    assert layout.box_pos == tuple(vault_contract.T4_VAULT_BOX_POS)
    assert abs(layout.box_front_x - (cx - hx)) < 1e-9
    assert abs(layout.box_back_x - (cx + hx)) < 1e-9
    assert abs(layout.box_back_x - 0.88) < 1e-9
    # Corridor y equals the box y-extent so a 1 m box fills the lane.
    assert abs(layout.corridor_y_min - (cy - hy)) < 1e-9
    assert abs(layout.corridor_y_max - (cy + hy)) < 1e-9
    assert layout.approach_gate_x < layout.box_front_x < layout.box_back_x < layout.exit_gate_x
    # Two side walls, no overlap with the box interior.
    assert len(layout.wall_aabbs) == 2
    box_min, box_max = layout.box_aabb
    for wmin, wmax in layout.wall_aabbs:
        overlap_y = not (wmax[1] <= box_min[1] or wmin[1] >= box_max[1])
        overlap_x = not (wmax[0] <= box_min[0] or wmin[0] >= box_max[0])
        assert not (overlap_y and overlap_x)


def test_reference_motion_is_in_front_of_approach_and_past_exit():
    layout = vault_eval.canonical_vault_corridor_layout()
    motion = load_t4_tracking_motion(vault_contract.T4_VAULT_MOTION_FILE)
    trunk = body_indices(motion, ("Trunk",))[0]
    pos = motion["body_pos_w"][:, trunk]
    assert pos[0, 0] < layout.approach_gate_x
    assert pos[-1, 0] > layout.exit_gate_x
    assert layout.corridor_y_min < pos[:, 1].min() <= pos[:, 1].max() < layout.corridor_y_max


def test_reference_motion_is_a_strict_success():
    motion = load_t4_tracking_motion(vault_contract.T4_VAULT_MOTION_FILE)
    trunk = body_indices(motion, ("Trunk",))[0]
    pos = motion["body_pos_w"][:, trunk]
    quat = motion["body_quat_w"][:, trunk]
    lin = np.linalg.norm(motion["body_lin_vel_w"][:, trunk], axis=1)
    ang = np.linalg.norm(motion["body_ang_vel_w"][:, trunk], axis=1)
    w, x, y, z = quat.T
    upright = 1.0 - 2.0 * (x * x + y * y)
    result = _judge(pos[:, 0], pos[:, 1], pos[:, 2], upright, lin, ang)
    assert result.success, result.reasons
    assert result.crossing_start_step is not None
    assert result.landing_start_step is not None
    assert result.reasons == ()


def test_zero_policy_still_path_is_a_classified_failure():
    layout = vault_eval.canonical_vault_corridor_layout()
    start_x = layout.approach_gate_x - 0.25
    xs, ys, zs, up, lin, ang = _traj(xs=[start_x] * 40)
    result = _judge(xs, ys, zs, up, lin, ang)
    assert result.success is False
    assert "crossing_not_sustained" in result.reasons
    assert "no_stable_landing" in result.reasons


def test_side_bypass_fails_left_corridor_even_if_x_crosses():
    layout = vault_eval.canonical_vault_corridor_layout()
    xs = np.linspace(layout.approach_gate_x - 0.2, layout.exit_gate_x + 0.2, 70)
    ys = np.linspace(0.2, layout.corridor_y_max + 0.4, 70)  # walk around +y
    zs = np.full(70, 0.85)
    result = _judge(xs, ys, zs, [0.99] * 70, [0.2] * 70, [0.2] * 70)
    assert result.success is False
    assert "left_corridor" in result.reasons


def test_skip_gate_when_approach_crossed_outside_y_window():
    layout = vault_eval.canonical_vault_corridor_layout()
    n = 70
    xs = np.linspace(layout.approach_gate_x - 0.2, layout.exit_gate_x + 0.2, n)
    ys = []
    for x in xs:
        if abs(x - layout.approach_gate_x) < 0.08:
            ys.append(layout.corridor_y_max + 0.25)
        else:
            ys.append(0.5 * (layout.corridor_y_min + layout.corridor_y_max))
    result = _judge(xs, ys, [0.85] * n, [0.99] * n, [0.15] * n, [0.15] * n)
    assert result.success is False
    assert "skipped_gate" in result.reasons or "left_corridor" in result.reasons


def test_reversed_gates_fail():
    layout = vault_eval.canonical_vault_corridor_layout()
    y = 0.5 * (layout.corridor_y_min + layout.corridor_y_max)
    # Walk forward past the exit, then backward through the approach.
    xs = list(np.linspace(layout.approach_gate_x - 0.2, layout.exit_gate_x + 0.15, 40))
    xs += list(np.linspace(layout.exit_gate_x + 0.15, layout.approach_gate_x - 0.2, 40))
    n = len(xs)
    result = _judge(xs, [y] * n, [0.85] * n, [0.99] * n, [0.15] * n, [0.15] * n)
    assert result.success is False
    assert "reversed_gate" in result.reasons or "no_stable_landing" in result.reasons


def test_start_already_crossed_is_a_failure():
    layout = vault_eval.canonical_vault_corridor_layout()
    y = 0.5 * (layout.corridor_y_min + layout.corridor_y_max)
    xs = [layout.box_back_x + 0.1] * 40
    result = _judge(xs, [y] * 40, [0.85] * 40, [0.99] * 40, [0.1] * 40, [0.1] * 40)
    assert result.success is False
    assert "start_already_crossed" in result.reasons


def test_hard_violation_is_a_failure_even_on_a_good_path():
    result = _judge(*_success_path(), hard_violation=True, termination_terms=("anchor_pos",))
    assert result.success is False
    assert "hard_violation" in result.reasons
    assert result.hard_violation is True


def test_happy_path_success_and_batch_zero_policy_rate_is_zero():
    happy = _judge(*_success_path())
    assert happy.success, happy.reasons
    layout = vault_eval.canonical_vault_corridor_layout()
    start_x = layout.approach_gate_x - 0.25
    still = np.full((30, 4), start_x)
    y = np.full((30, 4), 0.2)
    z = np.full((30, 4), 0.85)
    ones = np.ones((30, 4))
    batch = vault_eval.evaluate_vault_batch(
        still,
        y,
        z,
        ones,
        np.full((30, 4), 0.05),
        np.full((30, 4), 0.05),
        layout=layout,
    )
    assert batch.total == 4
    assert batch.successes == 0
    assert batch.success_rate == 0.0
    assert all(not ep.success for ep in batch.episodes)


def test_report_has_required_lineage_and_failure_fields():
    layout = vault_eval.canonical_vault_corridor_layout()
    still = np.full((12, 2), layout.approach_gate_x - 0.3)
    batch = vault_eval.evaluate_vault_batch(
        still,
        np.full((12, 2), 0.2),
        np.full((12, 2), 0.85),
        np.ones((12, 2)),
        np.full((12, 2), 0.05),
        np.full((12, 2), 0.05),
        layout=layout,
    )
    report = vault_eval.build_evaluation_report(
        batch,
        checkpoint="zero-policy",
        motion=str(vault_contract.T4_VAULT_MOTION_FILE),
        task="t4_vault_mimic_eval",
        seed=42,
        git_head="deadbeef",
        git_dirty=False,
        git_dirty_diff_sha256="0" * 64,
        bucket="g1_fixed_1m",
        policy="zero",
    )
    required = {
        "evaluator",
        "lineage",
        "seed",
        "checkpoint",
        "bucket",
        "policy",
        "success_definition",
        "summary",
        "episodes",
        "layout",
    }
    assert required.issubset(report)
    assert report["lineage"]["git_head"] == "deadbeef"
    assert report["lineage"]["motion_sha256"]
    assert report["summary"]["success_rate"] == 0.0
    assert report["summary"]["failures"] == 2
    assert report["summary"]["forbidden_contact_count"] == 0
    assert report["summary"]["hard_limit_count"] == 0
    assert "crossing_not_sustained" in report["summary"]["failure_reasons"]
    assert report["episodes"][0]["success"] is False
    assert report["episodes"][0]["reasons"]


def test_eval_env_cfg_consumes_the_layout():
    source = (ROOT / "legged_lab/envs/t4/vault_mimic/vault_env_cfg.py").read_text()
    init_source = (ROOT / "legged_lab/envs/t4/vault_mimic/__init__.py").read_text()
    script = (ROOT / "legged_lab/scripts/eval_t4_vault.py").read_text()
    assert "T4VaultMimicEvalEnvCfg" in source
    assert "canonical_vault_corridor_layout" in source
    assert "t4_vault_mimic_eval" in init_source
    assert "from legged_lab.envs.t4 import vault_eval" in script or "vault_eval" in script
    assert "--policy" in script
    assert "zero" in script


def _range_is_zero(bounds) -> bool:
    return bounds == (0.0, 0.0)


def test_play_and_eval_reset_use_zero_motion_jitter():
    """Play/eval start at frame 0 with no RSI pose/joint/vel jitter.

    Training keeps the non-zero ranges. Play/eval must not inherit them.
    """
    assert vault_contract.T4_VAULT_PLAY_SAMPLING_STRATEGY == "zero"
    assert vault_contract.T4_VAULT_PLAY_JOINT_POSITION_RANGE == (0.0, 0.0)
    for axis, bounds in vault_contract.T4_VAULT_PLAY_POSE_RANGE.items():
        assert _range_is_zero(bounds), axis
    for axis, bounds in vault_contract.T4_VAULT_PLAY_VELOCITY_RANGE.items():
        assert _range_is_zero(bounds), axis

    source = (ROOT / "legged_lab/envs/t4/vault_mimic/vault_env_cfg.py").read_text()
    assert "T4_VAULT_PLAY_SAMPLING_STRATEGY" in source
    assert "T4_VAULT_PLAY_POSE_RANGE" in source
    assert "T4_VAULT_PLAY_VELOCITY_RANGE" in source
    assert "T4_VAULT_PLAY_JOINT_POSITION_RANGE" in source
    play_init = source.split("class T4VaultMimicPlayEnvCfg")[1].split("class ")[0]
    assert "T4_VAULT_PLAY_SAMPLING_STRATEGY" in play_init
    assert "T4_VAULT_PLAY_POSE_RANGE" in play_init
    assert "T4_VAULT_PLAY_VELOCITY_RANGE" in play_init
    assert "T4_VAULT_PLAY_JOINT_POSITION_RANGE" in play_init


def test_eval_script_resets_each_batch_and_flushes_kinematics():
    """Each trial batch must start from a fresh RSI frame-0 state.

    The first m28500 100-trial JSON was 48/100 because the evaluator reused
    leftover poses across batches (16/16, 0/16, 16/16, ...). Step-0 wrist
    deaths were the cold-start batch before FK flush, not a policy failure.
    """
    script = (ROOT / "legged_lab/scripts/eval_t4_vault.py").read_text()
    assert "def _independent_reset(" in script
    loop = script.split("while completed < requested:")[1].split("result = VaultBatchResult")[0]
    assert "_independent_reset(env)" in loop
    helper = script.split("def _independent_reset(")[1].split("def evaluate")[0]
    assert helper.count("env.reset()") >= 2
    assert "env.step(" in helper
    assert "write_data_to_sim" in helper
    assert "sim.forward()" in helper

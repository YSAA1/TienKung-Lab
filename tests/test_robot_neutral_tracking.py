"""Tracking references are bound by names rather than a particular robot order."""

import numpy as np
import pytest

from legged_lab.motion_tracking.loader import body_indices, load_tracking_motion
from legged_lab.motion_tracking.schema import reorder_named_axis


def _payload(joints):
    count = len(joints)
    return {
        "fps": np.array([30.0]),
        "joint_names": np.array(joints),
        "body_names": np.array(["anchor", "sole_a", "sole_b"]),
        "joint_pos": np.arange(4 * count).reshape(4, count).astype(float),
        "joint_vel": -np.arange(4 * count).reshape(4, count).astype(float),
        "body_pos_w": np.zeros((4, 3, 3)),
        "body_quat_w": np.tile([1.0, 0.0, 0.0, 0.0], (4, 3, 1)),
        "body_lin_vel_w": np.zeros((4, 3, 3)),
        "body_ang_vel_w": np.zeros((4, 3, 3)),
    }


@pytest.mark.parametrize("count", [21, 27, 29])
def test_tracking_loader_reorders_arbitrary_robot_names(tmp_path, count):
    names = tuple(f"servo_{index}" for index in range(count))
    payload = _payload(names[::-1])
    path = tmp_path / "motion.npz"
    np.savez(path, **payload)
    motion = load_tracking_motion(path, joint_names=names)
    assert motion["joint_names"] == names
    assert np.array_equal(motion["joint_pos"], payload["joint_pos"][:, ::-1])
    assert np.array_equal(motion["joint_vel"], payload["joint_vel"][:, ::-1])
    assert body_indices(motion, ("sole_b", "anchor")) == (2, 0)


@pytest.mark.parametrize("corruption", ["missing_joint", "duplicate_body", "nonfinite", "shape", "fps"])
def test_tracking_reference_contract_fails_before_simulation(tmp_path, corruption):
    names = ("motor_a", "motor_b", "motor_c")
    payload = _payload(names)
    if corruption == "missing_joint":
        payload["joint_names"] = np.array(["motor_a", "motor_b", "different_motor"])
    elif corruption == "duplicate_body":
        payload["body_names"] = np.array(["anchor", "sole_a", "sole_a"])
    elif corruption == "nonfinite":
        payload["joint_pos"][0, 0] = np.nan
    elif corruption == "shape":
        payload["body_pos_w"] = payload["body_pos_w"][:, :2]
    else:
        payload["fps"] = np.array([0.0])
    path = tmp_path / "invalid.npz"
    np.savez(path, **payload)
    with pytest.raises(ValueError):
        load_tracking_motion(path, joint_names=names)


def test_runtime_axis_reorder_rejects_missing_names_and_duplicate_targets():
    data = np.arange(12).reshape(2, 3, 2)
    assert np.array_equal(
        reorder_named_axis(data, source_names=("a", "b", "c"), target_names=("c", "a"), axis=1, label="body"),
        data[:, [2, 0]],
    )
    for target in (("a", "absent"), ("a", "a")):
        with pytest.raises(ValueError):
            reorder_named_axis(data, source_names=("a", "b", "c"), target_names=target, axis=1, label="body")

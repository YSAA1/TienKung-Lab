import importlib

import mujoco
import pytest
import numpy as np
import pytest


def test_sparse_course_uses_training_layout_and_supports_both_teacher_contracts():
    module = importlib.import_module("legged_lab.scripts.play_t4_sparse_teacher_mujoco")

    layout = module.sparse_course_layout(0.5)
    assert layout["stone_width"] == pytest.approx(0.4)
    assert layout["stone_gap"] == pytest.approx(0.33)
    assert layout["stone_height"] == pytest.approx(0.23)
    assert layout["pillar_diameter"] == pytest.approx(0.44)
    assert layout["pillar_gap"] == pytest.approx(0.125)
    assert layout["pillar_height"] == pytest.approx(0.23)
    assert {geom["kind"] for geom in layout["geoms"]} >= {"stone", "pillar", "platform"}

    start_platform = next(geom for geom in layout["geoms"] if geom["name"] == "start_platform")
    assert 2.0 * start_platform["size"][1] == module._LAYOUT.T4_STONE_PLATFORM_WIDTH

    for kind in ("stone", "pillar"):
        footholds = [geom for geom in layout["geoms"] if geom["kind"] == kind]
        lateral_min = min(geom["pos"][1] - geom["size"][0] for geom in footholds)
        lateral_max = max(geom["pos"][1] + geom["size"][0] for geom in footholds)
        assert lateral_min <= -1.5
        assert lateral_max >= 1.5

    xml = module.build_sparse_model_xml(0.5)
    model = mujoco.MjModel.from_xml_string(xml)
    assert model.geom("sparse_pit_floor").pos[2] == -2.05
    assert model.geom("stone_0_0").size[0] == pytest.approx(0.2)
    assert model.geom("pillar_0_0").size[0] == pytest.approx(0.22)

    start_front = start_platform["pos"][0] + start_platform["size"][0]
    first_stone = next(geom for geom in layout["geoms"] if geom["name"] == "stone_0_3")
    first_inner = first_stone["pos"][0] - first_stone["size"][0]
    assert first_inner - start_front == pytest.approx(module._LAYOUT.platform_to_first_gap(0.5, layout["stone_width"]))

    assert module.supported_actor_obs_dim(module.TEACHER_ACTOR_OBS_DIM)
    assert module.supported_actor_obs_dim(module.TEACHER_PAPER_ACTOR_OBS_DIM)
    assert not module.supported_actor_obs_dim(module.TEACHER_PAPER_ACTOR_OBS_DIM + 1)


def test_teacher_scan_matches_isaac_xy_flatten_order():
    module = importlib.import_module("legged_lab.scripts.play_t4_sparse_teacher_mujoco")

    points = module.teacher_scan_local_points()
    num_x, num_y = module.TEACHER_SCAN_SHAPE
    assert len(points) == num_x * num_y
    assert points[0] == pytest.approx(
        (module.TEACHER_SCAN_FORWARD_RANGE[0], module.TEACHER_SCAN_LATERAL_RANGE[0])
    )
    assert points[num_x - 1] == pytest.approx(
        (module.TEACHER_SCAN_FORWARD_RANGE[1], module.TEACHER_SCAN_LATERAL_RANGE[0])
    )
    assert points[num_x] == pytest.approx(
        (module.TEACHER_SCAN_FORWARD_RANGE[0], module.TEACHER_SCAN_LATERAL_RANGE[0] + 0.1)
    )


def test_checkpoint_24999_geometry_is_pinned_to_its_training_lineage():
    module = importlib.import_module("legged_lab.scripts.play_t4_sparse_teacher_mujoco")

    layout = module.sparse_course_layout(0.5, terrain="raised_pillars", geometry="checkpoint_24999")
    assert layout["pillar_diameter"] == pytest.approx(0.36)
    assert layout["pillar_gap"] == pytest.approx(0.13)
    assert layout["pillar_height"] == pytest.approx(0.36)

    start = next(geom for geom in layout["geoms"] if geom["name"] == "start_platform")
    first = next(geom for geom in layout["geoms"] if geom["name"] == "pillar_0_3")
    start_front = start["pos"][0] + start["size"][0]
    first_left = first["pos"][0] - first["size"][0]
    assert first_left == pytest.approx(start_front)


def test_formal_teacher_loco_uses_isaac_scan_order():
    module = importlib.import_module("legged_lab.scripts.play_t4_teacher_viser")

    points = module.teacher_scan_local_points()
    num_x, num_y = module.TEACHER_SCAN_SHAPE
    assert len(points) == num_x * num_y
    assert points[0] == pytest.approx(
        (module.TEACHER_SCAN_FORWARD_RANGE[0], module.TEACHER_SCAN_LATERAL_RANGE[0])
    )
    assert points[num_x - 1] == pytest.approx(
        (module.TEACHER_SCAN_FORWARD_RANGE[1], module.TEACHER_SCAN_LATERAL_RANGE[0])
    )


def test_teacher_scan_points_match_isaac_xy_flatten_order():
    module = importlib.import_module("legged_lab.scripts.play_t4_sparse_teacher_mujoco")

    points = module.teacher_scan_local_points()
    assert len(points) == module.TEACHER_SCAN_DIM
    np.testing.assert_allclose(points[0], (0.2, -0.6))
    np.testing.assert_allclose(points[1], (0.3, -0.6))
    np.testing.assert_allclose(points[module.TEACHER_SCAN_SHAPE[0] - 1], (1.6, -0.6))
    np.testing.assert_allclose(points[module.TEACHER_SCAN_SHAPE[0]], (0.2, -0.5))

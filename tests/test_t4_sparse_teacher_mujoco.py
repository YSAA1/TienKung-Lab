import importlib

import mujoco
import pytest
import numpy as np
import pytest


def test_sparse_course_uses_training_layout_and_supports_both_teacher_contracts():
    module = importlib.import_module("legged_lab.scripts.play_t4_sparse_teacher_mujoco")

    layout = module.sparse_course_layout(0.5)
    assert layout["stone_width"] == pytest.approx(module._LAYOUT.stone_width(0.5))
    assert layout["stone_gap"] == pytest.approx(module._LAYOUT.stone_gap(0.5))
    assert layout["stone_height"] == pytest.approx(module._LAYOUT.stone_height(0.5))
    assert layout["pillar_diameter"] == pytest.approx(module._LAYOUT.pillar_diameter(0.5))
    assert layout["pillar_gap"] == pytest.approx(module._LAYOUT.pillar_gap(0.5))
    assert layout["pillar_height"] == pytest.approx(module._LAYOUT.pillar_height(0.5))
    assert {geom["kind"] for geom in layout["geoms"]} >= {"stone", "pillar", "platform"}

    start_platform = next(geom for geom in layout["geoms"] if geom["name"] == "start_platform")
    assert 2.0 * start_platform["size"][1] == module._LAYOUT.T4_STONE_PLATFORM_WIDTH

    for kind in ("stone", "pillar"):
        footholds = [geom for geom in layout["geoms"] if geom["kind"] == kind]
        lateral_min = min(geom["pos"][1] - geom["size"][0] for geom in footholds)
        lateral_max = max(geom["pos"][1] + geom["size"][0] for geom in footholds)
        assert lateral_min <= -1.0
        assert lateral_max >= 1.0

    xml = module.build_sparse_model_xml(0.5)
    model = mujoco.MjModel.from_xml_string(xml)
    assert model.geom("sparse_pit_floor").pos[2] == -2.05
    assert model.geom("stone_0_0").size[0] == pytest.approx(0.5 * layout["stone_width"])
    assert model.geom("pillar_0_0").size[0] == pytest.approx(0.5 * layout["pillar_diameter"])

    start_front = start_platform["pos"][0] + start_platform["size"][0]
    first_stone = next(geom for geom in layout["geoms"] if geom["name"] == "stone_0_3")
    first_inner = first_stone["pos"][0] - first_stone["size"][0]
    assert first_inner - start_front == pytest.approx(module._LAYOUT.platform_to_first_gap(0.5, layout["stone_width"]))

    assert module.supported_actor_obs_dim(module.TEACHER_ACTOR_OBS_DIM)
    assert module.supported_actor_obs_dim(module.TEACHER_PAPER_ACTOR_OBS_DIM)
    assert module.supported_actor_obs_dim(module.TEACHER_SPARSE_ACTOR_OBS_DIM)
    assert not module.supported_actor_obs_dim(module.TEACHER_SPARSE_ACTOR_OBS_DIM + 1)


def test_sparse_course_can_expand_the_local_diagnostic_scene_without_changing_defaults():
    module = importlib.import_module("legged_lab.scripts.play_t4_sparse_teacher_mujoco")

    default_layout = module.sparse_course_layout(0.39, terrain="stepping_stones")
    expanded_layout = module.sparse_course_layout(
        0.39,
        terrain="stepping_stones",
        lane_count=15,
        foothold_rows=30,
    )
    default_stones = [geom for geom in default_layout["geoms"] if geom["kind"] == "stone"]
    stones = [geom for geom in expanded_layout["geoms"] if geom["kind"] == "stone"]
    finish = next(geom for geom in expanded_layout["geoms"] if geom["name"] == "finish_platform")

    assert len(default_stones) == module.LANE_COUNT * module.FOOTHOLD_ROWS
    assert len(stones) == 15 * 30
    assert expanded_layout["lane_count"] == 15
    assert expanded_layout["foothold_rows"] == 30
    assert {geom["pos"][1] for geom in stones if geom["name"] == "stone_0_7"} == {0.0}
    assert finish["pos"][0] - finish["size"][0] > max(geom["pos"][0] + geom["size"][0] for geom in stones)


def test_loco_sparse_scene_reuses_the_loco_lane_and_current_footholds():
    module = importlib.import_module("legged_lab.scripts.play_t4_sparse_teacher_mujoco")

    layout = module.loco_sparse_layout(0.39, lane_count=15, foothold_rows=15)
    model = mujoco.MjModel.from_xml_string(
        module.build_sparse_model_xml(0.39, terrain="loco_sparse", lane_count=15, foothold_rows=15)
    )

    assert layout["terrain"] == "loco_sparse"
    assert layout["sparse_start_x"] == pytest.approx(module.LOCO_SPARSE_START_X)
    assert len([geom for geom in layout["geoms"] if geom["kind"] == "stone"]) == 15 * 15
    assert len([geom for geom in layout["geoms"] if geom["kind"] == "pillar"]) == 15 * 15
    assert model.geom("loco_hurdle_1_geom").id >= 0
    assert model.geom("loco_stair_up_1_geom").id >= 0
    alignment = model.geom("loco_sparse_alignment_platform")
    assert alignment.id >= 0
    assert alignment.size[1] == pytest.approx(1.1)
    assert model.geom("loco_sparse_stone_0_7").id >= 0
    assert model.geom("loco_sparse_pillar_0_7").id >= 0
    assert model.geom("ground").type[0] == mujoco.mjtGeom.mjGEOM_BOX
    assert model.geom("loco_sparse_pit_floor").pos[2] == pytest.approx(-2.05)


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

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from legged_lab.assets.t4 import schemas
from legged_lab.scripts.sim2sim_t4_depth_student import (
    COMMAND_RANGES,
    D455_ROS_ROT_WXYZ,
    DEPTH_CAM_POS,
    DEPTH_CAM_XYAXES,
    HEADING_STIFFNESS,
    LOCO_GOAL_XY,
    LOCO_STAIR2_RISE,
    LOCO_STAIR2_STEPS,
    LOCO_STAIR_RISE,
    LOCO_STAIR_STEPS,
    CourseNavigator,
    build_model_xml,
    camera_look_axes,
    classify_sensor_depth,
    course_waypoints_from_model,
    heading_velocity_command,
    polyline_lookahead,
    render_standing_sensor_depth,
    resolve_control_mode,
    resolve_course,
    ros_offset_to_mujoco_xyaxes,
    wrap_to_pi,
)


def test_rule_contract_course_contains_expected_obstacles():
    xml_path = Path(build_model_xml(False, True))
    xml = xml_path.read_text()

    assert xml.count('<body name="obstacle_2_hurdle_') == 10
    assert "obstacle_1_slope" in xml
    assert "obstacle_3_cross_slope" in xml
    assert "obstacle_4_pole_1" in xml
    assert "obstacle_5_up_slope" in xml
    assert "obstacle_6_stair_0" in xml
    assert "obstacle_7_bridge_1" in xml
    assert "obstacle_8_platform" in xml
    assert "obstacle_9_tunnel" in xml
    assert "obstacle_10_l_turn_x" in xml


def test_schema_camera_converts_to_pitched_forward_mujoco():
    xyaxes = ros_offset_to_mujoco_xyaxes(D455_ROS_ROT_WXYZ)
    np.testing.assert_allclose(xyaxes, DEPTH_CAM_XYAXES, atol=1e-3)
    axes = camera_look_axes(xyaxes)
    pitch = math.radians(schemas.DEPTH_CAMERA_PITCH_DEG)
    np.testing.assert_allclose(axes["look"], [math.cos(pitch), 0.0, -math.sin(pitch)], atol=1e-3)
    np.testing.assert_allclose(axes["right"], [0.0, -1.0, 0.0], atol=1e-3)


def test_training_camera_matches_the_schema_head_site():
    assert tuple(DEPTH_CAM_POS) == tuple(schemas.DEPTH_CAMERA_SITE_POS)
    xml = Path(build_model_xml(False, False)).read_text()
    assert f'pos="{" ".join(map(str, DEPTH_CAM_POS))}"' in xml
    look = camera_look_axes(DEPTH_CAM_XYAXES)["look"]
    pitch = math.degrees(math.atan2(-look[2], look[0]))
    assert pitch == pytest.approx(schemas.DEPTH_CAMERA_PITCH_DEG, abs=0.5)


def test_policy_depth_shows_forward_ground_not_a_rolled_vertical_band():
    depth = render_standing_sensor_depth()
    stats = classify_sensor_depth(depth)
    assert stats["orientation"] != "vertical_band"
    assert stats["hit_fraction"] > 0.55
    assert stats["median_hit_depth"] < 3.0


def test_heading_command_matches_training_stiffness():
    yaw_err = math.atan2(0.4, 4.0)
    cmd = heading_velocity_command(np.array([0.0, 0.0]), 0.0, np.array([4.0, 0.4]), 0.55)
    assert cmd[1] == 0.0
    assert cmd[2] == pytest.approx(HEADING_STIFFNESS * yaw_err)
    assert cmd[0] > 0.0
    stopped = heading_velocity_command(np.array([4.0, 0.0]), 0.0, np.array([4.05, 0.0]), 0.55)
    np.testing.assert_allclose(stopped, 0.0)


def test_lookahead_corrects_harder_than_a_distant_goal():
    root = np.array([5.0, -0.60])
    far = heading_velocity_command(root, 0.0, np.array([28.0, 0.0]), 0.55)
    carrot = polyline_lookahead(root, np.array([[0.0, 0.0], [28.0, 0.0]]), lookahead=1.4)
    near = heading_velocity_command(root, 0.0, carrot, 0.55)
    assert carrot[1] == pytest.approx(0.0)
    assert 5.5 < carrot[0] < 7.0
    assert abs(near[2]) > abs(far[2])
    assert near[2] > 0.0


def test_navigator_steers_back_to_the_centerline():
    nav = CourseNavigator(np.array([[0.0, 0.0], [4.0, 0.0], [8.0, 0.0]]), cruise_vx=0.55)
    cmd = nav.command(np.array([0.0, 0.4]), yaw=0.0)
    assert cmd[1] == 0.0
    assert cmd[2] < 0.0
    assert cmd[0] >= 0.0
    assert COMMAND_RANGES["yaw"][0] <= cmd[2] <= COMMAND_RANGES["yaw"][1]


def test_navigator_advances_when_the_robot_reaches_a_waypoint():
    nav = CourseNavigator(np.array([[1.0, 0.0], [4.0, 0.0]]), cruise_vx=0.55, reach=0.6)
    nav.command(np.array([1.05, 0.0]), yaw=0.0)
    assert nav.index == 1
    np.testing.assert_allclose(nav.current_waypoint, [4.0, 0.0])


def test_rule_contract_waypoints_include_weave_and_l_turn():
    import mujoco

    xml_path = build_model_xml(False, True)
    model = mujoco.MjModel.from_xml_path(xml_path)
    points = course_waypoints_from_model(model)
    assert points[0, 0] < 1.0
    assert points[-1, 0] >= 90.0
    assert np.any(points[:, 1] > 0.2)
    assert np.any(points[:, 1] < -0.2)


def test_loco_course_has_training_scale_obstacles_and_a_goal():
    xml = Path(build_model_xml(course="loco")).read_text()
    assert xml.count('<body name="loco_hurdle_') == 2
    assert xml.count('<body name="loco_rough_') >= 8
    assert xml.count('<body name="loco_boxes_') >= 4
    assert xml.count('<body name="loco_wave_') >= 8
    assert xml.count('<body name="loco_ramp_up_') >= 8
    assert xml.count('<body name="loco_ramp_down_') >= 8
    assert xml.count('<body name="loco_stair_up_') == LOCO_STAIR_STEPS
    assert xml.count('<body name="loco_stair_down_') == LOCO_STAIR_STEPS
    assert xml.count('<body name="loco_stair2_up_') == LOCO_STAIR2_STEPS
    assert xml.count('<body name="loco_stair2_down_') == LOCO_STAIR2_STEPS
    assert "loco_slope" not in xml
    assert 'name="wall_left"' in xml
    assert 'name="wall_right"' in xml
    assert 'name="goal"' in xml
    assert LOCO_STAIR_RISE >= 0.16
    assert LOCO_STAIR2_RISE >= 0.18
    import mujoco

    model = mujoco.MjModel.from_xml_path(build_model_xml(course="loco"))
    points = course_waypoints_from_model(model)
    np.testing.assert_allclose(points[-1], LOCO_GOAL_XY, atol=1e-6)
    assert points[0, 0] <= 0.05
    assert len(points) >= 8
    assert np.max(np.abs(points[:, 1])) < 1e-6


def test_wrap_to_pi_and_default_control_mode():
    assert wrap_to_pi(math.pi + 0.1) == pytest.approx(-math.pi + 0.1)
    args = SimpleNamespace(control=None, rule_contract=False, hurdles=False, course=None)
    assert resolve_control_mode(args) == "nav"
    assert resolve_course(args) == "loco"
    args.course = "flat"
    assert resolve_course(args) == "flat"
    args.control = "auto"
    assert resolve_control_mode(args) == "auto"

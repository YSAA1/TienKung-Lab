from pathlib import Path

from legged_lab.scripts.sim2sim_t4_depth_student import build_model_xml


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

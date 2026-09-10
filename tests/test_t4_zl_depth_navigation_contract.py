import ast
from pathlib import Path
import subprocess

import numpy as np
import pytest
import yaml
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]

# These contracts verify the zhuoqun-machine ``zl_deploy`` tree and the AME
# overlay tar; they only run on hosts that carry that deployment.
pytest.skip(
    "zl_deploy deployment tree not present on this host (zhuoqun contract tests)",
    allow_module_level=True,
) if not (ROOT / "zl_deploy").is_dir() else None


def test_zl_depth_course_nav_uses_shared_heading_command_contract():
    source = (ROOT / "zl_deploy/scripts/run_t4_depth_course_nav.py").read_text()
    assert "from legged_lab.assets.t4.navigation import CourseNavigator" in source


def test_zl_depth_course_nav_rate_limits_intents_to_policy_frequency():
    source = (ROOT / "zl_deploy/scripts/run_t4_depth_course_nav.py").read_text()
    assert "COMMAND_PUBLISH_PERIOD_S = 0.02" in source
    assert source.count("next_publish = now + COMMAND_PUBLISH_PERIOD_S") >= 2
    assert 'self.send_intent("status", "walk", velocity_command=command)' in source
    assert "msg.angular_z_radps = float(velocity_command[2])" in source
    assert "node.enter_walk(initial_command, timeout=5.0)" in source
    assert (
        'COMMAND_SOURCE = os.environ.get("T4_DEPTH_NAV_COMMAND_SOURCE", "nav")'
        in source
    )
    assert 'if COMMAND_SOURCE == "nav":' in source
    assert (
        'parser.add_argument("--ready-stable-seconds", type=float, default=2.0)'
        in source
    )
    assert (
        "node.spin_until_ready(args.ready_timeout, args.ready_stable_seconds)" in source
    )
    assert "self.publish_command(np.zeros(3" not in source
    assert "is_stable_platform_sample(" in source
    assert '"control_pipeline_complete": control_pipeline_complete' in source
    assert "and control_pipeline_complete" in source
    assert 'return 0 if summary["success"] else 2' in source
    assert '"feedback_steps_histogram": feedback_steps_histogram' in source

    bootstrap = (ROOT / "zl_deploy/scripts/bootstrap_t4_depth_sim.py").read_text()
    assert "QoSReliabilityPolicy.BEST_EFFORT" in bootstrap
    assert '"/robot/control/hardware_states"' in bootstrap
    assert "is_settled_ready_sample(" in bootstrap


def test_zl_depth_course_acceptance_rejects_fallen_or_low_arrivals():
    from zl_deploy.scripts.t4_depth_course_acceptance import (
        is_settled_ready_sample,
        is_stable_platform_sample,
    )

    assert is_stable_platform_sample(
        distance_m=0.1, height_m=1.91, roll_deg=2.0, pitch_deg=-3.0
    )
    assert not is_stable_platform_sample(
        distance_m=0.1, height_m=0.2, roll_deg=179.0, pitch_deg=-80.0
    )
    assert not is_stable_platform_sample(
        distance_m=0.1, height_m=1.75, roll_deg=2.0, pitch_deg=-3.0
    )
    assert not is_stable_platform_sample(
        distance_m=0.1, height_m=1.91, roll_deg=10.0, pitch_deg=0.0
    )
    assert not is_stable_platform_sample(
        distance_m=0.41, height_m=1.91, roll_deg=0.0, pitch_deg=0.0
    )
    assert is_settled_ready_sample(
        height_m=0.845, roll_deg=0.5, pitch_deg=-0.5, max_joint_speed=0.05
    )
    assert not is_settled_ready_sample(
        height_m=0.845, roll_deg=0.5, pitch_deg=-0.5, max_joint_speed=0.10
    )
    assert not is_settled_ready_sample(
        height_m=0.80, roll_deg=0.0, pitch_deg=0.0, max_joint_speed=0.0
    )
    assert not is_settled_ready_sample(
        height_m=0.845, roll_deg=2.0, pitch_deg=0.0, max_joint_speed=0.0
    )


def test_zl_depth_launcher_exposes_opt_in_course_navigation():
    source = (ROOT / "zl_deploy/scripts/run_t4_depth_sim2sim.sh").read_text()
    assert "T4_DEPTH_NAV_GOAL_X" in source
    assert "run_t4_depth_course_nav.py" in source
    assert "T4_DEPTH_NAV_VX:-0.8" in source
    assert "T4_DEPTH_NAV_MAX_YAW_RATE:-0.20" in source
    assert "T4_DEPTH_NAV_READY_STABLE_SECONDS:-2.0" in source
    assert "T4_DEPTH_MODEL_NAME:-t4_tienkung_depth_sim" in source
    assert "prepare_t4_depth_sim_xml.py" in source
    assert "prepare_t4_depth_sim_binary.py" in source
    assert 'export ZL_SIM_EXECUTABLE="${parity_sim_binary}"' in source
    assert "T4_DEPTH_SIM_DRIVER:-deterministic" in source
    assert "enable_vendor_mujoco_driver:=" in source
    assert "run_t4_depth_deterministic_driver.py" in source
    assert "T4_DEPTH_DETERMINISTIC_DRIVER=1" in source
    assert "T4_DEPTH_SIM_PLANT:-original" in source
    assert "prepare_t4_depth_sim_scene.py" in source
    assert 'enable_keyboard:="${T4_DEPTH_KEYBOARD:-false}"' in source
    assert "T4_DEPTH_SHOW_DEPTH" in source
    assert "view_t4_depth_image.py" in source
    assert "run_t4_linux_joy.py" in source
    assert "T4_DEPTH_JOY_DEVICE" in source
    assert "prepare_t4_ame_full_course_scene.py" in source
    assert "ame_implicit_xml" in source
    assert 'prepare_t4_depth_sim_xml.py"' in source
    assert 'T4_DEPTH_SIM_SCENE:-flat' in source
    assert 'T4_AME_FULL_COURSE_XML' in source
    assert 'sim_scene}" = "ame_full_course"' in source
    assert "--plant original" in source
    assert (
        'inference_feedback_decimation:="$(if [ "${sim_driver}" = "deterministic" ]; then echo 1; else echo 4; fi)"'
        in source
    )
    assert "mktemp -d" in source
    assert "T4_DEPTH_LOG_DIR" in source
    assert 'wait "${nav_pid}"' in source
    assert 'wait "${bootstrap_pid}"' in source
    assert "shutdown_bringup" in source
    assert "shutdown_runtime_processes" in source
    assert "trap cleanup EXIT INT TERM" in source
    assert 'implicit_xml="/tmp/' not in source
    assert 'parity_sim_binary="/tmp/' not in source
    assert source.index("source /opt/ros/humble/setup.bash") < source.index(
        "for command in"
    )
    assert source.index("source /opt/ros/humble/setup.bash") < source.index("set -u")


def test_depth_viewer_has_ros_depth_subscription_and_encoding_support():
    source = (ROOT / "zl_deploy/scripts/view_t4_depth_image.py").read_text()
    assert '"/robot/camera/depth/image_rect_raw"' in source
    assert '"32FC1"' in source
    assert '"16UC1"' in source
    assert "cv2.applyColorMap" in source


def test_linux_joy_reader_publishes_raw_js_axes_without_sdl_remapping():
    source = (ROOT / "zl_deploy/scripts/run_t4_linux_joy.py").read_text()
    assert '"/joy"' in source
    assert '"/dev/input/js0"' in source
    assert "value / 32767.0" in source
    assert "struct.unpack(\"IhBB\"" in source
    assert "self.raw_axes[3]" in source
    assert "self.raw_axes[6]" in source
    assert "-self.raw_axes[0]" in source
    assert "-self.raw_axes[1]" in source
    assert "-self.raw_axes[3]" in source


def test_ame_full_course_scene_merges_obstacles_without_touching_robot(tmp_path):
    source = Path("/home/ssy/桌面/t4_sim/t4_ame_overlay_20260811.tar")
    assert source.exists()
    import tarfile

    with tarfile.open(source) as archive:
        ame_member = next(
            member
            for member in archive.getmembers()
            if member.name.endswith("scenes/full_course/obstacle_scene.xml")
        )
        ame_path = tmp_path / "obstacle_scene.xml"
        with archive.extractfile(ame_member) as stream:
            ame_path.write_bytes(stream.read())

    vendor = ROOT / "zl_deploy/install/mj_sim/share/mj_sim/description/T4_std_add_head/xml/t4_std_add_head.xml"
    output = tmp_path / "t4-ame-full-course.xml"
    subprocess.run(
        [
            "/usr/bin/python3",
            str(ROOT / "zl_deploy/scripts/prepare_t4_ame_full_course_scene.py"),
            "--vendor-xml",
            str(vendor),
            "--ame-xml",
            str(ame_path),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )
    root = ET.parse(output).getroot()
    assert root.get("model") == "T4"
    assert len(root.find("actuator")) == 29
    assert any(camera.get("name") == "depth_cam" for camera in root.iter("camera"))
    assert len(root.find("asset")) == 34 + 108
    bodies = {body.get("name") for body in root.iter("body")}
    assert "obstacle100_full_course_00_ramp_series_frame" in bodies
    assert "obstacle100_full_course_09_l_turn_frame" in bodies
    assert root.find("worldbody/geom[@name='ground']") is not None
    assert any(camera.get("name") == "third_person" for camera in root.iter("camera"))
    assert root.find("worldbody/body[@name='Trunk']/camera[@name='third_person']").get("mode") == "fixed"


def test_ame_full_course_scene_preserves_implicit_drive_damping(tmp_path):
    vendor = ROOT / "zl_deploy/install/mj_sim/share/mj_sim/description/T4_std_add_head/xml/t4_std_add_head.xml"
    ame = tmp_path / "ame.xml"
    import tarfile

    with tarfile.open("/home/ssy/桌面/t4_sim/t4_ame_overlay_20260811.tar") as archive:
        member = next(x for x in archive.getmembers() if x.name.endswith("scenes/full_course/obstacle_scene.xml"))
        with archive.extractfile(member) as stream:
            ame.write_bytes(stream.read())
    merged = tmp_path / "merged.xml"
    implicit = tmp_path / "implicit.xml"
    subprocess.run(
        ["/usr/bin/python3", str(ROOT / "zl_deploy/scripts/prepare_t4_ame_full_course_scene.py"),
         "--vendor-xml", str(vendor), "--ame-xml", str(ame), "--output", str(merged)],
        cwd=ROOT, check=True,
    )
    subprocess.run(
        ["/usr/bin/python3", str(ROOT / "zl_deploy/scripts/prepare_t4_depth_sim_xml.py"),
         "--input", str(merged), "--output", str(implicit)],
        cwd=ROOT, check=True,
    )
    root = ET.parse(implicit).getroot()
    damping = {joint.get("name"): joint.get("damping") for joint in root.iter("joint")}
    assert damping["J_hip_l_pitch"] == "4.05"
    assert damping["J_ankle_l_roll"] == "1.05"


def test_original_scene_builder_runs_without_policy_runtime(tmp_path):
    output = tmp_path / "t4-original-stairs.xml"
    subprocess.run(
        [
            "/usr/bin/python3",
            str(ROOT / "zl_deploy/scripts/prepare_t4_depth_sim_scene.py"),
            "--plant",
            "original",
            "--scene",
            "stairs",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )
    xml_root = ET.parse(output).getroot()
    joints = {joint.get("name") for joint in xml_root.iter("joint")}
    actuators = list(xml_root.find("actuator"))
    bodies = {body.get("name") for body in xml_root.iter("body")}
    cameras = {camera.get("name") for camera in xml_root.iter("camera")}
    assert "J_head_yaw" not in joints
    assert "J_head_pitch" not in joints
    assert len(actuators) == 27
    assert {f"stair_probe_up_{index}" for index in range(6)}.issubset(bodies)
    assert {"depth_cam", "view_cam"}.issubset(cameras)


def test_combined_scene_has_bidirectional_stairs_ramps_and_hurdles(tmp_path):
    output = tmp_path / "t4-combined.xml"
    subprocess.run(
        [
            "/usr/bin/python3",
            str(ROOT / "zl_deploy/scripts/prepare_t4_depth_sim_scene.py"),
            "--plant",
            "zl",
            "--scene",
            "combined",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )
    xml = output.read_text()
    root = ET.parse(output).getroot()
    bodies = {body.get("name") for body in root.iter("body")}
    assert sum(name.startswith("combined_stair_up_") for name in bodies) == 6
    assert sum(name.startswith("combined_stair_down_") for name in bodies) == 6
    assert sum(name.startswith("combined_ramp_up_") for name in bodies) == 16
    assert sum(name.startswith("combined_ramp_down_") for name in bodies) == 16
    assert sum(name.startswith("combined_hurdle_") for name in bodies) == 3
    assert "combined_landing_up" in bodies

    manager = (
        ROOT
        / "zl_deploy/install/robot_rl_manager/lib/python3.10/site-packages/robot_rl_manager/rl_manager_node.py"
    ).read_text()
    assert 'declare_parameter("inference_feedback_decimation", 0)' in manager
    assert "feedback_trace_id = int(msg.last_command_trace_id)" in manager
    assert "self._feedback_sync_steps < self._inference_feedback_decimation" in manager
    assert "self._feedback_inference_due = True" in manager
    assert (
        "self._inference_feedback_decimation > 0 and not self._feedback_inference_due"
        in manager
    )
    assert "self._handle_command_publish_timer()" in manager

    bringup = (
        ROOT
        / "zl_deploy/install/robot_bringup/share/robot_bringup/launch/zl_robot.launch.py"
    ).read_text()
    assert (
        'DeclareLaunchArgument("inference_feedback_decimation", default_value="0")'
        in bringup
    )
    assert (
        'DeclareLaunchArgument("enable_vendor_mujoco_driver", default_value="true")'
        in bringup
    )
    assert '"inference_feedback_decimation": ParameterValue(' in bringup


def test_zl_depth_parity_batch_requires_every_cold_start_to_pass():
    source = (ROOT / "zl_deploy/scripts/run_t4_depth_parity_batch.sh").read_text()
    assert "T4_DEPTH_BATCH_RUNS:-3" in source
    assert 'bash "${SCRIPT_DIR}/run_t4_depth_sim2sim.sh"' in source
    assert 'row["launcher_status"] == 0' in source
    assert 'row["summary"].get("success") is True' in source
    assert 'row["summary"].get("control_pipeline_complete") is True' in source
    assert 'row["driver_trace_contract"].get("all_exactly_four") is True' in source
    assert '"all_exactly_four": bool(physics_steps)' in source
    assert '"all_passed": all_passed' in source

    sim_launch = (
        ROOT
        / "zl_deploy/install/sim_t4_std_add_head/share/sim_t4_std_add_head/launch/sim_t4_std_add_head_launch.py"
    ).read_text()
    assert "sim_executable = os.environ.get('ZL_SIM_EXECUTABLE')" in sim_launch
    assert "package=None if sim_executable else 'sim_t4_std_add_head'" in sim_launch


def test_zl_formal_mujoco_model_uses_training_timestep():
    xml = (
        ROOT
        / "zl_deploy/install/mj_sim/share/mj_sim/description/T4_std_add_head/xml/t4_std_add_head.xml"
    ).read_text()
    assert '<option timestep="0.005"/>' in xml


def test_zl_sim_profile_moves_kd_into_mujoco_implicit_damping(tmp_path):
    sim_config = yaml.safe_load(
        (
            ROOT
            / "zl_deploy/install/robot_rl_manager/share/robot_rl_manager/config/t4_tienkung_depth_sim.yaml"
        ).read_text()
    )
    assert sim_config["model"]["name"] == "t4_tienkung_depth_sim"
    assert all(
        joint["kd"] == 0.0 for joint in sim_config["joint_profiles"]["run"]["joints"]
    )
    sim_joints = {
        joint["joint_name"]: joint
        for joint in sim_config["joint_profiles"]["run"]["joints"]
    }
    assert sim_joints["left_ankle_pitch"]["kp"] == 20.0
    assert sim_joints["right_ankle_pitch"]["kp"] == 20.0

    real_config = yaml.safe_load(
        (
            ROOT
            / "zl_deploy/install/robot_rl_manager/share/robot_rl_manager/config/t4_tienkung_depth.yaml"
        ).read_text()
    )
    real_joints = {
        joint["joint_name"]: joint
        for joint in real_config["joint_profiles"]["run"]["joints"]
    }
    assert real_joints["left_ankle_pitch"]["kp"] == 80.0
    assert real_joints["left_ankle_pitch"]["kd"] == 4.0
    assert real_joints["right_ankle_pitch"]["kp"] == 80.0
    assert real_joints["right_ankle_pitch"]["kd"] == 4.0

    from zl_deploy.scripts.prepare_t4_depth_sim_xml import JOINT_DAMPING, prepare

    source = ROOT / "artifacts/eval/zl_deploy/t4_std_add_head_stairs.xml"
    output = tmp_path / "implicit.xml"
    prepare(source, output)
    xml_root = ET.parse(output).getroot()
    damping = {
        joint.get("name"): float(joint.get("damping"))
        for joint in xml_root.iter("joint")
        if joint.get("name")
    }
    assert damping["J_knee_l_pitch"] == 4.05
    assert damping["J_ankle_r_roll"] == 1.05
    assert set(JOINT_DAMPING).issubset(damping)


def test_zl_sim_parity_binary_neutralizes_vendor_speed_envelope(tmp_path):
    from zl_deploy.scripts.prepare_t4_depth_sim_binary import (
        EXPECTED_MAX_SPEEDS,
        PARITY_MAX_SPEEDS,
        _packed,
        prepare,
    )

    source = (
        ROOT
        / "zl_deploy/install/sim_t4_std_add_head/lib/sim_t4_std_add_head/sim_t4_std_add_head"
    )
    output = tmp_path / "sim_t4_std_add_head_tienkung_depth"
    prepare(source, output)
    source_bytes = source.read_bytes()
    output_bytes = output.read_bytes()
    assert source_bytes.count(_packed(EXPECTED_MAX_SPEEDS)) == 1
    assert output_bytes.count(_packed(EXPECTED_MAX_SPEEDS)) == 0
    assert output_bytes.count(_packed(PARITY_MAX_SPEEDS)) == 1
    assert output.stat().st_mode == source.stat().st_mode


def test_zl_deterministic_driver_holds_every_run_trace_for_exactly_four_steps():
    source = (
        ROOT / "zl_deploy/scripts/run_t4_depth_deterministic_driver.py"
    ).read_text()
    assert "RUN_DECIMATION = 4" in source
    assert "if trace_id in self._seen_run_traces:" in source
    assert "self._robot_state != RobotState.STATE_RL_ACTIVE" in source
    assert '"/robot/state"' in source
    assert "if self._robot_state != RobotState.STATE_RL_ACTIVE:" in source
    assert "self._hold_standing_pose()" in source
    assert "self.renderer.render()[:, ::-1]" in source
    assert "for step_index in range(RUN_DECIMATION):" in source
    assert "node.step_physics(" in source
    assert "run_command," in source
    assert "publish_feedback=policy_boundary" in source
    assert "physics_steps={RUN_DECIMATION}" in source
    assert "if node.run_handshake_active:" in source
    assert "pause both physics and observations" in source
    assert "One policy action owns one boundary feedback" in source
    assert "publish_boundary_snapshot" not in source
    assert '"/robot/control/hardware_commands"' in source
    assert '"/robot/driver/joint_states"' in source
    assert '"/robot/imu/nav_all"' in source
    assert '"/robot/camera/depth/image_rect_raw"' in source
    assert "def _configure_original_position_servos(self) -> None:" in source
    assert "self.model.dof_damping[self.dof_adr[name]] += kd" in source
    assert "self.data.ctrl[self.actuator_ids[name]] = position" in source
    assert "The pre-active RUN trace is the reference standing command" in source
    assert "wait for the first policy-produced" in source
    assert "self._latest_non_run = None" in source
    assert "gyro = np.asarray(self.data.qvel[3:6]" in source
    assert 'mjOBJ_SENSOR, "base_ang_vel"' not in source
    assert "DEPTH_HEIGHT = 270" in source
    assert "DEPTH_WIDTH = 480" in source
    assert "DEPTH_TRANSPORT_HEIGHT = 48" in source
    assert "DEPTH_TRANSPORT_WIDTH = 64" in source
    assert "_adaptive_area_resize(" in source
    assert "CourseNavigator" in source
    assert "node.publish_navigation_command()" in source


def test_zl_native_depth_transport_matches_adaptive_area_contract():
    source = (
        ROOT / "zl_deploy/scripts/run_t4_depth_deterministic_driver.py"
    ).read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_adaptive_area_resize"
    )
    namespace = {"np": np}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "<depth_resize>", "exec"),
        namespace,
    )
    resize = namespace["_adaptive_area_resize"]

    image = np.arange(35, dtype=np.float32).reshape(5, 7)
    actual = resize(image, 3, 4)
    expected = np.empty((3, 4), dtype=np.float32)
    for row in range(3):
        row_start = int(np.floor(row * 5 / 3))
        row_end = int(np.ceil((row + 1) * 5 / 3))
        for column in range(4):
            column_start = int(np.floor(column * 7 / 4))
            column_end = int(np.ceil((column + 1) * 7 / 4))
            expected[row, column] = image[
                row_start:row_end, column_start:column_end
            ].mean()
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0.0)


def test_zl_depth_runtime_rejects_missing_or_stale_depth():
    adapter = (
        ROOT
        / "zl_deploy/install/robot_rl_manager/lib/python3.10/site-packages/robot_rl_manager/policy/adapters/t4_tienkung_depth.py"
    ).read_text()
    manager = (
        ROOT
        / "zl_deploy/install/robot_rl_manager/lib/python3.10/site-packages/robot_rl_manager/rl_manager_node.py"
    ).read_text()
    assert 'return {"joint_states", "imu", "depth_image"}' in adapter
    assert 'declare_parameter("depth_image_timeout_sec", 0.25)' in manager
    assert "self._last_depth_time_sec" in manager
    assert "depth_image=self._fresh_depth_image()" in manager
    assert "if (h, w) == (DEPTH_POLICY_H, DEPTH_POLICY_W):" in adapter


def test_zl_depth_policy_owns_the_first_rl_active_frame():
    adapter = (
        ROOT
        / "zl_deploy/install/robot_rl_manager/lib/python3.10/site-packages/robot_rl_manager/policy/adapters/t4_tienkung_depth.py"
    ).read_text()
    base_adapter = (
        ROOT
        / "zl_deploy/install/robot_rl_manager/lib/python3.10/site-packages/robot_rl_manager/policy/adapter.py"
    ).read_text()
    manager = (
        ROOT
        / "zl_deploy/install/robot_rl_manager/lib/python3.10/site-packages/robot_rl_manager/rl_manager_node.py"
    ).read_text()
    assert "def requires_policy_seed_on_activation(self) -> bool:" in base_adapter
    assert "self._seed_observation_pending = True" in adapter
    assert "phase = [0.0, 0.0]" in adapter
    assert "_update_smoothed_command" not in adapter
    assert (
        "def requires_policy_seed_on_activation(self) -> bool:\n        return True"
        in adapter
    )
    assert "def _prepare_activation_seed(self) -> None:" in manager
    assert (
        "self._activation_seed_commands = self._policy_runtime.infer_commands("
        in manager
    )
    assert "elif self._activation_seed_commands is not None:" in manager
    assert "self._set_latest_commands(seed_commands)" in manager
    assert "self._activation_seed_trace_id = int(seed_commands.trace_id)" in manager
    assert "self._activation_seed_publish_count < 4" in manager
    assert "self._activation_seed_publish_count += 1" in manager
    assert (
        "not self._policy_runtime.adapter.requires_policy_seed_on_activation()"
        in manager
    )
    assert "self._smoother = OutputSmoother()" in manager


def test_zl_depth_deployment_has_machine_verifier():
    source = (ROOT / "zl_deploy/scripts/verify_t4_depth_deployment.py").read_text()
    assert "max_abs_diff" in source
    assert "checkpoint_sha256" in source
    assert "onnx_sha256" in source
    assert "required_inputs" in source
    assert "/opt/ros/humble/local/lib/python3.10/dist-packages" in source
    assert "zl_deploy/install/robot_rl_manager/lib/python3.10/site-packages" in source
    assert (
        "zl_deploy/install/robot_interfaces/local/lib/python3.10/dist-packages"
        in source
    )
    assert "direct_stairs_nav_vx080_seed_action.json" in source
    for run in range(1, 4):
        assert f"live_zl_original_policy_depth_vx080_final_run{run}.json" in source
    assert "t4_tienkung_depth_sim.yaml" in source
    assert 'registry["models"].get("t4_tienkung_depth_sim")' in source
    assert "T4_DEPTH_SIM_PLANT:-original" in source
    assert "wait for the first policy-produced" in source
    assert 'summary.get("success") is not True' in source


def test_zl_sim_knee_limits_do_not_clip_observed_policy_targets():
    sim_config = yaml.safe_load(
        (ROOT / "zl_deploy/config/humanoid/robot_joints_sim.yaml").read_text()
    )
    real_config = yaml.safe_load(
        (ROOT / "zl_deploy/config/humanoid/robot_joints.yaml").read_text()
    )

    def limits_by_name(config):
        return {joint["name"]: joint for joint in config["robot_joints"]["joints"]}

    sim_limits = limits_by_name(sim_config)
    real_limits = limits_by_name(real_config)
    for knee in ("left_knee", "right_knee"):
        assert sim_limits[knee]["max_pos"] >= 3.755
        assert real_limits[knee]["max_pos"] == 3.14

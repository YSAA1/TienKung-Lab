"""MuJoCo sim2sim evaluation for the T4 depth student policy (Stage S distillation).

Replicates the Stage S deployment contract on the migrated T4 MJCF:

- proprio: 96-dim frame x 10-step history, exactly ``schemas.PROPRIO_FIELDS``
- depth: 3-frame history of 48x64 planar depth, D455-style torso camera
  (87 deg HFOV, training CameraCfg offset, 270x480 -> 48x64 area resize,
  clip (0.2, 3.0) m, no-hit pixels fill the raw invalid value 1.0)
- MuJoCo position servo with the same gains and effort limits the IsaacLab
  asset uses (``legged_lab/assets/t4/t4.py``)
- gait clock: ``gait_time += step_dt / cycle`` only while moving, phase
  offsets 0.38 / 0.88, air ratio 0.38 / 0.38

The policy is the deployable ``DepthStudentTeacher`` / GRU student.
Teacher MLP, scan-decoder, and critic weights are stripped at load even
if the file is a full training checkpoint.

Usage (from the repo root so the vendored packages resolve):

    python -m legged_lab.scripts.sim2sim_t4_depth_student                 # auto-pick latest local checkpoint
    python -m legged_lab.scripts.sim2sim_t4_depth_student \
        --checkpoint path/to/model_1000.pt                                 # interactive, depth preview included
    python -m legged_lab.scripts.sim2sim_t4_depth_student \
        --checkpoint path/to/model_1000.pt --record out.mp4 --duration 20
    python -m legged_lab.scripts.sim2sim_t4_depth_student
        # default: training-scale loco course + goal nav, no keyboard needed
    python -m legged_lab.scripts.sim2sim_t4_depth_student --course flat
    python -m legged_lab.scripts.sim2sim_t4_depth_student --course rule
    python -m legged_lab.scripts.sim2sim_t4_depth_student --course stepping_stones --difficulty 0
    python -m legged_lab.scripts.sim2sim_t4_depth_student --course raised_pillars --difficulty 0
    python -m legged_lab.scripts.sim2sim_t4_depth_student --course sparse --difficulty 0

Sparse courses are the Isaac 8 m tile (1.6 m center pad, lattice, 0.75 m rim)
over a pit, not a 7-lane pier. Spawn is the pad center, same as training.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
import tempfile
from pathlib import Path

try:
    import cv2
except ModuleNotFoundError:  # MuJoCo-only course builders do not need previews.
    cv2 = None
import mujoco
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
# The vendored rsl_rl lives at <repo>/rsl_rl/rsl_rl; add it so the module
# resolves without installing the fork.
sys.path.insert(0, str(ROOT / "rsl_rl"))

from rsl_rl.modules.depth_student_teacher import (  # noqa: E402
    build_depth_student_policy,
    strip_deployable_state_dict,
)

from legged_lab.assets.t4.constants import T4_JOINT_NAMES  # noqa: E402
from legged_lab.assets.t4.navigation import (  # noqa: E402
    COMMAND_RANGES,
    HEADING_STIFFNESS,
    NAV_LOOKAHEAD,
    CourseNavigator,
    heading_velocity_command,
    polyline_lookahead,
    wrap_to_pi,
)
from legged_lab.assets.t4.schemas import (  # noqa: E402
    DEPTH_CAMERA_PITCH_DEG,
    DEPTH_CAMERA_SITE_POS,
    DEPTH_CLIP_RANGE,
    DEPTH_HISTORY_LENGTH,
    DEPTH_INVALID_VALUE,
    DEPTH_POLICY_SIZE,
    DEPTH_UPDATE_DECIMATION,
    PROPRIO_FRAME_DIM,
    PROPRIO_HISTORY_LENGTH,
    STUDENT_ACTOR_OBS_DIM,
    STUDENT_DEPTH_HISTORY_LENGTH,
    STUDENT_PROPRIO_HISTORY_LENGTH,
    TEACHER_ACTOR_OBS_DIM,
    TEACHER_SPARSE_ACTOR_OBS_DIM,
    depth_camera_mujoco_xyaxes,
    depth_camera_ros_quat_wxyz,
    sparse_teacher_latest_scan_range,
)

_DEPTH_NOISE_PATH = ROOT / "legged_lab" / "envs" / "t4" / "mdp" / "depth_noise.py"
_depth_noise_spec = importlib.util.spec_from_file_location("t4_sim2sim_depth_noise", _DEPTH_NOISE_PATH)
_depth_noise = importlib.util.module_from_spec(_depth_noise_spec)
assert _depth_noise_spec.loader is not None
_depth_noise_spec.loader.exec_module(_depth_noise)
LIGHTLP_DEPTH_SCALE_JITTER = _depth_noise.LIGHTLP_DEPTH_SCALE_JITTER
apply_metric_depth_noise = _depth_noise.apply_metric_depth_noise

MJCF = ROOT / "legged_lab/assets/t4/mjcf/t4_std.xml"
DEFAULT_CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "t4_depth_student_latest.pt"
DEFAULT_CHECKPOINT_ROOTS = (
    ROOT / "artifacts" / "checkpoints",
    ROOT / "logs" / "t4_loco_depth_student",
)

NUM_JOINTS = len(T4_JOINT_NAMES)

# IsaacLab T4 articulation contract (legged_lab/assets/t4/t4.py).
STANDING_POS = {
    "J_arm_l_01": 0.20, "J_arm_l_02": 0.13, "J_arm_l_04": -0.43,
    "J_arm_r_01": 0.20, "J_arm_r_02": -0.13, "J_arm_r_04": -0.43,
    "J_hip_l_pitch": -0.20, "J_knee_l_pitch": 0.42, "J_ankle_l_pitch": -0.24,
    "J_hip_r_pitch": -0.20, "J_knee_r_pitch": 0.42, "J_ankle_r_pitch": -0.24,
}
STANDING_POS = np.asarray([STANDING_POS.get(name, 0.0) for name in T4_JOINT_NAMES])

# PD gains per joint group (stiffness, damping); effort limits come from the
# MJCF actuatorfrcrange values, which mirror the USD drive limits.
_PD = {
    "arm_1_5": (20.0, 1.0), "arm_6_7": (10.0, 1.0),
    "waist": (50.0, 2.0),
    "hip_pitch": (100.0, 4.0), "hip_roll_yaw": (50.0, 2.0),
    "knee": (100.0, 4.0),
    "ankle_pitch": (80.0, 4.0), "ankle_roll": (20.0, 1.0),
}


def pd_group(name: str) -> tuple[float, float]:
    if name.startswith("J_arm_"):
        index = int(name.split("_")[-1])
        return _PD["arm_1_5"] if index <= 5 else _PD["arm_6_7"]
    if name == "J_waist_yaw":
        return _PD["waist"]
    if name.startswith("J_hip_"):
        return _PD["hip_pitch"] if name.endswith("pitch") else _PD["hip_roll_yaw"]
    if name.startswith("J_knee_"):
        return _PD["knee"]
    if name.startswith("J_ankle_"):
        return _PD["ankle_pitch"] if name.endswith("pitch") else _PD["ankle_roll"]
    raise KeyError(name)


def quat_rotate_inverse_wxyz(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate a world vector into the floating-base frame."""
    w, x, y, z = quat
    quat_vec = np.array([x, y, z])
    return (
        vec * (2.0 * w * w - 1.0)
        - np.cross(quat_vec, vec) * w * 2.0
        + quat_vec * np.dot(quat_vec, vec) * 2.0
    )


def _checkpoint_iteration(path: Path) -> int:
    match = re.search(r"model_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def _latest_checkpoint_in_dir(root: Path) -> Path | None:
    candidates = list(root.glob("model_*.pt"))
    if not candidates:
        candidates = [path for path in root.rglob("model_*.pt") if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (_checkpoint_iteration(path), path.stat().st_mtime, path.as_posix()))


def _latest_checkpoint_under_roots(roots: tuple[Path, ...]) -> Path | None:
    run_dirs: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file() and root.name.startswith("model_") and root.suffix == ".pt":
            run_dirs.append(root.parent)
            continue
        if any(root.glob("model_*.pt")):
            run_dirs.append(root)
            continue
        run_dirs.extend(path.parent for path in root.rglob("model_*.pt") if path.is_file())
    if not run_dirs:
        return None
    latest_run = max(run_dirs, key=lambda path: (path.name, path.as_posix()))
    return _latest_checkpoint_in_dir(latest_run)


def resolve_checkpoint(checkpoint: Path | None) -> Path:
    """Resolve a checkpoint path or the latest local candidate."""
    if checkpoint is not None:
        if checkpoint.is_dir():
            resolved = _latest_checkpoint_in_dir(checkpoint)
            if resolved is None:
                raise FileNotFoundError(f"no model_*.pt checkpoint found under directory: {checkpoint}")
            return resolved
        if checkpoint.is_file():
            return checkpoint
        if checkpoint.name.lower() not in {"latest", "auto"}:
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    if DEFAULT_CHECKPOINT.is_file():
        return DEFAULT_CHECKPOINT

    resolved = _latest_checkpoint_under_roots(DEFAULT_CHECKPOINT_ROOTS)
    if resolved is not None:
        return resolved

    raise FileNotFoundError(
        "no T4 depth student checkpoint found; pass --checkpoint explicitly or place one under "
        f"{DEFAULT_CHECKPOINT.parent} or logs/t4_loco_depth_student"
    )


# Stage S training constants (env.yaml + teacher_cfg.py).
SIM_DT = 0.005
DECIMATION = 4
STEP_DT = SIM_DT * DECIMATION
ACTION_SCALE = 0.25
CLIP_ACTIONS = 100.0
CLIP_OBS = 100.0
LOCO_GOAL_XY = (38.0, 0.0)
LOCO_STAIR_RISE = 0.18
LOCO_STAIR_STEPS = 6
LOCO_STAIR2_RISE = 0.20
LOCO_STAIR2_STEPS = 5
LOCO_LANE_HALF = 1.20
LOCO_OBSTACLE_WIDTH = 2.20
STAIR_PROBE_START_X = 1.8
STAIR_PROBE_TREAD = 0.30
STAIR_PROBE_WIDTH = 3.0
STAIR_PROBE_LANDING_LENGTH = 1.2
# T4GaitCfg defaults to fixed_clock; standing commands freeze the clock.
GAIT_CYCLE = 0.85
STANDING_COMMAND_THRESHOLD = 0.1
GAIT_AIR_RATIO = np.array([0.38, 0.38])
GAIT_PHASE_OFFSET = np.array([0.38, 0.88])

# D455 intrinsics + schema head-height pose (forward_camera site, 35 deg down).
DEPTH_HFOV_DEG = 87.0
DEPTH_WIDTH, DEPTH_HEIGHT = 480, 270
DEPTH_MAX_RANGE = 15.0  # D455 max_range; farther pixels are no-hit in IsaacLab
DEPTH_FOVY = math.degrees(2 * math.atan(math.tan(math.radians(DEPTH_HFOV_DEG / 2)) * DEPTH_HEIGHT / DEPTH_WIDTH))


def depth_vertical_fov_deg(
    image_height: int,
    image_width: int,
    horizontal_fov_deg: float = DEPTH_HFOV_DEG,
) -> float:
    """Vertical FOV that keeps ``horizontal_fov_deg`` at this image aspect."""
    return math.degrees(
        2.0 * math.atan(math.tan(math.radians(horizontal_fov_deg / 2.0)) * image_height / image_width)
    )


DEPTH_CAM_POS = DEPTH_CAMERA_SITE_POS
D455_ROS_ROT_WXYZ = depth_camera_ros_quat_wxyz()
DEPTH_CAM_XYAXES = depth_camera_mujoco_xyaxes()

SPAWN_Z = 0.85
# Warp training only hits `/World/ground`. Put MuJoCo terrain in this group and
# hide robot collision (default group 0) from the depth camera.
DEPTH_TERRAIN_GEOM_GROUP = 3


def quat_wxyz_to_mat(quat: tuple[float, float, float, float] | np.ndarray) -> np.ndarray:
    """Convert a ``(w, x, y, z)`` quaternion to a 3x3 rotation matrix."""
    w, x, y, z = (float(value) for value in quat)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def ros_offset_to_mujoco_xyaxes(rot_wxyz: tuple[float, float, float, float] | np.ndarray) -> np.ndarray:
    """Convert an Isaac Lab ROS camera offset quaternion to MuJoCo ``xyaxes``.

    Isaac Lab stores ``OffsetCfg.rot`` as ``(w, x, y, z)`` in the ROS optical
    frame (+Z forward, +X right, +Y down) and converts ROS -> OpenGL/USD with a
    180 deg rotation about X. MuJoCo cameras use that same OpenGL convention.
    """
    parent_from_ros = quat_wxyz_to_mat(rot_wxyz)
    ros_from_opengl = np.diag([1.0, -1.0, -1.0])
    parent_from_opengl = parent_from_ros @ ros_from_opengl
    return np.concatenate([parent_from_opengl[:, 0], parent_from_opengl[:, 1]])


def camera_look_axes(xyaxes: tuple[float, ...] | np.ndarray) -> dict[str, np.ndarray]:
    """Return MuJoCo camera right / up / look axes from an ``xyaxes`` vector."""
    axes = np.asarray(xyaxes, dtype=np.float64).reshape(6)
    right = axes[:3]
    up = axes[3:]
    look = -np.cross(right, up)
    return {"right": right, "up": up, "look": look}


def body_velocities_world(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    """Return body linear/angular velocities in world coordinates for MuJoCo 3.x."""
    linear = np.empty((model.nbody, 3), dtype=np.float64)
    angular = np.empty((model.nbody, 3), dtype=np.float64)
    spatial = np.empty(6, dtype=np.float64)
    for body_id in range(model.nbody):
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, spatial, 0)
        angular[body_id] = spatial[:3]
        linear[body_id] = spatial[3:]
    return linear, angular


def root_yaw_wxyz(quat: np.ndarray) -> float:
    """Yaw of a MuJoCo floating-base ``(w, x, y, z)`` quaternion."""
    w, x, y, z = (float(value) for value in quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def classify_sensor_depth(sensor_depth: np.ndarray, max_range: float = DEPTH_MAX_RANGE) -> dict:
    """Classify a raw MuJoCo depth frame as a vertical or horizontal hit band.

    The training D455 is a 90 deg rolled forward camera, so a flat floor is a
    tall, narrow vertical strip. An underfoot / 35 deg head camera would instead
    fill the lower image with a wide horizontal ground band.
    """
    hit = (sensor_depth >= 0.05) & (sensor_depth <= max_range)
    col_frac = hit.mean(axis=0)
    row_frac = hit.mean(axis=1)
    hit_cols = np.flatnonzero(col_frac > 0.25)
    band = (int(hit_cols[0]), int(hit_cols[-1]) + 1) if hit_cols.size else (0, 0)
    orientation = "vertical_band" if float(col_frac.std()) > float(row_frac.std()) else "horizontal_band"
    return {
        "hit_fraction": float(hit.mean()),
        "col_frac_std": float(col_frac.std()),
        "row_frac_std": float(row_frac.std()),
        "orientation": orientation,
        "band_cols": band,
        "band_width_frac": float((band[1] - band[0]) / max(sensor_depth.shape[1], 1)),
        "median_hit_depth": float(np.median(sensor_depth[hit])) if np.any(hit) else math.nan,
    }


def _named_body_pos(model: mujoco.MjModel, name: str) -> np.ndarray | None:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        return None
    return np.array(model.body_pos[body_id], dtype=np.float64)


def course_waypoints_from_model(model: mujoco.MjModel) -> np.ndarray:
    """Build a path to the goal. Weave only when the 100m poles are present."""
    named: list[tuple[str, np.ndarray]] = []
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not name or "wall" in name:
            continue
        if name.startswith("obstacle_") or name.startswith("hurdle_") or name.startswith("loco_"):
            named.append((name, np.array(model.body_pos[body_id], dtype=np.float64)))
    goal = _named_body_pos(model, "goal")
    poles = [pos for name, pos in named if "pole" in name]
    if poles:
        named.sort(key=lambda item: (float(item[1][0]), item[0]))
        points: list[list[float]] = [[0.5, 0.0]]
        for name, pos in named:
            x_pos, y_pos = float(pos[0]), float(pos[1])
            if "pole" in name:
                points.append([x_pos, -0.45 if y_pos >= 0.0 else 0.45])
            elif "l_turn_corner" in name:
                points.append([x_pos, 0.0])
                points.append([x_pos, -2.2])
            elif "bridge" in name:
                points.append([x_pos, y_pos])
            elif "finish" in name:
                points.append([x_pos, 0.0])
        if goal is not None:
            points.append([float(goal[0]), float(goal[1])])
        return np.asarray(points, dtype=np.float64)
    points = [[0.0, 0.0]]
    named.sort(key=lambda item: (float(item[1][0]), item[0]))
    for _name, pos in named:
        points.append([float(pos[0]), 0.0])
    if goal is not None:
        points.append([float(goal[0]), float(goal[1])])
    elif len(points) == 1:
        points.append([8.0, 0.0])
    return np.asarray(points, dtype=np.float64)


def depth_source_size_for_student(*, is_gru: bool) -> tuple[int, int]:
    """GRU LightLP students train on native 48x64 tiled RTX; Stage E still uses 270x480."""
    if is_gru:
        return DEPTH_POLICY_SIZE
    return (DEPTH_HEIGHT, DEPTH_WIDTH)


def terrain_only_depth_option() -> mujoco.MjvOption:
    """Hide robot geoms. Kept for ablations; deploy eval must see the body."""
    option = mujoco.MjvOption()
    option.geomgroup = np.zeros(6, dtype=np.uint8)
    option.geomgroup[DEPTH_TERRAIN_GEOM_GROUP] = 1
    return option


def robot_and_terrain_depth_option() -> mujoco.MjvOption:
    """RTX-style depth: terrain and robot both visible (self-occlusion)."""
    option = mujoco.MjvOption()
    option.geomgroup = np.ones(6, dtype=np.uint8)
    return option


def enable_sparse_terrain_in_mjv_option(option: mujoco.MjvOption) -> None:
    """Show geom group 3 in the interactive viewer.

    MuJoCo's default ``MjvOption.geomgroup`` is ``[1, 1, 1, 0, 0, 0]``. Sparse
    tiles, pit floor, and the start/finish pads are group 3 so the depth camera
    can isolate terrain. The offscreen follow-cam renderer still draws them;
    ``launch_passive`` does not unless this bit is turned on.
    """
    option.geomgroup[DEPTH_TERRAIN_GEOM_GROUP] = 1


def apply_terrain_only_depth_groups(model: mujoco.MjModel) -> None:
    """Move the MJCF ground plane onto the terrain depth group."""
    ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    if ground_id >= 0:
        model.geom_group[ground_id] = DEPTH_TERRAIN_GEOM_GROUP


def render_standing_sensor_depth(
    *, hurdles: bool = False, rule_contract: bool = False, include_robot: bool = True
) -> np.ndarray:
    """Render one raw MuJoCo depth frame at the T4 standing pose."""
    xml_path = build_model_xml(hurdles, rule_contract)
    model = mujoco.MjModel.from_xml_path(xml_path)
    apply_terrain_only_depth_groups(model)
    data = mujoco.MjData(model)
    data.qpos[2] = SPAWN_Z
    for name, target in zip(T4_JOINT_NAMES, STANDING_POS):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint_id]] = target
    mujoco.mj_forward(model, data)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "depth_cam")
    renderer = mujoco.Renderer(model, height=DEPTH_HEIGHT, width=DEPTH_WIDTH)
    try:
        renderer.enable_depth_rendering()
        scene_option = robot_and_terrain_depth_option() if include_robot else terrain_only_depth_option()
        renderer.update_scene(data, camera=cam_id, scene_option=scene_option)
        return np.ascontiguousarray(renderer.render()[:, ::-1])
    finally:
        renderer.close()


def _xml_body_block(
    name: str,
    pos: tuple[float, float, float],
    geoms: list[str],
    quat: tuple[float, float, float, float] | None = None,
) -> str:
    attrs = [f'name="{name}"', f'pos="{" ".join(f"{value:.4f}" for value in pos)}"']
    if quat is not None:
        attrs.append(f'quat="{" ".join(f"{value:.4f}" for value in quat)}"')
    lines = [f"\n    <body {' '.join(attrs)}>"]
    lines.extend(f"\n      {geom}" for geom in geoms)
    lines.append("\n    </body>")
    return "".join(lines)


def _xml_box_geom(
    name: str,
    size: tuple[float, float, float],
    *,
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    quat: tuple[float, float, float, float] | None = None,
    rgba: tuple[float, float, float, float] | None = None,
    density: float = 0.0,
) -> str:
    attrs = [f'name="{name}"', 'type="box"', f'size="{" ".join(f"{value:.4f}" for value in size)}"']
    if pos != (0.0, 0.0, 0.0):
        attrs.append(f'pos="{" ".join(f"{value:.4f}" for value in pos)}"')
    if quat is not None:
        attrs.append(f'quat="{" ".join(f"{value:.4f}" for value in quat)}"')
    if rgba is not None:
        attrs.append(f'rgba="{" ".join(f"{value:.4f}" for value in rgba)}"')
    attrs.append(f'density="{density:.1f}"')
    attrs.append(f'group="{DEPTH_TERRAIN_GEOM_GROUP}"')
    return f"<geom {' '.join(attrs)}/>"


def _xml_cylinder_geom(
    name: str,
    size: tuple[float, float],
    *,
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    quat: tuple[float, float, float, float] | None = None,
    rgba: tuple[float, float, float, float] | None = None,
    density: float = 0.0,
) -> str:
    attrs = [f'name="{name}"', 'type="cylinder"', f'size="{" ".join(f"{value:.4f}" for value in size)}"']
    if pos != (0.0, 0.0, 0.0):
        attrs.append(f'pos="{" ".join(f"{value:.4f}" for value in pos)}"')
    if quat is not None:
        attrs.append(f'quat="{" ".join(f"{value:.4f}" for value in quat)}"')
    if rgba is not None:
        attrs.append(f'rgba="{" ".join(f"{value:.4f}" for value in rgba)}"')
    attrs.append(f'density="{density:.1f}"')
    attrs.append(f'group="{DEPTH_TERRAIN_GEOM_GROUP}"')
    return f"<geom {' '.join(attrs)}/>"


def _quat_from_axis_angle(axis: tuple[float, float, float], angle_deg: float) -> tuple[float, float, float, float]:
    axis_vec = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis_vec)
    if norm == 0:
        raise ValueError("axis must be non-zero")
    axis_vec = axis_vec / norm
    half = math.radians(angle_deg) * 0.5
    s = math.sin(half)
    return (math.cos(half), float(axis_vec[0] * s), float(axis_vec[1] * s), float(axis_vec[2] * s))


def _add_step_ramp(
    parts: list[str],
    *,
    prefix: str,
    x_start: float,
    length: float,
    width: float,
    z_start: float,
    height: float,
    steps: int,
    rgba: tuple[float, float, float, float],
) -> float:
    step_length = length / steps
    step_height = abs(height) / steps
    direction = 1.0 if height >= 0.0 else -1.0
    for idx in range(steps):
        parts.append(
            _xml_body_block(
                f"{prefix}_{idx}",
                (x_start + (idx + 0.5) * step_length, 0.0, z_start + direction * (idx + 0.5) * step_height),
                [
                    _xml_box_geom(
                        f"{prefix}_{idx}_geom",
                        (step_length * 0.5, width * 0.5, step_height * 0.5),
                        rgba=rgba,
                    )
                ],
            )
        )
    return x_start + length


def _add_ramp(
    parts: list[str],
    *,
    name: str,
    x_start: float,
    length: float,
    width: float,
    angle_deg: float,
    z_start: float,
    axis: str = "y",
    thickness: float = 0.06,
    rgba: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0),
) -> float:
    quat = _quat_from_axis_angle((0.0, 1.0, 0.0) if axis == "y" else (1.0, 0.0, 0.0), angle_deg)
    center = (
        x_start + 0.5 * length * math.cos(math.radians(angle_deg)) if axis == "y" else x_start + 0.5 * length,
        0.0,
        z_start + 0.5 * length * math.sin(math.radians(angle_deg)) + 0.5 * thickness,
    )
    parts.append(
        _xml_body_block(
            name,
            center,
            [
                _xml_box_geom(
                    f"{name}_geom",
                    (length * 0.5, width * 0.5, thickness * 0.5),
                    quat=quat,
                    rgba=rgba,
                )
            ],
        )
    )
    return x_start + length


def build_rule_contract_course() -> str:
    """Build a conservative 100 m obstacle course from the local rule contract."""
    parts: list[str] = []
    x = 2.0

    parts.append("\n    <!-- 1. continuous slope -->")
    x = _add_ramp(
        parts,
        name="obstacle_1_slope",
        x_start=x,
        length=7.4,
        width=2.4,
        angle_deg=15.0,
        z_start=0.0,
        axis="y",
        thickness=0.08,
        rgba=(0.72, 0.72, 0.72, 1.0),
    )

    parts.append("\n    <!-- 2. continuous hurdles -->")
    hurdle_start = x + 2.0
    for idx in range(10):
        bar_x = hurdle_start + idx * 1.1
        parts.append(
            _xml_body_block(
                f"obstacle_2_hurdle_{idx + 1}",
                (bar_x, 0.0, 0.15),
                [
                    _xml_box_geom(
                        f"obstacle_2_hurdle_{idx + 1}_geom",
                        (0.025, 1.2, 0.15),
                        rgba=(0.88, 0.49, 0.12, 1.0),
                    )
                ],
            )
        )
    x = hurdle_start + 9 * 1.1 + 0.2

    parts.append("\n    <!-- 3. cross slope -->")
    parts.append(
        _xml_body_block(
            "obstacle_3_cross_slope",
            (x + 3.0, 0.0, 0.0),
            [
                _xml_box_geom(
                    "obstacle_3_cross_slope_geom",
                    (3.0, 1.5, 0.06),
                    quat=_quat_from_axis_angle((1.0, 0.0, 0.0), 15.0),
                    rgba=(0.64, 0.64, 0.76, 1.0),
                )
            ],
        )
    )
    x += 6.0

    parts.append("\n    <!-- 4. weave poles -->")
    pole_y = [0.7, -0.7, 0.7, -0.7, 0.7]
    for idx, y in enumerate(pole_y):
        parts.append(
            _xml_body_block(
                f"obstacle_4_pole_{idx + 1}",
                (x + idx * 1.1, y, 0.75),
                [
                    _xml_cylinder_geom(
                        f"obstacle_4_pole_{idx + 1}_geom",
                        (0.015, 0.75),
                        rgba=(0.93, 0.80, 0.18, 1.0),
                    )
                ],
            )
        )
    x += 5.0

    parts.append("\n    <!-- 5. symmetric slope -->")
    x = _add_ramp(
        parts,
        name="obstacle_5_up_slope",
        x_start=x,
        length=1.5,
        width=3.5,
        angle_deg=20.0,
        z_start=0.0,
        axis="y",
        thickness=0.08,
        rgba=(0.69, 0.69, 0.69, 1.0),
    )
    parts.append(
        _xml_body_block(
            "obstacle_5_peak",
            (x + 0.15, 0.0, 0.54),
            [
                _xml_box_geom(
                    "obstacle_5_peak_geom",
                    (0.15, 1.75, 0.04),
                    rgba=(0.69, 0.69, 0.69, 1.0),
                )
            ],
        )
    )
    x = _add_ramp(
        parts,
        name="obstacle_5_down_slope",
        x_start=x + 0.15,
        length=1.5,
        width=3.5,
        angle_deg=-20.0,
        z_start=0.54,
        axis="y",
        thickness=0.08,
        rgba=(0.69, 0.69, 0.69, 1.0),
    )

    parts.append("\n    <!-- 6. stairs -->")
    x = _add_step_ramp(
        parts,
        prefix="obstacle_6_stair",
        x_start=x + 1.0,
        length=5.04,
        width=3.0,
        z_start=0.0,
        height=0.9,
        steps=6,
        rgba=(0.58, 0.50, 0.42, 1.0),
    )
    x = _add_step_ramp(
        parts,
        prefix="obstacle_6_down",
        x_start=x,
        length=5.04,
        width=3.0,
        z_start=0.9,
        height=-0.9,
        steps=6,
        rgba=(0.58, 0.50, 0.42, 1.0),
    )

    parts.append("\n    <!-- 7. S bridge -->")
    bridge_segments = [
        (2.0, 0.0, 0.10, 0.0),
        (2.0, 0.15, 0.10, 8.0),
        (2.0, -0.15, 0.10, -8.0),
        (2.0, 0.15, 0.10, 8.0),
        (2.0, 0.0, 0.10, 0.0),
    ]
    bridge_x = x + 1.0
    for idx, (seg_len, offset_y, height, angle_deg) in enumerate(bridge_segments):
        quat = _quat_from_axis_angle((0.0, 0.0, 1.0), angle_deg)
        parts.append(
            _xml_body_block(
                f"obstacle_7_bridge_{idx + 1}",
                (bridge_x + seg_len * 0.5, offset_y, height),
                [
                    _xml_box_geom(
                        f"obstacle_7_bridge_{idx + 1}_deck",
                        (seg_len * 0.5, 0.25, 0.04),
                        quat=quat,
                        rgba=(0.34, 0.49, 0.74, 1.0),
                    ),
                    _xml_box_geom(
                        f"obstacle_7_bridge_{idx + 1}_rail_l",
                        (seg_len * 0.5, 0.03, 0.10),
                        pos=(0.0, 0.28, 0.10),
                        quat=quat,
                        rgba=(0.28, 0.40, 0.62, 1.0),
                    ),
                    _xml_box_geom(
                        f"obstacle_7_bridge_{idx + 1}_rail_r",
                        (seg_len * 0.5, 0.03, 0.10),
                        pos=(0.0, -0.28, 0.10),
                        quat=quat,
                        rgba=(0.28, 0.40, 0.62, 1.0),
                    ),
                ],
            )
        )
        bridge_x += seg_len
    x = bridge_x + 1.0

    parts.append("\n    <!-- 8. jump platform -->")
    parts.append(
        _xml_body_block(
            "obstacle_8_platform",
            (x + 1.2, 0.0, 0.5),
            [
                _xml_box_geom(
                    "obstacle_8_platform_geom",
                    (1.2, 1.2, 0.5),
                    rgba=(0.45, 0.31, 0.23, 1.0),
                )
            ],
        )
    )
    x += 4.0

    parts.append("\n    <!-- 9. crawl tunnel -->")
    tunnel_x = x + 0.5
    parts.append(
        _xml_body_block(
            "obstacle_9_tunnel",
            (tunnel_x + 3.0, 0.0, 0.0),
            [
                _xml_box_geom(
                    "obstacle_9_tunnel_floor",
                    (3.0, 1.0, 0.025),
                    pos=(0.0, 0.0, 0.025),
                    rgba=(0.58, 0.58, 0.58, 1.0),
                ),
                _xml_box_geom(
                    "obstacle_9_tunnel_roof",
                    (3.0, 1.0, 0.025),
                    pos=(0.0, 0.0, 0.525),
                    rgba=(0.36, 0.36, 0.36, 1.0),
                ),
                _xml_box_geom(
                    "obstacle_9_tunnel_wall_l",
                    (3.0, 0.05, 0.25),
                    pos=(0.0, 1.0, 0.25),
                    rgba=(0.42, 0.42, 0.42, 1.0),
                ),
                _xml_box_geom(
                    "obstacle_9_tunnel_wall_r",
                    (3.0, 0.05, 0.25),
                    pos=(0.0, -1.0, 0.25),
                    rgba=(0.42, 0.42, 0.42, 1.0),
                ),
            ],
        )
    )
    x = tunnel_x + 6.0

    parts.append("\n    <!-- 10. narrow L turn -->")
    l_x = x + 0.5
    parts.append(
        _xml_body_block(
            "obstacle_10_l_turn_x",
            (l_x + 1.75, 0.0, 0.02),
            [
                _xml_box_geom(
                    "obstacle_10_l_turn_x_floor",
                    (1.75, 0.30, 0.02),
                    rgba=(0.52, 0.52, 0.52, 1.0),
                ),
                _xml_box_geom(
                    "obstacle_10_l_turn_x_wall_l",
                    (1.75, 0.03, 0.60),
                    pos=(0.0, 0.33, 0.60),
                    rgba=(0.35, 0.35, 0.35, 1.0),
                ),
                _xml_box_geom(
                    "obstacle_10_l_turn_x_wall_r",
                    (1.75, 0.03, 0.60),
                    pos=(0.0, -0.33, 0.60),
                    rgba=(0.35, 0.35, 0.35, 1.0),
                ),
            ],
        )
    )
    parts.append(
        _xml_body_block(
            "obstacle_10_l_turn_corner",
            (l_x + 3.5, -1.25, 0.02),
            [
                _xml_box_geom(
                    "obstacle_10_l_turn_corner_floor",
                    (0.30, 1.25, 0.02),
                    rgba=(0.52, 0.52, 0.52, 1.0),
                ),
                _xml_box_geom(
                    "obstacle_10_l_turn_corner_wall_l",
                    (0.03, 1.25, 0.60),
                    pos=(0.33, 0.0, 0.60),
                    rgba=(0.35, 0.35, 0.35, 1.0),
                ),
                _xml_box_geom(
                    "obstacle_10_l_turn_corner_wall_r",
                    (0.03, 1.25, 0.60),
                    pos=(-0.33, 0.0, 0.60),
                    rgba=(0.35, 0.35, 0.35, 1.0),
                ),
            ],
            quat=_quat_from_axis_angle((0.0, 0.0, 1.0), -90.0),
        )
    )
    x = l_x + 4.0

    parts.append(_xml_body_block("obstacle_10_finish", (100.0, 0.0, 0.01), [_xml_box_geom("obstacle_10_finish_geom", (0.05, 1.5, 0.01), rgba=(0.20, 0.74, 0.34, 1.0))]))
    return "".join(parts)


def _goal_xml(x: float, y: float = 0.0) -> str:
    return _xml_body_block(
        "goal",
        (x, y, 0.0),
        [
            _xml_cylinder_geom("goal_pole", (0.04, 0.55), pos=(0.0, 0.0, 0.55), rgba=(0.12, 0.82, 0.28, 1.0)),
            _xml_box_geom("goal_pad", (0.25, 0.25, 0.02), pos=(0.0, 0.0, 0.02), rgba=(0.12, 0.82, 0.28, 1.0)),
        ],
    )


def _add_solid_wedge(
    parts: list[str],
    *,
    prefix: str,
    x_start: float,
    length: float,
    width: float,
    height: float,
    rising: bool = True,
    slices: int = 16,
    rgba: tuple[float, float, float, float] = (0.62, 0.62, 0.64, 1.0),
) -> float:
    """Ground-backed ramp: stacked boxes from z=0, not a floating plank."""
    slice_len = length / slices
    for index in range(slices):
        frac = (index + 1) / slices if rising else (slices - index) / slices
        slab_h = max(0.03, abs(height) * frac)
        parts.append(
            _xml_body_block(
                f"{prefix}_{index}",
                (x_start + (index + 0.5) * slice_len, 0.0, 0.5 * slab_h),
                [
                    _xml_box_geom(
                        f"{prefix}_{index}_geom",
                        (0.5 * slice_len, 0.5 * width, 0.5 * slab_h),
                        rgba=rgba,
                    )
                ],
            )
        )
    return x_start + length


def _add_rough_patch(
    parts: list[str],
    *,
    prefix: str,
    x_start: float,
    length: float,
    width: float,
    cell: float = 0.25,
    height_range: tuple[float, float] = (0.015, 0.08),
    seed: int = 7,
) -> float:
    """Stage E style random-rough: small ground-backed blocks."""
    rng = np.random.default_rng(seed)
    cols = max(1, int(round(length / cell)))
    rows = max(1, int(round(width / cell)))
    dx = length / cols
    dy = width / rows
    for ix in range(cols):
        for iy in range(rows):
            height = float(rng.uniform(*height_range))
            y_pos = -0.5 * width + (iy + 0.5) * dy
            parts.append(
                _xml_body_block(
                    f"{prefix}_{ix}_{iy}",
                    (x_start + (ix + 0.5) * dx, y_pos, 0.5 * height),
                    [
                        _xml_box_geom(
                            f"{prefix}_{ix}_{iy}_geom",
                            (0.48 * dx, 0.48 * dy, 0.5 * height),
                            rgba=(0.40, 0.46, 0.38, 1.0),
                        )
                    ],
                )
            )
    return x_start + length


def _add_box_grid(
    parts: list[str],
    *,
    prefix: str,
    x_start: float,
    length: float,
    width: float,
    cell: float = 0.45,
    heights: tuple[float, ...] = (0.06, 0.10, 0.14, 0.08),
) -> float:
    """Stage E MeshRandomGrid-style boxes, 0–15 cm class."""
    cols = max(1, int(round(length / cell)))
    rows = max(1, int(round(width / cell)))
    dx = length / cols
    dy = width / rows
    for ix in range(cols):
        for iy in range(rows):
            height = heights[(ix + 2 * iy) % len(heights)]
            y_pos = -0.5 * width + (iy + 0.5) * dy
            parts.append(
                _xml_body_block(
                    f"{prefix}_{ix}_{iy}",
                    (x_start + (ix + 0.5) * dx, y_pos, 0.5 * height),
                    [
                        _xml_box_geom(
                            f"{prefix}_{ix}_{iy}_geom",
                            (0.42 * dx, 0.42 * dy, 0.5 * height),
                            rgba=(0.50, 0.40, 0.30, 1.0),
                        )
                    ],
                )
            )
    return x_start + length


def _add_wave_patch(
    parts: list[str],
    *,
    prefix: str,
    x_start: float,
    length: float,
    width: float,
    amplitude: float = 0.12,
    waves: float = 2.5,
    slices: int = 20,
) -> float:
    """Stage E wave terrain as ground-backed lateral strips."""
    dx = length / slices
    for index in range(slices):
        phase = 2.0 * math.pi * waves * (index + 0.5) / slices
        height = max(0.02, 0.04 + 0.5 * amplitude * (1.0 + math.sin(phase)))
        parts.append(
            _xml_body_block(
                f"{prefix}_{index}",
                (x_start + (index + 0.5) * dx, 0.0, 0.5 * height),
                [
                    _xml_box_geom(
                        f"{prefix}_{index}_geom",
                        (0.5 * dx, 0.5 * width, 0.5 * height),
                        rgba=(0.36, 0.48, 0.56, 1.0),
                    )
                ],
            )
        )
    return x_start + length


def _corridor_walls(x_start: float, x_end: float, half_y: float = LOCO_LANE_HALF, height: float = 0.55) -> str:
    """Lane walls so the robot has to cross the obstacles instead of walking around."""
    length = x_end - x_start
    center_x = 0.5 * (x_start + x_end)
    geoms = []
    for side, name in ((1.0, "wall_left"), (-1.0, "wall_right")):
        geoms.append(
            _xml_body_block(
                name,
                (center_x, side * half_y, 0.5 * height),
                [
                    _xml_box_geom(
                        f"{name}_geom",
                        (0.5 * length, 0.04, 0.5 * height),
                        rgba=(0.28, 0.28, 0.30, 1.0),
                    )
                ],
            )
        )
    return "".join(geoms)


def build_loco_course() -> str:
    """Stage E style lane: rough, boxes, wave, hurdles, solid slopes, stairs."""
    half_w = 0.5 * LOCO_OBSTACLE_WIDTH
    parts: list[str] = ["\n    <!-- loco: Stage E buckets along a corridor -->"]
    x_cursor = 1.8
    x_cursor = _add_rough_patch(
        parts,
        prefix="loco_rough",
        x_start=x_cursor,
        length=4.0,
        width=LOCO_OBSTACLE_WIDTH,
        height_range=(0.015, 0.08),
    )
    x_cursor = _add_box_grid(
        parts,
        prefix="loco_boxes",
        x_start=x_cursor + 0.4,
        length=3.6,
        width=LOCO_OBSTACLE_WIDTH,
    )
    x_cursor = _add_wave_patch(
        parts,
        prefix="loco_wave",
        x_start=x_cursor + 0.4,
        length=4.0,
        width=LOCO_OBSTACLE_WIDTH,
        amplitude=0.16,
        waves=2.5,
    )
    x_cursor += 0.5
    for index, height in enumerate((0.25, 0.28, 0.30), start=1):
        parts.append(
            _xml_body_block(
                f"loco_hurdle_{index}",
                (x_cursor, 0.0, 0.5 * height),
                [
                    _xml_box_geom(
                        f"loco_hurdle_{index}_geom",
                        (0.035, half_w, 0.5 * height),
                        rgba=(0.88, 0.49, 0.12, 1.0),
                    )
                ],
            )
        )
        x_cursor += 1.2
    ramp_height = 3.6 * math.tan(math.radians(15.0))
    x_cursor = _add_solid_wedge(
        parts,
        prefix="loco_ramp_up",
        x_start=x_cursor + 0.4,
        length=3.6,
        width=LOCO_OBSTACLE_WIDTH,
        height=ramp_height,
        rising=True,
    )
    landing = 0.50
    parts.append(
        _xml_body_block(
            "loco_ramp_peak",
            (x_cursor + 0.5 * landing, 0.0, ramp_height - 0.05),
            [_xml_box_geom("loco_ramp_peak_geom", (0.5 * landing, half_w, 0.05), rgba=(0.62, 0.62, 0.64, 1.0))],
        )
    )
    x_cursor = _add_solid_wedge(
        parts,
        prefix="loco_ramp_down",
        x_start=x_cursor + landing,
        length=3.6,
        width=LOCO_OBSTACLE_WIDTH,
        height=ramp_height,
        rising=False,
    )
    stair1_height = LOCO_STAIR_RISE * LOCO_STAIR_STEPS
    x_cursor = _add_step_ramp(
        parts,
        prefix="loco_stair_up",
        x_start=x_cursor + 1.0,
        length=0.30 * LOCO_STAIR_STEPS,
        width=LOCO_OBSTACLE_WIDTH,
        z_start=0.0,
        height=stair1_height,
        steps=LOCO_STAIR_STEPS,
        rgba=(0.58, 0.50, 0.42, 1.0),
    )
    parts.append(
        _xml_body_block(
            "loco_landing",
            (x_cursor + 0.25, 0.0, stair1_height - 0.05),
            [_xml_box_geom("loco_landing_geom", (0.25, half_w, 0.05), rgba=(0.58, 0.50, 0.42, 1.0))],
        )
    )
    x_cursor = _add_step_ramp(
        parts,
        prefix="loco_stair_down",
        x_start=x_cursor + 0.50,
        length=0.30 * LOCO_STAIR_STEPS,
        width=LOCO_OBSTACLE_WIDTH,
        z_start=stair1_height,
        height=-stair1_height,
        steps=LOCO_STAIR_STEPS,
        rgba=(0.58, 0.50, 0.42, 1.0),
    )
    stair2_height = LOCO_STAIR2_RISE * LOCO_STAIR2_STEPS
    x_cursor = _add_step_ramp(
        parts,
        prefix="loco_stair2_up",
        x_start=x_cursor + 1.0,
        length=0.30 * LOCO_STAIR2_STEPS,
        width=LOCO_OBSTACLE_WIDTH,
        z_start=0.0,
        height=stair2_height,
        steps=LOCO_STAIR2_STEPS,
        rgba=(0.50, 0.42, 0.36, 1.0),
    )
    parts.append(
        _xml_body_block(
            "loco_landing2",
            (x_cursor + 0.25, 0.0, stair2_height - 0.05),
            [_xml_box_geom("loco_landing2_geom", (0.25, half_w, 0.05), rgba=(0.50, 0.42, 0.36, 1.0))],
        )
    )
    _add_step_ramp(
        parts,
        prefix="loco_stair2_down",
        x_start=x_cursor + 0.50,
        length=0.30 * LOCO_STAIR2_STEPS,
        width=LOCO_OBSTACLE_WIDTH,
        z_start=stair2_height,
        height=-stair2_height,
        steps=LOCO_STAIR2_STEPS,
        rgba=(0.50, 0.42, 0.36, 1.0),
    )
    parts.append(_goal_xml(*LOCO_GOAL_XY))
    parts.append(_corridor_walls(1.2, LOCO_GOAL_XY[0] + 0.6))
    return "".join(parts)


def build_stair_probe_course(*, include_goal: bool = True) -> str:
    """Build the Stage E 18 cm stair bucket directly in front of reset.

    The probe omits the rough, hurdle and ramp lead-in from ``loco`` so a
    failed rollout can be attributed to the stair interaction itself.
    """
    parts: list[str] = ["\n    <!-- stair probe: shared direct/ZL geometry -->"]
    stair_height = LOCO_STAIR_RISE * LOCO_STAIR_STEPS
    stair_end = _add_step_ramp(
        parts,
        prefix="stair_probe_up",
        x_start=STAIR_PROBE_START_X,
        length=STAIR_PROBE_TREAD * LOCO_STAIR_STEPS,
        width=STAIR_PROBE_WIDTH,
        z_start=0.0,
        height=stair_height,
        steps=LOCO_STAIR_STEPS,
        rgba=(0.58, 0.50, 0.42, 1.0),
    )
    parts.append(
        _xml_body_block(
            "stair_probe_landing",
            (stair_end + 0.5 * STAIR_PROBE_LANDING_LENGTH, 0.0, stair_height - 0.05),
            [
                _xml_box_geom(
                    "stair_probe_landing_geom",
                    (0.5 * STAIR_PROBE_LANDING_LENGTH, 0.5 * STAIR_PROBE_WIDTH, 0.05),
                    rgba=(0.58, 0.50, 0.42, 1.0),
                )
            ],
        )
    )
    if include_goal:
        parts.append(_goal_xml(stair_end + STAIR_PROBE_LANDING_LENGTH - 0.2, 0.0))
    return "".join(parts)


SPARSE_COURSE_TERRAIN = {
    "stepping_stones": "stepping_stones",
    "raised_pillars": "raised_pillars",
    "sparse": "sparse_course",
}
_GROUND_GEOM_RE = re.compile(r'<geom name="ground"[^>]*/>')


def build_sparse_foothold_course(course: str, difficulty: float = 0.0) -> str:
    """Isaac 8 m sparse tile (center pad + lattice + rim) over a real pit."""
    from legged_lab.scripts.play_t4_sparse_teacher_mujoco import _LAYOUT, _geom_xml

    LIGHTLP_STONE_TILE_SIZE = _LAYOUT.LIGHTLP_STONE_TILE_SIZE
    isaac_sparse_tile_geoms = _LAYOUT.isaac_sparse_tile_geoms

    if course not in SPARSE_COURSE_TERRAIN:
        raise ValueError(f"unsupported sparse course {course!r}")
    if course == "stepping_stones":
        geoms = isaac_sparse_tile_geoms(difficulty, "stepping_stones")
    elif course == "raised_pillars":
        geoms = isaac_sparse_tile_geoms(difficulty, "raised_pillars")
    else:
        geoms = isaac_sparse_tile_geoms(difficulty, "stepping_stones", finish_name=None)
        geoms.extend(
            isaac_sparse_tile_geoms(
                difficulty,
                "raised_pillars",
                origin_xy=(LIGHTLP_STONE_TILE_SIZE, 0.0),
                platform_name="transition_platform",
                finish_name="finish_platform",
                name_prefix="b_",
            )
        )
    left = min(float(geom["pos"][0]) - float(geom["size"][0]) for geom in geoms)
    right = max(float(geom["pos"][0]) + float(geom["size"][0]) for geom in geoms)
    pit = (
        f'\n    <geom name="sparse_pit_floor" type="box" pos="{0.5 * (left + right):.8g} 0 -2.05" '
        f'size="{0.5 * (right - left) + 1.0:.8g} 6 0.05" rgba="0.08 0.09 0.10 1" '
        f'condim="3" friction="1 0.005 0.0001" group="{DEPTH_TERRAIN_GEOM_GROUP}"/>'
    )
    terrain_xml = "\n    ".join(_geom_xml(geom) for geom in geoms)
    finish = next(geom for geom in geoms if geom["name"] == "finish_platform")
    return pit + "\n    " + terrain_xml + _goal_xml(float(finish["pos"][0]), 0.0)


def build_model_xml(
    hurdles: bool = False,
    rule_contract: bool = False,
    course: str | None = None,
    difficulty: float = 0.0,
) -> str:
    """Return a patched MJCF: camera on Trunk, optional obstacle course, all world geoms collision-active."""
    if course is None:
        course = "rule" if rule_contract else "hurdles" if hurdles else "flat"
    xml = MJCF.read_text()
    # The patched XML lives in the temp dir, so pin the mesh dir to an
    # absolute path (the original uses a relative `meshdir`).
    meshes_dir = (MJCF.parent / ".." / "meshes").resolve().as_posix()
    xml = xml.replace('meshdir="../meshes/"', f'meshdir="{meshes_dir}/"')
    # The migrated floor declares condim=1 (frictionless); PhysX training
    # terrain is frictional, so restore a frictional ground plane.
    xml = xml.replace('geom name="ground" type="plane" pos="0 0 0" size="0 0 1" material="matplane" condim="1"',
                      'geom name="ground" type="plane" pos="0 0 0" size="0 0 1" material="matplane" condim="3"')
    if course in SPARSE_COURSE_TERRAIN:
        xml, replacements = _GROUND_GEOM_RE.subn("", xml, count=1)
        if replacements != 1:
            raise RuntimeError("failed to remove the infinite ground plane for the sparse pit")
    anchor = '<body name="Trunk"'
    index = xml.index(anchor)
    index = xml.index(">", index) + 1
    # Schema head-height camera: Trunk/forward_camera + 35 deg down.
    xyaxes = " ".join(str(value) for value in DEPTH_CAM_XYAXES)
    camera = (
        f'\n      <camera name="depth_cam" pos="{" ".join(map(str, DEPTH_CAM_POS))}" '
        f'xyaxes="{xyaxes}" fovy="{DEPTH_FOVY:.4f}" mode="fixed"/>\n    '
    )
    xml = xml[:index] + camera + xml[index:]

    world_anchor = xml.index("<worldbody>")
    world_end = xml.index("</worldbody>")
    extras = ""
    if course == "rule":
        extras += build_rule_contract_course()
        extras += _goal_xml(100.0, 0.0)
    elif course == "hurdles":
        for step in range(1, 11):
            x = float(step) * 2.0
            extras += (
                f'\n    <body name="hurdle_{step}" pos="{x} 0 0.15" mocap="true">'
                f'\n      <geom name="hurdle_{step}_geom" type="box" size="0.6 0.02 0.15" density="0"/>\n    </body>'
            )
        extras += _goal_xml(22.0, 0.0)
    elif course == "loco":
        extras += build_loco_course()
    elif course == "stairs":
        extras += build_stair_probe_course()
    elif course in SPARSE_COURSE_TERRAIN:
        extras += build_sparse_foothold_course(course, difficulty)
    else:
        extras += _goal_xml(8.0, 0.0)
    extras += (
        '\n    <body name="view_cam_mount" pos="0 0 0" mocap="true">'
        '\n      <camera name="view_cam" pos="0 0 1.6" euler="0 -1.4 0" fovy="50" mode="fixed"/>\n    </body>'
    )
    xml = xml[: world_end] + extras + xml[world_end:]

    path = Path(tempfile.gettempdir()) / "t4_sim2sim_depth.xml"
    path.write_text(xml)
    return str(path)


class DepthStudentSim:
    """Loads the depth student checkpoint and runs it against the MuJoCo T4."""

    def __init__(
        self,
        checkpoint: str,
        hurdles: bool = False,
        rule_contract: bool = False,
        course: str | None = None,
        difficulty: float = 0.0,
        depth_source_size: tuple[int, int] | None = None,
        include_robot_in_depth: bool = True,
        depth_noise: bool = False,
    ):
        xml_path = build_model_xml(hurdles, rule_contract, course=course, difficulty=difficulty)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        apply_terrain_only_depth_groups(self.model)
        self.model.opt.timestep = SIM_DT
        self.data = mujoco.MjData(self.model)

        # Joint and actuator ids follow the T4 order (MJCF order == T4_JOINT_NAMES).
        self.joint_ids = np.asarray(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in T4_JOINT_NAMES]
        )
        self.qpos_adr = np.asarray([self.model.jnt_qposadr[j] for j in self.joint_ids])
        self.dof_adr = np.asarray([self.model.jnt_dofadr[j] for j in self.joint_ids])
        self.actuator_ids = np.asarray(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "M_" + name[2:]) for name in T4_JOINT_NAMES]
        )
        self.kp = np.asarray([pd_group(name)[0] for name in T4_JOINT_NAMES])
        self.kd = np.asarray([pd_group(name)[1] for name in T4_JOINT_NAMES])
        # The MJCF declares torque limits on the joints (actuatorfrcrange ->
        # jnt_actfrcrange); the actuator-level forcerange is unset (zero).
        self.effort_limit = np.asarray(self.model.jnt_actfrcrange[self.joint_ids, 1])
        self.trunk_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Trunk")
        self._configure_position_servos()

        # Policy: feed-forward Stage E student or S12 GRU student. Always strip
        # teacher / scan-decoder / critic weights so a training checkpoint cannot
        # load privileged weights into the MuJoCo inference path.
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or "model_state_dict" not in state:
            raise RuntimeError(f"checkpoint {checkpoint} does not contain model_state_dict")
        model_state = strip_deployable_state_dict(state["model_state_dict"])
        is_gru = any(key.startswith("memory_s") for key in model_state)
        self.is_gru_student = is_gru
        self.policy = build_depth_student_policy(
            model_state,
            NUM_JOINTS,
            num_teacher_obs=TEACHER_SPARSE_ACTOR_OBS_DIM if is_gru else TEACHER_ACTOR_OBS_DIM,
            depth_shape=(STUDENT_DEPTH_HISTORY_LENGTH if is_gru else DEPTH_HISTORY_LENGTH, *DEPTH_POLICY_SIZE),
            proprio_obs_dim=PROPRIO_FRAME_DIM
            * (STUDENT_PROPRIO_HISTORY_LENGTH if is_gru else PROPRIO_HISTORY_LENGTH),
            recon_scan_offset=sparse_teacher_latest_scan_range()[0],
        )
        self.policy.eval()
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        print(f"[INFO] loaded checkpoint {checkpoint} (iter {state.get('iter')})")

        # Depth renderer. Deploy-domain GRU students train on tiled RTX that
        # sees the robot; keep that self-occlusion here. ``--terrain-only-depth``
        # restores the old warp-style mask for ablations.
        self.include_robot_in_depth = bool(include_robot_in_depth)
        self.depth_noise = bool(depth_noise)
        self._depth_noise_scale = 1.0
        self.depth_source_size = depth_source_size or depth_source_size_for_student(is_gru=is_gru)
        source_height, source_width = self.depth_source_size
        self.renderer = mujoco.Renderer(self.model, height=source_height, width=source_width)
        self.renderer.enable_depth_rendering()
        self.depth_option = (
            robot_and_terrain_depth_option() if self.include_robot_in_depth else terrain_only_depth_option()
        )
        self.depth_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "depth_cam")
        self.model.cam_fovy[self.depth_cam_id] = depth_vertical_fov_deg(source_height, source_width)
        self.last_depth_sensor = np.full(self.depth_source_size, np.inf, dtype=np.float32)
        self.last_depth_raw = np.full(self.depth_source_size, DEPTH_INVALID_VALUE, dtype=np.float32)

        # Buffers. The env zero-fills history at reset (CircularBuffer.reset),
        # so start zeroed rather than repeating the first frame.
        self.proprio_history = np.zeros((PROPRIO_FRAME_DIM * PROPRIO_HISTORY_LENGTH,), dtype=np.float32)
        self.depth_history = np.zeros((DEPTH_HISTORY_LENGTH, *DEPTH_POLICY_SIZE), dtype=np.float32)
        self.depth_counter = 0
        self.previous_action = np.zeros(NUM_JOINTS)
        self.gait_time = 0.0
        self.command = np.zeros(3)
        self.targets = STANDING_POS.copy()
        self.qpos = self.data.qpos
        self.qvel = self.data.qvel

        self._reset_pose()

    def _reset_pose(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.qpos[2] = SPAWN_Z
        self.qpos[self.qpos_adr] = STANDING_POS
        mujoco.mj_forward(self.model, self.data)
        if self.depth_noise:
            self._depth_noise_scale = (1.0 - LIGHTLP_DEPTH_SCALE_JITTER) + 2.0 * LIGHTLP_DEPTH_SCALE_JITTER * float(
                np.random.random()
            )

    def reset_episode(self) -> None:
        """Reset pose, histories and the gait clock for a new attempt."""
        self._reset_pose()
        self.proprio_history[:] = 0.0
        self.depth_history[:] = 0.0
        self.depth_counter = 0
        self.previous_action[:] = 0.0
        self.gait_time = 0.0
        self.command[:] = 0.0
        self.targets = STANDING_POS.copy()
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def _configure_position_servos(self) -> None:
        """Mirror PhysX implicit joint drives with MuJoCo position servos."""
        for i, actuator_id in enumerate(self.actuator_ids):
            self.model.actuator_gaintype[actuator_id] = mujoco.mjtGain.mjGAIN_FIXED
            self.model.actuator_gainprm[actuator_id, :] = 0.0
            self.model.actuator_gainprm[actuator_id, 0] = self.kp[i]
            self.model.actuator_biastype[actuator_id] = mujoco.mjtBias.mjBIAS_AFFINE
            self.model.actuator_biasprm[actuator_id, :] = 0.0
            self.model.actuator_biasprm[actuator_id, 1] = -self.kp[i]
            self.model.actuator_ctrllimited[actuator_id] = 0
            self.model.actuator_forcelimited[actuator_id] = 1
            self.model.actuator_forcerange[actuator_id] = (-self.effort_limit[i], self.effort_limit[i])
            self.model.dof_damping[self.dof_adr[i]] = self.kd[i]
            self.model.dof_frictionloss[self.dof_adr[i]] = 0.0

    def _apply_position_targets(self, targets: np.ndarray) -> None:
        self.data.ctrl[self.actuator_ids] = targets

    def _read_depth(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=self.depth_cam_id, scene_option=self.depth_option)
        depth = self.renderer.render()  # (H, W) planar depth in camera frame; -1 where nothing hit
        # MuJoCo's image x-axis runs opposite to the IsaacLab sensor frame;
        # the server dump shows the ground band on the right, MuJoCo on the
        # left, so mirror horizontally.
        depth = np.ascontiguousarray(depth[:, ::-1])
        self.last_depth_sensor = depth
        # No-hit pixels come back as a large sentinel (not -1) in MuJoCo 3;
        # beyond the D455 max range they must fill the raw invalid value like
        # the IsaacLab stream does (raw 1.0 -> normalized 0.286).
        raw = np.where((depth < 0) | (depth > DEPTH_MAX_RANGE), DEPTH_INVALID_VALUE, depth)
        if self.depth_noise:
            gaussian = np.random.randn(*raw.shape).astype(np.float32)
            raw = apply_metric_depth_noise(raw, gaussian=gaussian, scale=np.float32(self._depth_noise_scale))
        self.last_depth_raw = raw
        raw = np.clip(raw, DEPTH_CLIP_RANGE[0], DEPTH_CLIP_RANGE[1])
        normalized = (raw - DEPTH_CLIP_RANGE[0]) / (DEPTH_CLIP_RANGE[1] - DEPTH_CLIP_RANGE[0])
        if tuple(normalized.shape) == DEPTH_POLICY_SIZE:
            return normalized.astype(np.float32)
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(tensor, size=DEPTH_POLICY_SIZE, mode="area").squeeze(0).squeeze(0)
        return resized.numpy()

    def raw_depth_preview_image(self) -> np.ndarray:
        """Return a color-mapped raw camera preview for debugging.

        No-hit pixels are painted dark gray. Training fills those same pixels
        with 1.0 m, which otherwise blends into the mid-range ground and makes
        the floor look like a thin stick.
        """
        sensor = np.asarray(self.last_depth_sensor, dtype=np.float32)
        hit = (sensor >= 0.05) & (sensor <= DEPTH_MAX_RANGE)
        depth = np.clip(np.where(hit, sensor, DEPTH_CLIP_RANGE[1]), DEPTH_CLIP_RANGE[0], DEPTH_CLIP_RANGE[1])
        normalized = (depth - DEPTH_CLIP_RANGE[0]) / (DEPTH_CLIP_RANGE[1] - DEPTH_CLIP_RANGE[0])
        preview = cv2.applyColorMap((normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
        preview = np.ascontiguousarray(preview)
        preview[~hit] = (36, 36, 36)
        cv2.putText(
            preview,
            "raw hits 0.2-15m",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return preview

    def policy_depth_preview_image(self) -> np.ndarray:
        """Return the exact processed depth frame fed to the policy."""
        frame = np.nan_to_num(
            self.depth_history[-1],
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        frame = np.clip(frame, 0.0, 1.0)
        preview = (frame * 255.0).astype(np.uint8)
        preview = cv2.applyColorMap(preview, cv2.COLORMAP_TURBO)
        preview = cv2.resize(preview, (256, 192), interpolation=cv2.INTER_NEAREST)
        cv2.putText(
            preview,
            "policy 48x64",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return preview

    def debug_preview_image(self) -> np.ndarray:
        """Side-by-side raw camera and policy input for the live overlay."""
        raw = cv2.resize(self.raw_depth_preview_image(), (256, 192), interpolation=cv2.INTER_NEAREST)
        combo = np.hstack([raw, self.policy_depth_preview_image()])
        cv2.putText(
            combo,
            "head 35deg: ground is the lower image",
            (8, 184),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return combo

    def dump_depth_diagnostics(self, output_dir: Path) -> dict:
        """Write raw/policy previews and camera-axis facts for a standing frame."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.depth_counter == 0:
            self._update_depth_history()
        stats = classify_sensor_depth(self.last_depth_sensor)
        axes = camera_look_axes(DEPTH_CAM_XYAXES)
        payload = {
            "camera_pos_trunk": list(DEPTH_CAM_POS),
            "camera_xyaxes": list(DEPTH_CAM_XYAXES),
            "look": axes["look"].tolist(),
            "right": axes["right"].tolist(),
            "up": axes["up"].tolist(),
            "matches_d455_training_offset": True,
            "schema_camera_site_pos": list(DEPTH_CAMERA_SITE_POS),
            "schema_camera_pitch_deg": DEPTH_CAMERA_PITCH_DEG,
            "matches_schema_head_site": list(DEPTH_CAM_POS) == list(DEPTH_CAMERA_SITE_POS),
            "depth": stats,
        }
        (output_dir / "depth_diagnostics.json").write_text(json.dumps(payload, indent=2) + "\n")
        cv2.imwrite(str(output_dir / "raw_depth.png"), self.raw_depth_preview_image())
        cv2.imwrite(str(output_dir / "policy_depth.png"), self.policy_depth_preview_image())
        cv2.imwrite(str(output_dir / "debug_preview.png"), self.debug_preview_image())
        return payload

    def _update_depth_history(self) -> None:
        frame = self._read_depth()
        if self.depth_counter == 0:
            self.depth_history[:] = frame
        elif self.depth_counter % DEPTH_UPDATE_DECIMATION == 0:
            self.depth_history[:-1] = self.depth_history[1:]
            self.depth_history[-1] = frame
        self.depth_counter += 1

    def _policy_observation(self) -> np.ndarray:
        """Stack history for Stage E CNN; GRU LightLP student takes the newest frame only."""
        if getattr(self, "is_gru_student", False):
            proprio = self.proprio_history[-PROPRIO_FRAME_DIM:]
            depth = self.depth_history[-1].reshape(-1)
            obs = np.concatenate([proprio, depth]).astype(np.float32)
            if obs.shape[0] != STUDENT_ACTOR_OBS_DIM:
                raise RuntimeError(f"GRU student obs width {obs.shape[0]} != {STUDENT_ACTOR_OBS_DIM}")
            return obs
        return np.concatenate([self.proprio_history, self.depth_history.reshape(-1)]).astype(np.float32)

    def _proprio_frame(self) -> np.ndarray:
        ang_vel_b = self.qvel[3:6]
        gravity_b = quat_rotate_inverse_wxyz(self.qpos[3:7], np.array([0.0, 0.0, -1.0]))
        joint_pos = self.qpos[self.qpos_adr] - STANDING_POS
        joint_vel = self.qvel[self.dof_adr]
        phase = (self.gait_time + GAIT_PHASE_OFFSET) % 1.0
        return np.concatenate(
            [
                ang_vel_b,
                gravity_b,
                self.command,
                joint_pos,
                joint_vel,
                self.previous_action,
                np.sin(2 * np.pi * phase),
                np.cos(2 * np.pi * phase),
                GAIT_AIR_RATIO,
            ]
        ).astype(np.float32)

    def step(self) -> tuple[np.ndarray, dict]:
        """Advance one policy step (decimation sim steps); return (obs, info)."""
        for _ in range(DECIMATION):
            self._apply_position_targets(self.targets)
            mujoco.mj_step(self.model, self.data)

        if np.linalg.norm(self.command[:2]) > STANDING_COMMAND_THRESHOLD:
            self.gait_time += STEP_DT / GAIT_CYCLE

        frame = self._proprio_frame()
        self.proprio_history = np.roll(self.proprio_history, shift=-PROPRIO_FRAME_DIM)
        self.proprio_history[-PROPRIO_FRAME_DIM:] = frame
        self._update_depth_history()

        obs = self._policy_observation()
        obs = np.clip(obs, -CLIP_OBS, CLIP_OBS)

        info = {
            "trunk_height": self.data.xpos[self.trunk_id][2],
            "root_lin_vel": self.qvel[:3].copy(),
        }
        return obs, info

    def observe(self) -> np.ndarray:
        """Build the current policy observation without stepping physics."""
        frame = self._proprio_frame()
        self.proprio_history = np.roll(self.proprio_history, shift=-PROPRIO_FRAME_DIM)
        self.proprio_history[-PROPRIO_FRAME_DIM:] = frame
        self._update_depth_history()
        obs = self._policy_observation()
        return np.clip(obs, -CLIP_OBS, CLIP_OBS)

    def act(self, obs: np.ndarray) -> None:
        """Run the policy on obs and store the PD targets."""
        with torch.no_grad():
            action = self.policy.act_inference(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
        action = np.clip(action, -CLIP_ACTIONS, CLIP_ACTIONS)
        self.targets = action * ACTION_SCALE + STANDING_POS
        self.previous_action = action.copy()


def interactive_run(
    sim: DepthStudentSim,
    duration: float,
    *,
    navigator: CourseNavigator | None = None,
    auto_command: tuple[float, float, float] | None = None,
) -> dict:
    """Keyboard or waypoint-commanded run; returns metrics.

    Uses MuJoCo's bundled passive viewer (works on Windows where mujoco-viewer
    has no wheels); commands come through the viewer key callback.
    """
    import time
    import mujoco.viewer

    control_mode = "nav" if navigator is not None else "auto" if auto_command is not None else "hold"

    def init_view(viewer) -> None:
        with viewer.lock():
            enable_sparse_terrain_in_mjv_option(viewer.opt)
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = sim.trunk_id
            viewer.cam.fixedcamid = -1
            viewer.cam.distance = 4.8
            viewer.cam.azimuth = 145.0
            viewer.cam.elevation = -18.0

    def apply_high_level_command() -> None:
        if control_mode == "nav" and navigator is not None:
            sim.command[:] = navigator.command(sim.qpos[:2], root_yaw_wxyz(sim.qpos[3:7]))
        elif control_mode == "auto" and auto_command is not None:
            sim.command[:] = auto_command

    def key_callback(key: int) -> None:
        # Do not bind I/J/K/L: those are MuJoCo visualize toggles (inertia/joints/labels).
        if key == 82 and navigator is not None:  # R reset
            sim.reset_episode()
            navigator.reset()

    viewer = mujoco.viewer.launch_passive(
        sim.model, sim.data, key_callback=key_callback, show_left_ui=False, show_right_ui=False
    )
    init_view(viewer)
    depth_viewport = mujoco.MjrRect(16, 16, 512, 192)
    print("[INFO] goal nav is running. Mouse orbits the tracking camera. R resets.")
    print("[INFO] Do not use I/J/K/L — MuJoCo uses those to toggle inertia/joints/labels.")
    print("[INFO] depth inset: raw D455 | policy 48x64. Head-height 35 deg down.")
    print("[INFO] enabled geom group 3 so sparse tiles are visible (viewer default hides 3-5).")

    steps = 0
    start = time.time()
    try:
        apply_high_level_command()
        obs = sim.observe()
        sim.act(obs)
        while steps < int(duration / STEP_DT) and viewer.is_running():
            apply_high_level_command()
            obs, info = sim.step()
            sim.act(obs)
            viewer.set_images((depth_viewport, sim.debug_preview_image()))
            target = ""
            if navigator is not None:
                waypoint = navigator.current_waypoint
                target = f" goal={waypoint[0]:.1f},{waypoint[1]:.1f}"
            viewer.set_texts(
                (
                    mujoco.mjtFontScale.mjFONTSCALE_150,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    "goal nav  |  mouse look  |  R reset",
                    (
                        f"cmd={sim.command[0]:+.2f},{sim.command[1]:+.2f},{sim.command[2]:+.2f} "
                        f"x={sim.qpos[0]:.1f} y={sim.qpos[1]:.2f}{target}"
                    ),
                ),
            )
            viewer.sync()
            steps += 1
            if steps % 250 == 0:
                print(
                    f"t={steps * STEP_DT:6.1f}s cmd={sim.command} "
                    f"xy=({sim.qpos[0]:.1f},{sim.qpos[1]:.2f}) h={info['trunk_height']:.2f} "
                    f"v={info['root_lin_vel'][:2]}"
                )
    finally:
        viewer.clear_images()
        viewer.clear_texts()
        viewer.close()
    return {"steps": steps, "wall_seconds": time.time() - start}


def recorded_run(
    sim: DepthStudentSim,
    output: str,
    duration: float,
    script: str,
    *,
    navigator: CourseNavigator | None = None,
    rollout_output: Path | None = None,
) -> None:
    """Headless scripted run rendered from the follow camera.

    When ``rollout_output`` is supplied, also save a tracking-compatible NPZ
    at policy rate (50 Hz).  The NPZ keeps MuJoCo body order and T4 joint order
    explicit; downstream tracking consumers can validate/reorder it by name.
    """
    import imageio.v2 as imageio

    sim.renderer.enable_depth_rendering()  # keep depth for obs
    rgb_renderer = mujoco.Renderer(sim.model, height=368, width=640)
    view_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, "view_cam")
    mount_body_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "view_cam_mount")
    mount_mocap_id = sim.model.body_mocapid[mount_body_id]
    if mount_mocap_id < 0:
        raise RuntimeError("view_cam_mount is not a mocap body")

    scripts = {
        "forward": lambda t: (0.6, 0.0, 0.0),
        "turn": lambda t: (0.4 if t < 8 else 0.0, 0.0, 0.6 if t >= 8 else 0.0),
        "slalom": lambda t: (0.5, 0.4 * math.sin(2 * math.pi * t / 8.0), 0.0),
        "nav": None,
    }
    if script not in scripts:
        raise KeyError(f"unknown record script {script!r}")
    if script == "nav" and navigator is None:
        raise ValueError("recorded script 'nav' requires a CourseNavigator")
    command_fn = scripts[script]
    use_nav = navigator is not None and script == "nav"

    writer = imageio.get_writer(output, fps=int(1 / STEP_DT))
    rollout = {
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
        "action": [],
        "command": [],
    }

    def capture() -> None:
        body_lin_vel_w, body_ang_vel_w = body_velocities_world(sim.model, sim.data)
        rollout["joint_pos"].append(sim.qpos[sim.qpos_adr].copy())
        rollout["joint_vel"].append(sim.qvel[sim.dof_adr].copy())
        rollout["body_pos_w"].append(sim.data.xpos.copy())
        rollout["body_quat_w"].append(sim.data.xquat.copy())
        rollout["body_lin_vel_w"].append(body_lin_vel_w)
        rollout["body_ang_vel_w"].append(body_ang_vel_w)
        rollout["action"].append(sim.previous_action.copy())
        rollout["command"].append(sim.command.copy())

    steps = int(duration / STEP_DT)
    if use_nav:
        sim.command[:] = navigator.command(sim.qpos[:2], root_yaw_wxyz(sim.qpos[3:7]))
    else:
        sim.command[:] = command_fn(0.0)
    obs = sim.observe()
    sim.act(obs)
    if rollout_output is not None:
        capture()
    for step in range(steps):
        t = step * STEP_DT
        if use_nav:
            sim.command[:] = navigator.command(sim.qpos[:2], root_yaw_wxyz(sim.qpos[3:7]))
        else:
            sim.command[:] = command_fn(t)
        obs, info = sim.step()
        sim.act(obs)
        if rollout_output is not None:
            capture()
        # Follow camera two metres behind the trunk.
        pos = sim.data.xpos[sim.trunk_id]
        sim.data.mocap_pos[mount_mocap_id] = [pos[0] - 2.0, pos[1], 0.0]
        rgb_renderer.update_scene(sim.data, camera=view_id)
        writer.append_data(rgb_renderer.render())
        if step % 500 == 0:
            print(f"t={t:6.1f}s h={info['trunk_height']:.2f} cmd={sim.command}")
    writer.close()
    if rollout_output is not None:
        rollout_output = Path(rollout_output)
        rollout_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            rollout_output,
            schema_version=np.asarray("t4_tracking_rollout.v1"),
            fps=np.asarray(1.0 / STEP_DT, dtype=np.float32),
            joint_names=np.asarray(T4_JOINT_NAMES),
            body_names=np.asarray(
                [mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}" for i in range(sim.model.nbody)]
            ),
            **{key: np.asarray(value, dtype=np.float32) for key, value in rollout.items()},
        )
        print(f"[INFO] wrote rollout NPZ to {rollout_output} ({len(rollout['joint_pos'])} frames)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Path to a Stage S model_*.pt, a run directory, or 'latest'. "
            "Defaults to artifacts/checkpoints/t4_depth_student_latest.pt, then the newest local run."
        ),
    )
    scene = parser.add_mutually_exclusive_group()
    scene.add_argument(
        "--course",
        choices=["loco", "flat", "hurdles", "stairs", "rule", "stepping_stones", "raised_pillars", "sparse"],
        default=None,
        help=(
            "Scene: training-scale loco (default), stair probe, flat, hurdles, 100m rule, "
            "踏石 stepping_stones, 圆桩 raised_pillars, or both (sparse)"
        ),
    )
    scene.add_argument("--hurdles", action="store_true", help="Alias for --course hurdles")
    scene.add_argument("--rule-contract", action="store_true", help="Alias for --course rule")
    control = parser.add_mutually_exclusive_group()
    control.add_argument("--nav", dest="control", action="store_const", const="nav", help="Force goal navigator")
    control.add_argument(
        "--auto-forward",
        dest="control",
        action="store_const",
        const="auto",
        help="Lock a constant forward command",
    )
    parser.set_defaults(control=None)
    parser.add_argument("--cruise", type=float, default=0.55, help="Navigator forward speed in m/s")
    parser.add_argument(
        "--difficulty",
        type=float,
        default=0.0,
        help="Sparse foothold curriculum in [0, 1]; 0 is easy 踏石/圆桩, 1 is hard. Ignored for loco/flat.",
    )
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument("--record", type=Path, default=None, help="Write an MP4 and exit instead of opening the GUI")
    parser.add_argument(
        "--rollout",
        type=Path,
        default=None,
        help="Alongside --record, write a tracking-compatible rollout NPZ at policy rate",
    )
    parser.add_argument(
        "--script",
        type=str,
        default="forward",
        choices=["forward", "turn", "slalom", "nav"],
        help="Command script for --record runs",
    )
    parser.add_argument(
        "--dump-depth",
        type=Path,
        default=None,
        help="Write raw/policy depth previews and camera-axis JSON, then continue",
    )
    parser.add_argument(
        "--depth-noise",
        action="store_true",
        help="Apply the same LightLP Python depth noise used in RTX distill",
    )
    parser.add_argument(
        "--terrain-only-depth",
        action="store_true",
        help="Hide the robot from the depth camera (old warp-style ablation)",
    )
    return parser.parse_args()


def resolve_course(args: argparse.Namespace) -> str:
    if getattr(args, "course", None):
        return args.course
    if args.rule_contract:
        return "rule"
    if args.hurdles:
        return "hurdles"
    return "loco"


def resolve_control_mode(args: argparse.Namespace) -> str:
    if args.control is not None:
        return args.control
    return "nav"


def main() -> None:
    args = parse_args()
    checkpoint = resolve_checkpoint(args.checkpoint)
    course = resolve_course(args)
    print(f"[INFO] using checkpoint {checkpoint}")
    print(f"[INFO] course={course}")
    sim = DepthStudentSim(
        str(checkpoint),
        args.hurdles,
        args.rule_contract,
        course=course,
        difficulty=args.difficulty,
        include_robot_in_depth=not bool(args.terrain_only_depth),
        depth_noise=bool(args.depth_noise),
    )
    if args.dump_depth is not None:
        payload = sim.dump_depth_diagnostics(args.dump_depth)
        print(f"[INFO] wrote depth diagnostics to {args.dump_depth}: {payload['depth']}")
    control_mode = resolve_control_mode(args)
    navigator = None
    if control_mode == "nav":
        navigator = CourseNavigator(course_waypoints_from_model(sim.model), cruise_vx=args.cruise)
        goal = navigator.waypoints[-1]
        print(f"[INFO] goal nav ON -> ({goal[0]:.1f}, {goal[1]:.1f}), cruise={args.cruise:.2f}")
    if args.record is not None:
        script = "nav" if control_mode == "nav" and args.script == "forward" else args.script
        recorded_run(sim, str(args.record), args.duration, script, navigator=navigator, rollout_output=args.rollout)
    elif args.rollout is not None:
        raise SystemExit("--rollout requires --record so the rollout has a fixed scripted command")
    else:
        auto_command = (0.55, 0.0, 0.0) if control_mode == "auto" else None
        metrics = interactive_run(sim, args.duration, navigator=navigator, auto_command=auto_command)
        print(f"[DONE] {metrics['steps']} steps in {metrics['wall_seconds']:.1f}s "
              f"({metrics['steps'] * STEP_DT / metrics['wall_seconds']:.1f}x realtime)")


if __name__ == "__main__":
    sys.exit(main())

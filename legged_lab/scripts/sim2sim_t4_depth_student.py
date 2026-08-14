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

The policy is the ``DepthStudentTeacher`` head of the rsl-rl checkpoint;
the frozen teacher network is loaded but never queried.

Usage (from the repo root so the vendored packages resolve):

    python -m legged_lab.scripts.sim2sim_t4_depth_student                 # auto-pick latest local checkpoint
    python -m legged_lab.scripts.sim2sim_t4_depth_student \
        --checkpoint path/to/model_1000.pt                                 # interactive, depth preview included
    python -m legged_lab.scripts.sim2sim_t4_depth_student \
        --checkpoint path/to/model_1000.pt --record out.mp4 --duration 20
    python -m legged_lab.scripts.sim2sim_t4_depth_student \
        --checkpoint path/to/model_1000.pt --hurdles      # hurdle course
    python -m legged_lab.scripts.sim2sim_t4_depth_student \
        --checkpoint path/to/model_1000.pt --rule-contract # 100m obstacle course
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import tempfile
from pathlib import Path

import cv2
import mujoco
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
# The vendored rsl_rl lives at <repo>/rsl_rl/rsl_rl; add it so the module
# resolves without installing the fork.
sys.path.insert(0, str(ROOT / "rsl_rl"))

from rsl_rl.modules.depth_student_teacher import DepthStudentTeacher  # noqa: E402

from legged_lab.assets.t4.constants import T4_JOINT_NAMES  # noqa: E402
from legged_lab.assets.t4.schemas import (  # noqa: E402
    DEPTH_CLIP_RANGE,
    DEPTH_HISTORY_LENGTH,
    DEPTH_INVALID_VALUE,
    DEPTH_POLICY_SIZE,
    DEPTH_UPDATE_DECIMATION,
    PROPRIO_FRAME_DIM,
    PROPRIO_HISTORY_LENGTH,
    STUDENT_ACTOR_OBS_DIM,
    TEACHER_ACTOR_OBS_DIM,
)

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
COMMAND_RANGES = {"vx": (-0.6, 1.0), "vy": (-0.5, 0.5), "yaw": (-1.57, 1.57)}
# T4GaitCfg defaults to fixed_clock; standing commands freeze the clock.
GAIT_CYCLE = 0.85
STANDING_COMMAND_THRESHOLD = 0.1
GAIT_AIR_RATIO = np.array([0.38, 0.38])
GAIT_PHASE_OFFSET = np.array([0.38, 0.88])

# D455-style depth camera (d455_depth_config.py).
DEPTH_HFOV_DEG = 87.0
DEPTH_WIDTH, DEPTH_HEIGHT = 480, 270
DEPTH_MAX_RANGE = 15.0  # D455 max_range; farther pixels are no-hit in IsaacLab
DEPTH_FOVY = math.degrees(2 * math.atan(math.tan(math.radians(DEPTH_HFOV_DEG / 2)) * DEPTH_HEIGHT / DEPTH_WIDTH))
DEPTH_CAM_POS = (0.10, 0.0, 0.03)

SPAWN_Z = 0.85


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


def build_model_xml(hurdles: bool, rule_contract: bool) -> str:
    """Return a patched MJCF: camera on Trunk, optional obstacle course, all world geoms collision-active."""
    xml = MJCF.read_text()
    # The patched XML lives in the temp dir, so pin the mesh dir to an
    # absolute path (the original uses a relative `meshdir`).
    meshes_dir = (MJCF.parent / ".." / "meshes").resolve().as_posix()
    xml = xml.replace('meshdir="../meshes/"', f'meshdir="{meshes_dir}/"')
    # The migrated floor declares condim=1 (frictionless); PhysX training
    # terrain is frictional, so restore a frictional ground plane.
    xml = xml.replace('geom name="ground" type="plane" pos="0 0 0" size="0 0 1" material="matplane" condim="1"',
                      'geom name="ground" type="plane" pos="0 0 0" size="0 0 1" material="matplane" condim="3"')
    anchor = '<body name="Trunk"'
    index = xml.index(anchor)
    index = xml.index(">", index) + 1
    # D455CameraCfg offset is pos=(0.10, 0.0, 0.03), rot=(0.707, 0, 0.707, 0)
    # with convention="ros". MuJoCo cameras look along -Z, so this xyaxes
    # choice points the camera forward along +X instead of back into the trunk.
    camera = (
        f'\n      <camera name="depth_cam" pos="{" ".join(map(str, DEPTH_CAM_POS))}" '
        f'xyaxes="0 0 -1 0 -1 0" fovy="{DEPTH_FOVY:.4f}" mode="fixed"/>\n    '
    )
    xml = xml[:index] + camera + xml[index:]

    world_anchor = xml.index("<worldbody>")
    world_end = xml.index("</worldbody>")
    extras = ""
    if rule_contract:
        extras += build_rule_contract_course()
    if hurdles:
        # Stage E hurdle course: 0.3 m tall, 0.4 m deep boxes every 2 m for 20 m.
        for step in range(1, 11):
            x = float(step) * 2.0
            extras += (
                f'\n    <body name="hurdle_{step}" pos="{x} 0 0.15" mocap="true">'
                f'\n      <geom name="hurdle_{step}_geom" type="box" size="0.6 0.02 0.15" density="0"/>\n    </body>'
            )
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

    def __init__(self, checkpoint: str, hurdles: bool, rule_contract: bool):
        xml_path = build_model_xml(hurdles, rule_contract)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
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

        # Policy: reconstruct DepthStudentTeacher and load the rsl-rl state dict.
        self.policy = DepthStudentTeacher(
            num_student_obs=STUDENT_ACTOR_OBS_DIM,
            num_teacher_obs=TEACHER_ACTOR_OBS_DIM,
            num_actions=NUM_JOINTS,
            depth_shape=(DEPTH_HISTORY_LENGTH, *DEPTH_POLICY_SIZE),
            proprio_obs_dim=PROPRIO_FRAME_DIM * PROPRIO_HISTORY_LENGTH,
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or "model_state_dict" not in state:
            raise RuntimeError(f"checkpoint {checkpoint} does not contain model_state_dict")
        self.policy.load_state_dict(state["model_state_dict"])
        self.policy.eval()
        print(f"[INFO] loaded checkpoint {checkpoint} (iter {state.get('iter')})")

        # Depth renderer at the native sensor resolution. Training keeps the
        # depth stream self-contained by disabling visual assets, so mirror the
        # collision-only view here.
        self.renderer = mujoco.Renderer(self.model, height=DEPTH_HEIGHT, width=DEPTH_WIDTH)
        self.renderer.enable_depth_rendering()
        self.depth_option = mujoco.MjvOption()
        self.depth_option.geomgroup = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        self.depth_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "depth_cam")
        self.last_depth_raw = np.full((DEPTH_HEIGHT, DEPTH_WIDTH), DEPTH_INVALID_VALUE, dtype=np.float32)

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
            self.model.dof_damping[self.dof_adr[i]] += self.kd[i]

    def _apply_position_targets(self, targets: np.ndarray) -> None:
        self.data.ctrl[self.actuator_ids] = targets

    def _read_depth(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=self.depth_cam_id, scene_option=self.depth_option)
        depth = self.renderer.render()  # (H, W) planar depth in camera frame; -1 where nothing hit
        # MuJoCo's image x-axis runs opposite to the IsaacLab sensor frame;
        # the server dump shows the ground band on the right, MuJoCo on the
        # left, so mirror horizontally.
        depth = np.ascontiguousarray(depth[:, ::-1])
        # No-hit pixels come back as a large sentinel (not -1) in MuJoCo 3;
        # beyond the D455 max range they must fill the raw invalid value like
        # the IsaacLab stream does (raw 1.0 -> normalized 0.286).
        raw = np.where((depth < 0) | (depth > DEPTH_MAX_RANGE), DEPTH_INVALID_VALUE, depth)
        self.last_depth_raw = raw
        raw = np.clip(raw, DEPTH_CLIP_RANGE[0], DEPTH_CLIP_RANGE[1])
        normalized = (raw - DEPTH_CLIP_RANGE[0]) / (DEPTH_CLIP_RANGE[1] - DEPTH_CLIP_RANGE[0])
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(tensor, size=DEPTH_POLICY_SIZE, mode="area").squeeze(0).squeeze(0)
        return resized.numpy()

    def raw_depth_preview_image(self) -> np.ndarray:
        """Return a color-mapped raw camera preview for debugging."""
        depth = np.nan_to_num(
            self.last_depth_raw,
            nan=DEPTH_INVALID_VALUE,
            posinf=DEPTH_INVALID_VALUE,
            neginf=DEPTH_CLIP_RANGE[0],
        )
        depth = np.clip(depth, DEPTH_CLIP_RANGE[0], DEPTH_CLIP_RANGE[1])
        normalized = (depth - DEPTH_CLIP_RANGE[0]) / (DEPTH_CLIP_RANGE[1] - DEPTH_CLIP_RANGE[0])
        preview = (normalized * 255.0).astype(np.uint8)
        preview = cv2.applyColorMap(preview, cv2.COLORMAP_TURBO)
        cv2.putText(
            preview,
            "depth_cam 0.2-3.0m",
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
            "policy depth input 48x64",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return preview

    def _update_depth_history(self) -> None:
        frame = self._read_depth()
        if self.depth_counter == 0:
            self.depth_history[:] = frame
        elif self.depth_counter % DEPTH_UPDATE_DECIMATION == 0:
            self.depth_history[:-1] = self.depth_history[1:]
            self.depth_history[-1] = frame
        self.depth_counter += 1

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

        obs = np.concatenate([self.proprio_history, self.depth_history.reshape(-1)]).astype(np.float32)
        obs = np.clip(obs, -CLIP_OBS, CLIP_OBS)

        info = {
            "trunk_height": self.data.xpos[self.trunk_id][2],
            "root_lin_vel": self.qvel[:3].copy(),
        }
        return obs, info

    def observe(self) -> np.ndarray:
        """Build the current policy observation without stepping physics."""
        ang_vel_b = self.qvel[3:6]
        gravity_b = quat_rotate_inverse_wxyz(self.qpos[3:7], np.array([0.0, 0.0, -1.0]))
        joint_pos = self.qpos[self.qpos_adr] - STANDING_POS
        joint_vel = self.qvel[self.dof_adr]
        phase = np.zeros(2)
        frame = np.concatenate(
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
        self.proprio_history = np.roll(self.proprio_history, shift=-PROPRIO_FRAME_DIM)
        self.proprio_history[-PROPRIO_FRAME_DIM:] = frame
        self._update_depth_history()
        obs = np.concatenate([self.proprio_history, self.depth_history.reshape(-1)]).astype(np.float32)
        return np.clip(obs, -CLIP_OBS, CLIP_OBS)

    def act(self, obs: np.ndarray) -> None:
        """Run the policy on obs and store the PD targets."""
        with torch.no_grad():
            action = self.policy.act_inference(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
        action = np.clip(action, -CLIP_ACTIONS, CLIP_ACTIONS)
        self.targets = action * ACTION_SCALE + STANDING_POS
        self.previous_action = action.copy()


def interactive_run(sim: DepthStudentSim, duration: float, auto_command: tuple[float, float, float] | None = None) -> dict:
    """Keyboard-commanded run; returns metrics.

    Uses MuJoCo's bundled passive viewer (works on Windows where mujoco-viewer
    has no wheels); commands come through the viewer key callback.
    """
    import time
    import mujoco.viewer

    def adjust(index: int, delta: float) -> None:
        bounds = (COMMAND_RANGES["vx"], COMMAND_RANGES["vy"], COMMAND_RANGES["yaw"])[index]
        sim.command[index] = float(np.clip(sim.command[index] + delta, bounds[0], bounds[1]))

    def reset_view(viewer) -> None:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.fixedcamid = -1
        viewer.cam.trackbodyid = -1
        viewer.cam.distance = 5.8
        viewer.cam.azimuth = 140.0
        viewer.cam.elevation = -20.0
        viewer.cam.lookat[:] = [sim.data.qpos[0], sim.data.qpos[1], 0.82]

    # GLFW key codes. Keep off WASD to avoid collisions with existing bindings.
    moves = {
        73: (0, 0.2),  # I
        75: (0, -0.2),  # K
        85: (1, 0.2),  # U
        79: (1, -0.2),  # O
        74: (2, 0.2),  # J
        76: (2, -0.2),  # L
        88: (0, 0.0),  # X: stop
        32: (0, 0.0),  # Space: stop
        328: (0, 0.2),  # KP_8
        330: (0, -0.2),  # KP_2
        331: (1, 0.2),  # KP_4
        333: (1, -0.2),  # KP_6
        327: (2, 0.2),  # KP_7
        329: (2, -0.2),  # KP_9
        336: (0, 0.0),  # KP_0: stop
    }

    def key_callback(key: int) -> None:
        if key in moves:
            index, delta = moves[key]
            if delta == 0.0 and index == 0:
                sim.command[:] = 0.0
            else:
                adjust(index, delta)

    viewer = mujoco.viewer.launch_passive(
        sim.model, sim.data, key_callback=key_callback, show_left_ui=False, show_right_ui=False
    )
    reset_view(viewer)
    depth_viewport = mujoco.MjrRect(16, 16, 256, 192)
    print("[INFO] controls: I/K vx, U/O vy, J/L yaw, Space/X stop, numpad 8/2/4/6/7/9")
    print("[INFO] depth preview: embedded policy depth input in MuJoCo viewer")

    steps = 0
    start = time.time()
    try:
        if auto_command is not None:
            sim.command[:] = auto_command
        obs = sim.observe()
        sim.act(obs)
        while steps < int(duration / STEP_DT) and viewer.is_running():
            if auto_command is not None:
                sim.command[:] = auto_command
            obs, info = sim.step()
            sim.act(obs)
            reset_view(viewer)
            viewer.set_images((depth_viewport, sim.policy_depth_preview_image()))
            viewer.set_texts(
                (
                    mujoco.mjtFontScale.mjFONTSCALE_150,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    "policy depth: embedded inset",
                    f"cmd={sim.command[0]:.2f},{sim.command[1]:.2f},{sim.command[2]:.2f}",
                ),
            )
            viewer.sync()
            steps += 1
            if steps % 250 == 0:
                print(
                    f"t={steps * STEP_DT:6.1f}s cmd={sim.command} h={info['trunk_height']:.2f} "
                    f"v={info['root_lin_vel'][:2]}"
                )
    finally:
        viewer.clear_images()
        viewer.clear_texts()
        viewer.close()
    return {"steps": steps, "wall_seconds": time.time() - start}


def recorded_run(sim: DepthStudentSim, output: str, duration: float, script: str) -> None:
    """Headless scripted run rendered from the follow camera."""
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
    }
    command_fn = scripts[script]

    writer = imageio.get_writer(output, fps=int(1 / STEP_DT))
    steps = int(duration / STEP_DT)
    sim.command[:] = command_fn(0.0)
    obs = sim.observe()
    sim.act(obs)
    for step in range(steps):
        t = step * STEP_DT
        sim.command[:] = command_fn(t)
        obs, info = sim.step()
        sim.act(obs)
        # Follow camera two metres behind the trunk.
        pos = sim.data.xpos[sim.trunk_id]
        sim.data.mocap_pos[mount_mocap_id] = [pos[0] - 2.0, pos[1], 0.0]
        rgb_renderer.update_scene(sim.data, camera=view_id)
        writer.append_data(rgb_renderer.render())
        if step % 500 == 0:
            print(f"t={t:6.1f}s h={info['trunk_height']:.2f} cmd={sim.command}")
    writer.close()


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
    scene.add_argument("--hurdles", action="store_true", help="Add the Stage E hurdle course")
    scene.add_argument(
        "--rule-contract",
        action="store_true",
        help="Build the conservative 100m obstacle course and auto-forward route",
    )
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--record", type=Path, default=None, help="Write an MP4 and exit instead of opening the GUI")
    parser.add_argument(
        "--script",
        type=str,
        default="forward",
        choices=["forward", "turn", "slalom"],
        help="Command script for --record runs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = resolve_checkpoint(args.checkpoint)
    print(f"[INFO] using checkpoint {checkpoint}")
    sim = DepthStudentSim(str(checkpoint), args.hurdles, args.rule_contract)
    if args.record is not None:
        recorded_run(sim, str(args.record), args.duration, args.script)
    else:
        auto_command = (0.6, 0.0, 0.0) if args.rule_contract else None
        if auto_command is not None:
            print("[INFO] auto route: forward command enabled for rule-contract scene")
        metrics = interactive_run(sim, args.duration, auto_command=auto_command)
        print(f"[DONE] {metrics['steps']} steps in {metrics['wall_seconds']:.1f}s "
              f"({metrics['steps'] * STEP_DT / metrics['wall_seconds']:.1f}x realtime)")


if __name__ == "__main__":
    sys.exit(main())

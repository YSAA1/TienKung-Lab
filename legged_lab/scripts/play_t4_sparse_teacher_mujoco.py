"""MuJoCo sim2sim viewer for sparse-foothold teachers.

The scene is a controlled corridor built from the same width, gap, and height
functions used by the Isaac training terrain:

    start platform -> stepping stones -> transition platform -> raised pillars

T4 contracts:

    - T-compat: 1155D = proprio history + privileged height scan.
    - T-paper: 1157D = T-compat + current left/right foot contact flags.
    - LightLP sparse: 1937D.

G1 LightLP sparse is 1997D (29 joints). Pass ``--robot g1`` or load a 29-DoF
checkpoint; PD/MJCF come from the official velocity plant, not T4.

Z2 LightLP sparse is also 1997D (29 joints) and shares the G1 observation
contract, so a 29-DoF checkpoint alone cannot disambiguate the robot: pass
``--robot z2``. PD/MJCF come from ``Z2_29DOF_WALK_POSE_DAMPED_PD_CFG`` and
``legged_lab/assets/z2/mjcf/assembly.xml``.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch

from legged_lab.assets.t4.constants import T4_JOINT_NAMES
from legged_lab.assets.t4.mujoco_sim2sim import (
    apply_isaac_contact_friction as apply_t4_contact_friction,
    apply_isaac_pd as apply_t4_pd,
    isaac_pd_gains as t4_isaac_pd_gains,
)
from legged_lab.assets.unitree_g1 import mujoco_sim2sim as g1_mujoco
from legged_lab.assets.unitree_g1.constants import (
    G1_29DOF_JOINT_NAMES,
    G1_VELOCITY_JOINT_POS,
    G1_VELOCITY_PELVIS_Z,
)
from legged_lab.assets.z2 import mujoco_sim2sim as z2_mujoco
from legged_lab.assets.z2.constants import (
    Z2_29DOF_JOINT_NAMES,
    Z2_STANDING_JOINT_POS,
    Z2_STANDING_PELVIS_Z,
)
from legged_lab.locomotion.schemas import ObservationLayout
from legged_lab.assets.t4.navigation import CourseNavigator
from legged_lab.assets.t4.schemas import (
    PROPRIO_HISTORY_LENGTH,
    TEACHER_ACTOR_OBS_DIM,
    TEACHER_PAPER_ACTOR_OBS_DIM,
    TEACHER_SCAN_CLIP,
    TEACHER_SCAN_DIM,
    TEACHER_SCAN_FORWARD_RANGE,
    TEACHER_SCAN_HEIGHT_OFFSET,
    TEACHER_SCAN_LATERAL_RANGE,
    TEACHER_SCAN_SHAPE,
    TEACHER_SPARSE_ACTOR_OBS_DIM,
    TEACHER_SPARSE_SCAN_HISTORY_LENGTH,
)
from legged_lab.scripts.play_t4_teacher_viser import (
    ACTION_SCALE,
    ASSET_DIR,
    CLIP_ACTIONS,
    CLIP_OBS,
    DECIMATION,
    GAIT_CYCLE,
    INIT_ROOT_Z,
    MJCF_PATH,
    PHASE_OFFSET,
    PHASE_RATIO,
    PHYSICS_DT,
    STANDING_COMMAND_THRESHOLD,
    T4_STANDING_JOINT_POS,
    _root_yaw_wxyz,
    quat_rotate_inverse_wxyz,
)

G1_ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "unitree_g1"
G1_MJCF_PATH = G1_ASSET_DIR / "xmls" / "g1_actuated.xml"
G1_MESH_DIR = G1_ASSET_DIR / "urdf" / "meshes"
G1_SPARSE_ACTOR_OBS_DIM = ObservationLayout(
    len(G1_29DOF_JOINT_NAMES), scan_history=TEACHER_SPARSE_SCAN_HISTORY_LENGTH, actor_contact=True
).actor_dim
Z2_ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "z2"
Z2_MJCF_PATH = Z2_ASSET_DIR / "mjcf" / "assembly.xml"
Z2_MESH_DIR = Z2_ASSET_DIR / "mjcf" / "meshes"
Z2_SPARSE_ACTOR_OBS_DIM = ObservationLayout(
    len(Z2_29DOF_JOINT_NAMES), scan_history=TEACHER_SPARSE_SCAN_HISTORY_LENGTH, actor_contact=True
).actor_dim


@dataclass(frozen=True)
class SparseMujocoRobot:
    name: str
    joint_names: tuple[str, ...]
    mjcf_path: Path
    meshdir: Path
    meshdir_token: str
    root_body: str
    scan_body: str
    default_joint_pos: dict[str, float]
    init_root_z: float
    sparse_actor_obs_dim: int


T4_SPARSE_ROBOT = SparseMujocoRobot(
    name="t4",
    joint_names=T4_JOINT_NAMES,
    mjcf_path=MJCF_PATH,
    meshdir=ASSET_DIR / "meshes",
    meshdir_token='meshdir="../meshes/"',
    root_body="Trunk",
    scan_body="Trunk",
    default_joint_pos=T4_STANDING_JOINT_POS,
    init_root_z=INIT_ROOT_Z,
    sparse_actor_obs_dim=TEACHER_SPARSE_ACTOR_OBS_DIM,
)
G1_SPARSE_ROBOT = SparseMujocoRobot(
    name="g1",
    joint_names=G1_29DOF_JOINT_NAMES,
    mjcf_path=G1_MJCF_PATH,
    meshdir=G1_MESH_DIR,
    meshdir_token='meshdir="../urdf/meshes"',
    root_body="pelvis",
    scan_body="torso_link",
    default_joint_pos=G1_VELOCITY_JOINT_POS,
    init_root_z=G1_VELOCITY_PELVIS_Z,
    sparse_actor_obs_dim=G1_SPARSE_ACTOR_OBS_DIM,
)
Z2_SPARSE_ROBOT = SparseMujocoRobot(
    name="z2",
    joint_names=Z2_29DOF_JOINT_NAMES,
    mjcf_path=Z2_MJCF_PATH,
    meshdir=Z2_MESH_DIR,
    meshdir_token='meshdir="meshes/"',
    root_body="Z2_0_Lite_description_0429_1",
    scan_body="waist_roll_link",
    default_joint_pos=Z2_STANDING_JOINT_POS,
    init_root_z=Z2_STANDING_PELVIS_Z,
    sparse_actor_obs_dim=Z2_SPARSE_ACTOR_OBS_DIM,
)

_LAYOUT_PATH = (
    Path(__file__).resolve().parents[1] / "terrains" / "stepping_stone_layout.py"
)
_LAYOUT_SPEC = importlib.util.spec_from_file_location("t4_sparse_layout", _LAYOUT_PATH)
assert _LAYOUT_SPEC and _LAYOUT_SPEC.loader
_LAYOUT = importlib.util.module_from_spec(_LAYOUT_SPEC)
_LAYOUT_SPEC.loader.exec_module(_LAYOUT)

PIT_FLOOR_Z = -2.05
PIT_FLOOR_TOP_Z = -2.0
PLATFORM_THICKNESS = 0.10
# The Isaac terrain is an 8 m square lattice around the spawn platform, not a
# three-lane corridor.  Seven lanes preserve enough of that lateral support for
# the teacher's natural sim2sim drift while keeping the interactive scene small.
LANE_COUNT = 7
FOOTHOLD_ROWS = 10
# G1 local playback: Isaac tiles are 8 m; keep a wider/longer lattice so a
# 0.7 m/s walk can be watched for tens of seconds without walking off the edge.
G1_LANE_COUNT = 15
G1_FOOTHOLD_ROWS = 40
LOCO_SPARSE_START_X = 41.0


def supported_actor_obs_dim(num_obs: int) -> bool:
    return num_obs in (
        TEACHER_ACTOR_OBS_DIM,
        TEACHER_PAPER_ACTOR_OBS_DIM,
        TEACHER_SPARSE_ACTOR_OBS_DIM,
        G1_SPARSE_ACTOR_OBS_DIM,
    )


def teacher_scan_local_points() -> np.ndarray:
    """IsaacLab GridPatternCfg(ordering="xy"): y outer, x inner."""
    xs = np.linspace(*TEACHER_SCAN_FORWARD_RANGE, TEACHER_SCAN_SHAPE[0])
    ys = np.linspace(*TEACHER_SCAN_LATERAL_RANGE, TEACHER_SCAN_SHAPE[1])
    return np.asarray([(x, y) for y in ys for x in xs], dtype=np.float64)


def _platform(name: str, x_center: float, length: float, width: float = 2.0) -> dict:
    return {
        "name": name,
        "kind": "platform",
        "shape": "box",
        "pos": (x_center, 0.0, -0.5 * PLATFORM_THICKNESS),
        "size": (0.5 * length, 0.5 * width, 0.5 * PLATFORM_THICKNESS),
        "rgba": (0.28, 0.31, 0.34, 1.0),
    }


def _lattice_platform_width(lane_count: int, pitch: float, foothold_size: float) -> float:
    span = (lane_count - 1) * pitch + foothold_size + 0.8
    return max(_LAYOUT.LIGHTLP_STONE_PLATFORM_WIDTH, span)


def _geom_xy_extent(geom: dict) -> tuple[float, float, float, float]:
    x, y = float(geom["pos"][0]), float(geom["pos"][1])
    if geom["shape"] == "box":
        hx, hy = float(geom["size"][0]), float(geom["size"][1])
    else:
        hx = hy = float(geom["size"][0])
    return x - hx, x + hx, y - hy, y + hy


def _pit_xml(layout: dict, name: str = "sparse_pit_floor") -> str:
    extents = [_geom_xy_extent(geom) for geom in layout["geoms"]]
    x0 = min(item[0] for item in extents) - 2.0
    x1 = max(item[1] for item in extents) + 2.0
    y0 = min(item[2] for item in extents) - 2.0
    y1 = max(item[3] for item in extents) + 2.0
    return (
        f'<geom name="{name}" type="box" pos="{0.5 * (x0 + x1):.8g} {0.5 * (y0 + y1):.8g} {PIT_FLOOR_Z}" '
        f'size="{0.5 * (x1 - x0):.8g} {0.5 * (y1 - y0):.8g} 0.05" '
        f'rgba="0.22 0.24 0.27 1" condim="3" friction="1 0.005 0.0001" group="3"/>'
    )


def sparse_course_layout(
    difficulty: float,
    terrain: str = "sparse_course",
    geometry: str = "current",
    lane_count: int = LANE_COUNT,
    foothold_rows: int = FOOTHOLD_ROWS,
) -> dict:
    """Return a deterministic sparse corridor using the Isaac training dimensions."""
    if terrain not in {"sparse_course", "stepping_stones", "raised_pillars"}:
        raise ValueError(f"unsupported sparse terrain {terrain!r}")
    if lane_count < 1 or lane_count % 2 == 0:
        raise ValueError(f"lane_count must be a positive odd integer, got {lane_count}")
    if foothold_rows < 1:
        raise ValueError(f"foothold_rows must be positive, got {foothold_rows}")
    if geometry == "current":
        stone_width = _LAYOUT.stone_width(difficulty)
        stone_gap = _LAYOUT.stone_gap(difficulty)
        stone_pitch = _LAYOUT.foothold_pitch(difficulty)
        stone_height = _LAYOUT.stone_height(difficulty)
        stone_jitter = _LAYOUT.stone_height_jitter(difficulty)
        pillar_diameter = _LAYOUT.pillar_diameter(difficulty)
        pillar_gap = _LAYOUT.pillar_gap(difficulty)
        pillar_pitch = _LAYOUT.foothold_pitch(difficulty, _LAYOUT.LIGHTLP_PILLAR_PITCH_RANGE)
        pillar_height = _LAYOUT.pillar_height(difficulty)
        stone_leading_gap = _LAYOUT.platform_to_first_gap(difficulty, stone_width)
        pillar_leading_gap = _LAYOUT.platform_to_first_gap(
            difficulty, pillar_diameter, pitch_range=_LAYOUT.LIGHTLP_PILLAR_PITCH_RANGE
        )
    elif geometry == "checkpoint_24999":
        d = min(max(float(difficulty), 0.0), 1.0)
        stone_width = 0.45 + d * (0.30 - 0.45)
        stone_gap = 0.08 + d * (0.20 - 0.08)
        stone_pitch = stone_width + stone_gap
        stone_height = 0.36
        stone_jitter = d * 0.06
        pillar_diameter = 0.40 + d * (0.32 - 0.40)
        pillar_gap = 0.08 + d * (0.18 - 0.08)
        pillar_pitch = pillar_diameter + pillar_gap
        pillar_height = 0.30 + d * (0.42 - 0.30)
        stone_leading_gap = 0.0
        pillar_leading_gap = 0.0
    else:
        raise ValueError(f"unsupported sparse geometry {geometry!r}")

    if terrain == "raised_pillars":
        platform_width = _lattice_platform_width(lane_count, pillar_pitch, pillar_diameter)
    elif terrain == "stepping_stones":
        platform_width = _lattice_platform_width(lane_count, stone_pitch, stone_width)
    else:
        platform_width = max(
            _lattice_platform_width(lane_count, stone_pitch, stone_width),
            _lattice_platform_width(lane_count, pillar_pitch, pillar_diameter),
        )
    geoms = [_platform("start_platform", 0.0, 2.0, width=platform_width)]
    cursor = 1.0
    rng = np.random.default_rng(int(round(float(difficulty) * 1000.0)))

    if terrain in {"sparse_course", "stepping_stones"}:
        first_center = cursor + stone_leading_gap + 0.5 * stone_width
        lane_y = [(lane - lane_count // 2) * stone_pitch for lane in range(lane_count)]
        for row in range(foothold_rows):
            x = first_center + row * stone_pitch
            for lane, y in enumerate(lane_y):
                height = max(
                    0.04, stone_height + float(rng.uniform(-stone_jitter, stone_jitter))
                )
                geoms.append(
                    {
                        "name": f"stone_{row}_{lane}",
                        "kind": "stone",
                        "shape": "box",
                        "pos": (x, y, 0.5 * height),
                        "size": (0.5 * stone_width, 0.5 * stone_width, 0.5 * height),
                        "rgba": (0.72, 0.63, 0.42, 1.0),
                    }
                )
        cursor = (
            first_center + (foothold_rows - 1) * stone_pitch + 0.5 * stone_width + 0.30
        )
        if terrain == "sparse_course":
            geoms.append(
                _platform("transition_platform", cursor + 0.70, 1.40, width=platform_width)
            )
            cursor += 1.55

    if terrain in {"sparse_course", "raised_pillars"}:
        first_center = cursor + pillar_leading_gap + 0.5 * pillar_diameter
        lane_y = [(lane - lane_count // 2) * pillar_pitch for lane in range(lane_count)]
        for row in range(foothold_rows):
            x = first_center + row * pillar_pitch
            for lane, y in enumerate(lane_y):
                geoms.append(
                    {
                        "name": f"pillar_{row}_{lane}",
                        "kind": "pillar",
                        "shape": "cylinder",
                        "pos": (x, y, 0.5 * pillar_height),
                        "size": (0.5 * pillar_diameter, 0.5 * pillar_height),
                        "rgba": (0.35, 0.58, 0.72, 1.0),
                    }
                )
        cursor = (
            first_center
            + (foothold_rows - 1) * pillar_pitch
            + 0.5 * pillar_diameter
            + 0.30
        )

    geoms.append(_platform("finish_platform", cursor + 1.0, 2.0, width=platform_width))
    return {
        "difficulty": float(difficulty),
        "terrain": terrain,
        "geometry": geometry,
        "lane_count": lane_count,
        "foothold_rows": foothold_rows,
        "stone_width": stone_width,
        "stone_gap": stone_gap,
        "stone_height": stone_height,
        "pillar_diameter": pillar_diameter,
        "pillar_gap": pillar_gap,
        "pillar_height": pillar_height,
        "geoms": geoms,
    }


def loco_sparse_layout(
    difficulty: float,
    geometry: str = "current",
    lane_count: int = LANE_COUNT,
    foothold_rows: int = FOOTHOLD_ROWS,
) -> dict:
    """Append the sparse course to the reusable Stage-E continuous-locomotion lane.

    The existing lane finishes on a finite flat section at ``x=40``.  The
    sparse course starts immediately after it, preserving the current
    stepping-stone and raised-pillar dimensions while keeping their real holes.
    """
    layout = sparse_course_layout(
        difficulty,
        terrain="sparse_course",
        geometry=geometry,
        lane_count=lane_count,
        foothold_rows=foothold_rows,
    )
    geoms = []
    for geom in layout["geoms"]:
        shifted = dict(geom)
        shifted["name"] = f"loco_sparse_{geom['name']}"
        shifted["pos"] = (float(geom["pos"][0]) + LOCO_SPARSE_START_X, *geom["pos"][1:])
        if geom["name"] == "start_platform":
            # The preceding loco corridor is 2.2 m wide.  Preserve that width
            # for the entry platform so its lateral state can transfer to the
            # sparse lattice instead of falling through an artificial 30 cm
            # step-in narrowing at the ground/pit boundary.
            shifted["name"] = "loco_sparse_alignment_platform"
            shifted["size"] = (geom["size"][0], 1.1, geom["size"][2])
        geoms.append(shifted)
    return {
        **layout,
        "terrain": "loco_sparse",
        "sparse_start_x": LOCO_SPARSE_START_X,
        "geoms": geoms,
    }


def loco_sparse_navigation_waypoints(layout: dict) -> np.ndarray:
    """Return a collision-aware centerline for the local whole-course scene.

    Stage E leaves its non-goal marker at x=38, so the centerline briefly
    passes on its right before returning to the aligned sparse platform.  The
    remaining points select the middle lane of each local foothold lattice.
    """
    points: list[tuple[float, float]] = [
        (0.0, 0.0),
        (35.8, 0.0),
        (37.1, -0.7),
        (39.0, -0.7),
        (40.3, 0.0),
    ]
    centerline = [
        geom
        for geom in layout["geoms"]
        if geom["kind"] in {"platform", "stone", "pillar"}
        and abs(float(geom["pos"][1])) < 1e-6
        and geom["pos"][0] >= LOCO_SPARSE_START_X
    ]
    for geom in sorted(centerline, key=lambda item: float(item["pos"][0])):
        point = (float(geom["pos"][0]), float(geom["pos"][1]))
        if point != points[-1]:
            points.append(point)
    return np.asarray(points, dtype=np.float64)


def _rewrite_meshdir(xml: str, robot: SparseMujocoRobot) -> str:
    token = robot.meshdir_token
    rewritten = xml.replace(token, f'meshdir="{robot.meshdir.as_posix()}"')
    if rewritten == xml:
        raise RuntimeError(f"failed to rewrite meshdir token {token!r} in {robot.mjcf_path}")
    return rewritten


def _install_world_geoms(xml: str, extra: str) -> str:
    if "</worldbody>" not in xml:
        raise RuntimeError("MJCF is missing </worldbody>")
    return xml.replace("</worldbody>", f"{extra}\n  </worldbody>", 1)


def _replace_or_insert_ground(xml: str, ground: str) -> str:
    xml, replacements = re.subn(r'<geom name="ground"[^>]*/>', ground, xml, count=1)
    if replacements == 1:
        return xml
    return _install_world_geoms(xml, ground)


def _decorate_viewer_scene(xml: str) -> str:
    """G1/Z2 MJCFs have no skybox or world lights; the viewer would be black."""
    if 'type="skybox"' not in xml:
        xml = xml.replace(
            "<asset>",
            """<asset>
    <texture type="skybox" builtin="gradient" rgb1="0.85 0.58 0.55" rgb2="0.18 0.10 0.10" width="512" height="3072"/>
    <texture name="texplane" type="2d" builtin="checker" rgb1=".25 .32 .40" rgb2=".12 .16 .20" width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
    <material name="matplane" reflectance="0.25" texture="texplane" texrepeat="1 1" texuniform="true"/>""",
            1,
        )
    if 'directional="true"' not in xml:
        xml = xml.replace(
            "<worldbody>",
            """<worldbody>
    <light directional="true" diffuse=".45 .45 .45" specular="0.1 0.1 0.1" pos="0 0 8" dir="0 0 -1" castshadow="false"/>
    <light directional="true" diffuse=".65 .65 .65" specular="0.15 0.15 0.15" pos="2 3 6" dir="-0.2 -0.3 -1"/>""",
            1,
        )
    return xml


def _geom_xml(geom: dict) -> str:
    pos = " ".join(f"{value:.8g}" for value in geom["pos"])
    size = " ".join(f"{value:.8g}" for value in geom["size"])
    rgba = " ".join(f"{value:.8g}" for value in geom["rgba"])
    return (
        f'<geom name="{geom["name"]}" type="{geom["shape"]}" pos="{pos}" size="{size}" '
        f'rgba="{rgba}" condim="3" friction="1 0.005 0.0001" group="3"/>'
    )


def build_sparse_model_xml(
    difficulty: float,
    terrain: str = "sparse_course",
    geometry: str = "current",
    lane_count: int = LANE_COUNT,
    foothold_rows: int = FOOTHOLD_ROWS,
    robot: SparseMujocoRobot = T4_SPARSE_ROBOT,
) -> str:
    xml = _rewrite_meshdir(robot.mjcf_path.read_text(encoding="utf-8"), robot)
    if robot.name in ("g1", "z2"):
        xml = _decorate_viewer_scene(xml)
    if terrain == "flat":
        ground = (
            '<geom name="ground" type="plane" pos="0 0 0" size="0 0 1" '
            'material="matplane" condim="3" friction="1 0.005 0.0001"/>'
        )
        return _replace_or_insert_ground(xml, ground)
    if terrain == "loco_sparse":
        if robot.name != "t4":
            raise ValueError("loco_sparse is the T4 Stage-E course; use T4 or a sparse G1 terrain")
        from legged_lab.scripts.sim2sim_t4_depth_student import build_loco_course

        layout = loco_sparse_layout(
            difficulty,
            geometry=geometry,
            lane_count=lane_count,
            foothold_rows=foothold_rows,
        )
        ground = (
            '<geom name="ground" type="box" pos="12 0 -0.05" size="28 10 0.05" '
            'material="matplane" condim="3" friction="1 0.005 0.0001"/>'
        )
        xml, replacements = re.subn(r'<geom name="ground"[^>]*/>', ground, xml, count=1)
        if replacements != 1:
            raise RuntimeError("failed to replace the stock MuJoCo ground plane")
        pit = _pit_xml(layout, name="loco_sparse_pit_floor")
        sparse_xml = "\n    ".join(_geom_xml(geom) for geom in layout["geoms"])
        return xml.replace(
            "</worldbody>",
            f"{build_loco_course()}\n    {pit}\n    {sparse_xml}\n  </worldbody>",
        )

    layout = sparse_course_layout(
        difficulty,
        terrain=terrain,
        geometry=geometry,
        lane_count=lane_count,
        foothold_rows=foothold_rows,
    )
    pit = _pit_xml(layout)
    xml = _replace_or_insert_ground(xml, pit)
    terrain_xml = "\n    ".join(_geom_xml(geom) for geom in layout["geoms"])
    return _install_world_geoms(xml, f"    {terrain_xml}")


def load_actor(checkpoint_path: str) -> tuple[torch.nn.Module, int, int]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rsl_rl"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    actor_state = {
        key[len("actor.") :]: value
        for key, value in state.items()
        if key.startswith("actor.")
    }
    num_obs = int(actor_state["0.weight"].shape[1])
    weight_keys = [key for key in actor_state if key.endswith(".weight") and key.split(".", 1)[0].isdigit()]
    last_weight = max(weight_keys, key=lambda key: int(key.split(".", 1)[0]))
    num_actions = int(actor_state[last_weight].shape[0])
    if not supported_actor_obs_dim(num_obs):
        raise RuntimeError(
            f"checkpoint actor expects {num_obs} observations; supported contracts are "
            f"{TEACHER_ACTOR_OBS_DIM}, {TEACHER_PAPER_ACTOR_OBS_DIM}, "
            f"{TEACHER_SPARSE_ACTOR_OBS_DIM}, and {G1_SPARSE_ACTOR_OBS_DIM}"
        )
    dims = [num_obs, 512, 256, 128, num_actions]
    layers: list[torch.nn.Module] = []
    for index in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[index], dims[index + 1]))
        if index < len(dims) - 2:
            layers.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*layers)
    actor.load_state_dict(actor_state)
    actor.eval()
    return actor, num_obs, num_actions


def resolve_sparse_robot(num_obs: int, num_actions: int, requested: str | None) -> SparseMujocoRobot:
    if num_actions == len(G1_29DOF_JOINT_NAMES):
        # G1 and Z2 share the 1997D/29-action contract; the checkpoint alone
        # cannot disambiguate. Keep G1 as the implicit default for backward
        # compatibility and require an explicit --robot z2 for Z2 plants.
        inferred = Z2_SPARSE_ROBOT if requested == "z2" else G1_SPARSE_ROBOT
        expected_dim = inferred.sparse_actor_obs_dim
        if num_obs != expected_dim:
            raise RuntimeError(
                f"{inferred.name} actor has {num_obs} observations; expected LightLP sparse {expected_dim}"
            )
        if requested not in (None, "g1", "z2"):
            raise RuntimeError(f"--robot {requested} does not match a 29-DoF checkpoint")
    elif num_actions == len(T4_JOINT_NAMES):
        inferred = T4_SPARSE_ROBOT
        if requested is not None and requested != inferred.name:
            raise RuntimeError(
                f"--robot {requested} does not match checkpoint robot {inferred.name} "
                f"(obs={num_obs}, actions={num_actions})"
            )
    else:
        raise RuntimeError(f"unsupported actor action width {num_actions}")
    return inferred


class T4SparseTeacherMujocoRunner:
    def __init__(
        self,
        checkpoint: str,
        difficulty: float = 0.5,
        terrain: str = "sparse_course",
        geometry: str = "current",
        lane_count: int | None = None,
        foothold_rows: int | None = None,
        nav: bool = False,
        cruise: float = 0.6,
        robot: str | None = None,
    ):
        self.actor, self.actor_obs_dim, num_actions = load_actor(checkpoint)
        self.robot = resolve_sparse_robot(self.actor_obs_dim, num_actions, robot)
        wide_lattice = self.robot.name in ("g1", "z2")
        if lane_count is None:
            lane_count = G1_LANE_COUNT if wide_lattice else LANE_COUNT
        if foothold_rows is None:
            foothold_rows = G1_FOOTHOLD_ROWS if wide_lattice else FOOTHOLD_ROWS
        self.lane_count = lane_count
        self.foothold_rows = foothold_rows
        self.paper_contact_obs = self.actor_obs_dim in (
            TEACHER_PAPER_ACTOR_OBS_DIM,
            TEACHER_SPARSE_ACTOR_OBS_DIM,
            G1_SPARSE_ACTOR_OBS_DIM,
            Z2_SPARSE_ACTOR_OBS_DIM,
        )
        self.sparse_scan_history = self.actor_obs_dim in (
            TEACHER_SPARSE_ACTOR_OBS_DIM,
            G1_SPARSE_ACTOR_OBS_DIM,
            Z2_SPARSE_ACTOR_OBS_DIM,
        )
        self.scan_history_length = (
            TEACHER_SPARSE_SCAN_HISTORY_LENGTH if self.sparse_scan_history else 1
        )
        self.obs_layout = ObservationLayout(
            len(self.robot.joint_names),
            scan_history=self.scan_history_length,
            actor_contact=self.paper_contact_obs,
        )
        if terrain == "flat":
            self.layout = {
                "difficulty": float(difficulty),
                "terrain": "flat",
                "geometry": geometry,
                "lane_count": 0,
                "foothold_rows": 0,
                "stone_width": 0.0,
                "stone_gap": 0.0,
                "stone_height": 0.0,
                "pillar_diameter": 0.0,
                "pillar_gap": 0.0,
                "pillar_height": 0.0,
                "geoms": [],
            }
        elif terrain == "loco_sparse":
            self.layout = loco_sparse_layout(
                difficulty,
                geometry=geometry,
                lane_count=lane_count,
                foothold_rows=foothold_rows,
            )
        else:
            self.layout = sparse_course_layout(
                difficulty,
                terrain=terrain,
                geometry=geometry,
                lane_count=lane_count,
                foothold_rows=foothold_rows,
            )
        self.model = mujoco.MjModel.from_xml_string(
            build_sparse_model_xml(
                difficulty,
                terrain=terrain,
                geometry=geometry,
                lane_count=lane_count,
                foothold_rows=foothold_rows,
                robot=self.robot,
            )
        )
        self.model.opt.timestep = PHYSICS_DT
        if self.robot.name in ("g1", "z2"):
            if self.robot.name == "g1":
                g1_mujoco.apply_isaac_pd(self.model)
                g1_mujoco.apply_isaac_contact_friction(self.model)
                gains = np.array([g1_mujoco.isaac_pd_gains(name) for name in self.robot.joint_names])
            else:
                z2_mujoco.apply_isaac_pd(self.model)
                z2_mujoco.apply_isaac_contact_friction(self.model)
                gains = np.array([z2_mujoco.isaac_pd_gains(name) for name in self.robot.joint_names])
            self.ctrl_ids = np.array([self.model.actuator(name).id for name in self.robot.joint_names])
        else:
            apply_t4_pd(self.model)
            apply_t4_contact_friction(self.model)
            gains = np.array([t4_isaac_pd_gains(name) for name in self.robot.joint_names])
            self.ctrl_ids = np.array(
                [self.model.actuator(f"M{name[1:]}").id for name in self.robot.joint_names]
            )
        self.data = mujoco.MjData(self.model)

        self.qpos_adr = np.array(
            [self.model.jnt_qposadr[self.model.joint(name).id] for name in self.robot.joint_names]
        )
        self.qvel_adr = np.array(
            [self.model.jnt_dofadr[self.model.joint(name).id] for name in self.robot.joint_names]
        )
        self.kp, self.kd, self.effort_limit = gains[:, 0], gains[:, 1], gains[:, 2]
        self.default_pos = np.array(
            [self.robot.default_joint_pos[name] for name in self.robot.joint_names]
        )
        self._ray_geomgroup = np.ones(6, dtype=np.uint8)
        self._robot_body_id = self.model.body(self.robot.root_body).id
        self._foot_geom_side: dict[int, int] = {}
        if self.robot.name == "z2":
            # Z2 foot collision cylinders are unnamed; map geoms by their body.
            foot_bodies = {
                int(self.model.body("L_ankle_roll_link").id): 0,
                int(self.model.body("R_ankle_roll_link").id): 1,
            }
            for geom_id in range(self.model.ngeom):
                side = foot_bodies.get(int(self.model.geom_bodyid[geom_id]))
                if side is not None:
                    self._foot_geom_side[geom_id] = side
        else:
            for geom_id in range(self.model.ngeom):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
                if name.startswith("left_foot") and name.endswith("_collision"):
                    self._foot_geom_side[geom_id] = 0
                elif name.startswith("right_foot") and name.endswith("_collision"):
                    self._foot_geom_side[geom_id] = 1

        self.command = np.zeros(3, dtype=np.float32)
        self.navigator = (
            CourseNavigator(loco_sparse_navigation_waypoints(self.layout), cruise_vx=cruise)
            if terrain == "loco_sparse" and nav
            else None
        )
        self.reset_count = 0
        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = [0.0, 0.0, self.robot.init_root_z]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[self.qpos_adr] = self.default_pos
        mujoco.mj_forward(self.model, self.data)
        if self.navigator is not None:
            self.navigator.reset()
            self.command[:] = self.navigator.command(
                self.data.qpos[:2], _root_yaw_wxyz(self.data.qpos[3:7])
            )
        self.action = np.zeros(len(self.robot.joint_names), dtype=np.float32)
        self.gait_time = 0.0
        self.gait_phase = PHASE_OFFSET % 1.0
        frame = self._proprio_frame()
        self.history: deque[np.ndarray] = deque(
            [frame.copy() for _ in range(PROPRIO_HISTORY_LENGTH)],
            maxlen=PROPRIO_HISTORY_LENGTH,
        )
        scan = self._height_scan()
        self.scan_history: deque[np.ndarray] = deque(
            [scan.copy() for _ in range(self.scan_history_length)],
            maxlen=self.scan_history_length,
        )
        self.reset_count += 1

    def _proprio_frame(self) -> np.ndarray:
        quat = self.data.qpos[3:7]
        projected_gravity = quat_rotate_inverse_wxyz(quat, np.array([0.0, 0.0, -1.0]))
        frame = np.concatenate(
            [
                self.data.qvel[3:6],
                projected_gravity,
                self.command,
                self.data.qpos[self.qpos_adr] - self.default_pos,
                self.data.qvel[self.qvel_adr],
                self.action,
                np.sin(2.0 * np.pi * self.gait_phase),
                np.cos(2.0 * np.pi * self.gait_phase),
                PHASE_RATIO,
            ]
        ).astype(np.float32)
        if frame.shape[0] != self.obs_layout.proprio_dim:
            raise RuntimeError(
                f"proprio frame width {frame.shape[0]} != {self.obs_layout.proprio_dim}"
            )
        return frame

    def _height_scan(self) -> np.ndarray:
        if self.robot.name in ("g1", "z2"):
            baseline = self.data.xpos[self.model.body(self.robot.scan_body).id, 2] - TEACHER_SCAN_HEIGHT_OFFSET
        else:
            baseline = self.data.qpos[2] - TEACHER_SCAN_HEIGHT_OFFSET
        yaw = _root_yaw_wxyz(self.data.qpos[3:7])
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        values = []
        for x, y in teacher_scan_local_points():
            world_x = self.data.qpos[0] + cos_yaw * x - sin_yaw * y
            world_y = self.data.qpos[1] + sin_yaw * x + cos_yaw * y
            geom_id = np.zeros(1, dtype=np.int32)
            distance = mujoco.mj_ray(
                self.model,
                self.data,
                np.array([world_x, world_y, 5.0]),
                np.array([0.0, 0.0, -1.0]),
                self._ray_geomgroup,
                1,
                self._robot_body_id,
                geom_id,
            )
            terrain_z = PIT_FLOOR_TOP_Z if distance < 0.0 else 5.0 - distance
            values.append(np.clip(baseline - terrain_z, *TEACHER_SCAN_CLIP))
        scan = np.asarray(values, dtype=np.float32)
        if scan.shape[0] != TEACHER_SCAN_DIM:
            raise RuntimeError(
                f"height scan width {scan.shape[0]} != {TEACHER_SCAN_DIM}"
            )
        return scan

    def _feet_contact(self) -> np.ndarray:
        contact = np.zeros(2, dtype=np.float32)
        force = np.zeros(6, dtype=np.float64)
        for index in range(self.data.ncon):
            item = self.data.contact[index]
            sides = {
                self._foot_geom_side.get(int(item.geom1)),
                self._foot_geom_side.get(int(item.geom2)),
            }
            sides.discard(None)
            if not sides:
                continue
            mujoco.mj_contactForce(self.model, self.data, index, force)
            if np.linalg.norm(force[:3]) > 0.5:
                for side in sides:
                    contact[side] = 1.0
        return contact

    def observe(self) -> np.ndarray:
        self.history.append(self._proprio_frame())
        self.scan_history.append(self._height_scan())
        parts = [*self.history, *self.scan_history]
        if self.paper_contact_obs:
            parts.append(self._feet_contact())
        obs = np.concatenate(parts)
        if obs.shape[0] != self.actor_obs_dim:
            raise RuntimeError(
                f"actor observation width {obs.shape[0]} != checkpoint width {self.actor_obs_dim}"
            )
        return np.clip(obs, -CLIP_OBS, CLIP_OBS)

    def step(self) -> None:
        if self.navigator is not None:
            self.command[:] = self.navigator.command(
                self.data.qpos[:2], _root_yaw_wxyz(self.data.qpos[3:7])
            )
        obs = self.observe()
        with torch.no_grad():
            action = self.actor(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
        self.action = np.clip(action, -CLIP_ACTIONS, CLIP_ACTIONS).astype(np.float32)
        self.data.ctrl[self.ctrl_ids] = self.default_pos + ACTION_SCALE * self.action
        for _ in range(DECIMATION):
            mujoco.mj_step(self.model, self.data)
        if np.linalg.norm(self.command[:2]) > STANDING_COMMAND_THRESHOLD:
            self.gait_time += (PHYSICS_DT * DECIMATION) / GAIT_CYCLE
        self.gait_phase = (self.gait_time + PHASE_OFFSET) % 1.0

    @property
    def fallen(self) -> bool:
        gravity_z = quat_rotate_inverse_wxyz(
            self.data.qpos[3:7], np.array([0.0, 0.0, -1.0])
        )[2]
        return self.data.qpos[2] < 0.35 or gravity_z > -0.5


def run_native_viewer(runner: T4SparseTeacherMujocoRunner, vx: float) -> None:
    import mujoco.viewer

    if runner.navigator is None:
        runner.command[:] = [vx, 0.0, 0.0]
    step_dt = PHYSICS_DT * DECIMATION
    with mujoco.viewer.launch_passive(runner.model, runner.data) as viewer:
        viewer.opt.geomgroup[:] = 0
        viewer.opt.geomgroup[0] = 1
        viewer.opt.geomgroup[1] = 1
        viewer.opt.geomgroup[2] = 1
        # course terrain lives in the depth terrain group; keep it visible live
        viewer.opt.geomgroup[3] = 1
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 145.0
        viewer.cam.elevation = -18.0
        while viewer.is_running():
            start = time.time()
            runner.step()
            if runner.fallen:
                runner.reset()
                if runner.navigator is None:
                    runner.command[:] = [vx, 0.0, 0.0]
            viewer.cam.lookat[:] = [
                runner.data.qpos[0] + 0.8,
                runner.data.qpos[1],
                0.45,
            ]
            viewer.sync()
            remaining = step_dt - (time.time() - start)
            if remaining > 0.0:
                time.sleep(remaining)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--terrain",
        choices=("flat", "sparse_course", "stepping_stones", "raised_pillars", "loco_sparse"),
        default="sparse_course",
    )
    parser.add_argument("--difficulty", type=float, default=0.5)
    parser.add_argument(
        "--geometry",
        choices=("current", "checkpoint_24999"),
        default="current",
        help="Pin geometry to the current curriculum or the model_24999 training lineage.",
    )
    parser.add_argument("--vx", type=float, default=0.6)
    parser.add_argument(
        "--robot",
        choices=("t4", "g1", "z2"),
        default=None,
        help="Optional. Inferred from the checkpoint action width when omitted, "
        "except Z2 which shares the 29-DoF contract with G1 and needs --robot z2.",
    )
    parser.add_argument(
        "--nav",
        action="store_true",
        help="For loco_sparse only, follow the whole-course centerline with [vx, 0, wz] commands.",
    )
    parser.add_argument(
        "--cruise",
        type=float,
        default=0.6,
        help="Requested forward speed for --nav; it is clipped to the frozen vx command range.",
    )
    parser.add_argument(
        "--lane-count",
        type=int,
        default=None,
        help="Odd number of lateral foothold lanes. Default 7 for T4, 15 for G1/Z2.",
    )
    parser.add_argument(
        "--foothold-rows",
        type=int,
        default=None,
        help="Longitudinal foothold rows. Default 10 for T4, 40 for G1/Z2.",
    )
    args = parser.parse_args()
    if args.nav and args.terrain != "loco_sparse":
        parser.error("--nav is available only with --terrain loco_sparse")
    if args.cruise < 0.0:
        parser.error("--cruise must be non-negative")

    runner = T4SparseTeacherMujocoRunner(
        args.checkpoint,
        difficulty=args.difficulty,
        terrain=args.terrain,
        geometry=args.geometry,
        lane_count=args.lane_count,
        foothold_rows=args.foothold_rows,
        nav=args.nav,
        cruise=args.cruise,
        robot=args.robot,
    )
    layout = runner.layout
    if runner.actor_obs_dim in (G1_SPARSE_ACTOR_OBS_DIM, Z2_SPARSE_ACTOR_OBS_DIM):
        contract = f"{runner.robot.name.upper()} LightLP sparse {runner.actor_obs_dim}D"
    elif runner.sparse_scan_history:
        contract = f"LightLP sparse {TEACHER_SPARSE_ACTOR_OBS_DIM}D"
    elif runner.paper_contact_obs:
        contract = "T-paper 1157D"
    else:
        contract = "T-compat 1155D"
    print(
        f"[INFO] loaded {contract}; robot={runner.robot.name}; terrain={args.terrain}; geometry={args.geometry}; "
        f"difficulty={args.difficulty:.2f}; scene={layout['lane_count']} lanes x "
        f"{layout['foothold_rows']} rows; stone={layout['stone_width']:.3f}m "
        f"gap={layout['stone_gap']:.3f}m height={layout['stone_height']:.3f}m; "
        f"pillar={layout['pillar_diameter']:.3f}m gap={layout['pillar_gap']:.3f}m "
        f"height={layout['pillar_height']:.3f}m",
        flush=True,
    )
    if args.nav:
        goal = runner.navigator.waypoints[-1]
        print(
            f"[INFO] route navigation ON; cruise={args.cruise:.2f}; goal=({goal[0]:.1f}, {goal[1]:.1f})",
            flush=True,
        )
    run_native_viewer(runner, args.vx)


if __name__ == "__main__":
    main()

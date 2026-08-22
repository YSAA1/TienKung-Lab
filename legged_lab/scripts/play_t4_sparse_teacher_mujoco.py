"""MuJoCo sim2sim viewer for the two T4 sparse-foothold teachers.

The scene is a controlled corridor built from the same width, gap, and height
functions used by the Isaac training terrain:

    start platform -> stepping stones -> transition platform -> raised pillars

Both sparse teacher observation contracts are supported automatically:

- T-compat: 1155D = proprio history + privileged height scan.
- T-paper: 1157D = T-compat + current left/right foot contact flags.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import re
import sys
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import torch

from legged_lab.assets.t4.constants import T4_JOINT_NAMES
from legged_lab.assets.t4.mujoco_sim2sim import apply_isaac_contact_friction, apply_isaac_pd, isaac_pd_gains
from legged_lab.assets.t4.schemas import (
    PROPRIO_FRAME_DIM,
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


_LAYOUT_PATH = Path(__file__).resolve().parents[1] / "terrains" / "stepping_stone_layout.py"
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
LOCO_SPARSE_START_X = 41.0


def supported_actor_obs_dim(num_obs: int) -> bool:
    return num_obs in (TEACHER_ACTOR_OBS_DIM, TEACHER_PAPER_ACTOR_OBS_DIM, TEACHER_SPARSE_ACTOR_OBS_DIM)


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
        pillar_pitch = _LAYOUT.foothold_pitch(difficulty, _LAYOUT.T4_PILLAR_PITCH_RANGE)
        pillar_height = _LAYOUT.pillar_height(difficulty)
        stone_leading_gap = _LAYOUT.platform_to_first_gap(difficulty, stone_width)
        pillar_leading_gap = _LAYOUT.platform_to_first_gap(
            difficulty, pillar_diameter, pitch_range=_LAYOUT.T4_PILLAR_PITCH_RANGE
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

    geoms = [_platform("start_platform", 0.0, 2.0, width=_LAYOUT.T4_STONE_PLATFORM_WIDTH)]
    cursor = 1.0
    rng = np.random.default_rng(int(round(float(difficulty) * 1000.0)))

    if terrain in {"sparse_course", "stepping_stones"}:
        first_center = cursor + stone_leading_gap + 0.5 * stone_width
        lane_y = [(lane - lane_count // 2) * stone_pitch for lane in range(lane_count)]
        for row in range(foothold_rows):
            x = first_center + row * stone_pitch
            for lane, y in enumerate(lane_y):
                height = max(0.04, stone_height + float(rng.uniform(-stone_jitter, stone_jitter)))
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
        cursor = first_center + (foothold_rows - 1) * stone_pitch + 0.5 * stone_width + 0.30
        if terrain == "sparse_course":
            geoms.append(_platform("transition_platform", cursor + 0.70, 1.40))
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
        cursor = first_center + (foothold_rows - 1) * pillar_pitch + 0.5 * pillar_diameter + 0.30

    geoms.append(_platform("finish_platform", cursor + 1.0, 2.0))
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
        geoms.append(shifted)
    return {**layout, "terrain": "loco_sparse", "sparse_start_x": LOCO_SPARSE_START_X, "geoms": geoms}


def _geom_xml(geom: dict) -> str:
    pos = " ".join(f"{value:.8g}" for value in geom["pos"])
    size = " ".join(f"{value:.8g}" for value in geom["size"])
    rgba = " ".join(f"{value:.8g}" for value in geom["rgba"])
    return (
        f'<geom name="{geom["name"]}" type="{geom["shape"]}" pos="{pos}" size="{size}" '
        f'rgba="{rgba}" condim="3" friction="1 0.005 0.0001"/>'
    )


def build_sparse_model_xml(
    difficulty: float,
    terrain: str = "sparse_course",
    geometry: str = "current",
    lane_count: int = LANE_COUNT,
    foothold_rows: int = FOOTHOLD_ROWS,
) -> str:
    if terrain == "loco_sparse":
        from legged_lab.scripts.sim2sim_t4_depth_student import build_loco_course

        layout = loco_sparse_layout(
            difficulty,
            geometry=geometry,
            lane_count=lane_count,
            foothold_rows=foothold_rows,
        )
        xml = MJCF_PATH.read_text().replace('meshdir="../meshes/"', f'meshdir="{ASSET_DIR / "meshes"}"')
        # The continuous course is flat through x=40.  Past that edge the
        # sparse segment has a real -2 m pit beneath its platforms/footholds.
        ground = (
            '<geom name="ground" type="box" pos="12 0 -0.05" size="28 10 0.05" '
            'material="matplane" condim="3" friction="1 0.005 0.0001"/>'
        )
        xml, replacements = re.subn(r'<geom name="ground"[^>]*/>', ground, xml, count=1)
        if replacements != 1:
            raise RuntimeError("failed to replace the stock MuJoCo ground plane")
        left = min(geom["pos"][0] - geom["size"][0] for geom in layout["geoms"])
        right = max(geom["pos"][0] + geom["size"][0] for geom in layout["geoms"])
        pit = (
            f'<geom name="loco_sparse_pit_floor" type="box" pos="{0.5 * (left + right):.8g} 0 -2.05" '
            f'size="{0.5 * (right - left) + 1.0:.8g} 6 0.05" rgba="0.08 0.09 0.10 1" '
            'condim="3" friction="1 0.005 0.0001"/>'
        )
        sparse_xml = "\n    ".join(_geom_xml(geom) for geom in layout["geoms"])
        return xml.replace("</worldbody>", f"{build_loco_course()}\n    {pit}\n    {sparse_xml}\n  </worldbody>")

    layout = sparse_course_layout(
        difficulty,
        terrain=terrain,
        geometry=geometry,
        lane_count=lane_count,
        foothold_rows=foothold_rows,
    )
    xml = MJCF_PATH.read_text()
    xml = xml.replace('meshdir="../meshes/"', f'meshdir="{ASSET_DIR / "meshes"}"')
    pit = (
        '<geom name="sparse_pit_floor" type="box" pos="7 0 -2.05" size="20 6 0.05" '
        'rgba="0.08 0.09 0.10 1" condim="3" friction="1 0.005 0.0001"/>'
    )
    xml, replacements = re.subn(r'<geom name="ground"[^>]*/>', pit, xml, count=1)
    if replacements != 1:
        raise RuntimeError("failed to replace the stock MuJoCo ground plane")
    terrain_xml = "\n    ".join(_geom_xml(geom) for geom in layout["geoms"])
    return xml.replace("</worldbody>", f"    {terrain_xml}\n  </worldbody>")


def load_actor(checkpoint_path: str) -> tuple[torch.nn.Module, int]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rsl_rl"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    actor_state = {key[len("actor.") :]: value for key, value in state.items() if key.startswith("actor.")}
    num_obs = int(actor_state["0.weight"].shape[1])
    if not supported_actor_obs_dim(num_obs):
        raise RuntimeError(
            f"checkpoint actor expects {num_obs} observations; supported sparse contracts are "
            f"{TEACHER_ACTOR_OBS_DIM}, {TEACHER_PAPER_ACTOR_OBS_DIM}, and {TEACHER_SPARSE_ACTOR_OBS_DIM}"
        )
    dims = [num_obs, 512, 256, 128, len(T4_JOINT_NAMES)]
    layers: list[torch.nn.Module] = []
    for index in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[index], dims[index + 1]))
        if index < len(dims) - 2:
            layers.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*layers)
    actor.load_state_dict(actor_state)
    actor.eval()
    return actor, num_obs


class T4SparseTeacherMujocoRunner:
    def __init__(
        self,
        checkpoint: str,
        difficulty: float = 0.5,
        terrain: str = "sparse_course",
        geometry: str = "current",
        lane_count: int = LANE_COUNT,
        foothold_rows: int = FOOTHOLD_ROWS,
    ):
        self.actor, self.actor_obs_dim = load_actor(checkpoint)
        self.paper_contact_obs = self.actor_obs_dim in (
            TEACHER_PAPER_ACTOR_OBS_DIM,
            TEACHER_SPARSE_ACTOR_OBS_DIM,
        )
        self.sparse_scan_history = self.actor_obs_dim == TEACHER_SPARSE_ACTOR_OBS_DIM
        self.scan_history_length = TEACHER_SPARSE_SCAN_HISTORY_LENGTH if self.sparse_scan_history else 1
        self.layout = (
            loco_sparse_layout(difficulty, geometry=geometry, lane_count=lane_count, foothold_rows=foothold_rows)
            if terrain == "loco_sparse"
            else sparse_course_layout(
                difficulty,
                terrain=terrain,
                geometry=geometry,
                lane_count=lane_count,
                foothold_rows=foothold_rows,
            )
        )
        self.model = mujoco.MjModel.from_xml_string(
            build_sparse_model_xml(
                difficulty,
                terrain=terrain,
                geometry=geometry,
                lane_count=lane_count,
                foothold_rows=foothold_rows,
            )
        )
        self.model.opt.timestep = PHYSICS_DT
        apply_isaac_pd(self.model)
        apply_isaac_contact_friction(self.model)
        self.data = mujoco.MjData(self.model)

        self.qpos_adr = np.array([self.model.jnt_qposadr[self.model.joint(name).id] for name in T4_JOINT_NAMES])
        self.qvel_adr = np.array([self.model.jnt_dofadr[self.model.joint(name).id] for name in T4_JOINT_NAMES])
        self.ctrl_ids = np.array([self.model.actuator(f"M{name[1:]}").id for name in T4_JOINT_NAMES])
        gains = np.array([isaac_pd_gains(name) for name in T4_JOINT_NAMES])
        self.kp, self.kd, self.effort_limit = gains[:, 0], gains[:, 1], gains[:, 2]
        self.default_pos = np.array([T4_STANDING_JOINT_POS[name] for name in T4_JOINT_NAMES])
        self._ray_geomgroup = np.ones(6, dtype=np.uint8)
        self._robot_body_id = self.model.body("Trunk").id
        self._foot_geom_side: dict[int, int] = {}
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if name.startswith("left_foot") and name.endswith("_collision"):
                self._foot_geom_side[geom_id] = 0
            elif name.startswith("right_foot") and name.endswith("_collision"):
                self._foot_geom_side[geom_id] = 1

        self.command = np.zeros(3, dtype=np.float32)
        self.reset_count = 0
        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = [0.0, 0.0, INIT_ROOT_Z]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[self.qpos_adr] = self.default_pos
        mujoco.mj_forward(self.model, self.data)
        self.action = np.zeros(len(T4_JOINT_NAMES), dtype=np.float32)
        self.gait_time = 0.0
        self.gait_phase = PHASE_OFFSET % 1.0
        frame = self._proprio_frame()
        self.history: deque[np.ndarray] = deque(
            [frame.copy() for _ in range(PROPRIO_HISTORY_LENGTH)], maxlen=PROPRIO_HISTORY_LENGTH
        )
        scan = self._height_scan()
        self.scan_history: deque[np.ndarray] = deque(
            [scan.copy() for _ in range(self.scan_history_length)], maxlen=self.scan_history_length
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
        if frame.shape[0] != PROPRIO_FRAME_DIM:
            raise RuntimeError(f"proprio frame width {frame.shape[0]} != {PROPRIO_FRAME_DIM}")
        return frame

    def _height_scan(self) -> np.ndarray:
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
            raise RuntimeError(f"height scan width {scan.shape[0]} != {TEACHER_SCAN_DIM}")
        return scan

    def _feet_contact(self) -> np.ndarray:
        contact = np.zeros(2, dtype=np.float32)
        force = np.zeros(6, dtype=np.float64)
        for index in range(self.data.ncon):
            item = self.data.contact[index]
            sides = {self._foot_geom_side.get(int(item.geom1)), self._foot_geom_side.get(int(item.geom2))}
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
            raise RuntimeError(f"actor observation width {obs.shape[0]} != checkpoint width {self.actor_obs_dim}")
        return np.clip(obs, -CLIP_OBS, CLIP_OBS)

    def step(self) -> None:
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
        gravity_z = quat_rotate_inverse_wxyz(self.data.qpos[3:7], np.array([0.0, 0.0, -1.0]))[2]
        return self.data.qpos[2] < 0.35 or gravity_z > -0.5


def run_native_viewer(runner: T4SparseTeacherMujocoRunner, vx: float) -> None:
    import mujoco.viewer

    runner.command[:] = [vx, 0.0, 0.0]
    step_dt = PHYSICS_DT * DECIMATION
    with mujoco.viewer.launch_passive(runner.model, runner.data) as viewer:
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 145.0
        viewer.cam.elevation = -18.0
        while viewer.is_running():
            start = time.time()
            runner.step()
            if runner.fallen:
                runner.reset()
                runner.command[:] = [vx, 0.0, 0.0]
            viewer.cam.lookat[:] = [runner.data.qpos[0] + 0.8, runner.data.qpos[1], 0.45]
            viewer.sync()
            remaining = step_dt - (time.time() - start)
            if remaining > 0.0:
                time.sleep(remaining)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--terrain",
        choices=("sparse_course", "stepping_stones", "raised_pillars", "loco_sparse"),
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
        "--lane-count",
        type=int,
        default=LANE_COUNT,
        help=f"Odd number of lateral foothold lanes for this local diagnostic scene (default: {LANE_COUNT}).",
    )
    parser.add_argument(
        "--foothold-rows",
        type=int,
        default=FOOTHOLD_ROWS,
        help=f"Number of longitudinal foothold rows for this local diagnostic scene (default: {FOOTHOLD_ROWS}).",
    )
    args = parser.parse_args()

    runner = T4SparseTeacherMujocoRunner(
        args.checkpoint,
        difficulty=args.difficulty,
        terrain=args.terrain,
        geometry=args.geometry,
        lane_count=args.lane_count,
        foothold_rows=args.foothold_rows,
    )
    layout = runner.layout
    if runner.sparse_scan_history:
        contract = f"LightLP sparse {TEACHER_SPARSE_ACTOR_OBS_DIM}D"
    elif runner.paper_contact_obs:
        contract = "T-paper 1157D"
    else:
        contract = "T-compat 1155D"
    print(
        f"[INFO] loaded {contract}; terrain={args.terrain}; geometry={args.geometry}; "
        f"difficulty={args.difficulty:.2f}; scene={layout['lane_count']} lanes x "
        f"{layout['foothold_rows']} rows; stone={layout['stone_width']:.3f}m "
        f"gap={layout['stone_gap']:.3f}m height={layout['stone_height']:.3f}m; "
        f"pillar={layout['pillar_diameter']:.3f}m gap={layout['pillar_gap']:.3f}m "
        f"height={layout['pillar_height']:.3f}m",
        flush=True,
    )
    run_native_viewer(runner, args.vx)


if __name__ == "__main__":
    main()

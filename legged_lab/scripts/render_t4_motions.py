"""Render T4 source motions to video for the M0 human playback review.

The IsaacLab playback gate is headless on the training server, so the human
review needs an offline renderer. This drives the migrated MJCF directly from the
raw CSV frames - no policy, no physics integration - and also reports the
foot-height and contact-sliding hints the review is looking for.

Usage (from the repo root, so the package resolves without an editable install):
    python -m legged_lab.scripts.render_t4_motions --output-dir artifacts/motion_review
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from legged_lab.assets.t4.constants import T4_JOINT_NAMES

ROOT = Path(__file__).resolve().parents[2]
MJCF = ROOT / "legged_lab/assets/t4/mjcf/t4_std.xml"
RAW_WIDTH = 7 + len(T4_JOINT_NAMES)

FOOT_SITES = ("left_foot", "right_foot")
SOLE_GEOMS = tuple(tuple(f"{side}_foot{index}_collision" for index in range(1, 7)) for side in ("left", "right"))
# A sole this close to the ground is treated as standing on it, so horizontal motion
# in that band counts as sliding.
CONTACT_HEIGHT = 0.005

# Camera azimuths are offsets from the root yaw so that clips with an arbitrary
# world heading (and the turning clips) keep the same robot-relative framing. Foot
# penetration and sliding are only judgeable in profile, so every clip is rendered
# as a three-quarter view next to a profile view.
CAMERA_VIEWS = (
    {"name": "three_quarter", "yaw_offset": 145.0, "elevation": -10.0},
    {"name": "profile", "yaw_offset": -90.0, "elevation": -4.0},
    # Sole penetration is a few centimetres, so it only reads on a foot close-up.
    {"name": "feet", "yaw_offset": -90.0, "elevation": -3.0, "distance": 1.1, "lookat_height": 0.09},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render T4 motion clips to MP4 for human review.")
    parser.add_argument("--motion-dir", type=Path, default=Path("legged_lab/envs/t4/datasets/motion_source"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/motion_review"))
    parser.add_argument("--motions", nargs="*", default=None, help="Motion stems to render; default is all.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=512, help="Width of a single view.")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--distance", type=float, default=2.4)
    parser.add_argument("--lookat-height", type=float, default=0.55)
    parser.add_argument("--sheet-columns", type=int, default=3)
    parser.add_argument("--sheet-frames", type=int, default=6)
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open the MuJoCo viewer and play the clips instead of writing files.",
    )
    parser.add_argument("--loops", type=int, default=0, help="Interactive replays per clip; 0 loops forever.")
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Refresh the manifest from kinematics without re-rendering the videos.",
    )
    return parser.parse_args()


def read_motion(path: Path) -> np.ndarray:
    with path.open(newline="") as stream:
        rows = [[float(value) for value in row] for row in csv.reader(stream) if row]
    frames = np.asarray(rows, dtype=np.float64)
    if frames.ndim != 2 or frames.shape[1] != RAW_WIDTH:
        raise ValueError(f"{path} must have width {RAW_WIDTH}, got shape {frames.shape}")
    if not np.isfinite(frames).all():
        raise ValueError(f"{path} contains non-finite values")
    return frames


def root_yaw(quat_xyzw: np.ndarray) -> float:
    x, y, z, w = quat_xyzw
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def joint_qpos_addresses(model: mujoco.MjModel) -> list[int]:
    addresses = []
    for name in T4_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise KeyError(f"joint {name!r} is missing from {MJCF}")
        addresses.append(int(model.jnt_qposadr[joint_id]))
    return addresses


def site_ids(model: mujoco.MjModel, names: tuple[str, ...]) -> list[int]:
    ids = []
    for name in names:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            raise KeyError(f"site {name!r} is missing from {MJCF}")
        ids.append(site_id)
    return ids


def sole_geom_ids(model: mujoco.MjModel) -> list[list[int]]:
    per_foot = []
    for names in SOLE_GEOMS:
        ids = []
        for name in names:
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id < 0:
                raise KeyError(f"geom {name!r} is missing from {MJCF}")
            ids.append(geom_id)
        per_foot.append(ids)
    return per_foot


def sole_lowest_z(model: mujoco.MjModel, data: mujoco.MjData, geom_ids: list[int]) -> float:
    """Lowest world z of a foot's sole capsules, so ground contact is exact."""
    lowest = math.inf
    for geom_id in geom_ids:
        radius, half_length = model.geom_size[geom_id][:2]
        axis = data.geom_xmat[geom_id].reshape(3, 3)[:, 2]
        center_z = data.geom_xpos[geom_id][2]
        lowest = min(lowest, center_z - abs(axis[2]) * half_length - radius)
    return float(lowest)


def contact_metrics(sole: np.ndarray, foot_xy: np.ndarray, fps: float) -> dict:
    """Contact and sliding statistics for a floor-normalised clip."""
    grounded = sole < CONTACT_HEIGHT + 1e-9
    step = np.linalg.norm(np.diff(foot_xy, axis=0), axis=2) * fps
    moving = grounded[1:] & grounded[:-1]
    slide = step[moving]
    return {
        "contact_frame_fraction": float(grounded.any(axis=1).mean()),
        "flight_frame_fraction": float((~grounded).all(axis=1).mean()),
        "double_support_fraction": float(grounded.all(axis=1).mean()),
        "slide_speed_mean": float(slide.mean()) if slide.size else 0.0,
        "slide_speed_p95": float(np.percentile(slide, 95)) if slide.size else 0.0,
        "slide_speed_max": float(slide.max()) if slide.size else 0.0,
    }


def contact_sheet(frames: list[np.ndarray], count: int, columns: int) -> np.ndarray:
    picks = np.linspace(0, len(frames) - 1, num=min(count, len(frames))).round().astype(int)
    tiles = [frames[index] for index in picks]
    rows = math.ceil(len(tiles) / columns)
    tile_h, tile_w, channels = tiles[0].shape
    sheet = np.zeros((rows * tile_h, columns * tile_w, channels), dtype=tiles[0].dtype)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        sheet[row * tile_h : (row + 1) * tile_h, column * tile_w : (column + 1) * tile_w] = tile
    return sheet


def apply_frame(model: mujoco.MjModel, data: mujoco.MjData, frame: np.ndarray, joint_addresses: list[int]) -> None:
    data.qpos[0:3] = frame[0:3]
    # CSV stores the root quaternion as xyzw; MuJoCo expects wxyz.
    data.qpos[3:7] = [frame[6], frame[3], frame[4], frame[5]]
    for column, address in enumerate(joint_addresses):
        data.qpos[address] = frame[7 + column]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def play_interactive(
    paths: list[Path],
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_addresses: list[int],
    args: argparse.Namespace,
) -> None:
    import time

    import mujoco.viewer

    frame_period = 1.0 / args.fps
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.distance = args.distance
        viewer.cam.elevation = CAMERA_VIEWS[0]["elevation"]
        for path in paths:
            motion = read_motion(path)
            print(f"playing {path.stem} ({motion.shape[0]} frames)", flush=True)
            replay = 0
            while viewer.is_running() and (args.loops <= 0 or replay < args.loops):
                for frame in motion:
                    if not viewer.is_running():
                        return
                    deadline = time.perf_counter() + frame_period
                    apply_frame(model, data, frame, joint_addresses)
                    viewer.cam.lookat[:] = [frame[0], frame[1], args.lookat_height]
                    viewer.sync()
                    time.sleep(max(0.0, deadline - time.perf_counter()))
                replay += 1
            if not viewer.is_running():
                return


def render_motion(
    path: Path,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer | None,
    cameras: list[mujoco.MjvCamera],
    joint_addresses: list[int],
    foot_site_ids: list[int],
    sole_ids: list[list[int]],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict:
    motion = read_motion(path)
    images: list[np.ndarray] = []
    sole_history: list[np.ndarray] = []
    foot_xy_history: list[np.ndarray] = []

    for frame in motion:
        apply_frame(model, data, frame, joint_addresses)

        foot_xy_history.append(np.array([data.site_xpos[site_id][:2] for site_id in foot_site_ids]))
        sole_history.append(np.array([sole_lowest_z(model, data, ids) for ids in sole_ids]))

        if renderer is None:
            continue
        yaw = math.degrees(root_yaw(frame[3:7]))
        views = []
        for camera, view in zip(cameras, CAMERA_VIEWS):
            camera.azimuth = yaw + view["yaw_offset"]
            camera.lookat[:] = [frame[0], frame[1], view.get("lookat_height", args.lookat_height)]
            renderer.update_scene(data, camera=camera)
            views.append(renderer.render().copy())
        images.append(np.hstack(views))

    sole = np.stack(sole_history)
    foot_xy = np.stack(foot_xy_history)
    worst_index = int(sole.min(axis=1).argmin())
    lowest_sole = float(sole.min())

    video_path = output_dir / f"{path.stem}.mp4"
    sheet_path = output_dir / f"{path.stem}.png"
    worst_path = output_dir / f"{path.stem}_worst_sole.png"
    if images:
        imageio.mimwrite(video_path, images, fps=args.fps, quality=8, macro_block_size=1)
        imageio.imwrite(sheet_path, contact_sheet(images, args.sheet_frames, args.sheet_columns))
        imageio.imwrite(worst_path, images[worst_index])

    per_frame_sole = sole.min(axis=1)
    stats = {
        "frames": int(motion.shape[0]),
        "duration_s": float(motion.shape[0] / args.fps),
        "lowest_sole_height": lowest_sole,
        "lowest_sole_frame": worst_index,
        "sole_penetration": max(0.0, -lowest_sole),
        "stance_sole_p05": float(np.percentile(per_frame_sole, 5)),
        "stance_sole_median": float(np.median(per_frame_sole)),
        "video": str(video_path.relative_to(ROOT)),
        "contact_sheet": str(sheet_path.relative_to(ROOT)),
        "worst_sole_frame_image": str(worst_path.relative_to(ROOT)),
    }
    # The raw clips sit at the wrong height relative to the floor, which both hides
    # and fakes ground contact, so contact-dependent numbers are reported on a
    # floor-normalised copy. The reference is the 5th percentile of the per-frame
    # lowest sole rather than the minimum, because a handful of deep-penetration
    # frames would otherwise lift the whole clip clear of the ground.
    stats["floor_offset"] = -stats["stance_sole_p05"]
    stats.update(contact_metrics(sole + stats["floor_offset"], foot_xy, args.fps))
    return stats


def main() -> None:
    args = parse_args()
    motion_dir = args.motion_dir if args.motion_dir.is_absolute() else ROOT / args.motion_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir

    stems = args.motions or sorted(path.stem for path in motion_dir.glob("*.csv"))
    paths = []
    for stem in stems:
        path = motion_dir / f"{stem}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"motion {stem!r} is missing at {path}")
        paths.append(path)

    model = mujoco.MjModel.from_xml_path(str(MJCF))
    # Raise the offscreen framebuffer here instead of editing the shared MJCF, which
    # is also the sim2sim and deployment asset.
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), args.width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), args.height)
    data = mujoco.MjData(model)
    joint_addresses = joint_qpos_addresses(model)
    foot_site_ids = site_ids(model, FOOT_SITES)
    sole_ids = sole_geom_ids(model)

    if args.interactive:
        play_interactive(paths, model, data, joint_addresses, args)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    cameras = []
    for view in CAMERA_VIEWS:
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.elevation = view["elevation"]
        camera.distance = view.get("distance", args.distance)
        cameras.append(camera)

    summary = {}
    renderer = None if args.metrics_only else mujoco.Renderer(model, height=args.height, width=args.width)
    try:
        for path in paths:
            stats = render_motion(
                path, model, data, renderer, cameras, joint_addresses, foot_site_ids, sole_ids, args, output_dir
            )
            summary[path.stem] = stats
            print(
                f"{path.stem}: {stats['frames']} frames, lowest sole {stats['lowest_sole_height'] * 1000:+6.1f} mm, "
                f"stance sole p05 {stats['stance_sole_p05'] * 1000:+6.1f} mm, "
                f"contact {stats['contact_frame_fraction']:.2f}, slide p95 {stats['slide_speed_p95']:.3f} m/s",
                flush=True,
            )
    finally:
        if renderer is not None:
            renderer.close()

    manifest = {
        "renderer": "mujoco",
        "mujoco_version": mujoco.__version__,
        "mjcf": str(MJCF.relative_to(ROOT)),
        "views": [view["name"] for view in CAMERA_VIEWS],
        "fps": args.fps,
        "contact_height_threshold": CONTACT_HEIGHT,
        "motions": summary,
    }
    (output_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"T4_MOTION_RENDER motions={len(summary)} output={output_dir}", flush=True)


if __name__ == "__main__":
    main()

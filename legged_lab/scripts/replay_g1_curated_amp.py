# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

"""Replay every selected segment from its exact source frames and prepared AMP q."""

import argparse
import json
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image, ImageDraw

from legged_lab.scripts.curate_g1_amp import ROOT, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--snapshot-dir", type=Path)
    args = parser.parse_args()
    model, joint_ids, _ = load_model()
    with tempfile.TemporaryDirectory(prefix="g1_curated_view_") as temp:
        xml = Path(temp) / "viewer.xml"
        mujoco.mj_saveLastXML(str(xml), model)
        tree = ET.parse(xml)
        world = tree.getroot().find("worldbody")
        ET.SubElement(world, "geom", name="floor", type="plane", size="200 200 .1", rgba=".28 .3 .34 1")
        ET.SubElement(world, "light", pos="0 0 5", dir="0 0 -1", directional="true")
        model = mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(), encoding="unicode"))
    data = mujoco.MjData(model)
    manifest = json.loads((args.data_dir / "_manifest.json").read_text())
    clips = []
    for name in manifest["clips"]:
        raw = json.loads((args.data_dir / f"{name}.txt").read_text())
        frames = np.asarray(raw["Frames"])
        rows = np.loadtxt(ROOT / raw["SourceMotion"], delimiter=",")[raw["SourceFrameStart"] : raw["SourceFrameStop"]]
        assert len(rows) == len(frames) and np.max(np.abs(rows[:, 7:] - frames[:, :29])) < 1e-5
        clips.append((name, rows, frames))

    def pose(rows, frames, k):
        data.qpos[:3] = rows[k, :3]
        data.qpos[3:7] = rows[k, [6, 3, 4, 5]]
        data.qpos[model.jnt_qposadr[joint_ids]] = frames[k, :29]
        mujoco.mj_forward(model, data)

    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.distance, camera.azimuth, camera.elevation = 2.6, 140, -15
        writer = imageio.get_writer(str(args.record), fps=15, codec="libx264", quality=8)
        for name, rows, frames in clips:
            for k in range(0, len(rows), 2):
                pose(rows, frames, k)
                camera.lookat[:] = rows[k, :3] + [0, 0, -0.15]
                renderer.update_scene(data, camera=camera)
                img = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(img)
                draw.rectangle((0, 0, 640, 44), fill="black")
                draw.text((8, 5), f"{name} | {k / 30:.2f}s / {len(rows) / 30:.2f}s", fill="white")
                draw.text((8, 24), "Exact expert replay | kinematics only | clip boundaries are separate", fill="white")
                writer.append_data(np.asarray(img))
                if args.snapshot_dir and k == (len(rows) // 4) * 2:
                    args.snapshot_dir.mkdir(parents=True, exist_ok=True)
                    img.save(args.snapshot_dir / f"{name}.png")
            print("recorded", name, flush=True)
        writer.close()
        renderer.close()
        return
    state = {"clip": 0, "paused": False, "frame": 0}

    def on_key(key):
        if key == 32:
            state["paused"] = not state["paused"]
        if key in (78, 80):
            state["clip"] = (state["clip"] + (1 if key == 78 else -1)) % len(clips)
            state["frame"] = 0

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.6, 140, -15
        while viewer.is_running():
            start = time.perf_counter()
            name, rows, frames = clips[state["clip"]]
            pose(rows, frames, state["frame"])
            viewer.cam.lookat[:] = rows[state["frame"], :3] + [0, 0, -0.15]
            viewer.set_texts(
                (
                    mujoco.mjtFontScale.mjFONTSCALE_150,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    f"{name}\nKinematic expert | Space pause | N/P clip",
                    "",
                )
            )
            viewer.sync()
            if not state["paused"]:
                state["frame"] += 1
                if state["frame"] == len(rows):
                    state["frame"] = 0
                    state["clip"] = (state["clip"] + 1) % len(clips)
            time.sleep(max(0, 1 / 30 - (time.perf_counter() - start)))


if __name__ == "__main__":
    main()

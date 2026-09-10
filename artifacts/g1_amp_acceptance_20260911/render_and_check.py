"""Visual + quantitative acceptance of the G1 AMP expert set used by stage-2 vital runs.

Renders every ``motion_amp_expert_unitree_v5`` clip from its exact source CSV frames
in MuJoCo (official URDF, kinematic replay), records two-view MP4s plus snapshots,
and computes arm/gait sanity statistics. Run from the repo root:

    /d/Z2_SSY/mjlab_ssy/.venv/Scripts/python.exe artifacts/g1_amp_acceptance_20260911/render_and_check.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from legged_lab.scripts.curate_g1_amp import load_model  # noqa: E402

DATA_DIR = ROOT / "legged_lab/envs/g1/datasets/motion_amp_expert_unitree_v5"
OUT_DIR = Path(__file__).resolve().parent
SNAPSHOT_FRACTIONS = (0.12, 0.33, 0.55, 0.77)
VIEWS = {"side34": 140, "side34_r": 40}  # two opposing 3/4 views so both arms are seen
ARM_SLICES = {
    "left_arm": slice(15, 22),
    "right_arm": slice(22, 29),
    "waist": slice(12, 15),
}
ARM_JOINT_NAMES = [
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
]


def pose(data, model, joint_ids, row):
    data.qpos[:3] = row[:3]
    data.qpos[3:7] = row[[6, 3, 4, 5]]
    data.qpos[model.jnt_qposadr[joint_ids]] = row[7:]
    mujoco.mj_forward(model, data)


def add_floor(model):
    with tempfile.TemporaryDirectory(prefix="g1_accept_") as temp:
        xml = Path(temp) / "viewer.xml"
        mujoco.mj_saveLastXML(str(xml), model)
        tree = ET.parse(xml)
        world = tree.getroot().find("worldbody")
        ET.SubElement(world, "geom", name="floor", type="plane", size="200 200 .1", rgba=".28 .3 .34 1")
        ET.SubElement(world, "light", pos="0 0 5", dir="0 0 -1", directional="true")
        return mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(), encoding="unicode"))


def arm_report(rows, limits):
    q = rows[:, 7:]
    report = {}
    for group, sl in ARM_SLICES.items():
        report[group] = {}
        for k, name in enumerate(ARM_JOINT_NAMES if "arm" in group else ["waist_yaw", "waist_roll", "waist_pitch"]):
            j = sl.start + k
            lo, hi = limits[j]
            series = q[:, j]
            dq = np.gradient(series, 1 / 30.0)
            report[group][name] = {
                "min_rad": round(float(series.min()), 3),
                "max_rad": round(float(series.max()), 3),
                "mean_rad": round(float(series.mean()), 3),
                "std_rad": round(float(series.std()), 3),
                "at_limit_frac": round(float(((series <= lo + 1e-3) | (series >= hi - 1e-3)).mean()), 4),
                "max_dq_rad_s": round(float(np.abs(dq).max()), 2),
                "limit": [round(float(lo), 3), round(float(hi), 3)],
            }
    # anti-phase arm swing: left vs right shoulder_pitch should alternate
    lp, rp = q[:, 15], q[:, 22]
    corr = float(np.corrcoef(lp - lp.mean(), rp - rp.mean())[0, 1])
    report["arm_swing_correlation_L_vs_R_shoulder_pitch"] = round(corr, 3)
    return report


def main():
    model_raw, joint_ids, _ = load_model()
    limits = model_raw.jnt_range[joint_ids].copy()
    model = add_floor(model_raw)
    data = mujoco.MjData(model)
    manifest = json.loads((DATA_DIR / "_manifest.json").read_text())
    clip_names = list(manifest["clips"].keys())
    report = {"clips": {}, "passed_q_match": True, "max_feature_error_m": 0.0}

    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    snapshots = {}
    for name in clip_names:
        raw = json.loads((DATA_DIR / f"{name}.txt").read_text())
        frames = np.asarray(raw["Frames"])
        rows = np.loadtxt(ROOT / raw["SourceMotion"], delimiter=",")[: len(frames)]
        q_err = float(np.max(np.abs(rows[:, 7:] - frames[:, :29])))
        assert q_err < 1e-5, (name, q_err)

        stats = {
            "frames": len(frames),
            "q29_match_max_err": q_err,
            "root_z_min_max_m": [round(float(rows[:, 2].min()), 3), round(float(rows[:, 2].max()), 3)],
            "speed_mean_m_s": round(
                float(np.linalg.norm(np.gradient(rows[:, :3], 1 / 30.0, axis=0)[:, :2], axis=1).mean()), 3
            ),
            "arms": arm_report(rows, limits),
        }

        # AMP hand/foot features vs fresh FK on this model
        body_ids = [model.body(n).id for n in ("left_wrist_yaw_link", "right_wrist_yaw_link")]
        hand_err = 0.0
        for k in range(0, len(rows), 25):
            pose(data, model, joint_ids, rows[k])
            rotation = data.xmat[model.body("pelvis").id].reshape(3, 3)
            hands = ((data.xpos[body_ids] - rows[k, :3]) @ rotation).reshape(-1)
            hand_err = max(hand_err, float(np.abs(hands - frames[k, 58:64]).max()))
        stats["hand_feature_max_err_m"] = round(hand_err, 6)
        report["max_feature_error_m"] = max(report["max_feature_error_m"], hand_err)

        hands_root = frames[:, 58:64].reshape(-1, 2, 3)
        stats["hand_height_m"] = {
            "left_min_max": [round(float(hands_root[:, 0, 2].min()), 3), round(float(hands_root[:, 0, 2].max()), 3)],
            "right_min_max": [round(float(hands_root[:, 1, 2].min()), 3), round(float(hands_root[:, 1, 2].max()), 3)],
        }
        stats["hands_min_separation_m"] = round(float(np.linalg.norm(np.diff(hands_root, axis=1)[:, 0], axis=1).min()), 3)

        for view, azimuth in VIEWS.items():
            writer = imageio.get_writer(str(OUT_DIR / f"{name}_{view}.mp4"), fps=15, codec="libx264", quality=8)
            camera.distance, camera.azimuth, camera.elevation = 2.6, azimuth, -15
            for k in range(0, len(rows), 2):
                pose(data, model, joint_ids, rows[k])
                camera.lookat[:] = rows[k, :3] + [0, 0, -0.15]
                renderer.update_scene(data, camera=camera)
                img = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(img)
                draw.rectangle((0, 0, 640, 44), fill="black")
                draw.text((8, 5), f"{name} | {k / 30:.2f}s / {len(rows) / 30:.2f}s | {view}", fill="white")
                draw.text((8, 24), "AMP expert v5 exact source replay | kinematic", fill="white")
                writer.append_data(np.asarray(img))
                if view == "side34":
                    frac = k / len(rows)
                    nearest = min(SNAPSHOT_FRACTIONS, key=lambda f: abs(f - frac))
                    if abs(frac - nearest) < 1.0 / len(rows) and nearest not in snapshots.setdefault(name, {}):
                        snapshots[name][nearest] = img
            writer.close()
            print("recorded", name, view, flush=True)
        report["clips"][name] = stats

    snap_dir = OUT_DIR / "snapshots"
    snap_dir.mkdir(exist_ok=True)
    for name, by_frac in snapshots.items():
        for frac, img in sorted(by_frac.items()):
            img.save(snap_dir / f"{name}_f{int(frac * 100):02d}.png")

    (OUT_DIR / "arm_quantitative.json").write_text(json.dumps(report, indent=2) + "\n")
    renderer.close()
    print(json.dumps({"clips": len(clip_names), "max_feature_error_m": report["max_feature_error_m"]}))


if __name__ == "__main__":
    main()

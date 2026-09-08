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

"""Convert Z2 PKL/NPZ source motions into policy-order CSVs.

Isaac-free. Does not produce 70D AMP experts; those require Isaac FK through
``AmpFeatureBuilder``. Rejects G1 tensors and undeclared joint orders.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from legged_lab.assets.z2.constants import (
    NUM_Z2_29DOF_JOINTS,
    Z2_29DOF_JOINT_NAMES,
    Z2_SOURCE_JOINT_ORDER,
)
from legged_lab.assets.z2.schemas import (
    AMP_HELD_OUT_MOTIONS,
    AMP_MOTION_CLASSES,
    AMP_MOTION_SOURCE_DIR,
    AMP_MOTION_SOURCE_RAW_DIR,
    Z2_SOURCE_CSV_WIDTH,
    Z2_SOURCE_DATASET,
    amp_motion_class,
    amp_motion_weight,
)

ROOT = Path(__file__).resolve().parents[3]

WHITELIST: dict[str, str] = {
    "walk": "walk.pkl",
    "walk_l": "walk_l.pkl",
    "run": "run.pkl",
}

# Same-named upstream txt files are not the CSV lineage. Expert run.txt is 64D
# with bad qvel; visualization/run.txt is a different 70D schema and frame count.
UPSTREAM_TXT_NOT_SOURCE: dict[str, tuple[str, ...]] = {
    "walk": (
        "legged_lab/envs/z2/datasets/z2_data/motion_amp_expert/walk.txt",
        "legged_lab/envs/z2/datasets/z2_data/motion_amp_expert/walk_repaired.txt",
    ),
    "walk_l": (
        "legged_lab/envs/z2/datasets/z2_data/motion_amp_expert/walk_l.txt",
        "legged_lab/envs/z2/datasets/z2_data/motion_amp_expert/walk_l_repaired.txt",
    ),
    "run": (
        "legged_lab/envs/z2/datasets/z2_data/motion_amp_expert/run.txt",
        "legged_lab/envs/z2/datasets/z2_data/motion_visualization/run.txt",
    ),
}


class NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            try:
                return super().find_class(module, name)
            except ModuleNotFoundError:
                module = module.replace("numpy._core", "numpy.core", 1)
        elif module.startswith("numpy.core"):
            try:
                return super().find_class(module, name)
            except ModuleNotFoundError:
                module = module.replace("numpy.core", "numpy._core", 1)
        return super().find_class(module, name)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pickle_compat(path: Path):
    with path.open("rb") as stream:
        return NumpyCompatUnpickler(stream).load()


def load_motion(path: Path) -> dict:
    suffix = path.suffix.lower()
    if suffix == ".pkl":
        data = load_pickle_compat(path)
        if not isinstance(data, dict):
            raise ValueError(f"{path} pickle is {type(data).__name__}, expected dict")
        names = data.get("link_body_list")
        return {
            "fps": float(data["fps"]),
            "root_pos": np.asarray(data["root_pos"], dtype=np.float64),
            "root_rot": np.asarray(data["root_rot"], dtype=np.float64),
            "dof_pos": np.asarray(data["dof_pos"], dtype=np.float64),
            "source_joint_names": list(names) if names else list(Z2_SOURCE_JOINT_ORDER),
        }
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            missing = {"qpos", "fps"} - set(archive.files)
            if missing:
                raise KeyError(f"{path} missing NPZ keys {sorted(missing)}")
            qpos = np.asarray(archive["qpos"], dtype=np.float64)
            fps = float(np.asarray(archive["fps"]).reshape(-1)[0])
        expected = 7 + NUM_Z2_29DOF_JOINTS
        if qpos.ndim != 2 or qpos.shape[1] != expected:
            raise ValueError(f"{path} qpos must be [N,{expected}], got {qpos.shape}")
        root_rot_wxyz = qpos[:, 3:7]
        root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]
        return {
            "fps": fps,
            "root_pos": qpos[:, :3],
            "root_rot": root_rot_xyzw,
            "dof_pos": qpos[:, 7:],
            "source_joint_names": list(Z2_SOURCE_JOINT_ORDER),
        }
    raise ValueError(f"unsupported motion extension: {path.suffix}")


def reorder_dofs(dof_pos: np.ndarray, source_joint_names: list[str], target_joint_names=Z2_29DOF_JOINT_NAMES):
    if dof_pos.ndim != 2 or dof_pos.shape[1] != NUM_Z2_29DOF_JOINTS:
        raise ValueError(f"dof_pos must be [N,{NUM_Z2_29DOF_JOINTS}], got {dof_pos.shape}")
    if len(source_joint_names) != NUM_Z2_29DOF_JOINTS:
        raise ValueError(f"source joint list has {len(source_joint_names)} names, expected {NUM_Z2_29DOF_JOINTS}")
    if len(set(source_joint_names)) != NUM_Z2_29DOF_JOINTS:
        raise ValueError("source joint list is not unique")
    missing = [name for name in target_joint_names if name not in source_joint_names]
    extra = [name for name in source_joint_names if name not in target_joint_names]
    if missing or extra:
        raise ValueError(f"joint name mismatch; missing={missing}, extra={extra}")
    index = {name: i for i, name in enumerate(source_joint_names)}
    return dof_pos[:, [index[name] for name in target_joint_names]]


def reject_foreign_tensor(path: Path, dof_pos: np.ndarray) -> None:
    stem = path.stem
    if stem in AMP_HELD_OUT_MOTIONS or "g1" in stem.lower() or "subject" in stem.lower():
        raise ValueError(f"{path} is held out of the Z2 AMP whitelist (G1-compat or undeclared subject clip)")
    if dof_pos.shape[1] == 27:
        raise ValueError(f"{path} looks like T4 27DoF, not Z2 29DoF")
    if dof_pos.shape[1] not in (20, 23, 29):
        raise ValueError(f"{path} has {dof_pos.shape[1]} DoF, not a known Z2 variant")
    if dof_pos.shape[1] != 29:
        raise ValueError(f"{path} is a {dof_pos.shape[1]}DoF Z2 variant, not the 29DoF teacher")


def motion_to_csv_rows(motion: dict) -> np.ndarray:
    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    root_rot = np.asarray(motion["root_rot"], dtype=np.float64)
    dof = reorder_dofs(np.asarray(motion["dof_pos"], dtype=np.float64), motion["source_joint_names"])
    if root_pos.shape[0] != dof.shape[0] or root_rot.shape != (dof.shape[0], 4):
        raise ValueError("root and dof frame counts disagree")
    if not np.isfinite(root_pos).all() or not np.isfinite(root_rot).all() or not np.isfinite(dof).all():
        raise ValueError("non-finite source values")
    norms = np.linalg.norm(root_rot, axis=-1, keepdims=True)
    if np.any(norms < 1.0e-8):
        raise ValueError("source quaternion has a near-zero norm")
    root_rot = root_rot / norms
    rows = np.concatenate([root_pos, root_rot, dof], axis=1)
    if rows.shape[1] != Z2_SOURCE_CSV_WIDTH:
        raise ValueError(f"CSV width {rows.shape[1]} != {Z2_SOURCE_CSV_WIDTH}")
    return rows


def write_csv(path: Path, rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        for row in rows:
            writer.writerow([f"{value:.16g}" for value in row])


def convert_file(path: Path, output_csv: Path) -> dict:
    motion = load_motion(path)
    reject_foreign_tensor(path, np.asarray(motion["dof_pos"]))
    rows = motion_to_csv_rows(motion)
    write_csv(output_csv, rows)
    stem = output_csv.stem
    return {
        "stem": stem,
        "source": str(path.as_posix()),
        "source_sha256": sha256(path),
        "csv": str(output_csv.as_posix()),
        "csv_sha256": sha256(output_csv),
        "fps": float(motion["fps"]),
        "frames": int(rows.shape[0]),
        "width": int(rows.shape[1]),
        "source_joint_names": list(motion["source_joint_names"]),
        "policy_joint_names": list(Z2_29DOF_JOINT_NAMES),
        "motion_class": amp_motion_class(stem),
        "motion_weight": amp_motion_weight(stem),
        "dataset": Z2_SOURCE_DATASET,
    }


def convert_whitelist(raw_dir: Path | None = None, csv_dir: Path | None = None) -> dict:
    raw_dir = raw_dir or (ROOT / AMP_MOTION_SOURCE_RAW_DIR)
    csv_dir = csv_dir or (ROOT / AMP_MOTION_SOURCE_DIR)
    summaries = []
    for stem, filename in WHITELIST.items():
        if stem in AMP_HELD_OUT_MOTIONS:
            raise ValueError(f"whitelist stem {stem} is held out")
        if stem not in AMP_MOTION_CLASSES:
            raise ValueError(f"whitelist stem {stem} has no AMP class")
        source = raw_dir / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        summaries.append(convert_file(source, csv_dir / f"{stem}.csv"))
    manifest = {
        "schema": "z2_amp_source.v1",
        "dataset": Z2_SOURCE_DATASET,
        "policy_joint_names": list(Z2_29DOF_JOINT_NAMES),
        "source_joint_order_fallback": list(Z2_SOURCE_JOINT_ORDER),
        "held_out": list(AMP_HELD_OUT_MOTIONS),
        "upstream_txt_not_source": {stem: list(paths) for stem, paths in UPSTREAM_TXT_NOT_SOURCE.items()},
        "lineage_note": (
            "CSV frames equal PKL frames. They are not expert txt or visualization txt "
            "of the same stem. qvel is not copied from 64D experts; Isaac FK writes 70D later."
        ),
        "motions": summaries,
    }
    csv_dir.mkdir(parents=True, exist_ok=True)
    (csv_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest

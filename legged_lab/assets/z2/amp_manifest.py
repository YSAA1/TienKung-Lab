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

"""Isaac-free Z2 AMP source validation and training-manifest clips+sha256 contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from legged_lab.assets.z2.constants import Z2_29DOF_JOINT_NAMES, Z2_USD_LAYER_FILES
from legged_lab.assets.z2.schemas import (
    AMP_FORMAL_V1_STEMS,
    AMP_FRAME_DIM,
    AMP_HELD_OUT_MOTIONS,
    AMP_MOTION_CLASSES,
    AMP_SCHEMA_VERSION,
    Z2_SOURCE_DATASET,
    amp_motion_class,
)

PENDING_EXPERT_STATUS = "pending_isaac_fk"
COMPLETED_EXPERT_STATUS = "isaac_fk"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_raw_source_path(item: dict, motion_dir: Path) -> Path:
    declared = Path(str(item.get("source", "")))
    if declared.is_file():
        return declared
    # Portable fallback: keep the declared path's suffix below the last
    # ``motion_source_raw`` marker (preserving subdirectories like ``liyang/``);
    # fall back to the bare file name for legacy flat manifests.
    parts = declared.parts
    if "motion_source_raw" in parts:
        suffix = Path(*parts[parts.index("motion_source_raw") + 1 :])
    else:
        suffix = Path(declared.name)
    fallback = motion_dir.parent / "motion_source_raw" / suffix
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"raw source missing at declared {declared} and fallback {fallback}")


def validate_z2_source_manifest(source_manifest: dict, motion_dir: Path) -> dict[str, dict]:
    if not source_manifest.get("motions"):
        raise ValueError("Z2 source manifest is missing motions; refuse to generate experts")
    policy = source_manifest.get("policy_joint_names")
    if list(policy or []) != list(Z2_29DOF_JOINT_NAMES):
        raise ValueError("Z2 source manifest policy_joint_names is not Z2_29DOF_JOINT_NAMES")
    by_stem = {item["stem"]: item for item in source_manifest["motions"]}
    checked = {}
    # Manifest-driven: every declared clip must be classed, not held out, and
    # hash-clean. The manifest (not AMP_MOTION_CLASSES) defines the version's
    # clip set, so v1 and curated v2 sources both validate.
    for stem in by_stem:
        if stem in AMP_HELD_OUT_MOTIONS:
            raise ValueError(f"declared AMP motion {stem!r} is held out")
        amp_motion_class(stem)
        item = by_stem[stem]
        csv_path = motion_dir / f"{stem}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"declared AMP motion {stem!r} is missing at {csv_path}")
        actual_csv = sha256_file(csv_path)
        declared_csv = item.get("csv_sha256")
        if declared_csv != actual_csv:
            raise ValueError(f"CSV hash mismatch for {stem}: manifest {declared_csv} file {actual_csv}")
        declared_raw = item.get("source_sha256")
        if not isinstance(declared_raw, str) or not declared_raw.strip():
            raise ValueError(f"raw source SHA missing for {stem}")
        raw_path = resolve_raw_source_path(item, motion_dir)
        actual_raw = sha256_file(raw_path)
        if actual_raw != declared_raw:
            raise ValueError(f"raw source hash mismatch for {stem}: manifest {declared_raw} file {actual_raw}")
        checked[stem] = {
            "csv": csv_path,
            "csv_sha256": actual_csv,
            "source": raw_path,
            "source_sha256": actual_raw,
            "fps": float(item["fps"]),
            "frames": int(item["frames"]),
        }
    return checked


def training_clip_record(
    *,
    txt_path: Path,
    source_csv: Path,
    source_csv_sha256: str,
    source_raw_sha256: str | None,
    frames: int,
    fps: float,
    motion_class: str,
    motion_weight: float,
    root: Path,
) -> dict:
    return {
        "sha256": sha256_file(txt_path),
        "frames": int(frames),
        "fps": float(fps),
        "motion_class": motion_class,
        "motion_weight": float(motion_weight),
        "source_csv": str(source_csv.relative_to(root)).replace("\\", "/"),
        "source_csv_sha256": source_csv_sha256,
        "source_raw_sha256": source_raw_sha256,
    }


def usd_layer_shas(asset_dir: Path) -> dict[str, str]:
    layers = {}
    for rel in Z2_USD_LAYER_FILES:
        path = asset_dir / rel
        if not path.is_file():
            raise FileNotFoundError(f"USD layer missing: {path}")
        layers[rel.replace("\\", "/")] = sha256_file(path)
    return layers


def code_provenance(root: Path) -> dict:
    asset_dir = root / "legged_lab/assets/z2"
    return {
        "asset_mode": "upstream_usd",
        "usd_layers": usd_layer_shas(asset_dir),
        "urdf_sha256": sha256_file(asset_dir / "urdf/assembly.urdf"),
        "asset_cfg_sha256": sha256_file(asset_dir / "z2.py"),
        "generator_sha256": sha256_file(root / "legged_lab/scripts/generate_z2_amp_expert.py"),
        "amp_manifest_sha256": sha256_file(root / "legged_lab/assets/z2/amp_manifest.py"),
    }


def amp_training_manifest(
    *,
    clips: dict,
    motions: dict,
    kinematics_response: float,
    sim_dt: float,
    max_frames: int,
    converter: dict,
    plant: str,
    joint_order: list[str],
    held_out: list[str],
    frame_dim: int,
    provenance: dict | None = None,
) -> dict:
    if not clips:
        raise ValueError("AMP training manifest has no clips")
    return {
        "amp_schema_version": AMP_SCHEMA_VERSION,
        "frame_dim": frame_dim,
        "transition_dim": 2 * frame_dim,
        "joint_order": list(joint_order),
        "held_out_motions": list(held_out),
        "source_dataset": Z2_SOURCE_DATASET,
        "status": "isaac_fk",
        "kinematics_response": kinematics_response,
        "simulator": "isaaclab",
        "sim_dt": sim_dt,
        "max_frames": max_frames,
        "plant": plant,
        "converter": converter,
        "provenance": provenance or {},
        "clips": clips,
        "motions": motions,
        "note": "70D q29+dq29+hands6+feet6. train.py binds clips[stem].sha256 next to {stem}.txt.",
    }


def assert_pending_expert_tree(expert_dir: Path) -> None:
    manifest = json.loads((expert_dir / "_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != PENDING_EXPERT_STATUS:
        raise ValueError(f"pending expert status must be {PENDING_EXPERT_STATUS}, got {manifest.get('status')}")
    if manifest.get("clips"):
        raise ValueError("pending expert tree must have empty clips")
    for stem in AMP_MOTION_CLASSES:
        path = expert_dir / f"{stem}.txt"
        if path.exists():
            raise ValueError(f"pending expert tree must not contain {path.name}")


def assert_completed_expert_tree(  # noqa: C901
    expert_dir: Path, *, source_dir: Path, root: Path, expected_stems: tuple[str, ...] = AMP_FORMAL_V1_STEMS
) -> None:
    manifest = json.loads((expert_dir / "_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != COMPLETED_EXPERT_STATUS:
        raise ValueError(f"completed expert status must be {COMPLETED_EXPERT_STATUS}, got {manifest.get('status')}")
    if int(manifest.get("max_frames", -1)) != 0:
        raise ValueError(f"completed experts must set max_frames=0, got {manifest.get('max_frames')}")
    if int(manifest.get("frame_dim", -1)) != AMP_FRAME_DIM:
        raise ValueError(f"completed frame_dim must be {AMP_FRAME_DIM}")
    if list(manifest.get("joint_order") or []) != list(Z2_29DOF_JOINT_NAMES):
        raise ValueError("completed manifest joint_order must be Z2_29DOF_JOINT_NAMES")
    clips = manifest.get("clips") or {}
    if set(clips) != set(expected_stems):
        raise ValueError(f"completed clips must be {sorted(expected_stems)}, got {sorted(clips)}")
    provenance = manifest.get("provenance") or {}
    if provenance.get("asset_mode") != "upstream_usd":
        raise ValueError("completed provenance asset_mode must be upstream_usd")
    expected_layers = usd_layer_shas(root / "legged_lab/assets/z2")
    if provenance.get("usd_layers") != expected_layers:
        raise ValueError("completed provenance usd_layers must match the training USD files")
    source_manifest = json.loads((source_dir / "_manifest.json").read_text(encoding="utf-8"))
    checked = validate_z2_source_manifest(source_manifest, source_dir)
    for stem, source in checked.items():
        txt_path = expert_dir / f"{stem}.txt"
        if not txt_path.is_file():
            raise FileNotFoundError(f"completed expert missing {txt_path}")
        digest = sha256_file(txt_path)
        clip = clips[stem]
        if clip.get("sha256") != digest:
            raise ValueError(f"clip sha256 mismatch for {stem}: manifest {clip.get('sha256')} file {digest}")
        if clip.get("source_csv_sha256") != source["csv_sha256"]:
            raise ValueError(f"clip source_csv_sha256 mismatch for {stem}")
        if clip.get("source_raw_sha256") != source["source_sha256"]:
            raise ValueError(f"clip source_raw_sha256 mismatch for {stem}")
        payload = json.loads(txt_path.read_text(encoding="utf-8"))
        if list(payload.get("JointOrder") or []) != list(Z2_29DOF_JOINT_NAMES):
            raise ValueError(f"{stem}.txt JointOrder is not Z2_29DOF_JOINT_NAMES")
        frames = payload.get("Frames") or []
        if not frames or len(frames[0]) != AMP_FRAME_DIM:
            raise ValueError(f"{stem}.txt frame width must be {AMP_FRAME_DIM}")
        expected_frames = int(source["frames"]) - 1
        if len(frames) != expected_frames:
            raise ValueError(f"{stem}.txt has {len(frames)} frames, expected source_rows-1={expected_frames}")
        if int(clip.get("frames", -1)) != expected_frames:
            raise ValueError(f"clip frames for {stem} must be {expected_frames}")
        fps = float(source["fps"])
        if abs(float(clip.get("fps", 0.0)) - fps) > 1.0e-6:
            raise ValueError(f"clip fps for {stem} must be {fps}")
        duration = float(payload.get("FrameDuration", 0.0))
        if abs(duration - 1.0 / fps) > 1.0e-6:
            raise ValueError(f"{stem}.txt FrameDuration {duration} != 1/fps")


def assert_expert_tree_lifecycle(expert_dir: Path, *, source_dir: Path, root: Path) -> str:
    manifest = json.loads((expert_dir / "_manifest.json").read_text(encoding="utf-8"))
    status = manifest.get("status")
    if status == PENDING_EXPERT_STATUS:
        assert_pending_expert_tree(expert_dir)
        return status
    if status == COMPLETED_EXPERT_STATUS:
        assert_completed_expert_tree(expert_dir, source_dir=source_dir, root=root)
        return status
    raise ValueError(f"expert status must be pending or completed, got {status!r}")

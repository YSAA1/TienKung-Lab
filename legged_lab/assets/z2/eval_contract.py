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

"""Isaac-free Z2 training/eval contract and G1-comparison provenance.

``eval_locomotion.py`` loads the live task registry. It does not apply
``train.py`` ``--g1_progress_ab`` / ``--g1_motion_experiment`` flags. A fair
G1 vital_v1 baseline therefore needs the frozen env yaml from that run, not
those flags on the evaluator CLI.

This module lives under ``assets/z2`` so dump/tests can import it without
``legged_lab.envs`` (Isaac). Shared ``eval_locomotion.py`` must not import it.
"""

from __future__ import annotations

import json
from pathlib import Path

from legged_lab.assets.z2.schemas import (
    AMP_FIELDS,
    AMP_FRAME_DIM,
    UPSTREAM_VISUALIZATION_WIDTH,
)
from legged_lab.locomotion.mdp.sparse_signals import oob_linf_limit_m, promote_radius_m

ROOT = Path(__file__).resolve().parents[3]
# Matches ``AMP_LOCOMOTION_TERRAINS_CFG.size``; do not import the terrain cfg (Isaac).
LIGHTLP_TILE_SIZE_M = 8.0
PENDING_RUNTIME_CONFIG = "pending_isaac_export"
COMPLETED_EXPERT_RUNTIME = "isaac_fk"
ISAAC_PROBE_EVIDENCE = "artifacts/z2_migration/isaac_probe.json"
PHYSICS_USD_EVIDENCE = "artifacts/z2_migration/physics_original_usd/comparison.json"


def runtime_evidence_status(root: Path | None = None) -> str:
    base = root or ROOT
    expert = base / "legged_lab/envs/z2/datasets/motion_amp_expert/_manifest.json"
    if expert.is_file():
        status = json.loads(expert.read_text(encoding="utf-8")).get("status")
        if status == COMPLETED_EXPERT_RUNTIME:
            return COMPLETED_EXPERT_RUNTIME
    for rel in (ISAAC_PROBE_EVIDENCE, PHYSICS_USD_EVIDENCE):
        if (base / rel).is_file():
            return rel
    return PENDING_RUNTIME_CONFIG


def z2_training_progress_contract() -> dict:
    shared = (ROOT / "legged_lab/locomotion/teacher_cfg.py").read_text(encoding="utf-8")
    z2 = (ROOT / "legged_lab/envs/z2/teacher_cfg.py").read_text(encoding="utf-8")
    g1 = (ROOT / "legged_lab/envs/g1/teacher_cfg.py").read_text(encoding="utf-8")
    if 'self.lightlp_promotion_distance = "max_radial"' not in z2:
        raise ValueError("Z2 recipe must set max_radial in its own teacher_cfg")
    if "self.progress_monitor_enabled = True" not in z2:
        raise ValueError("Z2 recipe must enable the shared progress monitor")
    if "envs.g1" in z2 or "motion_experiment" in z2:
        raise ValueError("Z2 recipe must not import the G1 CLI/profile path")
    if "lightlp_promotion_distance" in g1 or "progress_monitor_enabled" in g1:
        raise ValueError("G1 recipe still does not own max_radial; do not copy a G1 field that is not there")
    if 'lightlp_promotion_distance: str = "path_length"' not in shared:
        raise ValueError("shared LightLP default is still path_length; do not silently rewrite it")
    promote_m = promote_radius_m(LIGHTLP_TILE_SIZE_M)
    oob_m = oob_linf_limit_m(LIGHTLP_TILE_SIZE_M)
    return {
        "task": "z2_loco_teacher",
        "lightlp_promotion_distance": "max_radial",
        "progress_monitor_enabled": True,
        "shared_default_promotion": "path_length",
        "tile_size_m": LIGHTLP_TILE_SIZE_M,
        "promotion_distance_m": promote_m,
        "oob_distance_m": oob_m,
        "oob_metric": "chebyshev_linf_from_tile_origin",
        "promotion_metric": "peak_planar_l2_from_tile_origin",
        "reach2m_source": "episode_max_command_direction_displacement",
        "monitor_radial_source": "terminal_episode_max_radial_dist_from_terrain_origin",
        "source": "legged_lab/envs/z2/teacher_cfg.py",
        "core_docs": (
            "docs/plans/2026-09-07--robot-neutral-locomotion.md",
            "legged_lab/locomotion/curriculum.py",
            "legged_lab/locomotion/mdp/sparse_signals.py",
        ),
        "not_from": ("train.py --g1_progress_ab", "train.py --g1_motion_experiment", "legged_lab.envs.g1"),
    }


def eval_locomotion_ignored_g1_flags(argv: list[str]) -> tuple[str, ...]:
    ignored = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg.startswith("--g1_"):
            ignored.append(arg)
            if "=" not in arg and i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                ignored.append(argv[i + 1])
                i += 1
        i += 1
    return tuple(ignored)


def reject_eval_g1_cli_flags(argv: list[str]) -> None:
    ignored = eval_locomotion_ignored_g1_flags(argv)
    if ignored:
        raise ValueError(
            "eval_locomotion.py does not apply train.py G1 experiment flags "
            f"{list(ignored)}. Use a frozen env yaml from the G1 run or an explicit "
            "profile helper. Registry config is used as-is."
        )


def z2_eval_provenance() -> dict:
    progress = z2_training_progress_contract()
    return {
        **progress,
        "amp_frame_dim": AMP_FRAME_DIM,
        "amp_schema": "+".join(f"{name}{width}" for name, width in AMP_FIELDS),
        "not_visualization_70d": "root_xyz3+euler3+q29+linvel3+angvel3+dq29",
        "visualization_width": UPSTREAM_VISUALIZATION_WIDTH,
        "plant": "Z2_29DOF_WALK_POSE_DAMPED_PD_CFG",
        "asset_mode": "upstream_usd",
        "usd_path": "legged_lab/assets/z2/usd/assembly.usd",
        "root_z": 0.75,
        "action_scale": 0.25,
        "sim_dt": 0.005,
        "decimation": 4,
        "resolved_runtime_config": runtime_evidence_status(),
        "g1_comparison": {
            "eval_cli_flags_apply_vital_v1": False,
            "eval_loads": "task_registry.get_cfgs(task)",
            "eval_does_not_load_saved_run_env_yaml": True,
            "requires": "frozen_run_env_yaml_or_explicit_profile_helper",
            "active_g1_note": "current G1 vital_v1 is a train.py profile, not eval_locomotion.py",
            "parent_owns_baseline_eval": True,
        },
    }

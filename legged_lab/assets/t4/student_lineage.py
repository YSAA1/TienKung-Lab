"""Isaac-free gates for sparse depth-student training.

The live student train entry must reject Stage E 1155D teachers and student
checkpoints, and must see evaluator manifests unless the operator explicitly
waives the teacher gate.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from legged_lab.assets.t4.schemas import TEACHER_ACTOR_OBS_DIM, TEACHER_SPARSE_ACTOR_OBS_DIM

SPARSE_TEACHER_TASK = "t4_loco_teacher_sparse"
STUDENT_WARMSTART_PREFIXES = ("depth_encoder.", "memory_s.", "student.", "scan_decoder.")


class StudentLineageError(ValueError):
    """Raised when a teacher checkpoint or evaluator manifest fails the gate."""


def inspect_teacher_payload(blob: Mapping[str, Any]) -> dict[str, Any]:
    if "model_state_dict" not in blob:
        raise StudentLineageError("checkpoint does not contain model_state_dict")
    state = blob["model_state_dict"]
    if any(key.startswith("student.") or key.startswith("depth_encoder.") for key in state):
        raise StudentLineageError("checkpoint looks like a depth student, not a frozen sparse teacher")
    if "actor.0.weight" not in state:
        raise StudentLineageError("checkpoint is missing actor.0.weight; expected a sparse teacher PPO file")
    obs_dim = int(state["actor.0.weight"].shape[1])
    return {"obs_dim": obs_dim, "iter": blob.get("iter")}


def _load_checkpoint_blob(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        raise StudentLineageError(f"{path} is not a dict checkpoint")
    return blob


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_eval_manifests(
    teacher_path: Path,
    eval_manifests: Iterable[Path],
    checkpoint_sha256: str,
) -> list[Path]:
    manifests = [Path(item) for item in eval_manifests]
    if not manifests:
        raise StudentLineageError(
            "sparse student train requires --teacher_eval_manifest JSON "
            "(or pass --allow_ungated_teacher for an explicit waiver)"
        )
    teacher_resolved = teacher_path.resolve()
    for manifest_path in manifests:
        if not manifest_path.is_file():
            raise StudentLineageError(f"teacher eval manifest not found: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise StudentLineageError(f"{manifest_path} is not a JSON object")
        task = payload.get("task")
        if task != SPARSE_TEACHER_TASK:
            raise StudentLineageError(f"{manifest_path} task={task!r}; expected {SPARSE_TEACHER_TASK!r}")
        listed = payload.get("checkpoint")
        if not listed:
            raise StudentLineageError(f"{manifest_path} is missing checkpoint")
        listed_path = Path(str(listed))
        if not listed_path.is_absolute():
            listed_path = manifest_path.parent / listed_path
        if listed_path.resolve() != teacher_resolved:
            raise StudentLineageError(
                f"{manifest_path} checkpoint {listed!r} does not match teacher path {teacher_resolved}"
            )
        listed_sha256 = payload.get("checkpoint_sha256")
        if listed_sha256 is not None and str(listed_sha256).lower() != checkpoint_sha256:
            raise StudentLineageError(f"{manifest_path} checkpoint_sha256 does not match the teacher checkpoint")
    return manifests


def require_sparse_teacher_checkpoint(
    path: str | Path,
    *,
    eval_manifests: Iterable[Path] = (),
    allow_ungated: bool = False,
) -> dict[str, Any]:
    """Validate a frozen S12 sparse teacher before distillation.

    Structural 1937D check is never skipped. Evaluator manifests are required
    unless ``allow_ungated`` is set.
    """
    teacher_path = Path(path)
    blob = _load_checkpoint_blob(teacher_path)
    info = inspect_teacher_payload(blob)
    obs_dim = int(info["obs_dim"])
    if obs_dim != TEACHER_SPARSE_ACTOR_OBS_DIM:
        hint = ""
        if obs_dim == TEACHER_ACTOR_OBS_DIM:
            hint = " (this looks like Stage E 1155D; do not distill into the sparse student)"
        raise StudentLineageError(
            f"{teacher_path} actor obs dim is {obs_dim}, expected {TEACHER_SPARSE_ACTOR_OBS_DIM}{hint}"
        )
    checkpoint_sha256 = _checkpoint_sha256(teacher_path)
    info["sha256"] = checkpoint_sha256
    if allow_ungated:
        info["ungated"] = True
        info["manifests"] = []
        return info
    info["ungated"] = False
    info["manifests"] = [str(item) for item in _require_eval_manifests(teacher_path, eval_manifests, checkpoint_sha256)]
    return info


def _is_student_warmstart_key(name: str) -> bool:
    return name == "std" or name.startswith(STUDENT_WARMSTART_PREFIXES)


def load_student_warmstart_checkpoint(policy, path: str | Path) -> dict[str, Any]:
    """Load only the deployable/reconstruction student stack and reset all training state.

    Teacher and critic parameters stay at their current values. The caller owns a
    freshly constructed optimizer, so Adam moments and safe-update counters are
    deliberately not migrated.
    """

    checkpoint_path = Path(path)
    blob = _load_checkpoint_blob(checkpoint_path)
    source_state = blob["model_state_dict"]
    if not isinstance(source_state, Mapping):
        raise StudentLineageError(f"{checkpoint_path} model_state_dict is not a mapping")

    target_state = policy.state_dict()
    expected_keys = {name for name in target_state if _is_student_warmstart_key(name)}
    source_keys = {name for name in source_state if _is_student_warmstart_key(name)}
    missing_keys = sorted(expected_keys - source_keys)
    unexpected_keys = sorted(source_keys - expected_keys)
    if missing_keys:
        raise StudentLineageError(
            f"{checkpoint_path} is not a complete GRU student warm-start; missing {missing_keys[:5]}"
        )
    if unexpected_keys:
        raise StudentLineageError(
            f"{checkpoint_path} student architecture does not match the target; unexpected {unexpected_keys[:5]}"
        )

    selected_state = {name: source_state[name] for name in sorted(expected_keys)}
    try:
        torch.nn.Module.load_state_dict(policy, selected_state, strict=False)
    except RuntimeError as exc:
        raise StudentLineageError(f"{checkpoint_path} student warm-start shape mismatch: {exc}") from exc

    return {
        "path": str(checkpoint_path),
        "sha256": _checkpoint_sha256(checkpoint_path),
        "iter": blob.get("iter"),
        "loaded_keys": len(selected_state),
        "optimizer_reset": True,
        "critic_reset": True,
    }

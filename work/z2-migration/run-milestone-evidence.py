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

"""Standalone eval/play wrapper that applies a milestone profile before env construction.

This does not modify ``eval_locomotion.py``, ``play.py``, ``train.py``, or training
packages. ``runpy.run_path(..., run_name='milestone_import')`` executes the target
script body (AppLauncher + imports) without the bottom ``__main__`` block. The
wrapper then patches ``task_registry.get_cfgs`` so ``evaluate()`` / ``play()``
receive the profile and seed 42 before they construct the env.

Profiles:
  registry   live task registry (Z2 default; also G1 without VITAL mutation)
  vital_v1   G1 only; calls apply_vital_motion_v1(env_cfg)
             (frozen G1 export; local G1 keeps it as a compatibility wrapper)

Wrapper flags are stripped from downstream ``sys.argv``. ``--g1_*`` is rejected
before simulation in both modes: frozen G1 eval/play would silently drop those
flags. ``--seed`` other than 42 is rejected so play.update_rsl_rl_cfg cannot
override the protocol seed after the wrapper snapshot.

Optional ``--amp-expert-manifest`` copies the exact clip+sha256 loop from train.py.
It is not a guessed full17 path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import runpy
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

MILESTONE_SEED = 42
ALLOWED_PROFILES = ("registry", "vital_v1")
ALLOWED_MODES = ("eval", "play")
RUN_NAME = "milestone_import"
EVAL_CLOSE_TIMEOUT_S = 90.0  # matches eval_locomotion.py __main__
PLAY_CLOSE_TIMEOUT_S = 90.0  # play.py has no hang timer; wrapper bounds close
TARGET_SCRIPTS = {
    "eval": Path("legged_lab") / "scripts" / "eval_locomotion.py",
    "play": Path("legged_lab") / "scripts" / "play.py",
}
SOURCE_HASH_RELATIVE = {
    "eval_locomotion.py": Path("legged_lab") / "scripts" / "eval_locomotion.py",
    "play.py": Path("legged_lab") / "scripts" / "play.py",
    "train.py": Path("legged_lab") / "scripts" / "train.py",
    "motion_experiment.py": Path("legged_lab") / "envs" / "g1" / "motion_experiment.py",
    "task_registry.py": Path("legged_lab") / "utils" / "task_registry.py",
}


def parse_wrapper_argv(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Milestone evidence wrapper for existing eval_locomotion.py / play.py. "
            "Wrapper flags are not forwarded. Pass script flags after the wrapper flags."
        )
    )
    parser.add_argument("--mode", required=True, choices=ALLOWED_MODES)
    parser.add_argument("--profile", required=True, choices=ALLOWED_PROFILES)
    parser.add_argument("--project-root", type=Path, required=True, help="Source checkout used for imports and scripts")
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument(
        "--amp-expert-manifest",
        "--amp_expert_manifest",
        dest="amp_expert_manifest",
        type=Path,
        default=None,
        help="Optional. Exact train.py clips+sha256 plumbing; do not guess a default path",
    )
    wrapper, downstream = parser.parse_known_args(argv)
    return wrapper, list(downstream)


def resolve_wrapper_paths(wrapper: argparse.Namespace, cwd: Path | None = None) -> SimpleNamespace:
    cwd = Path.cwd() if cwd is None else cwd
    project_root = wrapper.project_root if wrapper.project_root.is_absolute() else cwd / wrapper.project_root
    evidence = wrapper.evidence_manifest
    evidence = evidence if evidence.is_absolute() else cwd / evidence
    amp = wrapper.amp_expert_manifest
    if amp is not None:
        amp = amp if amp.is_absolute() else cwd / amp
        amp = amp.resolve()
    project_root = project_root.resolve()
    target = project_root / TARGET_SCRIPTS[wrapper.mode]
    if not target.is_file():
        raise FileNotFoundError(f"target script missing: {target}")
    if amp is not None and not amp.is_file():
        raise FileNotFoundError(f"AMP expert manifest missing: {amp}")
    return SimpleNamespace(
        project_root=project_root,
        evidence_manifest=evidence.resolve(),
        amp_expert_manifest=amp,
        target_script=target,
    )


def configure_source_root(project_root: Path) -> None:
    """Prefer an explicit G1/Z2 checkout over whatever PYTHONPATH the process inherited."""
    root = str(project_root.resolve())
    while root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    rsl = project_root / "rsl_rl"
    if rsl.is_dir():
        rsl_s = str(rsl.resolve())
        while rsl_s in sys.path:
            sys.path.remove(rsl_s)
        sys.path.insert(1, rsl_s)
    os.chdir(root)


def downstream_sys_argv(target_script: Path, downstream: list[str]) -> list[str]:
    return [str(target_script)] + list(downstream)


def downstream_flag_value(downstream: list[str], flag: str) -> str | None:
    prefix = flag + "="
    for index, item in enumerate(downstream):
        if item == flag:
            if index + 1 < len(downstream) and not downstream[index + 1].startswith("-"):
                return downstream[index + 1]
            return None
        if item.startswith(prefix):
            return item[len(prefix) :]
    return None


def reject_g1_train_flags(downstream: list[str]) -> None:
    leftover = [arg for arg in downstream if arg.startswith("--g1_")]
    if leftover:
        raise ValueError(
            "wrapper rejects train.py --g1_* flags before simulation "
            f"{leftover}. Use --profile vital_v1; frozen G1 eval/play would drop them."
        )


def _seed_cli_value(downstream: list[str]) -> str | None:
    """Read --seed / --seed= including negative values such as --seed -1."""
    prefix = "--seed="
    for index, item in enumerate(downstream):
        if item.startswith(prefix):
            return item[len(prefix) :]
        if item == "--seed":
            if index + 1 >= len(downstream):
                return None
            return downstream[index + 1]
    return None


def reject_non_protocol_seed(downstream: list[str]) -> None:
    value = _seed_cli_value(downstream)
    if value is None:
        return
    if value != str(MILESTONE_SEED):
        raise ValueError(
            f"--seed must be {MILESTONE_SEED} (protocol seed); got {value!r}. "
            "play.update_rsl_rl_cfg would otherwise override the wrapper snapshot."
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_if_file(path: Path | None) -> dict:
    if path is None:
        return {"path": None, "exists": False, "sha256": None}
    resolved = Path(path)
    exists = resolved.is_file()
    return {
        "path": str(resolved),
        "exists": exists,
        "sha256": sha256_file(resolved) if exists else None,
    }


def source_file_hashes(project_root: Path, wrapper_path: Path) -> dict:
    hashed = {"run-milestone-evidence.py": hash_if_file(wrapper_path)}
    for label, relative in SOURCE_HASH_RELATIVE.items():
        hashed[label] = hash_if_file(project_root / relative)
    return hashed


def apply_train_amp_expert_manifest(agent_cfg, manifest_path: Path) -> list[str]:
    """Copied from ``legged_lab/scripts/train.py`` clip+sha256 assignment. No extra discovery."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest["clips"]:
        raise ValueError("AMP manifest has no clips")
    files = []
    for name, clip in manifest["clips"].items():
        path = manifest_path.parent / f"{name}.txt"
        if hashlib.sha256(path.read_bytes()).hexdigest() != clip["sha256"]:
            raise ValueError(f"AMP clip hash mismatch: {path}")
        files.append(str(path))
    agent_cfg.amp_expert_dir = str(manifest_path.parent)
    agent_cfg.amp_motion_files = files
    return files


def load_apply_vital_motion_v1_file(path: Path):
    """Load apply_vital_motion_v1(env_cfg) from frozen or local motion_experiment.py."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"apply_vital_motion_v1 source missing: {path}")
    spec = importlib.util.spec_from_file_location("milestone_motion_experiment", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    apply_vital = getattr(module, "apply_vital_motion_v1", None)
    if apply_vital is None:
        raise AttributeError(f"{path} does not export apply_vital_motion_v1(env_cfg)")
    return apply_vital


def load_apply_vital_motion_v1(project_root: Path):
    return load_apply_vital_motion_v1_file(Path(project_root) / SOURCE_HASH_RELATIVE["motion_experiment.py"])


def apply_profile_to_cfgs(
    env_cfg,
    agent_cfg,
    *,
    profile: str,
    seed: int,
    amp_manifest: Path | None,
    apply_vital,
) -> None:
    if profile not in ALLOWED_PROFILES:
        raise ValueError(f"profile must be registry or vital_v1, got {profile!r}")
    if profile == "vital_v1":
        if apply_vital is None:
            raise RuntimeError("vital_v1 requires apply_vital_motion_v1")
        apply_vital(env_cfg)
    agent_cfg.seed = seed
    scene = getattr(env_cfg, "scene", None)
    if scene is not None and hasattr(scene, "seed"):
        scene.seed = seed
    if amp_manifest is not None:
        apply_train_amp_expert_manifest(agent_cfg, amp_manifest)


def _attr(root, *names):
    current = root
    for name in names:
        current = getattr(current, name, None)
        if current is None:
            return None
    return current


def _jsonable(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


def collect_effective_fields(task_name, env_cfg, agent_cfg, *, profile: str, seed: int, amp_manifest: Path | None):
    motion_files = _attr(agent_cfg, "amp_motion_files")
    motion_count = len(motion_files) if motion_files is not None else None
    amp_record = {
        "manifest_supplied": amp_manifest is not None,
        "manifest_path": str(amp_manifest) if amp_manifest is not None else None,
        "motion_file_count": motion_count,
    }
    if amp_manifest is not None:
        declared = json.loads(Path(amp_manifest).read_text(encoding="utf-8"))
        amp_record["clip_sha256"] = {name: clip.get("sha256") for name, clip in declared.get("clips", {}).items()}
    return {
        "task": task_name,
        "profile": profile,
        "seed": seed,
        "robot_spec_name": _attr(env_cfg, "robot_spec", "name"),
        "acceleration_termination_enabled": _attr(env_cfg, "acceleration_termination_enabled"),
        "deterministic_fall_limits": _jsonable(_attr(env_cfg, "deterministic_fall_limits")),
        "gait_tracking_gate_enabled": _attr(env_cfg, "gait", "tracking_gate_enabled"),
        "action_rate_l2_weight": _attr(env_cfg, "reward", "action_rate_l2", "weight"),
        "lightlp_promotion_distance": _attr(env_cfg, "lightlp_promotion_distance"),
        "progress_monitor_enabled": _attr(env_cfg, "progress_monitor_enabled"),
        "sparse_command_min_speed_scale": _attr(env_cfg, "sparse_command_min_speed_scale"),
        "actor_hidden_dims": _jsonable(_attr(agent_cfg, "policy", "actor_hidden_dims")),
        "critic_hidden_dims": _jsonable(_attr(agent_cfg, "policy", "critic_hidden_dims")),
        "amp_frame_dim": _attr(agent_cfg, "amp_frame_dim"),
        "amp_discr_hidden_dims": _jsonable(_attr(agent_cfg, "amp_discr_hidden_dims")),
        "amp": amp_record,
    }


def install_get_cfgs_wrapper(
    registry,
    *,
    profile: str,
    seed: int,
    amp_manifest: Path | None,
    apply_vital,
    snapshot: dict,
):
    original = registry.get_cfgs

    def wrapped_get_cfgs(name):
        env_cfg, agent_cfg = original(name)
        apply_profile_to_cfgs(
            env_cfg,
            agent_cfg,
            profile=profile,
            seed=seed,
            amp_manifest=amp_manifest,
            apply_vital=apply_vital,
        )
        snapshot.clear()
        snapshot.update(
            collect_effective_fields(
                name,
                env_cfg,
                agent_cfg,
                profile=profile,
                seed=seed,
                amp_manifest=amp_manifest,
            )
        )
        snapshot["get_cfgs_consumed"] = True
        return env_cfg, agent_cfg

    registry.get_cfgs = wrapped_get_cfgs
    return wrapped_get_cfgs


def import_target_script(target_script: Path, downstream: list[str]) -> dict:
    sys.argv = downstream_sys_argv(target_script, downstream)
    return runpy.run_path(str(target_script), run_name=RUN_NAME)


def resolve_checkpoint_for_hash(mode: str, downstream: list[str], eval_result=None) -> Path | None:
    if isinstance(eval_result, dict) and eval_result.get("checkpoint"):
        result_path = Path(eval_result["checkpoint"])
        if result_path.is_file():
            return result_path
    flags = ("--checkpoint_path", "--checkpoint") if mode == "play" else ("--checkpoint", "--checkpoint_path")
    for flag in flags:
        value = downstream_flag_value(downstream, flag)
        if not value:
            continue
        path = Path(value).expanduser()
        if path.is_file():
            return path.resolve()
    return None


def compute_evidence_success(
    *,
    mode: str,
    snapshot: dict,
    checkpoint: dict,
    eval_result,
    downstream: list[str],
    exception: BaseException | None,
) -> bool:
    if exception is not None:
        return False
    if not snapshot.get("get_cfgs_consumed"):
        return False
    if not checkpoint.get("exists") or not checkpoint.get("sha256"):
        return False
    if mode == "eval":
        return isinstance(eval_result, dict) and "completed_episodes" in eval_result
    if mode == "play":
        record = downstream_flag_value(downstream, "--record")
        return bool(record) and Path(record).expanduser().is_file()
    return False


def simulation_close_timeout_s(mode: str) -> float:
    return EVAL_CLOSE_TIMEOUT_S if mode == "eval" else PLAY_CLOSE_TIMEOUT_S


def close_simulation(simulation_app, exit_code: int, timeout_s: float) -> None:
    import threading

    if simulation_app is not None:
        threading.Timer(timeout_s, os._exit, args=(exit_code,)).start()
        simulation_app.close()
    os._exit(exit_code)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_evidence_manifest(
    *,
    wrapper: argparse.Namespace,
    paths: SimpleNamespace,
    full_argv: list[str],
    downstream: list[str],
    snapshot: dict,
    eval_result,
    exception: BaseException | None,
    exit_code: int,
    wrapper_path: Path,
) -> dict:
    checkpoint = resolve_checkpoint_for_hash(wrapper.mode, downstream, eval_result)
    checkpoint_info = hash_if_file(checkpoint) if checkpoint is not None else hash_if_file(None)
    success = compute_evidence_success(
        mode=wrapper.mode,
        snapshot=snapshot,
        checkpoint=checkpoint_info,
        eval_result=eval_result,
        downstream=downstream,
        exception=exception,
    )
    play_record = downstream_flag_value(downstream, "--record")
    eval_output = downstream_flag_value(downstream, "--output")
    recorded_exit = exit_code if success else (1 if exit_code == 0 else exit_code)
    status = "success" if success else ("exception" if exception is not None else "incomplete")
    return {
        "wrapper": "run-milestone-evidence.py",
        "mode": wrapper.mode,
        "profile": wrapper.profile,
        "seed": MILESTONE_SEED,
        "project_root": str(paths.project_root),
        "target_script": str(paths.target_script),
        "full_argv": list(full_argv),
        "downstream_argv": list(downstream),
        "wrapper_flags_in_downstream_argv": False,
        "checkpoint": checkpoint_info,
        "source_file_hashes": source_file_hashes(paths.project_root, wrapper_path),
        "amp_expert_manifest": hash_if_file(paths.amp_expert_manifest),
        "effective_profile": dict(snapshot) if snapshot else None,
        "get_cfgs_consumed": bool(snapshot.get("get_cfgs_consumed")),
        "eval_output": eval_output,
        "play_record": play_record,
        "success": success,
        "status": status,
        "exception_type": None if exception is None else type(exception).__name__,
        "exception_message": None if exception is None else str(exception),
        "exit_code": recorded_exit,
        "close_timeout_s": simulation_close_timeout_s(wrapper.mode),
        "notes": [
            "eval_locomotion.py / play.py / train.py were not modified.",
            "vital_v1 is G1-only via apply_vital_motion_v1(env_cfg).",
            "AMP files are taken from registry unless --amp-expert-manifest is supplied.",
            "G1 training env.yaml / full17 datapaths are not guessed here.",
            "Runtime simulation validation is parent-owned.",
        ],
    }


def run_wrapped_script(wrapper, downstream, paths, imported: dict, snapshot: dict):
    reject_g1_train_flags(downstream)
    reject_non_protocol_seed(downstream)
    configure_source_root(paths.project_root)
    apply_vital = None
    if wrapper.profile == "vital_v1":
        apply_vital = load_apply_vital_motion_v1(paths.project_root)
    imported.update(import_target_script(paths.target_script, downstream))
    registry = imported.get("task_registry")
    if registry is None:
        from legged_lab.utils import task_registry as registry  # noqa: PLC0415
    install_get_cfgs_wrapper(
        registry,
        profile=wrapper.profile,
        seed=MILESTONE_SEED,
        amp_manifest=paths.amp_expert_manifest,
        apply_vital=apply_vital,
        snapshot=snapshot,
    )
    if wrapper.mode == "eval":
        return imported["evaluate"]()
    imported["play"]()
    return None


def _partial_paths(wrapper: argparse.Namespace) -> SimpleNamespace:
    cwd = Path.cwd()
    project_root = wrapper.project_root if wrapper.project_root.is_absolute() else cwd / wrapper.project_root
    evidence = wrapper.evidence_manifest
    evidence = evidence if evidence.is_absolute() else cwd / evidence
    amp = wrapper.amp_expert_manifest
    if amp is not None:
        amp = amp if amp.is_absolute() else cwd / amp
    target = project_root / TARGET_SCRIPTS[wrapper.mode]
    return SimpleNamespace(
        project_root=project_root.resolve(),
        evidence_manifest=evidence.resolve(),
        amp_expert_manifest=None if amp is None else amp.resolve(),
        target_script=target,
    )


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv if argv is None else argv)
    wrapper, downstream = parse_wrapper_argv(argv[1:])
    imported: dict = {}
    snapshot: dict = {}
    eval_result = None
    exception: BaseException | None = None
    exit_code = 0
    paths = _partial_paths(wrapper)
    try:
        paths = resolve_wrapper_paths(wrapper)
        eval_result = run_wrapped_script(wrapper, downstream, paths, imported, snapshot)
    except BaseException as exc:
        traceback.print_exc()
        exception = exc
        code = getattr(exc, "code", 1)
        exit_code = 1 if code in (None, "") else int(code) if isinstance(code, int) else 1
    finally:
        manifest = build_evidence_manifest(
            wrapper=wrapper,
            paths=paths,
            full_argv=argv,
            downstream=downstream,
            snapshot=snapshot,
            eval_result=eval_result,
            exception=exception,
            exit_code=exit_code,
            wrapper_path=Path(__file__).resolve(),
        )
        exit_code = int(manifest["exit_code"])
        try:
            write_json(paths.evidence_manifest, manifest)
        except BaseException:
            traceback.print_exc()
            exit_code = 1
        close_simulation(
            imported.get("simulation_app"),
            exit_code,
            simulation_close_timeout_s(wrapper.mode),
        )


if __name__ == "__main__":
    main()

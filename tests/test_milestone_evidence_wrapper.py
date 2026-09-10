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

"""CPU-only stubs for the milestone eval/play wrapper. No Isaac / no training."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import runpy
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
WRAPPER_PATH = ROOT / "work/z2-migration/run-milestone-evidence.py"
CAPTURED_FROZEN_HELPER = ROOT / "artifacts/z2_migration/formal_v1/g1_reference/motion_experiment.py"
LOCAL_HELPER = ROOT / "legged_lab/envs/g1/motion_experiment.py"


def load_wrapper():
    spec = importlib.util.spec_from_file_location("run_milestone_evidence", WRAPPER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_captured_vital():
    return load_wrapper().load_apply_vital_motion_v1_file(CAPTURED_FROZEN_HELPER)


def g1_env():
    return NS(
        robot_spec=NS(name="g1"),
        acceleration_termination_enabled=True,
        deterministic_fall_limits=None,
        gait=NS(tracking_gate_enabled=True),
        reward=NS(action_rate_l2=NS(weight=-0.1)),
        lightlp_promotion_distance="path_length",
        progress_monitor_enabled=False,
        sparse_command_min_speed_scale=0.5,
        scene=NS(seed=7),
    )


def agent_cfg():
    return NS(
        seed=7,
        policy=NS(actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128]),
        amp_frame_dim=70,
        amp_discr_hidden_dims=[1024, 512, 256],
        amp_motion_files=["placeholder.txt"],
        amp_expert_dir="placeholder",
    )


def test_wrapper_flags_are_stripped_and_g1_flags_stay_in_downstream():
    module = load_wrapper()
    wrapper, downstream = module.parse_wrapper_argv(
        [
            "--mode",
            "eval",
            "--profile",
            "vital_v1",
            "--project-root",
            "D:/g1-frozen",
            "--evidence-manifest",
            "out.json",
            "--task",
            "g1_loco_teacher",
            "--load_run",
            "2026-09-08_16-25-01_vital_motion_v1",
            "--checkpoint",
            "model_1000.pt",
            "--output",
            "eval.json",
            "--g1_motion_experiment",
            "vital_v1",
            "--g1_progress_ab",
            "B",
        ]
    )
    assert wrapper.mode == "eval"
    assert wrapper.profile == "vital_v1"
    assert downstream == [
        "--task",
        "g1_loco_teacher",
        "--load_run",
        "2026-09-08_16-25-01_vital_motion_v1",
        "--checkpoint",
        "model_1000.pt",
        "--output",
        "eval.json",
        "--g1_motion_experiment",
        "vital_v1",
        "--g1_progress_ab",
        "B",
    ]
    argv = module.downstream_sys_argv(Path("legged_lab/scripts/eval_locomotion.py"), downstream)
    joined = " ".join(argv)
    assert "--mode" not in argv
    assert "--profile" not in argv
    assert "--project-root" not in argv
    assert "--evidence-manifest" not in argv
    assert "--amp-expert-manifest" not in argv
    assert "--task" in argv
    assert "run-milestone-evidence.py" not in joined
    with pytest.raises(ValueError, match="--g1_"):
        module.reject_g1_train_flags(downstream)


def test_amp_manifest_alias_is_wrapper_only_not_forwarded(tmp_path):
    module = load_wrapper()
    manifest = tmp_path / "_manifest.json"
    wrapper, downstream = module.parse_wrapper_argv(
        [
            "--mode",
            "play",
            "--profile",
            "registry",
            "--project-root",
            str(tmp_path),
            "--evidence-manifest",
            str(tmp_path / "e.json"),
            "--amp_expert_manifest",
            str(manifest),
            "--task",
            "z2_loco_teacher",
            "--record",
            "out.mp4",
        ]
    )
    assert wrapper.amp_expert_manifest == manifest
    assert "--amp_expert_manifest" not in downstream
    assert "--amp-expert-manifest" not in downstream
    assert downstream == ["--task", "z2_loco_teacher", "--record", "out.mp4"]


def test_profile_is_applied_before_get_cfgs_return_is_consumed():
    module = load_wrapper()
    apply_vital = load_captured_vital()
    env = g1_env()
    agent = agent_cfg()
    seen_inside_original = {}

    class Registry:
        def get_cfgs(self, name):
            seen_inside_original["gait"] = env.gait.tracking_gate_enabled
            seen_inside_original["action_rate"] = env.reward.action_rate_l2.weight
            seen_inside_original["seed"] = agent.seed
            seen_inside_original["name"] = name
            return env, agent

    snapshot = {}
    registry = Registry()
    module.install_get_cfgs_wrapper(
        registry,
        profile="vital_v1",
        seed=42,
        amp_manifest=None,
        apply_vital=apply_vital,
        snapshot=snapshot,
    )
    assert env.gait.tracking_gate_enabled is True
    consumed_env, consumed_agent = registry.get_cfgs("g1_loco_teacher")
    assert seen_inside_original == {
        "gait": True,
        "action_rate": -0.1,
        "seed": 7,
        "name": "g1_loco_teacher",
    }
    assert consumed_env is env
    assert consumed_env.gait.tracking_gate_enabled is False
    assert consumed_env.acceleration_termination_enabled is False
    assert consumed_env.deterministic_fall_limits == (0.8, 1.0)
    assert consumed_env.reward.action_rate_l2.weight == -0.01
    assert consumed_env.lightlp_promotion_distance == "max_radial"
    assert consumed_env.progress_monitor_enabled is True
    assert consumed_agent.seed == 42
    assert consumed_env.scene.seed == 42
    assert snapshot["get_cfgs_consumed"] is True
    assert snapshot["gait_tracking_gate_enabled"] is False
    assert snapshot["action_rate_l2_weight"] == -0.01
    assert snapshot["actor_hidden_dims"] == [512, 256, 128]
    assert snapshot["amp_frame_dim"] == 70


def test_registry_profile_sets_seed_without_vital_mutation():
    module = load_wrapper()
    env = g1_env()
    env.robot_spec.name = "z2"
    agent = agent_cfg()

    class Registry:
        def get_cfgs(self, name):
            return env, agent

    snapshot = {}
    registry = Registry()
    module.install_get_cfgs_wrapper(
        registry,
        profile="registry",
        seed=42,
        amp_manifest=None,
        apply_vital=None,
        snapshot=snapshot,
    )
    got_env, got_agent = registry.get_cfgs("z2_loco_teacher")
    assert got_env.gait.tracking_gate_enabled is True
    assert got_env.reward.action_rate_l2.weight == -0.1
    assert got_env.acceleration_termination_enabled is True
    assert got_env.lightlp_promotion_distance == "path_length"
    assert got_agent.seed == 42
    assert got_env.scene.seed == 42
    assert snapshot["profile"] == "registry"
    assert snapshot["robot_spec_name"] == "z2"


def test_vital_v1_rejects_non_g1_before_mutation():
    module = load_wrapper()
    env = g1_env()
    env.robot_spec.name = "z2"
    agent = agent_cfg()

    class Registry:
        def get_cfgs(self, name):
            return env, agent

    registry = Registry()
    module.install_get_cfgs_wrapper(
        registry,
        profile="vital_v1",
        seed=42,
        amp_manifest=None,
        apply_vital=load_captured_vital(),
        snapshot={},
    )
    with pytest.raises(ValueError, match="G1-specific"):
        registry.get_cfgs("z2_loco_teacher")
    assert env.gait.tracking_gate_enabled is True
    assert env.reward.action_rate_l2.weight == -0.1
    assert agent.seed == 7


def test_unknown_profile_is_rejected():
    module = load_wrapper()
    env = g1_env()
    agent = agent_cfg()
    registry = NS(get_cfgs=lambda name: (env, agent))
    module.install_get_cfgs_wrapper(
        registry,
        profile="not_a_profile",
        seed=42,
        amp_manifest=None,
        apply_vital=None,
        snapshot={},
    )
    with pytest.raises(ValueError, match="registry or vital_v1"):
        registry.get_cfgs("g1_loco_teacher")


def test_runpy_milestone_import_skips_main(tmp_path):
    script = tmp_path / "eval_locomotion.py"
    script.write_text(
        "called_main = False\n"
        "def evaluate():\n"
        "    return {'ok': True}\n"
        "if __name__ == '__main__':\n"
        "    called_main = True\n"
        "    raise RuntimeError('main must not run')\n",
        encoding="utf-8",
    )
    imported = runpy.run_path(str(script), run_name="milestone_import")
    assert imported["called_main"] is False
    assert imported["evaluate"]() == {"ok": True}
    with pytest.raises(RuntimeError, match="main must not run"):
        runpy.run_path(str(script), run_name="__main__")


def test_train_amp_manifest_plumbing_matches_clip_sha256(tmp_path):
    module = load_wrapper()
    clip_a = tmp_path / "walk.txt"
    clip_b = tmp_path / "run.txt"
    clip_a.write_bytes(b"walk-bytes")
    clip_b.write_bytes(b"run-bytes")
    manifest = {
        "clips": {
            "walk": {"sha256": hashlib.sha256(b"walk-bytes").hexdigest()},
            "run": {"sha256": hashlib.sha256(b"run-bytes").hexdigest()},
        }
    }
    manifest_path = tmp_path / "_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    agent = agent_cfg()
    files = module.apply_train_amp_expert_manifest(agent, manifest_path)
    assert files == [str(clip_a), str(clip_b)]
    assert agent.amp_expert_dir == str(tmp_path)
    assert agent.amp_motion_files == files
    bad = json.loads(manifest_path.read_text(encoding="utf-8"))
    bad["clips"]["walk"]["sha256"] = "0" * 64
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="AMP clip hash mismatch"):
        module.apply_train_amp_expert_manifest(agent_cfg(), bad_path)
    empty_path = tmp_path / "empty.json"
    empty_path.write_text(json.dumps({"clips": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="AMP manifest has no clips"):
        module.apply_train_amp_expert_manifest(agent_cfg(), empty_path)


def test_evidence_manifest_records_argv_seed_and_exception(tmp_path):
    module = load_wrapper()
    wrapper, downstream = module.parse_wrapper_argv(
        [
            "--mode",
            "eval",
            "--profile",
            "registry",
            "--project-root",
            str(ROOT),
            "--evidence-manifest",
            str(tmp_path / "e.json"),
            "--task",
            "z2_loco_teacher",
            "--checkpoint",
            "model_1000.pt",
        ]
    )
    paths = module.resolve_wrapper_paths(wrapper, cwd=Path.cwd())
    payload = module.build_evidence_manifest(
        wrapper=wrapper,
        paths=paths,
        full_argv=["run-milestone-evidence.py", "--mode", "eval", "--task", "z2_loco_teacher"],
        downstream=downstream,
        snapshot={"get_cfgs_consumed": True, "seed": 42, "profile": "registry"},
        eval_result=None,
        exception=ValueError("g1 flags not applied"),
        exit_code=1,
        wrapper_path=WRAPPER_PATH,
    )
    assert payload["seed"] == 42
    assert payload["success"] is False
    assert payload["exception_type"] == "ValueError"
    assert payload["downstream_argv"] == downstream
    assert "--mode" not in payload["downstream_argv"]
    assert payload["source_file_hashes"]["eval_locomotion.py"]["exists"] is True
    assert payload["source_file_hashes"]["eval_locomotion.py"]["sha256"]
    assert payload["checkpoint"]["exists"] is False
    assert payload["success"] is False


def test_captured_frozen_g1_apply_vital_motion_v1_fields():
    captured = CAPTURED_FROZEN_HELPER.read_text(encoding="utf-8")
    assert "def apply_vital_motion_v1(" in captured
    assert "def apply_vital_motion_experiment" not in captured
    module = load_wrapper()
    apply_vital = module.load_apply_vital_motion_v1_file(CAPTURED_FROZEN_HELPER)
    env = g1_env()
    apply_vital(env)
    assert env.acceleration_termination_enabled is False
    assert env.deterministic_fall_limits == (0.8, 1.0)
    assert env.gait.tracking_gate_enabled is False
    assert env.reward.action_rate_l2.weight == -0.01
    assert env.lightlp_promotion_distance == "max_radial"
    assert env.progress_monitor_enabled is True
    assert env.sparse_command_min_speed_scale == 1.0


def test_loader_uses_exact_captured_helper_from_project_layout(tmp_path):
    dest = tmp_path / "legged_lab/envs/g1/motion_experiment.py"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(CAPTURED_FROZEN_HELPER.read_bytes())
    assert dest.read_bytes() == CAPTURED_FROZEN_HELPER.read_bytes()
    apply_vital = load_wrapper().load_apply_vital_motion_v1(tmp_path)
    env = g1_env()
    apply_vital(env)
    assert env.gait.tracking_gate_enabled is False
    assert env.reward.action_rate_l2.weight == -0.01
    assert env.deterministic_fall_limits == (0.8, 1.0)


def test_local_helper_still_exports_apply_vital_motion_v1():
    apply_vital = load_wrapper().load_apply_vital_motion_v1_file(LOCAL_HELPER)
    env = g1_env()
    apply_vital(env)
    assert env.gait.tracking_gate_enabled is False
    assert env.reward.action_rate_l2.weight == -0.01
    assert env.acceleration_termination_enabled is False


def test_wrapper_rejects_g1_flags_before_simulation():
    module = load_wrapper()
    with pytest.raises(ValueError, match="--g1_"):
        module.reject_g1_train_flags(["--task", "g1_loco_teacher", "--g1_motion_experiment", "vital_v1"])
    with pytest.raises(ValueError, match="--g1_"):
        module.reject_g1_train_flags(["--g1_progress_ab=B"])
    module.reject_g1_train_flags(["--task", "g1_loco_teacher", "--checkpoint", "model_1000.pt"])


def test_rejects_seed_other_than_42_including_equals_form():
    module = load_wrapper()
    module.reject_non_protocol_seed([])
    module.reject_non_protocol_seed(["--seed", "42"])
    module.reject_non_protocol_seed(["--seed=42"])
    with pytest.raises(ValueError, match="42"):
        module.reject_non_protocol_seed(["--seed", "43"])
    with pytest.raises(ValueError, match="42"):
        module.reject_non_protocol_seed(["--seed=43"])
    with pytest.raises(ValueError, match="42"):
        module.reject_non_protocol_seed(["--seed", "-1"])
    with pytest.raises(ValueError, match="42"):
        module.reject_non_protocol_seed(["--seed=-1"])


def test_success_requires_get_cfgs_checkpoint_and_eval_or_record(tmp_path):
    module = load_wrapper()
    checkpoint = tmp_path / "model_1000.pt"
    checkpoint.write_bytes(b"ckpt")
    record = tmp_path / "replay.mp4"
    record.write_bytes(b"mp4")
    hashed = module.hash_if_file(checkpoint)
    snapshot = {"get_cfgs_consumed": True}
    eval_json = {"completed_episodes": 64, "checkpoint": str(checkpoint)}
    assert (
        module.compute_evidence_success(
            mode="eval",
            snapshot=snapshot,
            checkpoint=hashed,
            eval_result=eval_json,
            downstream=["--output", "eval.json"],
            exception=None,
        )
        is True
    )
    assert (
        module.compute_evidence_success(
            mode="eval",
            snapshot={},
            checkpoint=hashed,
            eval_result=eval_json,
            downstream=[],
            exception=None,
        )
        is False
    )
    assert (
        module.compute_evidence_success(
            mode="eval",
            snapshot=snapshot,
            checkpoint={"path": None, "exists": False, "sha256": None},
            eval_result=eval_json,
            downstream=[],
            exception=None,
        )
        is False
    )
    assert (
        module.compute_evidence_success(
            mode="eval",
            snapshot=snapshot,
            checkpoint=hashed,
            eval_result=None,
            downstream=[],
            exception=None,
        )
        is False
    )
    assert (
        module.compute_evidence_success(
            mode="eval",
            snapshot=snapshot,
            checkpoint=hashed,
            eval_result={"ok": True},
            downstream=[],
            exception=None,
        )
        is False
    )
    assert (
        module.compute_evidence_success(
            mode="eval",
            snapshot=snapshot,
            checkpoint=hashed,
            eval_result=eval_json,
            downstream=[],
            exception=SystemExit(0),
        )
        is False
    )
    assert (
        module.compute_evidence_success(
            mode="play",
            snapshot=snapshot,
            checkpoint=hashed,
            eval_result=None,
            downstream=["--record", str(record)],
            exception=None,
        )
        is True
    )
    assert (
        module.compute_evidence_success(
            mode="play",
            snapshot=snapshot,
            checkpoint=hashed,
            eval_result=None,
            downstream=["--record", str(tmp_path / "missing.mp4")],
            exception=None,
        )
        is False
    )
    assert (
        module.compute_evidence_success(
            mode="play",
            snapshot=snapshot,
            checkpoint=hashed,
            eval_result=None,
            downstream=[],
            exception=None,
        )
        is False
    )


def test_build_manifest_help_exit_is_not_success(tmp_path):
    module = load_wrapper()
    wrapper, downstream = module.parse_wrapper_argv(
        [
            "--mode",
            "eval",
            "--profile",
            "registry",
            "--project-root",
            str(ROOT),
            "--evidence-manifest",
            str(tmp_path / "e.json"),
            "--task",
            "z2_loco_teacher",
        ]
    )
    paths = module.resolve_wrapper_paths(wrapper, cwd=Path.cwd())
    payload = module.build_evidence_manifest(
        wrapper=wrapper,
        paths=paths,
        full_argv=["run-milestone-evidence.py", "--help"],
        downstream=downstream,
        snapshot={},
        eval_result=None,
        exception=SystemExit(0),
        exit_code=0,
        wrapper_path=WRAPPER_PATH,
    )
    assert payload["success"] is False
    assert payload["exit_code"] == 1
    assert payload["get_cfgs_consumed"] is False

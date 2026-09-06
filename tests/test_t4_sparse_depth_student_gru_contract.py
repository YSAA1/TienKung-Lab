# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""Isaac-free contracts for the S12 sparse GRU depth student."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from legged_lab.assets.t4 import schemas

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rsl_rl"))

from rsl_rl.algorithms.distillation import Distillation  # noqa: E402
from rsl_rl.modules.depth_student_teacher import (  # noqa: E402
    DepthStudentTeacher,
    DepthStudentTeacherRecurrent,
    build_depth_student_policy,
    strip_deployable_state_dict,
    write_deployable_checkpoint,
)

from legged_lab.assets.t4.student_lineage import (  # noqa: E402
    StudentLineageError,
    load_student_continuation_checkpoint,
    load_student_warmstart_checkpoint,
    require_json_gate,
    require_sparse_teacher_checkpoint,
)

_NOISE_PATH = ROOT / "legged_lab" / "envs" / "t4" / "mdp" / "depth_noise.py"
_noise_spec = importlib.util.spec_from_file_location("t4_depth_noise", _NOISE_PATH)
noise = importlib.util.module_from_spec(_noise_spec)
_noise_spec.loader.exec_module(noise)
_EXTRINSIC_PATH = ROOT / "legged_lab" / "envs" / "t4" / "mdp" / "camera_extrinsic.py"
_extrinsic_spec = importlib.util.spec_from_file_location("t4_camera_extrinsic", _EXTRINSIC_PATH)
extrinsic = importlib.util.module_from_spec(_extrinsic_spec)
_extrinsic_spec.loader.exec_module(extrinsic)

ENV_PY = ROOT / "legged_lab" / "envs" / "t4" / "depth_student_env.py"
RUNTIME_PY = ROOT / "legged_lab/locomotion/depth_env.py"
CFG_PY = ROOT / "legged_lab" / "envs" / "t4" / "depth_student_cfg.py"
INIT_PY = ROOT / "legged_lab" / "envs" / "__init__.py"
TEACHER_PY = ROOT / "legged_lab" / "locomotion" / "teacher_cfg.py"
TRAIN_PY = ROOT / "legged_lab" / "scripts" / "train_t4_sparse_depth_student.py"
FT_PY = ROOT / "legged_lab" / "scripts" / "train_t4_sparse_depth_student_ft.py"
SIM2SIM_PY = ROOT / "legged_lab" / "scripts" / "sim2sim_t4_depth_student.py"
T4_ENV_PY = ROOT / "legged_lab" / "locomotion" / "env.py"
PLAY_PY = ROOT / "legged_lab" / "scripts" / "play.py"
EVAL_PY = ROOT / "legged_lab" / "scripts" / "eval_locomotion.py"
PROBE_PY = ROOT / "legged_lab" / "scripts" / "probe_t4_student_depth_render.py"


def _tiny_gru(**kwargs):
    defaults = dict(
        num_student_obs=20,
        num_teacher_obs=12,
        num_actions=2,
        depth_shape=(1, 4, 4),
        proprio_obs_dim=4,
        depth_hidden_dim=8,
        student_hidden_dims=[8],
        teacher_hidden_dims=[8],
        rnn_type="gru",
        rnn_hidden_dim=8,
        rnn_num_layers=1,
        teacher_recurrent=False,
        recon_scan_dim=4,
        recon_scan_offset=4,
        init_noise_std=1.0e-6,
    )
    defaults.update(kwargs)
    return DepthStudentTeacherRecurrent(**defaults)


def _student_obs(n=3):
    proprio = 4
    depth = 1 * 4 * 4
    return torch.zeros(n, proprio + depth)


def test_sparse_student_env_inherits_s12_teacher_mdp():
    env_src = ENV_PY.read_text(encoding="utf-8") + RUNTIME_PY.read_text(encoding="utf-8")
    cfg_src = CFG_PY.read_text(encoding="utf-8")
    sparse_alg = cfg_src.split("class T4SparseDepthStudentDaggerAlgCfg", 1)[1].split(
        "class T4SparseDepthStudentFinalMainAlgCfg", 1
    )[0]
    final_main = cfg_src.split("class T4SparseDepthStudentFinalMainAlgCfg", 1)[1].split(
        "class T4SparseDepthStudentAgentCfg", 1
    )[0]
    agent = cfg_src.split("class T4SparseDepthStudentAgentCfg", 1)[1].split(
        "class T4SparseDepthStudentReprFirstAlgCfg", 1
    )[0]
    assert "class T4LocoSparseDepthStudentEnvCfg(T4LocoSparseTeacherEnvCfg)" in env_src
    assert "class LightLPDepthDistillationEnv(DepthDistillationEnv)" in env_src
    assert "self.observation_layout.actor_dim" in env_src
    assert "self.student_observation_dim" in env_src
    assert "student_depth_noise" in env_src
    assert 'class_name: str = "DepthStudentTeacherRecurrent"' in cfg_src
    assert 'rnn_type: str = "gru"' in cfg_src
    assert "teacher_recurrent: bool = False" in cfg_src
    assert "pg_coef: float = 0.0" in sparse_alg
    assert "pg_delay_iters: int = 0" in sparse_alg
    assert "teacher_mix_decay_iters: int = 1000" in sparse_alg
    assert "pg_coef: float = 0.2" in final_main
    assert "pg_coef_ramp_iters: int = 800" in final_main
    assert "pg_delay_iters: int = 2000" in final_main
    assert "critic_warmup_iters: int = 200" in final_main
    assert "teacher_mix_decay_iters: int = 2000" in final_main
    assert "learning_rate: float = 1.0e-4" in final_main
    assert "recon_coef: float = 1.0" in cfg_src
    assert 'experiment_name: str = "t4_loco_sparse_depth_student"' in cfg_src
    assert 'run_name: str = "s12_final_main"' in agent
    assert "max_iterations: int = 15000" in agent
    assert "T4SparseDepthStudentFinalMainAlgCfg" in agent
    assert "behavior_coef_decay_iters: int = 0" in sparse_alg
    assert "teacher_mix: float = 1.0" in sparse_alg
    assert 'schedule: str = "fixed"' in sparse_alg
    assert "learning_rate: float = 1.0e-4" in sparse_alg
    ft_agent = cfg_src.split("class T4SparseDepthStudentFtAgentCfg", 1)[1]
    assert "class T4SparseDepthStudentJointAlgCfg" in cfg_src
    assert "class T4SparseDepthStudentDeployFtAlgCfg" in cfg_src
    assert 'run_name: str = "s12_rtx_gated_joint"' in ft_agent
    assert 'run_name: str = "s12_rtx_deploy_ft_v2"' in cfg_src


def test_stage_e_student_lineage_is_unchanged():
    cfg_src = CFG_PY.read_text(encoding="utf-8")
    init_src = INIT_PY.read_text(encoding="utf-8")
    teacher_src = TEACHER_PY.read_text(encoding="utf-8")
    assert 'class_name: str = "DepthStudentTeacher"' in cfg_src.split("class T4SparseDepthStudentPolicyCfg")[0]
    assert 'run_name: str = "stage_s_head35"' in cfg_src
    assert 'experiment_name: str = "t4_loco_depth_student"' in cfg_src
    assert "t4_loco_sparse_depth_student" in init_src
    assert 'task_registry.register("t4_loco_teacher"' in init_src
    assert "rnn_type" not in teacher_src
    assert "DepthStudentTeacherRecurrent" not in teacher_src


def test_train_scripts_require_sparse_teacher_checkpoint():
    train_src = TRAIN_PY.read_text(encoding="utf-8")
    ft_src = FT_PY.read_text(encoding="utf-8")
    assert "T4LocoSparseDepthStudentReprFirstEnvCfg" in train_src
    assert "T4SparseDepthStudentReprFirstAgentCfg" in train_src
    assert "teacher_checkpoint" in train_src
    assert "student_warmstart_checkpoint" in train_src
    assert "student_lineage.json" in train_src
    assert "resume and --student_warmstart_checkpoint are mutually exclusive" in train_src
    assert "Resuming student from:" in train_src
    teacher_load_at = train_src.index("runner.load(str(teacher_path), load_optimizer=False)")
    resume_load_at = train_src.index("runner.load(resume_path, load_optimizer=load_optimizer)")
    assert teacher_load_at < resume_load_at
    assert "agent_cfg.algorithm.teacher_mix" in train_src
    assert "agent_cfg.algorithm.pg_coef" in train_src
    assert "agent_cfg.algorithm.schedule" in train_src
    assert "agent_cfg.algorithm.learning_rate" in train_src
    assert '"phase": "representation"' in train_src
    assert "attach_repr_runtime" in train_src
    assert "teacher_eval_manifest" in train_src
    assert "allow_ungated_teacher" in train_src
    assert "require_sparse_teacher_checkpoint" in train_src
    assert "legged_lab.assets.t4.student_lineage" in train_src
    assert "OnPolicyRunner" in train_src
    assert "args_cli.enable_cameras = True" in train_src
    assert 'parser.add_argument("--task_num_envs", type=int, default=256)' in train_src
    assert "pg_delay_iters" in train_src
    assert "student_depth_noise" in train_src
    assert "random_level_reset_min_level" in train_src
    assert "random_level_reset_max_level" in train_src
    assert "entropy_coef" in train_src
    assert "T4LocoSparseDepthStudentFtEnvCfg" in ft_src
    assert "student_checkpoint" in ft_src
    assert 'choices=("joint", "deploy_ft", "targeted_ft", "residual_ft", "plant_ft")' in ft_src
    assert "T4SparseDepthStudentTargetedFtAgentCfg" in ft_src
    assert "T4SparseDepthStudentResidualFtAgentCfg" in ft_src
    assert "T4LocoSparseDepthStudentResidualFtEnvCfg" in ft_src
    assert "T4SparseDepthStudentPlantFtAgentCfg" in ft_src
    assert "T4LocoSparseDepthStudentPlantFtEnvCfg" in ft_src
    assert 'phase = "plant_ft"' in ft_src
    assert "set_reference_policy_from_current" in ft_src
    assert 'phase = "targeted_ft"' in ft_src
    assert 'phase = "residual_ft"' in ft_src
    assert "capability_gate_json" in ft_src
    assert "teacher_eval_manifest" in ft_src
    assert 'required=True' in ft_src.split("--capability_gate_json", 1)[1].split(")", 1)[0]
    assert 'required=True' in ft_src.split("--student_checkpoint", 1)[1].split(")", 1)[0]
    assert 'required=True' in ft_src.split("--teacher_checkpoint", 1)[1].split(")", 1)[0]
    assert "allow_ungated" not in ft_src
    assert "load_student_continuation_checkpoint" in ft_src
    assert "require_sparse_teacher_checkpoint" in ft_src
    assert "T4LocoDepthStudentEnvCfg" not in train_src
    assert "T4DepthStudentAgentCfg" not in train_src
    assert "runner.current_learning_iteration = 0" in ft_src
    assert "runner.alg.num_updates = 0" in ft_src


def test_student_obs_has_no_scan_or_contact_privilege():
    assert schemas.STUDENT_ACTOR_OBS_DIM == (
        schemas.PROPRIO_FRAME_DIM * schemas.STUDENT_PROPRIO_HISTORY_LENGTH
        + schemas.DEPTH_POLICY_SIZE[0] * schemas.DEPTH_POLICY_SIZE[1] * schemas.STUDENT_DEPTH_HISTORY_LENGTH
    )
    start, end = schemas.sparse_teacher_latest_scan_range()
    assert end - start == schemas.TEACHER_SCAN_DIM
    scan_start, scan_end = schemas.sparse_teacher_scan_range()
    assert scan_end - scan_start == schemas.TEACHER_SCAN_DIM * schemas.TEACHER_SPARSE_SCAN_HISTORY_LENGTH
    assert schemas.TEACHER_SPARSE_ACTOR_OBS_DIM - scan_end == schemas.TEACHER_SPARSE_CONTACT_DIM
    assert schemas.PROPRIO_HISTORY_LENGTH == 10
    assert schemas.DEPTH_HISTORY_LENGTH == 3
    assert schemas.STUDENT_PROPRIO_HISTORY_LENGTH == 1
    assert schemas.STUDENT_DEPTH_HISTORY_LENGTH == 1
    assert schemas.STUDENT_ACTOR_OBS_DIM == (
        schemas.PROPRIO_FRAME_DIM * schemas.STUDENT_PROPRIO_HISTORY_LENGTH
        + schemas.DEPTH_POLICY_SIZE[0] * schemas.DEPTH_POLICY_SIZE[1] * schemas.STUDENT_DEPTH_HISTORY_LENGTH
    )
    assert schemas.STUDENT_ACTOR_OBS_DIM == 3168
    assert schemas.STAGE_E_STUDENT_ACTOR_OBS_DIM == (
        schemas.PROPRIO_FRAME_DIM * schemas.PROPRIO_HISTORY_LENGTH
        + schemas.DEPTH_POLICY_SIZE[0] * schemas.DEPTH_POLICY_SIZE[1] * schemas.DEPTH_HISTORY_LENGTH
    )
    assert schemas.STUDENT_ACTOR_OBS_DIM != schemas.TEACHER_SPARSE_ACTOR_OBS_DIM


def test_gru_rejects_lstm_and_recurrent_teacher():
    with pytest.raises(ValueError, match="gru"):
        _tiny_gru(rnn_type="lstm")
    with pytest.raises(ValueError, match="teacher_recurrent"):
        _tiny_gru(teacher_recurrent=True)


def test_gru_projects_raw_action_std_into_training_envelope():
    policy = _tiny_gru(min_action_std=0.05, max_action_std=0.2)
    with torch.no_grad():
        policy.std.copy_(torch.tensor([-0.1, 0.4]))

    policy.project_action_std_()
    policy.update_distribution(_student_obs(2))

    assert torch.allclose(policy.std, torch.tensor([0.05, 0.2]))
    assert policy.action_std.min().item() == pytest.approx(0.05)
    assert policy.action_std.max().item() == pytest.approx(0.2)


def test_gru_forward_reset_and_recon_are_training_only():
    policy = _tiny_gru()
    obs = _student_obs(4)
    teacher_obs = torch.zeros(4, 12)
    teacher_obs[:, 4:8] = 0.7
    first = policy.act_inference(obs)
    hidden_before = policy.get_hidden_states()[0].clone()
    recon = policy.reconstruct()
    assert recon.shape == (4, 4)
    assert torch.allclose(policy.teacher_scan(teacher_obs), teacher_obs[:, 4:8])
    second = policy.act_inference(obs)
    assert first.shape == (4, 2)
    assert not torch.equal(hidden_before, policy.get_hidden_states()[0])
    policy.reset()
    assert policy.get_hidden_states()[0] is None
    exported = policy.deployable_state_dict()
    assert not any(key.startswith("scan_decoder") for key in exported)
    assert not any(key.startswith("teacher.") for key in exported)
    assert any(key.startswith("memory_s") for key in exported)
    assert any(key.startswith("depth_encoder") for key in exported)
    assert "student.0.weight" in exported
    _ = second


def test_deployable_reload_has_no_decoder():
    policy = _tiny_gru()
    policy.act_inference(_student_obs(2))
    deployed = policy.deployable_state_dict()
    rebuilt = build_depth_student_policy(
        deployed,
        num_actions=2,
        num_teacher_obs=12,
        depth_shape=(1, 4, 4),
        proprio_obs_dim=4,
        recon_scan_offset=4,
    )
    assert rebuilt.scan_decoder is None
    out = rebuilt.act_inference(_student_obs(2))
    assert out.shape == (2, 2)


def _scratch_dir():
    return tempfile.TemporaryDirectory(prefix="t4-student-gate-")


def test_full_training_checkpoint_does_not_rebuild_decoder():
    policy = _tiny_gru(critic_hidden_dims=[8])
    policy.act_inference(_student_obs(2))
    full = policy.state_dict()
    assert "teacher.0.weight" in full
    assert "scan_decoder.2.weight" in full
    assert "critic.0.weight" in full
    rebuilt = build_depth_student_policy(
        full,
        num_actions=2,
        num_teacher_obs=12,
        depth_shape=(1, 4, 4),
        proprio_obs_dim=4,
        recon_scan_offset=4,
    )
    assert rebuilt.scan_decoder is None
    assert not any(key.startswith("scan_decoder") for key in rebuilt.state_dict())
    stripped = strip_deployable_state_dict(full)
    assert not any(key.startswith("teacher.") for key in stripped)
    assert not any(key.startswith("scan_decoder") for key in stripped)
    assert not any(key.startswith("critic.") for key in stripped)
    with _scratch_dir() as raw:
        scratch = Path(raw)
        src = scratch / "model_500.pt"
        dst = scratch / "model_500_deploy.pt"
        torch.save(
            {
                "model_state_dict": full,
                "optimizer_state_dict": {"not": "exported"},
                "iter": 500,
                "infos": None,
            },
            src,
        )
        written = write_deployable_checkpoint(src, dst)
        assert written["deployable"] is True
        assert "optimizer_state_dict" not in written
        assert not any(key.startswith("teacher.") for key in written["model_state_dict"])
        assert not any(key.startswith("scan_decoder") for key in written["model_state_dict"])
        assert not any(key.startswith("critic.") for key in written["model_state_dict"])


def test_student_warmstart_loads_control_stack_but_not_teacher_or_critic():
    source = _tiny_gru(critic_hidden_dims=[8], init_noise_std=0.1, min_action_std=0.05, max_action_std=0.2)
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            if name.startswith(("depth_encoder.", "memory_s.", "student.", "scan_decoder.")) or name == "std":
                parameter.fill_(0.25)
            elif name.startswith("teacher."):
                parameter.fill_(0.75)
            elif name.startswith("critic."):
                parameter.fill_(0.9)

    target = _tiny_gru(critic_hidden_dims=[8], init_noise_std=0.1, min_action_std=0.05, max_action_std=0.2)
    teacher_before = {name: value.clone() for name, value in target.teacher.state_dict().items()}
    critic_before = {name: value.clone() for name, value in target.critic.state_dict().items()}

    with _scratch_dir() as raw:
        checkpoint = Path(raw) / "model_3000.pt"
        torch.save({"model_state_dict": source.state_dict(), "iter": 3000}, checkpoint)
        info = load_student_warmstart_checkpoint(target, checkpoint)

    assert info["iter"] == 3000
    assert info["loaded_keys"] > 0
    assert len(info["sha256"]) == 64
    assert info["reset_action_std"] is True
    for name, value in target.state_dict().items():
        if name.startswith(("depth_encoder.", "memory_s.", "student.", "scan_decoder.")):
            assert torch.allclose(value, torch.full_like(value, 0.25))
        elif name == "std":
            assert torch.allclose(value, torch.full_like(value, 0.1))
    for name, value in target.teacher.state_dict().items():
        assert torch.equal(value, teacher_before[name])
    for name, value in target.critic.state_dict().items():
        assert torch.equal(value, critic_before[name])


def test_student_warmstart_resets_legacy_std_to_init_not_the_cap():
    source = _tiny_gru(
        critic_hidden_dims=[8],
        init_noise_std=0.1,
        min_action_std=0.05,
        max_action_std=0.8,
    )
    with torch.no_grad():
        source.std.fill_(0.4)
    target = _tiny_gru(
        critic_hidden_dims=[8],
        init_noise_std=0.1,
        min_action_std=0.05,
        max_action_std=0.2,
    )

    with _scratch_dir() as raw:
        checkpoint = Path(raw) / "model_3000.pt"
        torch.save({"model_state_dict": source.state_dict(), "iter": 3000}, checkpoint)
        info = load_student_warmstart_checkpoint(target, checkpoint)

    assert torch.allclose(target.std, torch.full_like(target.std, 0.1))
    assert info["source_action_std"]["max"] == pytest.approx(0.4)
    assert info["init_action_std"] == pytest.approx(0.1)
    assert info["reset_action_std"] is True
    assert info["effective_action_std"]["max"] == pytest.approx(0.1)
    assert info["effective_action_std"]["min"] == pytest.approx(0.1)


def test_continuation_loads_full_student_and_critic_resets_std_and_pins_teacher():
    source = _tiny_gru(critic_hidden_dims=[8], init_noise_std=0.1, min_action_std=0.05, max_action_std=0.2)
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            if name.startswith(("depth_encoder.", "memory_s.", "student.", "scan_decoder.", "critic.")):
                parameter.fill_(0.25)
            elif name.startswith("teacher."):
                parameter.fill_(0.75)
        source.std.fill_(0.19)
    target = _tiny_gru(critic_hidden_dims=[8], init_noise_std=0.1, min_action_std=0.05, max_action_std=0.2)

    with _scratch_dir() as raw:
        scratch = Path(raw)
        parent = scratch / "model_4000.pt"
        teacher = scratch / "model_21500.pt"
        teacher_state = {
            name.replace("teacher.", "actor.", 1): value.clone()
            for name, value in source.state_dict().items()
            if name.startswith("teacher.")
        }
        torch.save({"model_state_dict": source.state_dict(), "iter": 4000}, parent)
        torch.save({"model_state_dict": teacher_state, "iter": 21500}, teacher)
        info = load_student_continuation_checkpoint(target, parent, teacher, reset_action_std=0.08)

    assert info["iter"] == 4000
    assert info["reset_action_std"] == pytest.approx(0.08)
    assert len(info["sha256"]) == 64
    assert len(info["teacher_sha256"]) == 64
    assert torch.allclose(target.std, torch.full_like(target.std, 0.08))
    for name, value in target.state_dict().items():
        if name.startswith(("depth_encoder.", "memory_s.", "student.", "scan_decoder.", "critic.")):
            assert torch.allclose(value, torch.full_like(value, 0.25))
        elif name.startswith("teacher."):
            assert torch.allclose(value, torch.full_like(value, 0.75))


def test_continuation_rejects_mismatched_teacher_and_deployable_parent():
    source = _tiny_gru(critic_hidden_dims=[8])
    target = _tiny_gru(critic_hidden_dims=[8])
    with _scratch_dir() as raw:
        scratch = Path(raw)
        parent = scratch / "parent.pt"
        deployable = scratch / "deployable.pt"
        teacher = scratch / "teacher.pt"
        teacher_state = {
            name.replace("teacher.", "actor.", 1): value.clone() + 1.0
            for name, value in source.state_dict().items()
            if name.startswith("teacher.")
        }
        torch.save({"model_state_dict": source.state_dict()}, parent)
        torch.save({"model_state_dict": source.deployable_state_dict(), "deployable": True}, deployable)
        torch.save({"model_state_dict": teacher_state}, teacher)
        with pytest.raises(StudentLineageError, match="does not match"):
            load_student_continuation_checkpoint(target, parent, teacher)
        with pytest.raises(StudentLineageError, match="deployable-only"):
            load_student_continuation_checkpoint(target, deployable, teacher)


def test_capability_gate_requires_json_object_and_records_sha():
    with _scratch_dir() as raw:
        scratch = Path(raw)
        gate = scratch / "gate.json"
        gate.write_text(json.dumps({"strict": 24}), encoding="utf-8")
        info = require_json_gate(gate)
        assert info["path"] == str(gate.resolve())
        assert len(info["sha256"]) == 64
        bad = scratch / "bad.json"
        bad.write_text("[]", encoding="utf-8")
        with pytest.raises(StudentLineageError, match="JSON object"):
            require_json_gate(bad)


def test_distillation_records_recon_and_pg_without_double_gru_step():
    policy = _tiny_gru()
    with torch.no_grad():
        for parameter in policy.teacher.parameters():
            parameter.fill_(0.2)
    encoder_ids = {id(parameter) for parameter in policy.depth_encoder.parameters()}
    memory_ids = {id(parameter) for parameter in policy.memory_s.parameters()}
    algorithm = Distillation(
        policy,
        collect_mode="student",
        pg_coef=0.5,
        behavior_coef=1.0,
        recon_coef=1.0,
        teacher_mix=0.5,
        device="cpu",
        num_learning_epochs=1,
        gradient_length=2,
    )
    opt_ids = {id(parameter) for group in algorithm.optimizer.param_groups for parameter in group["params"]}
    assert encoder_ids <= opt_ids
    assert memory_ids <= opt_ids
    algorithm.init_storage("distillation", 4, 2, [20], [12], [2])
    student_obs = _student_obs(4)
    teacher_obs = torch.zeros(4, 12)
    teacher_obs[:, 4:8] = 0.3
    for _ in range(2):
        algorithm.act(student_obs, teacher_obs)
        algorithm.process_env_step(torch.ones(4), torch.zeros(4), {})

    original_forward = policy.memory_s.forward
    calls = {"n": 0}

    def _count_forward(*args, **kwargs):
        calls["n"] += 1
        return original_forward(*args, **kwargs)

    policy.memory_s.forward = _count_forward
    before = [parameter.detach().clone() for parameter in policy.depth_encoder.parameters()]
    loss_dict = algorithm.update()
    after = list(policy.depth_encoder.parameters())
    assert "behavior" in loss_dict
    assert "pg" in loss_dict
    assert "recon" in loss_dict
    assert calls["n"] == 2
    assert any(not torch.equal(old, new.detach()) for old, new in zip(before, after))


def test_distillation_stores_logprob_of_executed_teacher_mix_action():
    policy = _tiny_gru()
    algorithm = Distillation(policy, collect_mode="student", teacher_mix=1.0, device="cpu")
    student_obs = _student_obs(5)
    teacher_obs = torch.ones(5, 12)
    executed = algorithm.act(student_obs, teacher_obs)
    teacher_actions = policy.evaluate(teacher_obs)
    assert torch.allclose(executed, teacher_actions)
    expected = policy.distribution.log_prob(executed).sum(dim=-1)
    assert torch.allclose(algorithm.transition.actions_log_prob, expected)
    assert not algorithm.transition.student_action_mask.any()


def test_teacher_injected_actions_do_not_enter_on_policy_pg():
    policy = _tiny_gru()
    algorithm = Distillation(
        policy,
        collect_mode="student",
        teacher_mix=1.0,
        pg_coef=1.0,
        behavior_coef=1.0,
        device="cpu",
        num_learning_epochs=1,
        gradient_length=2,
    )
    algorithm.init_storage("distillation", 4, 2, [20], [12], [2])
    for reward in (1.0, -1.0):
        algorithm.act(_student_obs(4), torch.ones(4, 12))
        algorithm.process_env_step(torch.full((4,), reward), torch.zeros(4), {})

    loss_dict = algorithm.update()
    assert loss_dict["pg"] == pytest.approx(0.0)


def test_lightlp_depth_noise_matches_paper_constants():
    depth = np.array([1.0, 2.0])
    std = noise.range_dependent_std(depth)
    np.testing.assert_allclose(std, [0.025, 0.045])
    assert noise.sample_global_scale(0.0) == pytest.approx(0.95)
    assert noise.sample_global_scale(1.0) == pytest.approx(1.05)
    noisy = noise.apply_metric_depth_noise(
        depth,
        gaussian=np.zeros_like(depth),
        scale=1.05,
        dropout_mask=np.array([True, False]),
        dropout_value=3.0,
    )
    np.testing.assert_allclose(noisy, [3.0, 2.1])
    assert noise.LIGHTLP_DEPTH_DELAY_STEPS == (2, 3)
    assert noise.LIGHTLP_DEPTH_HOLD_STEPS == 2


def test_feedforward_stage_e_policy_class_is_not_recurrent():
    policy = DepthStudentTeacher(
        num_student_obs=20,
        num_teacher_obs=12,
        num_actions=2,
        depth_shape=(1, 4, 4),
        proprio_obs_dim=4,
        depth_hidden_dim=8,
        student_hidden_dims=[8],
        teacher_hidden_dims=[8],
    )
    assert policy.is_recurrent is False
    assert not hasattr(policy, "memory_s")
    assert any(isinstance(module, torch.nn.Conv2d) for module in policy.depth_encoder.modules())


def test_recurrent_depth_encoder_is_mlp_not_conv():
    policy = _tiny_gru()
    assert not any(isinstance(module, torch.nn.Conv2d) for module in policy.depth_encoder.modules())
    assert any(isinstance(module, torch.nn.Linear) for module in policy.depth_encoder.modules())
    shared_ids = {id(parameter) for parameter in policy.shared_encoder_parameters()}
    encoder_ids = {id(parameter) for parameter in policy.depth_encoder.parameters()}
    memory_ids = {id(parameter) for parameter in policy.memory_s.parameters()}
    assert shared_ids == encoder_ids | memory_ids
    obs = _student_obs(3)
    out = policy.act_inference(obs)
    assert out.shape == (3, 2)


def test_sim2sim_loads_gru_builder():
    source = SIM2SIM_PY.read_text(encoding="utf-8")
    assert "build_depth_student_policy" in source
    assert "strip_deployable_state_dict" in source
    assert "sparse_teacher_latest_scan_range" in source


def _write_teacher_ckpt(path, obs_dim, iter_n=21500):
    torch.save(
        {
            "model_state_dict": {
                "actor.0.weight": torch.zeros(8, obs_dim),
                "actor.0.bias": torch.zeros(8),
                "std": torch.ones(2),
            },
            "iter": iter_n,
        },
        path,
    )


def test_teacher_gate_accepts_sparse_1937_with_matching_manifest():
    with _scratch_dir() as raw:
        scratch = Path(raw)
        ckpt = scratch / "model_21500.pt"
        _write_teacher_ckpt(ckpt, schemas.TEACHER_SPARSE_ACTOR_OBS_DIM)
        manifest = scratch / "eval.json"
        manifest.write_text(
            json.dumps(
                {
                    "evaluator": "t4_terrain_perception_v3",
                    "task": "t4_loco_teacher_sparse",
                    "checkpoint": str(ckpt),
                }
            ),
            encoding="utf-8",
        )
        info = require_sparse_teacher_checkpoint(ckpt, eval_manifests=[manifest])
        assert info["obs_dim"] == schemas.TEACHER_SPARSE_ACTOR_OBS_DIM
        assert info["ungated"] is False
        assert len(info["sha256"]) == 64


def test_teacher_gate_rejects_stage_e_and_student_files():
    with _scratch_dir() as raw:
        scratch = Path(raw)
        stage_e = scratch / "stage_e.pt"
        _write_teacher_ckpt(stage_e, schemas.TEACHER_ACTOR_OBS_DIM)
        with pytest.raises(StudentLineageError, match="Stage E 1155D"):
            require_sparse_teacher_checkpoint(stage_e, allow_ungated=True)
        student = scratch / "student.pt"
        torch.save(
            {
                "model_state_dict": {
                    "student.0.weight": torch.zeros(8, 4),
                    "depth_encoder.0.weight": torch.zeros(2, 1, 3, 3),
                }
            },
            student,
        )
        with pytest.raises(StudentLineageError, match="depth student"):
            require_sparse_teacher_checkpoint(student, allow_ungated=True)


def test_teacher_gate_requires_manifest_unless_ungated():
    with _scratch_dir() as raw:
        scratch = Path(raw)
        ckpt = scratch / "model_21500.pt"
        _write_teacher_ckpt(ckpt, schemas.TEACHER_SPARSE_ACTOR_OBS_DIM)
        with pytest.raises(StudentLineageError, match="teacher_eval_manifest"):
            require_sparse_teacher_checkpoint(ckpt)
        info = require_sparse_teacher_checkpoint(ckpt, allow_ungated=True)
        assert info["ungated"] is True
        other = scratch / "wrong.json"
        other.write_text(
            json.dumps(
                {
                    "task": "t4_loco_teacher_sparse",
                    "checkpoint": "/tmp/model_19000.pt",
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(StudentLineageError, match="does not match teacher path"):
            require_sparse_teacher_checkpoint(ckpt, eval_manifests=[other])


def test_teacher_gate_rejects_same_filename_from_a_different_run():
    with _scratch_dir() as raw:
        scratch = Path(raw)
        ckpt = scratch / "run_a" / "model_21500.pt"
        ckpt.parent.mkdir()
        _write_teacher_ckpt(ckpt, schemas.TEACHER_SPARSE_ACTOR_OBS_DIM)
        wrong = scratch / "run_b" / "model_21500.pt"
        manifest = scratch / "eval.json"
        manifest.write_text(
            json.dumps(
                {
                    "task": "t4_loco_teacher_sparse",
                    "checkpoint": str(wrong),
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(StudentLineageError, match="does not match teacher path"):
            require_sparse_teacher_checkpoint(ckpt, eval_manifests=[manifest])


def test_nan_guard_rejects_ingest_nan_before_storage():
    policy = _tiny_gru()
    algorithm = Distillation(policy, collect_mode="student", teacher_mix=0.0, device="cpu", nan_guard=True)
    obs = _student_obs(3)
    obs[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match=r"non-finite ingest: student_obs"):
        algorithm.act(obs, torch.zeros(3, 12))
    assert algorithm.transition.actions is None


def test_nan_guard_blocks_optimizer_step_on_inf_grad():
    policy = _tiny_gru()
    algorithm = Distillation(policy, device="cpu", nan_guard=True, max_grad_norm=1.0)
    param = next(p for p in algorithm.optimizer.param_groups[0]["params"] if p.requires_grad)
    handle = param.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    stepped = {"n": 0}
    original_step = algorithm.optimizer.step

    def _count_step(*args, **kwargs):
        stepped["n"] += 1
        return original_step(*args, **kwargs)

    algorithm.optimizer.step = _count_step
    try:
        with pytest.raises(RuntimeError, match=r"non-finite gradient:"):
            algorithm._optimizer_step(param.square().sum())
    finally:
        handle.remove()
        algorithm.optimizer.step = original_step
    assert stepped["n"] == 0


def test_vectorized_block_stamp_matches_python_loop():
    height, width, block_h, block_w = 48, 64, 1, 1
    n_env = 7
    env_ids = torch.arange(n_env)
    torch.manual_seed(11)
    row_draw = torch.rand(n_env)
    rows = noise.edge_biased_block_rows(row_draw, height, block_h)
    cols = torch.randint(0, width - block_w + 1, (n_env,))
    vectorized = torch.zeros(n_env, height, width, dtype=torch.bool)
    looped = torch.zeros(n_env, height, width, dtype=torch.bool)
    noise.stamp_rectangular_blocks(vectorized, env_ids, rows, cols, block_h, block_w)
    for env_id in range(n_env):
        row = int(rows[env_id])
        col = int(cols[env_id])
        looped[env_id, row : row + block_h, col : col + block_w] = True
    assert torch.equal(vectorized, looped)


def test_policy_dropout_fills_max_range_not_near_clip():
    depth = torch.zeros(2, 4, 4)
    mask = torch.zeros(2, 4, 4, dtype=torch.bool)
    mask[0, 1, 2] = True
    filled = noise.apply_normalized_block_dropout(depth, mask, fill_value=1.0)
    assert filled[0, 1, 2].item() == pytest.approx(1.0)
    assert filled[1, 0, 0].item() == pytest.approx(0.0)


def test_targeted_depth_boundary_corruption_flips_both_sides_of_an_edge():
    depth = torch.tensor([[[1.0, 1.0, 3.0, 3.0], [1.0, 1.0, 3.0, 3.0]]])
    force = torch.zeros_like(depth)
    corrupted = noise.apply_depth_boundary_corruption(
        depth,
        dropout_draw=force,
        false_hit_draw=force,
        probability=1.0,
        edge_threshold_m=0.25,
        invalid_depth_m=3.0,
    )
    assert torch.all(corrupted[:, :, 1] == 3.0)
    assert torch.all(corrupted[:, :, 2] == 1.0)
    assert torch.all(corrupted[:, :, 0] == 1.0)
    assert torch.all(corrupted[:, :, 3] == 3.0)


def test_depth_refresh_plan_keeps_reset_and_skips_idle_steps():
    assert noise.depth_refresh_plan(1, 3, False, False) == (False, False, False)
    assert noise.depth_refresh_plan(0, 3, False, False) == (True, True, True)
    assert noise.depth_refresh_plan(3, 3, False, False) == (True, True, True)
    # Clean distill: a subset reset must not force a full tiled capture.
    assert noise.depth_refresh_plan(1, 3, True, False) == (False, False, False)
    assert noise.depth_refresh_plan(1, 3, False, True) == (True, False, False)
    assert noise.depth_refresh_plan(1, 3, True, True) == (True, True, False)


def test_student_cfg_exposes_nan_guard_camera_period_and_noise_overrides():
    cfg_src = CFG_PY.read_text(encoding="utf-8")
    env_src = ENV_PY.read_text(encoding="utf-8") + RUNTIME_PY.read_text(encoding="utf-8")
    assert "nan_guard: bool = True" in cfg_src
    assert "max_grad_norm: float = 1.0" in cfg_src
    sparse_env = env_src.split("class T4LocoSparseDepthStudentEnvCfg", 1)[1].split(
        "class T4LocoSparseDepthStudentReprFirstEnvCfg", 1
    )[0]
    assert "student_depth_camera_update_period: float = 0.02" in sparse_env
    assert "student_depth_noise: bool = True" in sparse_env
    assert "student_camera_pos_jitter_m" in sparse_env
    assert "student_camera_ori_jitter_rad" in sparse_env
    assert "sparse_curriculum_demote: bool = False" in sparse_env
    assert "random_level_reset_fraction: float = 0.10" in sparse_env
    assert "random_level_reset_min_level: int | None = None" in sparse_env
    assert "random_level_reset_max_level: int | None = None" in sparse_env
    assert "self.random_level_reset_fraction = 0.10" in sparse_env
    assert "self.random_level_reset_min_level = None" in sparse_env
    assert "self.random_level_reset_max_level = None" in sparse_env
    assert "random_level_reset_fraction: float = 0.50" not in sparse_env
    assert "random_level_reset_min_level: int = 6" not in sparse_env
    assert "self.scene.terrain_generator" not in sparse_env
    assert "self.scene.max_init_terrain_level" not in sparse_env
    assert "self.commands" not in sparse_env
    assert "self.reward =" not in sparse_env
    assert "self.use_lightlp_terminations" not in sparse_env
    assert "self.noise.add_noise = True" in sparse_env
    ft_block = env_src.split("class T4LocoSparseDepthStudentFtEnvCfg", 1)[1].split("\nclass ", 1)[0]
    assert "student_depth_camera_update_period: float = 0.02" in ft_block
    assert "student_depth_d455_sensor_noise: bool = False" in env_src
    assert "student_depth_dropout_after_resize: bool = True" in env_src
    camera_fn = env_src.split("def _t4_student_depth_camera", 1)[1].split("\n@configclass", 1)[0]
    assert "TiledD455CameraCfg" in camera_fn
    assert "RayCasterCameraCfg" not in camera_fn
    assert "mesh_prim_paths" not in camera_fn
    assert 'prim_body_name="Trunk/depth_camera"' in camera_fn
    assert "_apply_camera_extrinsic_jitter" in env_src
    assert "capture_nominal_camera_pose" in env_src
    assert "if self.student_depth_noise:" in env_src
    assert "stamp_rectangular_blocks" in env_src
    assert "assert_finite_grads" in (ROOT / "rsl_rl" / "rsl_rl" / "algorithms" / "distillation.py").read_text(
        encoding="utf-8"
    )


def test_residual_ft_recipe_is_not_plant_or_d3():
    cfg_src = CFG_PY.read_text(encoding="utf-8")
    env_src = ENV_PY.read_text(encoding="utf-8") + RUNTIME_PY.read_text(encoding="utf-8")
    ft_src = FT_PY.read_text(encoding="utf-8")
    residual_alg = cfg_src.split("class T4SparseDepthStudentResidualFtAlgCfg", 1)[1].split(
        "class T4SparseDepthStudentResidualFtAgentCfg", 1
    )[0]
    residual_agent = cfg_src.split("class T4SparseDepthStudentResidualFtAgentCfg", 1)[1]
    residual_env = env_src.split("class T4LocoSparseDepthStudentResidualFtEnvCfg", 1)[1].split(
        "class T4TargetedFtEventCfg", 1
    )[0]
    final_main = cfg_src.split("class T4SparseDepthStudentFinalMainAlgCfg", 1)[1].split(
        "class T4SparseDepthStudentAgentCfg", 1
    )[0]
    main_agent = cfg_src.split("class T4SparseDepthStudentAgentCfg", 1)[1].split(
        "class T4SparseDepthStudentReprFirstAlgCfg", 1
    )[0]
    sparse_env = env_src.split("class T4LocoSparseDepthStudentEnvCfg", 1)[1].split(
        "class T4LocoSparseDepthStudentReprFirstEnvCfg", 1
    )[0]
    assert "max_iterations: int = 1000" in residual_agent
    assert 'run_name: str = "s12_residual_ft"' in residual_agent
    assert "teacher_mix: float = 0.0" in residual_alg
    assert "critic_warmup_iters: int = 200" in residual_alg
    assert "pg_coef: float = 0.5" in residual_alg
    assert "pg_coef_ramp_iters: int = 400" in residual_alg
    assert "behavior_coef: float = 0.5" in residual_alg
    assert "behavior_coef_end: float = 0.5" in residual_alg
    assert "recon_coef: float = 0.5" in residual_alg
    assert 'schedule: str = "fixed"' in residual_alg
    assert "learning_rate: float = 3.0e-5" in residual_alg
    assert "reference_action_coef: float = 0.0" in residual_alg
    assert "pg_coef: float = 0.1" not in residual_alg
    assert "behavior_coef: float = 1.0" not in residual_alg
    assert "reference_action_coef: float = 0.25" not in residual_alg
    assert "pg_coef: float = 1.0" not in residual_alg
    assert "behavior_coef: float = 0.0" not in residual_alg
    assert "recon_coef: float = 0.25" not in residual_alg
    assert "random_level_reset_fraction: float = 0.50" in residual_env
    assert "random_level_reset_min_level: int | None = 6" in residual_env
    assert "self.random_level_reset_fraction = 0.50" in residual_env
    assert "self.random_level_reset_min_level = 6" in residual_env
    assert "student_depth_noise = True" in residual_env
    assert "student_depth_boundary_corruption = False" in residual_env
    assert "action_delay.enable = False" in residual_env
    assert 'sub.proportion = 0.30' in residual_env
    assert '("stepping_stones", "raised_pillars")' in residual_env
    assert "targeted_manufacturing_variation = False" in residual_env
    assert "targeted_layout_seed = False" in residual_env
    assert "randomize_actuator_gains" not in residual_env
    assert 'choices=("joint", "deploy_ft", "targeted_ft", "residual_ft", "plant_ft")' in ft_src
    assert "T4LocoSparseDepthStudentResidualFtEnvCfg" in ft_src
    assert "T4SparseDepthStudentResidualFtAgentCfg" in ft_src
    assert 'phase = "residual_ft"' in ft_src
    assert "teacher_eval_manifest" in ft_src
    assert "load_student_continuation_checkpoint" in ft_src
    assert 'required=True' in ft_src.split("--capability_gate_json", 1)[1].split(")", 1)[0]
    assert 'required=True' in ft_src.split("--student_checkpoint", 1)[1].split(")", 1)[0]
    assert 'required=True' in ft_src.split("--teacher_checkpoint", 1)[1].split(")", 1)[0]
    assert "allow_ungated" not in ft_src
    assert "require_sparse_teacher_checkpoint" in ft_src
    assert "pg_coef: float = 0.2" in final_main
    assert "teacher_mix_decay_iters: int = 2000" in final_main
    assert "behavior_coef: float = 1.0" in final_main
    assert "max_iterations: int = 15000" in main_agent
    assert "random_level_reset_fraction: float = 0.10" in sparse_env
    assert "random_level_reset_min_level: int | None = None" in sparse_env


def test_plant_ft_recipe_is_delay_actuator_not_manufacturing():
    cfg_src = CFG_PY.read_text(encoding="utf-8")
    env_src = ENV_PY.read_text(encoding="utf-8") + RUNTIME_PY.read_text(encoding="utf-8")
    ft_src = FT_PY.read_text(encoding="utf-8")
    plant_alg = cfg_src.split("class T4SparseDepthStudentPlantFtAlgCfg", 1)[1].split(
        "class T4SparseDepthStudentPlantFtAgentCfg", 1
    )[0]
    plant_agent = cfg_src.split("class T4SparseDepthStudentPlantFtAgentCfg", 1)[1]
    plant_env = env_src.split("class T4LocoSparseDepthStudentPlantFtEnvCfg", 1)[1].split("\nclass ", 1)[0]
    residual_env = env_src.split("class T4LocoSparseDepthStudentResidualFtEnvCfg", 1)[1].split(
        "class T4TargetedFtEventCfg", 1
    )[0]
    assert "max_iterations: int = 1000" in plant_agent
    assert 'run_name: str = "s12_plant_ft"' in plant_agent
    assert "learning_rate: float = 1.0e-5" in plant_alg
    assert "pg_coef: float = 0.1" in plant_alg
    assert "pg_coef_ramp_iters: int = 400" in plant_alg
    assert "critic_warmup_iters: int = 200" in plant_alg
    assert "behavior_coef: float = 1.0" in plant_alg
    assert "recon_coef: float = 1.0" in plant_alg
    assert "reference_action_coef: float = 0.25" in plant_alg
    assert "teacher_mix: float = 0.0" in plant_alg
    assert "pg_coef: float = 0.5" not in plant_alg
    assert "random_level_reset_fraction = 0.10" in plant_env
    assert "random_level_reset_min_level = None" in plant_env
    assert "student_depth_noise = True" in plant_env
    assert "student_depth_boundary_corruption = False" in plant_env
    assert "action_delay.enable = True" in plant_env
    assert '"max_delay": 2' in plant_env
    assert "T4TargetedFtEventCfg()" in plant_env
    assert "targeted_manufacturing_variation" not in plant_env
    assert "targeted_layout_seed" not in plant_env
    assert "sub.proportion = 0.30" not in plant_env
    assert "random_level_reset_fraction = 0.50" in residual_env
    assert "action_delay.enable = False" in residual_env
    assert 'choices=("joint", "deploy_ft", "targeted_ft", "residual_ft", "plant_ft")' in ft_src
    assert "T4LocoSparseDepthStudentPlantFtEnvCfg" in ft_src
    assert "T4SparseDepthStudentPlantFtAgentCfg" in ft_src
    assert 'phase = "plant_ft"' in ft_src


def test_targeted_ft_env_isolated_domain_randomization_contract():
    env_src = ENV_PY.read_text(encoding="utf-8") + RUNTIME_PY.read_text(encoding="utf-8")
    events = env_src.split("class T4TargetedFtEventCfg", 1)[1].split(
        "class T4LocoSparseDepthStudentTargetedFtEnvCfg", 1
    )[0]
    targeted = env_src.split("class T4LocoSparseDepthStudentTargetedFtEnvCfg", 1)[1].split("\nclass ", 1)[0]
    assert "self.domain_rand.action_delay.enable = True" in targeted
    assert 'self.domain_rand.action_delay.params = {"min_delay": 0, "max_delay": 2}' in targeted
    assert "randomize_actuator_gains" in events
    assert "randomize_joint_parameters" in events
    assert "randomize_joint_effort_limits" in events
    assert "base_events = self.domain_rand.events" in targeted
    assert "setattr(targeted_events, name, getattr(base_events, name))" in targeted
    assert "student_depth_boundary_corruption: bool = True" in targeted
    assert "self.student_depth_boundary_corruption = bool" in env_src
    assert "targeted_manufacturing_variation" in targeted
    train_src = (ROOT / "legged_lab" / "scripts" / "train_t4_sparse_depth_student_ft.py").read_text(encoding="utf-8")
    assert "T4LocoSparseDepthStudentTargetedFtEnvCfg" in train_src
    assert "env_cfg = T4LocoSparseDepthStudentTargetedFtEnvCfg()" in train_src
    assert '"actuator_domain_randomization"' in train_src
    assert '"terrain_manufacturing_variation"' in train_src
    assert '"pillar_top_tilt_rad"' in train_src


def test_scene_cfg_preserves_positive_depth_camera_update_period():
    scene_src = (ROOT / "legged_lab" / "utils" / "env_utils" / "scene.py").read_text(encoding="utf-8")
    start = scene_src.index("def resolve_depth_camera_update_period")
    end = scene_src.index("\n@configclass", start)
    ns: dict[str, object] = {}
    exec(scene_src[start:end], ns)
    resolve = ns["resolve_depth_camera_update_period"]
    assert resolve(0.0, 0.02) == 0.02
    assert resolve(None, 0.02) == 0.02
    assert resolve(0.02, 0.02) == 0.02
    assert resolve(0.06, 0.02) == 0.06
    rtx_interval = ns["rtx_render_interval_physics"]
    rtx_due = ns["rtx_render_due"]
    pending = ns["pending_post_reset_rtx"]
    assert rtx_interval(0.06, 0.005) == 12
    assert rtx_interval(0.02, 0.005) == 4
    assert rtx_due(12, 0.005, 0.06) is True
    assert rtx_due(24, 0.005, 0.06) is True
    assert rtx_due(11, 0.005, 0.06) is False
    assert rtx_due(1, 0.005, 0.06) is False
    assert pending(100, 96) is True
    assert pending(100, 100) is True
    assert pending(100, 108) is False
    is_warp = ns["sensor_is_warp_raycast"]
    assert is_warp(None) is False

    class _Warp:
        pass

    _Warp.__name__ = "RayCasterCamera"
    assert is_warp(_Warp()) is True

    class _Tiled:
        pass

    _Tiled.__name__ = "TiledCamera"
    assert is_warp(_Tiled()) is False
    depth_block = scene_src.split("if config.depth_camera.enable_depth_camera:", 1)[1]
    assert "resolve_depth_camera_update_period(" in depth_block
    assert "update_period=step_dt" not in depth_block.split("self.depth_camera =", 1)[1].split(")", 1)[0]


def test_t4_headless_step_schedules_rtx_render_before_scene_update():
    t4_src = T4_ENV_PY.read_text(encoding="utf-8")
    env_src = ENV_PY.read_text(encoding="utf-8") + RUNTIME_PY.read_text(encoding="utf-8")
    play_src = PLAY_PY.read_text(encoding="utf-8")
    eval_src = EVAL_PY.read_text(encoding="utf-8")
    probe_src = PROBE_PY.read_text(encoding="utf-8")
    physics_loop = t4_src.split("for _ in range(self.cfg.sim.decimation):", 1)[1].split(
        "self.avg_feet_force_per_step /= self.cfg.sim.decimation", 1
    )[0]
    assert "rtx_render_due(" in physics_loop
    assert "self.sim.render()" in physics_loop
    assert "self.last_rtx_sim_step" in physics_loop
    assert "self.scene.update(dt=self.physics_dt)" in physics_loop
    assert physics_loop.index("self.sim.render()") < physics_loop.index("self.scene.update(dt=self.physics_dt)")
    assert "has_warp" in physics_loop or "has_warp_depth" in t4_src
    assert "_has_warp_depth_camera" in t4_src
    assert "sensor_is_warp_raycast" in t4_src
    assert "schedule_rtx_render" in t4_src
    assert "_rtx_reset_sim_step[env_ids]" in t4_src
    assert "_pending_post_reset_rtx_mask" in t4_src
    assert "writable = ~pending" in env_src
    assert 'if "sensor" in task_name or "depth_student" in task_name:' in play_src
    assert 'if "depth_student" in (args_cli.task or ""):' in eval_src
    assert "args_cli.enable_cameras = True" in eval_src
    assert "skip_rtx" in probe_src
    assert "checksum" in probe_src
    assert "collection_meets_fallback" in probe_src
    assert "collection_s" in probe_src
    assert '"backend": "tiled_rtx"' in probe_src
    assert "has_rtx_sensors" in probe_src
    assert "terrain_generator.num_rows = 1" not in probe_src
    assert "mask_sparse_curriculum_demote" in t4_src
    assert "sparse_curriculum_demote" in env_src


def test_camera_extrinsic_jitter_stays_inside_lightlp_table_ii():
    assert extrinsic.LIGHTLP_CAMERA_POS_JITTER_M == pytest.approx(0.01)
    assert extrinsic.LIGHTLP_CAMERA_ORI_JITTER_RAD == pytest.approx(0.025)
    identity = extrinsic.euler_xyz_to_quat_wxyz(0.0, 0.0, 0.0)
    assert identity[0] == pytest.approx(1.0)
    composed = extrinsic.quat_mul_wxyz(identity, extrinsic.euler_xyz_to_quat_wxyz(0.01, -0.02, 0.015))
    pos, quat = extrinsic.compose_camera_offset(
        (0.085, 0.0, 0.42), identity, (0.01, -0.01, 0.0), (0.025, 0.0, -0.025)
    )
    assert pos[0] == pytest.approx(0.095)
    assert pos[1] == pytest.approx(-0.01)
    assert abs(quat[0] - 1.0) < 0.01
    assert abs(composed[0] - 1.0) < 0.01


def test_repr_first_is_default_train_recipe_and_keeps_final_main():
    cfg_src = CFG_PY.read_text(encoding="utf-8")
    env_src = ENV_PY.read_text(encoding="utf-8") + RUNTIME_PY.read_text(encoding="utf-8")
    train_src = TRAIN_PY.read_text(encoding="utf-8")
    repr_alg = cfg_src.split("class T4SparseDepthStudentReprFirstAlgCfg", 1)[1].split(
        "class T4SparseDepthStudentReprFirstAgentCfg", 1
    )[0]
    repr_agent = cfg_src.split("class T4SparseDepthStudentReprFirstAgentCfg", 1)[1].split(
        "class T4SparseDepthStudentJointAlgCfg", 1
    )[0]
    repr_env = env_src.split("class T4LocoSparseDepthStudentReprFirstEnvCfg", 1)[1].split(
        "class T4LocoSparseDepthStudentFtEnvCfg", 1
    )[0]
    final_main = cfg_src.split("class T4SparseDepthStudentFinalMainAlgCfg", 1)[1].split(
        "class T4SparseDepthStudentAgentCfg", 1
    )[0]
    assert "repr_first: bool = True" in repr_alg
    assert "pg_coef: float = 0.0" in repr_alg
    assert "teacher_mix: float = 1.0" in repr_alg
    assert "teacher_mix_decay_iters: int = 0" in repr_alg
    assert "action_mix_decay_iters: int = 2000" in repr_alg
    assert "repr_cap_iters: int = 4000" in repr_alg
    assert "repr_probe_interval: int = 100" in repr_alg
    assert "repr_probe_patience: int = 3" in repr_alg
    assert "action_level_reset_fraction: float = 0.10" in repr_alg
    assert 'run_name: str = "s12_repr_first"' in repr_agent
    assert "max_iterations: int = 14000" in repr_agent
    assert "self.random_level_reset_fraction = 0.50" in repr_env
    assert "self.random_level_reset_min_level = 6" in repr_env
    assert "pg_coef: float = 0.2" in final_main
    assert 'run_name: str = "s12_final_main"' in cfg_src
    assert "T4SparseDepthStudentReprFirstAgentCfg" in train_src
    assert "T4LocoSparseDepthStudentReprFirstEnvCfg" in train_src
    assert '"phase": "representation"' in train_src
    assert '"repr_first": True' in train_src
    assert "representation_curriculum" in train_src
    assert "action_curriculum" in train_src
    assert "attach_repr_runtime" in train_src
    assert "switch_reason" in train_src
    t4_src = T4_ENV_PY.read_text(encoding="utf-8")
    assert 'fraction = float(getattr(self.cfg, "random_level_reset_fraction", 0.0) or 0.0)' in t4_src
    assert train_src.find("attach(env, on_switch=on_repr_switch)") < train_src.find(
        'getattr(runner.alg, "_repr_state"'
    )

from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
RSL_RL_ROOT = ROOT / "rsl_rl"
if str(RSL_RL_ROOT) not in sys.path:
    sys.path.insert(0, str(RSL_RL_ROOT))

from rsl_rl.algorithms.safe_recurrent_distillation import SafeRecurrentDistillation  # noqa: E402
from rsl_rl.modules.depth_student_teacher import DepthStudentTeacherRecurrent  # noqa: E402
from rsl_rl.runners.on_policy_runner import OnPolicyRunner  # noqa: E402

CFG_PATH = ROOT / "legged_lab" / "envs" / "t4" / "depth_student_cfg.py"
RUNNER_PATH = ROOT / "rsl_rl" / "rsl_rl" / "runners" / "on_policy_runner.py"


def _tiny_policy() -> DepthStudentTeacherRecurrent:
    policy = DepthStudentTeacherRecurrent(
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
        recon_scan_dim=4,
        recon_scan_offset=4,
        recon_hidden_dim=8,
        critic_hidden_dims=[8],
        min_action_std=0.05,
        max_action_std=0.8,
    )
    with torch.no_grad():
        for parameter in policy.teacher.parameters():
            parameter.fill_(0.2)
    return policy


def _student_obs(num_envs: int, step: int = 0) -> torch.Tensor:
    values = torch.linspace(0.0, 1.0, num_envs * 20, dtype=torch.float32).reshape(num_envs, 20)
    return values + 0.01 * step


def _teacher_obs(num_envs: int, step: int = 0) -> torch.Tensor:
    values = torch.zeros(num_envs, 12)
    values[:, :4] = 0.05 * step
    values[:, 4:8] = 0.3 + 0.01 * step
    values[:, 8:] = 0.1
    return values


def _algorithm(
    *,
    max_kl: float = 10.0,
    max_kl_emergency: float = 100.0,
    learning_rate: float = 3.0e-3,
    teacher_mix: float = 0.0,
    teacher_mix_end: float = 0.0,
    teacher_mix_decay_iters: int = 2000,
    behavior_coef: float = 1.0,
    behavior_coef_end: float = 0.0,
    behavior_coef_decay_iters: int = 2000,
) -> SafeRecurrentDistillation:
    return SafeRecurrentDistillation(
        _tiny_policy(),
        device="cpu",
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=learning_rate,
        gamma=0.99,
        lam=0.95,
        clip_param=0.2,
        behavior_coef=behavior_coef,
        behavior_coef_end=behavior_coef_end,
        behavior_coef_decay_iters=behavior_coef_decay_iters,
        pg_coef=0.5,
        recon_coef=1.0,
        value_loss_coef=1.0,
        teacher_mix=teacher_mix,
        teacher_mix_end=teacher_mix_end,
        teacher_mix_decay_iters=teacher_mix_decay_iters,
        desired_kl=0.01,
        max_kl=max_kl,
        max_kl_emergency=max_kl_emergency,
        max_behavior_drift=10.0,
        rollback_lr_factor=0.5,
        max_learning_rate=0.1,
    )


def _collect(algorithm: SafeRecurrentDistillation, *, num_envs: int = 4, num_steps: int = 4) -> None:
    algorithm.init_storage("safe_distillation", num_envs, num_steps, [20], [12], [2])
    for step in range(num_steps):
        algorithm.act(_student_obs(num_envs, step), _teacher_obs(num_envs, step))
        rewards = torch.linspace(-0.5, 1.0, num_envs) + 0.1 * step
        dones = torch.zeros(num_envs)
        if step == 1:
            dones[0] = 1.0
        algorithm.process_env_step(rewards, dones, {"time_outs": torch.zeros(num_envs)})
    algorithm.compute_returns(_teacher_obs(num_envs, num_steps))


def _clone_policy_state(policy: DepthStudentTeacherRecurrent) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in policy.state_dict().items()}


def test_safe_rollout_saves_actor_hidden_and_computes_gae():
    algorithm = _algorithm()
    _collect(algorithm)

    assert algorithm.storage.training_type == "safe_distillation"
    assert algorithm.storage.saved_hidden_states_a is not None
    assert algorithm.storage.saved_hidden_states_c in (None, [])
    assert torch.isfinite(algorithm.storage.values).all()
    assert torch.isfinite(algorithm.storage.returns).all()
    assert torch.isfinite(algorithm.storage.advantages).all()
    assert algorithm.storage.advantages.std() > 0


def test_safe_update_uses_sequence_batches_without_mutating_live_hidden():
    algorithm = _algorithm()
    _collect(algorithm)
    policy = algorithm.policy
    live_hidden = policy.get_hidden_states()[0].detach().clone()

    calls: list[bool] = []
    original_forward = policy.memory_s.forward

    def _record_forward(input_tensor, masks=None, hidden_states=None):
        calls.append(masks is not None)
        return original_forward(input_tensor, masks=masks, hidden_states=hidden_states)

    policy.memory_s.forward = _record_forward
    before = _clone_policy_state(policy)
    report = algorithm.update()

    assert calls and all(calls)
    assert torch.allclose(policy.get_hidden_states()[0], live_hidden)
    assert report["update_accepted"] == pytest.approx(1.0)
    for key in (
        "behavior",
        "pg",
        "recon",
        "value_function",
        "kl_mean",
        "kl_p95",
        "kl_max",
        "ratio_p95",
        "clip_fraction",
        "grad_norm_behavior",
        "grad_norm_pg",
        "grad_norm_recon",
        "grad_cos_behavior_pg",
        "grad_cos_behavior_recon",
        "grad_cos_pg_recon",
        "grad_conflict_recon_control",
        "behavior_coef",
        "rollback_kl_p95",
        "rollback_kl_emergency",
        "accepted_update_count",
    ):
        assert key in report
        assert torch.isfinite(torch.tensor(report[key]))
    assert any(not torch.equal(before[name], value) for name, value in policy.state_dict().items())


def test_configured_multi_batch_update_replays_detached_sequences():
    torch.manual_seed(7)
    algorithm = _algorithm(max_kl=1.0e6, max_kl_emergency=1.0e7, learning_rate=3.0e-4)
    algorithm.num_mini_batches = 4
    algorithm.num_learning_epochs = 2
    _collect(algorithm, num_envs=8, num_steps=6)

    report = algorithm.update()

    assert report["update_accepted"] == pytest.approx(1.0)
    assert report["accepted_update_count"] == pytest.approx(1.0)


def test_safe_update_rolls_back_policy_and_adam_state_when_kl_budget_is_breached():
    algorithm = _algorithm(max_kl=0.0, learning_rate=5.0e-2)
    _collect(algorithm)
    policy_before = _clone_policy_state(algorithm.policy)
    optimizer_before = copy.deepcopy(algorithm.optimizer.state_dict())
    lr_before = algorithm.learning_rate

    report = algorithm.update()

    assert report["update_accepted"] == pytest.approx(0.0)
    assert report["rollback_kl"] == pytest.approx(1.0)
    assert report["kl_max"] > 0.0
    for name, value in algorithm.policy.state_dict().items():
        assert torch.equal(value, policy_before[name])
    assert algorithm.optimizer.state_dict()["state"] == optimizer_before["state"]
    assert algorithm.learning_rate == pytest.approx(lr_before * 0.5)


def test_all_teacher_rollout_has_no_on_policy_ratio_samples():
    algorithm = _algorithm(teacher_mix=1.0, teacher_mix_end=1.0)
    _collect(algorithm)

    report = algorithm.update()

    assert report["pg"] == pytest.approx(0.0)
    assert report["ratio_p95"] == pytest.approx(1.0)
    assert report["ratio_max"] == pytest.approx(1.0)
    assert report["clip_fraction"] == pytest.approx(0.0)


def test_kl_gate_uses_p95_with_a_separate_emergency_maximum():
    algorithm = _algorithm(max_kl=0.03, max_kl_emergency=0.3)
    baseline = {
        "kl_mean": 0.01,
        "kl_p95": 0.02,
        "kl_max": 0.10,
        "post_behavior": 0.1,
    }

    assert algorithm._rollback_reasons(baseline, pre_behavior=0.1) == (False, False, False, False)

    p95_breach = {**baseline, "kl_p95": 0.031}
    assert algorithm._rollback_reasons(p95_breach, pre_behavior=0.1) == (True, True, False, False)

    emergency_breach = {**baseline, "kl_max": 0.301}
    assert algorithm._rollback_reasons(emergency_breach, pre_behavior=0.1) == (True, False, True, False)


def test_rejected_update_does_not_advance_teacher_mix_schedule():
    algorithm = _algorithm(
        max_kl=0.0,
        max_kl_emergency=0.0,
        learning_rate=5.0e-2,
        teacher_mix=0.5,
        teacher_mix_end=0.0,
        teacher_mix_decay_iters=2,
        behavior_coef=1.0,
        behavior_coef_end=0.0,
        behavior_coef_decay_iters=2,
    )
    _collect(algorithm)
    mix_before = algorithm.current_teacher_mix()

    report = algorithm.update()

    assert report["update_accepted"] == pytest.approx(0.0)
    assert algorithm.num_updates == 1
    assert algorithm.num_accepted_updates == 0
    assert algorithm.current_teacher_mix() == pytest.approx(mix_before)
    assert algorithm.current_behavior_coef() == pytest.approx(1.0)


def test_safe_algorithm_checkpoint_state_restores_schedule_and_learning_rate():
    algorithm = _algorithm(
        teacher_mix=0.5,
        teacher_mix_end=0.0,
        teacher_mix_decay_iters=10,
        behavior_coef_decay_iters=10,
    )
    algorithm.num_updates = 7
    algorithm.num_accepted_updates = 4
    algorithm.rollback_count = 3
    algorithm._set_learning_rate(7.5e-4)

    restored = _algorithm(
        teacher_mix=0.5,
        teacher_mix_end=0.0,
        teacher_mix_decay_iters=10,
        behavior_coef_decay_iters=10,
    )
    restored.load_checkpoint_state_dict(algorithm.checkpoint_state_dict())

    assert restored.num_updates == 7
    assert restored.num_accepted_updates == 4
    assert restored.rollback_count == 3
    assert restored.learning_rate == pytest.approx(7.5e-4)
    assert restored.current_teacher_mix() == pytest.approx(0.3)
    assert restored.current_behavior_coef() == pytest.approx(0.6)


def test_runner_checkpoint_round_trip_preserves_safe_algorithm_state():
    algorithm = _algorithm(
        teacher_mix=0.5,
        teacher_mix_end=0.0,
        teacher_mix_decay_iters=10,
        behavior_coef_decay_iters=10,
    )
    algorithm.num_updates = 7
    algorithm.num_accepted_updates = 4
    algorithm.rollback_count = 3
    algorithm._set_learning_rate(7.5e-4)

    with tempfile.TemporaryDirectory(prefix="safe-recurrent-checkpoint-") as raw:
        runner = object.__new__(OnPolicyRunner)
        runner.alg = algorithm
        runner.current_learning_iteration = 123
        runner.empirical_normalization = False
        runner.logger_type = "tensorboard"
        runner.disable_logs = False
        checkpoint = Path(raw) / "model_123.pt"
        runner.save(str(checkpoint))

        restored = _algorithm(
            teacher_mix=0.5,
            teacher_mix_end=0.0,
            teacher_mix_decay_iters=10,
            behavior_coef_decay_iters=10,
        )
        restored_runner = object.__new__(OnPolicyRunner)
        restored_runner.alg = restored
        restored_runner.current_learning_iteration = 0
        restored_runner.empirical_normalization = False
        restored_runner.load(str(checkpoint))

        assert restored_runner.current_learning_iteration == 123
        assert restored.num_updates == 7
        assert restored.num_accepted_updates == 4
        assert restored.rollback_count == 3
        assert restored.learning_rate == pytest.approx(7.5e-4)
        assert restored.current_teacher_mix() == pytest.approx(0.3)
        assert restored.current_behavior_coef() == pytest.approx(0.6)


def test_runner_model_only_load_resets_safe_optimizer_schedule():
    algorithm = _algorithm(teacher_mix=0.5, teacher_mix_end=0.0, teacher_mix_decay_iters=10)
    algorithm.num_updates = 7
    algorithm.num_accepted_updates = 4
    algorithm.rollback_count = 3
    algorithm._set_learning_rate(7.5e-4)

    with tempfile.TemporaryDirectory(prefix="safe-recurrent-model-only-") as raw:
        runner = object.__new__(OnPolicyRunner)
        runner.alg = algorithm
        runner.current_learning_iteration = 123
        runner.empirical_normalization = False
        runner.logger_type = "tensorboard"
        runner.disable_logs = False
        checkpoint = Path(raw) / "model_123.pt"
        runner.save(str(checkpoint))

        restored = _algorithm(teacher_mix=0.5, teacher_mix_end=0.0, teacher_mix_decay_iters=10)
        initial_lr = restored.learning_rate
        restored_runner = object.__new__(OnPolicyRunner)
        restored_runner.alg = restored
        restored_runner.current_learning_iteration = 0
        restored_runner.empirical_normalization = False
        restored_runner.load(str(checkpoint), load_optimizer=False)

        assert restored.num_updates == 0
        assert restored.num_accepted_updates == 0
        assert restored.rollback_count == 0
        assert restored.learning_rate == pytest.approx(initial_lr)


def test_reconstruction_gradient_is_projected_when_it_conflicts_with_control():
    control = [torch.tensor([1.0, 0.0])]
    reconstruction = [torch.tensor([-2.0, 1.0])]

    projected, conflict = SafeRecurrentDistillation._project_auxiliary_gradient(reconstruction, control)

    assert conflict.item() == pytest.approx(1.0)
    assert SafeRecurrentDistillation._gradient_dot(projected, control).item() >= -1.0e-7
    assert torch.equal(control[0], torch.tensor([1.0, 0.0]))


def test_training_critic_is_removed_from_deployable_state():
    policy = _tiny_policy()
    assert any(name.startswith("critic.") for name in policy.state_dict())
    deployed = policy.deployable_state_dict()
    assert not any(name.startswith("critic.") for name in deployed)


def test_sparse_student_cfg_selects_safe_recurrent_improvement():
    source = CFG_PATH.read_text(encoding="utf-8")
    assert 'class_name: str = "SafeRecurrentDistillation"' in source
    assert "critic_hidden_dims" in source
    assert "num_mini_batches: int = 4" in source
    assert "desired_kl: float = 0.01" in source
    assert "max_kl: float = 0.03" in source
    assert "max_kl_emergency: float = 0.3" in source
    assert "max_behavior_drift: float = 0.02" in source
    assert "behavior_coef_end: float = 0.0" in source
    assert "behavior_coef_decay_iters: int = 2000" in source
    assert "pg_coef: float = 0.5" in source


def test_runner_maps_safe_algorithm_to_teacher_observation_contract():
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert 'self.alg_cfg["class_name"] == "SafeRecurrentDistillation"' in source
    assert 'self.training_type = "safe_distillation"' in source
    assert 'self.training_type in {"distillation", "safe_distillation"}' in source
    assert 'self.training_type in {"rl", "safe_distillation"}' in source
    assert 'self.training_type in {"distillation", "safe_distillation"}' in source
    assert 'saved_dict["algorithm_state_dict"]' in source
    assert 'loaded_dict["algorithm_state_dict"]' in source

from __future__ import annotations

import inspect
import sys
import tempfile
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
RSL_RL_ROOT = ROOT / "rsl_rl"
if str(RSL_RL_ROOT) not in sys.path:
    sys.path.insert(0, str(RSL_RL_ROOT))

from rsl_rl.algorithms.distillation import mix_teacher_student_actions  # noqa: E402
from rsl_rl.algorithms.ppo import (  # noqa: E402
    PPO,
    adapt_ppo_learning_rate,
    gaussian_kl,
    ppo_clipped_surrogate_loss,
    ppo_clipped_value_loss,
)
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
    learning_rate: float = 3.0e-3,
    teacher_mix: float = 0.0,
    teacher_mix_end: float = 0.0,
    teacher_mix_decay_iters: int = 2000,
    behavior_coef: float = 1.0,
    behavior_coef_end: float = 0.0,
    behavior_coef_decay_iters: int = 2000,
    critic_warmup_iters: int = 0,
    desired_kl: float | None = 0.01,
    pg_coef: float = 0.5,
    pg_coef_ramp_iters: int = 0,
    pg_delay_iters: int = 0,
    schedule: str = "fixed",
    reference_action_coef: float = 0.0,
    repr_first: bool = False,
    repr_cap_iters: int = 4000,
    repr_probe_interval: int = 100,
    action_mix_decay_iters: int = 2000,
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
        pg_coef=pg_coef,
        pg_coef_ramp_iters=pg_coef_ramp_iters,
        pg_delay_iters=pg_delay_iters,
        critic_warmup_iters=critic_warmup_iters,
        recon_coef=1.0,
        value_loss_coef=1.0,
        teacher_mix=teacher_mix,
        teacher_mix_end=teacher_mix_end,
        teacher_mix_decay_iters=teacher_mix_decay_iters,
        desired_kl=desired_kl,
        max_learning_rate=0.1,
        schedule=schedule,
        reference_action_coef=reference_action_coef,
        repr_first=repr_first,
        repr_cap_iters=repr_cap_iters,
        repr_probe_interval=repr_probe_interval,
        action_mix_decay_iters=action_mix_decay_iters,
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


def test_cpu_ingest_nan_still_rejected_and_cuda_cadence_is_explicit():
    algorithm = _algorithm()
    obs = _student_obs(4)
    obs[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite ingest: student_obs"):
        algorithm.act(obs, _teacher_obs(4))
    algorithm.init_storage("safe_distillation", 4, 4, [20], [12], [2])
    assert algorithm._should_check_ingest(torch.zeros(2)) is True
    algorithm.storage.step = 3
    assert algorithm._should_check_ingest(torch.zeros(2)) is True
    source = (RSL_RL_ROOT / "rsl_rl" / "algorithms" / "safe_recurrent_distillation.py").read_text(encoding="utf-8")
    assert 'tensor.device.type != "cuda"' in source
    assert "return step == 0" in source


def test_safe_recurrent_distillation_is_teacher_ppo_subclass():
    assert issubclass(SafeRecurrentDistillation, PPO)
    source = (RSL_RL_ROOT / "rsl_rl" / "algorithms" / "safe_recurrent_distillation.py").read_text(encoding="utf-8")
    assert "class SafeRecurrentDistillation(PPO)" in source
    assert "super().process_env_step" in source
    assert "ppo_clipped_surrogate_loss" in source
    assert "ppo_clipped_value_loss" in source
    assert "gaussian_kl" in source
    assert "mix_teacher_student_actions" in source
    assert "def _coordinated_shared_gradients" not in source
    assert "autograd.grad" not in source
    assert source.count("retain_graph=True") == 1
    assert "control_loss.backward(retain_graph=True)" in source
    assert "def _candidate_metrics" not in source
    storage_source = (RSL_RL_ROOT / "rsl_rl" / "storage" / "rollout_storage.py").read_text(encoding="utf-8")
    assert "def _padded_recurrent_minibatches" in storage_source
    assert "for batch in self._padded_recurrent_minibatches" in storage_source


def test_reference_actor_records_recurrent_actions_and_resets_with_rollout():
    algorithm = _algorithm(reference_action_coef=0.25)
    algorithm.set_reference_policy_from_current()
    algorithm.init_storage("safe_distillation", 4, 2, [20], [12], [2])

    algorithm.act(_student_obs(4), _teacher_obs(4))
    assert algorithm.transition.reference_actions is not None
    assert torch.allclose(algorithm.transition.reference_actions, algorithm.policy.action_mean)
    rewards = torch.zeros(4)
    dones = torch.tensor([1.0, 0.0, 0.0, 0.0])
    algorithm.process_env_step(rewards, dones, {"time_outs": torch.zeros(4)})

    assert torch.allclose(algorithm.storage.reference_actions[0], algorithm.storage.mu[0])
    assert algorithm.reference_policy.memory_s.hidden_states[:, 0].abs().sum() == pytest.approx(0.0)
    assert all(not parameter.requires_grad for parameter in algorithm.reference_policy.parameters())


def test_reference_action_anchor_reports_cumulative_parent_drift():
    algorithm = _algorithm(reference_action_coef=0.5, pg_coef=0.0)
    algorithm.set_reference_policy_from_current()
    with torch.no_grad():
        algorithm.policy.student[-1].bias.add_(0.25)
    _collect(algorithm)

    report = algorithm.update()

    assert report["reference_action"] > 0.0
    assert report["reference_action_coef"] == pytest.approx(0.5)


def test_shared_ppo_primitives_match_teacher_formulas():
    old_mu = torch.zeros(4, 2)
    old_sigma = torch.ones(4, 2)
    new_mu = torch.ones(4, 2)
    new_sigma = torch.full((4, 2), 2.0)
    expected_kl = torch.sum(
        torch.log(new_sigma / old_sigma + 1.0e-5)
        + (old_sigma.square() + (old_mu - new_mu).square()) / (2.0 * new_sigma.square())
        - 0.5,
        dim=-1,
    )
    assert torch.allclose(gaussian_kl(old_mu, old_sigma, new_mu, new_sigma), expected_kl)

    advantages = torch.tensor([1.0, -0.5])
    new_log = torch.tensor([0.2, -0.1])
    old_log = torch.tensor([0.0, 0.0])
    loss, ratio = ppo_clipped_surrogate_loss(advantages, new_log, old_log, 0.2)
    unclipped = torch.exp(new_log - old_log)
    assert torch.allclose(ratio, unclipped)
    squeezed = advantages
    surrogate = torch.max(-squeezed * unclipped, -squeezed * unclipped.clamp(0.8, 1.2)).mean()
    assert torch.allclose(loss, surrogate)

    values = torch.tensor([[1.5], [0.5]])
    targets = torch.tensor([[1.0], [0.0]])
    returns = torch.tensor([[2.0], [1.0]])
    clipped = ppo_clipped_value_loss(values, targets, returns, 0.2, True)
    value_clipped = targets + (values - targets).clamp(-0.2, 0.2)
    expected_value = torch.max((values - returns).pow(2), (value_clipped - returns).pow(2)).mean()
    assert torch.allclose(clipped, expected_value)
    assert adapt_ppo_learning_rate(3.0e-3, 0.03, 0.01) == pytest.approx(3.0e-3 / 1.5)
    assert adapt_ppo_learning_rate(3.0e-3, 0.001, 0.01) == pytest.approx(3.0e-3 * 1.5)


def test_dagger_mix_helper_is_shared_with_distillation():
    student = torch.zeros(4, 2)
    teacher = torch.ones(4, 2)
    executed, mask = mix_teacher_student_actions(student, teacher, 0.0)
    assert torch.equal(executed, student)
    assert torch.all(mask)
    executed, mask = mix_teacher_student_actions(student, teacher, 1.0)
    assert torch.equal(executed, teacher)
    assert torch.all(~mask)
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
    for key in (
        "behavior",
        "pg",
        "recon",
        "value_function",
        "kl_mean",
        "kl_p95",
        "kl_p99",
        "kl_max",
        "ratio_p95",
        "clip_fraction",
        "accepted_minibatches",
        "planned_minibatches",
        "behavior_coef",
        "pg_coef",
        "critic_warmup",
        "accepted_update_count",
        "action_std_min",
        "action_std_mean",
        "action_std_max",
        "recon_control_cosine",
        "recon_grad_norm",
        "control_grad_norm",
    ):
        assert key in report
        assert torch.isfinite(torch.tensor(report[key]))
    assert "grad_norm_behavior" not in report
    assert any(not torch.equal(before[name], value) for name, value in policy.state_dict().items())


def test_configured_multi_batch_update_replays_detached_sequences():
    torch.manual_seed(7)
    algorithm = _algorithm(learning_rate=3.0e-4)
    algorithm.num_mini_batches = 4
    algorithm.num_learning_epochs = 2
    _collect(algorithm, num_envs=8, num_steps=6)

    report = algorithm.update()

    assert report["accepted_update_count"] == pytest.approx(1.0)
    assert report["accepted_minibatches"] == pytest.approx(8.0)


def test_critic_warmup_holds_pg_until_the_value_head_has_data():
    algorithm = _algorithm(critic_warmup_iters=2)
    assert algorithm.current_pg_coef() == pytest.approx(0.0)

    _collect(algorithm)
    first = algorithm.update()
    assert first["pg_coef"] == pytest.approx(0.0)
    assert first["critic_warmup"] == pytest.approx(1.0)

    _collect(algorithm)
    second = algorithm.update()
    assert second["pg_coef"] == pytest.approx(0.0)

    _collect(algorithm)
    third = algorithm.update()
    assert third["pg_coef"] == pytest.approx(0.5)
    assert third["critic_warmup"] == pytest.approx(0.0)


def test_pg_coef_ramps_after_critic_warmup():
    algorithm = _algorithm(critic_warmup_iters=1, pg_coef=0.2, pg_coef_ramp_iters=2)
    _collect(algorithm)
    first = algorithm.update()
    assert first["pg_coef"] == pytest.approx(0.0)
    _collect(algorithm)
    second = algorithm.update()
    assert second["pg_coef"] == pytest.approx(0.1)
    _collect(algorithm)
    third = algorithm.update()
    assert third["pg_coef"] == pytest.approx(0.2)


def test_critic_warmup_freezes_actor_and_still_updates_critic():
    algorithm = _algorithm(critic_warmup_iters=1, pg_coef=0.5, behavior_coef=1.0, learning_rate=3.0e-3)
    critic_ids = {id(parameter) for parameter in algorithm.policy.critic.parameters()}

    def _actor_snapshot():
        return {
            name: parameter.detach().clone()
            for name, parameter in algorithm.policy.named_parameters()
            if id(parameter) not in critic_ids
        }

    _collect(algorithm)
    actor_before = _actor_snapshot()
    critic_before = [parameter.detach().clone() for parameter in algorithm.policy.critic.parameters()]
    first = algorithm.update()
    assert first["pg_coef"] == pytest.approx(0.0)
    assert first["actor_frozen"] == pytest.approx(1.0)
    actor_after_warmup = _actor_snapshot()
    for name, before in actor_before.items():
        assert torch.equal(before, actor_after_warmup[name]), name
    assert any(
        not torch.equal(before, after)
        for before, after in zip(critic_before, list(algorithm.policy.critic.parameters()))
    )

    _collect(algorithm)
    second = algorithm.update()
    assert second["pg_coef"] == pytest.approx(0.5)
    assert second["actor_frozen"] == pytest.approx(0.0)
    actor_after_ppo = _actor_snapshot()
    assert any(not torch.equal(actor_after_warmup[name], actor_after_ppo[name]) for name in actor_after_warmup)


def test_mean_kl_adapts_learning_rate_like_teacher_ppo():
    algorithm = _algorithm(learning_rate=3.0e-3, desired_kl=0.01, schedule="adaptive")
    old_mu = torch.zeros(8, 2)
    old_sigma = torch.ones(8, 2)
    shifted_mu = torch.ones(8, 2)

    algorithm._adapt_learning_rate(old_mu, old_sigma, shifted_mu, old_sigma)
    assert algorithm.learning_rate == pytest.approx(3.0e-3 / 1.5)

    algorithm._set_learning_rate(3.0e-3)
    algorithm._adapt_learning_rate(old_mu, old_sigma, old_mu + 1.0e-2, old_sigma)
    assert algorithm.learning_rate == pytest.approx(3.0e-3 * 1.5)


def test_fixed_schedule_does_not_adapt_learning_rate_from_kl():
    algorithm = _algorithm(learning_rate=1.0e-4, desired_kl=0.01, schedule="fixed")
    algorithm._adapt_learning_rate_from_mean_kl(10.0)
    assert algorithm.learning_rate == pytest.approx(1.0e-4)
    algorithm._adapt_learning_rate(
        torch.zeros(8, 2),
        torch.ones(8, 2),
        torch.ones(8, 2),
        torch.ones(8, 2),
    )
    assert algorithm.learning_rate == pytest.approx(1.0e-4)


def test_delayed_pg_allows_mix_start_when_decay_fits_delay():
    algorithm = _algorithm(
        teacher_mix=1.0,
        teacher_mix_end=0.0,
        teacher_mix_decay_iters=4,
        pg_coef=0.2,
        pg_delay_iters=4,
        critic_warmup_iters=2,
        behavior_coef=1.0,
        behavior_coef_end=1.0,
        behavior_coef_decay_iters=0,
    )
    algorithm.num_updates = 0
    assert algorithm.current_teacher_mix() == pytest.approx(1.0)
    assert algorithm.current_pg_coef() == pytest.approx(0.0)
    assert algorithm._in_critic_warmup() is False
    algorithm.num_updates = 3
    assert algorithm.current_teacher_mix() > 0.0
    assert algorithm.current_pg_coef() == pytest.approx(0.0)
    algorithm.num_updates = 4
    assert algorithm.current_teacher_mix() == pytest.approx(0.0)
    assert algorithm.current_pg_coef() == pytest.approx(0.0)
    algorithm.num_updates = 5
    assert algorithm._in_critic_warmup() is True
    assert algorithm.current_pg_coef() == pytest.approx(0.0)
    algorithm.num_updates = 6
    assert algorithm._in_critic_warmup() is True
    algorithm.num_updates = 7
    assert algorithm._in_critic_warmup() is False
    assert algorithm.current_pg_coef() == pytest.approx(0.2)


def test_nonzero_teacher_mix_raises_when_pg_coef_is_positive():
    with pytest.raises(ValueError, match="teacher_mix=0"):
        _algorithm(teacher_mix=0.5, pg_coef=0.5)
    with pytest.raises(ValueError, match="teacher_mix=0"):
        _algorithm(teacher_mix=0.0, teacher_mix_end=0.3, pg_coef=0.5)
    algorithm = _algorithm(teacher_mix=0.5, teacher_mix_end=0.0, pg_coef=0.0)
    assert algorithm.current_teacher_mix() == pytest.approx(0.5)


def test_all_teacher_rollout_has_no_on_policy_ratio_samples():
    algorithm = _algorithm(teacher_mix=1.0, teacher_mix_end=1.0, pg_coef=0.0)
    _collect(algorithm)

    report = algorithm.update()

    assert report["pg"] == pytest.approx(0.0)
    assert report["ratio_p95"] == pytest.approx(1.0)
    assert report["ratio_max"] == pytest.approx(1.0)
    assert report["clip_fraction"] == pytest.approx(0.0)


def test_teacher_mix_advances_on_every_applied_update():
    algorithm = _algorithm(
        teacher_mix=0.5,
        teacher_mix_end=0.0,
        teacher_mix_decay_iters=2,
        behavior_coef=1.0,
        behavior_coef_end=0.0,
        behavior_coef_decay_iters=2,
        pg_coef=0.0,
    )
    _collect(algorithm)
    mix_before = algorithm.current_teacher_mix()

    report = algorithm.update()

    assert algorithm.num_updates == 1
    assert algorithm.num_accepted_updates == 1
    assert algorithm.current_teacher_mix() < mix_before
    assert report["teacher_mix"] == pytest.approx(algorithm.current_teacher_mix())


def test_safe_algorithm_checkpoint_state_restores_schedule_and_learning_rate():
    algorithm = _algorithm(
        teacher_mix=0.5,
        teacher_mix_end=0.0,
        teacher_mix_decay_iters=10,
        behavior_coef_decay_iters=10,
        pg_coef=0.0,
    )
    algorithm.num_updates = 4
    algorithm.num_accepted_updates = 4
    algorithm.rollback_count = 0
    algorithm._set_learning_rate(7.5e-4)

    restored = _algorithm(
        teacher_mix=0.5,
        teacher_mix_end=0.0,
        teacher_mix_decay_iters=10,
        behavior_coef_decay_iters=10,
        pg_coef=0.0,
    )
    restored.load_checkpoint_state_dict(algorithm.checkpoint_state_dict())

    assert restored.num_updates == 4
    assert restored.num_accepted_updates == 4
    assert restored.rollback_count == 0
    assert restored.learning_rate == pytest.approx(7.5e-4)
    assert restored.current_teacher_mix() == pytest.approx(0.3)
    assert restored.current_behavior_coef() == pytest.approx(0.6)


def test_safe_algorithm_checkpoint_restores_frozen_reference_actor():
    algorithm = _algorithm(reference_action_coef=0.25)
    algorithm.set_reference_policy_from_current()
    with torch.no_grad():
        algorithm.reference_policy.student[-1].bias.add_(0.125)

    restored = _algorithm(reference_action_coef=0.25)
    restored.load_checkpoint_state_dict(algorithm.checkpoint_state_dict())

    assert restored.reference_policy is not None
    assert all(not parameter.requires_grad for parameter in restored.reference_policy.parameters())
    expected = algorithm.reference_policy.deployable_state_dict()
    actual = restored.reference_policy.deployable_state_dict()
    assert expected.keys() == actual.keys()
    for name in expected:
        assert torch.equal(expected[name], actual[name]), name


def test_runner_checkpoint_round_trip_preserves_safe_algorithm_state():
    algorithm = _algorithm(
        teacher_mix=0.5,
        teacher_mix_end=0.0,
        teacher_mix_decay_iters=10,
        behavior_coef_decay_iters=10,
        pg_coef=0.0,
    )
    algorithm.num_updates = 4
    algorithm.num_accepted_updates = 4
    algorithm.rollback_count = 0
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
            pg_coef=0.0,
        )
        restored_runner = object.__new__(OnPolicyRunner)
        restored_runner.alg = restored
        restored_runner.current_learning_iteration = 0
        restored_runner.empirical_normalization = False
        restored_runner.load(str(checkpoint))

        assert restored_runner.current_learning_iteration == 123
        assert restored.num_updates == 4
        assert restored.num_accepted_updates == 4
        assert restored.rollback_count == 0
        assert restored.learning_rate == pytest.approx(7.5e-4)
        assert restored.current_teacher_mix() == pytest.approx(0.3)
        assert restored.current_behavior_coef() == pytest.approx(0.6)


def test_runner_model_only_load_resets_safe_optimizer_schedule():
    algorithm = _algorithm(teacher_mix=0.5, teacher_mix_end=0.0, teacher_mix_decay_iters=10, pg_coef=0.0)
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

        restored = _algorithm(teacher_mix=0.5, teacher_mix_end=0.0, teacher_mix_decay_iters=10, pg_coef=0.0)
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


def test_reconstruction_gradient_norm_is_capped_relative_to_control():
    control = [torch.tensor([1.0, 0.0])]
    reconstruction = [torch.tensor([100.0, 100.0])]

    limited, scale = SafeRecurrentDistillation._limit_auxiliary_gradient_norm(
        reconstruction,
        control,
        coefficient=1.0,
        max_ratio=1.0,
    )

    control_norm = torch.sqrt(SafeRecurrentDistillation._gradient_dot(control, control))
    limited_norm = torch.sqrt(SafeRecurrentDistillation._gradient_dot(limited, limited))
    assert scale.item() < 0.01
    assert limited_norm.item() <= control_norm.item() + 1.0e-6


def test_reconstruction_cannot_move_shared_trunk_without_control_gradient():
    control = [torch.zeros(2)]
    reconstruction = [torch.tensor([1.0, 2.0])]

    limited, scale = SafeRecurrentDistillation._limit_auxiliary_gradient_norm(
        reconstruction,
        control,
        coefficient=1.0,
        max_ratio=1.0,
    )

    assert scale.item() == pytest.approx(0.0)
    assert torch.equal(limited[0], torch.zeros(2))


def test_optimizer_step_projects_trainable_action_std():
    algorithm = _algorithm(learning_rate=0.1)
    algorithm.policy.max_action_std = 0.2
    with torch.no_grad():
        algorithm.policy.std.fill_(0.19)

    algorithm._optimizer_step(-100.0 * algorithm.policy.std.sum())

    assert torch.allclose(algorithm.policy.std, torch.full_like(algorithm.policy.std, 0.2))


def test_training_critic_is_removed_from_deployable_state():
    policy = _tiny_policy()
    assert any(name.startswith("critic.") for name in policy.state_dict())
    deployed = policy.deployable_state_dict()
    assert not any(name.startswith("critic.") for name in deployed)


def test_update_projects_conflicting_recon_gradient_on_shared_encoder():
    seen = {"calls": 0}
    original = SafeRecurrentDistillation._project_auxiliary_gradient.__func__

    def _spy(cls, auxiliary, control):
        seen["calls"] += 1
        return original(cls, auxiliary, control)

    SafeRecurrentDistillation._project_auxiliary_gradient = classmethod(_spy)
    try:
        algorithm = _algorithm()
        _collect(algorithm)
        report = algorithm.update()
        assert seen["calls"] >= 1
        assert "recon_control_cosine" in report
        assert "recon_grad_norm" in report
        assert "control_grad_norm" in report
        assert torch.isfinite(torch.tensor(report["recon_control_cosine"]))
        assert torch.isfinite(torch.tensor(report["recon_grad_norm"]))
        assert torch.isfinite(torch.tensor(report["control_grad_norm"]))
        source = inspect.getsource(SafeRecurrentDistillation.update)
        assert "_optimizer_step_with_recon_projection" in source
        helper = inspect.getsource(SafeRecurrentDistillation._optimizer_step_with_recon_projection)
        assert "_project_auxiliary_gradient" in helper
        assert "_limit_auxiliary_gradient_norm" in helper
    finally:
        SafeRecurrentDistillation._project_auxiliary_gradient = classmethod(original)


def test_distributed_recon_projection_reduces_before_projecting():
    order: list[str] = []
    algorithm = _algorithm()
    _collect(algorithm)
    batch = next(algorithm.storage.safe_recurrent_mini_batch_generator(1, 1))
    control_loss, recon_term, *_ = algorithm._batch_losses(batch)
    original_project = SafeRecurrentDistillation._project_auxiliary_gradient.__func__

    def _spy_project(cls, auxiliary, control):
        order.append("project")
        return original_project(cls, auxiliary, control)

    def _spy_reduce(tensors):
        order.append("reduce")

    algorithm.is_multi_gpu = True
    algorithm.gpu_world_size = 2
    algorithm._all_reduce_tensors = _spy_reduce
    SafeRecurrentDistillation._project_auxiliary_gradient = classmethod(_spy_project)
    try:
        algorithm._optimizer_step_with_recon_projection(control_loss, recon_term, check_finite=False)
        assert "reduce" in order
        assert "project" in order
        assert order.index("reduce") < order.index("project")
        source = inspect.getsource(SafeRecurrentDistillation._optimizer_step_with_recon_projection)
        assert source.find("_all_reduce_tensors") < source.find("_project_auxiliary_gradient")
        assert "_reduce_gradients()" not in source
    finally:
        SafeRecurrentDistillation._project_auxiliary_gradient = classmethod(original_project)


def test_sparse_student_cfg_selects_safe_recurrent_improvement():
    source = CFG_PATH.read_text(encoding="utf-8")
    distill = source.split("class T4SparseDepthStudentDaggerAlgCfg", 1)[1].split(
        "class T4SparseDepthStudentFinalMainAlgCfg", 1
    )[0]
    final_main = source.split("class T4SparseDepthStudentFinalMainAlgCfg", 1)[1].split(
        "class T4SparseDepthStudentAgentCfg", 1
    )[0]
    joint_alg = source.split("class T4SparseDepthStudentJointAlgCfg", 1)[1].split(
        "class T4SparseDepthStudentDeployFtAlgCfg", 1
    )[0]
    ft_alg = source.split("class T4SparseDepthStudentDeployFtAlgCfg", 1)[1].split(
        "class T4SparseDepthStudentTargetedFtAlgCfg", 1
    )[0]
    targeted_alg = source.split("class T4SparseDepthStudentTargetedFtAlgCfg", 1)[1].split(
        "class T4SparseDepthStudentFtAgentCfg", 1
    )[0]
    residual_alg = source.split("class T4SparseDepthStudentResidualFtAlgCfg", 1)[1].split(
        "class T4SparseDepthStudentResidualFtAgentCfg", 1
    )[0]
    residual_agent = source.split("class T4SparseDepthStudentResidualFtAgentCfg", 1)[1]
    agent = source.split("class T4SparseDepthStudentAgentCfg", 1)[1].split(
        "class T4SparseDepthStudentReprFirstAlgCfg", 1
    )[0]
    assert 'class_name: str = "SafeRecurrentDistillation"' in distill
    assert "critic_hidden_dims" in source
    assert "num_mini_batches: int = 4" in distill
    assert 'schedule: str = "fixed"' in distill
    assert "teacher_mix: float = 1.0" in distill
    assert "teacher_mix_end: float = 0.0" in distill
    assert "learning_rate: float = 1.0e-4" in distill
    assert "use_clipped_value_loss: bool = True" in distill
    assert "desired_kl: float = 0.01" in distill
    assert "critic_warmup_iters: int = 0" in distill
    assert "pg_delay_iters: int = 0" in distill
    assert "behavior_coef_end: float = 1.0" in distill
    assert "behavior_coef_decay_iters: int = 0" in distill
    assert "teacher_mix_decay_iters: int = 1000" in distill
    assert 'run_name: str = "s12_final_main"' in agent
    assert "max_iterations: int = 15000" in agent
    assert "T4SparseDepthStudentFinalMainAlgCfg" in agent
    assert "pg_coef: float = 0.0" in distill
    assert "pg_coef: float = 0.2" in final_main
    assert "pg_delay_iters: int = 2000" in final_main
    assert "critic_warmup_iters: int = 200" in final_main
    assert "teacher_mix_decay_iters: int = 2000" in final_main
    assert "learning_rate: float = 1.0e-4" in final_main
    assert "max_recon_grad_ratio: float = 1.0" in distill
    assert "max_action_std: float = 0.2" in source
    assert "pg_coef: float = 0.2" in joint_alg
    assert "pg_coef_ramp_iters: int = 800" in joint_alg
    assert "critic_warmup_iters: int = 200" in joint_alg
    assert "learning_rate: float = 3.0e-5" in joint_alg
    assert 'schedule: str = "fixed"' in joint_alg
    assert "pg_coef: float = 0.5" in ft_alg
    assert "pg_coef_ramp_iters: int = 400" in ft_alg
    assert "behavior_coef: float = 0.5" in ft_alg
    assert "behavior_coef_end: float = 0.5" in ft_alg
    assert "learning_rate: float = 1.0e-5" in targeted_alg
    assert "pg_coef: float = 0.1" in targeted_alg
    assert "reference_action_coef: float = 0.25" in targeted_alg
    assert "behavior_coef: float = 1.0" in targeted_alg
    assert "recon_coef: float = 1.0" in targeted_alg
    assert 'run_name: str = "s12_rtx_gated_joint"' in source
    assert 'run_name: str = "s12_rtx_deploy_ft_v2"' in source
    assert 'run_name: str = "s12_rtx_targeted_robust_ft"' in source
    assert 'run_name: str = "s12_residual_ft"' in residual_agent
    assert "max_iterations: int = 1000" in residual_agent
    assert "teacher_mix: float = 0.0" in residual_alg
    assert "critic_warmup_iters: int = 200" in residual_alg
    assert "pg_coef: float = 0.5" in residual_alg
    assert "pg_coef_ramp_iters: int = 400" in residual_alg
    assert "behavior_coef: float = 0.5" in residual_alg
    assert "recon_coef: float = 0.5" in residual_alg
    assert "learning_rate: float = 3.0e-5" in residual_alg
    assert "reference_action_coef: float = 0.0" in residual_alg
    assert "pg_coef: float = 0.1" not in residual_alg
    assert "behavior_coef: float = 1.0" not in residual_alg
    assert "reference_action_coef: float = 0.25" not in residual_alg
    assert "pg_coef: float = 1.0" not in residual_alg
    assert "behavior_coef: float = 0.0" not in residual_alg
    assert "max_kl:" not in source
    assert "rollback_lr_factor" not in source
    params = inspect.signature(SafeRecurrentDistillation.__init__).parameters
    for name in (
        "pg_coef",
        "pg_coef_ramp_iters",
        "pg_delay_iters",
        "teacher_mix",
        "teacher_mix_end",
        "teacher_mix_decay_iters",
        "behavior_coef",
        "behavior_coef_end",
        "behavior_coef_decay_iters",
        "critic_warmup_iters",
        "schedule",
        "learning_rate",
        "reference_action_coef",
    ):
        assert name in params, name


def test_residual_ft_pg_waits_for_critic_warmup_and_keeps_bc_floor():
    algorithm = _algorithm(
        teacher_mix=0.0,
        teacher_mix_end=0.0,
        teacher_mix_decay_iters=0,
        pg_coef=0.5,
        pg_coef_ramp_iters=400,
        pg_delay_iters=0,
        critic_warmup_iters=200,
        behavior_coef=0.5,
        behavior_coef_end=0.5,
        behavior_coef_decay_iters=0,
        schedule="fixed",
        reference_action_coef=0.0,
    )
    algorithm.num_updates = 0
    assert algorithm.current_teacher_mix() == pytest.approx(0.0)
    assert algorithm.current_pg_coef() == pytest.approx(0.0)
    algorithm.num_updates = 200
    assert algorithm.current_pg_coef() == pytest.approx(0.0)
    algorithm.num_updates = 400
    assert algorithm.current_pg_coef() == pytest.approx(0.25)
    algorithm.num_updates = 600
    assert algorithm.current_pg_coef() == pytest.approx(0.5)
    assert algorithm.current_behavior_coef() == pytest.approx(0.5)
    source = CFG_PATH.read_text(encoding="utf-8")
    residual_alg = source.split("class T4SparseDepthStudentResidualFtAlgCfg", 1)[1].split(
        "class T4SparseDepthStudentResidualFtAgentCfg", 1
    )[0]
    assert "pg_coef: float = 0.5" in residual_alg
    assert "behavior_coef: float = 0.5" in residual_alg
    assert "reference_action_coef: float = 0.0" in residual_alg
    assert "pg_coef: float = 1.0" not in residual_alg
    assert "behavior_coef: float = 0.0" not in residual_alg


def test_final_main_pg_waits_for_mix_zero_and_critic_warmup():
    algorithm = _algorithm(
        teacher_mix=1.0,
        teacher_mix_end=0.0,
        teacher_mix_decay_iters=2000,
        pg_coef=0.2,
        pg_coef_ramp_iters=800,
        pg_delay_iters=2000,
        critic_warmup_iters=200,
        behavior_coef=1.0,
        behavior_coef_end=1.0,
        behavior_coef_decay_iters=0,
        schedule="fixed",
    )
    algorithm.num_updates = 0
    assert algorithm.current_teacher_mix() == pytest.approx(1.0)
    assert algorithm.current_pg_coef() == pytest.approx(0.0)
    algorithm.num_updates = 1000
    assert algorithm.current_teacher_mix() == pytest.approx(0.5)
    assert algorithm.current_pg_coef() == pytest.approx(0.0)
    algorithm.num_updates = 2000
    assert algorithm.current_teacher_mix() == pytest.approx(0.0)
    assert algorithm.current_pg_coef() == pytest.approx(0.0)
    algorithm.num_updates = 2200
    assert algorithm.current_pg_coef() == pytest.approx(0.0)
    algorithm.num_updates = 2600
    assert algorithm.current_pg_coef() == pytest.approx(0.1)
    algorithm.num_updates = 3000
    assert algorithm.current_pg_coef() == pytest.approx(0.2)
    assert algorithm.current_behavior_coef() == pytest.approx(1.0)


def test_sparse_distill_cfg_objects_match_gated_three_phase_recipe():
    pytest.importorskip("isaaclab")
    pytest.importorskip("warp")
    from legged_lab.envs.t4.depth_student_cfg import (
        T4SparseDepthStudentDeployFtAgentCfg,
        T4SparseDepthStudentAgentCfg,
        T4SparseDepthStudentDaggerAlgCfg,
        T4SparseDepthStudentFinalMainAlgCfg,
        T4SparseDepthStudentFtAgentCfg,
        T4SparseDepthStudentJointAlgCfg,
        T4SparseDepthStudentReprFirstAgentCfg,
        T4SparseDepthStudentResidualFtAgentCfg,
        T4SparseDepthStudentResidualFtAlgCfg,
        T4SparseDepthStudentPlantFtAgentCfg,
        T4SparseDepthStudentPlantFtAlgCfg,
        T4SparseDepthStudentTargetedFtAgentCfg,
        T4SparseDepthStudentTargetedFtAlgCfg,
    )
    from legged_lab.envs.t4.depth_student_env import (
        T4LocoSparseDepthStudentEnvCfg,
        T4LocoSparseDepthStudentPlantFtEnvCfg,
        T4LocoSparseDepthStudentReprFirstEnvCfg,
        T4LocoSparseDepthStudentResidualFtEnvCfg,
        T4LocoSparseDepthStudentTargetedFtEnvCfg,
    )

    distill = T4SparseDepthStudentDaggerAlgCfg()
    assert distill.pg_coef == pytest.approx(0.0)
    assert distill.pg_delay_iters == 0
    assert distill.teacher_mix == pytest.approx(1.0)
    assert distill.teacher_mix_end == pytest.approx(0.0)
    assert distill.teacher_mix_decay_iters == 1000
    assert distill.schedule == "fixed"
    assert distill.learning_rate == pytest.approx(1.0e-4)
    assert distill.behavior_coef == pytest.approx(1.0)
    assert distill.behavior_coef_end == pytest.approx(1.0)
    assert distill.behavior_coef_decay_iters == 0
    assert distill.critic_warmup_iters == 0
    agent = T4SparseDepthStudentAgentCfg()
    assert isinstance(agent.algorithm, T4SparseDepthStudentFinalMainAlgCfg)
    assert agent.run_name == "s12_final_main"
    assert agent.max_iterations == 15000
    assert agent.save_interval == 500
    assert agent.algorithm.pg_coef == pytest.approx(0.2)
    assert agent.algorithm.pg_coef_ramp_iters == 800
    assert agent.algorithm.pg_delay_iters == 2000
    assert agent.algorithm.critic_warmup_iters == 200
    assert agent.algorithm.teacher_mix == pytest.approx(1.0)
    assert agent.algorithm.teacher_mix_end == pytest.approx(0.0)
    assert agent.algorithm.teacher_mix_decay_iters == 2000
    assert agent.algorithm.learning_rate == pytest.approx(1.0e-4)
    assert agent.algorithm.behavior_coef == pytest.approx(1.0)
    assert agent.algorithm.behavior_coef_end == pytest.approx(1.0)
    assert agent.algorithm.behavior_coef_decay_iters == 0
    assert agent.algorithm.recon_coef == pytest.approx(1.0)
    assert agent.algorithm.entropy_coef == pytest.approx(0.0)
    assert agent.algorithm.schedule == "fixed"
    repr_agent = T4SparseDepthStudentReprFirstAgentCfg()
    repr_env = T4LocoSparseDepthStudentReprFirstEnvCfg()
    assert repr_agent.run_name == "s12_repr_first"
    assert repr_agent.max_iterations == 14000
    assert repr_agent.algorithm.repr_first is True
    assert repr_agent.algorithm.pg_coef == pytest.approx(0.0)
    assert repr_agent.algorithm.teacher_mix == pytest.approx(1.0)
    assert repr_agent.algorithm.teacher_mix_decay_iters == 0
    assert repr_agent.algorithm.action_mix_decay_iters == 2000
    assert repr_agent.algorithm.repr_cap_iters == 4000
    assert repr_env.random_level_reset_fraction == pytest.approx(0.50)
    assert repr_env.random_level_reset_min_level == 6
    joint_agent = T4SparseDepthStudentFtAgentCfg()
    joint = joint_agent.algorithm
    assert isinstance(joint, T4SparseDepthStudentJointAlgCfg)
    assert joint_agent.run_name == "s12_rtx_gated_joint"
    assert joint.pg_coef == pytest.approx(0.2)
    assert joint.pg_coef_ramp_iters == 800
    assert joint.critic_warmup_iters == 200
    ft_agent = T4SparseDepthStudentDeployFtAgentCfg()
    ft = ft_agent.algorithm
    assert ft_agent.run_name == "s12_rtx_deploy_ft_v2"
    assert ft.pg_coef == pytest.approx(0.5)
    assert ft.pg_coef_ramp_iters == 400
    assert ft.critic_warmup_iters == 200
    targeted_agent = T4SparseDepthStudentTargetedFtAgentCfg()
    targeted = targeted_agent.algorithm
    assert isinstance(targeted, T4SparseDepthStudentTargetedFtAlgCfg)
    assert targeted_agent.run_name == "s12_rtx_targeted_robust_ft"
    assert targeted_agent.max_iterations == 800
    assert targeted.learning_rate == pytest.approx(1.0e-5)
    assert targeted.pg_coef == pytest.approx(0.1)
    assert targeted.reference_action_coef == pytest.approx(0.25)
    dagger = _algorithm(
        teacher_mix=distill.teacher_mix,
        teacher_mix_end=distill.teacher_mix_end,
        teacher_mix_decay_iters=distill.teacher_mix_decay_iters,
        behavior_coef=distill.behavior_coef,
        behavior_coef_end=distill.behavior_coef_end,
        behavior_coef_decay_iters=distill.behavior_coef_decay_iters,
        critic_warmup_iters=distill.critic_warmup_iters,
        pg_coef=distill.pg_coef,
        pg_delay_iters=distill.pg_delay_iters,
    )
    assert dagger.current_pg_coef() == pytest.approx(0.0)
    assert dagger.current_teacher_mix() == pytest.approx(1.0)
    joint_alg = _algorithm(
        teacher_mix=joint.teacher_mix,
        teacher_mix_end=joint.teacher_mix_end,
        teacher_mix_decay_iters=joint.teacher_mix_decay_iters,
        behavior_coef=joint.behavior_coef,
        behavior_coef_end=joint.behavior_coef_end,
        critic_warmup_iters=joint.critic_warmup_iters,
        pg_coef=joint.pg_coef,
        pg_coef_ramp_iters=joint.pg_coef_ramp_iters,
    )
    joint_alg.num_updates = 200
    assert joint_alg.current_pg_coef() == pytest.approx(0.0)
    joint_alg.num_updates = 600
    assert joint_alg.current_pg_coef() == pytest.approx(0.1)
    ft_alg = _algorithm(
        teacher_mix=ft.teacher_mix,
        teacher_mix_end=ft.teacher_mix_end,
        teacher_mix_decay_iters=ft.teacher_mix_decay_iters,
        behavior_coef=ft.behavior_coef,
        behavior_coef_end=ft.behavior_coef_end,
        critic_warmup_iters=ft.critic_warmup_iters,
        pg_coef=ft.pg_coef,
        pg_coef_ramp_iters=ft.pg_coef_ramp_iters,
        pg_delay_iters=ft.pg_delay_iters,
        schedule=ft.schedule,
    )
    ft_alg.num_updates = 400
    assert ft_alg.current_pg_coef() == pytest.approx(0.25)
    assert ft_alg.current_behavior_coef() == pytest.approx(0.5)
    assert ft_alg.current_teacher_mix() == pytest.approx(0.0)
    residual_agent = T4SparseDepthStudentResidualFtAgentCfg()
    residual = residual_agent.algorithm
    assert isinstance(residual, T4SparseDepthStudentResidualFtAlgCfg)
    assert residual_agent.run_name == "s12_residual_ft"
    assert residual_agent.max_iterations == 1000
    assert residual.teacher_mix == pytest.approx(0.0)
    assert residual.teacher_mix_end == pytest.approx(0.0)
    assert residual.critic_warmup_iters == 200
    assert residual.pg_coef == pytest.approx(0.5)
    assert residual.pg_coef_ramp_iters == 400
    assert residual.pg_delay_iters == 0
    assert residual.behavior_coef == pytest.approx(0.5)
    assert residual.behavior_coef_end == pytest.approx(0.5)
    assert residual.recon_coef == pytest.approx(0.5)
    assert residual.learning_rate == pytest.approx(3.0e-5)
    assert residual.schedule == "fixed"
    assert residual.reference_action_coef == pytest.approx(0.0)
    assert residual.pg_coef != pytest.approx(0.1)
    assert residual.behavior_coef != pytest.approx(1.0)
    assert residual.behavior_coef != pytest.approx(0.0)
    assert residual.pg_coef != pytest.approx(1.0)
    residual_alg = _algorithm(
        teacher_mix=residual.teacher_mix,
        teacher_mix_end=residual.teacher_mix_end,
        teacher_mix_decay_iters=residual.teacher_mix_decay_iters,
        behavior_coef=residual.behavior_coef,
        behavior_coef_end=residual.behavior_coef_end,
        critic_warmup_iters=residual.critic_warmup_iters,
        pg_coef=residual.pg_coef,
        pg_coef_ramp_iters=residual.pg_coef_ramp_iters,
        pg_delay_iters=residual.pg_delay_iters,
        schedule=residual.schedule,
        reference_action_coef=residual.reference_action_coef,
    )
    residual_alg.num_updates = 0
    assert residual_alg.current_pg_coef() == pytest.approx(0.0)
    assert residual_alg.current_teacher_mix() == pytest.approx(0.0)
    residual_alg.num_updates = 200
    assert residual_alg.current_pg_coef() == pytest.approx(0.0)
    residual_alg.num_updates = 400
    assert residual_alg.current_pg_coef() == pytest.approx(0.25)
    residual_alg.num_updates = 600
    assert residual_alg.current_pg_coef() == pytest.approx(0.5)
    assert residual_alg.current_behavior_coef() == pytest.approx(0.5)
    plant_agent = T4SparseDepthStudentPlantFtAgentCfg()
    plant = plant_agent.algorithm
    assert isinstance(plant, T4SparseDepthStudentPlantFtAlgCfg)
    assert plant_agent.run_name == "s12_plant_ft"
    assert plant_agent.max_iterations == 1000
    assert plant.learning_rate == pytest.approx(1.0e-5)
    assert plant.pg_coef == pytest.approx(0.1)
    assert plant.behavior_coef == pytest.approx(1.0)
    assert plant.recon_coef == pytest.approx(1.0)
    assert plant.reference_action_coef == pytest.approx(0.25)
    assert plant.critic_warmup_iters == 200
    main_env = T4LocoSparseDepthStudentEnvCfg()
    residual_env = T4LocoSparseDepthStudentResidualFtEnvCfg()
    targeted_env = T4LocoSparseDepthStudentTargetedFtEnvCfg()
    plant_env = T4LocoSparseDepthStudentPlantFtEnvCfg()
    assert main_env.random_level_reset_fraction == pytest.approx(0.10)
    assert main_env.random_level_reset_min_level is None
    assert residual_env.random_level_reset_fraction == pytest.approx(0.50)
    assert residual_env.random_level_reset_min_level == 6
    assert residual_env.student_depth_noise is True
    assert residual_env.student_depth_boundary_corruption is False
    assert residual_env.domain_rand.action_delay.enable is False
    residual_stones = residual_env.scene.terrain_generator.sub_terrains["stepping_stones"]
    residual_pillars = residual_env.scene.terrain_generator.sub_terrains["raised_pillars"]
    main_stones = main_env.scene.terrain_generator.sub_terrains["stepping_stones"]
    main_pillars = main_env.scene.terrain_generator.sub_terrains["raised_pillars"]
    assert residual_stones.proportion == pytest.approx(0.30)
    assert residual_pillars.proportion == pytest.approx(0.30)
    assert residual_stones.proportion > main_stones.proportion
    assert residual_pillars.proportion > main_pillars.proportion
    assert not getattr(residual_stones, "targeted_layout_seed", False)
    assert not getattr(residual_pillars, "targeted_manufacturing_variation", False)
    assert targeted_env.student_depth_boundary_corruption is True
    assert getattr(
        targeted_env.scene.terrain_generator.sub_terrains["raised_pillars"],
        "targeted_manufacturing_variation",
        False,
    )
    assert plant_env.random_level_reset_fraction == pytest.approx(0.10)
    assert plant_env.random_level_reset_min_level is None
    assert plant_env.student_depth_noise is True
    assert plant_env.student_depth_boundary_corruption is False
    assert plant_env.domain_rand.action_delay.enable is True
    assert plant_env.domain_rand.action_delay.params["max_delay"] == 2
    plant_pillars = plant_env.scene.terrain_generator.sub_terrains["raised_pillars"]
    assert plant_pillars.proportion == pytest.approx(main_pillars.proportion)
    assert not getattr(plant_pillars, "targeted_manufacturing_variation", False)
    assert hasattr(plant_env.domain_rand.events, "actuator_gains")


def test_runner_maps_safe_algorithm_to_teacher_observation_contract():
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert 'self.alg_cfg["class_name"] == "SafeRecurrentDistillation"' in source
    assert 'self.training_type = "safe_distillation"' in source
    assert 'self.training_type in {"distillation", "safe_distillation"}' in source
    assert 'self.training_type in {"rl", "safe_distillation"}' in source
    assert 'self.training_type in {"distillation", "safe_distillation"}' in source
    assert 'saved_dict["algorithm_state_dict"]' in source
    assert 'loaded_dict["algorithm_state_dict"]' in source
    assert 'tag = key if "/" in key else f"Loss/{key}"' in source


def test_repr_first_locks_mix_and_freezes_actor_while_updating_scan():
    algorithm = _algorithm(
        teacher_mix=1.0,
        teacher_mix_end=0.0,
        teacher_mix_decay_iters=0,
        pg_coef=0.0,
        behavior_coef=1.0,
        behavior_coef_end=1.0,
        behavior_coef_decay_iters=0,
        repr_first=True,
        repr_cap_iters=10000,
        repr_probe_interval=10000,
        learning_rate=3.0e-3,
    )
    assert algorithm.current_teacher_mix() == pytest.approx(1.0)
    assert algorithm.current_behavior_coef() == pytest.approx(0.0)
    assert algorithm.current_pg_coef() == pytest.approx(0.0)

    frozen_prefixes = ("student.", "std", "critic.")
    trained_prefixes = ("depth_encoder.", "scan_decoder.", "memory_s.")
    before = _clone_policy_state(algorithm.policy)
    _collect(algorithm)
    report = algorithm.update()
    after = algorithm.policy.state_dict()
    assert report["actor_frozen"] == pytest.approx(1.0)
    assert report["teacher_mix"] == pytest.approx(1.0)
    assert report["Distill/phase"] == pytest.approx(0.0)
    for name, tensor in before.items():
        if name == "std" or name.startswith(frozen_prefixes):
            assert torch.equal(tensor, after[name]), name
    assert any(
        not torch.equal(before[name], after[name])
        for name in before
        if any(name.startswith(prefix) for prefix in trained_prefixes)
    )


def test_repr_first_skips_recon_projection_until_action_phase():
    original = SafeRecurrentDistillation._project_auxiliary_gradient.__func__
    seen = {"calls": 0}

    def _spy(cls, auxiliary, control):
        seen["calls"] += 1
        return original(cls, auxiliary, control)

    SafeRecurrentDistillation._project_auxiliary_gradient = classmethod(_spy)
    try:
        representation = _algorithm(
            pg_coef=0.0,
            repr_first=True,
            repr_cap_iters=10000,
            repr_probe_interval=10000,
        )
        _collect(representation)
        representation.update()
        assert seen["calls"] == 0

        action = _algorithm(pg_coef=0.0, repr_first=True, repr_cap_iters=10000, repr_probe_interval=10000)
        action._repr_state.phase = "action"
        action._repr_state.switch_iter = 0
        _collect(action)
        action.update()
        assert seen["calls"] >= 1
        helper = inspect.getsource(SafeRecurrentDistillation._optimizer_step_with_recon_projection)
        assert "protect_recon" in helper
        assert "_project_auxiliary_gradient(shared_control, shared_recon)" in helper
        assert "_limit_auxiliary_gradient_norm" in helper
    finally:
        SafeRecurrentDistillation._project_auxiliary_gradient = classmethod(original)


def test_repr_first_mix_decays_from_switch_iter_and_curriculum_snaps():
    algorithm = _algorithm(
        repr_first=True,
        action_mix_decay_iters=2000,
        pg_coef=0.0,
        behavior_coef=1.0,
        behavior_coef_end=1.0,
        behavior_coef_decay_iters=0,
    )
    algorithm.num_updates = 80
    assert algorithm.current_teacher_mix() == pytest.approx(1.0)

    class _Cfg:
        random_level_reset_fraction = 0.50
        random_level_reset_min_level = 6

    class _Env:
        cfg = _Cfg()

    env = _Env()
    algorithm.attach_repr_runtime(env)
    algorithm._repr_state.phase = "action"
    algorithm._repr_state.switch_iter = 5
    algorithm._repr_state.switch_reason = "metric"
    algorithm._apply_action_curriculum()
    algorithm.num_updates = 5
    assert algorithm.current_teacher_mix() == pytest.approx(1.0)
    algorithm.num_updates = 1005
    assert algorithm.current_teacher_mix() == pytest.approx(0.5)
    algorithm.num_updates = 2005
    assert algorithm.current_teacher_mix() == pytest.approx(0.0)
    assert algorithm.current_behavior_coef() == pytest.approx(1.0)
    assert env.cfg.random_level_reset_fraction == pytest.approx(0.10)
    assert env.cfg.random_level_reset_min_level is None


def test_repr_first_cap_switches_once_and_checkpoint_restores_phase():
    algorithm = _algorithm(repr_first=True, repr_cap_iters=2, repr_probe_interval=1, pg_coef=0.0)
    switched = []
    algorithm.attach_repr_runtime(None, on_switch=lambda alg: switched.append(alg._repr_state.switch_reason))
    _collect(algorithm)
    first = algorithm.update()
    assert first["Distill/phase"] == pytest.approx(0.0)
    assert algorithm._repr_state.phase == "representation"
    _collect(algorithm)
    second = algorithm.update()
    assert algorithm._repr_state.phase == "action"
    assert algorithm._repr_state.switch_reason == "baseline_invalid"
    assert second["Distill/phase"] == pytest.approx(1.0)
    assert switched == ["baseline_invalid"]
    frozen_reason = algorithm._repr_state.switch_reason
    frozen_iter = algorithm._repr_state.switch_iter
    _collect(algorithm)
    algorithm.update()
    assert algorithm._repr_state.phase == "action"
    assert algorithm._repr_state.switch_reason == frozen_reason
    assert algorithm._repr_state.switch_iter == frozen_iter

    restored = _algorithm(repr_first=True, pg_coef=0.0)
    restored.load_checkpoint_state_dict(algorithm.checkpoint_state_dict())
    assert restored._repr_state.phase == "action"
    assert restored._repr_state.switch_reason == "baseline_invalid"
    assert restored._repr_state.switch_iter == frozen_iter
    assert restored.current_teacher_mix() < 1.0


def test_repr_first_attach_after_action_checkpoint_snaps_curriculum():
    saved = _algorithm(repr_first=True, pg_coef=0.0)
    saved._repr_state.phase = "action"
    saved._repr_state.switch_iter = 12
    saved._repr_state.switch_reason = "metric"
    payload = saved.checkpoint_state_dict()

    class _Cfg:
        random_level_reset_fraction = 0.50
        random_level_reset_min_level = 6

    class _Env:
        cfg = _Cfg()

    restored = _algorithm(repr_first=True, pg_coef=0.0)
    restored.load_checkpoint_state_dict(payload)
    assert restored._repr_state.phase == "action"
    env = _Env()
    assert env.cfg.random_level_reset_fraction == pytest.approx(0.50)
    restored.attach_repr_runtime(env)
    assert env.cfg.random_level_reset_fraction == pytest.approx(0.10)
    assert env.cfg.random_level_reset_min_level is None


def test_repr_first_probe_sync_sums_counts_across_ranks():
    algorithm = _algorithm(repr_first=True, pg_coef=0.0)
    algorithm.is_multi_gpu = True
    algorithm._probe_acc.sse_global = 4.0
    algorithm._probe_acc.count_global = 4
    algorithm._probe_acc.sse_stones = 2.0
    algorithm._probe_acc.count_stones = 2
    algorithm._probe_acc.agree_stones = 1.0
    algorithm._probe_acc.sse_pillars = 2.0
    algorithm._probe_acc.count_pillars = 2
    algorithm._probe_acc.agree_pillars = 1.0
    ops = []

    def _sum_reduce(tensor, op=None):
        ops.append(op)
        tensor.mul_(2)

    import torch.distributed as dist

    original = dist.all_reduce
    dist.all_reduce = _sum_reduce
    try:
        algorithm._sync_probe_accumulator()
    finally:
        dist.all_reduce = original

    assert ops == [dist.ReduceOp.SUM]
    assert algorithm._probe_acc.count_global == 8
    assert algorithm._probe_acc.count_stones == 4
    assert algorithm._probe_acc.count_pillars == 4
    assert algorithm._probe_acc.sse_global == pytest.approx(8.0)
    source = inspect.getsource(SafeRecurrentDistillation._advance_repr_phase)
    assert source.find("_sync_probe_accumulator") < source.find("finalize()")


def test_repr_first_probe_accumulates_before_transition_clear():
    algorithm = _algorithm(repr_first=True, pg_coef=0.0)
    algorithm.init_storage("safe_distillation", 4, 4, [20], [12], [2])
    algorithm.act(_student_obs(4), _teacher_obs(4))
    assert algorithm._probe_acc.count_global == 0
    algorithm.process_env_step(torch.zeros(4), torch.zeros(4), {"time_outs": torch.zeros(4)})
    assert algorithm._probe_acc.count_global == 4
    assert algorithm.transition.privileged_observations is None
    source = inspect.getsource(SafeRecurrentDistillation.process_env_step)
    assert source.find("_accumulate_scan_probe") < source.find("super().process_env_step")

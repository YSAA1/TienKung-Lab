from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms.distillation import assert_finite_grads, assert_finite_tensor
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import unpad_trajectories


class SafeRecurrentDistillation:
    """Recurrent student improvement with DAgger anchors and transactional PPO updates."""

    def __init__(
        self,
        policy,
        num_learning_epochs=2,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        behavior_coef=1.0,
        behavior_coef_end=0.0,
        behavior_coef_decay_iters=2000,
        pg_coef=0.5,
        recon_coef=1.0,
        max_recon_grad_ratio=1.0,
        teacher_mix=0.5,
        teacher_mix_end=0.0,
        teacher_mix_decay_iters=2000,
        desired_kl=0.01,
        max_kl=0.03,
        max_kl_emergency=0.3,
        max_behavior_drift=0.02,
        rollback_lr_factor=0.5,
        min_learning_rate=1.0e-5,
        max_learning_rate=1.0e-3,
        max_grad_norm=1.0,
        nan_guard=True,
        device="cpu",
        multi_gpu_cfg: dict | None = None,
    ):
        if not getattr(policy, "is_recurrent", False):
            raise ValueError("SafeRecurrentDistillation requires a recurrent student policy")
        if getattr(policy, "critic", None) is None:
            raise ValueError("SafeRecurrentDistillation requires a training-time asymmetric critic")
        if num_mini_batches <= 0:
            raise ValueError("num_mini_batches must be positive")
        if max_kl < 0.0:
            raise ValueError("max_kl must be non-negative")
        if max_kl_emergency < max_kl:
            raise ValueError("max_kl_emergency must be greater than or equal to max_kl")
        if max_recon_grad_ratio < 0.0:
            raise ValueError("max_recon_grad_ratio must be non-negative")

        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is None:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1
        else:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]

        self.rnd = None
        self.policy = policy.to(device)
        self.optimizer = optim.Adam(list(policy.student_parameters()), lr=learning_rate)
        self.storage = None
        self.transition = RolloutStorage.Transition()

        self.num_learning_epochs = int(num_learning_epochs)
        self.num_mini_batches = int(num_mini_batches)
        self.learning_rate = float(learning_rate)
        self.clip_param = float(clip_param)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.value_loss_coef = float(value_loss_coef)
        self.entropy_coef = float(entropy_coef)
        self.behavior_coef = float(behavior_coef)
        self.behavior_coef_end = float(behavior_coef_end)
        self.behavior_coef_decay_iters = int(behavior_coef_decay_iters)
        self.pg_coef = float(pg_coef)
        self.recon_coef = float(recon_coef)
        self.max_recon_grad_ratio = float(max_recon_grad_ratio)
        self.teacher_mix = float(teacher_mix)
        self.teacher_mix_end = float(teacher_mix_end)
        self.teacher_mix_decay_iters = int(teacher_mix_decay_iters)
        self.desired_kl = None if desired_kl is None else float(desired_kl)
        self.max_kl = float(max_kl)
        self.max_kl_emergency = float(max_kl_emergency)
        self.max_behavior_drift = float(max_behavior_drift)
        self.rollback_lr_factor = float(rollback_lr_factor)
        self.min_learning_rate = float(min_learning_rate)
        self.max_learning_rate = float(max_learning_rate)
        self.max_grad_norm = float(max_grad_norm)
        self.nan_guard = bool(nan_guard)
        self.num_updates = 0
        self.num_accepted_updates = 0
        self.rollback_count = 0

    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, student_obs_shape, teacher_obs_shape, actions_shape
    ):
        if training_type != "safe_distillation":
            raise ValueError(f"expected training_type='safe_distillation', got {training_type!r}")
        if num_envs % self.num_mini_batches != 0:
            raise ValueError("num_envs must be divisible by num_mini_batches")
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            student_obs_shape,
            teacher_obs_shape,
            actions_shape,
            None,
            self.device,
        )

    def _accepted_update_schedule(self, start, end, decay_iters):
        if decay_iters <= 0:
            return end
        progress = min(float(self.num_accepted_updates) / float(decay_iters), 1.0)
        return start + (end - start) * progress

    def current_teacher_mix(self):
        return self._accepted_update_schedule(self.teacher_mix, self.teacher_mix_end, self.teacher_mix_decay_iters)

    def current_behavior_coef(self):
        return self._accepted_update_schedule(
            self.behavior_coef,
            self.behavior_coef_end,
            self.behavior_coef_decay_iters,
        )

    def _guard_step(self):
        storage_step = self.storage.step if self.storage is not None else -1
        return f"update={self.num_updates},storage={storage_step}"

    def _assert_finite(self, tensor, name, stage):
        if self.nan_guard:
            assert_finite_tensor(
                tensor,
                name,
                step=self._guard_step(),
                rank=self.gpu_global_rank,
                stage=stage,
            )

    @classmethod
    def _detach_hidden_states(cls, hidden_states):
        if hidden_states is None:
            return None
        if isinstance(hidden_states, tuple):
            return tuple(cls._detach_hidden_states(hidden_state) for hidden_state in hidden_states)
        return hidden_states.detach()

    def act(self, obs, teacher_obs):
        self._assert_finite(obs, "student_obs", "ingest")
        self._assert_finite(teacher_obs, "teacher_obs", "ingest")
        self.transition.hidden_states = self._detach_hidden_states(self.policy.get_hidden_states())
        student_actions = self.policy.act(obs).detach()
        teacher_actions = self.policy.evaluate(teacher_obs).detach()
        values = self.policy.evaluate_value(teacher_obs).detach()
        mix = self.current_teacher_mix()
        if mix >= 1.0:
            executed = teacher_actions
            student_action_mask = torch.zeros(student_actions.shape[0], dtype=torch.bool, device=self.device)
        elif mix <= 0.0:
            executed = student_actions
            student_action_mask = torch.ones(student_actions.shape[0], dtype=torch.bool, device=self.device)
        else:
            take_teacher = torch.rand(student_actions.shape[0], device=self.device) < mix
            executed = torch.where(take_teacher.unsqueeze(-1), teacher_actions, student_actions)
            student_action_mask = ~take_teacher

        actions_log_prob = self.policy.distribution.log_prob(executed).sum(dim=-1).detach()
        for name, tensor in (
            ("teacher_actions", teacher_actions),
            ("executed_actions", executed),
            ("actions_log_prob", actions_log_prob),
            ("values", values),
        ):
            self._assert_finite(tensor, name, "ingest")

        self.transition.observations = obs
        self.transition.privileged_observations = teacher_obs
        self.transition.actions = executed
        self.transition.privileged_actions = teacher_actions
        self.transition.student_action_mask = student_action_mask
        self.transition.values = values
        self.transition.actions_log_prob = actions_log_prob
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        return executed

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_teacher_obs):
        last_values = self.policy.evaluate_value(last_teacher_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam, normalize_advantage=False)
        self._normalize_advantages()

    def _normalize_advantages(self):
        advantages = self.storage.advantages
        moments = torch.stack(
            [advantages.sum(), advantages.square().sum(), advantages.new_tensor(float(advantages.numel()))]
        )
        if self.is_multi_gpu:
            torch.distributed.all_reduce(moments, op=torch.distributed.ReduceOp.SUM)
        mean = moments[0] / moments[2]
        variance = torch.clamp(moments[1] / moments[2] - mean.square(), min=0.0)
        self.storage.advantages.copy_((advantages - mean) / (torch.sqrt(variance) + 1.0e-8))

    @staticmethod
    def _gaussian_kl(old_mu, old_sigma, new_mu, new_sigma):
        old_sigma = torch.clamp(old_sigma, min=1.0e-6)
        new_sigma = torch.clamp(new_sigma, min=1.0e-6)
        return torch.sum(
            torch.log(new_sigma / old_sigma)
            + (old_sigma.square() + (old_mu - new_mu).square()) / (2.0 * new_sigma.square())
            - 0.5,
            dim=-1,
        )

    def _reduce_gradients(self, excluded_parameters=()):
        excluded_ids = {id(parameter) for parameter in excluded_parameters}
        params = [parameter for group in self.optimizer.param_groups for parameter in group["params"]]
        params = [parameter for parameter in params if parameter.grad is not None and id(parameter) not in excluded_ids]
        if not params:
            return
        flat = torch.cat([parameter.grad.view(-1) for parameter in params])
        torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
        flat /= self.gpu_world_size
        offset = 0
        for parameter in params:
            numel = parameter.numel()
            parameter.grad.copy_(flat[offset : offset + numel].view_as(parameter.grad))
            offset += numel

    @staticmethod
    def _gradient_dot(left, right):
        return sum((lhs * rhs).sum() for lhs, rhs in zip(left, right))

    @classmethod
    def _gradient_cosine(cls, left, right):
        dot = cls._gradient_dot(left, right)
        left_norm = torch.sqrt(torch.clamp(cls._gradient_dot(left, left), min=0.0))
        right_norm = torch.sqrt(torch.clamp(cls._gradient_dot(right, right), min=0.0))
        denominator = left_norm * right_norm
        safe_denominator = torch.clamp(denominator, min=torch.finfo(denominator.dtype).eps)
        return torch.where(denominator > 0.0, dot / safe_denominator, dot.new_zeros(()))

    @classmethod
    def _project_auxiliary_gradient(cls, auxiliary, control):
        dot = cls._gradient_dot(auxiliary, control)
        control_norm_sq = cls._gradient_dot(control, control)
        conflict = (dot < 0.0) & (control_norm_sq > 0.0)
        safe_norm_sq = torch.clamp(control_norm_sq, min=torch.finfo(control_norm_sq.dtype).eps)
        coefficient = torch.where(conflict, dot / safe_norm_sq, dot.new_zeros(()))
        projected = [aux - coefficient * reference for aux, reference in zip(auxiliary, control)]
        return projected, conflict.to(dtype=dot.dtype)

    @classmethod
    def _limit_auxiliary_gradient_norm(cls, auxiliary, control, *, coefficient, max_ratio):
        auxiliary_norm = torch.sqrt(torch.clamp(cls._gradient_dot(auxiliary, auxiliary), min=0.0))
        control_norm = torch.sqrt(torch.clamp(cls._gradient_dot(control, control), min=0.0))
        weighted_coefficient = abs(float(coefficient))
        if weighted_coefficient == 0.0 or float(max_ratio) == 0.0:
            scale = auxiliary_norm.new_zeros(())
        else:
            denominator = torch.clamp(
                auxiliary_norm * weighted_coefficient,
                min=torch.finfo(auxiliary_norm.dtype).eps,
            )
            requested_scale = float(max_ratio) * control_norm / denominator
            scale = torch.where(
                (auxiliary_norm > 0.0) & (control_norm > 0.0),
                torch.clamp(requested_scale, max=1.0),
                auxiliary_norm.new_zeros(()),
            )
        return [gradient * scale for gradient in auxiliary], scale

    def _objective_gradients(self, loss, parameters):
        if not loss.requires_grad:
            return [torch.zeros_like(parameter) for parameter in parameters]
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        gradients = [
            torch.zeros_like(parameter) if gradient is None else gradient
            for parameter, gradient in zip(parameters, gradients)
        ]
        return gradients

    def _synchronize_gradient_sets(self, gradient_sets, parameters):
        if not self.is_multi_gpu:
            return gradient_sets
        packed = torch.stack(
            [torch.cat([gradient.reshape(-1) for gradient in gradients]) for gradients in gradient_sets]
        )
        torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
        packed /= self.gpu_world_size
        synchronized_sets = []
        for flat in packed:
            synchronized = []
            offset = 0
            for parameter in parameters:
                numel = parameter.numel()
                synchronized.append(flat[offset : offset + numel].view_as(parameter))
                offset += numel
            synchronized_sets.append(synchronized)
        return synchronized_sets

    def _coordinated_shared_gradients(self, behavior_loss, pg_loss, recon_loss, behavior_coef):
        shared_parameters = list(self.policy.shared_encoder_parameters())
        behavior_gradients = self._objective_gradients(behavior_loss, shared_parameters)
        pg_gradients = self._objective_gradients(pg_loss, shared_parameters)
        recon_gradients = self._objective_gradients(recon_loss, shared_parameters)
        behavior_gradients, pg_gradients, recon_gradients = self._synchronize_gradient_sets(
            [behavior_gradients, pg_gradients, recon_gradients], shared_parameters
        )
        control_gradients = [
            behavior_coef * behavior + self.pg_coef * pg for behavior, pg in zip(behavior_gradients, pg_gradients)
        ]
        projected_recon, recon_conflict = self._project_auxiliary_gradient(recon_gradients, control_gradients)
        limited_recon, recon_grad_scale = self._limit_auxiliary_gradient_norm(
            projected_recon,
            control_gradients,
            coefficient=self.recon_coef,
            max_ratio=self.max_recon_grad_ratio,
        )
        combined = [control + self.recon_coef * recon for control, recon in zip(control_gradients, limited_recon)]
        diagnostics = {
            "grad_norm_behavior": torch.sqrt(
                torch.clamp(self._gradient_dot(behavior_gradients, behavior_gradients), min=0.0)
            ),
            "grad_norm_pg": torch.sqrt(torch.clamp(self._gradient_dot(pg_gradients, pg_gradients), min=0.0)),
            "grad_norm_recon": torch.sqrt(torch.clamp(self._gradient_dot(recon_gradients, recon_gradients), min=0.0)),
            "grad_cos_behavior_pg": self._gradient_cosine(behavior_gradients, pg_gradients),
            "grad_cos_behavior_recon": self._gradient_cosine(behavior_gradients, recon_gradients),
            "grad_cos_pg_recon": self._gradient_cosine(pg_gradients, recon_gradients),
            "grad_conflict_recon_control": recon_conflict,
            "recon_grad_scale": recon_grad_scale,
        }
        return shared_parameters, combined, diagnostics

    def _optimizer_step(self, loss, shared_parameters=None, shared_gradients=None):
        self.optimizer.zero_grad()
        loss.backward()
        if self.is_multi_gpu:
            self._reduce_gradients(excluded_parameters=shared_parameters or ())
        if shared_parameters is not None and shared_gradients is not None:
            for parameter, gradient in zip(shared_parameters, shared_gradients):
                parameter.grad = gradient.detach().clone()
        if self.nan_guard:
            optimized_ids = {id(parameter) for group in self.optimizer.param_groups for parameter in group["params"]}
            named = [
                (name, parameter)
                for name, parameter in self.policy.named_parameters()
                if id(parameter) in optimized_ids
            ]
            assert_finite_grads(named, step=self._guard_step(), rank=self.gpu_global_rank)
        nn.utils.clip_grad_norm_(
            [parameter for group in self.optimizer.param_groups for parameter in group["params"]], self.max_grad_norm
        )
        self.optimizer.step()
        self.policy.project_action_std_()

    def _batch_losses(self, batch):
        (
            obs_batch,
            teacher_obs_batch,
            actions_batch,
            teacher_actions_batch,
            student_action_mask_batch,
            old_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            actor_hidden_batch,
            masks_batch,
        ) = batch

        self.policy.update_distribution(obs_batch, masks=masks_batch, hidden_states=actor_hidden_batch)
        log_prob = self.policy.distribution.log_prob(actions_batch).sum(dim=-1)
        mu = self.policy.action_mean
        student_rows = student_action_mask_batch.squeeze(-1)
        if student_rows.any():
            ratio = torch.exp(log_prob[student_rows] - old_actions_log_prob_batch.squeeze(-1)[student_rows])
            advantages = advantages_batch.squeeze(-1)[student_rows]
            surrogate = -advantages * ratio
            surrogate_clipped = -advantages * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            pg_loss = torch.max(surrogate, surrogate_clipped).mean()
        else:
            ratio = mu.new_ones(1)
            pg_loss = mu.new_zeros(())

        behavior_loss = nn.functional.mse_loss(mu, teacher_actions_batch)
        behavior_coef = self.current_behavior_coef()
        value = self.policy.evaluate_value(teacher_obs_batch, masks=masks_batch)
        value_clipped = old_values_batch + (value - old_values_batch).clamp(-self.clip_param, self.clip_param)
        value_loss = torch.max((value - returns_batch).square(), (value_clipped - returns_batch).square()).mean()
        teacher_obs_unpadded = unpad_trajectories(teacher_obs_batch, masks_batch)
        recon_loss = nn.functional.mse_loss(self.policy.reconstruct(), self.policy.teacher_scan(teacher_obs_unpadded))
        entropy = self.policy.distribution.entropy().sum(dim=-1).mean()
        loss = (
            behavior_coef * behavior_loss
            + self.pg_coef * pg_loss
            + self.recon_coef * recon_loss
            + self.value_loss_coef * value_loss
            - self.entropy_coef * entropy
        )
        for name, tensor in (
            ("behavior_loss", behavior_loss),
            ("pg_loss", pg_loss),
            ("recon_loss", recon_loss),
            ("value_loss", value_loss),
            ("total_loss", loss),
        ):
            self._assert_finite(tensor, name, "loss")
        shared_parameters, shared_gradients, gradient_diagnostics = self._coordinated_shared_gradients(
            behavior_loss, pg_loss, recon_loss, behavior_coef
        )
        return (
            loss,
            behavior_loss,
            pg_loss,
            recon_loss,
            value_loss,
            ratio,
            shared_parameters,
            shared_gradients,
            gradient_diagnostics,
        )

    def _distributed_mean(self, value_sum, count):
        pair = torch.stack([value_sum, value_sum.new_tensor(float(count))])
        if self.is_multi_gpu:
            torch.distributed.all_reduce(pair, op=torch.distributed.ReduceOp.SUM)
        return (pair[0] / torch.clamp(pair[1], min=1.0)).item()

    def _gather_1d(self, values):
        values = values.reshape(-1)
        if not self.is_multi_gpu:
            return values

        local_count = torch.tensor([values.numel()], device=values.device, dtype=torch.long)
        gathered_counts = [torch.zeros_like(local_count) for _ in range(self.gpu_world_size)]
        torch.distributed.all_gather(gathered_counts, local_count)
        counts = [int(count.item()) for count in gathered_counts]
        padded_size = max(1, max(counts))
        padded = values.new_zeros(padded_size)
        padded[: values.numel()].copy_(values)
        gathered_values = [torch.empty_like(padded) for _ in range(self.gpu_world_size)]
        torch.distributed.all_gather(gathered_values, padded)
        if sum(counts) == 0:
            return values.new_empty(0)
        return torch.cat([rank_values[:count] for rank_values, count in zip(gathered_values, counts)])

    @torch.no_grad()
    def _candidate_metrics(self):
        kl_values = []
        ratio_values = []
        behavior_sum = torch.zeros((), device=self.device)
        behavior_count = 0
        generator = self.storage.safe_recurrent_mini_batch_generator(self.num_mini_batches, 1)
        for batch in generator:
            (
                obs_batch,
                _teacher_obs_batch,
                actions_batch,
                teacher_actions_batch,
                student_action_mask_batch,
                _old_values_batch,
                _advantages_batch,
                _returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                actor_hidden_batch,
                masks_batch,
            ) = batch
            self.policy.update_distribution(obs_batch, masks=masks_batch, hidden_states=actor_hidden_batch)
            new_mu = self.policy.action_mean
            new_sigma = self.policy.action_std
            kl_values.append(self._gaussian_kl(old_mu_batch, old_sigma_batch, new_mu, new_sigma).reshape(-1))
            behavior_sum += nn.functional.mse_loss(new_mu, teacher_actions_batch, reduction="sum")
            behavior_count += teacher_actions_batch.numel()
            student_rows = student_action_mask_batch.squeeze(-1)
            if student_rows.any():
                new_log_prob = self.policy.distribution.log_prob(actions_batch).sum(dim=-1)
                ratio = torch.exp(new_log_prob[student_rows] - old_actions_log_prob_batch.squeeze(-1)[student_rows])
                ratio_values.append(ratio.reshape(-1))

        kl_global = self._gather_1d(torch.cat(kl_values))
        kl_mean = kl_global.mean()
        kl_p95 = torch.quantile(kl_global, 0.95)
        kl_max = kl_global.max()

        ratio_local = torch.cat(ratio_values) if ratio_values else kl_global.new_empty(0)
        ratio_global = self._gather_1d(ratio_local)
        if ratio_global.numel() > 0:
            ratio_p95 = torch.quantile(ratio_global, 0.95)
            ratio_max = ratio_global.max()
            clip_fraction = (torch.abs(ratio_global - 1.0) > self.clip_param).float().mean()
        else:
            ratio_p95 = kl_mean.new_tensor(1.0)
            ratio_max = kl_mean.new_tensor(1.0)
            clip_fraction = kl_mean.new_zeros(())

        behavior_pair = torch.stack([behavior_sum, behavior_sum.new_tensor(float(behavior_count))])
        if self.is_multi_gpu:
            torch.distributed.all_reduce(behavior_pair, op=torch.distributed.ReduceOp.SUM)
        return {
            "kl_mean": kl_mean.item(),
            "kl_p95": kl_p95.item(),
            "kl_max": kl_max.item(),
            "ratio_p95": ratio_p95.item(),
            "ratio_max": ratio_max.item(),
            "clip_fraction": clip_fraction.item(),
            "post_behavior": (behavior_pair[0] / torch.clamp(behavior_pair[1], min=1.0)).item(),
        }

    def _set_learning_rate(self, learning_rate):
        self.learning_rate = float(max(self.min_learning_rate, min(self.max_learning_rate, learning_rate)))
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate

    def checkpoint_state_dict(self):
        return {
            "num_updates": self.num_updates,
            "num_accepted_updates": self.num_accepted_updates,
            "rollback_count": self.rollback_count,
            "learning_rate": self.learning_rate,
        }

    def load_checkpoint_state_dict(self, state_dict):
        num_updates = int(state_dict.get("num_updates", 0))
        num_accepted_updates = int(state_dict.get("num_accepted_updates", num_updates))
        rollback_count = int(state_dict.get("rollback_count", num_updates - num_accepted_updates))
        if min(num_updates, num_accepted_updates, rollback_count) < 0 or num_accepted_updates > num_updates:
            raise ValueError("invalid safe recurrent checkpoint counters")
        self.num_updates = num_updates
        self.num_accepted_updates = num_accepted_updates
        self.rollback_count = rollback_count
        self._set_learning_rate(float(state_dict.get("learning_rate", self.learning_rate)))

    def _rollback_reasons(self, metrics, pre_behavior):
        finite_metrics = all(math.isfinite(value) for value in metrics.values())
        rollback_kl_p95 = finite_metrics and metrics["kl_p95"] > self.max_kl
        rollback_kl_emergency = finite_metrics and metrics["kl_max"] > self.max_kl_emergency
        rollback_kl = (not finite_metrics) or rollback_kl_p95 or rollback_kl_emergency
        rollback_behavior = finite_metrics and metrics["post_behavior"] > pre_behavior + self.max_behavior_drift
        return rollback_kl, rollback_kl_p95, rollback_kl_emergency, rollback_behavior

    def update(self):
        self.num_updates += 1
        policy_backup = {name: value.detach().clone() for name, value in self.policy.state_dict().items()}
        optimizer_backup = copy.deepcopy(self.optimizer.state_dict())
        pre_behavior_error = (self.storage.mu - self.storage.privileged_actions).square()
        pre_behavior = self._distributed_mean(pre_behavior_error.sum(), pre_behavior_error.numel())

        totals = {
            name: torch.zeros((), device=self.device)
            for name in (
                "behavior",
                "pg",
                "recon",
                "value_function",
                "grad_norm_behavior",
                "grad_norm_pg",
                "grad_norm_recon",
                "grad_cos_behavior_pg",
                "grad_cos_behavior_recon",
                "grad_cos_pg_recon",
                "grad_conflict_recon_control",
                "recon_grad_scale",
            )
        }
        num_batches = 0
        generator = self.storage.safe_recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch in generator:
            (
                loss,
                behavior,
                pg,
                recon,
                value,
                _ratio,
                shared_parameters,
                shared_gradients,
                gradient_diagnostics,
            ) = self._batch_losses(batch)
            self._optimizer_step(loss, shared_parameters, shared_gradients)
            totals["behavior"] += behavior.detach()
            totals["pg"] += pg.detach()
            totals["recon"] += recon.detach()
            totals["value_function"] += value.detach()
            for name, metric in gradient_diagnostics.items():
                totals[name] += metric.detach()
            num_batches += 1

        metrics = self._candidate_metrics()
        rollback_kl, rollback_kl_p95, rollback_kl_emergency, rollback_behavior = self._rollback_reasons(
            metrics, pre_behavior
        )
        accepted = not rollback_kl and not rollback_behavior
        if not accepted:
            self.policy.load_state_dict(policy_backup)
            self.optimizer.load_state_dict(optimizer_backup)
            self.rollback_count += 1
            self._set_learning_rate(self.learning_rate * self.rollback_lr_factor)
        else:
            self.num_accepted_updates += 1
            if self.desired_kl is not None:
                if metrics["kl_mean"] > self.desired_kl * 2.0:
                    self._set_learning_rate(self.learning_rate / 1.5)
                elif 0.0 < metrics["kl_mean"] < self.desired_kl / 2.0:
                    self._set_learning_rate(self.learning_rate * 1.5)

        self.storage.clear()
        divisor = max(1, num_batches)
        report = {name: (value / divisor).item() for name, value in totals.items()}
        report.update(metrics)
        report.update(
            {
                "pre_behavior": pre_behavior,
                "behavior_drift": metrics["post_behavior"] - pre_behavior,
                "teacher_mix": self.current_teacher_mix(),
                "behavior_coef": self.current_behavior_coef(),
                "learning_rate": self.learning_rate,
                "update_accepted": float(accepted),
                "rollback_kl": float(rollback_kl),
                "rollback_kl_p95": float(rollback_kl_p95),
                "rollback_kl_emergency": float(rollback_kl_emergency),
                "rollback_behavior": float(rollback_behavior),
                "rollback_count": float(self.rollback_count),
                "accepted_update_count": float(self.num_accepted_updates),
                "action_std_min": self.policy.std.detach().min().item(),
                "action_std_mean": self.policy.std.detach().mean().item(),
                "action_std_max": self.policy.std.detach().max().item(),
            }
        )
        return report

    def broadcast_parameters(self):
        model_params = [self.policy.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])

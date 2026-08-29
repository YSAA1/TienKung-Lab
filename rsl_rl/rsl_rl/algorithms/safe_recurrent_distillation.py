from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms.distillation import assert_finite_grads, assert_finite_tensor, mix_teacher_student_actions
from rsl_rl.algorithms.ppo import PPO, adapt_ppo_learning_rate, gaussian_kl, ppo_clipped_surrogate_loss, ppo_clipped_value_loss
from rsl_rl.utils import unpad_trajectories


class SafeRecurrentDistillation(PPO):
    """DAgger extras on top of teacher PPO: GAE, clip, and mean-KL LR come from ``PPO``."""

    def __init__(
        self,
        policy,
        num_learning_epochs=2,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        behavior_coef=1.0,
        behavior_coef_end=0.0,
        behavior_coef_decay_iters=2000,
        pg_coef=0.5,
        pg_coef_ramp_iters=0,
        pg_delay_iters=0,
        critic_warmup_iters=200,
        recon_coef=1.0,
        max_recon_grad_ratio=1.0,
        reference_action_coef=0.0,
        teacher_mix=0.0,
        teacher_mix_end=0.0,
        teacher_mix_decay_iters=2000,
        desired_kl=0.01,
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
        if critic_warmup_iters < 0:
            raise ValueError("critic_warmup_iters must be non-negative")
        if pg_coef_ramp_iters < 0:
            raise ValueError("pg_coef_ramp_iters must be non-negative")
        if pg_delay_iters < 0:
            raise ValueError("pg_delay_iters must be non-negative")
        if max_recon_grad_ratio < 0.0:
            raise ValueError("max_recon_grad_ratio must be non-negative")
        if reference_action_coef < 0.0:
            raise ValueError("reference_action_coef must be non-negative")
        if float(pg_coef) > 0.0 and abs(float(teacher_mix_end)) > 0.0:
            raise ValueError(
                "SafeRecurrentDistillation requires teacher_mix=0 whenever pg_coef>0; "
                f"got teacher_mix={teacher_mix}, teacher_mix_end={teacher_mix_end}, pg_coef={pg_coef}"
            )
        if float(pg_coef) > 0.0 and abs(float(teacher_mix)) > 0.0:
            mix_zero_at = max(0, int(teacher_mix_decay_iters))
            if mix_zero_at > int(pg_delay_iters):
                raise ValueError(
                    "SafeRecurrentDistillation requires teacher_mix=0 whenever pg_coef>0; "
                    "set pg_delay_iters >= teacher_mix_decay_iters so mix hits 0 before PPO "
                    f"(got mix_decay={teacher_mix_decay_iters}, pg_delay={pg_delay_iters})"
                )

        super().__init__(
            policy,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            device=device,
            normalize_advantage_per_mini_batch=False,
            min_learning_rate=min_learning_rate,
            max_learning_rate=max_learning_rate,
            rnd_cfg=None,
            symmetry_cfg=None,
            multi_gpu_cfg=multi_gpu_cfg,
        )
        self.optimizer = optim.Adam(list(policy.student_parameters()), lr=learning_rate)
        self.behavior_coef = float(behavior_coef)
        self.behavior_coef_end = float(behavior_coef_end)
        self.behavior_coef_decay_iters = int(behavior_coef_decay_iters)
        self.pg_coef = float(pg_coef)
        self.pg_coef_ramp_iters = int(pg_coef_ramp_iters)
        self.pg_delay_iters = int(pg_delay_iters)
        self.critic_warmup_iters = int(critic_warmup_iters)
        self.recon_coef = float(recon_coef)
        self.max_recon_grad_ratio = float(max_recon_grad_ratio)
        self.reference_action_coef = float(reference_action_coef)
        self.reference_policy = None
        self.teacher_mix = float(teacher_mix)
        self.teacher_mix_end = float(teacher_mix_end)
        self.teacher_mix_decay_iters = int(teacher_mix_decay_iters)
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
        super().init_storage(
            training_type,
            num_envs,
            num_transitions_per_env,
            student_obs_shape,
            teacher_obs_shape,
            actions_shape,
        )

    def _update_schedule(self, start, end, decay_iters):
        if decay_iters <= 0:
            return end
        progress = min(float(self.num_updates) / float(decay_iters), 1.0)
        return start + (end - start) * progress

    def current_teacher_mix(self):
        return self._update_schedule(self.teacher_mix, self.teacher_mix_end, self.teacher_mix_decay_iters)

    def current_behavior_coef(self):
        return self._update_schedule(
            self.behavior_coef,
            self.behavior_coef_end,
            self.behavior_coef_decay_iters,
        )

    def current_pg_coef(self):
        if self.num_updates <= self.pg_delay_iters:
            return 0.0
        if self._in_critic_warmup():
            return 0.0
        if self.pg_coef_ramp_iters <= 0:
            return self.pg_coef
        elapsed = self.num_updates - self.pg_delay_iters - self.critic_warmup_iters
        if elapsed <= 0:
            return 0.0
        progress = min(float(elapsed) / float(self.pg_coef_ramp_iters), 1.0)
        return self.pg_coef * progress

    def _in_critic_warmup(self) -> bool:
        if self.pg_coef <= 0.0 or self.critic_warmup_iters <= 0:
            return False
        if self.num_updates <= self.pg_delay_iters:
            return False
        elapsed = self.num_updates - self.pg_delay_iters
        return elapsed <= self.critic_warmup_iters

    def _critic_parameter_ids(self) -> set[int]:
        critic = getattr(self.policy, "critic", None)
        if critic is None:
            return set()
        return {id(parameter) for parameter in critic.parameters()}

    def _zero_non_critic_grads(self) -> None:
        """Parkour-in-the-Wild: freeze the cloned actor until the value head has on-policy data."""
        critic_ids = self._critic_parameter_ids()
        for group in self.optimizer.param_groups:
            for parameter in group["params"]:
                if id(parameter) not in critic_ids and parameter.grad is not None:
                    parameter.grad = None

    def _assert_student_only_when_ppo(self):
        pg_coef = self.current_pg_coef()
        teacher_mix = self.current_teacher_mix()
        if pg_coef > 0.0 and abs(teacher_mix) > 0.0:
            raise RuntimeError(
                "SafeRecurrentDistillation requires teacher_mix=0 whenever pg_coef>0; "
                f"got teacher_mix={teacher_mix}, pg_coef={pg_coef}"
            )

    def _guard_step(self):
        storage_step = self.storage.step if self.storage is not None else -1
        return f"update={self.num_updates},storage={storage_step}"

    def _should_check_ingest(self, tensor) -> bool:
        """CPU always checks. CUDA ingest checks once per rollout.

        ``assert_finite_tensor`` uses a 0-dim CUDA bool in a Python ``if``, which
        ``.item()``-syncs. Doing that on obs/actions every control step drains the
        physics pipeline. Loss/grad guards still run every optimizer step. The v3
        collapse showed up in the updater, not ingest.
        """
        if not self.nan_guard or not torch.is_tensor(tensor):
            return False
        if tensor.device.type != "cuda":
            return True
        step = 0 if self.storage is None else int(getattr(self.storage, "step", 0))
        return step == 0

    def _assert_finite(self, tensor, name, stage):
        if self.nan_guard:
            assert_finite_tensor(
                tensor,
                name,
                step=self._guard_step(),
                rank=self.gpu_global_rank,
                stage=stage,
            )

    def set_reference_policy_from_current(self):
        """Freeze the loaded parent actor so cumulative FT drift remains observable and bounded."""
        self.reference_policy = copy.deepcopy(self.policy).to(self.device)
        self.reference_policy.eval()
        self.reference_policy.reset()
        for parameter in self.reference_policy.parameters():
            parameter.requires_grad_(False)
        return self.reference_policy

    @classmethod
    def _detach_hidden_states(cls, hidden_states):
        if hidden_states is None:
            return None
        if isinstance(hidden_states, tuple):
            return tuple(cls._detach_hidden_states(hidden_state) for hidden_state in hidden_states)
        return hidden_states.detach()

    def act(self, obs, teacher_obs):
        self._assert_student_only_when_ppo()
        if self._should_check_ingest(obs):
            self._assert_finite(obs, "student_obs", "ingest")
            self._assert_finite(teacher_obs, "teacher_obs", "ingest")
        self.transition.hidden_states = self._detach_hidden_states(self.policy.get_hidden_states())
        student_actions = self.policy.act(obs).detach()
        if self.reference_policy is None:
            if self.reference_action_coef > 0.0:
                raise RuntimeError("reference_action_coef>0 requires set_reference_policy_from_current()")
            reference_actions = self.policy.action_mean.detach()
        else:
            with torch.no_grad():
                reference_actions = self.reference_policy.act_inference(obs).detach()
        teacher_actions = self.policy.evaluate(teacher_obs).detach()
        values = self.policy.evaluate_value(teacher_obs).detach()
        executed, student_action_mask = mix_teacher_student_actions(
            student_actions, teacher_actions, self.current_teacher_mix()
        )
        actions_log_prob = self.policy.distribution.log_prob(executed).sum(dim=-1).detach()
        if self._should_check_ingest(executed):
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
        self.transition.reference_actions = reference_actions
        self.transition.student_action_mask = student_action_mask
        self.transition.values = values
        self.transition.actions_log_prob = actions_log_prob
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        return executed

    def process_env_step(self, rewards, dones, infos):
        super().process_env_step(rewards, dones, infos)
        if self.reference_policy is not None:
            self.reference_policy.reset(dones)

    def compute_returns(self, last_teacher_obs):
        last_values = self.policy.evaluate_value(last_teacher_obs).detach()
        if self.is_multi_gpu:
            self.storage.compute_returns(last_values, self.gamma, self.lam, normalize_advantage=False)
            self._normalize_advantages()
            return
        self.storage.compute_returns(
            last_values,
            self.gamma,
            self.lam,
            normalize_advantage=not self.normalize_advantage_per_mini_batch,
        )

    def _normalize_advantages(self):
        advantages = self.storage.advantages
        moments = torch.stack(
            [advantages.sum(), advantages.square().sum(), advantages.new_tensor(float(advantages.numel()))]
        )
        torch.distributed.all_reduce(moments, op=torch.distributed.ReduceOp.SUM)
        mean = moments[0] / moments[2]
        variance = torch.clamp(moments[1] / moments[2] - mean.square(), min=0.0)
        self.storage.advantages.copy_((advantages - mean) / (torch.sqrt(variance) + 1.0e-8))

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

    def _all_reduce_tensors(self, tensors):
        tensors = [tensor for tensor in tensors if tensor is not None]
        if not self.is_multi_gpu or not tensors:
            return
        flat = torch.cat([tensor.reshape(-1) for tensor in tensors])
        torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
        flat /= float(self.gpu_world_size)
        offset = 0
        for tensor in tensors:
            numel = tensor.numel()
            tensor.copy_(flat[offset : offset + numel].view_as(tensor))
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

    def _optimizer_step(self, loss, *, check_finite=True):
        self.optimizer.zero_grad()
        loss.backward()
        if self.is_multi_gpu:
            self._reduce_gradients()
        if self._in_critic_warmup():
            self._zero_non_critic_grads()
        if self.nan_guard and check_finite:
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
        if not self._in_critic_warmup():
            self.policy.project_action_std_()

    def _shared_encoder_params(self):
        shared = getattr(self.policy, "shared_encoder_parameters", None)
        if shared is None:
            return []
        return list(shared())

    @staticmethod
    def _clone_grad(parameter):
        if parameter.grad is None:
            return None
        return parameter.grad.detach().clone()

    def _optimizer_step_with_recon_projection(self, control_loss, recon_loss, *, check_finite=True):
        """Split control vs recon backward, then project recon off the shared encoder."""
        shared = self._shared_encoder_params()
        optimized = [parameter for group in self.optimizer.param_groups for parameter in group["params"]]
        shared_ids = {id(parameter) for parameter in shared}
        recon_needs_backward = (
            recon_loss is not None
            and recon_loss.requires_grad
            and float(self.recon_coef) != 0.0
            and not self._in_critic_warmup()
        )

        self.optimizer.zero_grad()
        if not recon_needs_backward or not shared:
            total = control_loss if not recon_needs_backward else control_loss + recon_loss
            self._optimizer_step(total, check_finite=check_finite)
            return {
                "recon_control_cosine": 0.0,
                "recon_grad_norm": 0.0,
                "control_grad_norm": 0.0,
                "recon_grad_scale": 0.0,
                "recon_conflict": 0.0,
            }

        control_loss.backward(retain_graph=True)
        control_by_id = {id(parameter): self._clone_grad(parameter) for parameter in optimized}
        shared_control = [
            parameter.grad.detach().clone() if parameter.grad is not None else torch.zeros_like(parameter)
            for parameter in shared
        ]
        self.optimizer.zero_grad()
        recon_loss.backward()
        shared_recon = [
            parameter.grad.detach().clone() if parameter.grad is not None else torch.zeros_like(parameter)
            for parameter in shared
        ]
        if self.is_multi_gpu:
            self._all_reduce_tensors([grad for grad in control_by_id.values() if grad is not None])
            extra_recon = [
                parameter.grad
                for parameter in optimized
                if id(parameter) not in shared_ids and parameter.grad is not None
            ]
            self._all_reduce_tensors([*shared_recon, *extra_recon])
            shared_control = [
                control_by_id[id(parameter)]
                if control_by_id.get(id(parameter)) is not None
                else torch.zeros_like(parameter)
                for parameter in shared
            ]
        cosine = self._gradient_cosine(shared_recon, shared_control)
        projected, conflict = self._project_auxiliary_gradient(shared_recon, shared_control)
        limited, scale = self._limit_auxiliary_gradient_norm(
            projected,
            shared_control,
            coefficient=1.0,
            max_ratio=self.max_recon_grad_ratio,
        )
        for parameter in optimized:
            if id(parameter) in shared_ids:
                continue
            control_grad = control_by_id.get(id(parameter))
            if control_grad is None:
                continue
            if parameter.grad is None:
                parameter.grad = control_grad
            else:
                parameter.grad = control_grad + parameter.grad
        for parameter, recon_grad in zip(shared, limited):
            control_grad = control_by_id.get(id(parameter))
            if control_grad is None:
                parameter.grad = recon_grad
            else:
                parameter.grad = control_grad + recon_grad

        if self._in_critic_warmup():
            self._zero_non_critic_grads()
        if self.nan_guard and check_finite:
            optimized_ids = {id(parameter) for parameter in optimized}
            named = [
                (name, parameter)
                for name, parameter in self.policy.named_parameters()
                if id(parameter) in optimized_ids
            ]
            assert_finite_grads(named, step=self._guard_step(), rank=self.gpu_global_rank)
        nn.utils.clip_grad_norm_(optimized, self.max_grad_norm)
        self.optimizer.step()
        if not self._in_critic_warmup():
            self.policy.project_action_std_()
        recon_norm = torch.sqrt(torch.clamp(self._gradient_dot(shared_recon, shared_recon), min=0.0))
        control_norm = torch.sqrt(torch.clamp(self._gradient_dot(shared_control, shared_control), min=0.0))
        return {
            "recon_control_cosine": float(cosine.detach().item()),
            "recon_grad_norm": float(recon_norm.detach().item()),
            "control_grad_norm": float(control_norm.detach().item()),
            "recon_grad_scale": float(scale.detach().item()),
            "recon_conflict": float(conflict.detach().item()),
        }

    def _batch_losses(self, batch):
        (
            obs_batch,
            teacher_obs_batch,
            actions_batch,
            teacher_actions_batch,
            reference_actions_batch,
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
        sigma = self.policy.action_std
        kl = gaussian_kl(old_mu_batch, old_sigma_batch, mu, sigma)
        student_rows = student_action_mask_batch.squeeze(-1)
        if student_rows.any():
            pg_loss, ratio = ppo_clipped_surrogate_loss(
                advantages_batch.squeeze(-1)[student_rows],
                log_prob[student_rows],
                old_actions_log_prob_batch.squeeze(-1)[student_rows],
                self.clip_param,
            )
        else:
            ratio = None
            pg_loss = mu.new_zeros(())

        behavior_loss = nn.functional.mse_loss(mu, teacher_actions_batch)
        reference_action_loss = nn.functional.mse_loss(mu, reference_actions_batch)
        behavior_coef = self.current_behavior_coef()
        pg_coef = self.current_pg_coef()
        value = self.policy.evaluate_value(teacher_obs_batch, masks=masks_batch)
        value_loss = ppo_clipped_value_loss(
            value,
            old_values_batch,
            returns_batch,
            self.clip_param,
            self.use_clipped_value_loss,
        )
        teacher_obs_unpadded = unpad_trajectories(teacher_obs_batch, masks_batch)
        recon_loss = nn.functional.mse_loss(self.policy.reconstruct(), self.policy.teacher_scan(teacher_obs_unpadded))
        entropy = self.policy.distribution.entropy().sum(dim=-1).mean()
        control_loss = (
            behavior_coef * behavior_loss
            + self.reference_action_coef * reference_action_loss
            + pg_coef * pg_loss
            + self.value_loss_coef * value_loss
            - self.entropy_coef * entropy
        )
        recon_term = self.recon_coef * recon_loss
        loss = control_loss + recon_term
        for name, tensor in (
            ("behavior_loss", behavior_loss),
            ("reference_action_loss", reference_action_loss),
            ("pg_loss", pg_loss),
            ("recon_loss", recon_loss),
            ("value_loss", value_loss),
            ("total_loss", loss),
        ):
            self._assert_finite(tensor, name, "loss")
        return (
            control_loss,
            recon_term,
            behavior_loss,
            reference_action_loss,
            pg_loss,
            recon_loss,
            value_loss,
            ratio,
            kl,
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

    def _kl_ratio_metrics(self, kl_chunks, ratio_chunks):
        kl_local = torch.cat(kl_chunks) if kl_chunks else torch.zeros(1, device=self.device)
        kl_global = self._gather_1d(kl_local)
        kl_mean = kl_global.mean()
        kl_p95 = torch.quantile(kl_global, 0.95)
        kl_p99 = torch.quantile(kl_global, 0.99)
        kl_max = kl_global.max()
        if ratio_chunks:
            ratio_global = self._gather_1d(torch.cat(ratio_chunks))
        else:
            ratio_global = kl_global.new_empty(0)
        if ratio_global.numel() > 0:
            ratio_p95 = torch.quantile(ratio_global, 0.95)
            ratio_max = ratio_global.max()
            clip_fraction = (torch.abs(ratio_global - 1.0) > self.clip_param).float().mean()
        else:
            ratio_p95 = kl_mean.new_tensor(1.0)
            ratio_max = kl_mean.new_tensor(1.0)
            clip_fraction = kl_mean.new_zeros(())
        return {
            "kl_mean": kl_mean.item(),
            "kl_p95": kl_p95.item(),
            "kl_p99": kl_p99.item(),
            "kl_max": kl_max.item(),
            "ratio_p95": ratio_p95.item(),
            "ratio_max": ratio_max.item(),
            "clip_fraction": clip_fraction.item(),
        }

    def _adapt_learning_rate_from_mean_kl(self, kl_mean):
        if self.desired_kl is None or self.schedule != "adaptive":
            return
        if self.gpu_global_rank == 0:
            self.learning_rate = adapt_ppo_learning_rate(
                self.learning_rate,
                float(kl_mean),
                self.desired_kl,
                min_lr=self.min_learning_rate,
                max_lr=self.max_learning_rate,
            )
        if self.is_multi_gpu:
            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(lr_tensor, src=0)
            self.learning_rate = float(lr_tensor.item())
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate

    def _set_learning_rate(self, learning_rate):
        self.learning_rate = float(max(self.min_learning_rate, min(self.max_learning_rate, learning_rate)))
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate

    def checkpoint_state_dict(self):
        state = {
            "num_updates": self.num_updates,
            "num_accepted_updates": self.num_accepted_updates,
            "rollback_count": self.rollback_count,
            "learning_rate": self.learning_rate,
        }
        if self.reference_policy is not None:
            state["reference_model_state_dict"] = {
                key: value.detach().cpu() for key, value in self.reference_policy.deployable_state_dict().items()
            }
        return state

    def load_checkpoint_state_dict(self, state_dict):
        num_updates = int(state_dict.get("num_updates", 0))
        num_accepted_updates = int(state_dict.get("num_accepted_updates", num_updates))
        rollback_count = int(state_dict.get("rollback_count", 0))
        if min(num_updates, num_accepted_updates, rollback_count) < 0:
            raise ValueError("invalid safe recurrent checkpoint counters")
        self.num_updates = num_updates
        self.num_accepted_updates = num_accepted_updates
        self.rollback_count = rollback_count
        self._set_learning_rate(float(state_dict.get("learning_rate", self.learning_rate)))
        reference_state = state_dict.get("reference_model_state_dict")
        if reference_state is not None:
            reference = self.set_reference_policy_from_current()
            reference.load_state_dict(reference_state, strict=False)

    def update(self):
        self._assert_student_only_when_ppo()
        self.num_updates += 1
        self.num_accepted_updates = self.num_updates
        pre_behavior_error = (self.storage.mu - self.storage.privileged_actions).square()
        pre_behavior = self._distributed_mean(pre_behavior_error.sum(), pre_behavior_error.numel())
        planned_minibatches = self.num_mini_batches * self.num_learning_epochs

        totals = {
            name: torch.zeros((), device=self.device)
            for name in ("behavior", "reference_action", "pg", "recon", "value_function")
        }
        projection_totals = {
            "recon_control_cosine": 0.0,
            "recon_grad_norm": 0.0,
            "control_grad_norm": 0.0,
            "recon_grad_scale": 0.0,
            "recon_conflict": 0.0,
        }
        kl_chunks = []
        ratio_chunks = []
        num_batches = 0
        generator = self.storage.safe_recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch in generator:
            control_loss, recon_term, behavior, reference_action, pg, recon, value, ratio, kl = self._batch_losses(
                batch
            )
            projection = self._optimizer_step_with_recon_projection(
                control_loss,
                recon_term,
                check_finite=(num_batches + 1 >= planned_minibatches),
            )
            totals["behavior"] += behavior.detach()
            totals["reference_action"] += reference_action.detach()
            totals["pg"] += pg.detach()
            totals["recon"] += recon.detach()
            totals["value_function"] += value.detach()
            for key in projection_totals:
                projection_totals[key] += float(projection[key])
            kl_chunks.append(kl.reshape(-1).detach())
            if ratio is not None:
                ratio_chunks.append(ratio.reshape(-1).detach())
            num_batches += 1

        metrics = self._kl_ratio_metrics(kl_chunks, ratio_chunks)
        self._adapt_learning_rate_from_mean_kl(metrics["kl_mean"])
        self.storage.clear()
        divisor = max(1, num_batches)
        report = {name: (value / divisor).item() for name, value in totals.items()}
        post_behavior = report["behavior"]
        report.update(metrics)
        report.update({key: value / divisor for key, value in projection_totals.items()})
        report.update(
            {
                "pre_behavior": pre_behavior,
                "post_behavior": post_behavior,
                "behavior_drift": post_behavior - pre_behavior,
                "teacher_mix": self.current_teacher_mix(),
                "behavior_coef": self.current_behavior_coef(),
                "reference_action_coef": self.reference_action_coef,
                "pg_coef": self.current_pg_coef(),
                "critic_warmup": float(self._in_critic_warmup()),
                "actor_frozen": float(self._in_critic_warmup()),
                "learning_rate": self.learning_rate,
                "accepted_minibatches": float(num_batches),
                "planned_minibatches": float(planned_minibatches),
                "accepted_update_count": float(self.num_accepted_updates),
                "action_std_min": self.policy.std.detach().min().item(),
                "action_std_mean": self.policy.std.detach().mean().item(),
                "action_std_max": self.policy.std.detach().max().item(),
            }
        )
        return report

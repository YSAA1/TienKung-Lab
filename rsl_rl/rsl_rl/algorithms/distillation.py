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

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# rsl-rl
from rsl_rl.modules import StudentTeacher, StudentTeacherRecurrent
from rsl_rl.storage import RolloutStorage


def assert_finite_tensor(tensor, name, *, step, rank, stage):
    """Raise if ``tensor`` contains NaN/Inf. ``stage`` distinguishes ingest / loss / grad."""
    if tensor is None or not torch.is_tensor(tensor):
        return
    if torch.isfinite(tensor).all():
        return
    n_nan = int(torch.isnan(tensor).sum().item())
    n_inf = int(torch.isinf(tensor).sum().item())
    raise RuntimeError(
        f"non-finite {stage}: {name} step={step} rank={rank} shape={tuple(tensor.shape)} "
        f"n_nan={n_nan} n_inf={n_inf} dtype={tensor.dtype}"
    )


def assert_finite_grads(named_params, *, step, rank):
    """Raise on the first non-finite gradient so optimizer.step is never reached."""
    for name, param in named_params:
        if param.grad is None:
            continue
        if torch.isfinite(param.grad).all():
            continue
        grad = param.grad
        n_nan = int(torch.isnan(grad).sum().item())
        n_inf = int(torch.isinf(grad).sum().item())
        raise RuntimeError(
            f"non-finite gradient: {name} step={step} rank={rank} shape={tuple(grad.shape)} "
            f"n_nan={n_nan} n_inf={n_inf}"
        )


class Distillation:
    """Distillation algorithm for training a student model to mimic a teacher model."""

    policy: StudentTeacher | StudentTeacherRecurrent
    """The student teacher model."""

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        gradient_length=15,
        learning_rate=1e-3,
        loss_type="mse",
        device="cpu",
        collect_mode="student",
        pg_coef=0.0,
        behavior_coef=1.0,
        clip_param=0.2,
        gamma=0.99,
        teacher_mix=0.0,
        teacher_mix_end=None,
        teacher_mix_decay_iters=0,
        recon_coef=0.0,
        nan_guard=True,
        max_grad_norm=1.0,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ):
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.rnd = None  # TODO: remove when runner has a proper base class

        # distillation components
        self.policy = policy
        self.policy.to(self.device)
        self.storage = None  # initialized later
        student_params = getattr(self.policy, "student_parameters", None)
        params = list(student_params()) if callable(student_params) else list(self.policy.student.parameters())
        self.optimizer = optim.Adam(params, lr=learning_rate)
        self.transition = RolloutStorage.Transition()
        self.last_hidden_states = None

        # distillation parameters
        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length
        self.learning_rate = learning_rate
        if collect_mode not in {"student", "teacher"}:
            raise ValueError(f"Unknown collect_mode={collect_mode!r}; expected 'student' or 'teacher'")
        self.collect_mode = collect_mode
        self.pg_coef = float(pg_coef)
        self.behavior_coef = float(behavior_coef)
        self.clip_param = float(clip_param)
        self.gamma = float(gamma)
        self.teacher_mix = float(teacher_mix)
        self.teacher_mix_end = None if teacher_mix_end is None else float(teacher_mix_end)
        self.teacher_mix_decay_iters = int(teacher_mix_decay_iters)
        self.recon_coef = float(recon_coef)
        self.nan_guard = bool(nan_guard)
        self.max_grad_norm = float(max_grad_norm)

        # initialize the loss function
        if loss_type == "mse":
            self.loss_fn = nn.functional.mse_loss
        elif loss_type == "huber":
            self.loss_fn = nn.functional.huber_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported types are: mse, huber")

        self.num_updates = 0

    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, student_obs_shape, teacher_obs_shape, actions_shape
    ):
        # create rollout storage
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

    def current_teacher_mix(self):
        """Fraction of envs that execute the teacher action this step."""
        if self.collect_mode == "teacher":
            return 1.0
        start = self.teacher_mix
        end = start if self.teacher_mix_end is None else self.teacher_mix_end
        decay = self.teacher_mix_decay_iters
        if decay <= 0:
            return start
        progress = min(float(self.num_updates) / float(decay), 1.0)
        return start + (end - start) * progress

    def _guard_step(self):
        storage_step = self.storage.step if self.storage is not None else -1
        return f"update={self.num_updates},storage={storage_step}"

    def _assert_finite(self, tensor, name, stage):
        if not self.nan_guard:
            return
        assert_finite_tensor(tensor, name, step=self._guard_step(), rank=self.gpu_global_rank, stage=stage)

    def _student_named_parameters(self):
        opt_ids = {id(param) for group in self.optimizer.param_groups for param in group["params"]}
        return [(name, param) for name, param in self.policy.named_parameters() if id(param) in opt_ids]

    def _optimizer_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        if self.is_multi_gpu:
            self.reduce_parameters()
        if self.nan_guard:
            assert_finite_grads(
                self._student_named_parameters(),
                step=self._guard_step(),
                rank=self.gpu_global_rank,
            )
        nn.utils.clip_grad_norm_(
            [param for group in self.optimizer.param_groups for param in group["params"]],
            self.max_grad_norm,
        )
        self.optimizer.step()
        self.policy.detach_hidden_states()

    def act(self, obs, teacher_obs):
        self._assert_finite(obs, "student_obs", "ingest")
        self._assert_finite(teacher_obs, "teacher_obs", "ingest")
        student_actions = self.policy.act(obs)
        student_actions = student_actions.detach()
        teacher_actions = self.policy.evaluate(teacher_obs).detach()
        self._assert_finite(teacher_actions, "teacher_actions", "ingest")
        mix = self.current_teacher_mix()
        if mix >= 1.0:
            executed = teacher_actions
            student_action_mask = torch.zeros(student_actions.shape[0], dtype=torch.bool, device=student_actions.device)
        elif mix <= 0.0:
            executed = student_actions
            student_action_mask = torch.ones(student_actions.shape[0], dtype=torch.bool, device=student_actions.device)
        else:
            take_teacher = torch.rand(student_actions.shape[0], device=student_actions.device) < mix
            executed = torch.where(take_teacher.unsqueeze(-1), teacher_actions, student_actions)
            student_action_mask = ~take_teacher
        self._assert_finite(executed, "executed_actions", "ingest")
        # Ratio must use log π of the action that actually ran, including DAgger
        # teacher-mix rows. Storing the unused student sample makes pg garbage.
        actions_log_prob = self.policy.distribution.log_prob(executed).sum(dim=-1).detach()
        self._assert_finite(actions_log_prob, "actions_log_prob", "ingest")
        self.transition.privileged_actions = teacher_actions
        self.transition.observations = obs
        self.transition.privileged_observations = teacher_obs
        self.transition.actions = executed
        self.transition.student_action_mask = student_action_mask
        self.transition.actions_log_prob = actions_log_prob
        return executed

    def process_env_step(self, rewards, dones, infos):
        # record the rewards and dones
        self.transition.rewards = rewards
        self.transition.dones = dones
        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def _return_advantages(self):
        rewards = self.storage.rewards
        dones = self.storage.dones.float()
        returns = torch.zeros_like(rewards)
        running = torch.zeros(rewards.shape[1], 1, device=rewards.device, dtype=rewards.dtype)
        for step in reversed(range(rewards.shape[0])):
            running = rewards[step] + self.gamma * running * (1.0 - dones[step])
            returns[step] = running
        advantages = (returns - returns.mean()) / (returns.std() + 1.0e-8)
        return advantages

    def update(self):
        self.num_updates += 1
        behavior_acc = torch.zeros((), device=self.device)
        pg_acc = torch.zeros((), device=self.device)
        recon_acc = torch.zeros((), device=self.device)
        loss = 0
        cnt = 0
        use_pg = self.pg_coef > 0.0 and self.collect_mode == "student"
        use_recon = self.recon_coef > 0.0
        advantages = self._return_advantages() if use_pg else None

        for epoch in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            step_index = 0
            loss = 0
            for (
                obs,
                privileged_obs,
                executed_actions,
                privileged_actions,
                dones,
                student_action_mask,
            ) in self.storage.generator():
                # Rollout collection runs under ``torch.inference_mode``.  The
                # stored tensors therefore carry inference-mode metadata, but
                # the student forward below must save activations for backward.
                # Clone at the update boundary to materialize regular tensors.
                with torch.inference_mode(False):
                    obs = obs.clone()
                    privileged_obs = privileged_obs.clone()
                    privileged_actions = privileged_actions.clone()
                    executed_actions = executed_actions.clone()

                # inference the student for gradient computation
                actions = self.policy.act_inference(obs)

                # behavior cloning loss
                behavior_loss = self.loss_fn(actions, privileged_actions)
                self._assert_finite(behavior_loss, "behavior_loss", "loss")
                step_loss = self.behavior_coef * behavior_loss
                if use_pg:
                    if not getattr(self.policy, "is_recurrent", False):
                        self.policy.update_distribution(obs)
                    elif self.policy.distribution is None:
                        self.policy.update_distribution(obs)
                    log_prob = self.policy.distribution.log_prob(executed_actions).sum(dim=-1)
                    old_log_prob = self.storage.actions_log_prob[step_index].squeeze(-1).detach()
                    student_rows = student_action_mask.squeeze(-1)
                    if student_rows.any():
                        log_ratio = log_prob[student_rows] - old_log_prob[student_rows]
                        ratio = torch.exp(log_ratio)
                        adv = advantages[step_index].squeeze(-1).detach()[student_rows]
                        clipped = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                        pg_loss = -torch.min(ratio * adv, clipped * adv).mean()
                    else:
                        pg_loss = actions.new_zeros(())
                    self._assert_finite(pg_loss, "pg_loss", "loss")
                    step_loss = self.behavior_coef * behavior_loss + self.pg_coef * pg_loss
                    pg_acc = pg_acc + pg_loss.detach()
                if use_recon:
                    recon_loss = self.loss_fn(self.policy.reconstruct(), self.policy.teacher_scan(privileged_obs))
                    self._assert_finite(recon_loss, "recon_loss", "loss")
                    step_loss = step_loss + self.recon_coef * recon_loss
                    recon_acc = recon_acc + recon_loss.detach()

                self._assert_finite(step_loss, "total_loss", "loss")
                loss = loss + step_loss
                behavior_acc = behavior_acc + behavior_loss.detach()
                cnt += 1
                step_index += 1

                # gradient step
                if cnt % self.gradient_length == 0:
                    self._optimizer_step(loss)
                    loss = 0

                # reset dones
                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

            if isinstance(loss, torch.Tensor):
                self._optimizer_step(loss)

        mean_behavior_loss = (behavior_acc / cnt).item()
        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        # construct the loss dictionary
        loss_dict = {"behavior": mean_behavior_loss, "teacher_mix": self.current_teacher_mix()}
        if use_pg:
            loss_dict["pg"] = (pg_acc / cnt).item()
        if use_recon:
            loss_dict["recon"] = (recon_acc / cnt).item()

        return loss_dict

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel

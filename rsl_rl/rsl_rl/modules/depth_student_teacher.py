"""Depth-encoder student with a frozen flat-observation teacher."""

from __future__ import annotations

import torch
import torch.nn as nn

from .student_teacher import StudentTeacher


class DepthStudentTeacher(StudentTeacher):
    """Distillation policy consuming proprioception plus depth history."""

    def __init__(
        self,
        num_student_obs,
        num_teacher_obs,
        num_actions,
        depth_shape=(3, 48, 64),
        proprio_obs_dim=960,
        depth_hidden_dim=128,
        student_hidden_dims=(512, 256, 128),
        teacher_hidden_dims=(512, 256, 128),
        activation="elu",
        init_noise_std=0.1,
        **kwargs,
    ):
        del num_student_obs
        super().__init__(
            num_student_obs=depth_hidden_dim + proprio_obs_dim,
            num_teacher_obs=num_teacher_obs,
            num_actions=num_actions,
            student_hidden_dims=student_hidden_dims,
            teacher_hidden_dims=teacher_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            **kwargs,
        )
        self.depth_shape = tuple(depth_shape)
        self.proprio_obs_dim = int(proprio_obs_dim)
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(self.depth_shape[0], 32, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(64 * ((self.depth_shape[1] + 7) // 8) * ((self.depth_shape[2] + 7) // 8), depth_hidden_dim),
            nn.ELU(),
        )

        activation_cls = nn.ELU if activation == "elu" else nn.ReLU
        layers: list[nn.Module] = [
            nn.Linear(depth_hidden_dim + proprio_obs_dim, student_hidden_dims[0]),
            activation_cls(),
        ]
        for index, width in enumerate(student_hidden_dims):
            if index == len(student_hidden_dims) - 1:
                layers.append(nn.Linear(width, num_actions))
            else:
                layers.extend([nn.Linear(width, student_hidden_dims[index + 1]), activation_cls()])
        self.student = nn.Sequential(*layers)

    def _student_features(self, observations: torch.Tensor) -> torch.Tensor:
        leading_shape = observations.shape[:-1]
        flat_observations = observations.reshape(-1, observations.shape[-1])
        proprio = flat_observations[:, : self.proprio_obs_dim]
        depth = flat_observations[:, self.proprio_obs_dim :]
        depth = depth.reshape(flat_observations.shape[0], *self.depth_shape)
        features = torch.cat([proprio, self.depth_encoder(depth)], dim=-1)
        return features.reshape(*leading_shape, features.shape[-1])

    def update_distribution(self, observations):
        mean = self.student(self._student_features(observations))
        std = self.std.expand_as(mean)
        self.distribution = torch.distributions.Normal(mean, std)

    def act_inference(self, observations):
        return self.student(self._student_features(observations))


class DepthStudentTeacherRecurrent(DepthStudentTeacher):
    """Depth CNN + proprio fusion, then GRU. Teacher stays a feed-forward MLP.

    Reconstruction decoder maps the GRU hidden state to the teacher's newest
    HeightScan frame and is training-only. ``deployable_state_dict`` drops it.
    """

    is_recurrent = True

    def __init__(
        self,
        num_student_obs,
        num_teacher_obs,
        num_actions,
        depth_shape=(3, 48, 64),
        proprio_obs_dim=960,
        depth_hidden_dim=128,
        student_hidden_dims=(512, 256, 128),
        teacher_hidden_dims=(512, 256, 128),
        activation="elu",
        init_noise_std=0.1,
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        teacher_recurrent=False,
        recon_scan_dim=0,
        recon_scan_offset=0,
        recon_hidden_dim=128,
        critic_hidden_dims=None,
        min_action_std=1.0e-3,
        max_action_std=None,
        **kwargs,
    ):
        del kwargs
        if teacher_recurrent:
            raise ValueError("sparse depth student keeps teacher_recurrent=False")
        super().__init__(
            num_student_obs=num_student_obs,
            num_teacher_obs=num_teacher_obs,
            num_actions=num_actions,
            depth_shape=depth_shape,
            proprio_obs_dim=proprio_obs_dim,
            depth_hidden_dim=depth_hidden_dim,
            student_hidden_dims=student_hidden_dims,
            teacher_hidden_dims=teacher_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )
        from rsl_rl.networks import Memory

        self.teacher_recurrent = False
        self.rnn_type = str(rnn_type).lower()
        if self.rnn_type != "gru":
            raise ValueError(f"sparse depth student requires rnn_type='gru', got {rnn_type!r}")
        self.rnn_hidden_dim = int(rnn_hidden_dim)
        self.recon_scan_dim = int(recon_scan_dim)
        self.recon_scan_offset = int(recon_scan_offset)
        self.min_action_std = float(min_action_std)
        self.max_action_std = None if max_action_std is None else float(max_action_std)
        fused_dim = int(depth_hidden_dim) + int(proprio_obs_dim)
        self.memory_s = Memory(
            fused_dim, type=self.rnn_type, num_layers=int(rnn_num_layers), hidden_size=self.rnn_hidden_dim
        )
        activation_cls = nn.ELU if activation == "elu" else nn.ReLU
        layers: list[nn.Module] = [nn.Linear(self.rnn_hidden_dim, student_hidden_dims[0]), activation_cls()]
        for index, width in enumerate(student_hidden_dims):
            if index == len(student_hidden_dims) - 1:
                layers.append(nn.Linear(width, num_actions))
            else:
                layers.extend([nn.Linear(width, student_hidden_dims[index + 1]), activation_cls()])
        self.student = nn.Sequential(*layers)
        if self.recon_scan_dim > 0:
            self.scan_decoder = nn.Sequential(
                nn.Linear(self.rnn_hidden_dim, int(recon_hidden_dim)),
                activation_cls(),
                nn.Linear(int(recon_hidden_dim), self.recon_scan_dim),
            )
        else:
            self.scan_decoder = None
        if critic_hidden_dims is not None:
            critic_widths = [int(width) for width in critic_hidden_dims]
            if not critic_widths:
                raise ValueError("critic_hidden_dims must contain at least one layer")
            critic_layers: list[nn.Module] = [nn.Linear(num_teacher_obs, critic_widths[0]), activation_cls()]
            for index, width in enumerate(critic_widths):
                if index == len(critic_widths) - 1:
                    critic_layers.append(nn.Linear(width, 1))
                else:
                    critic_layers.extend([nn.Linear(width, critic_widths[index + 1]), activation_cls()])
            self.critic = nn.Sequential(*critic_layers)
        else:
            self.critic = None
        self._last_hidden = None

    def student_parameters(self):
        params = (
            list(self.depth_encoder.parameters()) + list(self.memory_s.parameters()) + list(self.student.parameters())
        )
        params.append(self.std)
        if self.scan_decoder is not None:
            params.extend(self.scan_decoder.parameters())
        if self.critic is not None:
            params.extend(self.critic.parameters())
        return params

    def shared_encoder_parameters(self):
        return list(self.depth_encoder.parameters()) + list(self.memory_s.parameters())

    def _bounded_std(self, mean: torch.Tensor) -> torch.Tensor:
        if self.max_action_std is None:
            std = torch.clamp(self.std, min=self.min_action_std)
        else:
            std = torch.clamp(self.std, min=self.min_action_std, max=self.max_action_std)
        return std.expand_as(mean)

    def _step_memory(self, observations: torch.Tensor, masks=None, hidden_states=None) -> torch.Tensor:
        fused = self._student_features(observations)
        hidden = self.memory_s(fused, masks=masks, hidden_states=hidden_states)
        if masks is None and hidden.dim() == 3:
            hidden = hidden.squeeze(0)
        self._last_hidden = hidden
        return hidden

    def reset(self, dones=None, hidden_states=None):
        if hidden_states is None:
            hidden_states = (None, None)
        memory_hidden = hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
        self.memory_s.reset(dones, memory_hidden)
        if dones is None:
            self._last_hidden = None

    def get_hidden_states(self):
        return self.memory_s.hidden_states, None

    def detach_hidden_states(self, dones=None):
        self.memory_s.detach_hidden_states(dones)

    def update_distribution(self, observations, masks=None, hidden_states=None):
        mean = self.student(self._step_memory(observations, masks=masks, hidden_states=hidden_states))
        std = self._bounded_std(mean)
        self.distribution = torch.distributions.Normal(mean, std)

    def act(self, observations, masks=None, hidden_states=None):
        self.update_distribution(observations, masks=masks, hidden_states=hidden_states)
        return self.distribution.sample()

    def act_inference(self, observations, masks=None, hidden_states=None):
        mean = self.student(self._step_memory(observations, masks=masks, hidden_states=hidden_states))
        std = self._bounded_std(mean)
        self.distribution = torch.distributions.Normal(mean, std)
        return mean

    def evaluate_value(self, teacher_observations: torch.Tensor, masks=None) -> torch.Tensor:
        if self.critic is None:
            raise RuntimeError("safe recurrent distillation requires critic_hidden_dims")
        if masks is not None:
            from rsl_rl.utils import unpad_trajectories

            teacher_observations = unpad_trajectories(teacher_observations, masks)
        return self.critic(teacher_observations)

    def reconstruct(self) -> torch.Tensor:
        if self.scan_decoder is None:
            raise RuntimeError("reconstruction decoder is disabled")
        if self._last_hidden is None:
            raise RuntimeError("reconstruct() requires a prior student forward")
        return self.scan_decoder(self._last_hidden)

    def teacher_scan(self, teacher_obs: torch.Tensor) -> torch.Tensor:
        if self.recon_scan_dim <= 0:
            raise RuntimeError("recon_scan_dim is 0")
        start = self.recon_scan_offset
        end = start + self.recon_scan_dim
        return teacher_obs[..., start:end]

    def deployable_state_dict(self):
        """Export: CNN + GRU + actor. No teacher, no recon decoder, no critic."""
        return strip_deployable_state_dict(self.state_dict())


DEPLOY_SKIP_PREFIXES = ("teacher.", "scan_decoder.", "critic.")


def strip_deployable_state_dict(state_dict):
    """Drop privileged teacher, scan-decoder, and critic weights from a training checkpoint."""
    return {key: value for key, value in state_dict.items() if not key.startswith(DEPLOY_SKIP_PREFIXES)}


def write_deployable_checkpoint(src, dst):
    """Write a slim deploy package: no optimizer, teacher MLP, scan decoder, or critic."""
    blob = torch.load(src, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or "model_state_dict" not in blob:
        raise ValueError(f"{src} does not contain model_state_dict")
    out = {
        "model_state_dict": strip_deployable_state_dict(blob["model_state_dict"]),
        "iter": blob.get("iter"),
        "infos": blob.get("infos"),
        "deployable": True,
    }
    torch.save(out, dst)
    return out


def _sequential_linear_out_dims(state_dict, prefix: str) -> list[int]:
    dims: list[int] = []
    index = 0
    while index < 64:
        key = f"{prefix}{index}.weight"
        if key in state_dict:
            dims.append(int(state_dict[key].shape[0]))
        elif dims:
            # Skip activation modules that have no weight.
            if f"{prefix}{index}.bias" not in state_dict and not any(
                key.startswith(f"{prefix}{index}.") for key in state_dict
            ):
                index += 1
                continue
        index += 1
        if index > 1 and not dims:
            break
    return dims


def build_depth_student_policy(
    state_dict,
    num_actions,
    *,
    num_teacher_obs,
    depth_shape=(3, 48, 64),
    proprio_obs_dim=960,
    recon_scan_offset=0,
):
    """Rebuild a depth student (feed-forward or GRU) from a checkpoint dict.

    Training checkpoints still contain teacher, scan-decoder, and critic weights.
    This builder strips them before loading. The compatibility policy class may
    still allocate an unused teacher shell, but no privileged weights enter the
    deploy state or inference path.
    """
    state_dict = strip_deployable_state_dict(state_dict)
    student_outs = _sequential_linear_out_dims(state_dict, "student.")
    student_hidden_dims = student_outs[:-1] if len(student_outs) >= 2 else [512, 256, 128]
    depth_hidden_dim = 128
    encoder_outs = _sequential_linear_out_dims(state_dict, "depth_encoder.")
    if encoder_outs:
        depth_hidden_dim = encoder_outs[-1]
    kwargs = {
        "num_student_obs": 1,
        "num_teacher_obs": int(num_teacher_obs),
        "num_actions": int(num_actions),
        "depth_shape": tuple(depth_shape),
        "proprio_obs_dim": int(proprio_obs_dim),
        "depth_hidden_dim": int(depth_hidden_dim),
        "student_hidden_dims": student_hidden_dims,
        "teacher_hidden_dims": student_hidden_dims,
    }
    if any(key.startswith("memory_s") for key in state_dict):
        rnn_hidden_dim = 256
        if "memory_s.rnn.weight_hh_l0" in state_dict:
            rnn_hidden_dim = int(state_dict["memory_s.rnn.weight_hh_l0"].shape[1])
        policy = DepthStudentTeacherRecurrent(
            **kwargs,
            recon_scan_dim=0,
            recon_scan_offset=int(recon_scan_offset),
            rnn_hidden_dim=rnn_hidden_dim,
        )
    else:
        policy = DepthStudentTeacher(**kwargs)
    nn.Module.load_state_dict(policy, state_dict, strict=False)
    return policy

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
        layers: list[nn.Module] = [nn.Linear(depth_hidden_dim + proprio_obs_dim, student_hidden_dims[0]), activation_cls()]
        for index, width in enumerate(student_hidden_dims):
            if index == len(student_hidden_dims) - 1:
                layers.append(nn.Linear(width, num_actions))
            else:
                layers.extend([nn.Linear(width, student_hidden_dims[index + 1]), activation_cls()])
        self.student = nn.Sequential(*layers)

    def _student_features(self, observations: torch.Tensor) -> torch.Tensor:
        proprio = observations[:, : self.proprio_obs_dim]
        depth = observations[:, self.proprio_obs_dim :]
        depth = depth.reshape(observations.shape[0], *self.depth_shape)
        return torch.cat([proprio, self.depth_encoder(depth)], dim=-1)

    def update_distribution(self, observations):
        mean = self.student(self._student_features(observations))
        std = self.std.expand_as(mean)
        self.distribution = torch.distributions.Normal(mean, std)

    def act_inference(self, observations):
        return self.student(self._student_features(observations))

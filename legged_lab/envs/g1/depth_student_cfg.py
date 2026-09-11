# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""G1 depth-student agent/algorithm configs (B2).

Mirrors the T4 S12 recipe family from ``legged_lab/envs/t4/depth_student_cfg.py``
with G1 (29DoF) observation arithmetic: proprio frame 102, teacher actor 1997.
Launch entry: ``legged_lab/scripts/train_g1_sparse_depth_student.py``.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from legged_lab.envs.g1.depth_student_contract import (
    G1_PROPRIO_FRAME_DIM,
    g1_sparse_teacher_latest_scan_range,
)
from legged_lab.locomotion.schemas import (
    DEPTH_POLICY_SIZE,
    STUDENT_DEPTH_HISTORY_LENGTH,
    STUDENT_PROPRIO_HISTORY_LENGTH,
    TEACHER_SCAN_DIM,
)


@configclass
class G1DepthStudentPolicyCfg:
    """MLP depth encoder + GRU student; teacher MLP matches the G1 sparse actor."""

    class_name: str = "DepthStudentTeacherRecurrent"
    init_noise_std: float = 0.1
    depth_shape: tuple[int, int, int] = (STUDENT_DEPTH_HISTORY_LENGTH, *DEPTH_POLICY_SIZE)
    proprio_obs_dim: int = G1_PROPRIO_FRAME_DIM * STUDENT_PROPRIO_HISTORY_LENGTH
    rnn_type: str = "gru"
    rnn_hidden_dim: int = 256
    rnn_num_layers: int = 1
    teacher_recurrent: bool = False
    recon_scan_dim: int = TEACHER_SCAN_DIM
    recon_scan_offset: int = g1_sparse_teacher_latest_scan_range()[0]
    recon_hidden_dim: int = 128
    critic_hidden_dims: list[int] = [512, 256, 128]
    min_action_std: float = 0.05
    max_action_std: float = 0.2


@configclass
class G1DepthDistillationAlgCfg:
    class_name: str = "Distillation"
    num_learning_epochs: int = 2
    gradient_length: int = 15
    learning_rate: float = 1.0e-3
    loss_type: str = "mse"
    collect_mode: str = "student"
    pg_coef: float = 0.0
    behavior_coef: float = 1.0
    nan_guard: bool = True
    max_grad_norm: float = 1.0


@configclass
class G1SparseDepthStudentDaggerAlgCfg(G1DepthDistillationAlgCfg):
    """Phase A: gated DAgger only; PPO is never enabled in this lineage."""

    class_name: str = "SafeRecurrentDistillation"
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-4
    collect_mode: str = "student"
    behavior_coef: float = 1.0
    behavior_coef_end: float = 1.0
    behavior_coef_decay_iters: int = 0
    pg_coef: float = 0.0
    pg_coef_ramp_iters: int = 0
    pg_delay_iters: int = 0
    critic_warmup_iters: int = 0
    recon_coef: float = 1.0
    max_recon_grad_ratio: float = 1.0
    teacher_mix: float = 1.0
    teacher_mix_end: float = 0.0
    teacher_mix_decay_iters: int = 1000
    gamma: float = 0.99
    lam: float = 0.95
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.0
    use_clipped_value_loss: bool = True
    schedule: str = "fixed"
    desired_kl: float = 0.01
    min_learning_rate: float = 1.0e-5
    max_learning_rate: float = 1.0e-3


@configclass
class G1SparseDepthStudentReprFirstAlgCfg(G1SparseDepthStudentDaggerAlgCfg):
    """One-run representation-first distill: teacher-driven scan, then Ross DAgger. No PPO."""

    learning_rate: float = 1.0e-4
    pg_coef: float = 0.0
    pg_coef_ramp_iters: int = 0
    pg_delay_iters: int = 0
    critic_warmup_iters: int = 0
    behavior_coef: float = 1.0
    behavior_coef_end: float = 1.0
    behavior_coef_decay_iters: int = 0
    recon_coef: float = 1.0
    teacher_mix: float = 1.0
    teacher_mix_end: float = 0.0
    teacher_mix_decay_iters: int = 0
    schedule: str = "fixed"
    entropy_coef: float = 0.0
    repr_first: bool = True
    repr_cap_iters: int = 4000
    repr_probe_interval: int = 100
    repr_probe_patience: int = 3
    repr_min_probe_count: int = 8
    repr_baseline_start_iter: int = 200
    repr_baseline_end_iter: int = 400
    action_mix_decay_iters: int = 2000
    action_level_reset_fraction: float = 0.10
    action_level_reset_min_level: int | None = None


@configclass
class G1SparseDepthStudentAgentCfg:
    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = 24
    max_iterations: int = 14000
    save_interval: int = 500
    experiment_name: str = "g1_loco_sparse_depth_student"
    run_name: str = "g1_repr_first"
    empirical_normalization: bool = False
    logger: str = "tensorboard"
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    policy: G1DepthStudentPolicyCfg = G1DepthStudentPolicyCfg()
    algorithm: G1SparseDepthStudentReprFirstAlgCfg = G1SparseDepthStudentReprFirstAlgCfg()

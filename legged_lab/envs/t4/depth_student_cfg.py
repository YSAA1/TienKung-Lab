"""RSL-RL configuration for the independent T4 depth student lineage."""

from __future__ import annotations

from isaaclab.utils import configclass

from legged_lab.assets.t4.schemas import (
    DEPTH_HISTORY_LENGTH,
    DEPTH_POLICY_SIZE,
    PROPRIO_FRAME_DIM,
    PROPRIO_HISTORY_LENGTH,
    TEACHER_SCAN_DIM,
    sparse_teacher_latest_scan_range,
)


@configclass
class T4DepthStudentPolicyCfg:
    class_name: str = "DepthStudentTeacher"
    init_noise_std: float = 0.1
    depth_shape: tuple[int, int, int] = (DEPTH_HISTORY_LENGTH, *DEPTH_POLICY_SIZE)
    proprio_obs_dim: int = PROPRIO_FRAME_DIM * PROPRIO_HISTORY_LENGTH
    depth_hidden_dim: int = 128
    student_hidden_dims: list[int] = [512, 256, 128]
    teacher_hidden_dims: list[int] = [512, 256, 128]
    activation: str = "elu"


@configclass
class T4DepthDistillationAlgCfg:
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
class T4DepthStudentAgentCfg:
    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = 24
    max_iterations: int = 25000
    save_interval: int = 500
    experiment_name: str = "t4_loco_depth_student"
    run_name: str = "stage_s_head35"
    empirical_normalization: bool = False
    logger: str = "tensorboard"
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    policy: T4DepthStudentPolicyCfg = T4DepthStudentPolicyCfg()
    algorithm: T4DepthDistillationAlgCfg = T4DepthDistillationAlgCfg()


@configclass
class T4DepthStudentFtAlgCfg(T4DepthDistillationAlgCfg):
    """PPO on the existing student; do not clone the plowing teacher."""

    pg_coef: float = 1.0
    behavior_coef: float = 0.0


@configclass
class T4DepthStudentFtAgentCfg(T4DepthStudentAgentCfg):
    experiment_name: str = "t4_loco_depth_student_ft"
    run_name: str = "hurdle030_ft"
    max_iterations: int = 8000
    algorithm: T4DepthStudentFtAlgCfg = T4DepthStudentFtAlgCfg()


@configclass
class T4SparseDepthStudentPolicyCfg(T4DepthStudentPolicyCfg):
    """CNN + GRU student; teacher MLP matches the S12 sparse actor."""

    class_name: str = "DepthStudentTeacherRecurrent"
    rnn_type: str = "gru"
    rnn_hidden_dim: int = 256
    rnn_num_layers: int = 1
    teacher_recurrent: bool = False
    recon_scan_dim: int = TEACHER_SCAN_DIM
    recon_scan_offset: int = sparse_teacher_latest_scan_range()[0]
    recon_hidden_dim: int = 128
    critic_hidden_dims: list[int] = [512, 256, 128]
    min_action_std: float = 0.05
    max_action_std: float = 0.8


@configclass
class T4SparseDepthDistillationAlgCfg(T4DepthDistillationAlgCfg):
    """Safe recurrent DAgger + PPO with critic/GAE and transactional KL guards."""

    class_name: str = "SafeRecurrentDistillation"
    num_mini_batches: int = 4
    learning_rate: float = 3.0e-4
    collect_mode: str = "student"
    behavior_coef: float = 1.0
    behavior_coef_end: float = 0.0
    behavior_coef_decay_iters: int = 2000
    pg_coef: float = 0.5
    recon_coef: float = 1.0
    teacher_mix: float = 0.5
    teacher_mix_end: float = 0.0
    teacher_mix_decay_iters: int = 2000
    gamma: float = 0.99
    lam: float = 0.95
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.0
    desired_kl: float = 0.01
    # Main trust-region gate over the global per-state KL distribution.
    max_kl: float = 0.03
    # A single rare state may exceed the p95 budget, but not catastrophically.
    max_kl_emergency: float = 0.3
    max_behavior_drift: float = 0.02
    rollback_lr_factor: float = 0.5
    min_learning_rate: float = 1.0e-5
    max_learning_rate: float = 1.0e-3


@configclass
class T4SparseDepthStudentAgentCfg(T4DepthStudentAgentCfg):
    experiment_name: str = "t4_loco_sparse_depth_student"
    run_name: str = "s12_gru_safe_recurrent"
    resume: bool = False
    policy: T4SparseDepthStudentPolicyCfg = T4SparseDepthStudentPolicyCfg()
    algorithm: T4SparseDepthDistillationAlgCfg = T4SparseDepthDistillationAlgCfg()


@configclass
class T4SparseDepthStudentFtAlgCfg(T4SparseDepthDistillationAlgCfg):
    """Noise-model short FT: PPO on the distilled GRU student."""

    pg_coef: float = 1.0
    behavior_coef: float = 0.0
    recon_coef: float = 0.25
    teacher_mix: float = 0.0
    teacher_mix_end: float = 0.0
    teacher_mix_decay_iters: int = 0


@configclass
class T4SparseDepthStudentFtAgentCfg(T4SparseDepthStudentAgentCfg):
    experiment_name: str = "t4_loco_sparse_depth_student_ft"
    run_name: str = "s12_gru_depth_noise_ft"
    max_iterations: int = 8000
    algorithm: T4SparseDepthStudentFtAlgCfg = T4SparseDepthStudentFtAlgCfg()

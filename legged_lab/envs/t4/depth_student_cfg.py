"""RSL-RL configuration for the independent T4 depth student lineage."""

from __future__ import annotations

from isaaclab.utils import configclass

from legged_lab.assets.t4.schemas import (
    DEPTH_HISTORY_LENGTH,
    DEPTH_POLICY_SIZE,
    PROPRIO_FRAME_DIM,
    PROPRIO_HISTORY_LENGTH,
    STUDENT_DEPTH_HISTORY_LENGTH,
    STUDENT_PROPRIO_HISTORY_LENGTH,
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
    """MLP depth encoder + GRU student; teacher MLP matches the S12 sparse actor."""

    class_name: str = "DepthStudentTeacherRecurrent"
    depth_shape: tuple[int, int, int] = (STUDENT_DEPTH_HISTORY_LENGTH, *DEPTH_POLICY_SIZE)
    proprio_obs_dim: int = PROPRIO_FRAME_DIM * STUDENT_PROPRIO_HISTORY_LENGTH
    rnn_type: str = "gru"
    rnn_hidden_dim: int = 256
    rnn_num_layers: int = 1
    teacher_recurrent: bool = False
    recon_scan_dim: int = TEACHER_SCAN_DIM
    recon_scan_offset: int = sparse_teacher_latest_scan_range()[0]
    recon_hidden_dim: int = 128
    critic_hidden_dims: list[int] = [512, 256, 128]
    min_action_std: float = 0.05
    max_action_std: float = 0.2


@configclass
class T4SparseDepthStudentDaggerAlgCfg(T4DepthDistillationAlgCfg):
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
class T4SparseDepthStudentAgentCfg(T4DepthStudentAgentCfg):
    experiment_name: str = "t4_loco_sparse_depth_student"
    run_name: str = "s12_rtx_gated_dagger"
    max_iterations: int = 8000
    save_interval: int = 500
    resume: bool = False
    policy: T4SparseDepthStudentPolicyCfg = T4SparseDepthStudentPolicyCfg()
    algorithm: T4SparseDepthStudentDaggerAlgCfg = T4SparseDepthStudentDaggerAlgCfg()


@configclass
class T4SparseDepthStudentJointAlgCfg(T4SparseDepthStudentDaggerAlgCfg):
    """Phase B: student-only DAgger + conservative PPO after critic calibration."""

    learning_rate: float = 3.0e-5
    pg_coef: float = 0.2
    pg_coef_ramp_iters: int = 800
    pg_delay_iters: int = 0
    critic_warmup_iters: int = 200
    behavior_coef: float = 1.0
    behavior_coef_end: float = 1.0
    recon_coef: float = 1.0
    teacher_mix: float = 0.0
    teacher_mix_end: float = 0.0
    teacher_mix_decay_iters: int = 0
    schedule: str = "fixed"


@configclass
class T4SparseDepthStudentDeployFtAlgCfg(T4SparseDepthStudentJointAlgCfg):
    """Phase C: 1000-iteration deployment-noise FT anchored by BC and reconstruction."""

    pg_coef: float = 0.5
    pg_coef_ramp_iters: int = 400
    behavior_coef: float = 0.5
    behavior_coef_end: float = 0.5
    recon_coef: float = 0.25
    entropy_coef: float = 0.0


@configclass
class T4SparseDepthStudentTargetedFtAlgCfg(T4SparseDepthStudentJointAlgCfg):
    """Deployment robustness FT with low PPO pressure and a frozen Phase-B action anchor."""

    learning_rate: float = 1.0e-5
    pg_coef: float = 0.1
    pg_coef_ramp_iters: int = 400
    behavior_coef: float = 1.0
    behavior_coef_end: float = 1.0
    recon_coef: float = 1.0
    reference_action_coef: float = 0.25
    entropy_coef: float = 0.0


@configclass
class T4SparseDepthStudentFtAgentCfg(T4SparseDepthStudentAgentCfg):
    experiment_name: str = "t4_loco_sparse_depth_student_ft"
    run_name: str = "s12_rtx_gated_joint"
    max_iterations: int = 6000
    save_interval: int = 250
    algorithm: T4SparseDepthStudentJointAlgCfg = T4SparseDepthStudentJointAlgCfg()


@configclass
class T4SparseDepthStudentDeployFtAgentCfg(T4SparseDepthStudentAgentCfg):
    experiment_name: str = "t4_loco_sparse_depth_student_ft"
    run_name: str = "s12_rtx_deploy_ft_v2"
    max_iterations: int = 1000
    save_interval: int = 250
    algorithm: T4SparseDepthStudentDeployFtAlgCfg = T4SparseDepthStudentDeployFtAlgCfg()


@configclass
class T4SparseDepthStudentTargetedFtAgentCfg(T4SparseDepthStudentAgentCfg):
    experiment_name: str = "t4_loco_sparse_depth_student_ft"
    run_name: str = "s12_rtx_targeted_robust_ft"
    max_iterations: int = 800
    save_interval: int = 100
    algorithm: T4SparseDepthStudentTargetedFtAlgCfg = T4SparseDepthStudentTargetedFtAlgCfg()

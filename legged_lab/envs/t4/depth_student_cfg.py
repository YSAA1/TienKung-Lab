"""RSL-RL configuration for the independent T4 depth student lineage."""

from __future__ import annotations

from isaaclab.utils import configclass

from legged_lab.assets.t4.schemas import (
    DEPTH_HISTORY_LENGTH,
    DEPTH_POLICY_SIZE,
    PROPRIO_FRAME_DIM,
    PROPRIO_HISTORY_LENGTH,
    TEACHER_ACTOR_OBS_DIM,
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

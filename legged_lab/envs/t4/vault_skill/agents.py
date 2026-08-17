"""Distillation runner for the G2 heightscan vault skill."""

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class StudentTeacherCfg:
    class_name: str = "StudentTeacher"
    init_noise_std: float = 0.1
    student_hidden_dims: list[int] = [512, 256, 128]
    teacher_hidden_dims: list[int] = [512, 256, 128]
    activation: str = "elu"


@configclass
class DistillationAlgCfg:
    class_name: str = "Distillation"
    num_learning_epochs: int = 2
    gradient_length: int = 15
    learning_rate: float = 1.0e-3
    loss_type: str = "mse"
    collect_mode: str = "teacher"
    pg_coef: float = 0.0
    clip_param: float = 0.2
    gamma: float = 0.99
    teacher_mix: float = 0.0
    teacher_mix_end: float = 0.0
    teacher_mix_decay_iters: int = 0


@configclass
class T4VaultSkillDistillRunnerCfg:
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = "t4_vault_skill"
    run_name = ""
    empirical_normalization = True
    logger = "tensorboard"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"
    policy: StudentTeacherCfg = StudentTeacherCfg()
    algorithm: DistillationAlgCfg = DistillationAlgCfg()

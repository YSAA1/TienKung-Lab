"""Layer-0 contracts for the G2 heightscan vault skill task.

These tests stay importable without IsaacLab. They freeze the 1155D student
layout, the 150D G1 teacher query, and the requirement that the student policy
group contains no reference / obstacle privilege.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import torch

from legged_lab.assets.t4 import vault_skill_contract as contract
from legged_lab.assets.t4.schemas import TEACHER_ACTOR_OBS_DIM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rsl_rl"))
SKILL_DIR = ROOT / "legged_lab/envs/t4/vault_skill"
OBS_PATH = SKILL_DIR / "mdp/observations.py"
CFG_PATH = SKILL_DIR / "skill_env_cfg.py"
AGENT_PATH = SKILL_DIR / "agents.py"
INIT_PATH = SKILL_DIR / "__init__.py"
TRAIN_PATH = ROOT / "legged_lab/scripts/train_t4_vault_skill.py"


def _source(path: Path) -> str:
    assert path.is_file(), f"missing {path}"
    return path.read_text()


def test_box_overlay_paints_footprint_and_leaves_ground():
    ground = torch.full((1, 4), 0.2)
    hits = torch.tensor([[[0.38, 0.2], [3.0, 3.0], [0.38, 0.2], [-2.0, 0.2]]])
    sensor_z = torch.tensor([0.9])
    box_center = torch.tensor([[0.38, 0.2, 0.5]])
    painted = contract.overlay_box_on_height_scan(ground, hits, sensor_z, box_center)
    # box top z=1.0 -> 0.9 - 1.0 - 0.5 = -0.6
    assert abs(float(painted[0, 0]) - (-0.6)) < 1e-5
    assert abs(float(painted[0, 2]) - (-0.6)) < 1e-5
    assert abs(float(painted[0, 1]) - 0.2) < 1e-5
    assert abs(float(painted[0, 3]) - 0.2) < 1e-5


def test_analytic_scan_grid_and_box_hit():
    grid = contract.teacher_scan_local_xy()
    assert grid.shape == (contract.G2_SCAN_DIM, 2)
    trunk_xy = torch.zeros(1, 2)
    yaw = torch.zeros(1)
    trunk_z = torch.tensor([0.9])
    box_center = torch.tensor([[0.38, 0.2, 0.5]])
    scan = contract.analytic_vault_height_scan(trunk_xy, yaw, trunk_z, box_center)
    assert scan.shape == (1, contract.G2_SCAN_DIM)
    # A ray over the box footprint is closer (more negative) than flat ground 0.4.
    assert float(scan.min()) < 0.0
    assert float(scan.max()) > 0.0


def test_g2_policy_dim_matches_stage_e_teacher_actor():
    assert contract.G2_POLICY_OBS_DIM == TEACHER_ACTOR_OBS_DIM == 1155
    assert contract.G2_TEACHER_OBS_DIM == 150
    assert contract.G2_ACTION_DIM == 27
    assert contract.G2_ACTION_SCALE == 0.25
    assert (
        contract.G2_PROPRIO_FRAME_DIM * contract.G2_PROPRIO_HISTORY_LENGTH + contract.G2_SCAN_DIM
        == contract.G2_POLICY_OBS_DIM
    )


def test_student_policy_source_has_no_reference_or_box_privilege():
    source = _source(OBS_PATH)
    tree = ast.parse(source)
    assigned = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned.append(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assigned.append(node.target.id)
    leaked = [
        name for name in assigned if any(token in name.lower() for token in contract.G2_POLICY_FORBIDDEN_SUBSTRINGS)
    ]
    assert leaked == [], f"student observation helpers look like privilege/reference: {leaked}"
    assert "g2_actor_obs" in source
    assert "g2_height_scan" in source
    cfg = _source(CFG_PATH)
    assert "g2_actor_obs" in cfg
    assert "class PolicyCfg" in cfg
    assert "class TeacherCfg" in cfg


def test_env_cfg_wires_1155d_student_and_150d_teacher():
    cfg = _source(CFG_PATH)
    init = _source(INIT_PATH)
    agents = _source(AGENT_PATH)
    assert "t4_vault_skill" in init
    assert "T4VaultSkillEnvCfg" in cfg
    assert "teacher" in cfg
    assert "StudentTeacher" in agents
    assert "Distillation" in agents
    assert "G2_ACTION_SCALE" in cfg or "T4_VAULT_ACTION_SCALE" in cfg
    assert "g2_height_scan" in _source(OBS_PATH)
    assert "analytic_vault_height_scan" in _source(OBS_PATH)


def test_train_script_loads_g1_as_teacher_not_as_student():
    source = _source(TRAIN_PATH)
    assert "teacher_checkpoint" in source
    assert "student_checkpoint" in source
    assert "collect_mode" in source
    assert "pg_coef" in source
    assert "teacher_mix" in source
    assert "student_noise" in source
    assert "load_optimizer=False" in source
    assert "t4_vault_skill" in source
    assert "OnPolicyRunner" in source


def test_g2_defaults_to_teacher_drive_without_push_or_walk_clock():
    agents = _source(AGENT_PATH)
    cfg = _source(CFG_PATH)
    obs = _source(OBS_PATH)
    assert 'collect_mode: str = "teacher"' in agents
    assert "push_robot = None" in cfg
    assert "new_zeros((env.num_envs, 6))" in obs
    assert "G2_GAIT_CYCLE" not in obs


def test_distillation_teacher_collect_steps_with_teacher_actions():
    from rsl_rl.algorithms.distillation import Distillation
    from rsl_rl.modules import StudentTeacher

    policy = StudentTeacher(
        num_student_obs=4,
        num_teacher_obs=3,
        num_actions=2,
        student_hidden_dims=[8],
        teacher_hidden_dims=[8],
        init_noise_std=1.0e-6,
    )
    with torch.no_grad():
        for parameter in policy.teacher.parameters():
            parameter.fill_(0.25)
        for parameter in policy.student.parameters():
            parameter.fill_(-0.4)
    algorithm = Distillation(policy, collect_mode="teacher", device="cpu")
    student_obs = torch.ones(5, 4)
    teacher_obs = torch.ones(5, 3)
    stepped = algorithm.act(student_obs, teacher_obs)
    teacher_actions = policy.evaluate(teacher_obs)
    student_actions = policy.act_inference(student_obs)
    assert torch.allclose(stepped, teacher_actions)
    assert torch.allclose(algorithm.transition.privileged_actions, teacher_actions)
    assert not torch.allclose(stepped, student_actions)


def test_distillation_student_collect_steps_with_student_actions():
    from rsl_rl.algorithms.distillation import Distillation
    from rsl_rl.modules import StudentTeacher

    policy = StudentTeacher(
        num_student_obs=4,
        num_teacher_obs=3,
        num_actions=2,
        student_hidden_dims=[8],
        teacher_hidden_dims=[8],
        init_noise_std=1.0e-6,
    )
    with torch.no_grad():
        for parameter in policy.teacher.parameters():
            parameter.fill_(0.25)
        for parameter in policy.student.parameters():
            parameter.fill_(-0.4)
    algorithm = Distillation(policy, collect_mode="student", pg_coef=0.5, device="cpu")
    student_obs = torch.ones(5, 4)
    teacher_obs = torch.ones(5, 3)
    stepped = algorithm.act(student_obs, teacher_obs)
    teacher_actions = policy.evaluate(teacher_obs)
    student_actions = policy.act_inference(student_obs)
    assert torch.allclose(stepped, student_actions, atol=1.0e-4)
    assert torch.allclose(algorithm.transition.privileged_actions, teacher_actions)
    assert not torch.allclose(stepped, teacher_actions)
    algorithm.init_storage("distillation", 5, 1, [4], [3], [2])
    algorithm.process_env_step(torch.ones(5), torch.zeros(5), {})
    loss_dict = algorithm.update()
    assert "behavior" in loss_dict
    assert "pg" in loss_dict


def _tiny_student_teacher():
    from rsl_rl.modules import StudentTeacher

    policy = StudentTeacher(
        num_student_obs=4,
        num_teacher_obs=3,
        num_actions=2,
        student_hidden_dims=[8],
        teacher_hidden_dims=[8],
        init_noise_std=1.0e-6,
    )
    with torch.no_grad():
        for parameter in policy.teacher.parameters():
            parameter.fill_(0.25)
        for parameter in policy.student.parameters():
            parameter.fill_(-0.4)
    return policy


def test_distillation_teacher_mix_one_executes_teacher():
    from rsl_rl.algorithms.distillation import Distillation

    policy = _tiny_student_teacher()
    algorithm = Distillation(policy, collect_mode="student", teacher_mix=1.0, device="cpu")
    stepped = algorithm.act(torch.ones(5, 4), torch.ones(5, 3))
    assert torch.allclose(stepped, policy.evaluate(torch.ones(5, 3)))


def test_distillation_teacher_mix_anneals_to_end_value():
    from rsl_rl.algorithms.distillation import Distillation

    policy = _tiny_student_teacher()
    algorithm = Distillation(
        policy,
        collect_mode="student",
        teacher_mix=0.8,
        teacher_mix_end=0.2,
        teacher_mix_decay_iters=10,
        device="cpu",
    )
    assert abs(algorithm.current_teacher_mix() - 0.8) < 1.0e-6
    algorithm.num_updates = 5
    assert abs(algorithm.current_teacher_mix() - 0.5) < 1.0e-6
    algorithm.num_updates = 10
    assert abs(algorithm.current_teacher_mix() - 0.2) < 1.0e-6
    algorithm.num_updates = 99
    assert abs(algorithm.current_teacher_mix() - 0.2) < 1.0e-6

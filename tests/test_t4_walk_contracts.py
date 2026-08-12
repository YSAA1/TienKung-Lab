# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""Layer-0 contracts for the TienKung-native T4 walk baseline.

These tests stay importable without Isaac Sim. They parse the walk cfg AST so a
swap to Stage E stairs, 20s resampling, or a live height scan cannot hide behind
an unused import of the frozen constant names.
"""

from __future__ import annotations

import ast
from pathlib import Path

from legged_lab.assets.t4 import schemas

ROOT = Path(__file__).resolve().parents[1]
WALK_CFG = ROOT / "legged_lab" / "envs" / "t4" / "walk_cfg.py"
ENV_INIT = ROOT / "legged_lab" / "envs" / "__init__.py"
T4_ENV = ROOT / "legged_lab" / "envs" / "t4" / "t4_env.py"
TEACHER_CFG = ROOT / "legged_lab" / "envs" / "t4" / "teacher_cfg.py"


def _class_def(path: Path, name: str) -> ast.ClassDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"{path} has no class {name}")


def _ann_assign(cls: ast.ClassDef, name: str) -> ast.AST:
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            if node.value is None:
                raise AssertionError(f"{cls.name}.{name} has no value")
            return node.value
    raise AssertionError(f"{cls.name} has no annotated assignment {name}")


def _call_kwarg(call: ast.AST, name: str) -> ast.AST:
    assert isinstance(call, ast.Call), f"expected Call, got {type(call).__name__}"
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    raise AssertionError(f"call is missing keyword {name}")


def _name_id(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    raise AssertionError(f"cannot read identifier from {ast.dump(node)}")


def test_walk_role_is_a_first_class_policy_role():
    assert "walk" in schemas.POLICY_ROLES
    assert schemas.WALK_TASK_NAME == "t4_walk"
    assert schemas.WALK_EXPERIMENT_NAME == "t4_walk"
    assert schemas.WALK_COMMAND_RESAMPLING_S == 10.0
    assert schemas.WALK_AMP_TERRAIN_SCHEDULE_ENABLE is False
    assert schemas.WALK_GAIT_MODE == "fixed_clock"
    assert "Shank_.*" in schemas.WALK_TERMINATE_CONTACTS


def test_walk_actor_has_no_terrain_privilege_width():
    assert schemas.WALK_ACTOR_OBS_DIM == schemas.PROPRIO_FRAME_DIM * schemas.PROPRIO_HISTORY_LENGTH
    assert schemas.WALK_ACTOR_OBS_DIM < schemas.TEACHER_ACTOR_OBS_DIM


def test_walk_cfg_ast_freezes_gravel_ten_second_commands_and_no_scan():
    cls = _class_def(WALK_CFG, "T4WalkEnvCfg")
    policy_role = _ann_assign(cls, "policy_role")
    assert isinstance(policy_role, ast.Constant) and policy_role.value == "walk"

    scene = _ann_assign(cls, "scene")
    assert _name_id(_call_kwarg(scene, "terrain_generator")) == "GRAVEL_TERRAINS_CFG"
    height_scanner = _call_kwarg(scene, "height_scanner")
    enable = _call_kwarg(height_scanner, "enable_height_scan")
    assert isinstance(enable, ast.Constant) and enable.value is False

    commands = _ann_assign(cls, "commands")
    resampling = _call_kwarg(commands, "resampling_time_range")
    assert isinstance(resampling, ast.Tuple) and len(resampling.elts) == 2
    for element in resampling.elts:
        assert _name_id(element) == "WALK_COMMAND_RESAMPLING_S"

    robot = _ann_assign(cls, "robot")
    terminate = _call_kwarg(robot, "terminate_contacts_body_names")
    assert isinstance(terminate, ast.Call) and _name_id(terminate.func) == "list"
    assert _name_id(terminate.args[0]) == "WALK_TERMINATE_CONTACTS"

    source = WALK_CFG.read_text(encoding="utf-8")
    assert "T4_STAGE_E_TERRAINS_CFG" not in source


def test_teacher_cfg_ast_keeps_stage_e_stairs_and_scan():
    """Walk must not be implemented by silently editing the teacher cfg."""
    cls = _class_def(TEACHER_CFG, "T4LocoTeacherEnvCfg")
    scene = _ann_assign(cls, "scene")
    assert _name_id(_call_kwarg(scene, "terrain_generator")) == "T4_STAGE_E_TERRAINS_CFG"
    enable = _call_kwarg(_call_kwarg(scene, "height_scanner"), "enable_height_scan")
    assert isinstance(enable, ast.Constant) and enable.value is True
    commands = _ann_assign(cls, "commands")
    resampling = _call_kwarg(commands, "resampling_time_range")
    assert isinstance(resampling, ast.Tuple)
    values = [elt.value for elt in resampling.elts if isinstance(elt, ast.Constant)]
    assert values == [20.0, 20.0]


def test_walk_task_is_registered_under_its_own_name():
    tree = ast.parse(ENV_INIT.read_text(encoding="utf-8"))
    registered = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "register":
            if node.args and isinstance(node.args[0], ast.Constant):
                registered.append(node.args[0].value)
    assert "t4_walk" in registered
    assert "t4_loco_teacher" in registered


def test_scan_concat_is_inside_the_teacher_branch():
    tree = ast.parse(T4_ENV.read_text(encoding="utf-8"))
    method = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "compute_observations":
            method = node
            break
    assert method is not None

    concat_scan = False
    teacher_guarded_concat = False
    for node in ast.walk(method):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        is_teacher = (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Attribute)
            and test.left.attr == "policy_role"
            and any(isinstance(comp, ast.Constant) and comp.value == "teacher" for comp in test.comparators)
        )
        if not is_teacher:
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute) and child.func.attr == "cat":
                dumped = ast.dump(child)
                if "height_scan" in dumped and "actor_obs" in dumped:
                    teacher_guarded_concat = True
    for node in ast.walk(method):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "cat":
            dumped = ast.dump(node)
            if "height_scan" in dumped and "actor_obs" in dumped:
                concat_scan = True
    assert concat_scan
    assert teacher_guarded_concat

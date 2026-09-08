"""The boundary gate catches the robot inheritance paths that caused this migration."""

import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/audit_robot_boundaries.py"
spec = importlib.util.spec_from_file_location("robot_boundary_audit", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _workspace(tmp_path):
    for name in ("t4", "future_biped"):
        (tmp_path / "legged_lab/envs" / name).mkdir(parents=True)
    (tmp_path / "legged_lab/locomotion").mkdir()
    return tmp_path


@pytest.mark.parametrize("statement", [
    "from legged_lab.envs.t4.teacher_cfg import T4LocoTeacherEnvCfg",
    "from ..t4.teacher_cfg import T4LocoTeacherEnvCfg",
    "from .. import t4",
    "from legged_lab.envs import T4LocoTeacherEnvCfg",
])
def test_rejects_new_robot_importing_previous_robot(tmp_path, statement):
    root = _workspace(tmp_path)
    (root / "legged_lab/envs/future_biped/teacher_cfg.py").write_text(statement)
    errors = module.audit(root)["violations"]
    assert any(error["error"] == "robot recipe imports another robot" for error in errors)


@pytest.mark.parametrize("statement", [
    "from legged_lab.assets import future_biped",
    "from ..assets.future_biped import model",
    "from legged_lab.envs import registry",
])
def test_shared_algorithms_cannot_import_future_robot_or_registration(tmp_path, statement):
    root = _workspace(tmp_path)
    (root / "legged_lab/locomotion/env.py").write_text(statement)
    errors = module.audit(root)["violations"]
    assert any(error["error"] == "shared algorithm depends on robot or task registry" for error in errors)


def test_robot_can_combine_shared_algorithm_with_its_own_asset(tmp_path):
    root = _workspace(tmp_path)
    (root / "legged_lab/envs/future_biped/teacher_cfg.py").write_text(
        "from legged_lab.locomotion.teacher_cfg import LightLPLocomotionEnvCfg\n"
        "from legged_lab.assets.future_biped import model\n"
        "class FutureBipedCfg(LightLPLocomotionEnvCfg): pass\n"
    )
    assert module.audit(root)["violations"] == []

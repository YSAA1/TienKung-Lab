from pathlib import Path


ENV = Path(__file__).resolve().parents[1] / "legged_lab" / "envs" / "t4" / "t4_env.py"
CFG = Path(__file__).resolve().parents[1] / "legged_lab" / "envs" / "t4" / "teacher_cfg.py"
INIT = Path(__file__).resolve().parents[1] / "legged_lab" / "envs" / "__init__.py"


def test_s1d_kitchen_sink_is_removed():
    source = CFG.read_text()
    assert "class T4LocoSparseStableEnvCfg" not in source
    assert "class T4LocoSparseAllInEnvCfg" not in source
    assert "DualValueActorCritic" not in source
    assert "legal_foothold" not in source
    assert "T4SparseSplitCfg" not in source
    assert "T4LocoSparsePaperTeacherEnvCfg" not in source


def test_paper_task_is_unregistered():
    source = INIT.read_text()
    assert "t4_loco_teacher_sparse_paper" not in source
    assert "t4_loco_teacher_sparse" in source


def test_sparse_teacher_is_lightlp_v4():
    source = CFG.read_text()
    assert "t_sparse_lightlp_v4" in source
    assert "soft_sparse_terrain" in source
    assert "append_critic_foot_scan" in source
    assert "opposite_direction" in source
    assert "foot_acceleration_penalty" in source
    assert "self.scene.max_init_terrain_level = 2" in source
    assert "apply_soft_sparse_stage" in source
    assert "soft_fill" in source


def test_soft_scan_uses_true_hole_support_mask():
    source = ENV.read_text()
    assert "_sparse_support_mask_xy" in source
    assert "_apply_algebraic_sparse_scan" in source
    # Must not reintroduce the "near floor z ⇒ hole" platform-destroying heuristic.
    assert "near_floor" not in source


def test_tensorboard_exposes_per_terrain_and_sparse_band_outcomes():
    source = ENV.read_text()
    assert 'logs[f"Terrain/{name}/{metric}"]' in source
    for metric in (
        "success_rate",
        "reach_1m_rate",
        "reach_2m_rate",
        "reach_4m_rate",
        "fall_rate",
        "timeout_rate",
        "pit_fall_rate",
        "progress_m",
    ):
        assert f'"{metric}"' in source
    assert '"Terrain/{name}/episodes"' in source
    assert 'logs[f"Terrain/{name}/{band}_{metric}"]' in source

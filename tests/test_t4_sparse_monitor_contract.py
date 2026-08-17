from pathlib import Path


ENV = Path(__file__).resolve().parents[1] / "legged_lab" / "envs" / "t4" / "t4_env.py"
CFG = Path(__file__).resolve().parents[1] / "legged_lab" / "envs" / "t4" / "teacher_cfg.py"


def test_s1d_kitchen_sink_is_removed():
    source = CFG.read_text()
    assert "class T4LocoSparseStableEnvCfg" not in source
    assert "class T4LocoSparseAllInEnvCfg" not in source
    assert "DualValueActorCritic" not in source
    assert "append_critic_foot_scan" not in source
    assert "legal_foothold" not in source
    assert "T4SparseSplitCfg" not in source


def test_sparse_teacher_starts_in_easy_rows():
    source = CFG.read_text()
    assert "self.scene.max_init_terrain_level = 2" in source


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

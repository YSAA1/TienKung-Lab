from pathlib import Path


ENV = Path(__file__).resolve().parents[1] / "legged_lab" / "locomotion" / "env.py"
CFG = Path(__file__).resolve().parents[1] / "legged_lab" / "locomotion" / "teacher_cfg.py"
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


def test_sparse_teacher_is_one_stage_lightlp():
    source = CFG.read_text()
    assert 'run_name = "t_sparse_lightlp_s12_rim_yaw40"' in (Path(__file__).resolve().parents[1] / "legged_lab/envs/t4/teacher_cfg.py").read_text()
    assert "use_lightlp_terminations" in source
    assert "append_critic_foot_scan" in source
    assert "append_critic_immunity" in source
    assert "opposite_direction" in source
    assert "foot_acceleration_penalty" in source
    assert "self.scene.max_init_terrain_level = 2" in source
    assert "random_level_reset_fraction = 0.10" in source
    assert "self.random_level_reset_max_level = None" in source
    assert "self.random_level_reset_max_level = 4" not in source
    assert "apply_soft_sparse_stage" not in source
    assert "soft_sparse_terrain" not in source
    assert "cfg.soft_fill = False" in source


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
        "promotion_rate",
        "timeout_success_rate",
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
    assert "TerrainCol/" not in source
    assert "assign_curriculum_columns" in source
    assert "random_level_reset_fraction" in source
    assert "random_level_reset_low" in source
    assert "lightlp_terrain_level_moves" in source
    assert 'logs[f"Reset/{name}"]' in source
    assert "promotion_rate" in source
    assert "timeout_success_rate" in source
    assert "random_level_before_mean" in source
    assert "Command/bin_vx_sparse_forward" in source
    assert 'self.extras.pop("log", None)' in source
    assert "LOG_COUNT_SUFFIX" in source or "__n" in source
    assert "mask_recent_push_accel(" in source
    assert "accel_for_gate" in source
    assert "lightlp_sparse_promotion_guard(" in source
    lightlp_reset = source.split("def _check_reset_lightlp", 1)[1].split("def reset(", 1)[0]
    assert "foot_z_all.min(dim=1).values" in lightlp_reset
    assert "foot_z_all.mean(dim=1)" not in lightlp_reset
    assert "cold_start_max_terrain_level" not in source
    assert "self.last_root_accel_mps2.copy_(accel)" in lightlp_reset
    assert "accel_mps2=accel_for_gate" in lightlp_reset
    assert "accel_mps2=accel," not in lightlp_reset
    assert "collapsed_pelvis_above_feet_mask(" in lightlp_reset
    terrain_fn = source.split("def update_terrain_levels", 1)[1]
    before_stage_e_sparse = terrain_fn.split("if not self.use_lightlp_terminations:", 1)[0]
    assert "lightlp_sparse_promotion_guard(" in before_stage_e_sparse


def test_sparse_push_event_is_tagged_stage_e_is_not():
    source = CFG.read_text()
    stage_e = source.split("class LightLPLocomotionEnvCfg", 1)[0]
    sparse = source.split("class LightLPLocomotionEnvCfg", 1)[1]
    assert "func=mdp.push_by_setting_velocity," in stage_e
    assert "push_by_setting_velocity_tagged" not in stage_e
    assert "push_by_setting_velocity_tagged" in sparse
    mdp_init = (Path(__file__).resolve().parents[1] / "legged_lab" / "mdp" / "__init__.py").read_text()
    events = (Path(__file__).resolve().parents[1] / "legged_lab" / "mdp" / "events.py").read_text()
    assert "from .events import *" in mdp_init
    assert "def push_by_setting_velocity_tagged(" in events
    assert "def push_by_setting_velocity(" not in events

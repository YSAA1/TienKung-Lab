from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "legged_lab" / "scripts" / "eval_t4_hurdle.py"


def test_sparse_progress_evaluator_accepts_both_foothold_terrains():
    source = SCRIPT.read_text()

    assert '"stepping_stones"' in source
    assert '"raised_pillars"' in source
    assert "timed_out and progress >= args_cli.progress_m" in source
    assert '"fall_or_early_termination_episodes"' in source
    assert '"pit_fall_episodes"' in source
    assert '"reach_1m_rate"' in source
    assert '"reach_2m_rate"' in source
    assert '"reach_4m_rate"' in source
    assert '"terrain_type": args_cli.terrain_type' in source
    assert "--hard_sparse_pits" in source
    assert "apply_soft_sparse_stage" in source
    assert '"soft_sparse_terrain"' in source

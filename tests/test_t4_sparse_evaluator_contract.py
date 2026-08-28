from pathlib import Path

import pytest
import torch

from legged_lab.scripts.recurrent_policy_eval import evaluate_counterfactual_actions

SCRIPT = Path(__file__).resolve().parents[1] / "legged_lab" / "scripts" / "eval_t4_hurdle.py"
ENV = Path(__file__).resolve().parents[1] / "legged_lab" / "envs" / "t4" / "t4_env.py"
PLAY = Path(__file__).resolve().parents[1] / "legged_lab" / "scripts" / "play.py"


class _StatefulPolicy:
    is_recurrent = True

    def __init__(self):
        self.hidden = torch.zeros(1, 2)

    def get_hidden_states(self):
        return self.hidden, None

    def reset(self, dones=None, hidden_states=None):
        assert dones is None
        self.hidden = hidden_states[0].clone()

    def infer(self, observations):
        self.hidden = self.hidden + observations
        return self.hidden.clone()


def test_counterfactual_actions_advance_recurrent_state_only_for_selected_observation():
    observations = {
        "normal": torch.tensor([[1.0, 2.0]]),
        "zero": torch.tensor([[0.0, 0.0]]),
        "permuted": torch.tensor([[2.0, 1.0]]),
    }
    policy = _StatefulPolicy()

    actions = evaluate_counterfactual_actions(policy, policy.infer, observations, selected_mode="normal")

    assert torch.equal(actions["normal"], torch.tensor([[1.0, 2.0]]))
    assert torch.equal(actions["zero"], torch.tensor([[0.0, 0.0]]))
    assert torch.equal(actions["permuted"], torch.tensor([[2.0, 1.0]]))
    assert torch.equal(policy.hidden, torch.tensor([[1.0, 2.0]]))


def test_sparse_evaluator_resets_recurrent_state_at_episode_boundaries():
    source = SCRIPT.read_text()

    assert "obs, _, dones, extras = env.step(actions)" in source
    assert "runner.alg.policy.reset(dones)" in source


def test_sparse_progress_evaluator_accepts_both_foothold_terrains():
    source = SCRIPT.read_text()

    assert '"stepping_stones"' in source
    assert '"raised_pillars"' in source
    assert "clean_completion and progress >= args_cli.progress_m" in source
    assert "terminal_episode_max_radial_dist" in source
    assert '"progress_source": "episode_max_command_direction_displacement"' in source
    assert '"final_forward_progress_mean_m"' in source
    assert '"fall_or_early_termination_episodes"' in source
    assert '"pit_fall_episodes"' in source
    assert '"reset_reason_counts"' in source
    assert '"reach_1m_rate"' in source
    assert '"reach_2m_rate"' in source
    assert '"reach_4m_rate"' in source
    assert '"terrain_type": args_cli.terrain_type' in source
    assert "--hard_sparse_pits" not in source
    assert "apply_soft_sparse_stage" not in source
    assert "soft_fill = False" in source


def test_sparse_evaluator_keeps_app_env_and_runner_on_the_requested_device():
    source = SCRIPT.read_text()

    assert "env_cfg.device = args_cli.device" in source
    assert "env_cfg.sim.device = args_cli.device" in source
    assert "agent_cfg.device = args_cli.device" in source


def test_sparse_evaluator_can_run_a_stationary_collision_smoke():
    source = SCRIPT.read_text()

    assert 'parser.add_argument("--command_vx", type=float, default=0.7)' in source
    assert 'parser.add_argument("--zero_actions", action="store_true")' in source
    assert "float(abs(args_cli.command_vx) <= 1.0e-9)" in source
    assert "(args_cli.command_vx, args_cli.command_vx)" in source
    assert "terrain_aware_commands = False" in source
    assert "actions = torch.zeros_like(deterministic_actions[args_cli.scan_mode])" in source
    assert '"requested_command_vx_mps": args_cli.command_vx' in source
    assert '"zero_actions": args_cli.zero_actions' in source


def _layout():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "legged_lab" / "terrains" / "stepping_stone_layout.py"
    spec = importlib.util.spec_from_file_location("t4_sparse_spawn_layout", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pinned_sparse_spawn_keeps_eight_cm_on_the_pad_and_rejects_holes():
    layout = _layout()
    from legged_lab.assets.t4.constants import T4_NOMINAL_FEET_Y_DISTANCE

    assert layout.PINNED_SPARSE_SPAWN_FEET_Y_DISTANCE == pytest.approx(T4_NOMINAL_FEET_Y_DISTANCE)
    centered = layout.resolve_pinned_sparse_spawn(0.0, 0.0, terrain_type="stepping_stones")
    assert centered["y_offset_m"] == 0.0
    assert centered["yaw_rad"] == 0.0
    assert centered["pose_range"]["x"] == (0.0, 0.0)
    assert centered["pose_range"]["y"] == (0.0, 0.0)
    assert centered["pose_range"]["yaw"] == (0.0, 0.0)
    assert centered["velocity_range"]["x"] == (0.0, 0.0)
    assert centered["joint_position_range"] == (1.0, 1.0)

    offset = layout.resolve_pinned_sparse_spawn(0.08, 0.0, terrain_type="raised_pillars")
    assert offset["y_offset_m"] == pytest.approx(0.08)
    assert offset["pose_range"]["y"] == (0.08, 0.08)
    assert layout.pinned_spawn_stays_on_platform(0.08, 0.0) is True

    yawed = layout.resolve_pinned_sparse_spawn(0.08, 8.0, terrain_type="stepping_stones")
    assert yawed["yaw_deg"] == pytest.approx(8.0)
    assert yawed["yaw_rad"] == pytest.approx(8.0 * 3.141592653589793 / 180.0)
    assert layout.pinned_spawn_stays_on_platform(0.08, yawed["yaw_rad"]) is True

    assert layout.pinned_spawn_stays_on_platform(0.75, 0.0) is False
    with pytest.raises(ValueError, match="y offset"):
        layout.resolve_pinned_sparse_spawn(0.50, 0.0, terrain_type="stepping_stones")
    with pytest.raises(ValueError, match="yaw"):
        layout.resolve_pinned_sparse_spawn(0.0, 20.0, terrain_type="stepping_stones")
    with pytest.raises(ValueError, match="hurdles"):
        layout.resolve_pinned_sparse_spawn(0.0, 0.0, terrain_type="hurdles")
    assert layout.resolve_pinned_sparse_spawn(None, None, terrain_type="stepping_stones") is None
    unset_yaw = layout.resolve_pinned_sparse_spawn(0.08, None, terrain_type="stepping_stones")
    assert unset_yaw["yaw_deg"] == 0.0
    unset_y = layout.resolve_pinned_sparse_spawn(None, 0.0, terrain_type="raised_pillars")
    assert unset_y["y_offset_m"] == 0.0


def test_sparse_evaluator_pins_spawn_only_when_offset_or_yaw_is_set():
    source = SCRIPT.read_text()

    assert "--spawn_y_offset_m" in source
    assert "--spawn_yaw_deg" in source
    assert "resolve_pinned_sparse_spawn(" in source
    assert 'params["pose_range"] = pinned_spawn["pose_range"]' in source
    assert 'params["velocity_range"] = pinned_spawn["velocity_range"]' in source
    assert '["position_range"] = pinned_spawn["joint_position_range"]' in source
    assert '"spawn_pose_pinned"' in source
    assert '"spawn_y_offset_m"' in source
    assert '"spawn_yaw_deg"' in source
    assert "if pinned_spawn is not None" in source


def test_sparse_evaluator_has_scan_ablation_and_first_step_instrumentation():
    source = SCRIPT.read_text()

    assert 'choices=("normal", "zero", "permuted")' in source
    assert "scan.index_select(-1, scan_permutation)" in source
    assert '"counterfactual_normal_zero_action_l1_mean"' in source
    assert '"counterfactual_normal_permuted_action_l1_mean"' in source
    assert '"first_swing_foot_max_lift_mean_m"' in source
    assert '"first_off_platform_contact_legal_rate"' in source
    assert '"first_off_platform_swing_peak_lift_mean_m"' in source
    assert '"first_off_platform_touchdown_lift_mean_m"' in source
    assert '"diagnostic_joint_action_abs_mean"' in source


def test_sparse_evaluator_balances_episode_sampling_across_envs():
    source = SCRIPT.read_text()

    assert "quota_base, quota_remainder = divmod(args_cli.episodes, env.num_envs)" in source
    assert "active_envs = recorded_episode_counts < episode_quota" in source
    assert "recorded_episode_counts[env_id] += 1" in source
    assert '"episode_sampling": "balanced_per_env_quota"' in source
    assert '"recorded_unique_envs"' in source
    assert '"episode_records"' in source
    assert '"env_episode_index"' in source


def test_first_swing_lift_freezes_at_its_first_touchdown():
    source = SCRIPT.read_text()

    assert "first_swing_complete" in source
    assert "tracking_first_swing = has_first_swing & (~first_swing_complete)" in source
    assert "first_swing_complete |= tracking_first_swing & selected_contact" in source


def test_departure_swing_peak_is_matched_to_the_first_off_platform_touchdown():
    source = SCRIPT.read_text()

    assert "swing_peak_lift" in source
    assert "first_illegal, first_contact_foot = torch.min(candidate_illegal, dim=-1)" in source
    assert "first_contact_swing_peak = swing_peak_lift.gather" in source
    assert '"first_off_platform_swing_peak_lift_m"' in source


def test_terminal_snapshot_is_not_overwritten_by_reset_pose():
    source = ENV.read_text()
    step_source = source.split("    def step(self, actions: torch.Tensor):", 1)[1].split(
        "    def _resample_impact_immunity", 1
    )[0]
    reset_source = source.split("    def reset(self, env_ids):", 1)[1].split("    def update_terrain_levels", 1)[0]

    assert "prev_step_root_pos_w" in source
    assert "terminal_root_pos_w" in source
    assert "terminal_root_quat_w" in source
    assert "terminal_feet_pos_w" in source
    assert "self.terminal_root_pos_w[env_ids] = self.robot.data.root_pos_w[env_ids]" in step_source
    assert "self.terminal_episode_max_radial_dist[env_ids]" in step_source
    assert "self.prev_step_root_pos_w[env_ids] = self.robot.data.root_pos_w[env_ids]" in reset_source
    assert "terminal_root_pos_w[env_ids]" not in reset_source
    assert "last_step_root_pos_w" not in source


def test_play_recording_reads_the_terminal_snapshot_contract():
    source = PLAY.read_text()

    assert 'parser.add_argument("--command_vx", type=float, default=0.6' in source
    assert "(args_cli.command_vx, args_cli.command_vx)" in source
    assert '"--disable_self_collisions"' in source
    assert "env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False" in source
    assert 'getattr(env, "terminal_root_pos_w", None)' in source
    assert 'getattr(env, "terminal_episode_max_radial_dist", None)' in source
    assert "terminal_root_accel_mps2" in source
    assert "terminal_diagnostic_contact_force_n" in source
    assert 'diagnostic_path = os.path.splitext(os.path.abspath(output_path))[0] + ".diagnostics.json"' in source
    assert '"contact_body_names": list(env.diagnostic_contact_body_names)' in source
    assert '"trace": diagnostic_trace' in source


def test_terminal_snapshot_includes_impact_diagnostics():
    source = ENV.read_text()

    assert 'self.diagnostic_contact_body_names = ("Trunk", "Shank_Left", "Shank_Right")' in source
    assert "self.terminal_root_lin_vel_w[env_ids]" in source
    assert "self.terminal_root_accel_mps2[env_ids]" in source
    assert "self.terminal_tilt_rad[env_ids]" in source
    assert "self.terminal_diagnostic_contact_force_n[env_ids]" in source
    assert "last_step_root_pos_w" not in source
    assert "last_step_episode_max_radial_dist" not in source

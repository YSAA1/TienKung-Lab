from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "legged_lab" / "scripts" / "eval_t4_hurdle.py"
ENV = Path(__file__).resolve().parents[1] / "legged_lab" / "envs" / "t4" / "t4_env.py"
PLAY = Path(__file__).resolve().parents[1] / "legged_lab" / "scripts" / "play.py"


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
    assert "actions = torch.zeros_like(deterministic_actions[args_cli.scan_mode])" in source
    assert '"requested_command_vx_mps": args_cli.command_vx' in source
    assert '"zero_actions": args_cli.zero_actions' in source


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

"""Check the environment/algorithm contract before loading AMP datasets."""


def validate_training_contract(env, train_cfg):
    spec = env.cfg.robot_spec
    spec.validate_articulation(env.robot)
    layout = env.observation_layout
    obs, extras = env.get_observations()
    if obs.shape[-1] != layout.actor_dim or extras["observations"]["critic"].shape[-1] != layout.critic_dim:
        raise ValueError(f"{spec.name}: runtime actor/critic dimensions disagree with configured layout")
    if train_cfg["amp_frame_dim"] != spec.amp_frame_dim:
        raise ValueError(f"{spec.name}: AMP config width {train_cfg['amp_frame_dim']} != robot width {spec.amp_frame_dim}")
    if tuple(train_cfg.get("amp_joint_order", ())) != spec.joint_names:
        raise ValueError(f"{spec.name}: AMP expert joint order must equal the policy joint order")
    if env.get_amp_obs_for_expert_trans().shape[-1] != spec.amp_frame_dim:
        raise ValueError(f"{spec.name}: runtime AMP feature width disagrees with robot spec")
    if len(train_cfg["min_normalized_std"]) != len(spec.joint_names):
        raise ValueError(f"{spec.name}: exploration standard deviation count must equal the number of actions")
    schedule = env.cfg.amp_terrain_schedule
    if schedule.enable and (schedule.mode != "linear_decay" or not 0 <= schedule.min_scale <= 1
                            or not 0 <= schedule.decay_start_difficulty < 1):
        raise ValueError("AMP schedule requires linear_decay, min_scale in [0,1], start in [0,1)")

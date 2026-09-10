"""Isaac side of the G1 Isaac-vs-MuJoCo paired rollout probe.

Mirrors `probe_mujoco_paired.py`: flat ground, fixed [vx, 0, 0] command,
nominal reset (no pose/joint/velocity perturbation, no DR, no noise), and the
deterministic actor. Records one 1997D policy obs vector per policy step plus
post-step root/joint/contact states, so the two npz files can be diffed
channel by channel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

from legged_lab.scripts.isaaclab_runtime_compat import (
    patch_missing_physx_material_attributes,
    patch_physx_backward_compatibility_setting,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="g1_loco_teacher")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--vx", type=float, required=True)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--output", required=True)
patch_physx_backward_compatibility_setting(AppLauncher)
AppLauncher.add_app_launcher_args(parser)
args_cli, unknown_cli = parser.parse_known_args()
if unknown_cli:
    raise ValueError(f"unknown CLI arguments: {unknown_cli}")
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from legged_lab.envs import *  # noqa: F401,F403,E402
from legged_lab.utils import task_registry  # noqa: E402
from rsl_rl.runners import (  # noqa: E402,F401 -- selected by runner class name
    AmpOnPolicyRunner,
    OnPolicyRunner,
)

patch_missing_physx_material_attributes()


def main() -> None:
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)
    env_cfg.device = args_cli.device
    env_cfg.sim.device = args_cli.device
    agent_cfg.device = args_cli.device
    env_cfg.scene.num_envs = 1
    env_cfg.scene.terrain_generator.curriculum = False
    env_cfg.scene.terrain_generator.difficulty_range = (0.0, 0.0)
    sub_terrains = env_cfg.scene.terrain_generator.sub_terrains
    if "flat" not in sub_terrains:
        raise ValueError(f"flat terrain is unavailable: {sorted(sub_terrains)}")
    flat_cfg = sub_terrains["flat"]
    flat_cfg.proportion = 1.0
    env_cfg.scene.terrain_generator.sub_terrains = {"flat": flat_cfg}
    env_cfg.scene.terrain_generator.num_rows = 1
    env_cfg.scene.terrain_generator.num_cols = 1

    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.rel_heading_envs = 0.0
    env_cfg.commands.heading_command = False
    env_cfg.commands.resampling_time_range = (20.0, 20.0)
    env_cfg.commands.ranges.lin_vel_x = (args_cli.vx, args_cli.vx)
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.ranges.heading = (0.0, 0.0)
    if hasattr(env_cfg, "terrain_aware_commands"):
        env_cfg.terrain_aware_commands = False
    env_cfg.noise.add_noise = False
    for name in ("physics_material", "add_base_mass", "push_robot"):
        setattr(env_cfg.domain_rand.events, name, None)
    if hasattr(env_cfg.domain_rand.events, "actuator_gains"):
        env_cfg.domain_rand.events.actuator_gains = None
    env_cfg.domain_rand.action_delay.enable = False
    # Nominal reset: exact init_state pose, zero velocities, exact joint targets.
    env_cfg.domain_rand.events.reset_base.params["pose_range"] = {
        key: (0.0, 0.0) for key in ("x", "y", "yaw", "roll", "pitch")
    }
    env_cfg.domain_rand.events.reset_base.params["velocity_range"] = {}
    env_cfg.domain_rand.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    env_cfg.domain_rand.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    env_cfg.scene.seed = agent_cfg.seed

    env_class = task_registry.get_task_class(args_cli.task)
    env = env_class(env_cfg, headless=True)

    root_path = Path(args_cli.checkpoint)
    runner_class = eval(getattr(agent_cfg, "runner_class_name", "OnPolicyRunner"))
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=str(root_path.parent), device=env.device)
    runner.load(str(root_path), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)

    n_joints = env.num_actions
    obs_dim = int(env.actor_obs_buffer.buffer.shape[-1] * env.cfg.robot.actor_obs_history_length)
    obs_dim += int(env.observation_layout.scan_dim * max(1, env.teacher_scan_history_length))
    if getattr(env.cfg.robot, "append_actor_feet_contact", False) or env.observation_layout.actor_contact:
        obs_dim += 2
    obs_log = np.zeros((args_cli.steps, obs_dim), dtype=np.float32)
    action_log = np.zeros((args_cli.steps, n_joints), dtype=np.float32)
    ctrl_log = np.zeros((args_cli.steps, n_joints), dtype=np.float32)
    root_log = np.zeros((args_cli.steps, 13), dtype=np.float32)
    joint_log = np.zeros((args_cli.steps, 2, n_joints), dtype=np.float32)
    contact_log = np.zeros((args_cli.steps, 2), dtype=np.float32)

    for t in range(args_cli.steps):
        obs, _ = env.get_observations()
        if t == 0 and obs.shape[-1] != obs_dim:
            raise RuntimeError(f"obs width {obs.shape[-1]} != expected {obs_dim}")
        obs_log[t] = obs[0].detach().cpu().numpy()
        with torch.no_grad():
            action = policy(obs)
        env.step(action)
        action_log[t] = env.policy_action[0].detach().cpu().numpy()
        ctrl_log[t] = (env.action * env.action_scale + env.robot.data.default_joint_pos)[0].detach().cpu().numpy()
        robot = env.robot.data
        root_log[t] = (
            torch.cat([robot.root_pos_w[0, :3], robot.root_quat_w[0], robot.root_lin_vel_w[0], robot.root_ang_vel_b[0]])
            .detach()
            .cpu()
            .numpy()
        )
        joint_log[t, 0] = robot.joint_pos[0, env.policy_joint_ids].detach().cpu().numpy()
        joint_log[t, 1] = robot.joint_vel[0, env.policy_joint_ids].detach().cpu().numpy()
        net_contact_forces = env.contact_sensor.data.net_forces_w_history
        contact_log[t] = (
            torch.max(torch.norm(net_contact_forces[:, :, env.feet_cfg.body_ids], dim=-1), dim=1)[0] > 0.5
        )[0].float().detach().cpu().numpy()

    meta = {
        "sim": "isaac",
        "checkpoint": str(root_path.resolve()),
        "vx": args_cli.vx,
        "steps": args_cli.steps,
        "physics_dt": float(env_cfg.sim.dt),
        "decimation": int(env_cfg.sim.decimation),
        "action_scale": float(env.action_scale),
        "joint_names": list(env.cfg.robot_spec.joint_names),
        "init_root_z": float(env_cfg.scene.robot.init_state.pos[2]),
        "history_zeroed_at_reset": True,
    }
    output = Path(args_cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        obs=obs_log,
        action=action_log,
        ctrl=ctrl_log,
        root=root_log,
        joint=joint_log,
        contact=contact_log,
    )
    meta_path = output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] wrote {output} and {meta_path}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

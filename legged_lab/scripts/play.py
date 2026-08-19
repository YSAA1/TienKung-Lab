# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

import argparse
import os

import torch
from isaaclab.app import AppLauncher

from legged_lab.scripts.isaaclab_runtime_compat import (
    patch_missing_physx_material_attributes,
    patch_physx_backward_compatibility_setting,
)
from legged_lab.utils import task_registry
from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--terrain",
    action="store_true",
    help="Keep the task terrain generator instead of flattening to a plane.",
)
parser.add_argument(
    "--difficulty",
    type=float,
    default=0.85,
    help="Fixed terrain difficulty in [0, 1] when --terrain is set.",
)
parser.add_argument(
    "--terrain_types",
    type=str,
    default=None,
    help="Comma-separated sub-terrain names to keep when --terrain is set (e.g. 'stairs_up_30,hurdles').",
)
parser.add_argument("--record", type=str, default=None, help="Write an MP4 and exit instead of looping the GUI.")
parser.add_argument("--duration", type=float, default=12.0, help="Recorded seconds when --record is set.")
parser.add_argument(
    "--cam_eye",
    type=str,
    default="-3.4,-2.6,2.4",
    help="Follow-cam eye offset as x,y,z meters (used with --record).",
)
parser.add_argument(
    "--cam_look",
    type=str,
    default="0.8,0.0,0.15",
    help="Follow-cam look offset as x,y,z meters relative to the root (used with --record).",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
patch_physx_backward_compatibility_setting(AppLauncher)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# Start camera rendering
if "sensor" in args_cli.task:
    args_cli.enable_cameras = True
if args_cli.record:
    args_cli.enable_cameras = True
    args_cli.headless = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg

import gymnasium as gym  # noqa: E402
import legged_lab.envs.t4.vault_mimic  # noqa: F401, E402
import legged_lab.envs.t4.vault_skill  # noqa: F401, E402
from legged_lab.envs.t4.vault_mimic.rsl_rl_compat import RslRlVecEnvWrapper  # noqa: E402

patch_missing_physx_material_attributes()


def _gym_play_task_name(task: str) -> str:
    play_name = f"{task}_play"
    if play_name in gym.registry and not task.endswith(("_play", "_eval")):
        return play_name
    return task


def play():
    runner: OnPolicyRunner
    env_cfg: BaseEnvCfg  # noqa:F405

    env_class_name = args_cli.task
    if env_class_name not in task_registry.train_cfgs:
        _play_gym_manager_task(env_class_name)
        return
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.scene.max_episode_length_s = 40.0
    env_cfg.scene.num_envs = 50
    env_cfg.scene.env_spacing = 2.5
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.ranges.lin_vel_x = (0.6, 0.6)
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)

    if args_cli.terrain:
        env_cfg.scene.terrain_generator.curriculum = False
        env_cfg.scene.terrain_generator.difficulty_range = (args_cli.difficulty, args_cli.difficulty)
    else:
        env_cfg.scene.terrain_generator = None
        env_cfg.scene.terrain_type = "plane"

    if env_cfg.scene.terrain_generator is not None:
        # One tile per env so the robot sits in front of the default camera.
        n_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        if n_envs <= 4:
            env_cfg.scene.terrain_generator.num_rows = 1
            env_cfg.scene.terrain_generator.num_cols = max(1, int(n_envs))
        else:
            env_cfg.scene.terrain_generator.num_rows = 5
            env_cfg.scene.terrain_generator.num_cols = 5
        env_cfg.scene.terrain_generator.curriculum = False
        if args_cli.terrain_types:
            keep = [name.strip() for name in args_cli.terrain_types.split(",") if name.strip()]
            sub_terrains = env_cfg.scene.terrain_generator.sub_terrains
            missing = [name for name in keep if name not in sub_terrains]
            if missing:
                raise ValueError(f"unknown sub-terrains {missing}; available: {sorted(sub_terrains)}")
            env_cfg.scene.terrain_generator.sub_terrains = {name: sub_terrains[name] for name in keep}
            for sub_cfg in env_cfg.scene.terrain_generator.sub_terrains.values():
                sub_cfg.proportion = 1.0 / len(keep)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)

    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    runner_class: OnPolicyRunner | AmpOnPolicyRunner = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)

    policy = runner.get_inference_policy(device=env.device)

    if not args_cli.record:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(runner.alg.policy, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(
            runner.alg.policy, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
        )

    if not args_cli.headless:
        from legged_lab.utils.keyboard import Keyboard

        keyboard = Keyboard(env)  # noqa:F841

    obs, _ = env.get_observations()
    if not args_cli.headless:
        root0 = env.robot.data.root_pos_w[0].detach().cpu().tolist()
        print(f"[INFO] robot root at ({root0[0]:.2f}, {root0[1]:.2f}, {root0[2]:.2f}); viewport will follow", flush=True)
        _follow_gui_camera(env.robot.data.root_pos_w[0])

    if args_cli.record:
        _record_play_video(
            env,
            policy,
            obs,
            args_cli.record,
            args_cli.duration,
            cam_eye=_parse_xyz(args_cli.cam_eye),
            cam_look=_parse_xyz(args_cli.cam_look),
        )
        return

    while simulation_app.is_running():

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        if not args_cli.headless:
            _follow_gui_camera(env.robot.data.root_pos_w[0])


def _play_gym_manager_task(task: str) -> None:
    """Play G1/G2 vault tasks registered through gym, not task_registry."""
    gym_task = _gym_play_task_name(task)
    if gym_task not in gym.registry:
        known = sorted(list(task_registry.train_cfgs) + [k for k in gym.registry if k.startswith("t4_")])
        raise KeyError(f"unknown task {task!r}; known: {known}")
    spec = gym.spec(gym_task)
    env_cfg = spec.kwargs["env_cfg_entry_point"]()
    agent_cfg = spec.kwargs["rsl_rl_cfg_entry_point"]()
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    else:
        env_cfg.scene.num_envs = 1
    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else agent_cfg.device
    env = RslRlVecEnvWrapper(gym.make(gym_task, cfg=env_cfg))

    log_root_path = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=os.path.dirname(resume_path), device=env.device)
    runner.load(resume_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    obs, _ = env.get_observations()
    print(f"[INFO] playing gym task {gym_task} from {resume_path}", flush=True)
    if args_cli.record:
        _record_play_video(
            env,
            policy,
            obs,
            args_cli.record,
            args_cli.duration,
            cam_eye=_parse_xyz(args_cli.cam_eye),
            cam_look=_parse_xyz(args_cli.cam_look),
        )
        return
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)


def _write_play_frames(output_path: str, frames: list, fps: int) -> str:
    """Write MP4 via imageio or system ffmpeg; fall back to a subsampled GIF."""
    import shutil
    import subprocess

    import imageio.v2 as imageio
    import numpy as np

    if not frames:
        raise RuntimeError("no camera frames captured")
    height, width = frames[0].shape[:2]
    try:
        imageio.mimwrite(output_path, frames, fps=fps, quality=8, macro_block_size=1)
        return output_path
    except Exception as imageio_err:
        print(f"[WARN] imageio mp4 failed ({imageio_err}); trying ffmpeg", flush=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            assert proc.stdin is not None
            for frame in frames:
                proc.stdin.write(np.ascontiguousarray(frame[..., :3], dtype=np.uint8).tobytes())
            proc.stdin.close()
            stderr = proc.communicate()[1]
            if proc.returncode == 0 and os.path.isfile(output_path) and os.path.getsize(output_path) > 1000:
                return output_path
            print(f"[WARN] ffmpeg mp4 failed rc={proc.returncode}: {stderr[-400:]!r}", flush=True)
        except Exception as ffmpeg_err:
            print(f"[WARN] ffmpeg pipe failed ({ffmpeg_err})", flush=True)
    gif_path = os.path.splitext(output_path)[0] + ".gif"
    stride = max(1, int(round(fps / 12.5)))
    imageio.mimwrite(gif_path, frames[::stride], fps=max(1, fps // stride), loop=0)
    return gif_path


def _parse_xyz(text: str) -> tuple[float, float, float]:
    parts = [float(item.strip()) for item in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"expected x,y,z got {text!r}")
    return parts[0], parts[1], parts[2]


def _follow_gui_camera(root) -> None:
    """Keep the Isaac viewport on the robot. GUI play has no default follow cam."""
    try:
        from isaacsim.core.utils.viewports import set_camera_view
    except ImportError:
        from omni.isaac.core.utils.viewports import set_camera_view

    eye_off = torch.tensor(_parse_xyz(args_cli.cam_eye), device=root.device, dtype=root.dtype)
    look_off = torch.tensor(_parse_xyz(args_cli.cam_look), device=root.device, dtype=root.dtype)
    eye = (root + eye_off).detach().cpu().tolist()
    target = (root + look_off).detach().cpu().tolist()
    set_camera_view(eye=eye, target=target)


def _record_event_line(env, step_idx: int, root) -> str | None:
    """One line per reset of env 0 so a recording can be read without guessing.

    ``env.step`` already teleports the robot back to the pad, so the death pose
    has to come from the pre-reset buffers.
    """
    reset_buf = getattr(env, "reset_buf", None)
    if reset_buf is None or not bool(reset_buf[0].item()):
        return None
    death = getattr(env, "last_step_root_pos_w", None)
    pose = death[0] if death is not None else root
    origin = getattr(getattr(env, "scene", None), "env_origins", None)
    if origin is not None:
        radial = float(torch.norm(pose[:2] - origin[0, :2]).item())
    else:
        radial = float(torch.norm(pose[:2]).item())
    peak = getattr(env, "last_step_episode_max_radial_dist", None)
    if peak is not None:
        radial_peak = float(peak[0].item())
    else:
        radial_peak = radial
    flags = []
    if bool(getattr(env, "time_out_buf", torch.zeros(1))[0].item()):
        flags.append("timeout")
    if bool(getattr(env, "pit_fall_buf", torch.zeros(1))[0].item()):
        flags.append("pit")
    reasons = getattr(env, "reset_reason_masks", None) or {}
    for name, mask in reasons.items():
        if bool(mask[0].item()):
            flags.append(name)
    return (
        f"reset step={step_idx} xy=({float(pose[0]):.2f},{float(pose[1]):.2f}) "
        f"z={float(pose[2]):.2f} r={radial:.2f} peak={radial_peak:.2f} "
        f"flags={','.join(flags) or 'unknown'}"
    )


def _record_play_video(
    env,
    policy,
    obs,
    output_path: str,
    duration_s: float,
    cam_eye: tuple[float, float, float] = (-3.4, -2.6, 2.4),
    cam_look: tuple[float, float, float] = (0.8, 0.0, 0.15),
) -> None:
    import imageio.v2 as imageio
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import Camera, CameraCfg

    camera = Camera(
        CameraCfg(
            prim_path="/World/play_cam",
            update_period=0.0,
            height=480,
            width=960,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.08, 40.0),
            ),
        )
    )
    # Play.py creates the camera after the timeline is already playing, so the
    # SensorBase PLAY callback never fires. Initialize it explicitly.
    camera._initialize_callback(None)
    camera.reset()

    step_dt = getattr(env, "step_dt", None) or env.unwrapped.step_dt
    robot = getattr(env, "robot", None)
    if robot is None:
        robot = env.unwrapped.scene["robot"]
    n_steps = max(1, int(round(duration_s / step_dt)))
    frames: list = []
    look_offset = torch.tensor(cam_look, device=env.device)
    eye_offset = torch.tensor(cam_eye, device=env.device)
    event_path = os.path.splitext(os.path.abspath(output_path))[0] + ".txt"
    events: list[str] = []

    for step_idx in range(n_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        root = robot.data.root_pos_w[0]
        event = _record_event_line(env, step_idx, root)
        if event:
            print(f"[RESET] {event}", flush=True)
            events.append(event)
        camera.set_world_poses_from_view(
            eyes=(root + eye_offset).unsqueeze(0),
            targets=(root + look_offset).unsqueeze(0),
        )
        sim = getattr(env, "sim", None) or env.unwrapped.sim
        sim.render()
        camera.update(dt=step_dt)
        rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy()
        if rgb.size:
            frames.append(rgb.copy())

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(event_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(events) + ("\n" if events else ""))
    fps = max(1, int(round(1.0 / step_dt)))
    written = _write_play_frames(output_path, frames, fps)
    print(f"[INFO] wrote {written} ({len(frames)} frames, {len(events)} resets)")


if __name__ == "__main__":
    play()
    if args_cli.record:
        os._exit(0)
    simulation_app.close()

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

import json

import numpy as np
import torch


class AMPLoader:
    """Schema-driven AMP expert loader.

    The frame width is supplied by the caller, so robots with different joint
    counts share one loader without a per-robot compatibility layer. Every motion
    file must declare its own frame width, frame duration, sampling weight and -
    when the caller pins one - joint order, so a silent schema drift between the
    expert data and the runtime observation is rejected at load time.
    """

    def __init__(
        self,
        device,
        time_between_frames,
        frame_dim,
        motion_files,
        preload_transitions=False,
        num_preload_transitions=1000000,
        expected_joint_order=None,
    ):
        """
        Args:
            device: Torch device that holds the expert trajectories.
            time_between_frames: Seconds between the two frames of an AMP transition.
            frame_dim: Frozen width of a single AMP state.
            motion_files: Expert motion files to load.
            preload_transitions: Whether to sample and cache transitions up front.
            num_preload_transitions: Number of transitions to cache when preloading.
            expected_joint_order: Joint order every motion file must declare, or ``None``.
        """
        if frame_dim is None or int(frame_dim) <= 0:
            raise ValueError(f"AMPLoader requires a positive frame_dim, got {frame_dim!r}")
        if not motion_files:
            raise ValueError("AMPLoader requires at least one motion file")

        self.device = device
        self.time_between_frames = time_between_frames
        self.frame_dim = int(frame_dim)
        self.expected_joint_order = list(expected_joint_order) if expected_joint_order is not None else None

        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Traj length in seconds.
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []

        for i, motion_file in enumerate(motion_files):
            frames, frame_duration, motion_weight = self._load_motion_file(motion_file)
            trajectory = torch.tensor(frames, dtype=torch.float32, device=device)
            self.trajectory_names.append(motion_file)
            self.trajectories.append(trajectory)
            self.trajectories_full.append(trajectory)
            self.trajectory_idxs.append(i)
            self.trajectory_weights.append(motion_weight)
            self.trajectory_frame_durations.append(frame_duration)
            traj_len = (frames.shape[0] - 1) * frame_duration
            self.trajectory_lens.append(traj_len)
            self.trajectory_num_frames.append(float(frames.shape[0]))
            print(f"Loaded {traj_len:.3f}s AMP motion from {motion_file} (weight {motion_weight:.4f}).")

        # Trajectory weights are used to sample some trajectories more than others.
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # Preload transitions.
        self.preload_transitions = preload_transitions
        if self.preload_transitions:
            print(f"Preloading {num_preload_transitions} transitions")
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs)

            self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times)
            self.preloaded_s_next = self.get_full_frame_at_time_batch(traj_idxs, times + self.time_between_frames)
            print("Finished preloading")

        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    def _load_motion_file(self, motion_file):
        """Read one motion file and reject anything that violates the AMP contract."""
        with open(motion_file) as stream:
            motion_json = json.load(stream)

        for key in ("Frames", "FrameDuration", "MotionWeight"):
            if key not in motion_json:
                raise ValueError(f"{motion_file} is missing required AMP key {key!r}")

        if self.expected_joint_order is not None:
            declared_order = motion_json.get("JointOrder")
            if declared_order is None:
                raise ValueError(f"{motion_file} must declare 'JointOrder' when a joint order is pinned")
            if list(declared_order) != self.expected_joint_order:
                raise ValueError(
                    f"{motion_file} declares joint order {list(declared_order)} "
                    f"but the runtime contract expects {self.expected_joint_order}"
                )

        frames = np.asarray(motion_json["Frames"], dtype=np.float64)
        if frames.ndim != 2:
            raise ValueError(f"{motion_file} frames must be 2D, got shape {frames.shape}")
        if frames.shape[1] != self.frame_dim:
            raise ValueError(
                f"{motion_file} frame width {frames.shape[1]} does not match schema width {self.frame_dim}"
            )
        if frames.shape[0] < 2:
            raise ValueError(f"{motion_file} needs at least two frames to form a transition")
        if not np.isfinite(frames).all():
            raise ValueError(f"{motion_file} contains non-finite frame values")

        frame_duration = float(motion_json["FrameDuration"])
        if not np.isfinite(frame_duration) or frame_duration <= 0.0:
            raise ValueError(f"{motion_file} has non-positive FrameDuration {frame_duration}")

        motion_weight = float(motion_json["MotionWeight"])
        if not np.isfinite(motion_weight) or motion_weight <= 0.0:
            raise ValueError(f"{motion_file} has non-positive MotionWeight {motion_weight}")

        return frames, frame_duration, motion_weight

    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """Batch sample traj idxs."""
        return np.random.choice(self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True)

    def traj_time_sample(self, traj_idx):
        """Sample random time for traj."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx]
        return max(0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """Sample random time for multiple trajectories."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]

        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst

        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, frame1, frame2, blend):
        return (1.0 - blend) * frame1 + blend * frame2

    def get_trajectory(self, traj_idx):
        """Returns trajectory of AMP observations."""
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low

        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times):
        """Returns frame for the given trajectory at the specified time."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int64), np.ceil(p * n).astype(np.int64)
        all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        """Returns full frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.slerp(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int64), np.ceil(p * n).astype(np.int64)
        all_frame_amp_starts = torch.zeros(len(traj_idxs), self.frame_dim, device=self.device)
        all_frame_amp_ends = torch.zeros(len(traj_idxs), self.frame_dim, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_amp_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_amp_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)

        return self.slerp(all_frame_amp_starts, all_frame_amp_ends, blend)

    def get_frame(self):
        """Returns random frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        """Returns random full frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames):
        if self.preload_transitions:
            idxs = np.random.choice(self.preloaded_s.shape[0], size=num_frames)
            return self.preloaded_s[idxs]
        else:
            traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
            times = self.traj_time_sample_batch(traj_idxs)
            return self.get_full_frame_at_time_batch(traj_idxs, times)

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """Generates a batch of AMP transitions."""
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                idxs = np.random.choice(self.preloaded_s.shape[0], size=mini_batch_size)
                s = self.preloaded_s[idxs]
                s_next = self.preloaded_s_next[idxs]
            else:
                s, s_next = [], []
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                for traj_idx, frame_time in zip(traj_idxs, times):
                    s.append(self.get_frame_at_time(traj_idx, frame_time))
                    s_next.append(self.get_frame_at_time(traj_idx, frame_time + self.time_between_frames))

                s = torch.vstack(s)
                s_next = torch.vstack(s_next)
            yield s, s_next

    @property
    def observation_dim(self):
        """Size of AMP observations."""
        return self.frame_dim

    @property
    def num_motions(self):
        return len(self.trajectory_names)

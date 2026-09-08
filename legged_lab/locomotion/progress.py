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

# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""Read-only displacement diagnostics; never used by rewards or observations."""

import torch


class ProgressMonitor:
    """Track actual spawn displacement and a reset-safe 0.4-second velocity window."""

    def __init__(self, num_envs, device, dt, window_s=0.4):
        self.dt = dt
        self.window = max(1, round(window_s / dt))
        self.index = 0
        self.spawn = torch.zeros(num_envs, 2, device=device)
        self.previous = self.spawn.clone()
        self.delta_history = torch.zeros(self.window, num_envs, 2, device=device)
        self.steps = torch.zeros(num_envs, device=device)
        self.path = self.steps.clone()
        self.peak = self.steps.clone()
        self.directed = self.steps.clone()
        self.requested = self.steps.clone()
        self.moving_steps = self.steps.clone()
        self.window_speed_sum = self.steps.clone()

    def reset(self, ids, xy):
        self.spawn[ids] = xy
        self.previous[ids] = xy
        self.delta_history[:, ids] = 0
        for value in (
            self.steps,
            self.path,
            self.peak,
            self.directed,
            self.requested,
            self.moving_steps,
            self.window_speed_sum,
        ):
            value[ids] = 0

    def update(self, xy, command_xy_world):
        delta = xy - self.previous
        self.previous.copy_(xy)
        self.steps += 1
        self.path += delta.norm(dim=-1)
        self.peak = torch.maximum(self.peak, (xy - self.spawn).norm(dim=-1))
        speed = command_xy_world.norm(dim=-1)
        moving = speed > 0.1
        self.directed += (delta * command_xy_world).sum(-1) / speed.clamp(min=1e-6) * moving
        self.requested += speed * self.dt * moving
        self.moving_steps += moving
        self.delta_history[self.index] = delta
        self.index = (self.index + 1) % self.window
        window_velocity = self.delta_history.sum(0) / (self.steps.clamp(max=self.window) * self.dt)[:, None]
        self.window_speed_sum += window_velocity.norm(dim=-1)

    def metrics(self, ids):
        net = (self.previous[ids] - self.spawn[ids]).norm(dim=-1)
        return {
            "net_displacement_m": net,
            "max_spawn_displacement_m": self.peak[ids],
            "path_length_m": self.path[ids],
            "net_path_ratio": net / self.path[ids].clamp(min=1e-6),
            "command_progress_m": self.directed[ids],
            "command_completion": self.directed[ids] / self.requested[ids].clamp(min=1e-6),
            "command_speed_mps": self.requested[ids] / (self.moving_steps[ids] * self.dt).clamp(min=self.dt),
            "directed_speed_mps": self.directed[ids] / (self.moving_steps[ids] * self.dt).clamp(min=self.dt),
            "window_net_speed_mps": self.window_speed_sum[ids] / self.steps[ids].clamp(min=1),
        }

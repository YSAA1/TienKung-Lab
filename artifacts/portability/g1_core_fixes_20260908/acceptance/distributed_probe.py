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

"""Run the production trainer for two updates, with rank-asymmetric reset injection."""

import hashlib
import json
import os
import threading
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

import legged_lab.scripts.train as training

OUTPUT = Path("artifacts/portability/g1_core_fixes_20260908/acceptance")
original_learn = training.AmpOnPolicyRunner.learn


def vector(module):
    return torch.cat([parameter.detach().flatten() for parameter in module.parameters()])


def max_difference(value):
    gathered = [torch.empty_like(value) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, value)
    return max(float((other - gathered[0]).abs().max()) for other in gathered)


def checked_learn(self, num_learning_iterations, init_at_random_ep_len=False):
    assert num_learning_iterations == 2 and self.num_steps_per_env == 24
    assert self.alg.amp_data.num_motions == 17
    rank = dist.get_rank()
    result = {
        "ok": False,
        "rank": rank,
        "world_size": dist.get_world_size(),
        "steps_per_update": self.num_steps_per_env,
        "num_envs": self.env.num_envs,
        "expert_count": self.alg.amp_data.num_motions,
        "steps": [],
    }
    output = OUTPUT / f"distributed_rank{rank}.json"
    original_step = self.env.step
    calls = 0

    def step(actions):
        nonlocal calls
        # Buffer injections only; thresholds, physics and reset events stay unchanged.
        if calls == 0:
            self.env.episode_length_buf.zero_()
        if rank == 0 and calls in (0, 24):
            self.env.episode_length_buf[:2] = self.env.max_episode_length - 1
        elif rank == 1 and calls == 25:
            self.env.episode_length_buf[2:5] = self.env.max_episode_length - 1
        obs, reward, done, info = original_step(actions)
        result["steps"].append(
            {
                "step": calls,
                "done": int(done.sum()),
                "timeouts": int(info["time_outs"].sum()),
                "bootstrap": int(info["bootstrap_mask"].sum()),
            }
        )
        calls += 1
        return obs, reward, done, info

    self.env.step = step
    # Exercise the actual optimizer projection from a below-floor initial state.
    with torch.no_grad():
        self.alg.policy.std.fill_(0.01)
    self.alg.enforce_min_std()
    try:
        original_learn(self, 2, init_at_random_ep_len=False)
        assert calls == 48
        result["differences"] = {
            "policy": max_difference(vector(self.alg.policy)),
            "discriminator": max_difference(vector(self.alg.discriminator)),
            "amp_normalizer": max_difference(
                torch.as_tensor(
                    np.concatenate(
                        (self.alg.amp_normalizer.mean, self.alg.amp_normalizer.var, [self.alg.amp_normalizer.count])
                    ),
                    device=self.device,
                )
            ),
        }
        assert all(value == 0 for value in result["differences"].values()), result["differences"]
        result["minimum_std"] = float(self.alg.policy.std.min())
        assert result["minimum_std"] >= 0.05 - 1e-7
        checkpoint = OUTPUT / "distributed_reload.pt"
        if rank == 0:
            self.save(str(checkpoint), infos={"acceptance": "core_fixes"})
        dist.barrier()
        state = torch.load(checkpoint, weights_only=False, map_location=self.device)
        assert state["transition_contract"]["version"] == 1
        before = vector(self.alg.policy).clone()
        loaded_info = self.load(str(checkpoint))
        torch.testing.assert_close(vector(self.alg.policy), before, atol=0, rtol=0)
        assert loaded_info == {"acceptance": "core_fixes"}
        result["checkpoint"] = {
            "path": str(checkpoint),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "transition_contract": state["transition_contract"],
            "reload_equal": True,
        }
        result["ok"] = True
    finally:
        output.write_text(json.dumps(result, indent=2) + "\n")


training.AmpOnPolicyRunner.learn = checked_learn
code = 0
try:
    training.train()
    dist.barrier()
    dist.destroy_process_group()
except BaseException:
    traceback.print_exc()
    code = 1
finally:
    threading.Timer(30, os._exit, args=(code,)).start()
    training.simulation_app.close()
    os._exit(code)

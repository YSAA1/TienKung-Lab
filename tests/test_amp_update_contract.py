"""AMP statistics and gradient regularization must use consistent coordinates."""

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rsl_rl"))
from rsl_rl.algorithms.amp_ppo import AMPPPO
from rsl_rl.modules import ActorCritic, Discriminator
from rsl_rl.utils import Normalizer


def _record_update():
    torch.manual_seed(42)
    raw = torch.tensor([[10.0, -20.0, 30.0], [12.0, -22.0, 34.0]]).repeat(4, 1)
    next_raw = raw + 0.25
    norm = Normalizer(3)
    norm.mean[:] = [2.0, -3.0, 4.0]
    norm.var[:] = [4.0, 9.0, 16.0]
    expected = (norm.normalize_torch(raw, "cpu"), norm.normalize_torch(next_raw, "cpu"))
    updates, gradient_inputs = [], []
    update = norm.update

    def record_update(arr):
        updates.append(arr.copy())
        update(arr)

    norm.update = record_update
    discriminator = Discriminator(6, 0.3, [8], "cpu", 0.7)
    grad_pen = discriminator.compute_grad_pen

    def record_grad_pen(*args, **kwargs):
        gradient_inputs.append([x.clone() for x in args])
        return grad_pen(*args, **kwargs)

    discriminator.compute_grad_pen = record_grad_pen
    policy = ActorCritic(5, 5, 2, actor_hidden_dims=[8], critic_hidden_dims=[8])
    loader = SimpleNamespace(feed_forward_generator=lambda *args: iter([(raw, next_raw)]))
    alg = AMPPPO(policy, discriminator, loader, norm, amp_replay_buffer_size=16, device="cpu")
    alg.init_storage("rl", 4, 2, [5], [5], [2])
    for _ in range(2):
        obs = torch.randn(4, 5)
        alg.act(obs, obs, raw[:4])
        alg.process_env_step(torch.ones(4), torch.zeros(4, dtype=torch.bool), {}, next_raw[:4])
    alg.compute_returns(torch.randn(4, 5))
    alg.amp_storage.feed_forward_generator = lambda *args: iter([(raw, next_raw)])
    alg.update()
    return raw, expected, updates, gradient_inputs


def test_running_statistics_receive_raw_amp_samples():
    raw, _, updates, _ = _record_update()
    assert len(updates) == 2
    for samples in updates:
        np.testing.assert_allclose(samples, raw.numpy())


def test_gradient_penalty_uses_same_coordinates_as_discriminator_loss():
    _, expected, _, gradient_inputs = _record_update()
    for actual, reference in zip(gradient_inputs[0], expected):
        torch.testing.assert_close(actual, reference)

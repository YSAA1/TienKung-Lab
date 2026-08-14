"""Vendored rsl-rl must ignore Isaac Lab 2.3 PPO keys such as optimizer."""

from __future__ import annotations

import torch

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.utils.utils import filter_init_kwargs


def test_filter_drops_isaaclab23_optimizer():
    cfg = {
        "learning_rate": 1e-3,
        "clip_param": 0.2,
        "optimizer": "adam",
        "share_cnn": False,
    }
    filtered = filter_init_kwargs(PPO.__init__, cfg)
    assert "optimizer" not in filtered
    assert "share_cnn" not in filtered
    assert filtered["learning_rate"] == 1e-3


def test_ppo_constructs_after_filtering_isaaclab23_dict():
    policy = ActorCritic(150, 276, 27, actor_hidden_dims=[32], critic_hidden_dims=[32])
    alg_cfg = {
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "entropy_coef": 0.005,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 1e-3,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
        "normalize_advantage_per_mini_batch": False,
        "symmetry_cfg": None,
        "rnd_cfg": None,
        "optimizer": "adam",
    }
    ppo = PPO(policy, device="cpu", **filter_init_kwargs(PPO.__init__, alg_cfg), multi_gpu_cfg=None)
    assert ppo.device == "cpu"
    dummy = torch.zeros(1, 150)
    _ = ppo.policy.act_inference(dummy)

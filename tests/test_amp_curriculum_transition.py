"""AMP weights belong to the transition before an automatic episode reset."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _env_methods():
    tree = ast.parse((ROOT / "legged_lab/locomotion/env.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LocomotionEnv")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in
               ("amp_reward_coef_scale", "_decay_scale")]
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[ast.ClassDef(name="Env", bases=[], keywords=[], body=methods,
                                              decorator_list=[], lineno=1, col_offset=0)],
                            type_ignores=[]), "env_amp", "exec"), scope)
    return scope["Env"]


def test_original_amp_schedule_and_sparse_zeroing_across_ten_levels():
    env = _env_methods()()
    env.num_envs, env.device = 20, "cpu"
    tree = ast.parse((ROOT / "legged_lab/locomotion/teacher_cfg.py").read_text(encoding="utf-8"))
    schedule_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AmpTerrainScheduleCfg")
    schedule_cls.decorator_list = []
    scope = {}
    exec(compile(ast.Module(body=[schedule_cls], type_ignores=[]), "schedule", "exec"), scope)
    env.cfg = SimpleNamespace(amp_terrain_schedule=scope["AmpTerrainScheduleCfg"]())
    env.terrain_difficulty = lambda: torch.arange(10).repeat(2).float() / 9
    env.sparse_foothold_type_ids = [1]
    env.sparse_tile_mask = torch.tensor([False] * 10 + [True] * 10)
    env.refresh_sparse_tile_mask = lambda: None
    scales = env.amp_reward_coef_scale().reshape(2, 10)
    # Frozen original rule: full weight through difficulty .3, floor .3 at d=1.
    expected = [1.0 - max(0.0, (level / 9 - 0.3) / 0.7) * 0.7 for level in range(10)]
    assert scales[0].tolist() == pytest.approx(expected)
    assert scales[1].tolist() == pytest.approx([0.0] * 10)


def test_runner_captures_amp_weight_before_auto_reset():
    path = ROOT / "rsl_rl/rsl_rl/runners/amp_on_policy_runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For)
                and ast.unparse(n.iter) == "range(self.num_steps_per_env)")
    # Execute the actual collection call sites with a resetting fake environment.
    stop = next(i for i, n in enumerate(loop.body)
                if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "rewards")
    nodes = loop.body[:stop]
    weights = torch.tensor([1.0, 0.3])
    env = SimpleNamespace(device="cpu", reset_env_ids=torch.tensor([], dtype=torch.long))
    def step(actions):
        weights.copy_(torch.tensor([0.3, 1.0]))
        return torch.zeros(2, 1), torch.zeros(2), torch.zeros(2), {}
    env.step = step
    env.get_amp_obs_for_expert_trans = lambda: torch.zeros(2, 1)
    runner = SimpleNamespace(env=env, device="cpu", privileged_obs_type=None,
                             obs_normalizer=lambda x: x,
                             amp_reward_coef_scale_fn=lambda: weights,
                             alg=SimpleNamespace(act=lambda *x: torch.zeros(2, 1)))
    scope = dict(self=runner, torch=torch, obs=None, privileged_obs=None, amp_obs=None)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    assert scope["coef_scale"].tolist() == pytest.approx([1.0, 0.3])

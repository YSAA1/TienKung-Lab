# Z2 action-rate candidate (local only)

Single-variable trial in this worktree. Not applied to frozen baseline `933e08a`. Not a claimed behavior fix. Parent owns whether to freeze a new remote checkout/lineage after baseline 3k JSON/video.

## Changed field

File: `legged_lab/envs/z2/teacher_cfg.py`
Location: `Z2LocoTeacherEnvCfg.__post_init__`, immediately after `self.reward = Z2SparseTeacherRewardCfg()`.

| | |
| --- | --- |
| Field | `self.reward.action_rate_l2.weight` |
| Before | `-0.1` (shared `LightLPRewardCfg` default; Z2 recipe did not override) |
| After | `-0.01` (explicit Z2-only assignment) |

Comment in source: single-variable action-rate trial from `-0.1`; not a claimed fix.

Nothing else in this file was changed (asset/PD/action scale `0.25`/experts/AMP/termination/gait gate/terrain/commands/seeds/PPO). G1 and shared `LightLPRewardCfg` defaults were not edited. No new CLI flag, subclass, task, or abstraction.

## Checks

```
set PYTHONPATH=D:\TienKung-Lab-z2-teacher-20260909;D:\TienKung-Lab-z2-teacher-20260909\rsl_rl
D:\anaconda\envs\pytorch\python.exe -m pytest tests/test_z2_asset_contract.py --basetemp artifacts/z2_migration/pytest-tmp
```

**31 passed in 4.24s.**

Isolated pre-commit on `legged_lab/envs/z2/teacher_cfg.py` only (`PRE_COMMIT_HOME=work/z2-migration/precommit-cache`, `uv tool run --from pre-commit==3.7.1`): **pass** (black, flake8, isort, license, codespell).

No new tests, no full suite, no Isaac/GPU, no git commit, no SSH/deploy.

## Not validated

This does not prove the stationary 2000-iter flat result (mean max-forward 0.116 m, 0 OOB / 0 reach2m on frozen `933e08a`) was caused by action-rate cost. Behavior, new lineage, and remote freeze remain parent-owned.

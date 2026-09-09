# Z2 G1 VITAL recipe (local candidate)

User cancelled single-variable trials. Formal Z2 training is to use all three G1 VITAL settings. Local only; parent deploys/trains. Frozen baseline `933e08a` is unchanged here.

## Change

`legged_lab/envs/z2/teacher_cfg.py` `Z2LocoTeacherEnvCfg.__post_init__`, after `self.reward = Z2SparseTeacherRewardCfg()`:

| Field | Before (shared LightLP default) | After |
| --- | --- | --- |
| `reward.action_rate_l2.weight` | `-0.1` | `-0.01` |
| `acceleration_termination_enabled` | `True` | `False` |
| `gait.tracking_gate_enabled` | `True` | `False` |

Comment: G1 VITAL recipe (action-rate, accel termination, gait tracking gate). No other VITAL extras (`deterministic_fall_limits`, etc.) were copied. Assets/PD/experts/shared defaults, flags, and tasks were not changed.

## Validation

```
PYTHONPATH=<repo>;<repo>/rsl_rl
D:\anaconda\envs\pytorch\python.exe -m pytest tests/test_z2_asset_contract.py --basetemp artifacts/z2_migration/pytest-tmp
```

**31 passed in 5.69s.**

Isolated pre-commit on `legged_lab/envs/z2/teacher_cfg.py` only: **pass**.

No Isaac/GPU, no SSH, no git commit, no status-doc edits.

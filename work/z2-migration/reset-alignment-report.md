# Z2 reset/contact alignment

Authorized correction after diagnosis: Z2 was missing G1 VITAL fall limits, collapse reset, and `undesired_contacts=-1`. Local `teacher_cfg.py` only. Parent deploys/restarts. Frozen `933e08a` untouched.

## Delta (`legged_lab/envs/z2/teacher_cfg.py`)

| Field | Before | After |
| --- | --- | --- |
| `reward.undesired_contacts.weight` | `-2.0` (LightLP default; Z2 did not override after role bind) | `-1.0` after `bind_reward_roles` |
| `deterministic_fall_limits` | unset / shared default | `(0.8, 1.0)` |
| `collapse_reset_pelvis_above_feet_m` | unset (comment said not configured) | `0.20` |
| `collapse_reset_grace_s` | unset | `0.20` |
| `collapse_reset_respects_impact_immunity` | unset | `True` |

Kept: `action_rate_l2.weight=-0.01`, `acceleration_termination_enabled=False`, `gait.tracking_gate_enabled=False`. Assets/PD/AMP/task params unchanged. Obsolete “threshold not configured” comment replaced: 0.20 m is a collapse floor below standing pelvis-foot clearance (~0.64 m); brief low pose is recovery; impact immunity kept.

Existing `tests/test_z2_asset_contract.py` had asserted `collapse_reset_pelvis_above_feet_m` **absent**. That one guard was updated to the authorized assignments so the requested contract test can pass. No new test file.

## Checks

```
PYTHONPATH=<repo>;<repo>/rsl_rl
D:\anaconda\envs\pytorch\python.exe -m pytest tests/test_z2_asset_contract.py tests/test_collapse_recovery_contract.py tests/test_g1_collapse_reset_numerics.py --basetemp artifacts/z2_migration/pytest-tmp
```

**42 passed, 1 skipped in 2.57s.**

Isolated pre-commit on `legged_lab/envs/z2/teacher_cfg.py`: **pass**.

No Isaac/GPU, no SSH, no git commit, no status docs.

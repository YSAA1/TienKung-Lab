# Findings log

## 2026-08-22 S12 MDP 审查验证

### Finding #1 push × accel 门 — CONFIRMED（语义需更正）

- nubot 只读：`~/IsaacLab` = `v2.1.0-110-g3d5ea25ddb`。`push_by_setting_velocity` 是 **additive**（`vel_w += U(range)`），不是把 x/y 覆写成 U(-1,1)。缺的轴用 (0,0) 即加零，不是置零。
- `write_root_com_velocity_to_sim` **立刻**写 `_data.root_com_vel_w`；`root_lin_vel_w` 是其切片。假设「下次 scene.update 才刷新」被证伪。
- `t4_env.step`：物理循环 → `event_manager.apply(interval)` → `check_reset`。尖峰落在**同一控制步**的 `_check_reset_lightlp`（`||Δv||/0.02`）。
- 数值：行走 (0.7,0,0) 上加 U(-1,1)²，约 49% 样本 accel>40；SET 覆写约 63%。90% 非免疫 env 会被推搡本身 reset。
- 修复（只动 sparse）：`push_by_setting_velocity_tagged` 打 `_push_step_marker`；`mask_recent_push_accel` 在 `0 <= diff <= decimation` 把门用 accel 置 0；诊断缓冲仍记原始 accel。Stage E 仍用原版 push。

### Finding #2 掉坑仍晋级 — CONFIRMED

- LightLP 分支跳过 `sparse_curriculum_moves`；S12 `terminate_on_pit_fall=False`。`pit_fall_buf` 在 `check_reset` 写入，`reset` → `update_terrain_levels` 前已刷新。
- 掉坑后若还能走满 path>4m 且 tracking≥0.5，会 `move_up` 并被 `monitor_outcome_flags` 记 timeout_success。
- 修复：`lightlp_sparse_promotion_guard` 去掉 sparse∧pit_fall 的晋级；不加 demotion。

### 测试

`python -m pytest tests/test_t4_sparse_reward_contracts.py tests/test_t4_sparse_monitor_contract.py tests/test_t4_sparse_command_contract.py tests/test_t4_terrain_curriculum.py tests/test_t4_sparse_evaluator_contract.py tests/test_t4_stepping_stone_contracts.py -q` → **80 passed**。

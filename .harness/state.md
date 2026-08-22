# Current State

- Living index: `docs/README.md`
- Context: `PROJECT_CONTEXT.md`
- Dual track（各一份计划，不是双真相）:
  - 梅花桩：`docs/plans/2026-08-22--t4-sparse-s12-rim-yaw-student-plan.md`
  - 翻箱：`docs/plans/2026-08-15--t4-vault-g1-g2-recovery-plan.md`
- Approved specs: `docs/specs/2026-08-12--t4-unified-depth-locomotion.md`（走跑/学生合同）、`docs/specs/2026-08-13--t4-vault-loco-merge.md`（G1→G2→G3 目标）、`docs/specs/2026-08-15--t4-stepping-stones-and-hurdle-stable.md`（梅花桩目标；执行已改单阶段）

## 梅花桩（nubot）

- 任务 `t4_loco_teacher_sparse`。S12 老师 worktree `/home/nubot/phn_ws/t4_train/TienKung-Lab-s12-from-s11b-5k`，run `t_sparse_lightlp_s12_from_s11b_5k`，从 S11b `model_19000` 加载后冻在 `model_21500.pt`。
- 配方：0.75 m 收尾边框 + 踏石/圆桩 40% 轻转；`vx` 非洞 `[-0.6, 2.0]`，洞上 `[0.6, 2.0]`。终止阈值不改。`random_level_reset_max_level=None` 是 S11 起的老师课表，学生继承，不是 v3 单变量差。
- 学生 `fixed-v3-nanguard` 已在 iteration 3305 左右策略坍塌，冻结为失败取证 lineage；不得续 `model_3500/4000`。最近 SSH 探针超时，tmux/进程是否仍存活未刷新。
- 本地后继实现是 `SafeRecurrentDistillation`；新 lineage 尚未同步/启动。建议冻结教师 `model_21500` + 显式 warm-start 崩塌前 `model_3000` student stack，重建 critic/Adam/counters；若拿不到 3000 才冷启，不用 3500/4000。
- S11b 已停（`model_19000.pt`）。S10/S9/S6–S8 只读对照。S7/S8 是污染 plant，不续训。
- Stage E `t4_loco_teacher` 1155D 未改。能力声明要等 evaluator + 回放；TB success / length→1000 不算。

## 翻箱（zhuoqun）

- 执行面仍是 G1/G2 recovery。G3 / 与走跑合并被 G2 过箱挡住。
- 不要动 nubot 梅花桩 checkout 去盖 zhuoqun 翻箱。

## 已关闭（不要当待办）

- Stage E 走跑 + 跨栏 bucket + 深度学生 `model_24999` / 部署候选 `model_25746`。
- 走跑完成 ≠ 100m `rule`、真机障碍、梅花桩、翻箱 G3。

## 验证

- 本机：`python -m pytest tests/test_t4_sparse_reward_contracts.py tests/test_t4_sparse_monitor_contract.py tests/test_t4_sparse_command_contract.py tests/test_t4_terrain_column_map.py tests/test_t4_stepping_stone_contracts.py tests/test_distributed_log_reduce.py tests/test_t4_sparse_evaluator_contract.py -q`
- 梅花桩 GRU 学生：`python -m pytest tests/test_t4_sparse_depth_student_gru_contract.py`（需要 torch）
- nubot：`scripts/nubot_run.sh`；zhuoqun：`scripts/zhuoqun_run.sh`
- 长任务一律 tmux。

## deferred_cleanup

- `docs/research/*` 历史调研：已用 `docs/research/README.md` 标明非权威；文内旧计划路径本次已改指向 S12，笔记正文不重写。
- 根目录 `findings.md` 已迁到 `docs/research/2026-08-22--s12-mdp-findings.md`。
- `docs/archive/plans/` 文头 living 指针：归档正文保持冻结；只允许一行 living 入口，不在本切片回写历史计划。
- `artifacts/eval/*`、视频、checkpoints：对照证据，未逐项核对 lineage，不删。
- `artifacts/diagnostics/eval_t4_impact_probe.py`、`open_s11b_sparse_stairs_viewer.py`：一次性诊断/viewer，可能仍能本地复现；正式评估走 `eval_t4_hurdle.py`。
  reevaluate_when: 下次 cleanup 或 S12 老师冻结后。
- `scripts/setup_local_isaac_docker.sh`：会话前已有本地改动，不混入提交。
- `.harness/progress.md` 中 S6–S10 超长 TB 流水：已有 S12 段在文首；全文压缩等下次专门收口。
  reevaluate_when: 下次 cleanup。
- pytest cache / 未跟踪 ckpt：不动。

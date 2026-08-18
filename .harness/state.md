# Current State

- Living index: `docs/README.md`
- Context: `PROJECT_CONTEXT.md`
- Dual track（各一份计划，不是双真相）:
  - 梅花桩：`docs/plans/2026-08-17--t4-sparse-lightlp-complete-plan.md`
  - 翻箱：`docs/plans/2026-08-15--t4-vault-g1-g2-recovery-plan.md`
- Approved specs: `docs/specs/2026-08-12--t4-unified-depth-locomotion.md`（走跑/学生合同）、`docs/specs/2026-08-13--t4-vault-loco-merge.md`（G1→G2→G3 目标）、`docs/specs/2026-08-15--t4-stepping-stones-and-hurdle-stable.md`（梅花桩目标；执行已改单阶段）

## 梅花桩（nubot）

- 任务 `t4_loco_teacher_sparse`，run `t_sparse_lightlp_s4`，四卡从零，真洞，无 soft/hard。
- tmux `t4-sparse-lightlp-s4`，日志 `logs/t4-sparse-lightlp-s4.log`，TB `:8007`（`http://100.100.188.39:8007/`）。
- Stage E `t4_loco_teacher` 1155D 未改。
- 旧 v4 软/硬与更早 sparse ckpt 只留对照，不加载。
- 本切片不训学生。能力声明要等 evaluator + 回放。

## 翻箱（zhuoqun）

- 执行面仍是 G1/G2 recovery。G3 / 与走跑合并被 G2 过箱挡住。
- 不要动 nubot 梅花桩 checkout 去盖 zhuoqun 翻箱。

## 已关闭（不要当待办）

- Stage E 走跑 + 跨栏 bucket + 深度学生 `model_24999` / 部署候选 `model_25746`。
- 走跑完成 ≠ 100m `rule`、真机障碍、梅花桩、翻箱 G3。

## 验证

- 本机：`python -m pytest tests/test_t4_asset_migration.py tests/test_t4_observation_contracts.py tests/test_t4_sparse_reward_contracts.py tests/test_t4_sparse_monitor_contract.py -q`
- nubot：`scripts/nubot_run.sh`；zhuoqun：`scripts/zhuoqun_run.sh`
- 长任务一律 tmux。

## deferred_cleanup

- `docs/research/*` 历史调研：已用 `docs/research/README.md` 标明非权威，不搬迁。
- `artifacts/eval/*` 与视频：可能是对照证据，未逐项核对 lineage，不删。
- 梅花桩学生蒸馏、G3 合并：等各自 teacher/G2 gate，不要提前写第二套执行计划。

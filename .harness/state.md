# Current State

- Living index: `docs/README.md`
- Context: `PROJECT_CONTEXT.md`
- Dual track（各一份计划，不是双真相）:
  - 梅花桩：`docs/plans/2026-08-19--t4-sparse-easy-geometry-s6-plan.md`（s5 计划已 superseded）
  - 翻箱：`docs/plans/2026-08-15--t4-vault-g1-g2-recovery-plan.md`
- Approved specs: `docs/specs/2026-08-12--t4-unified-depth-locomotion.md`（走跑/学生合同）、`docs/specs/2026-08-13--t4-vault-loco-merge.md`（G1→G2→G3 目标）、`docs/specs/2026-08-15--t4-stepping-stones-and-hurdle-stable.md`（梅花桩目标；执行已改单阶段）

## 梅花桩（nubot）

- 任务仍是 `t4_loco_teacher_sparse`。当前活跃 lineage 为 S9 `2026-08-20_16-07-04_t_sparse_lightlp_s9_capsulefix`，远端 worktree `/home/nubot/phn_ws/t4_train/TienKung-Lab-s9-capsulefix`，HEAD `fd84ee3`。训练 tmux `t4-sparse-lightlp-s9-capsulefix`，TensorBoard `http://100.100.188.39:8012/`。
- s5（`2026-08-19_00-20-19_t_sparse_lightlp_s5`）已停，仅作第一脚负结果。s4 仍只当列号错位基线。
- s6 已停，`model_31000.pt` 已作为 S9 warm-start，并重置 optimizer。S7/S8 均保留为失败证据：两者使用的 Shank capsule 与脚 collider 持续自重叠，不能再归因为 `termination_penalty=0` 的策略逃逸。S9 不加 `-200`，速度范围保持 `[-0.6, 1.0]`。
- `model_6000.pt` 配对诊断已排除“不抬脚”与 scan 断链；旧 9×9 格点在踏石/圆桩只支撑到约 2.20/2.45 m，与 4 m 课程 path / evaluator 前进门槛不一致，OOB 在 4.25 m。隔离 tile-filling grid 因果对照显著改善进度、减少掉坑。
- tile-filling grid 保留。`Trunk` box 保留；每侧 Shank 仍有膝部和胫骨 primitive collision。胫骨 URDF cylinder segment 已从 0.28 m 改为 0.20 m，使 Isaac capsule 的最终外包络为 0.28 m，与 MJCF 一致且不再插入脚 collider。
- `model_33000.pt` 修复前 evaluator：flat strict 13/16，踏石/圆桩 strict 均 0/16。修复后同 checkpoint 三类场景均 0/16 strict，flat 也 16/16 early，证明策略已适应污染 plant，不能续训。证据：`artifacts/diagnostics/s8_model33000_local_replay/summary.md`。
- S9 启动后 event step `31143` 早期窗口：`Episode_Reward/shank_contacts=0`，`undesired_contacts` 最近 20 点约 `-0.015`，`Reset/torso≈0.066`，`Reset/accel≈0.556`。假小腿接触未复现，但还没有新 checkpoint 的 fixed evaluator，不宣称稀疏地形能力。
- 当前 active slice：保持 S9 配方不变跑到 250/500 iter 门，`model_31500.pt` 出炉后做 flat + easy 踏石 + easy 圆桩 fixed evaluator 和连续回放。
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

- 本机：`python -m pytest tests/test_t4_asset_migration.py tests/test_t4_observation_contracts.py tests/test_t4_sparse_reward_contracts.py tests/test_t4_sparse_monitor_contract.py tests/test_t4_terrain_column_map.py -q`
- nubot：`scripts/nubot_run.sh`；zhuoqun：`scripts/zhuoqun_run.sh`
- 长任务一律 tmux。

## deferred_cleanup

- `docs/research/*` 历史调研：已用 `docs/research/README.md` 标明非权威，不搬迁。
- `artifacts/eval/*` 与视频：可能是对照证据，未逐项核对 lineage，不删。
- `artifacts/checkpoints/t4_depth_student_head35_model_20000.pt`：用户未跟踪 ckpt，本 cleanup 不动。
- `scripts/setup_local_isaac_docker.sh`：会话前已有本地改动，本 cleanup 不混入提交。
- 梅花桩学生蒸馏、G3 合并：等各自 teacher/G2 gate，不要提前写第二套执行计划。

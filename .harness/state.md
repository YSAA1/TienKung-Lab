# Current State

- 2026-09-07：排查后的 v3 已独立冷启动 30000 轮，统一 action_scale=0.5、原动作惩罚，撤回逐关节尺度补偿。之后按用户要求关闭旧 v2，未部署任何接续任务。
- Living index: `docs/README.md`
- 当前 G1 唯一执行计划：`docs/plans/2026-09-07--robot-neutral-locomotion.md`。v3 远端 `TienKung-Lab-g1-amp-fix-20260907`，代码冻结于 `d1f7f52`，tmux `g1-teacher-v3`，GPU 0/2、TB 8032；run `2026-09-07_10-08-58_g1_lightlp_teacher_v3`。原 AMP 难度曲线和 10% 全 0～9 保持。
- 新修复：AMP 统计更新使用原始样本、梯度惩罚与判别器使用同一坐标、多卡判别器参数/梯度与 AMP 统计同步。双 GPU 实测一致；启动核验 iteration 18、model_0、保存预算 30000。证据在 `artifacts/portability/v3/`，不把数值健康当作学习成功。
- 旧 v2 按用户要求在 iteration 9662 停止：tmux `g1-full-levels-v2` 与原 5 个训练进程全部退出，GPU 1/3 释放，20 个 checkpoint 保留；TB 8031 仍可查旧曲线。停止证据 `artifacts/portability/v3/stopped_v2.json`。共享教师、深度学生、动作跟踪与全库接入整理已完成；各候选行为能力仍由固定评估和连续回放判定。
- S12 学生 `s12_repr_first` `model_13999` Isaac hard 过门，MuJoCo 残留交给老师 plant 重训。老师本机 `artifacts/checkpoints/nubot/s12_teacher/model_21500.pt`。
- Approved specs: `docs/specs/2026-08-12--t4-unified-depth-locomotion.md`（走跑/学生合同）、`docs/specs/2026-08-13--t4-vault-loco-merge.md`（G1→G2→G3 目标）、`docs/specs/2026-08-15--t4-stepping-stones-and-hurdle-stable.md`（梅花桩目标；执行已改单阶段）、`docs/specs/2026-09-01--t4-s12-repr-first-distill.md`（表示先行蒸馏）

## 梅花桩（nubot）

- 任务 `t4_loco_teacher_sparse`。S12 老师 worktree `/home/nubot/phn_ws/t4_train/TienKung-Lab-s12-from-s11b-5k`，run `t_sparse_lightlp_s12_from_s11b_5k`，从 S11b `model_19000` 加载后冻在 `model_21500.pt`。
- 配方：0.75 m 收尾边框 + 踏石/圆桩 40% 轻转；`vx` 非洞 `[-0.6, 2.0]`，洞上 `[0.6, 2.0]`。终止阈值不改。`random_level_reset_max_level=None` 是 S11 起的老师课表，学生继承，不是 v3 单变量差。
- 学生 `fixed-v3-nanguard` 已在 iteration 3305 左右策略坍塌；失败取证链保留，不得续 `model_3500/4000`。TB 8017 进程仍活。
- 学生 lineage：worktree `TienKung-Lab-s12-gru-ppo`。**保留 D1** `s12_lightlp_dagger_only/model_10000.pt` 作对照。D3/`model_10500`、D3b、D3c 都不续。mix0 仍不 FT。
- **学生最终状态（2026-08-29）：** Phase B `s12_rtx_gated_joint/model_5999.pt` 已完成。修复 evaluator 的 GRU live hidden 污染与 episode reset 后，easy 踏石/圆桩 strict `32/32`、`29/32`，hard `18/32`、`27/32`，hard reach_2m `28/32`、`30/32`。两类 hard 连续回放、lineage 与 deploy-only 包齐全；3168-D 本机 dummy inference 输出 27-D finite action，GRU reset 通过。Phase C 跳过。
- **蒸馏成本切片墙钟已达标；质量未过。** 旧对照 `s12_lightlp_raycast` 已停在 `model_14000.pt`。近窗 collection p50 2.185 s / total p50 2.320 s。**不是** v3 坍塌。成本计划已归档。
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

- `docs/research/*` 与 `docs/archive/plans/` 历史正文不重写。
- `artifacts/eval/*`、视频、checkpoints：对照证据，未逐项核对 lineage，不删。
- `artifacts/diagnostics/*`：一次性 probe/viewer/JSON；正式评估走 `eval_locomotion.py`，旧 `eval_t4_hurdle.py` 为兼容入口。
- `scripts/setup_local_isaac_docker.sh`：会话前已有本地改动，不混入提交。
- `legged_lab/envs/t4/depth_student_cfg.py` 多代 AlgCfg（Dagger/Joint/DeployFt/TargetedFt/ResidualFt/PlantFt）+ `train_t4_sparse_depth_student_ft.py` 多 `--mode`：失败 lineage 对照与合同测试仍引用。删会改行为。
  reason: 不是未引用死代码，是叠代配方。
  reevaluate_when: 表示先行 Spec 批准并落地新 AlgCfg 之后，再单独 implement 切片收口旧 mode。
- pytest cache / 未跟踪 ckpt：不动。
- 本次独立 reviewer 的临时目录 `C:/Users/shash/AppData/Local/Temp/codex-tracking-review-20260907-a`：自动批准审查拒绝其删除，理由 `blocked by policy`。暂留；不影响工作树与训练，无替代工具删除尝试。

# 交接：T4 梅花桩 S12（2026-08-22）

> 本文是 **session 快照**，不是执行计划。开训 / 改配方认：
>
> - 入口：`docs/README.md`
> - 梅花桩计划：`docs/plans/2026-08-22--t4-sparse-s12-rim-yaw-student-plan.md`
> - 翻箱计划：`docs/plans/2026-08-15--t4-vault-g1-g2-recovery-plan.md`
> - 规格：`docs/specs/2026-08-15--t4-stepping-stones-and-hurdle-stable.md`
> - S6–S11 已归档：`docs/archive/plans/`（S10/S11b 只作对照，不热补旧 logdir）
>
> 不要用 reward、episode length、TB `success_rate` 或 checkpoint 存在宣称梅花桩能力。

## 一句话

S12 老师冻在 `model_21500.pt`。GRU 学生 `fixed-v3-nanguard` 已在 iteration 3305 左右由破坏性更新触发策略坍塌，冻结为失败证据。本地 safe recurrent 后继已实现，但尚未同步或启动远端新 lineage。S10/S11b 只读对照；翻箱仍在 zhuoqun。

## 当前 work surface

| 项 | 值 |
| --- | --- |
| 轨道 | 梅花桩 LightLP 老师 `t4_loco_teacher_sparse` 已冻；学生任务 `t4_loco_sparse_depth_student` 在修复 |
| 活跃 lineage | 无可继续的学生 lineage。`fixed-v3-nanguard` 已失败；下一条必须是新 logdir 的 safe recurrent lineage |
| 相对 S11b | S11b 已停，最后 `model_19000.pt`。S12 老师不再续到 24000 |
| 几何 | tile-filling grid + 每块梅花桩砖 **0.75 m** 实地边框；真 `Trunk` box + 双 `Shank`；胫骨 capsule 外包络 0.28 m |
| 命令 | 非洞 `vx ∈ [-0.6, 2.0]`；踏石/圆桩 `vx ∈ [0.6, 2.0]`、`vy=0`、60% `wz=0` / 40% `[-0.3, 0.3]` |
| 终止 | LightLP：horizon 20 s、OOB Chebyshev 4.25 m、关节 50 rad/s、Trunk>1 N、根加速度>40（1 s 热身）、倾角 63° 且 p=0.01。掉洞只记日志 |
| 观测 | Actor 1937D。Stage E 1155D 不动 |
| 翻箱 | zhuoqun G1/G2，不要动 nubot |

## 机器与进程（2026-08-22 晚）

| 项 | 值 |
| --- | --- |
| 训练机 | `nubot@100.100.188.39`（密码：一个空格） |
| worktree | `/home/nubot/phn_ws/t4_train/TienKung-Lab-s12-from-s11b-5k` |
| tmux | `fixed-v3` 最后会话名 `t4-sparse-s12-gru-student-21500-v3`；最近 SSH 探针超时，当前进程状态未刷新 |
| 老师专家 | `logs/t4_loco_teacher_sparse/2026-08-22_14-54-33_t_sparse_lightlp_s12_from_s11b_5k/model_21500.pt` |
| 启动 | `unset LD_LIBRARY_PATH` 后 `bash scripts/nubot_run.sh ...` |
| AMP | `T4_AMP_EXPERT_DIR=/home/nubot/phn_ws/t4_train/TienKung-Lab/artifacts/amp_expert_provisional` |
| zhuoqun | 翻箱，**不要动** |

S12 老师 / S11b / S10 / S9 产物只读对照。不要在旧 logdir 上热补。现行学生是用户授权的 ungated 开训；新入口需要 `--teacher_eval_manifest` 或 `--allow_ungated_teacher`。

## 证据口径

固定评估用 `legged_lab/scripts/eval_t4_hurdle.py`。看 JSON 的 `reach_2m` / `reach_4m` / strict + `reset_reason_counts`。**TB success / reward / episode length 不能当能力。** rank 0 TensorBoard 只覆盖约 1024 个环境，不是四卡 4096 的加权汇总。

失败链已钉死：3302 PG≈0.023、behavior≈0.135、reward≈9.56；3305 环境仍正常但 PG≈0.152、behavior≈0.896；3313 reward 才降到约 -2.57、length≈74。`model_3000→3500` depth encoder 相对变化约 92%。因此触发器是 updater，不是 TensorBoard、课程或环境先坏。

S11b `model_19000` 第一步侧偏钉死评估（d=0.8，`vx=0.8`，32 局）：`artifacts/eval/s12_firststep_pin/`。8 cm 侧偏没有更差。不据此热补。

## 不要重试什么

- 用 TB `success_rate` / 全局 reward / 课表均值 / length→1000 当梅花桩能力。
- 在 S10 / S11b / S12 老师 logdir 上热补速度、accel、边框。
- 从 `fixed-v3-nanguard` 的 `model_3500/4000` 续训，或只加 NaN guard/降 PG 再赌一次。
- 把新方案退化成永久 `pg_coef=0`、冻结 encoder 或纯蒸馏。
- 用 conda `env_isaaclab` 或本机 Windows 无 Docker 环境做正式 play。
- 占 nubot 训练四卡开 GUI play。
- 动 zhuoqun 翻箱 checkout。
- resume Stage E `stage_s_head35` 当梅花桩学生。
- 把 S7/S8 污染 ckpt 当续训。

## 风险 / blocker

- nubot 进程状态本轮未刷新；任何停训、同步或新启动前先做只读 tmux/process/GPU 核对。
- 新 lineage 的教师仍是用户授权的 `model_21500`，必须写明 `--allow_ungated_teacher`；学生 warm-start 只允许崩塌前 `model_3000`，并重建 critic/Adam/counters。
- `random_level_reset_max_level=None` 是 S11/S12 老师课表，学生继承，不是 v3 相对 v2 的单变量差。
- 翻箱 G3 仍被 G2 过箱挡住。

## 下一步（禁止只写 continue）

本地 safe recurrent 修复已通过 review 与 9 文件回归。获得远端动作确认后，只读核对 nubot，再同步本提交；用新 run name 启动 `SafeRecurrentDistillation` GPU smoke。优先 `--student_warmstart_checkpoint <fixed-v3/model_3000.pt>`，该入口只迁移 CNN/GRU/actor/recon/std，teacher/critic/Adam/counters 全部重建。smoke 先看 `update_accepted`、KL p95/emergency、rollback、behavior coefficient，再决定正式 lineage；能力仍看 evaluator + 消融 + 连续回放。

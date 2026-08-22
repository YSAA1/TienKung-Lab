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

S12 在 nubot 从 S11b `model_19000` **热启** 5k（reset optimizer，目标 24000）：0.75 m 收尾边框 + 踏石/圆桩 40% 轻转。GRU 学生代码已落地，waiter 等 `model_23999.pt`。S10/S11b 只读对照。翻箱仍在 zhuoqun，不要动 nubot 去盖。

## 当前 work surface

| 项 | 值 |
| --- | --- |
| 轨道 | 梅花桩 LightLP 单阶段老师 `t4_loco_teacher_sparse`；学生任务 `t4_loco_sparse_depth_student`（未开训） |
| 活跃 lineage | **S12** `t_sparse_lightlp_s12_from_s11b_5k`（19000→24000） |
| 相对 S11b | S11b 已停，最后 `model_19000.pt`。S12 **新 worktree/logdir**，加载该 ckpt 并 `--reset_optimizer` |
| 几何 | tile-filling grid + 每块梅花桩砖 **0.75 m** 实地边框；真 `Trunk` box + 双 `Shank`；胫骨 capsule 外包络 0.28 m |
| 命令 | 非洞 `vx ∈ [-0.6, 2.0]`；踏石/圆桩 `vx ∈ [0.6, 2.0]`、`vy=0`、60% `wz=0` / 40% `[-0.3, 0.3]` |
| 终止 | LightLP：horizon 20 s、OOB Chebyshev 4.25 m、关节 50 rad/s、Trunk>1 N、根加速度>40（1 s 热身）、倾角 63° 且 p=0.01。掉洞只记日志 |
| 观测 | Actor 1937D。Stage E 1155D 不动 |
| 翻箱 | zhuoqun G1/G2，不要动 nubot |

## 机器与进程（2026-08-22）

| 项 | 值 |
| --- | --- |
| 训练机 | `nubot@100.100.188.39`（密码：一个空格） |
| worktree | `/home/nubot/phn_ws/t4_train/TienKung-Lab-s12-from-s11b-5k` |
| tmux | `t4-sparse-lightlp-s12-from-s11b-5k`；学生 waiter `t4-s12-gru-student-wait` |
| TB | [http://100.100.188.39:8016/](http://100.100.188.39:8016/) |
| 启动 | `unset LD_LIBRARY_PATH` 后 `bash scripts/nubot_run.sh ...` |
| AMP | `T4_AMP_EXPERT_DIR=/home/nubot/phn_ws/t4_train/TienKung-Lab/artifacts/amp_expert_provisional` |
| zhuoqun | 翻箱，**不要动** |

S11b / S10 / S9 产物只读对照。不要在旧 logdir 上热补。

## 证据口径

固定评估用 `legged_lab/scripts/eval_t4_hurdle.py`。看 JSON 的 `reach_2m` / `reach_4m` / strict + `reset_reason_counts`。**TB success 不能当能力。** 不要把 mean episode length → 1000 写成成功。

S11b `model_19000` 第一步侧偏钉死评估（d=0.8，`vx=0.8`，32 局）：`artifacts/eval/s12_firststep_pin/`。8 cm 侧偏没有更差。不据此热补 S12。

S10 `model_30000` / `model_39999` 矩阵仍在 `artifacts/eval/`，只当旧物理+旧命令对照。

## 不要重试什么

- 用 TB `success_rate` / 全局 reward / 课表均值 / length→1000 当梅花桩能力。
- 在 S10 / S11b logdir 上热补速度、accel、边框。
- 删 accel + torso，只靠倒下复位。
- 用 conda `env_isaaclab` 或本机 Windows 无 Docker 环境做正式 play。
- 占 nubot 训练四卡开 GUI play。
- 动 zhuoqun 翻箱 checkout。
- resume Stage E `stage_s_head35` 当梅花桩学生。
- 把 S7/S8 污染 ckpt 当续训。

## 风险 / blocker

- S12 占 nubot 四卡到 24000；改配方必须等它停或另开机器。
- 老师未到 `model_23999.pt` 前不要手开学生。
- 边框若只加视觉 mesh、没进碰撞，观感仍会掉洞。
- 翻箱 G3 仍被 G2 过箱挡住。

## 下一步（禁止只写 continue）

等 S12 到 24000：fixed evaluator + 边框观感回放。过门后让 waiter 开 GRU 蒸馏。能力仍看固定 evaluator，不看 TB。

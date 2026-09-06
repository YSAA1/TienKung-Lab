# T4 文档入口

本 checkout 在上游 TienKung-Lab 上做 **T4 27DoF** 感知走跑与技能。
先读本文，再进代码。`docs/archive/` 与 `docs/research/` **不是**开训或接票权威。

## 当前工作面

当前执行轨道：宇树 G1 越障老师（复用 T4 LightLP 稀疏地形），以及翻箱 G1/G2。T4 S12 老师 plant 重训暂停。表示先行学生 `model_13999` 是 Isaac 对照候选，不是部署包。

| 轨道 | 现在做什么 | 执行计划 | 规格 |
| --- | --- | --- | --- |
| 梅花桩 | paused：G1 越障老师优先。T4 plant DR 计划仍在 | [老师 plant 重训](plans/2026-09-02--t4-s12-teacher-plant-retrain-plan.md) | [表示先行蒸馏](specs/2026-09-01--t4-s12-repr-first-distill.md)（学生对照） |
| Unitree G1 越障 teacher | active：`g1_lightlp_amp_full_levels_v2` 已启动，GPU 1、3；原 AMP 与 10% 全 0～9 等级，TensorBoard 8031 | [机器人无关算法整理](plans/2026-09-07--robot-neutral-locomotion.md) | `legged_lab/locomotion` + 各机器人资产/spec；能力待评估 |
| 翻箱 | G1 跟踪专家 → G2 heightscan 技能（zhuoqun）。**G2 学生与 G3 合并（走跑+翻箱成一条策略）等 G2 过箱后再开。** | [翻箱 recovery](plans/2026-08-15--t4-vault-g1-g2-recovery-plan.md) | [合并规格](specs/2026-08-13--t4-vault-loco-merge.md) |

运行时切片与机器占用：`.harness/work_index.md`、`.harness/state.md`。

## 已经关掉的阶段

不要把这些再当待办。

- **Stage E 走跑老师** `t4_loco_teacher`：1155D，平地 / rough / 楼梯 / 跨栏 bucket。主线 `stage_e_prov7_hurdle`。
- **深度学生** `pi_loco`：`stage_s_head35/model_24999.pt`，后续 FT `model_25746.pt`。导出只有深度 + 本体史 + 速度命令 + 上一动作。
- **跨栏**：并入 Stage E 地形，不再单独开 skill。
- 失败或被取代的梅花桩尝试（旧 25k、S1d、软/硬 v4）只留对照 ckpt 与归档计划。

走跑完成 **不等于** 100m `rule` 全路线、真机障碍验收、梅花桩能力或翻箱 G3。

2026-08-20 共享 T4 URDF 新增了躯干/双小腿碰撞。旧 Stage E、depth student 和 vault checkpoint 仍是有效产物，但它们过去的仿真 evaluator 证据属于旧 plant；在新资产上做新能力声明前必须重评。

## 冻结合同（改了就开新 lineage）

| 项 | 真值 |
| --- | --- |
| 关节序 | `legged_lab/assets/t4/constants.py::T4_JOINT_NAMES` |
| 默认 Stage E Actor | 1155D（本体史 10×96 + 前向 scan 195）。翻箱 G2 / 旧学生仍吃这个。 |
| Sparse teacher Actor | 1937D（scan×5 + 接触 2）。独立任务 `t4_loco_teacher_sparse`。 |
| 足底 scan | 只进 Critic / 奖励，不上 Actor、不上实机 |
| 部署学生 | 深度 + 本体史 + 命令 + 上一动作；无 HeightScan / 接触真值 / 路线进度 |
| 命令范围（现行走跑） | Stage E：`vx ∈ [-0.6, 1.0]`。S12 sparse 非洞地形：`vx ∈ [-0.6, 2.0]`。踏石/圆桩：`vx ∈ [0.6, 2.0]`、`vy=0`、60% `wz=0` / 40% `[-0.3, 0.3]`。每块梅花桩砖有 0.75 m 实地边框 |
| 能力声明 | evaluator JSON + lineage + 连续回放。Reward / TB / ckpt 存在都不算。 |

架构规格：[特权老师 → 深度学生](specs/2026-08-12--t4-unified-depth-locomotion.md)。

## 怎么跑

| 目的 | 入口 |
| --- | --- |
| nubot Isaac 训练 | `bash scripts/nubot_run.sh …`（tmux） |
| zhuoqun Isaac | `bash scripts/zhuoqun_run.sh …`（docker `t4-isaac-jammy:v2`） |
| 本机 play 拉下来的 ckpt | [本机 Isaac Docker](runbooks/local-isaac-docker.md) → `scripts/local_run.sh` |
| 老师 / 深度学生上机合同 | [部署手册](runbooks/t4-teacher-and-depth-deployment.md) |
| 资产合同 | `python -m pytest tests/test_t4_asset_migration.py` |
| 观测合同 | `python -m pytest tests/test_t4_observation_contracts.py` |
| 稀疏奖励 / 终止 / 列映射 / S12 命令与边框合同 | `python -m pytest tests/test_t4_sparse_reward_contracts.py tests/test_t4_sparse_monitor_contract.py tests/test_t4_sparse_command_contract.py tests/test_t4_terrain_column_map.py tests/test_t4_stepping_stone_contracts.py tests/test_distributed_log_reduce.py` |

已注册 T4 任务：`t4_loco_teacher`、`t4_loco_teacher_sparse`、`t4_loco_sparse_depth_student`、`t4_vault_mimic`、`t4_vault_skill`。上游 TienKung 的 `walk` / `run` 仍在，不是本 T4 路线。Stage E 深度学生仍走独立 train 脚本，不改任务名。

## 文档货架

| 目录 | 角色 |
| --- | --- |
| 本文、`AGENTS.md` | 入口。不写 session 流水。 |
| `.harness/` | 工作面索引与短状态。 |
| `docs/specs/` | 已批准行为 / 架构。文头 `Status: draft` 的不是执行权威。现行学生配方是 [表示先行蒸馏](specs/2026-09-01--t4-s12-repr-first-distill.md)（已批准）。 |
| `docs/plans/` | **只放现行执行计划**（梅花桩 S12 + 部署向 RTX 蒸馏、翻箱 G1/G2）。 |
| `docs/runbooks/` | 现在还能照着跑的操作。 |
| `docs/research/` | 调研与论文摘录，非权威。 |
| `docs/archive/` | 已完成或已取代的执行纸（含已交付的学生成本/RTX 切片）。 |

上游框架安装与 walk/run 演示仍在仓库根 [README.md](../README.md)。

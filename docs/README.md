# T4 文档入口

本 checkout 在上游 TienKung-Lab 上做 **T4 27DoF** 感知走跑与技能。
先读本文，再进代码。`docs/archive/` 与 `docs/research/` **不是**开训或接票权威。

## 当前工作面

两条并行轨道，各有一份执行计划。不要再开第三份「唯一计划」。

| 轨道 | 现在做什么 | 执行计划 | 规格 |
| --- | --- | --- | --- |
| 梅花桩 | nubot 四卡从零 `t_sparse_lightlp_s5`（列号已修）。s4 只当错误基线。**本切片不蒸学生。** | [梅花桩计划](plans/2026-08-18--t4-sparse-terrain-index-fix-plan.md) | [稀疏规格](specs/2026-08-15--t4-stepping-stones-and-hurdle-stable.md) |
| 翻箱 | G1 跟踪专家 → G2 heightscan 技能（zhuoqun）。**G2 学生与 G3 合并（走跑+翻箱成一条策略）等 G2 过箱后再开。** | [翻箱 recovery](plans/2026-08-15--t4-vault-g1-g2-recovery-plan.md) | [合并规格](specs/2026-08-13--t4-vault-loco-merge.md) |

运行时切片与机器占用：`.harness/work_index.md`、`.harness/state.md`。

## 已经关掉的阶段

不要把这些再当待办。

- **Stage E 走跑老师** `t4_loco_teacher`：1155D，平地 / rough / 楼梯 / 跨栏 bucket。主线 `stage_e_prov7_hurdle`。
- **深度学生** `pi_loco`：`stage_s_head35/model_24999.pt`，后续 FT `model_25746.pt`。导出只有深度 + 本体史 + 速度命令 + 上一动作。
- **跨栏**：并入 Stage E 地形，不再单独开 skill。
- 失败或被取代的梅花桩尝试（旧 25k、S1d、软/硬 v4）只留对照 ckpt 与归档计划。

走跑完成 **不等于** 100m `rule` 全路线、真机障碍验收、梅花桩能力或翻箱 G3。

## 冻结合同（改了就开新 lineage）

| 项 | 真值 |
| --- | --- |
| 关节序 | `legged_lab/assets/t4/constants.py::T4_JOINT_NAMES` |
| 默认 Stage E Actor | 1155D（本体史 10×96 + 前向 scan 195）。翻箱 G2 / 旧学生仍吃这个。 |
| Sparse teacher Actor | 1937D（scan×5 + 接触 2）。独立任务 `t4_loco_teacher_sparse`。 |
| 足底 scan | 只进 Critic / 奖励，不上 Actor、不上实机 |
| 部署学生 | 深度 + 本体史 + 命令 + 上一动作；无 HeightScan / 接触真值 / 路线进度 |
| 命令范围（现行走跑） | `vx ∈ [-0.6, 1.0]` m/s |
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
| 稀疏奖励 / 终止 / 列映射合同 | `python -m pytest tests/test_t4_sparse_reward_contracts.py tests/test_t4_sparse_monitor_contract.py tests/test_t4_terrain_column_map.py` |

已注册 T4 任务：`t4_loco_teacher`、`t4_loco_teacher_sparse`、`t4_vault_mimic`、`t4_vault_skill`。上游 TienKung 的 `walk` / `run` 仍在，不是本 T4 路线。

## 文档货架

| 目录 | 角色 |
| --- | --- |
| 本文、`PROJECT_CONTEXT.md`、`AGENTS.md` | 入口。不写 session 流水。 |
| `docs/specs/` | 已批准行为 / 架构。half-living 的文头会标明。 |
| `docs/plans/` | **只放现行执行计划**（现在就上面两份）。 |
| `docs/runbooks/` | 现在还能照着跑的操作。 |
| `docs/research/` | 调研与论文摘录，非权威。 |
| `docs/archive/` | 已完成或已取代的执行纸。 |

背景与边界的短文：[PROJECT_CONTEXT.md](../PROJECT_CONTEXT.md)。
上游框架安装与 walk/run 演示仍在仓库根 [README.md](../README.md)。

# Progress

## 2026-08-19 s6 easy 几何

s5 @17.5k 踏石/圆桩 `reach_2m` 仍 3%/2%、进度 1.1 m。回放第一脚 accel。代码改 easy 踏石 40/10/9 cm、圆桩高 8 cm、随机 level cap=4。pytest 50 passed。nubot 停 s5，从零 `2026-08-19_12-51-05_t_sparse_lightlp_s6`，`resume: false`，TB `:8009`。14:20 CST 刷新：训练/TB 存活，ckpt 到 `model_2000.pt`，step≈2171；两类 easy `reach_2m mean20` 仍约 1.1% / 2.3%，进度约 1.27 m，未到 3k 健康门，不提前停。

14:56 CST 3k 健康门：踏石 easy 通过（`reach_2m mean20≈12.2%`、`progress≈1.40 m`），圆桩 easy 未通过（`reach_2m≈3.1%`、`progress≈1.22 m`）。S6 整体标高风险，但按计划继续到 5k 决策门；未授权不改配方。

16:09 CST 5k 前：step≈4870。踏石 easy `reach_2m mean20≈29.3%`、`progress≈1.61 m`；圆桩 easy `reach_2m≈14.7%`、`progress≈1.33 m`，还没过 20%/1.40，但从 3k 持续上升，不是平台期。用户确认不要机械停；若 5k 仍上升且未触发硬停线，继续到 6k/7k 再判。

16:17 CST 5k 后：踏石 easy 过门（`reach_2m mean20≈29.7%`、`progress≈1.60 m`）；圆桩 easy 未过 reach 门但接近 progress 门（`reach_2m≈15.9%`、`progress≈1.38 m`），仍在上升，未触发硬停线，继续训。hard 曲线全 0 因踏石/圆桩 `hard_episodes=0`；mid 样本很少，只作参考。

## 2026-08-19 cleanup

删除根目录审计草稿 `findings.md` / `progress.md` / `task_plan.md`（内容已进现行计划与 `.harness`）。s4 计划迁到 `docs/archive/plans/`。TB 抓数 scratch 不入库。

## 2026-08-19 开训 s5

停 s4 tmux，ckpt 留到 `model_12500.pt`。nubot 工作区覆盖了列映射/MDP 源码后从零启动 `t_sparse_lightlp_s5`。logdir `logs/t4_loco_teacher_sparse/2026-08-19_00-20-19_t_sparse_lightlp_s5`，无 load ckpt。TB `:8008`。

## 2026-08-18 阶段 1–2 代码落地

列映射 helper `terrain_columns.py`；`t4_env` / `rewards` 走列号；TB 按真名聚合并打 `TerrainCol/*/occupancy` 与 `Curriculum/level_*_frac`。LightLP：Eq.4 求和、Eq.5 泄漏积分、opposite 点积、10% 随机 level、路径长度晋级。审查后又修了两处：play 单类型 `curriculum=False` 时全列绑定；horizon 超时重抽命令不再翻 standing/moving。相关 pytest 74 passed / 1 skipped。未开 s5、未停 s4。

## 2026-08-18 s4 列号审计与 TB 重读

IsaacLab 2.1.0 `terrain_types` 是 20 列列号。s4 代码用 13 个 `sub_terrains` 下标，TB 名整体错位。nubot EventAccumulator @ iter 12018：连续地形 reach_2m 0.70–0.77、progress 3.3–3.5 m；四列踏石与三列可见圆桩 reach_2m 0.02–0.05、progress 1.17–1.30 m。他人「踏石 71%」实为跨栏列。执行面改到 `docs/plans/2026-08-18--t4-sparse-terrain-index-fix-plan.md`。

## 2026-08-18 文档收口与 s4

入口改为 `docs/README.md`。S1d / rollback / 软硬 v4 计划已进 `docs/archive/plans/`。现行训练 `t_sparse_lightlp_s4`（nubot 四卡，真洞从零）。翻箱仍走 G1/G2 recovery。

## 2026-08-17 梅花桩完整补全改从零

用户裁定：不续训；不做短 FT；踏石方砖太大必须改；sparse Actor 1155D 可打破。执行面改为 `docs/plans/2026-08-17--t4-sparse-lightlp-complete-plan.md`。默认 Stage E 1155D 不动。


## 2026-08-17 S1d 失败抽检

原始表：`artifacts/eval/t4_sparse_ab_tb_dump.json`。nubot A/B 当时仍在跑，ckpt ≥ `model_11000.pt`。

3.5k 对齐：v2 reward 57.0 / 踏石 0.75；稳定组 48.1 / 0.76；对照组 43.6 / 0.75。圆桩三条 success 都是 0、progress ≈1.1 m。

末值：稳定组 11629 reward 56.4 踏石 0.81/success 0.45 圆桩 0.019/success 0；对照组 11795 reward 57.5 踏石 0.79 圆桩 ≈0。`legal_foothold` 稳定组 4e-6。

## 2026-08-17 S1d 回退与续训

- 代码已去掉 stable/allin、双 Critic、脚下 scan、legal_foothold、稀疏 gait/AMP/stumble 缩放。
- 本机合同 `42 passed`。
- nubot A/B tmux 已停，ckpt 保留。
- 续训 tmux `t4-sparse-v2-resume`，CUDA 0,1，任务 `t4_loco_teacher_sparse`，从 `2026-08-16_13-18-07_t_compat_sparse_v2/model_3500.pt` 加载，run `2026-08-17_01-50-07_t_compat_sparse_v2_resume`，日志 `logs/t4-sparse-v2-resume.log`。已见 PPO 从 3500+ 继续。

# Progress

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

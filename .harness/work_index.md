# Work Index

| Work surface | Status | Active slice | Source |
| --- | --- | --- | --- |
| T4 走跑与 1m 翻箱合并（LightLP V-B/V-C） | superseded | V7 G3 被 recovery 接管；旧 G2 配方作废 | `docs/archive/plans/2026-08-13--t4-vault-loco-merge-plan.md` |
| T4 翻箱 G1 专家补齐与 G2 蒸馏重做 | active | 混合 DAgger 15k 已开（`g2_dagger_mix15k`） | `docs/plans/2026-08-15--t4-vault-g1-g2-recovery-plan.md` |
| T4 梅花桩双 teacher 与跨栏 0.30 | superseded | S1d 失败后经回退，现执行面为 LightLP 完整补全 | `docs/plans/2026-08-15--t4-stepping-stones-and-hurdle-stable-plan.md` |
| T4 稀疏落足 S1d 回退与 v2 续训 | superseded | 用户改从零 + 收窄踏石 + 允许新观测维 | `docs/plans/2026-08-17--t4-sparse-ab-rollback-plan.md` |
| T4 梅花桩 LightLP 完整补全 | active | 阶段 3：nubot 软阶段从零开训 `t_sparse_lightlp_v4`（4096 env × 35k） | `docs/plans/2026-08-17--t4-sparse-lightlp-complete-plan.md` |
| T4 连续跨栏并入 Stage E 课程（100m 障碍 #2） | done | 已并入 loco 阶段；通过 `model_24000.pt` MuJoCo `loco` 交互回放人工复核 | `docs/archive/plans/2026-08-13--t4-hurdle-skill-plan.md` |
| T4 统一深度感知 Locomotion | done | Stage E teacher + depth student 阶段完成；student `stage_s_head35/model_24999.pt`，MuJoCo `loco` 交互回放通过 | `docs/archive/plans/2026-08-12--t4-unified-depth-locomotion-plan.md` |
| T4 depth student ZL/direct stair parity | done | 原始 27DoF plant + 首帧 seed action 对齐；`model_25746.pt`、`vx=0.8` 独立冷启动 3/3 通过 | `docs/runbooks/t4-teacher-and-depth-deployment.md` |

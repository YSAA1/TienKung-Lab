# Work Index

权威入口：`docs/README.md`。本表只列工作面状态，不写 session 流水。

| Work surface | Status | Active slice | Source |
| --- | --- | --- | --- |
| T4 梅花桩 LightLP teacher | frozen | S12 从 S11b `model_19000` 热启后冻在 `model_21500.pt`（用户授权提前蒸学生，不再等 23999） | `docs/plans/2026-08-22--t4-sparse-s12-rim-yaw-student-plan.md` |
| T4 梅花桩 GRU 深度学生 | repair ready | `fixed-v3-nanguard` 已策略坍塌并冻结取证。本地 safe recurrent updater 已实现；新 lineage 尚未远端启动，拟用 S12 `model_21500` + 可选 pre-collapse `model_3000` student-only warm-start | `docs/plans/2026-08-22--t4-sparse-s12-rim-yaw-student-plan.md` |
| T4 翻箱 G1/G2 | active | G1 专家已可用；G2 蒸馏 / 学生跟车在 zhuoqun。G3 合并未开 | `docs/plans/2026-08-15--t4-vault-g1-g2-recovery-plan.md` |
| T4 统一深度感知走跑 | done | Stage E + depth student；跨栏已并入 Stage E | `docs/specs/2026-08-12--t4-unified-depth-locomotion.md` |
| T4 depth student ZL/direct 楼梯对齐 | done | `model_25746.pt`，`vx=0.8` 冷启动 3/3 | `docs/runbooks/t4-teacher-and-depth-deployment.md` |
| T4 梅花桩 S6–S11 | superseded | 对照 lineage；计划已归档 | `docs/archive/plans/` |
| 旧翻箱一次性合并到 G3 | superseded | 被 G1/G2 recovery 接管 | `docs/archive/plans/2026-08-13--t4-vault-loco-merge-plan.md` |

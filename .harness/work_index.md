# Work Index

权威入口：`docs/README.md`。本表只列工作面状态，不写 session 流水。

| Work surface | Status | Active slice | Source |
| --- | --- | --- | --- |
| 项目统一整理 | active | `D:/TienKung-Lab` 的 `develop`；整合 T4/G1/Z2、公共接口与测试，最终 main + develop、单工作树；远端训练不热覆盖 | `docs/plans/2026-09-09--project-consolidation.md` |
| Z2 29DoF teacher | done / 30k 训练完成 + 双仿真器验收 | `reset_aligned_v1` `model_29999`（SHA256 `491ff4…`）：Isaac d=0 flat/踏石/圆桩 64/64、64/64、63/64 零摔，d=0.85 踏石 60/64；本机 MuJoCo 平地 10/10、踏石 sim2sim gap。本机 ckpt+params `artifacts/checkpoints/nubot/z2_reset_aligned_v1/`；结论与验收入口 `artifacts/z2_migration/sim2sim_20260911/FINDINGS.md`；数据体检 `artifacts/z2_migration/data_check_20260911/` | `docs/plans/2026-09-09--z2-teacher-migration.md`；`artifacts/z2_migration/reset_aligned_v1/lineage.json` |
| T4 梅花桩 LightLP teacher | paused | G1 老师开训优先。T4 plant DR 计划仍在 `docs/plans/2026-09-02--t4-s12-teacher-plant-retrain-plan.md`，不开 | `docs/plans/2026-09-02--t4-s12-teacher-plant-retrain-plan.md` |
| Unitree G1 越障 teacher | active / vital_v3 + vital_v2 训练中 | v3=curated `unitree_v6` AMP + ramp DR；nubot `TienKung-Lab-g1-vital-v3-20260911`；tmux `g1-vital-v3`；GPU0/2；TB8051。v2（`unitree_v5` 全量 DR）GPU1/3 26.5k+/30k 对照 | `docs/plans/2026-09-11--g1-vital-v3-curated-amp-ramped-dr-plan.md`；`artifacts/g1_amp_acceptance_20260911/` |
| T4 梅花桩 GRU 深度学生 | done | `s12_repr_first` `model_13999` Isaac hard 过门；MuJoCo 稀疏仍摔。下一刀跟新老师，不续 14k 蒸馏 | `docs/plans/2026-09-01--t4-s12-repr-first-distill-plan.md` |
| T4 梅花桩学生 LightLP 成本 | done | warp + 3168/MLP + 单次 backward；collection p50 ~2.2 s。质量未过，已转配方修复 | `docs/archive/plans/2026-08-23--t4-sparse-s12-student-lightlp-distill-cost-plan.md` |
| T4 学生 headless RTX 出图正确性 | done | `sim.render()` 调度 + checksum probe 绿；`s12_gru_ppo_rtx167` 已按 16.7 Hz 真出图 | `docs/archive/plans/2026-08-23--t4-sparse-s12-student-rtx-correctness-plan.md` |
| T4 翻箱 G1/G2 | active | G1 专家已可用；G2 蒸馏 / 学生跟车在 zhuoqun。G3 合并未开 | `docs/plans/2026-08-15--t4-vault-g1-g2-recovery-plan.md` |
| T4 统一深度感知走跑 | done | Stage E + depth student；跨栏已并入 Stage E | `docs/specs/2026-08-12--t4-unified-depth-locomotion.md` |
| T4 depth student ZL/direct 楼梯对齐 | done | `model_25746.pt`，`vx=0.8` 冷启动 3/3 | `docs/runbooks/t4-teacher-and-depth-deployment.md` |
| T4 梅花桩 S6–S11 | superseded | 对照 lineage；计划已归档 | `docs/archive/plans/` |
| 旧翻箱一次性合并到 G3 | superseded | 被 G1/G2 recovery 接管 | `docs/archive/plans/2026-08-13--t4-vault-loco-merge-plan.md` |

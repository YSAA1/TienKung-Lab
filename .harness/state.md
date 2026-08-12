# Current State

- Planning surface: `docs/plans/2026-08-13--t4-tienkung-native-walk-plan.md`
- Approved Spec: `docs/specs/2026-08-13--t4-tienkung-native-walk.md`
- Active item: W2 zhuoqun Isaac 运行时发现（blocked）
- Branch / worktree: `t4-walk` @ `D:\TienKung-Lab-t4-walk`，基线 commit `5c70898`
- Verification path: 本机 `python -m pytest tests/test_t4_observation_contracts.py tests/test_t4_walk_contracts.py tests/test_t4_terrain_curriculum.py tests/test_t4_asset_migration.py` runnable；Isaac/tmux gates 在 zhuoqun **blocked**。
- Next skill: 用户提供 Isaac 运行时后 `implement` W2/W3；否则 `harness-builder`
- Long-running rule: 所有 TienKung-Lab 训练必须在 Isaac + tmux 中运行；禁止占用 nubot Stage E。
- Parallel lineage (do not touch): nubot `stage_e_prov2`，session `t4-stage-e`，task `t4_loco_teacher`。
- Route W frozen: task `t4_walk`，`policy_role=walk`，Actor = proprio history only，terrain = `GRAVEL_TERRAINS_CFG`（无课程），command resampling 10s，AMP coef 恒定 0.3，shank 接触终止。
- W1 evidence: 35 passed on the four contract files (2026-08-13, worktree).
- Current blocker: zhuoqun 无 Isaac Sim `python.sh` / IsaacLab source；四卡被 MjLab `t4_stair_traversal` 占用。未开训。

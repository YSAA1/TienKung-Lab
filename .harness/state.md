# Current State

- Planning surface: `docs/plans/2026-08-12--t4-unified-depth-locomotion-plan.md`
- Approved Spec: `docs/specs/2026-08-12--t4-unified-depth-locomotion.md`
- Active item: M0 T4 资产、相机与 motion 事实闭环
- Verification path: offline M0 audit runnable in current Windows shell; IsaacLab/tmux gates require target Linux GPU runtime.
- Next skill: `implement`
- Long-running rule: 所有训练、GPU probe、批量 playback 和 evaluator 必须在 tmux 中运行。
- Stop gate: M0 未通过前不生成正式 AMP expert，不启动长训练。
- Latest M0 evidence: `artifacts/eval/t4_motion_audit.json` machine-audited 18 motions; 17 accepted, `t4_run` rejected for hard joint limit violations plus holdout rule.
- Current blocker: this shell has no `isaaclab` Python module and no `tmux`, so IsaacLab spawn/camera smoke and human playback review are still pending.
- New M0 simulator entrypoint: `legged_lab/scripts/playback_t4_motions.py` writes raw T4 frames into the IsaacLab T4 articulation and emits `artifacts/eval/t4_motion_playback.json` in the target nubot runtime.
- Nubot target: `nubot@100.100.188.39:/home/nubot/phn_ws/t4_train/TienKung-Lab` is on commit `039b8ac` (`feat(t4): 增加M0资产动作审计入口`).
- Nubot evidence: `python3 legged_lab/scripts/audit_t4_motions.py --task t4_loco_teacher --output artifacts/eval/t4_motion_audit.remote.json` passed with 18 motions, 17 accepted, 1 rejected.
- Nubot blocker: IsaacLab 2.1 source imports, but the conda `isaaclab` runtime resolves pip Isaac Sim 4.2 extensions and fails `isaacsim.core.api` / `omni.kit.usd`; Isaac Sim 5.1 standalone can start after newer `libstdc++`, but its Python cannot import a matching torch stack yet.

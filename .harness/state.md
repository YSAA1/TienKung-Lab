# Current State

- Planning surface: `docs/plans/2026-08-12--t4-unified-depth-locomotion-plan.md`
- Approved Spec: `docs/specs/2026-08-12--t4-unified-depth-locomotion.md`
- Active item: M0 T4 资产、相机与 motion 事实闭环
- Verification path: offline M0 audit runnable in current Windows shell; IsaacLab/tmux gates verified on target nubot Linux GPU runtime.
- Next skill: `verify`
- Long-running rule: 所有训练、GPU probe、批量 playback 和 evaluator 必须在 tmux 中运行。
- Stop gate: M0 未通过前不生成正式 AMP expert，不启动长训练。
- Latest M0 evidence: `artifacts/eval/t4_motion_audit.json` machine-audited 18 motions; 17 accepted, `t4_run` rejected for hard joint limit violations plus holdout rule.
- Latest simulator evidence: nubot IsaacLab spawn smoke loaded T4 with 27 joints and 30 bodies; joint name set matches `T4_JOINT_NAMES`, runtime order differs and playback reorders by name.
- Latest playback evidence: `artifacts/eval/t4_motion_playback_smoke.json` simulated `t4_stand` 5 frames with 0 rejects; `artifacts/eval/t4_motion_playback.json` simulated all 18 motions with 0 rejects and 18 human playback approvals still pending.
- Current blocker: no machine blocker for M0 headless spawn/playback; human visual playback review is still pending before formal AMP expert generation.
- New M0 simulator entrypoint: `legged_lab/scripts/playback_t4_motions.py` writes raw T4 frames into the IsaacLab T4 articulation and emits `artifacts/eval/t4_motion_playback.json` in the target nubot runtime.
- Nubot target: `nubot@100.100.188.39:/home/nubot/phn_ws/t4_train/TienKung-Lab`; M0 headless spawn/playback evidence was collected on commit `dfb69d1` (`fix(t4): 按名称重排动作关节`).
- Nubot runtime: use `/home/nubot/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh` with IsaacLab source paths in `PYTHONPATH`; torch verified as `2.5.1+cu124`, CUDA visible on 4 GPUs.
- Nubot caveat: GitHub fetch can fail with `GnuTLS recv error (-110)`; latest commits were synced through local git bundles over SSH when needed.

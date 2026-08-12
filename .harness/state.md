# Current State

- Planning surface: `docs/plans/2026-08-12--t4-unified-depth-locomotion-plan.md`
- Approved Spec: `docs/specs/2026-08-12--t4-unified-depth-locomotion.md`
- Active item: M0 T4 资产、相机与 motion 事实闭环
- Verification path: runnable
- Next skill: `implement`
- Long-running rule: 所有训练、GPU probe、批量 playback 和 evaluator 必须在 tmux 中运行。
- Stop gate: M0 未通过前不生成正式 AMP expert，不启动长训练。

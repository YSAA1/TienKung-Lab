# Work Index

| Work surface | Status | Active slice | Source |
| --- | --- | --- | --- |
| T4 走跑与 1m 翻箱合并（LightLP V-B/V-C） | active | V4 G1 训练挂机（zhuoqun tmux `t4_vault_g1`）+ V2 evaluator 并行；V0/V1/V3 完成 2026-08-13 | `docs/plans/2026-08-13--t4-vault-loco-merge-plan.md` |
| T4 连续跨栏 skill（100m 障碍 #2） | parallel | H0 布局真值与碰栏判定纯 Python 合同（本机可验证，待开工指令） | `docs/plans/2026-08-13--t4-hurdle-skill-plan.md` |
| T4 统一深度感知 Locomotion | parallel | M1/M2 与 Stage E teacher（nubot `stage_e_prov5` 训练中，等待 teacher evaluator；本 checkout 不改训练 MDP） | `docs/plans/2026-08-12--t4-unified-depth-locomotion-plan.md` |
| T4 TienKung-native Walk AMP（Route W） | parallel | W1 已提交 `73afddd`；zhuoqun worktree `/home/zhuoqun/workspace/TienKung-Lab-t4-walk`；原「无 IsaacLab」blocker 已被 vault V3 docker runtime（`t4-isaac-jammy:v2` + `scripts/zhuoqun_run.sh`）解除 | `D:\TienKung-Lab-t4-walk\docs\plans\2026-08-13--t4-tienkung-native-walk-plan.md` |

# Decisions

- 2026-08-13: Route W 选择 TienKung 原版 `walk` 配方（gravel + proprio-only + 恒定 AMP），而不是直接 depth Actor 或第二条 Stage E。理由：主线已经在测特权+楼梯；对照实验需要把 T4/AMP 能否学会平地行走从课程/特权里拆出来。
- 2026-08-13: 训练目标机为 zhuoqun，代码面为 git worktree `t4-walk`。nubot 与 `t4_loco_teacher` MDP 冻结。
- 2026-08-13: `t4_walk` 复用 `T4LocoEnv` 加 `policy_role=walk`，不复制第二套 env 类。teacher 角色继续强制 HeightScan。
- 2026-08-13: Route W 不继承 Stage E 的「膝盖接触不终止」规则；`WALK_TERMINATE_CONTACTS` 含 `Shank_.*`。
- 2026-08-13: zhuoqun 当前不能跑 IsaacLab。W2/W3 blocked，不把本机 pytest 当开训证据。

# Spec - T4 TienKung-native Walk AMP 基线（Route W）

> 状态 / Status: user-approved
> Owner: user / agent
> Date: 2026-08-13
> Revision: parallel-control-vs-stage-e
> 来源请求 / Source request: 在 nubot Stage E teacher 之外，另开一条基于 TienKung 原版 walk 配方的 T4 路线，于 zhuoqun 独立 worktree 训练

## 背景

主线 Spec `docs/specs/2026-08-12--t4-unified-depth-locomotion.md` 已经冻结
「特权 teacher + adaptive curriculum → 深度学生蒸馏」。该 lineage
（`stage_e_prov2`）正在 nubot 四卡上跑，本 Spec **不替换、不暂停、不续训** 它。

TienKung-Lab 原版成功配方是另一件事：`walk` 任务在 gravel 上做 PPO+AMP，
Actor 只有本体历史 + 速度命令 + 上一动作，**没有** HeightScan、楼梯课程或深度。
Stage E 把特权扫描和楼梯从第一天绑在一起；若它再死，将无法判断失败来自
T4/AMP 本身还是来自课程/特权。Route W 就是这条对照实验。

## 目标

- 在 T4 27DoF 上复现 TienKung 原版 `walk` 训练配方，任务名 `t4_walk`。
- 交付一条独立 AMP+PPO lineage，证明（或证伪）T4 能在无特权、无楼梯的
  gravel 上学会自然站立/走/侧移/转向。
- 与 Stage E 共享冻结的 27DoF 顺序、66D AMP schema 和已审核 motion；
  不共享 HeightScan、楼梯地形、20s 穿越课程或 AMP 难度衰减。
- 训练只在 zhuoqun 的独立 git worktree 中进行，不占用 nubot、不改
  `t4_loco_teacher` MDP。

## 非目标（Non-goals）

- 不把 `t4_walk` 当作部署策略 `pi_loco`，不导出、不蒸馏。
- 不训练楼梯、强 rough、route，也不把 gravel 课程晋级当成能力声明。
- 不引入 depth CNN、HeightScan 或 teacher privilege。
- 不恢复 walk/run 双策略作为最终产品形态；本路线是对照基线，不是主线替代。
- 不在 nubot 上启动第二条训练，不停止 `stage_e_prov2`。
- 不把 reward、episode length 或 checkpoint 存在当作 walk 能力通过。

## 行为规格

```text
pi_walk (training-only baseline):
    proprio history + local velocity command + previous action
        -> PPO + AMP (constant style weight)
        -> 27DoF action in T4_JOINT_NAMES order
```

- 地形：原版 `GRAVEL_TERRAINS_CFG`（`curriculum=False`，轻量 random_rough）。
- HeightScan：关闭。Actor 观测宽度 = `PROPRIO_FRAME_DIM * 10`，不含 scan/depth。
- 命令：沿用原版 `walk` 的 10s resampling、`rel_standing_envs=0.2`、
  `vx∈[-0.6,1.0]`、`vy∈[-0.5,0.5]`、`yaw∈[-1.57,1.57]`。
- AMP：同一 66D expert（`t4_run` hold out），`amp_reward_coef=0.3` 恒定，
  不做 terrain-difficulty 衰减。
- Gait：`fixed_clock`，与原版 TienKung walk 相同。
- Reward：T4 关节/body 名称映射后的原版 `LiteRewardCfg`（已存在于
  `T4TeacherRewardCfg`），不新增楼梯 traversal 项。

## 与主线的边界

| 项 | Stage E (`t4_loco_teacher`) | Route W (`t4_walk`) |
| --- | --- | --- |
| 服务器 | nubot | zhuoqun |
| Actor 输入 | proprio + 前向 HeightScan | proprio only |
| 地形 | `T4_STAGE_E_TERRAINS_CFG` 楼梯课程 | `GRAVEL_TERRAINS_CFG` |
| 命令时长 | 20s（一次穿越） | 10s（原版 walk） |
| AMP 系数 | 随 terrain difficulty 衰减 | 恒定 0.3 |
| 角色 | `pi_teacher`，将来蒸馏 | 对照基线，不蒸馏 |
| 部署 | 否 | 否 |

共享合同：`T4_JOINT_NAMES`、66D AMP、action scale 0.25、50 Hz / 200 Hz、
已通过 M0 的 17 条 motion。任一共享合同变更必须同时反映到两条路线的
lineage 记录，且不得无条件 resume。

## 成功标准

本 Spec 的「路线落地」成功（本轮 plan/implement 范围）是：

1. `t4_walk` 已注册，观测宽度、无特权泄漏、gravel 无课程、10s resampling
   均有纯 Python 合同测试。
2. zhuoqun 上存在独立 worktree checkout，Isaac 运行时入口可启动该任务。
3. tmux 中已出现 `t4_walk` lineage 的 iteration 日志；与 nubot `t4-stage-e`
   进程互不干扰。

「T4 已经会走」不是本 Spec 本轮成功标准。能力声明仍要求固定 evaluator JSON、
lineage manifest 和连续回放，另开工作项。

## 验证策略

- 第 0 层：无 Isaac 的 schema/role/观测宽度合同。
- 第 1 层：zhuoqun 上 1-env smoke（finite obs/action，任务名 `t4_walk`）。
- 第 2 层：正式多卡训练启动证据（tmux、iteration、log path、commit）。
- 负向：`t4_walk` Actor 不得含 `teacher_scan` / `height_scan` / depth；
  不得注册成 `t4_loco_teacher` 的别名。

## Plan 交接

- Active slice: 注册 `t4_walk`、把 `T4LocoEnv` 从 teacher-only 扩到 proprio-only
  walk 角色、在 zhuoqun worktree 开训。
- Suggested next skill: implement

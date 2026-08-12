# Executable Plan - T4 TienKung-native Walk AMP 基线（Route W）

> Status: active
> Date: 2026-08-13
> Spec: `docs/specs/2026-08-13--t4-tienkung-native-walk.md`
> Branch: `t4-walk`
> Local worktree: `D:\TienKung-Lab-t4-walk`
> Remote worktree: zhuoqun 上独立 checkout（部署时记录绝对路径）
> Planning surface: docs plan
> Training route: TienKung-native proprio-only PPO+AMP on gravel
> Parallel to: `docs/plans/2026-08-12--t4-unified-depth-locomotion-plan.md`（nubot Stage E，本计划不改、不停）

## Objective

把 TienKung 原版 `walk` 配方迁到 T4 27DoF，注册任务 `t4_walk`，并在 zhuoqun
的独立 git worktree 上启动一条对照 lineage。它与 Stage E 共享 27DoF 与 66D AMP，
但 Actor 没有 HeightScan、地形是 gravel 且无楼梯课程。

```text
Stage E (nubot, do not touch):
    proprio + forward HeightScan + stairs curriculum -> pi_teacher

Route W (zhuoqun, this plan):
    proprio history + command + last action
        -> PPO + AMP (constant 0.3)
        -> t4_walk baseline, not a deployment policy
```

## Active Slice

在 `t4-walk` worktree 中落地 `t4_walk` 任务合同（schema + cfg + env 角色扩展 +
注册 + 纯 Python 测试），部署到 zhuoqun 独立 worktree，并用 tmux 启动训练。

## Non-goals

- 不停止、不修改、不续训 nubot 上的 `stage_e_prov2` / `t4_loco_teacher`。
- 不把 `t4_walk` 当 `pi_loco`，不做蒸馏、导出、depth CNN、HeightScan。
- 不训练楼梯 / 强 rough / route。
- 不把本轮训练曲线当作「T4 会走」的能力声明。
- 不把 Route W 的 commits 直接推到 nubot 的 `t4-train` checkout。

## Success Criteria

1. `policy_role="walk"` 的 Actor 观测宽度等于 `PROPRIO_FRAME_DIM * 10`，合同测试断言
   其字段列表不含 `teacher_scan` / `height_scan` / depth。
2. 任务 `t4_walk` 已注册；cfg 使用 `GRAVEL_TERRAINS_CFG`（`curriculum=False`）和
   10s command resampling。
3. `T4LocoEnv` 在 `walk` 角色下不创建、不拼接 HeightScan；`teacher` 角色行为不变。
4. zhuoqun 上存在独立 worktree，Isaac 入口能启动 `--task t4_walk`。
5. tmux 日志出现 `Learning iteration` 且实验名为 `t4_walk`。

## Verification Path

```text
纯 Python 合同（walk role / 观测宽度 / 无特权泄漏）
  -> 本地 pytest 与现有 T4 观测/课程合同回归
  -> zhuoqun Isaac 运行时发现
  -> 独立 worktree checkout + 1-env smoke
  -> tmux 多卡/单卡正式启动
  -> 日志出现 iteration（本轮截止）
```

能力 evaluator 不在本轮。

### Verification Path Status

`runnable`

代码与第 0 层测试可在 Windows 本机完成。zhuoqun 的 Isaac 绝对路径在 W2 发现；
若发现失败则 verification path 转为 `blocked`，不得假装已开训。

## Required Capabilities

- 本地 git worktree（已建：`D:\TienKung-Lab-t4-walk`，branch `t4-walk`）。
- zhuoqun SSH：`zhuoqun@100.95.109.48`，四 GPU，tmux。
- zhuoqun 上可用的 Isaac Sim / IsaacLab 运行时（W2 发现，写入 `scripts/zhuoqun_run.sh`）。
- 与 Stage E 相同的 provisional/formal AMP expert 文件；若 zhuoqun 尚无 expert，
  从 nubot 或本地 `artifacts/amp_expert_provisional` / `legged_lab/envs/t4/datasets/motion_amp_expert`
  拷贝，并用 `T4_AMP_EXPERT_DIR` 指向。

## Fallback Evidence

- 若 zhuoqun 暂时没有 IsaacLab：W1 仍可完成并提交；W2/W3 标 blocked，不得把
  本机 pytest 当作已开训。
- 若 AMP expert 目录缺失：允许用 provisional expert + 环境变量，但 lineage
  记录必须写 `provisional`。
- 无 fallback 可以替代「tmux 里出现 iteration」这一启动证据。

## Final Integration Claim

`final_integration_claim`: 仓库中存在可注册的 TienKung-native T4 walk 任务
`t4_walk`（proprio-only Actor、gravel、恒定 AMP），并在 zhuoqun 独立 worktree
上以独立 lineage 开始训练；它不替代 Stage E，也不是部署策略。

## 冻结合同（本路线）

- 任务名：`t4_walk`；`policy_role`: `walk`。
- Actor 观测：`WALK_ACTOR_OBS_DIM = PROPRIO_FRAME_DIM * PROPRIO_HISTORY_LENGTH`。
- 地形：`GRAVEL_TERRAINS_CFG`，`curriculum=False`。
- `resampling_time_range=(10.0, 10.0)`。
- `amp_terrain_schedule.enable=False`；`amp_reward_coef=0.3`。
- `gait.mode="fixed_clock"`。
- `experiment_name="t4_walk"`；checkpoint 目录 `logs/t4_walk/`。
- 共享：`T4_JOINT_NAMES`、66D AMP、`t4_run` hold out、action scale 0.25。
- 改上述任一项即新 lineage。

## 工作项

- [x] W0：对照基线已存在（Stage E 在 nubot 跑；T4/AMP 合同已冻结）
  - acceptance_criteria: 主线 Spec/Plan 仍 active；`t4-walk` worktree 从
    `5c70898` 分出，不共享 nubot checkout。
  - verification_commands: `git -C D:\TienKung-Lab-t4-walk rev-parse --abbrev-ref HEAD`; `git -C D:\TienKung-Lab-t4-walk rev-parse HEAD`
  - success_definition: 两条路线的代码面和训练面已经物理隔离。

- [x] W1：注册 `t4_walk` 并冻结 proprio-only 合同
  - acceptance_criteria: `POLICY_ROLES` 含 `walk`；walk Actor 宽度无 scan；
    `assert_no_privilege_leakage("walk", ["teacher_scan"])` 失败；`t4_walk`
    已 `task_registry.register`；`T4LocoEnv` 在 walk 角色跳过 HeightScan，
    teacher 角色仍强制扫描；AST 合同锁定 gravel / 10s / 无 scan / shank 终止；
    现有观测与课程合同不回归。
  - verification_commands: `python -m pytest tests/test_t4_observation_contracts.py tests/test_t4_walk_contracts.py tests/test_t4_terrain_curriculum.py tests/test_t4_asset_migration.py -q`
  - success_definition: 无 Isaac 即可证明 walk 不是 teacher 的别名，且 teacher 合同未破。

- [ ] W2：zhuoqun 运行时发现 + 独立 worktree 部署（当前，blocked）
  - acceptance_criteria: 记录 Isaac Sim `python.sh`、IsaacLab source、conda lib、
    GPU 列表、绝对 checkout；`scripts/zhuoqun_run.sh` 能启动 repo 脚本；
    远程 worktree 在 `t4-walk` commit 上，与任何 nubot 路径无关。
  - verification_commands: `ssh zhuoqun@100.95.109.48 'nvidia-smi -L; ls -d $HOME/isaac-sim* $HOME/IsaacLab 2>/dev/null; git -C <worktree> rev-parse --abbrev-ref HEAD'`
  - success_definition: 远程有可复述的启动入口和隔离 checkout。
  - blocker: 2026-08-13 探测：无 Isaac Sim `python.sh`、无 IsaacLab source；四卡被
    MjLab `t4_stair_traversal` 占用（约 4–5 GiB/卡，50–60% util）。代码 worktree
    仍应落地；在用户提供 Isaac 运行时或明确允许安装之前不得开训。

- [ ] W3：tmux 启动 `t4_walk` lineage（下一步）
  - acceptance_criteria: tmux session 存在；日志出现 `Learning iteration`；
    `experiment_name=t4_walk`；lineage 名、commit、GPU、env 数写入
    `.harness/state.md`。
  - verification_commands: `ssh zhuoqun@100.95.109.48 'tmux ls; tail -n 80 <log>'`
  - success_definition: 对照训练已经在跑，而不是只存在配置。

## Commit Units

1. `feat(t4): 增加 TienKung-native t4_walk 对照任务`
   - work items: W1（及本 plan/spec/recovery）。
   - 前置：实现完成 + review 无 Critical + 本机 pytest PASS。
2. 训练启动本身不入库；W2/W3 只更新 `.harness/state.md` 与远程 checkout。

## Known Risks / Blockers

- zhuoqun 的 Isaac Sim 路径/版本可能与 nubot 不同；W2 必须实测，不能抄
  `scripts/nubot_run.sh` 的绝对路径。
- zhuoqun 可能没有 AMP expert 文件；启动前必须确认 `amp_motion_files` 存在。
- 在 `T4LocoEnv` 上加 walk 角色若误改 teacher 观测拼接，会污染 nubot 主线；
  回归 `test_t4_observation_contracts.py` 是硬门。
- gravel `curriculum=False` 时仍可能有轻微 rough；这是原版 walk 行为，不是 bug。
- 本机 Windows 不能跑 Isaac smoke；启动证据只认 zhuoqun 日志。

## Recovery Protocol

```bash
git -C D:\TienKung-Lab-t4-walk status --short --branch
sed -n "1,80p" D:\TienKung-Lab-t4-walk\.harness\state.md
rg -n "^[-] \\[[ x]\\]" D:\TienKung-Lab-t4-walk\docs\plans\2026-08-13--t4-tienkung-native-walk-plan.md
ssh zhuoqun@100.95.109.48 "tmux ls"
```

nubot 的 `t4-stage-e` 不属于本 plan 的恢复范围。

## Next Skill

`harness-builder` 或等待用户提供 zhuoqun Isaac 运行时后再 `implement` W2/W3。

Reason: W1 代码与合同已落地；W2/W3 被 zhuoqun 缺少 IsaacLab 挡住，不能把本机 pytest 当成已开训。

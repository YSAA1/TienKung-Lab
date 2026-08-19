# Executable Plan - T4 梅花桩双 teacher 与跨栏 0.30 学生微调

> **Status: archived（非权威）** — 双 teacher / S1d 路线已否。现行梅花桩执行面：`docs/plans/2026-08-18--t4-sparse-terrain-index-fix-plan.md`。入口：`docs/README.md`。

> Status: superseded
> Date: 2026-08-15
> Updated: 2026-08-18 归档；中间曾改到 rollback plan，现一并作废
> Spec: `docs/specs/2026-08-15--t4-stepping-stones-and-hurdle-stable.md`（user-approved）
> Branch: `t4-train`
> Planning surface: docs plan
> GPU: nubot 2（T-compat）+ nubot 2（T-paper）+ 本机 1（旧学生 0.30 FT）；zhuoqun 翻箱不动

## Objective

把 LightLP 式稀疏落脚（桩阵 + 踏石）并入新的 Stage E teacher 课表，并并行稳定 0.30 跨栏：

```text
新地形 + illegal-footstep + velocity-slack + 碰杆惩罚
    -> T-compat 1155D 从零（nubot 2 卡）
    -> T-paper 1157D 从零（nubot 2 卡，Actor +接触）
    -> 旧 depth student 本机 PPO FT 0.30（不蒸旧老师）
```

旧 `t4_loco_teacher` 默认课表不改。

## Active Slice

S1d：稳定组 vs 对照组。**已失败并被 2026-08-17 回退计划取代。** 10k 时全局走跑弱于 Stage E，踏石/圆桩均未成功；不要续训这两条。

## Non-goals

- 作废 `stage_s_head35`；改冻结 1155D 默认常量。
- 无老师学生从零踩桩；窄板/绕桩/翻箱。
- 杆高 > 0.35；strict 500-trial evaluator（后续）。
- 打断 zhuoqun 翻箱。
- 以 reward 代替行为验收。

## Success Criteria

1. sparse v2 合同测试绿；T-compat 1155D、T-paper 1157D。
2. nubot 两条 2-GPU 训练从零启动，真实 PPO iteration 持续、无 NaN/OOM。
3. TensorBoard 能分别观察 stepping stones / raised pillars，以及 easy/mid/hard 的 `reach_1m`、`reach_2m`、`reach_4m`、strict success、fall、pit-fall 和 progress。
4. 训练曲线只作监控，不据此声明梅花桩能力；10k 起用 fixed evaluator，最终按每种地形与难度独立验收。

## Verification Path

```text
sparse v2 合同 pytest（本机 + nubot）
  -> 本机 32-env Isaac smoke，检查真实 TensorBoard tags
  -> 旧 checkpoint 放入新 d=0.5 几何，验证 pit-fall 分类
  -> nubot 2+2 从零开训
  -> 1k/5k optimization + 分段 reach 趋势检查
  -> 10k/20k fixed evaluator + 连续回放
  -> 40k final evaluator
```

### Verification Path Status

`running`。v2 @2.1k：踏石 `reach_2m≈0.79` / success≈0.33 / progress≈4.5 m；圆桩 `reach_2m≈0` / fall≈0.65 / progress≈1.0 m。本机合同 `47 passed`（本切片相关）。T-compat v2 留在 CUDA 0,1；v3 圆桩探针用 CUDA 2,3。

## Required Capabilities

- nubot 4 卡空闲中的 4 张：0,1 与 2,3 两条 torchrun。
- 本机 1 卡 Isaac / 现有 student 训练环境。
- tmux；nubot 代码用 git bundle 如 fetch 失败。

## Fallback Evidence

- 本机无 Isaac：S0 只跑纯 Python 合同；smoke 记为 deferred。
- nubot OOM：每卡 env 2048→1024。
- 本机 student FT 入口若深度相机起不来：先只开两条 teacher，FT 记 blocker，不挡 S0 老师线。

## Final integration claim

`final_integration_claim`: sparse v2 交付了可训练的双 teacher、正确的稀疏几何/终止/课程和可区分学习阶段的监控指标，并已在 nubot 2+2 从零运行。不声明当前 checkpoint 已具备过桩能力。

## 工作项

- [x] 阶段 S0：旧版合同、地形、双任务、开训
  - verdict: 训练运行成功但行为失败；旧 25k T-compat 4/16、T-paper 2/16，不能作为能力证据
- [x] 阶段 S1：sparse v2 合同与从零重启
  - acceptance_criteria: 0.16/0.18 m 起步、固定 9×9、40% sparse、掉坑终止、严格 sparse 课程、分 terrain/band/reach 指标；本机 smoke 与 nubot 2+2 启动
  - verification_commands: `python -m pytest tests/test_t4_sparse_teacher_mujoco.py tests/test_t4_stepping_stone_contracts.py tests/test_t4_sparse_reward_contracts.py tests/test_t4_sparse_monitor_contract.py tests/test_t4_terrain_curriculum.py tests/test_t4_observation_contracts.py tests/test_t4_sparse_evaluator_contract.py -q`
  - evidence: 本机 51 passed；nubot 49 passed（仅缺可选 `mujoco` 包）；PPO iter 与新 tags 已出现
- [x] 阶段 S1c：圆桩 v3 探针
  - verdict: 几何/正奖励单独不够（用户目视仍挂）；转入步态/AMP/掉坑对照
- [x] 阶段 S1d：稳定组 vs 对照组
  - verdict: 失败。踏石 11.6k 才回到 v2 3.8k 的 `reach_2m=0.80`（慢约 3 倍）；圆桩稳定组 0.019、对照组≈0。全局 reward 52–62 低于 Stage E。执行面移交 2026-08-17 回退计划
- [ ] 阶段 S2：中期 gate（下一步）
  - 1k: metrics finite、episode count 持续增加、无 NaN/OOM/command collapse
  - 5k: T-compat v2 踏石 easy `reach_2m` 保持上行；v3 圆桩 easy `reach_2m` 离开 0
  - 若圆桩 5k 仍为 0，先诊断而不是盲跑 40k
  - 10k/20k: stepping stones 与 raised pillars 分别跑 fixed d=0.0/0.5 evaluator（至少 16 episodes）和连续回放
- [ ] 阶段 S3：最终行为 evaluator
  - acceptance_criteria: 每个 teacher、每种 sparse terrain、d=0.0/0.5/1.0 独立 JSON；至少 100 episodes/格；另保留连续回放
  - target_gate: easy ≥90%，mid ≥80%，hard ≥60%，pit-fall ≤5%；任何一格未过都不能用全局平均掩盖
  - success_definition: fixed evaluator + 回放共同支持能力，不以 reward、terrain level 或 checkpoint 存在代替

## Commit units

1. `S0-contracts`：布局、奖励、任务注册、测试（review 后）。
2. 训练产物不进 git。

## Known risks / blockers

- Scene 加足底 RayCaster 可能增显存。
- 本机 RAM 曾因 play+train OOM；FT 单独 tmux，不并行批量渲染。
- 翻箱与本工作面并行，禁止改 zhuoqun checkout。

## Next skill

本计划已 superseded。现行梅花桩执行面：`docs/plans/2026-08-18--t4-sparse-terrain-index-fix-plan.md`。

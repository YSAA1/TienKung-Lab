# Executable Plan - S11 sparse teacher MDP 最小修复包

> **Status: archived（非权威）** — living 梅花桩: `docs/plans/2026-08-22--t4-sparse-s12-rim-yaw-student-plan.md`。入口: `docs/README.md`。
>
> Status: superseded
> Date: 2026-08-21
> Superseded-by: `docs/plans/2026-08-22--t4-sparse-s12-rim-yaw-student-plan.md`
> Spec: `docs/specs/2026-08-15--t4-stepping-stones-and-hurdle-stable.md`（目标规格；双老师路线已废）
> 证据: S10 固定评估与 `.harness/decisions.md` 2026-08-21 条目（根目录 `findings.md` 已删）
> 取代: `docs/archive/plans/2026-08-19--t4-sparse-easy-geometry-s6-plan.md`（S6–S10 物理/课表基线已落地；S10 保留为对照 lineage）
> Branch: `t4-train`
> Planning surface: docs plan
> GPU: nubot 四卡。S10 作基线保留，不热补。zhuoqun 翻箱不动。Stage E 1155D 不覆盖。

## Objective

S10 在 easy / 训练均值难度上已经会朝前走过踏石和圆桩，但 hard d=1 仍大量真实 Trunk/Shank 撞击，且训练命令/课表采样把 hard 曝光和有效直行样本稀释。本切片落地 **最小 MDP 修复包** 并开 **S11 冷启动**，不改终止阈值，不搬 Go2 PIE 数值。

## Active slice

实现 sparse teacher 的课表/命令修复 + 纯监控修复，并在 nubot 开冷启动 lineage `t_sparse_lightlp_s11_vx2`。用户 2026-08-21 覆盖：不加 `body_orientation_l2` / `upright_orientation`；sparse 全局 `vx` 上限 2.0。

行为修复：

1. 保留 10% random level reset，`random_level_reset_max_level=None`，覆盖 level 0–9。
2. 仅对 `stepping_stones` / `raised_pillars` 使用 terrain-aware command：无 standing；`vx ∈ [0.6, 2.0]`；`vy=0`；80% 精确 `wz=0`；20% gentle yaw `[-0.3, 0.3]`；禁止 sparse 上倒退、侧移和 full heading。非 sparse 地形 `vx ∈ [-0.6, 2.0]`，其余全向合同不变。
3. sparse **不加** orientation 塑形：`body_orientation_l2=0`、`upright_orientation=0`。Stage E 的 ori −2 不动。

监控修复（不得改变 PPO 梯度）：

- `promotion_rate=move_up`；旧 `success_rate=move_up & timed_out` 降为 `timeout_success_rate`，并保留 deprecated 别名。
- 分开 horizon / OOB / joint / accel / torso / fall_over。
- episode-weighted sum/count 跨 rank 归约后再写 TB。
- 增加 random-reset 前后 level、command bins、reset reason × terrain、per-terrain level histogram。

## Non-goals

- 不恢复 sparse `body_orientation_l2=-2` / `upright_orientation=+1`（用户明确先不加）。
- Stage E 全局 `vx` 仍为 `1.0`。S11 sparse 上限为 `2.0`，不再升到 3.0/4.0。
- 根加速度硬门保持 `40 m/s²`，1 s warmup 不变。
- Trunk 硬接触门保持 `1 N`；硬终止仅 `Trunk`。
- 不改 20 s command resampling、sparse AMP/gait 置零、reset 随机化、pit-fall 合同。
- 不把 `parkour_mjlab` 的网络、四足 gait、base height 或 10 N 接触阈值搬到 T4。
- 不热补/resume S10，不用旧 lineage 名称。
- 不蒸学生，不改 Stage E 1155D，不动 zhuoqun。
- 不跑 `current/mild/nominal` spawn reset ablation（S11 10k hard gate 失败后的下一调查分支）。
- 不用 TB / reward / checkpoint 存在宣称梅花桩能力。

## Success criteria

1. 合同测试证明：sparse command 采样、full-level random reset、orientation 权重、promotion vs timeout-success、跨 rank 加权归约。
2. play / eval 固定 `command_vx` 时 **不** 被 terrain-aware sampler 覆盖。
3. Stage E `t4_loco_teacher` 命令、终止、奖励权重不变。
4. 相关 pytest 绿。本机无 Isaac，不把 GPU smoke 当本阶段完成条件。
5. 开训阶段：新 logdir、`--resume` 关、tmux、记录 SHA / dirty / 启动命令。约 10k 用同一 fixed evaluator 对照 S10 `model_32500` 的 d=0 / 0.39 / 1.0 与 `vx=0.7/1.0`。

## Verification path

```text
python -m pytest tests/test_t4_sparse_reward_contracts.py tests/test_t4_sparse_monitor_contract.py tests/test_t4_sparse_command_contract.py tests/test_t4_terrain_curriculum.py tests/test_distributed_log_reduce.py tests/test_t4_sparse_evaluator_contract.py -q
  -> review（多视角对抗审查）
  -> nubot 隔离 worktree 冷启动 S11（tmux）
  -> ~10k 同一 fixed evaluator 对照 S10
```

### Verification path status

`runnable` 对本机合同测试与文档。GPU 开训/evaluator 在 nubot 上 runnable，本机 Windows 无 Isaac Docker，开训证据走 nubot tmux + JSON。

## Required capabilities

- 本机 pytest（无 Isaac）。
- nubot 4×GPU + `scripts/nubot_run.sh` + tmux（阶段 2）。
- S10 logdir / `model_32500` 及后续 ckpt 作对照，不覆盖。

## Fallback evidence

- 本机无 Isaac：命令/课表/监控代数 + 源码合同。64-env smoke 放到 nubot 开训后看 log 无即崩。
- 跨 rank 归约在本机用 `world_size=1` 加权均值单测；四卡 all-reduce 用开训后 TB 是否不再只等于 rank0 的 1024 env 作运行时确认。
- S11 若 10k hard gate 无明确提升：停长训，转 spawn-reset / support-relative base-height 调查，不先松 accel/torso。

## Final integration claim

`final_integration_claim`: S11 是 S10 之后的冷启动 MDP 修复包（full-level random reset + sparse 正向命令 + `vx max=2.0` + 监控语义，不加 orientation 塑形）。它不声明 hard d=1 已会走。能力声明仍要 evaluator JSON、lineage manifest 和连续回放。

## 工作项

- [ ] 阶段 1：S11 MDP 合同与最小实现（当前）
  - acceptance_criteria: sparse 仅踏石/圆桩采样 `vx∈[0.6,2.0]`、`vy=0`、80/20 yaw；非 sparse `vx∈[-0.6,2.0]`；`random_level_reset_max_level is None`；sparse ori/upright 权重均为 0；TB 有 `promotion_rate`；eval/play 固定命令不被覆盖；相邻 sparse pytest 绿
  - verification_commands: `python -m pytest tests/test_t4_sparse_reward_contracts.py tests/test_t4_sparse_monitor_contract.py tests/test_t4_sparse_command_contract.py tests/test_t4_terrain_curriculum.py tests/test_distributed_log_reduce.py tests/test_t4_sparse_evaluator_contract.py -q`
  - success_definition: 代码合同与 S11 配方一致，训练行为只改上述三件，监控不进梯度
- [ ] 阶段 2：nubot 冷启动 S11 并对 10k 做 hard gate
  - acceptance_criteria: 新 worktree/logdir；resume false；不停止 S10 产物；约 10k 用 `eval_t4_hurdle.py` 对踏石/圆桩 d=0、0.39、1.0 和 `vx=0.7/1.0` 出 JSON
  - verification_commands: nubot `tmux ls`；`params/agent.yaml` `resume: false`；`artifacts/eval/s11_model10000_*`
  - success_definition: 新 lineage 在跑，且 10k 有与 S10 同口径的固定评估，而不是 TB success
- [x] 阶段 0：S10 诊断基线
  - acceptance_criteria: hard d=1 固定直行仍失败；只改 random cap 被否决；最小修复包已写进 findings
  - verification_commands: `artifacts/eval/s10_model30000_fixed/`；`artifacts/diagnostics/s10_model32500_hard/`；`.harness/decisions.md` 2026-08-21
  - success_definition: 根因与不改项已冻结，本切片不再重复调查

## Commit units

1. `sparse-s11-mdp-repair-contract`：阶段 1 代码、测试、梅花桩计划与 recovery 同步。
2. 训练产物不进 git。

提交前置：实现完成 + review 无 Critical + 对应 pytest 绿。不在 review 前做正式里程碑 commit。

## Known risks / blockers

- S10 可能仍占 nubot 四卡。阶段 2 必须 fresh 查询；未结束则等自然退出或开隔离 worktree，不 SIGINT 抢卡，除非用户明确要求。
- UniformVelocityCommand 在 `heading_command=True` 时每步改 `wz`。sparse overlay 必须在 `compute()` 之后强制写回，并把 sparse 的 heading 误差清零，否则 80% 精确直行会被 heading PD 破坏。
- 固定命令 eval 若忘记关 terrain-aware，会把 `vx=0.7` 盖成 `[0.6,1.0]` 随机。必须同时用显式 flag 和“命令范围已钉死则跳过”双保险。
- 跨 rank 归约只影响 TB；实现错误不得 all-reduce 奖励或观测。
- 一次改了课表曝光、命令和姿态三项，S11 只能声明组合效果，不能做单变量归因。这是用户授权的最小包，不是偷偷加第四项的许可。

## Next skill

`implement`

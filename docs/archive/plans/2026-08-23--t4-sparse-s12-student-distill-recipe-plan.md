# Executable Plan - S12 深度学生蒸馏配方修复（mix=0 / 固定 LR / recon 投影）

> **Status: archived（非权威）** — living work surface: `.harness/work_index.md`。

> Status: superseded
> Date: 2026-08-23
> Superseded-by: `docs/archive/plans/2026-08-25--t4-sparse-s12-rtx-deploy-distill-plan.md`（部署向 RTX 蒸馏；不续 D1/D3）
> Parent: `docs/archive/plans/2026-08-22--t4-sparse-s12-rim-yaw-student-plan.md`
> Predecessor: `docs/archive/plans/2026-08-23--t4-sparse-s12-student-lightlp-distill-cost-plan.md`（墙钟已交付）
> Spec: `docs/specs/2026-08-12--t4-unified-depth-locomotion.md`
> Reference: `docs/research/2608.02653v1/auto/2608.02653v1.md`（LightLP §IV-B / §VI）；现状 `.harness/state.md`
> Branch: `t4-train`
> Planning surface: docs plan

## Objective

把现行 `s12_lightlp_raycast` 的蒸馏配方改成论文对齐的 **纯学生 DAgger+PPO**，并另开 lineage。不热补 7740，不重训老师，不把 Actor 接触位当成作弊拆掉。

要修的三件事：

1. **PPO 开着时 `teacher_mix=0`。** 论文 §VI 是在学生自己访问的状态 `d^{π_θ}` 上用老师动作当标签；每步掷硬币掺老师会让回报含救援，TB 也被抬高。
2. **蒸馏期不用老师那套 `desired_kl=0.01` 自适应 LR。** 该门把 BC 当过大更新，iter 50 就把 LR 钉在 `1e-5`。改为固定 LR，用短健康门在 `1e-4` 与 `3e-5` 之间选，**不直接钉 `3e-4`**（开局 KL≈47）。
3. **真正接上 recon 冲突投影。** `_project_auxiliary_gradient` / `_limit_auxiliary_gradient_norm` 已存在但未进入 `update()`；现在 `recon_coef=1.0` 直接加进总损失。

## 背景证据（已测，不要重测墙钟 / v3 坍塌）

`s12_lightlp_raycast` @~7740（后窗只读到 ~8712）：不是 v3@3305 那种 PG/behavior 爆炸。课表 4k 窗 2.24 → ~1.71；踏石/圆桩继续掉，flat 稳住；仍有约 21% 老师步。`behavior` 4k→现在大约 0.28→0.26，克隆基本停住。LR 从 iter 50 钉在地板。

论文 §IV-B：**Actor 有 privileged contact flag，Critic 另有足底 scan。** S12 1937D 按这个做。`TEACHER_FORBIDDEN_PRIVILEGE_FIELDS` 的 `contact_truth` 禁的是导出学生 / Stage E 泄漏，不是稀疏老师 Actor。本计划不去特权老师。

## Active slice

阶段 D3b（修复后重开）：崩掉的 `s12_gru_depth_noise_ft` 已杀，不续 `model_10500`。新 lineage `s12_gru_ft_critic_bc` 仍从 D1 `model_10000.pt` 热启。配方：`critic_warmup_iters=200` 期间冻 actor 只训 critic、`behavior_coef=0.5` 不降、`schedule=fixed`、`pg_coef=1`、mix=0、深度噪声开、1000 iter（计数从 0 重计）。不是过桩。`nan_guard` 仍只扫最后 minibatch。

## Non-goals

- 不热补 `s12_lightlp_raycast` 的 7740/8712 或任何中间 ckpt 上改 LR 续训。
- 不 resume v3 `model_3500/4000`、nansync、rtx167、CNN `model_3000`（形状已是 MLP）。
- **不**开老师去接触 lineage；不把「Actor 有 `feet_contact`」判成老师不合格。接触消融只读、不阻塞本计划。
- 不把老师 12 条固定 evaluator 当开训硬门（ungated 过程债另记；不挡配方修复）。
- 不改 S12 老师 MDP、地形、命令、终止；不动 `model_21500.pt`。
- 不把蒸馏期「一直 mix>0」或「BC 退火到 0.25 同时开 PPO」当修法。Ross β 只覆盖前 1000 iter 且当时 `pg_coef=0`。
- 不直接固定 `learning_rate=3e-4`。
- 不砍 env、不换回 RTX 相机、不动 zhuoqun、不在 nubot 四卡开 GUI play。
- 本计划完成 ≠ 梅花桩能力验收。阶段 5 过桩仍另开。

## Success criteria

1. 蒸馏默认 `pg_coef=0`。`teacher_mix` 只允许在 `pg_coef=0` 时 >0（Ross 前 1000 iter 1→0）；`pg_coef>0` 时 mix 必须为 0。FT 仍 `pg_coef=1`、`mix=0`，但 **BC 留 0.5**，且前 200 iter 冻 actor 只训 critic。
2. 蒸馏 cfg `schedule="fixed"`。FT 也用 `schedule="fixed"`（不再对 DAgger 学生开 adaptive KL）。
3. `behavior_coef=behavior_coef_end=1.0`（不退火）；蒸馏 14000 步内 BC 不降。
4. 每个 minibatch：control（BC+PG+value）与 recon 分开反传；recon 对共享 encoder 的梯度先冲突投影，再按 `max_recon_grad_ratio` 限幅；TB 记录 `recon_control_cosine`、两边范数。
5. 合同测试覆盖 mix、fixed LR、投影被调用、cfg 数值；本机 pytest 绿。
6. 新 lineage：新 tmux / 新 logdir / 新 `run_name`；`student_lineage.json` 写明 `pg_coef=0`、mix 1→0（1000 iter）、fixed LR、warm-start null。
7. 200 iter 健康门：LR 不是地板；`pg_coef=0`；mix 仍在退火（约 0.8，不是 0）；无 NaN/OOM；`Loss/behavior` 相对 iter 0 下降。前 1000 步 TB 含老师执行，不当学生能力。过门再训到 14000。
8. **本计划完成 ≠ 会过桩。** 能力仍只认 student-only evaluator + 回放。

## Verification path

```text
# 本机（阶段 1，不需要 GPU）
D:\anaconda\envs\pytorch\python.exe -m pytest tests/test_safe_recurrent_distillation.py tests/test_t4_sparse_depth_student_gru_contract.py -q

# 阶段 2：用户确认后停训，再 student-only 评估（mix 必须为 0，即任务默认执行学生动作）
# nubot tmux kill t4-s12-lightlp-raycast 前先确认最新 model_*.pt 已落盘
bash scripts/nubot_run.sh legged_lab/scripts/eval_t4_hurdle.py --task t4_loco_sparse_depth_student \
  --load_run 2026-08-23_14-26-19_s12_lightlp_raycast --checkpoint model_<pick>.pt \
  --num_envs 32 --headless

# 阶段 3：新 lineage（示例；LR 以健康门为准）
bash scripts/nubot_run.sh -m torch.distributed.run --nnodes=1 --nproc_per_node=4 \
  legged_lab/scripts/train_t4_sparse_depth_student.py \
  --teacher_checkpoint <model_21500.pt> --allow_ungated_teacher \
  --task_num_envs 1024 --distributed --headless --seed 42 \
  --run_name s12_lightlp_dagger_only \
  [--student_warmstart_checkpoint <picked.pt>]
```

### Verification path status

`runnable` —— 阶段 1 本机 pytest 可跑。mix0 已结束，四卡应空闲；阶段 D1 开训前仍须再确认 GPU 空闲，且不得与 1024-env probe 并发（会 OOM）。

## Required capabilities

本机 `D:\anaconda\envs\pytorch\python.exe` 跑 pytest。nubot 4×RTX 4090 + `scripts/nubot_run.sh` + tmux。远端 worktree：`/home/nubot/phn_ws/t4_train/TienKung-Lab-s12-gru-ppo`。老师 ckpt 仍走 `--allow_ungated_teacher`。

## Fallback evidence

- 200 iter 健康门：若 `kl_mean` 近窗 p50 **> 1.0** 或出现非有限 loss，停掉，改 `1e-4`→`3e-5` 重开同一配方，不回 `3e-4`。
- 若 `behavior` 200 步几乎不降且 KL 近窗 p50 **< 0.002**，视为 LR 过小，只允许升到 `1e-4`（若已是 `3e-5`），仍不升到 `3e-4`。
- student-only 评估若所有 raycast ckpt 都明显劣于「从零 + 新配方」，warm-start 选 `null`。
- 接触消融若日后证明老师动作对 `feet_contact` 极敏感，**另开计划**去特权老师；不在本切片改 1937D。

## Final integration claim

`final_integration_claim`: 交付一条纯 DAgger 的 S12 学生 lineage（D1 停在用户授权的 `model_10000.pt`），并用修复后的配方重开 §VI 噪声 FT（`s12_gru_ft_critic_bc`：critic warmup + 冻 actor + BC=0.5 + fixed LR）。不声明过桩。不从 mix0 ckpt FT。不续崩掉的 `model_10500`。

## 工作项

- [x] 阶段 0：失败形状与论文对齐（基线）
  - acceptance_criteria: 已区分 v3 坍塌 vs 现行缓降；KL 自适应过紧；mix 抬高 TB；论文 Actor 接触 flag 合法；recon 投影未接线。见 `.harness/state.md` / `.harness/decisions.md`
  - verification_commands: 只读 TB/日志（`artifacts/tmp_s12_lightlp_health2.py`）；论文 §IV-B；`safe_recurrent_distillation.py` 未调用投影函数
  - success_definition: 修复方向可证伪，不再把「拆老师接触」当第一刀

- [x] 阶段 1：配方代码与合同（当前）
  - acceptance_criteria:
    - `T4SparseDepthDistillationAlgCfg`：`pg_coef=0`；`teacher_mix=1.0`、`teacher_mix_end=0`、`teacher_mix_decay_iters=1000`；`critic_warmup_iters=0`；`schedule="fixed"`；`learning_rate=1e-4`；`behavior_coef=behavior_coef_end=1.0`、`behavior_coef_decay_iters=0`；`max_iterations=14000`；`run_name=s12_lightlp_dagger_only`
    - `T4SparseDepthStudentFtAlgCfg`：`schedule="adaptive"`、`teacher_mix=0`、`behavior_coef=0`、`pg_coef=1.0`（FT 才开 KL 门）
    - `SafeRecurrentDistillation`：`pg_coef>0` 时若 mix≠0 直接 raise；`update()` 对 shared encoder 做 recon 投影+限幅并写入 loss_dict；4 卡时先 all-reduce 控制/recon 克隆再投影，禁止投影后再 `_reduce_gradients()`
    - 多卡：control/recon 克隆先 `_all_reduce_tensors`，再投影/限幅；投影路径不再事后 `_reduce_gradients()`
    - 测试：mix 合同、fixed 时 LR 不随 KL 变、冲突 recon 梯度被投影、多卡 reduce-before-project 顺序、cfg 源码/对象断言
  - verification_commands: `D:\anaconda\envs\pytorch\python.exe -m pytest tests/test_safe_recurrent_distillation.py tests/test_t4_sparse_depth_student_gru_contract.py -q`
  - success_definition: 不启动 GPU 也能证明新配方与旧 mix/adaptive/直接加 recon 不同

- [x] 阶段 2：停旧 run + 开新 lineage（用户覆盖：跳过 student-only 评估，warm-start=null；已停 `t4-s12-lightlp-raycast`，末 ckpt `model_14000.pt`）
  - acceptance_criteria: 下一份 `model_*.pt` 落盘后停 `t4-s12-lightlp-raycast`；保留 logdir。对 `model_2000` / `4000` / `7500` / 最新做 **student-only** 固定评估（踏石+圆桩至少 d=0，`vx=0.8`，32 局）。只加载 MLP/GRU/actor/recon；重置 critic/Adam/std/调度。禁止 CNN v3 ckpt。
  - verification_commands: nubot `eval_t4_hurdle.py --task t4_loco_sparse_depth_student`；JSON 写入 `artifacts/eval/s12_lightlp_raycast_student_only/`
  - success_definition: 有一份书面选择（某个 raycast ckpt 或从零），不是看训练 TB

- [x] 阶段 3：mix0 lineage 已跑完 14k（能力失败，不 FT）
  - acceptance_criteria: mix=0、LR=`1e-4`、无 NaN/OOM。hard 踏石峰值 0.43 vs 老师 0.83，终点课表 1.55。判定为克隆未完成就开 PPO。
  - success_definition: 配方健康门过了，过桩门没过；下一刀改蒸馏而不是 FT

- [x] 阶段 D1：纯 DAgger 新 lineage（用户覆盖：10k 即停，不跑满 14k）
  - acceptance_criteria: 新 tmux `t4-s12-lightlp-dagger`；`pg_coef=0`；BC=1 不退火；mix 仅前 1000 iter 1→0；无 student warm-start。用户授权以 `model_10000.pt` 为蒸馏终点。
  - verification_commands: 远端 log/TB；`student_lineage.json`；ckpt `model_10000.pt` 31MB @ 17:57
  - success_definition: 蒸馏段不再用 PPO 主导；mix 在 pg=0 时才允许 >0。10k 后转 FT 是用户覆盖，不是能力门。

- [x] 阶段 4：D1 蒸馏到 14000（健康监控）——用户覆盖跳过，10k 转 D3

- [x] 阶段 D3：§VI 噪声 FT 第一次尝试（已崩，已停）
  - acceptance_criteria: 当时 `pg_coef=1`、BC=0、`schedule=adaptive`、无 critic warmup。第一步 `kl_mean=61.8`，LR 钉死，`Reset/torso` 0.03→0.37。logdir 保留；`model_10500` 不要续。
  - success_definition: 证明「无 BC + adaptive KL + 未预热 critic」不可用。不是过桩。

- [ ] 阶段 D3b：修复后噪声 FT（当前）
  - acceptance_criteria: 新 tmux `t4-s12-lightlp-ft`；父 ckpt 仅为 D1 `model_10000.pt`；`critic_warmup_iters=200` 且 warmup 期冻 actor；`behavior_coef=behavior_coef_end=0.5`；`schedule=fixed`；`pg_coef=1`；mix=0；深度噪声开；计数从 0 重计；无 NaN/OOM。禁止 mix0 `13999`。禁止续 `model_10500`。
  - verification_commands: 本机 `python -m pytest tests/test_safe_recurrent_distillation.py tests/test_t4_sparse_depth_student_gru_contract.py -q`；远端 log/TB `:8021`；`student_lineage.json`。硬停：warmup 200 步内 `Reset/torso` 不应爆炸；PPO 打开后 200 步课表/hard 单边塌则停。
  - success_definition: 短 FT 按修复配方开训并过 200 步健康门。本阶段完成 ≠ 过桩。

- [ ] 阶段 5：能力验收（非本切片当前）
  - acceptance_criteria: student-only evaluator + lineage + 连续回放；深度消融。混合老师的 TB 不进能力。
  - verification_commands: `eval_t4_hurdle.py`（学生任务）
  - success_definition: 另开或本阶段完成后才允许能力声明

## Commit units

1. `student-distill-recipe-mix0-recon`：阶段 1 代码 + 测试 + 本计划 + `.harness` 同步。
2. 训练产物、评估 JSON 不进 git。

提交前置：阶段 1 实现完成 + review 无 Critical + 上列 pytest 绿。用户未要求则不 commit。

## Known risks / blockers

- 阶段 2 停训需要执行时再确认一次；未停前不能 1024-env eval/probe（13:04 已 OOM）。
- 前 1000 iter `teacher_mix>0`，课表/reach TB 含老师执行，不能当学生能力。1000 步后 mix 锁 0 才开始看学生自己的 hard 踏石。
- 固定 `1e-4` 仍可能一步过大；健康门必须真的会停，不准「先看着」。
- warm-start 来自旧 mix>0 配方，可能带偏；评估差就从零。
- warp 深度仍只打 `/World/ground`。
- 老师仍 ungated；本计划不补 12 矩阵硬门。
- 远端 rsync worktree，不要把 Windows `scripts/nubot_run.sh` 同步上去。

## Next skill

`review`（D3b 开训后：warmup 200 步 actor 应冻住、LR 固定；随后 200 步 PPO+BC 课表/hard 不单边塌）。不是过桩。

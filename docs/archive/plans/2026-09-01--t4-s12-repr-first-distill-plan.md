# S12 表示先行深度蒸馏执行计划

> Status: done
> Scope: Isaac 学生候选 `model_13999` 已交付 JSON / deploy-only / 回放。MuJoCo plant 残留交给 `docs/plans/2026-09-02--t4-s12-teacher-plant-retrain-plan.md`。
> Spec: user-approved by request to plan+implement（2026-09-01）

## Objective

把 S12 GRU 学生主蒸馏改成可测的表示先行相位机：老师开车学分层 scan → 按探针自动切 Ross DAgger；旧 `s12_final_main` 并列保留，不再当默认开训入口。

## Active slice

阶段 2 已开训：nubot GPU2×256，`t4-s12-repr-first`，logdir `2026-09-01_09-39-29_s12_repr_first`。未宣称 24/32。

## Non-goals

- 不改老师 `21500` / 1937D / 稀疏 MDP。
- 不改部署观测；不减噪。
- 阶段 2 可开 nubot 14k，但仍不跑 Isaac evaluator、不跑 MuJoCo 能力声明（那是阶段 3）。
- 不开 PPO、`residual_ft`、`plant_ft`。
- 不热启 `13500`/`5999`。
- 不热补 `T4SparseDepthStudentFinalMainAlgCfg` 的旧旋钮。
- 不动翻箱。
- 不扩展 stairs/rough evaluator。

## Success criteria

1. 新 cfg `s12_repr_first`：`pg_coef=0`，表示段 mix 锁 1，动作段 mix 从切点 1→0/2000；表示段课表 `0.50/min_level=6`，切后 `0.10/None`。
2. 切段只认分层踏石/圆桩 recon 规则；全局 recon 变好但稀疏列差不得切；样本不足探针作废；相位只前进一次；到期 cap 切并写 `switch_reason`。
3. 表示段 actor/critic/std 不更新；recon 不被 control 投影掉。动作段冲突投影主从反转（保护 recon）。
4. `student_lineage.json` 含 phase / switch_iter / switch_reason / 两段课表。
5. 旧 `s12_final_main` 配置测试仍绿。
6. 本 slice **不**宣称 hard 踏石 24/32；那是阶段 2 训练后的 Spec 能力门。

## Verification path

```powershell
D:\anaconda\envs\pytorch\python.exe -m pytest tests/test_repr_first_switch.py tests/test_t4_sparse_depth_student_gru_contract.py tests/test_safe_recurrent_distillation.py tests/test_t4_observation_contracts.py tests/test_t4_sparse_command_contract.py tests/test_t4_stepping_stone_contracts.py -q
```

## Verification path status

`runnable`：阶段 1 本机 pytest。阶段 2 开训/评测依赖 nubot Isaac，本机 blocked；fallback 是写清 tmux 命令，不把命令存在当过桩。

## Required capabilities

- 本机 `D:\anaconda\envs\pytorch`（torch，无 Isaac 也可跑合同测试）
- 阶段 2：nubot IsaacLab + `scripts/nubot_run.sh` + tmux

## Fallback evidence

阶段 1 只接受 focused pytest。阶段 2 未跑前不得用 TB/ckpt 存在代替 Spec 能力门。

## final_integration_claim

完成后：一条 `train_t4_sparse_depth_student.py` 默认表示先行配方，合同测试可证伪切段/冻结/投影主从；旧 final_main 仍可对照。Isaac hard 稀疏过门与 sim2sim 不在本 claim。

## 工作项

- [x] 阶段 1：表示先行代码合同
  - acceptance_criteria: 分层切段纯函数覆盖负向（全局 recon 不得切、样本不足作废、只前进一次、cap/`baseline_invalid`）；SafeRecurrentDistillation 表示段冻 actor、不投影掉 recon；动作段保护 recon；ReprFirst cfg/env/train/lineage 落地；final_main 断言仍成立。
  - verification_commands: `D:\anaconda\envs\pytorch\python.exe -m pytest tests/test_repr_first_switch.py tests/test_t4_sparse_depth_student_gru_contract.py tests/test_safe_recurrent_distillation.py tests/test_t4_observation_contracts.py tests/test_t4_sparse_command_contract.py tests/test_t4_stepping_stone_contracts.py -q`
  - success_definition: 本机 focused pytest 绿，开训入口指向 `s12_repr_first`。
- [x] 阶段 2：nubot 从零开训
  - acceptance_criteria: tmux 一条 run，每卡 256，teacher `21500`，warmstart null；数 iter 可见 `Distill/phase=representation` 与分层 recon tag。
  - verification_commands: nubot tmux + `scripts/nubot_run.sh`；默认 4×256，PhysX 失败则单卡 256。入口已是 `s12_repr_first`，不要再传 `s12_final_main`。

```bash
tmux new -s t4-s12-repr-first
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/nubot_run.sh -m torch.distributed.run --nnodes=1 --nproc_per_node=4 \
  legged_lab/scripts/train_t4_sparse_depth_student.py \
  --teacher_checkpoint <model_21500.pt> \
  --teacher_eval_manifest <teacher_gate.json> \
  --task_num_envs 256 --distributed --headless --seed 42
```

多卡 PhysX 失败时去掉 distributed，改单卡 256。数 iter 后确认 TB `Distill/phase=representation` 与 `Recon/mse_stepping_stones` / `Recon/mse_raised_pillars`。
  - success_definition: 训练进程存活且相位/探针日志符合 Spec，不是过桩。
- [x] 阶段 3：动作段后 student-only 选 ckpt（Isaac JSON + deploy-only 已出；Isaac 回放录制中）
  - acceptance_criteria: 与 `13500` 同口径 32-ep；达 Spec 成功标准才宣称打破困局。
  - verification_commands: `eval_t4_hurdle.py` student-only 矩阵
  - success_definition: evaluator JSON + lineage + 回放；否则记失败归因。
  - 2026-09-02：`model_13999` JSON 在 `artifacts/eval/s12_repr_first_m13999_student_only/`。hard 踏石 strict 27/32、reach_2m 30/32；圆桩 30/32、31/32。相对 `13500` 未掉超过 4/32。Isaac 数字过 Spec 门。
  - 2026-09-02 续：`model_13999_deploy.pt` 已导出（无 teacher/scan decoder/critic）；本机 3168-D→27-D finite，GRU reset 过。MuJoCo 六课已跑：踏石 ~6.3 s、圆桩 ~4.2 s 摔，记 plant 残留，不上真机。Isaac 连续回放 GIF：`artifacts/replay/s12_repr_first_m13999/`（hard 踏石/圆桩 20 s 内均为站立 oob/timeout，peak ~5 m；easy 圆桩有一次 accel）。待人看，未宣称人工验收。
- [x] 阶段 0：Spec 与失败基线
  - acceptance_criteria: Grill Q1–Q5 已定；`13500`/老师/Phase B 数字与噪声消融已留存。
  - verification_commands: `docs/specs/2026-09-01--t4-s12-repr-first-distill.md`；`artifacts/eval/s12_final_main_m13500_student_only/`
  - success_definition: 新 lineage 不把失败旋钮当默认。

## Commit units

1. `落地S12表示先行蒸馏合同`：阶段 1 完成，focused tests PASS，review 无 Critical。

训练 checkpoint / 视频 / TB 不进 Git。

## Known risks / blockers

- `_limit_auxiliary_gradient_norm` 在 control=0 时会把 recon 缩成 0；表示段必须走 recon-only 路径，不能复用旧投影。
- 切段相对基线无历史标定；`baseline_invalid` 时只能 cap 切。
- 本机不能替 nubot 开训；阶段 1 完成 ≠ 过桩。
- 老师接触真值学生没有，表示先行只补 scan。

## Next skill

deploy-only、dummy 推理、Isaac 四格回放已落盘。下一步：人看 `artifacts/replay/s12_repr_first_m13999/` hard GIF。MuJoCo plant 残留不在本 Spec 修。ready 仍走 `review`。不上真机。

# Executable Plan - S12 门控 DAgger → DAgger+PPO → 锚定 FT

> **Status: archived（非权威）** — living work surface: `.harness/work_index.md`。

> Status: done（Phase B `model_5999.pt` 验收通过；Phase C 因无必要且 PPO/KL 风险较高而跳过）
> Date: 2026-08-26
> Parent: `docs/archive/plans/2026-08-22--t4-sparse-s12-rim-yaw-student-plan.md`
> Replaces: 未启动的 D5 `s12_rtx_hardmix`
> Teacher: frozen S12 `model_21500.pt`
> GPU: nubot；所有长运行命令必须在 tmux 内执行，不覆盖远端 `scripts/nubot_run.sh`

## Objective

把固定 iteration 自动开 PPO 的 D5 单 lineage 拆成三个独立 lineage。阶段切换只看 student-only evaluator；每次 continuation 加载完整 student/GRU/decoder/critic，但重置 optimizer、iteration、算法计数和 action std，并用同一份经 Gate 0 校验的 S12 teacher 重新装载 teacher。

保持 tiled RTX、48×64、35°相机、完整深度噪声/延迟、外参 DR、1937D teacher、3168D student、hard reset `fraction=0.50/min_level=6` 不变。

## Gate 0：老师资格

- 固定 `model_21500.pt`，踏石/圆桩分别评 `d=0`、`d=0.8`、`vx=0.8`，每条件 32 局。
- hard 两类都必须 strict `>=24/32`；否则停止学生方案，先修 teacher 或 evaluator。
- RTX checksum probe 必须证明 processed policy frame 持续更新。
- 新开所有 lineage；不续 D4、D1、D3。teacher evaluator JSON 通过 `--teacher_eval_manifest` 进入 lineage。

## Phase A：纯 DAgger

- `s12_rtx_gated_dagger`，最多 8000 iter，500 iter 保存。
- `teacher_mix 1→0/1000`；1000 后 student-only rollout。
- `pg_coef=0`、`behavior_coef=1`、`recon_coef=1`、fixed LR `1e-4`。
- 2k/4k/6k/8k evaluator；2k 只诊断，4k 后才允许过门。

进入 Phase B 必须同时满足：easy 踏石 strict `>=24/32`、圆桩 `>=12/32`；hard 两类 `reach_2m>=8/32` 且 strict 合计 `>=4/64`；无 NaN/OOM；std 未长期顶到 0.2；连续回放证明不是 teacher mix 假象。8k 未过门即停，不加 PPO 救场。

## Phase B：联合 DAgger+PPO

- 从 Phase A 过门的最佳 checkpoint 启动 `--mode joint` / `s12_rtx_gated_joint`，最多 6000 iter，250 iter 保存。
- optimizer/iteration/算法计数重置，std 重置 `0.08`；`teacher_mix=0`。
- 前 200 iter actor freeze、critic-only；随后 800 iter `pg_coef 0→0.2`。
- `behavior_coef=1`、`recon_coef=1`、fixed LR `3e-5`；每 1000 iter evaluator。
- `kl_mean` 滚动高位 >0.05、单次 >0.5、easy 明显退化或 behavior loss 比 parent 上升 >25% 即停。

进入 FT：hard 两类均 strict `>=8/32` 且 `reach_2m>=16/32`；easy 仍满足 Phase A 门；evaluator JSON、lineage、连续回放齐全。不满足则不 FT、不延长同配方。

## Phase C：部署噪声锚定 FT

- 从 Phase B 最佳 checkpoint 启动 `--mode deploy_ft` / `s12_rtx_deploy_ft_v2`，严格 1000 iter。
- optimizer/iteration/算法计数重置，std `0.08`；前 200 iter critic-only，随后 400 iter `pg_coef 0→0.5`。
- `behavior_coef=0.5`、`recon_coef=0.25`、`teacher_mix=0`、fixed LR `3e-5`、entropy `0`。
- 500 iter 中门：hard/easy 任一明显退化即停并保留 Phase B parent。1000 后不自动续训，不现场提高 PPO 或关闭 BC。

## Continuation 与 lineage 合同

- `train_t4_sparse_depth_student_ft.py --mode joint|deploy_ft` 强制提供 `--student_checkpoint`、`--teacher_checkpoint`、`--teacher_eval_manifest` 和 `--capability_gate_json`。
- parent 必须包含完整 depth encoder、GRU、actor、reconstruction decoder、critic 和内嵌 teacher；deploy-only 或缺 critic 的 checkpoint 拒绝。
- parent 内嵌 teacher 必须逐张量匹配指定的 S12 teacher；来源不明或不匹配直接拒绝。
- `student_lineage.json` 记录 phase、parent/teacher SHA256、能力门 JSON 路径与 SHA、损失系数、ramp/warmup、std reset 和噪声模型。

## Verification

```powershell
D:\anaconda\envs\pytorch\python.exe -m pytest `
  tests/test_t4_sparse_depth_student_gru_contract.py `
  tests/test_safe_recurrent_distillation.py `
  tests/test_t4_observation_contracts.py `
  tests/test_t4_sparse_reward_contracts.py `
  tests/test_t4_stepping_stone_contracts.py -q
```

nubot 每阶段先做小规模 smoke，再在 tmux 启动正式 lineage。最终 3 个固定 seed、每 terrain/difficulty 32 局：easy aggregate strict `>=75%`；hard 每类 aggregate strict `>=50%`，任一 seed 不低于 `12/32`。交付 evaluator JSON、checkpoint/teacher SHA、lineage manifest 和连续回放。FT 未改善或破坏能力时，Phase B checkpoint 是最终候选。

## Current status

- [x] 三阶段配置、续训加载、teacher pin、std reset、lineage 字段与纯 Python 合同实现。
- [x] 本机 focused tests 通过。
- [x] Gate 0 teacher evaluator 与 RTX checksum 复核：hard 踏石 `28/32`、圆桩 `32/32`；probe `ok=true`。
- [x] nubot Phase A smoke 与正式 lineage：GPU2×256，tmux `t4-s12-rtx-gated-dagger`。
- [x] Phase A `model_6000.pt` student-only gate；Phase B strict smoke 与正式 lineage 已启动。
- [x] 修复 evaluator 的 GRU live hidden 污染与跨局 reset 后，Phase B `model_5999.pt` 通过 student-only gate：easy 踏石/圆桩 strict `32/32`、`29/32`；hard 踏石/圆桩 strict `18/32`、`27/32`，reach_2m `28/32`、`30/32`。
- [x] 两类 hard 连续回放已生成并人工复核；lineage、evaluator JSON 和回放证据齐全。
- [x] 导出 deploy-only checkpoint；只含 depth encoder、GRU、student actor 与 std。本机 3168-D dummy inference 输出 27-D finite action，GRU reset 可复现首步输出。
- [x] Phase C 跳过：Phase B 已超过验收门，继续使用 `pg_coef=0.5` 的 FT 没有必要，并会引入已有历史证据支持的 actor/KL 破坏风险。

最终交付清单：`artifacts/checkpoints/nubot/s12_rtx_gated_joint/delivery_manifest.json`。训练源 checkpoint SHA256 为 `d799a966dddf039a54d0c9a4896953de0d39dca8126c15645876f0d6de5fa20c`；deploy 包 SHA256 为 `f06d29a316fe012ef449d41eb5001f27554bcf10a73ea543b9d1cfd058942bf5`。

训练产物、视频和 evaluator JSON 不进 Git。

# Executable Plan - S12 部署向 RTX 蒸馏（D4a 纯 BC → D4b DAgger+PPO）

> **Status: archived（非权威）** — living work surface: `.harness/work_index.md`。

> Status: done
> Date: 2026-08-25
> Successor: `docs/archive/plans/2026-08-26--t4-sparse-s12-gated-dagger-joint-ft-plan.md`（D5 hardmix 未启动即被门控三阶段路线替换）
> Parent: `docs/archive/plans/2026-08-22--t4-sparse-s12-rim-yaw-student-plan.md`
> Predecessor: `docs/archive/plans/2026-08-23--t4-sparse-s12-student-distill-recipe-plan.md`（superseded：D1 干净 warp / D3 噪声 FT 均不作部署候选）
> Spec: `docs/specs/2026-08-12--t4-unified-depth-locomotion.md`；梅花桩目标 `docs/specs/2026-08-15--t4-stepping-stones-and-hurdle-stable.md`
> Reference: `docs/research/2608.02653v1/auto/2608.02653v1.md`（LightLP §IV / §VI / Table II / Table V，**不是**执行权威）
> Branch: `t4-train`
> Planning surface: docs plan
> GPU: nubot 四卡；worktree `/home/nubot/phn_ws/t4_train/TienKung-Lab-s12-gru-ppo`。zhuoqun 翻箱不动

## Objective

按 LightLP §VI 做**部署向**深度蒸馏：tiled RTX + 深度噪声 + 相机外参 DR，D4a 纯 BC（mix 1→0）再 D4b 小权 DAgger+PPO。目标是真机域，不是再刷一条只在 Isaac 干净深度里好看的学生。老师冻 `model_21500.pt`。不续 D1/D3。D4c（损失切到任务回报的短 RL）与真机上电不在本 slice。

对照原文，纠正上一版「纯 DAgger、不开 PPO、从 0 就满噪声但永远不 FT」以及 D3「克隆未 usable 就满权 PPO」的偏差。

## Active slice

落地 D4 代码合同并开训 `s12_rtx_deploy_distill`（RTX、噪声从 iter 0、外参 DR、D4a 纯 BC → D4b `pg_coef=0.2` 且 BC=1）。`warmstart=null`。D4c FT 与真机不在本 slice。

## Non-goals

- 不续 D1 `s12_lightlp_dagger_only` / D3 / D3b / D3c ckpt，不重训老师 `model_21500.pt`
- 不把「先干净深度再加噪声」写成论文对齐；若 RTX+满噪声连 easy 5k 门都过不了，才允许工程上把噪声从 0.5× 升到 1.0×，并在 lineage 标明这是仓库补丁
- 不回 v3 CNN / L2T / Cross-Attention
- 不为了快退回 warp；OOM 只降 env（512 试一次 → 256 → 128）
- 不在本切片重开老师侧 Kt/COM 全套（学生外参 + 本体噪声 + 已有摩擦/质量/push 先做）
- 本切片不开 D4c、不上真机；没有「每种行为的 usable version」时不准加满权 PPO
- 本计划完成 ≠ 梅花桩能力验收

## 流水线（部署域，不是只仿真）

老师仍冻 `model_21500.pt`。新 lineage。

```text
S12 scan teacher 21500
  → D4a 0–2k: RTX+噪声+DR, mix 1→0, pg=0, BC=1
  → D4b 2k–14k: mix=0, DAgger+PPO pg=0.2, BC=1, critic warmup 200, fixed LR
  → Isaac student-only d=0 与 d=0.8 门
  →（过门后另开）D4c 1k: 同环境，BC 1.0→0.5，pg 0.2→0.5–1
  → MuJoCo 自遮挡 + 同课程 → 部署候选
```

网络仍按 LightLP：48×64 MLP 深度 + 本体 + GRU + scan recon（recon 训练用、部署丢掉）。

### A. 眼睛：全程 RTX

- 稀疏学生从 warp `/World/ground` 改回 tiled RTX D455，schema 头高 + **35°**（不改角度）。
- 训练 `enable_cameras=True`，沿用已修好的 headless `sim.render()`。
- 自遮挡打开：RTX 必须打到地形和机器人。`disable_visual_assets=True` 只跳过 Nucleus 材质/穹顶灯，保留本机机器人 USD 与地形 mesh。
- 默认 **256 env/卡 × 4**。禁止退回 warp。
- 每 reset：相机外参 **±1 cm / ±0.025 rad**（Table II）。

### B. 视觉噪声：蒸馏期就开（§VI）

`depth_noise.py` 从 **iter 0** 打开。σ = 0.005 + 0.02 d；全局尺度 ±5%；块状 dropout；30–60 ms 延迟（2–3 步 @ 50 Hz）+ 持帧约 25–33 Hz。相机 `update_period=0.02`。Isaac `SensorNoiseCfg` 保持关，避免叠两套。

### C. 物理 / 本体 DR

蒸馏 env **不要** `add_noise=False`。打开本体噪声（`ang_vel` / `projected_gravity` / `joint_pos` / `joint_vel`；`height_scan` 尺度 0）。保留老师摩擦、Trunk 质量、push、reset 扰动。本切片最小补齐相机外参抖动。不开 RGB 纹理 DR。

### D. 算法：先覆盖 hard，再按 §VI 开 PPO，BC 不降

相对「论文全程 DAgger+PPO」的有意偏差：D3 证明 critic 未热、pg 过大、BC=0 会毁策略。λ_RL 论文没写。

- **D4a（0–2000）**：`pg_coef` 目标 0.2 但 `pg_delay_iters=2000` 使有效 PG=0；`behavior_coef=1`；`teacher_mix` 1→0 / 2000 iter。噪声和 DR 已经开着。
- **D4b（2000–14000）**：`teacher_mix=0`；其后 200 iter 冻 actor 只训 critic，再 `pg_coef=0.2`；`behavior_coef=1.0` **不退火**；`schedule=fixed`；`lr=1e-4`。禁止有效 `pg>0` 且 mix≠0。禁止 adaptive KL。
- **D4c 不在本 slice**。过门后另开：BC 1.0→0.5（不到 0），pg 0.2→约 0.5–1，不再冻 actor 200 步。不是真机微调，也不是把 D4b 多跑 1k。

稀疏课表：蒸馏期关掉 sparse `move_down` + `random_level_reset_fraction=0.25`。

### E. Sim2sim

MuJoCo 必须像 RTX：深度相机看见机器人；同一 48×64 / 87° / 35°；评测可开同一套 Python 深度噪声；课程 `d=0` 与 `d=0.8`，踏石+圆桩，`vx=0.8`。能力只认 student-only JSON + 回放。

## Success criteria

1. 稀疏学生深度相机是 tiled RTX D455，48×64，schema 头高 35°，训练 `enable_cameras=True`，默认 256 env/卡。
2. 蒸馏从 iter 0 开 LightLP Python 深度噪声；本体 `add_noise=True`；每 reset 外参 ±1 cm / ±0.025 rad。
3. `T4SparseDepthDistillationAlgCfg`：mix 1→0 / 2000；`pg_coef=0.2`；`pg_delay_iters=2000`；`critic_warmup_iters=200`；BC=1 不退火；`schedule=fixed`；`lr=1e-4`；`run_name=s12_rtx_deploy_distill`；`max_iterations=14000`。
4. `SafeRecurrentDistillation` 允许「cfg 上 mix 起点>0 且 pg 目标>0」，但有效 PG 与 mix 不得重叠；D4a 期间 actor 必须能被 BC 更新（不能整段冻 actor）。
5. 蒸馏期 sparse 列不 `move_down`；`random_level_reset_fraction=0.25`。老师 MDP 默认不变。
6. MuJoCo 默认深度看见机器人；`--depth-noise` 可开同等 Python 噪声。
7. 本机合同 pytest 绿。nubot 在 checksum probe 后开训；不并行 1024-env probe。
8. **本计划完成 ≠ 会过桩。** 健康门与 Isaac 5k/10k/14k 门见下；达不到硬门则停，改眼睛/课表，不准加 pg。

健康 / 能力门（开训后监控，不是本 slice 代码验收）：

- ~200 iter：无 NaN/OOM；mix 在退火；behavior 相对 iter 0 下降
- Isaac 5k：easy 相对 D1（21/32、26/32）不能崩到 <10/32
- Isaac 10k/14k：`d=0.8` 不再 0/32；目标 strict≥8/32 或 reach_2m≥0.25。达不到：**停，改眼睛/课表，不准加 pg**

## Verification path

```text
# 本机（不需要 GPU / Isaac）
D:\anaconda\envs\pytorch\python.exe -m pytest tests/test_t4_sparse_depth_student_gru_contract.py tests/test_safe_recurrent_distillation.py tests/test_sim2sim_t4_depth_student.py tests/test_t4_sparse_reward_contracts.py tests/test_t4_stepping_stone_contracts.py -q

# nubot checksum probe（小 env，禁止 1024 并发）
bash scripts/nubot_run.sh legged_lab/scripts/probe_t4_student_depth_render.py \
  --num_envs 8 --steps 24 --enable_cameras \
  --output artifacts/diagnostics/s12_rtx_deploy_probe.json

# nubot 开训（warmstart 空；老师 ungated 21500）
bash scripts/nubot_run.sh -m torch.distributed.run --nnodes=1 --nproc_per_node=4 \
  legged_lab/scripts/train_t4_sparse_depth_student.py \
  --teacher_checkpoint <model_21500.pt> --allow_ungated_teacher \
  --task_num_envs 256 --distributed --headless --seed 42 \
  --run_name s12_rtx_deploy_distill
```

### Verification path status

`runnable` —— 本机 pytest 可跑。nubot 开训依赖 SSH/GPU 空闲与 worktree rsync；不得与 1024-env probe 并发。

## Required capabilities

本机 `D:\anaconda\envs\pytorch\python.exe` 跑 pytest。nubot 4×RTX 4090 + `scripts/nubot_run.sh` + tmux。远端 worktree：`/home/nubot/phn_ws/t4_train/TienKung-Lab-s12-gru-ppo`。老师 ckpt 仍走 `--allow_ungated_teacher`。不要把 Windows `scripts/nubot_run.sh` 同步上去。

## Fallback evidence

- RTX OOM：只降 `--task_num_envs`（512 试一次 → 256 → 128），禁止 warp。
- 满噪声连 easy 5k <10/32：允许噪声尺度 0.5×→1.0× 课程，lineage 标明仓库补丁，不是 LightLP。
- SSH/GPU 忙：本机代码+测试+计划落地仍算本 slice 的实现步；开训命令写入 `.harness/progress.md`，不伪造 run。
- hard 10k/14k 仍 0/32：停，改眼睛/课表，不准加 pg、不准开 D4c。

## Final integration claim

`final_integration_claim`: 交付部署域 D4 代码合同与 `s12_rtx_deploy_distill` 开训入口（tiled RTX、噪声从 0、外参 DR、D4a 纯 BC → D4b 小 PPO）。不声明过桩。不开 D4c。不续 D1/D3。

## 工作项

- [x] 阶段 0：论文对照与失败实验（基线）
  - acceptance_criteria: §VI 是 DAgger+PPO 再短 FT；Table II 含相机外参；干净深度会 transfer 差。D1 干净 warp easy 会、hard 0。D3 pg=1 BC=0 毁 easy。D4c 是损失权重切换，不是再蒸一遍。
  - verification_commands: 只读 LightLP §IV/§VI/Table II/V；`artifacts/eval/s12_d3c_vs_d1_student_only/`
  - success_definition: 不再把「纯 DAgger 到底」或「从 D1 再 FT」当部署方案

- [x] 阶段 1：计划、相机/配方/MuJoCo 合同、本机测试（当前）
  - acceptance_criteria: 本计划 active；旧 08-23 配方计划 superseded；RTX 48×64 + 外参抖动 + 噪声从 0 + D4 系数 + sparse 不降级 + MuJoCo 自遮挡；pytest 绿
  - verification_commands: 上列本机 pytest
  - success_definition: 不启动 GPU 也能证明与 D1 warp / D3 FT 配方不同

- [x] 阶段 2：nubot probe + 开训
  - acceptance_criteria: checksum probe `ok`（已满足）；tmux `s12_rtx_deploy_distill` 已开（**单卡 GPU2×256**；双卡 2/3 PhysX 失败已弃）；`student_lineage.json` 写明 mix/pg_delay/noise/cameras/256 env；warmstart null。logdir `2026-08-25_23-35-06_s12_rtx_deploy_distill`
  - verification_commands: probe JSON；远端 log `artifacts/diagnostics/s12_rtx_deploy_train_1gpu.log`；`student_lineage.json`
  - success_definition: 部署域上已开训。不是过桩。四卡扩容等 0/1 空闲另开

- [x] 阶段 3：健康门与 Isaac 能力门（开训后，非本实现步）
  - acceptance_criteria: 200 iter 健康；5k easy 不崩；10k/14k `d=0.8` 不再 0/32 否则停
  - verification_commands: TB；`eval_t4_hurdle.py --task t4_loco_sparse_depth_student` → `artifacts/eval/s12_rtx_deploy_student_only/`
  - success_definition: **未过 hard 门**（d=0.8 两边 0/32）。按 fallback 停，改课表，见 successor。不开 D4c

## Commit units

1. `s12-rtx-deploy-distill-contract`：阶段 1 代码 + 测试 + 本计划 + `.harness` 同步。
2. 训练产物、评估 JSON 不进 git。

提交前置：阶段 1 实现完成 + 上列 pytest 绿。用户未要求则不 commit。

## Known risks / blockers

- tiled RTX 256 env/卡仍可能 OOM；只降 env。
- 前 2000 iter mix>0，课表/reach TB 含老师执行，不能当学生能力。
- `disable_visual_assets=True` 若日后证明 RTX 打不到机器人，再改，不准为此退回 warp。
- 老师仍 ungated；本计划不补 12 矩阵硬门。
- 远端 rsync worktree，不要把 Windows `scripts/nubot_run.sh` 同步上去。

## Next skill

`review`（阶段 1 代码合同 + probe 绿）。四卡开训等 GPU 0/1 空闲后按 `.harness/progress.md` 命令启动。开训后的能力门仍走本计划阶段 3，不是本代码步的 ready。

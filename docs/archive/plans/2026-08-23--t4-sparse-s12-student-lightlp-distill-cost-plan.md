# Executable Plan - S12 GRU 深度学生回到 LightLP 最小蒸馏配方

> Status: done
> Date: 2026-08-23
> Successor: `docs/plans/2026-08-23--t4-sparse-s12-student-distill-recipe-plan.md`（蒸馏配方修复）
> Parent: `docs/plans/2026-08-22--t4-sparse-s12-rim-yaw-student-plan.md`
> Predecessor: `docs/archive/plans/2026-08-23--t4-sparse-s12-student-rtx-correctness-plan.md`（正确性已交付）
> Spec: `docs/specs/2026-08-12--t4-unified-depth-locomotion.md`
> Reference: `docs/research/2608.02653v1/auto/2608.02653v1.md`（LightLP §IV-A / §VI）
> Branch: `t4-train`
> Planning surface: docs plan

## Objective

把 S12 GRU 深度学生的蒸馏成本从实测 **7.22 s/iter × 25000 ≈ 50 h** 降到论文量级（目标 **≈3 s/iter × 15000 ≈ 12 h**），手段是把学生侧的深度后端、观测形状、编码器和更新期回退到 LightLP §IV-A/§VI 的最小配方，而不是继续加固当前这套。老师 MDP、地形和能力验收标准都不动。

## 背景证据（已测，不要重测）

`s12_gru_ppo_rtx167` 第 147 iteration，四卡 RTX 4090（PCIe Gen4，无 NVLink，GPU3 跨 NUMA）：

| 指标 | p50 | p90 | max | mean |
| --- | --- | --- | --- | --- |
| collection | 5.391 s | 5.568 s | 6.063 s | 5.390 s |
| learning | 1.128 s | 3.732 s | 7.359 s | 1.832 s |
| 合计 | 6.504 s | 9.210 s | 13.328 s | 7.223 s |

GPU 利用率 22–27%，显存控制器利用率 1–14%，显存 20196–23880 / 24564 MiB。

collection 拆解（对照 `.harness/decisions.md` 已钉死的基线）：

| 组成 | 耗时 | 占比 |
| --- | --- | --- |
| 物理 + S12 MDP + height scan（老师无相机） | 1.96 s | 36% |
| RTX 相机固定管线（相机在场景但不出图） | +2.28 s | 42% |
| 真实 `sim.render()`（8 次 / iter） | +1.15 s | 21% |

四条根因：

1. **GPU 空闲 75%。** 瓶颈在 CPU 侧 Isaac/Kit/Python 与同步，不在算力。所以砍分辨率、砍相机频率、加卡都无效（50 Hz→16.7 Hz 只省 2%）。
2. **RTX 相机的固定开销是出图本身的 2 倍。** tiled `annotator.get_data()`、1024 个相机 prim 的 Kit/Fabric 位姿同步，加上 Python ingest 里的 `torch.any()` 同步、37 MB `depth_history` 布尔掩码 gather/scatter、每步 42 MB 观测 concat+clip。
3. **网络与观测偏离论文。** 论文 §IV-A 明说 "we encode the depth with an MLP … pass the result through an RNN"，并且 "The convolutional encoder yielded no measurable gain yet noticeably lengthened training, so we use an MLP encoder in all experiments"。仓库用 3 层 Conv2d（约 45× FLOPs），且在已有 GRU 的前提下仍堆 3 帧深度 + 10 帧本体感（obs 10176 维，storage 1.0 GB/rank）。
4. **更新期比论文重 4–5 倍。** `_coordinated_shared_gradients` 每个 minibatch 走 3 次 `autograd.grad(retain_graph=True)` + 1 次 `backward()`；再加梯度投影/范数限制、`assert_finite_grads` 全参数同步、每 minibatch 一次 KL all_reduce+broadcast、结束后 `_candidate_metrics()` 再遍历一遍全部 minibatch。约 38 次 NCCL 集合通信/iter。`retain_graph` 钉住约 1 GB CNN 激活，配合 1.0 GB obs storage 把 GPU2 顶到 23880/24564 MiB —— learn 的 1 s→7.4 s 双峰是分配器停顿，不是算法慢。

额外发现（与速度无关但影响结果）：`behavior_coef` 与 `teacher_mix` 都在 2000 iteration 归零，意味着 25000 步里只有前 2000 步在蒸馏，后 23000 步是带重建损失的纯 PPO。论文是 14000 步全程 `L_DAgger + λ_RL·L_PPO`，再单独 1000 步 fine-tune。

## Active slice

阶段 1–4 已落地：warp `RayCasterCamera` + 3168/MLP + 单次 backward + 新 lineage `s12_lightlp_raycast` 已在 nubot 开训，墙钟近窗 total p50 2.32 s。阶段 5 能力验收仍是 non-goal。

**训练健康（2026-08-23 20:00，不是本切片成功标准）：** 约 7740/15000 仍在跑，**不是** v3@3305 坍塌。4k 后课表回落、`behavior` 停住、LR 从 iter 50 钉在 `1e-5`。判断：感知/MLP 前 2k 能学；按现行配方把 15k 跑完希望不大。主因是老师 PPO 的 `desired_kl=0.01` 自适应 LR 与蒸馏 BC 冲突，同时 `teacher_mix`/`behavior_coef` 仍在往 0 撤。用户未授权停训。完整指标与排除项见 `.harness/decisions.md`。

## Non-goals

- 不改 S12 老师 MDP、不重训老师、不动 `model_21500.pt`。
- 不粗化 S12 地形、不降 command 难度、不砍环境数当加速手段。
- 不把 GPU 利用率或 s/iter 写成能力成功标准；能力仍只认 evaluator JSON + lineage manifest + 连续回放。
- 不换 per-env RTX `Camera` 后备（同样是 RTX，且样本减半）。
- 不为了加速再往下砍相机频率。
- 不 resume v3 `model_3500/4000`、nansync、holdcap、`13-04-08`。
- 不动 zhuoqun；不在 nubot 四卡开 GUI play。
- 本切片不完成梅花桩能力验收。

## Success criteria

1. `RayCasterCamera` 深度后端在 1024 env / `update_period=0.06` 下 collection 中位数 **≤ 3.0 s**（对照当前 5.391 s）；且 headless 不再需要 `enable_cameras` / `sim.render()`。
2. 深度图非全 invalid，且只在 16.7 Hz tick 变化——沿用现有 `probe_t4_student_depth_render.py` 的 checksum 合同。
3. 学生观测回到论文形状：单帧深度 + 单帧本体感，宽度 = `PROPRIO_FRAME_DIM + DEPTH_POLICY_SIZE[0] * DEPTH_POLICY_SIZE[1]` = 96 + 3072 = 3168；**老师 1937 维观测与 `PROPRIO_HISTORY_LENGTH` 不变**。
4. 深度编码器为 MLP；`shared_encoder_parameters()` 仍只含 depth encoder + GRU。
5. 更新期每 minibatch 只有一次图遍历；`retain_graph` 从 `_coordinated_shared_gradients` 移除或该函数整体退役。
6. 新 lineage：老师 `model_21500.pt` + 从零学生，15000 步蒸馏（全程 DAgger+PPO 同开）+ 1000 步 §VI noise/latency FT。
7. 合同测试全绿；能力仍只交给后续 evaluator。**本计划完成 ≠ 会过桩。**

## Verification path

```text
# 本机（不需要 GPU）
D:\anaconda\envs\pytorch\python.exe -m pytest tests/test_t4_sparse_depth_student_gru_contract.py tests/test_safe_recurrent_distillation.py tests/test_t4_observation_contracts.py -q

# nubot（需要先停训，见 Known risks）
bash scripts/nubot_run.sh legged_lab/scripts/probe_t4_student_depth_render.py --headless --num_envs 1024 --steps 24 --output artifacts/diagnostics/s12_student_raycast_probe.json

# 新 lineage
bash scripts/nubot_run.sh -m torch.distributed.run --nnodes=1 --nproc_per_node=4 \
  legged_lab/scripts/train_t4_sparse_depth_student.py \
  --teacher_checkpoint <model_21500.pt> --allow_ungated_teacher \
  --task_num_envs 1024 --distributed --headless --seed 42 --run_name s12_lightlp_raycast

# 墙钟复核
D:\anaconda\python.exe artifacts\tmp_s12_rtx_remote.py perf
```

### Verification path status

`runnable` —— 本机 pytest 74 passed；nubot 1024-env probe `ok=true`、`collection_s=1.833`；train `s12_lightlp_raycast` 已开，early perf n=16 collection p50 2.202 s / total p50 2.342 s。15000 步尚未跑完；阶段 5 未开。

## Required capabilities

本机 `D:\anaconda\envs\pytorch\python.exe` 跑 pytest；本机 `D:\anaconda\python.exe` + paramiko 做 SSH 编排；nubot 4×RTX 4090 + `scripts/nubot_run.sh` + tmux。`RayCasterCamera` 已就位：`/home/nubot/IsaacLab/source/isaaclab/isaaclab/sensors/ray_caster/ray_caster_camera.py`；仓库已在对 `/World/ground` 做 warp 光追（`height_scanner` / `foot_scanner`，见 `legged_lab/utils/env_utils/scene.py:124-157`），无需新建基础设施。

## Fallback evidence

若 `RayCasterCamera` probe 的 collection **> 4.0 s**（即节省不足 26%），说明 2.28 s 的固定开销主要不在 RTX 管线而在 Python ingest：此时停下来先用 `torch.cuda.synchronize` 分段计时定位，**不要**顺手降 env 或降分辨率充数。若光追深度出现全 invalid 或看不到梅花桩边缘，先查 `mesh_prim_paths` 与 `clipping_range`，不准靠加大 ingest 假装在看。

## Final integration claim

`final_integration_claim`: 交付一条与 LightLP §IV-A/§VI 同形状的 S12 深度学生蒸馏路径——warp 光追深度、单帧 MLP 编码器 + GRU、单次 backward、15000 蒸馏 + 1000 noise FT——并给出 probe 与 `perf` 两份墙钟证据。不声明过桩能力。

## 工作项

- [x] 阶段 1：warp 光追深度后端 + 墙钟 probe（当前）
  - 改 `legged_lab/envs/t4/depth_student_env.py::_t4_student_depth_camera` 返回 `RayCasterCameraCfg`（`PinholeCameraPatternCfg` 用 `d455_depth_config.py` 的 `D455_FOCAL_LENGTH_CM` / `D455_HORIZONTAL_APERTURE_CM` 保持 87° HFOV，`mesh_prim_paths=["/World/ground"]`，`data_types=["distance_to_image_plane"]`，直接渲 48×64 取消 72×128→resize）；`legged_lab/envs/t4/t4_env.py::_has_rtx_sensors` 对光追相机返回 False 以停掉 `sim.render()` 调度；`legged_lab/utils/env_utils/scene.py` 接线新 cfg 类
  - acceptance_criteria: 1024 env probe `ok=true`，checksum 只在 16.7 Hz tick 变化且非全 invalid；collection 中位数 ≤ 3.0 s
  - verification_commands: `bash scripts/nubot_run.sh legged_lab/scripts/probe_t4_student_depth_render.py --headless --num_envs 1024 --steps 24 --output artifacts/diagnostics/s12_student_raycast_probe.json`
  - success_definition: headless 学生不经 RTX 也能拿到正确的 16.7 Hz 深度，且采集期至少省下 2.4 s/iter
- [x] 阶段 2：学生观测与编码器回到论文形状（下一步）
  - 学生只取 `actor_obs_buffer` 的最新一帧（**不改** `PROPRIO_HISTORY_LENGTH`，老师 1937 维不动），深度历史长度走学生专属常量（**不改** `DEPTH_HISTORY_LENGTH`，`sim2sim_t4_depth_student.py` 部署路径不动）；`rsl_rl/rsl_rl/modules/depth_student_teacher.py` 的 `depth_encoder` 从 Conv2d 栈换成 MLP
  - acceptance_criteria: `STUDENT_ACTOR_OBS_DIM` = 3168；老师观测合同测试不变；`shared_encoder_parameters()` 仍只含 depth encoder + GRU
  - verification_commands: `D:\anaconda\envs\pytorch\python.exe -m pytest tests/test_t4_sparse_depth_student_gru_contract.py tests/test_t4_observation_contracts.py -q`
  - success_definition: rollout storage 从 1.0 GB 降到约 311 MB / rank，CNN 三重冗余消失
- [x] 阶段 3：更新期瘦身到一次 backward
  - `rsl_rl/rsl_rl/algorithms/safe_recurrent_distillation.py`：退役 `_coordinated_shared_gradients` 的三次 `autograd.grad(retain_graph=True)`，改为单次融合 backward；`_candidate_metrics` 降频或退役；`assert_finite_grads` 降频；每 minibatch 的 KL all_reduce 合并
  - acceptance_criteria: 每 minibatch 仅一次图遍历；NCCL 集合通信数量较现状显著下降；learn p90 与 p50 的比值 < 1.5
  - verification_commands: `D:\anaconda\envs\pytorch\python.exe -m pytest tests/test_safe_recurrent_distillation.py -q`
  - success_definition: learn 的双峰停顿消失，显存峰值回到 20 GiB 以下
- [x] 阶段 4：新 lineage 开训
  - `legged_lab/envs/t4/depth_student_cfg.py`：`max_iterations` 25000→15000；`behavior_coef_decay_iters` / `teacher_mix_decay_iters` 覆盖全程而非 2000；FT 用现成的 `T4SparseDepthStudentFtAgentCfg`（1000 步）
  - acceptance_criteria: 新 tmux / 新 logdir / 新 run_name；`perf` 报告 total 中位数 ≤ 3.5 s；DAgger 项在 15000 步内不提前归零
  - verification_commands: `D:\anaconda\python.exe artifacts\tmp_s12_rtx_remote.py perf`
  - success_definition: 学生第一次在论文配方下蒸馏，总预算落到约 12–15 h
- [ ] 阶段 5：能力验收
  - acceptance_criteria: evaluator JSON + lineage manifest + 连续回放，梅花桩 / raised pillars 通过率有据可查
  - verification_commands: `legged_lab/scripts/eval_t4_hurdle.py`（学生需开深度传感器）
  - success_definition: 能力声明有第一手证据，而非 reward/length/checkpoint 存在
- [x] 阶段 0：性能根因已定位
  - acceptance_criteria: 拿到 n=147 的 collection/learn 分位数与 GPU 利用率；把 5.39 s collection 拆成 1.96 / +2.28 / +1.15；确认 GPU 空闲 75%
  - verification_commands: `D:\anaconda\python.exe artifacts\tmp_s12_rtx_remote.py perf`；`... inspect`
  - success_definition: 不再把「修好 RTX 出图」当成性能已解决

## Commit units

1. `student-raycast-depth-backend`：阶段 1 代码 + probe 证据 + 本计划 + `.harness` 同步。
2. `student-lightlp-obs-encoder`：阶段 2 观测/编码器 + 合同测试。
3. `distillation-single-backward`：阶段 3 更新期瘦身 + 测试。
4. `student-lightlp-recipe-cfg`：阶段 4 配方参数。

训练产物不进 git。用户未要求则不 commit。

## Known risks / blockers

- **停训授权是硬阻塞。** GPU 余量 <1 GiB，probe 不能与训练并发（13:04 已因此 OOM 一次）。
- **阶段 2 会作废 `model_3000` 热启动**，学生从零开始。交接文档已确认该 checkpoint 的 CNN 全程只见过陈旧 buffer，代价可接受，但要明确告知而不是默默丢弃。
- **光追深度看不见机器人自身肢体和其他机器人**，只看地形网格。对静态梅花桩找落脚点是想要的行为（论文 §VIII 也把上半身遮挡列为待解决的限制），但和真机 D435 的自遮挡有差异，需要在 §VI noise FT 阶段和 sim2sim 复核时说明。
- 2.28 s 归因于 RTX 管线是从基线差值推的，**不是 profiler 结果**；阶段 1 的 probe 就是用来证伪它的。
- 不要把 Windows 的 `scripts/nubot_run.sh` rsync 到 nubot。
- 远端 worktree 走 rsync 而非 git：`/home/nubot/phn_ws/t4_train/TienKung-Lab-s12-gru-ppo`。

## Next skill

阶段 1–4 墙钟已交付。质量问题转交 `docs/plans/2026-08-23--t4-sparse-s12-student-distill-recipe-plan.md`。Ready 只覆盖配方与开训墙钟，不覆盖过桩。

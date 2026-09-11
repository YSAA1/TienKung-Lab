# G1 视觉学生主线 + 教师巩固（v3 验收 / v3.1）计划

2026-09-11 立项。brainstorm 收敛结论：不走盲走中间档，主轴直指视觉学生；G1 真机在视觉策略就绪后直接部署。论文暂不锁 venue，素材按 RA-L 节奏随做随攒。

## 目标

1. **视觉学生线（主轴）**：把 T4 Stage E 深度学生链路确认、整理成"关键代码地图 + 换机器人接入清单"，平移到 G1 并落地感知补齐，训练出 G1 深度学生，通过 Isaac + MuJoCo sim2sim 双验收。
2. **教师线（副线，G1 优先）**：v3 跑满后三方对照验收；通过即 v3.1（编码器偏置 / COM / 高度图 dropout）重训 G1，作为学生蒸馏的教师基座。
3. T4/Z2 的 v3.1 重训排队跟进，非阻塞。

## 轨道 A：教师线

- [ ] A1 v3 验收（v3 30k 跑满后，ETA 09-11 晨）
  - Isaac evaluator 三方对照（v1 基准 / v3 / v2）：d=0 stones、d=1.0 vx=0.5，v1 基准 63/64、71.9% 作参照。
  - 本机 MuJoCo 探针 KPI（`artifacts/g1_gap_20260910/FINDINGS.md`）：d=1.0 vx=0.5 reach_2m ≥ 5/10，d≤0.5 与平地不回归。
  - 15k 监控点：tracking ≥ 0.5（v1 轨迹对照）；若 promotion 异常按 v1 对照复盘，不改门限。
  - 产物：evaluator JSON + lineage + 证据归档 `artifacts/g1_vital_v3/`。
- [x] A2 v3.1 三项 DR 增量（代码与单测已落，2026-09-11；重训待 A1 通过）
  - 编码器偏置 ±0.015 rad：`EncoderBiasCfg`（`legged_lab/config.py`），env 观测组装处 ramp 施加（构造期采样一次、全程恒定），`VITAL_V31_ENCODER_BIAS` 与 v3 同窗。
  - 根 COM 偏移：`vital_v31` profile 挂 `randomize_rigid_body_com` startup 事件，torso ±0.05 m 三轴；IsaacLab 2.1.0 正式版缺该函数，由 `legged_lab/mdp/events.py` 重导出 `motion_tracking` 的本地实现（静态合同测试锁定来源）。
  - 教师特权高度图 dropout：`HeightScannerCfg.occlusion_*` + `legged_lab/locomotion/mdp/scan_occlusion.py`（RPL 式侧向带遮挡，按扫描 y 外 x 内展平布局 `iy*nx+ix` 构造，仅 actor 流，critic 保持特权干净）。
  - 验收：`tests/test_g1_dr_ramp.py`（遮挡纯函数，按物理布局断言）+ `tests/test_g1_motion_experiment.py`（v31 profile 合同 + COM 函数来源合同）通过；本地全量测试绿；black/flake8 干净。审查修复轮（subagent 对抗审查）修正了遮挡展平方向与 COM 函数来源两个缺陷。
- [ ] A3 v3.1 G1 重训（依赖 A2）：单卡 ~29h；nubot tmux，沿用 v3 启动模板。
  - 验收：同 A1 KPI 对 v3 不回归（MuJoCo 探针是门不是参考）。
- [ ] A4 T4/Z2 v3.1 跟进（排队，非阻塞，单独开切片时再细化）。

## 轨道 B：视觉学生线（主轴）

- [x] B1 T4 Stage E 链路确认与整理（2026-09-11 完成）
  - 现行链路判定：S12 稀疏线（`train_t4_sparse_depth_student{,_ft}.py`，教师 1937D）；旧 Stage E 走跑线只标注不动。
  - 产出 [docs/runbooks/depth-student-line.md](../runbooks/depth-student-line.md)：关键代码地图 + 换机器人接入清单（第 3 节，B2 执行入口）；README 挂链接。
  - 澄清：现行学生线的**相机外参 DR 已存在**（`_apply_camera_extrinsic_jitter`，`student_camera_pos_jitter_m`/`ori_jitter_rad`，DPL Table II 式）；真缺口只剩内参（IsaacLab 相机内参全局共享，per-env 不可行——需自定义渲染或跨 run 随机，挂 backlog）。
  - 澄清 4 个训练脚本的沿革与现行链路：`train_t4_sparse_depth_student{,_ft}.py`（现行 s12 repr-first）vs `train_t4_depth_student{,_ft}.py`（旧 Stage E 走跑线），明确唯一权威入口。
  - 现行链路盘点：`DepthDistillationEnv`（`legged_lab/locomotion/depth_env.py`）→ `legged_lab/envs/t4/depth_student_env.py` + `depth_student_cfg.py`（三阶段配方 ReprFirst/DAgger/FinalMain + ResidualFt/PlantFt/Joint 变体）→ GRU 学生 → `sim2sim_t4_depth_student.py` / `eval_t4_depth_student_sim2sim.py`。
  - 产出 `docs/runbooks/depth-student-line.md`：关键代码地图（哪些是核心、哪些是历史）、训练-验收-导出命令、**换机器人接入清单**（`LocomotionRobotSpec` + 相机配置 + `depth_student_cfg` 平移步骤 + 测试入口，对照 AGENTS.md 测试表"共享深度学生运行时"行）。
  - 验收：文档路径/命令可执行（纯文档按 AGENTS 校对差异与路径）；`artifacts/portability/depth/import_depth_env.json` 探针现状写入文档。不顺手重构旧脚本——只标注，不动代码。
- [x] B2 G1 深度学生资产与环境（代码落地 2026-09-12，B4 启动前以 Isaac 实启为准）
  - 纯合同 `legged_lab/envs/g1/depth_student_contract.py`（102/1997 维数、相机四元数、扫描区间，本地可测）；环境 `depth_student_env.py`（TiledD455 原生 48×64，torso_link 挂载 0.10/0/0.25 + 35° 下视）；配方 `depth_student_cfg.py`（repr-first 主配方镜像 T4 数值）；入口 `train_g1_sparse_depth_student.py`（1997D 教师门 + manifest 对账）。
  - 测试 `tests/test_g1_depth_student_b2_contract.py` 4 项；本地全量 503 绿。
- [ ] B2 挂起项：G1 版 sim2sim 深度学生脚本（B4 训练中期探针需要，届时按 T4 `sim2sim_t4_depth_student.py` 模式做 G1 MJCF 版）。
  - G1 深度相机配置（模拟真实部署相机 FOV/安装位）；`depth_student_env` 的 G1 版（沿用 `LocomotionRobotSpec` 模式，不继承 T4 任务实现）。
  - 验收：`tests/test_robot_neutral_depth_env.py` + `tests/test_t4_sparse_depth_student_gru_contract.py` 的 G1 对应项通过（需要 torch 的环境）。
- [ ] B3 感知补齐四项（2026-09-11 部分完成）
  - [x] 块 dropout 按地形调档：`central_band_column_draw`（`depth_noise.py`）+ `student_depth_spare_lateral_for_sparse` 旋钮（默认关，向后兼容）——踏石 env 的 dropout 块避开两侧 1/4 边距。
  - [x] 相机内参/外参随机化：**外参已存在**（B1 澄清，T4 学生 cfg 已启用 `LIGHTLP_CAMERA_POS/ORI_JITTER`）；内参 per-env 不可行（渲染器限制），挂 backlog。
  - [ ] ResidualFt 在 G1 学生线启用（B2 建 G1 学生 cfg 时一并）。
  - [ ] 深度→高度图辅助监督头（B4 训练前冻结时一并）。
  - 相机内参/外参随机化（startup per-env，RPL/HPL 区间为参照）。
  - 块 dropout 按地形调档（踏石保留侧向视野；先做 0% vs 现状快速消融再定档）。
  - ResidualFt 在 G1 学生线启用（残差限幅=无盲走档的风险兜底）。
  - 深度→高度图辅助监督头（特权高度图现成标签，辅助权重 0.1–0.5，可拆卸）。
  - 验收：每项配单测（噪声/遮挡函数纯逻辑可本地测）；不动已验证的 `depth_noise` 既有函数语义。
- [ ] B4 G1 深度学生训练（依赖 B2+B3+A3：教师用 v3.1 checkpoint）
  - nubot，repr-first 三阶段；训练中期（DAgger 段）即插入 MuJoCo sim2sim 深度探针，不等终训——T4 s12 "Isaac 过门但 MuJoCo 稀疏仍摔"的教训：sim2sim 提前、当门用。
  - 蒸馏输入一致性（B2 立项时显式决策）：v3.1 教师训练期 actor 流带编码器偏置/扫描遮挡，而 `DepthDistillationEnv` 的 teacher_obs 是干净特权流——B2 的 G1 学生环境需决定是否对教师推理流施加同款腐蚀，避免教师"训时有噪、蒸馏时无噪"的输入分布失配。
- [ ] B5 双仿真器验收
  - Isaac evaluator：对照 T4 s12 的 Isaac hard 门定 G1 KPI（踏石/圆桩 reach_2m）。
  - MuJoCo sim2sim 深度探针（复用 `sim2sim_t4_depth_student.py` 模式做 G1 版）：稀疏地形 KPI 过门 + 平地不回归。
  - 产物：evaluator JSON + lineage manifest + 连续回放证据（AGENTS 能力声明合同）。

## 风险与依赖

- **学生线 sim2sim gap（T4 已证存在）**：无盲走档，风险兜底=B3 的 ResidualFt 限幅 + 内参随机化 + B4 的中期探针。若中期探针暴露系统性 gap，回退讨论（教师 v3.1 DR 加档或感知补齐加档），不硬冲训练。
- GPU 排队：v3.1（A3 单卡）与 B1–B3（本地无 GPU）天然并行；B4 需要 A3 产物 + GPU。
- 范围蔓延防护：B1 只文档化不重构；B3 四项之外的新想法（重建器整案、Δ相位/Δvx）挂 backlog 不进本轮。

## final_integration_claim

G1 深度学生端到端成立：以 v3.1 教师为蒸馏源，G1 深度策略在 Isaac 稀疏地形 evaluator 达到（对照 T4 s12 门标定的）KPI，且本机 MuJoCo sim2sim 深度探针过门、平地不回归；训练-验收-导出链路对 G1 完整可复用，第三机器人接入路径有文档与测试入口；教师侧 v3.1 相对 v3 在双方仿真器均不回归。

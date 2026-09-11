# 深度学生线操作手册（关键代码地图与换机器人接入）

2026-09-11 由 B1 整理产出（计划：`docs/plans/2026-09-11--g1-vision-student-and-teacher-v31-plan.md`）。
本手册回答三件事：**哪条是现行链路、关键代码在哪、换一个机器人要动哪几处**。

## 1. 两代链路沿革（只认 S12 线）

| 线 | 训练入口 | 教师 | 状态 |
| --- | --- | --- | --- |
| 旧 Stage E 走跑线 | `legged_lab/scripts/train_t4_depth_student.py`（+`_ft.py`） | Stage E HeightScan 教师（1155D） | done，历史保留，**不再开新训** |
| **S12 稀疏线（现行）** | `legged_lab/scripts/train_t4_sparse_depth_student.py`（+`_ft.py`） | `t4_loco_teacher_sparse` S12 教师（1937D） | 现行；`s12_repr_first` Isaac 过门、MuJoCo 稀疏仍摔，下一刀跟新教师 |

判定规则：教师是 1937D `t4_loco_teacher_sparse` 即现行线；入口脚本 docstring 自带此合同，误传 Stage E checkpoint 会被拒。

## 2. 现行链路关键代码地图

**共享层（机器人无关，G1/Z2 复用不再动）**

- `legged_lab/locomotion/depth_env.py` — `DepthDistillationEnv` / `LightLPDepthDistillationEnv`：教师特权流（proprio 历史 + 高度扫描）与学生可部署流（proprio + 深度历史）双观测组装。portability 探针：`artifacts/portability/depth/import_depth_env.json`（证明不依赖任务 registry）。
- `legged_lab/locomotion/schemas.py` — 观测合同唯一真值：`PROPRIO_FIELDS`（96D/帧）、`PROPRIO_HISTORY_LENGTH=10`、`DEPTH_POLICY_SIZE=(48,64)`、`DEPTH_HISTORY_LENGTH=3`、`DEPTH_UPDATE_DECIMATION=3`。
- `legged_lab/locomotion/mdp/depth_noise.py` — 深度真实感：距离相关噪声、块状 dropout（边缘偏向）、近/远边腐蚀、刷新计划。
- `rsl_rl/rsl_rl/algorithms/safe_recurrent_distillation.py` — `SafeRecurrentDistillation(PPO)`：门控 DAgger/BC+PG 系数调度（pg 0→0.1→0.2、behavior 1.0）。
- `rsl_rl/rsl_rl/algorithms/repr_first_switch.py` — repr-first 阶段切换。
- `legged_lab/utils/rsl_rl_compat.py` — IsaacLab/RSL-RL 观测适配。

**T4 特化层（换机器人要对应新建的部分）**

- `legged_lab/envs/t4/depth_student_env.py` — 环境族：`T4LocoSparseDepthStudentEnvCfg`（基）→ `…ReprFirstEnvCfg`（表征阶段上 hard 稀疏行）→ `…FtEnvCfg` → `…ResidualFtEnvCfg` / `…TargetedFtEnvCfg` / `…PlantFtEnvCfg`。机器人差异点全在这里：继承自机器人的教师 cfg、`_t4_student_depth_camera()`（D455 风格躯干相机，87° HFOV，270×480 出图）。
- `legged_lab/envs/t4/depth_student_cfg.py` — 配方族（agent/algorithm）：
  - `T4SparseDepthStudentReprFirstAgentCfg/…AlgCfg`：**现行主配方**（一轮式表征先行：教师驱动扫描 → Ross DAgger，无 PPO）；
  - `…FinalMainAlgCfg`（mix 2k + critic 200 + DAgger + pg 0.2 一轮式主蒸馏）；
  - `…JointAlgCfg`（Phase B 学生 DAgger + 保守 PPO）、`…DeployFtAgentCfg`、`…ResidualFtAgentCfg`、`…PlantFtAgentCfg`（续训变体）。
- `legged_lab/assets/t4/student_lineage.py` — S12 教师 checkpoint 与 evaluator manifest 门控（`--teacher_eval_manifest` 必须匹配冻结教师； waiver 需显式 `--allow_ungated_teacher`）。
- `legged_lab/scripts/train_t4_sparse_depth_student.py` — 主训入口：直接构造 env/agent cfg（**不走 task registry**，Stage E 学生线历来独立入口）。
- `legged_lab/scripts/train_t4_sparse_depth_student_ft.py` — 门控续训入口：`--mode joint|deploy_ft|targeted_ft|residual_ft|plant_ft`。

**验收层（本机 Windows，无 Isaac）**

- `legged_lab/scripts/sim2sim_t4_depth_student.py` — 交互式 MuJoCo 部署合同复现：96D proprio×10、48×64×3 深度（clip (0.2,3.0) m、no-hit 填 1.0）、MuJoCo 位置伺服增益/力矩限与 IsaacLab 资产（`legged_lab/assets/t4/t4.py`）一致、步态时钟仅在移动时推进（相位 0.38/0.88，air ratio 0.38）。
- `legged_lab/scripts/eval_t4_depth_student_sim2sim.py` — headless 固定命令评估器：固定命令/导航、进度/生存/摔倒 JSON、跟机 MP4。**这是"中期 MuJoCo 探针"的执行体**。

**测试入口（AGENTS.md 验证表"共享深度学生运行时"行）**

- `tests/test_robot_neutral_depth_env.py` — 机器人无关性（按关节数参数化，G1 接入时此测试直接覆盖）。
- `tests/test_t4_sparse_depth_student_gru_contract.py` — GRU 学生网络合同。
- `tests/test_sim2sim_t4_depth_student_runtime_contract.py` — sim2sim 运行时合同（需 mujoco）。
- `tests/test_safe_recurrent_distillation.py` — 配方系数调度合同。

## 3. 换机器人接入清单（以 G1 为例，B2 执行时照此逐项）

1. **教师就绪**：目标机器人的稀疏教师过 evaluator 门（G1=vital_v3.1，A 轨产物）。学生线入口的 manifest 门控即挂此验收 JSON。
2. **深度相机 cfg**：在 G1 侧新建 `…_student_depth_camera()`：选真实部署相机（G1 机头 D435 类 87° HFOV 对齐 T4 合同，或按实机改 `DEPTH_POLICY_SIZE`——改尺寸即改合同，须同步 schemas、sim2sim、导出三处）。
3. **环境 cfg**：`legged_lab/envs/g1/depth_student_env.py` 新建，继承 G1 教师 cfg（`motion_experiment` 的 vital profile），按 T4 的 `T4LocoSparseDepthStudentEnvCfg` 逐段对照抄写；机器人通过 `LocomotionRobotSpec` 接入，不继承 T4 任务实现（AGENTS 合同）。
4. **配方 cfg**：G1 版 agent/algorithm cfg（ReprFirst 主配方 + ResidualFt 变体先行），策略输入维度由 `ObservationLayout` 推导，不手写。
5. **训练脚本**：克隆 `train_t4_sparse_depth_student.py` 的 G1 版（改教师 lineage 门控与 cfg import），沿用"不走 registry"惯例。
6. **sim2sim**：`sim2sim_g1_depth_student.py` + headless 评估器 G1 版：G1 MJCF、伺服增益/力矩限取自 `legged_lab/assets/g1/`、步态时钟参数用 G1 教师合同值。
7. **测试**：跑 `tests/test_robot_neutral_depth_env.py`（参数化自动覆盖）+ G1 版 GRU/sim2sim 合同测试。
8. **验收**：B4 训练中期起用 G1 版 headless 评估器做 MuJoCo 探针（Isaac/MuJoCo 双曲线），B5 按 evaluator JSON + lineage + 连续回放出证据（AGENTS 能力声明合同）。

## 4. 已知教训（G1 线必读）

- T4 `s12_repr_first` Isaac hard 过门但 **MuJoCo 稀疏仍摔**：Isaac 指标不替代 sim2sim；G1 线把 MuJoCo 探针提前到 DAgger 中期并当验收门。
- 旧 Stage E 脚本与归档计划（`docs/archive/plans/2026-08-2*.md`）是配方演化史，只读不改；现行配方以本手册第 2 节为准。

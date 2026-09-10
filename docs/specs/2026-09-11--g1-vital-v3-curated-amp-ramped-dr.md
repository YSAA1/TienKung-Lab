# Spec - G1 VITAL v3：curated AMP + 端到端 ramp 式 plant DR

> 状态 / Status: user-approved
> Owner: user / agent
> Date: 2026-09-11
> 来源请求 / Source request: 用户确认 v5 AMP 数据初始帧 T-pose 污染（双手展平），要求先证明数据可用性、参考成熟开源配方给出迁移修复终极方案，端到端单 run（教师阶段 DR 从头到尾），不用热启动修补；落完后对抗审查。

## 背景

配对回放（`artifacts/g1_gap_20260910/FINDINGS.md`）证明 Isaac↔MuJoCo 的 gap 是"难例余量耗尽"而非管线 bug：部署链路 t0 一致（1.9e-6），首步踝部瞬态发散、踝/肩/腕出力常饱和。`vital_v2` 冷启动全量 DR（delay 0–2、摩擦 0.4–1.2/0.3–1.0、reset 速度 ±0.3）把 tracking_mean 压到 0.437 < 0.5 晋级门，地形等级停滞 ~3。

AMP 数据验收（`artifacts/g1_amp_acceptance_20260911/`）发现 `unitree_v5`（每段前 899 帧）包含：每段前 ~60 帧 T-pose 标定姿态（|肩外展|>0.6 rad 占比 92–100%）、walk1/walk3 脚底打滑（support proxy p95 = 1.85/1.06 m/s）、大段低速内容（walk4 前 30s 均速 0.10 m/s）。视觉 MCP 与定量统计双重证实。开源数据集调研（unitree_rl_gym、humanoid-gym、unitree_rl_mjlab、BeamDojo）表明：所有 mocap 源都有标定起始帧，换数据集不解决该问题；BeamDojo 在同类稀疏落脚任务用 ±10% 出力+±15% Kp/Kd 从零收敛；没有一个成熟项目用 reset 初速度随机；硬动作延迟仅 humanoid-gym 用软混合且无人用 0–2 步硬延迟；IsaacLab 维护者明确建议 DR 课程化（"不要从第一步上强 DR"）。

## 目标

- AMP expert 换为 `motion_amp_expert_unitree_v6`：`curate_g1_amp.py` 从同源 LAFAN CSV 精选的 8 段干净步态周期（T-pose/打滑/低速段全部排除），通过 Isaac `validate_g1_curated_amp.py` 与视觉验收。
- 新增端到端 `vital_v3` 配方：保留 vital_v1 全部奖励/终止/步态设定，plant DR 从第 0 步生效并按线性 ramp 在 36000 策略步（≈1500 iter）内从 v1 等效条件渐进到满幅——增益 (0.9,1.1)、摩擦 static (0.6,1.2)/dynamic (0.5,1.0)、动作延迟 (0,1)。
- 30k 冷启动训练在 nubot GPU0/2（v2 仍占 GPU1/3；Z2 已完成释放 GPU0/2）。

## 非目标（Non-goals）

- 不改 T4/Z2 配置与共享 `DomainRandCfg` 默认值。
- 不加 reset 初速度随机（v2 回退项）；不加编码器零偏 DR（留待真机阶段）。
- 不修 MJCF 脚部几何（配对回放结论的 plant 对齐路线另行处理）。
- 不在本切片宣称 MuJoCo 稀疏落脚已修复；以评测器 JSON + MuJoCo 探针为后续验收。

## 用户 / 调用者（Users / Callers）

- 训练：`scripts/train_g1_vital_v3.sh` → `train.py --g1_motion_experiment vital_v3 --amp_expert_manifest .../motion_amp_expert_unitree_v6/_manifest.json`。
- 评测/回放：与 vital_v1 相同评测配方，AMP 数据集按 manifest 指定。
- 本机 MuJoCo：`play_t4_sparse_teacher_mujoco.py --robot g1`（不变）。

## 行为规格（Behavior Spec）

### 正常路径（Happy Path）

- `--g1_motion_experiment vital_v3` 且冷启动、显式 AMP manifest 指向 `unitree_v6`。
- 先应用与 `vital_v1` 完全相同的终止/步态/action_rate；再挂三类 ramp 事件：
  - `physics_material`（startup）保持 v1 区间，`physics_material_ramp`（interval 45–60s）按 ramp 插值重采样；
  - `actuator_gains`（reset）以 1.0 为锚点渐扩到 (0.9,1.1)；
  - `action_delay_reset`（reset）使 P(延迟=1) 随 ramp 增长，上限 1 步（20ms）。
- ramp 进度 = `sim_step_counter // decimation / 36000`，夹在 [0,1]。

### 边界情况（Edge Cases）

- 非 G1、resume、或缺少 AMP manifest：拒绝（沿用 train.py 现有守卫）。
- 评测加载 registry cfg 时无该 flag，DR 事件不存在、delay 关闭。
- ramp 完成后（≥1500 iter）各项 DR 恒满幅，与地形课程解耦。

### 接口 / 状态（Interfaces / State）

- `legged_lab/mdp/ramp.py`（新）：纯 ramp 数学，可脱离 Isaac 导入。
- `legged_lab/mdp/events.py`：`randomize_rigid_body_material_ramped`、`randomize_actuator_gains_ramped`、`randomize_action_delay_ramped`。
- `legged_lab/envs/g1/motion_experiment.py`：`vital_v3` profile 与常量。
- `legged_lab/scripts/train.py`：CLI choices 增加 `vital_v3`。
- `legged_lab/envs/g1/datasets/motion_amp_expert_unitree_v6/`：8 段 + `_manifest.json`（sha256）。

## 验收（Acceptance）

- 本机 `pytest tests/test_g1_motion_experiment.py tests/test_g1_dr_ramp.py tests/test_g1_asset_contract.py` 全过。
- nubot `validate_g1_curated_amp.py --data-dir .../unitree_v6` 全过（70D 特征回读误差 < 2e-4）。
- 开训核验：`params/env.yaml` 含三个 ramp 事件、delay (0,1)、无 reset 速度；`agent.yaml` AMP 文件为 v6 的 8 段；model_0 存在；GPU0/2 各 ~8.5GB。
- 中止线（训练期监控）：~3000 iter 时 tracking_mean < 0.45 或地形等级落后 v1 轨迹 ≥2 级 → 停止并复盘。

## 约束（Constraints）

- GPU1/3 留给正在跑的 vital_v2（26.5k/30k），v3 只用 GPU0/2。
- 控制步 20ms；delay 上限 1 步。
- 不热覆盖任何正在运行的工作树；v3 使用独立目录 `TienKung-Lab-g1-vital-v3-20260911`。
- 组合实验（curated AMP + ramp DR 同跑），不声称单变量因果。

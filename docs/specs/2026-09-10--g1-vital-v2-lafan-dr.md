# Spec - G1 VITAL v2：LAFAN AMP + 训练期 plant DR

> 状态 / Status: user-approved
> Owner: user / agent
> Date: 2026-09-10
> 来源请求 / Source request: 替换 T4 rob2rob AMP 为正确人体重定向数据，并加针对性 DR 后冷启动 3 万轮；用户确认方案 B 及全部推荐数值，并要求不再询问、直接执行。

## 背景

G1 VITAL `model_29999` 用 T4→G1 rob2rob 17 段 AMP。走路专家肘锁在约 0.13 rad，策略慢走肩 pitch 约 0.88，比专家更夸张。仓库已有 LAFAN1 人体重定向 6 段（`motion_amp_expert_unitree_v5`），肘约 0.84。Isaac 高难慢走能过桩，本机 MuJoCo 同速常过不去；部分差距是几何/评测协议，部分是延迟与 PD/接触。G1 默认 `action_delay` 关闭、reset 速度为空。

## 目标

- 新冷启动 30k 教师使用 LAFAN `unitree_v5`，不再使用 T4 rob2rob 17 段。
- 仅在训练期对 G1 施加一小包 plant DR（delay、摩擦、reset xy/yaw、PD 增益）。
- 保留 VITAL 终止、步态门控关闭、action_rate -0.01。
- 在 nubot GPU1/3 开训；不碰 Z2 占用的 GPU0/2。

## 非目标（Non-goals）

- 不重做 T4→G1 手臂 IK。
- 不改 T4 / Z2 默认 `DomainRandCfg` 或正在跑的 Z2 源码。
- 不改稀疏最低速度 0.6、不改 Isaac 砖几何、不加梅花桩制造抖动。
- 不把关节 reset 缩放到 0.5–1.5，不恢复 z/roll/pitch reset 速度。
- 不在本切片注册或开 G1 深度学生。
- 开训健康 ≠ 行为验收；30k 跑完前不宣称 Isaac/MuJoCo/实机已对齐。

## 用户 / 调用者（Users / Callers）

- 训练：`scripts/train_g1_vital_v2.sh` → `train.py --g1_motion_experiment vital_v2` + LAFAN manifest。
- 评测/回放：与 VITAL 相同终止配方（`--profile vital_v1`），AMP 用 `unitree_v5`；评测不启用 delay。
- 本机 MuJoCo：`play_t4_sparse_teacher_mujoco.py --robot g1`。

## 行为规格（Behavior Spec）

### 正常路径（Happy Path）

- `--g1_motion_experiment vital_v2` 且冷启动、显式 AMP manifest 指向 `unitree_v5`。
- 先应用与 `vital_v1` 相同的终止/步态/action_rate，再应用 plant DR。
- 默认 `g1_loco_teacher` 无该 flag 时 delay 仍为关，摩擦/reset 保持 G1 教师原值。

### 边界情况（Edge Cases）

- 非 G1、resume、或缺少 AMP manifest：拒绝。
- 与 `--g1_progress_ab` 互斥。
- 评测加载 registry cfg 时不得偷偷打开 delay。

### 接口 / 状态（Interfaces / State）

- `legged_lab/envs/g1/motion_experiment.py` 增加 `vital_v2`。
- `legged_lab/scripts/train.py` CLI choices 同步。
- `unitree_v5/_manifest.json` 增加可供 `train.py` 校验的 `clips`+sha256。
- 启动脚本与远端独立工作树、tmux、TensorBoard 端口写入计划与 `.harness/`。

## 约束（Constraints）

- GPU0/2 留给 Z2，不得占用。
- 控制步 20 ms；delay 仅 0–2 步。
- 远端不热覆盖 VITAL 或 Z2 冻结目录。
- 组合实验：AMP 与 DR 同跑，不声称单变量因果。

## 选定方案（Chosen Approach）

`vital_v2` = `vital_v1` + 训练期 plant DR + LAFAN `unitree_v5`。评测用 `vital_v1` 终止开关、默认 delay=关。

DR 数值：delay 0–2；静/动摩擦 (0.4, 1.2)/(0.3, 1.0)；reset 速度仅 xy、yaw ±0.3；关节 reset 保持 (1.0, 1.0)；kp/kd × 0.9–1.1；质量与推扰不变。

## 拒绝方案（Rejected Options）

- 只换 AMP 不加 DR：无法覆盖本次 sim2sim 目标。
- 重做 T4 IK：今天无法开训，且不是人体风格。
- delay 0–5 步：稀疏踏石 30k 过猛。
- 把 DR 写进 G1 `teacher_cfg.py`：污染默认任务与 Z2 对照。

## 验证策略（Verification Strategy）

### 基线证据（Baseline Evidence）

- VITAL `model_29999` Isaac 评测与硬+慢回放；T4 rob2rob 与 LAFAN 肘/肩统计。

### 自动检查（Automated Checks）

- `python -m pytest tests/test_g1_motion_experiment.py tests/test_g1_asset_contract.py`
- `pre-commit run --files` 本次修改文件

### Smoke / E2E 检查

- 远端冷启动：tmux 存活、GPU1/3 占用、model_0、loss 有限、保存的 env.yaml 含 LAFAN 路径与 delay 开。
- 30k 后（本切片不阻塞开训）：Isaac `vx=0.7` easy 踏石/圆桩；本机 MuJoCo 同命令；平地手臂观感。

### 负向 / 边界检查（Negative / Boundary Checks）

- `vital_v1` 不打开 delay。
- 非 G1 调用 `vital_v2` 在突变前失败。
- G1 `teacher_cfg.py` 仍写 `action_delay.enable = False`。

### 文档 / 状态检查（Documentation / State Checks）

- `docs/README.md`、`.harness/work_index.md`、计划与启动工件。

### 完成前所需 fresh evidence

- 开训：远端 batch 日志 + 保存的 env/agent yaml + GPU 快照。
- 能力：30k 后 evaluator JSON + 回放；本切片只证明开训合同。

## 能力缺口（Capability Gaps）

- Windows 本机无 Isaac Docker，开训与 Isaac 评测必须在 nubot。
- 实机部署本切片无法测；MuJoCo 为 sim2sim 代理。
- Fallback：CPU 合同测试 + 远端启动日志。

## 成功标准（Success Criteria）

- `vital_v2` 在无 Isaac 的单元测试中应用推荐 DR 与 VITAL 奖励/终止；`vital_v1` 行为不变。
- LAFAN 6 段 clip 哈希与 `unitree_v5` 文件一致，训练 CLI 钉死该 manifest。
- nubot 新工作树冷启动 30k，GPU1/3，未使用 GPU0/2；model_0 存在且 AMP 目录不是 rob2rob full17。

## 残余风险（Residual Risks）

- 历史上 G1 冷启动 delay+增益曾伤害学习（v6）；本次有更好 plant 与 LAFAN，仍可能学慢。缓解：保留 VITAL 为对照，不改默认 G1 cfg。
- AMP 与 DR 绑定，无法拆因果。
- Isaac 砖 vs MuJoCo 走廊仍在。

## Plan 交接（Plan Handoff）

- 当前切片 / Active slice: 落地 `vital_v2` + LAFAN 训练钉扎，并在 nubot GPU1/3 冷启动 30k
- 建议下一 skill / Suggested next skill: plan
- 计划提示 / Planning notes: 先测后改；远端复制独立目录 overlay，不热覆盖 VITAL/Z2
- 建议里程碑 / Suggested milestones: 合同与启动；30k 行为验收（后续切片）
- 里程碑验收提示 / Per-milestone acceptance hints: 本切片过门=测试绿+远端 model_0+yaml 合同；能力声明另开切片

# Executable Plan - S12 梅花桩收尾边框 + 轻转，以及后续 GRU 深度学生

> Status: active
> Date: 2026-08-22
> Spec: `docs/specs/2026-08-15--t4-stepping-stones-and-hurdle-stable.md`（梅花桩目标）；学生合同 `docs/specs/2026-08-12--t4-unified-depth-locomotion.md`；GRU/蒸馏参考 `docs/research/2608.02653v1`（LightLP §IV / §VI，**不是**执行权威）
> 取代: `docs/archive/plans/2026-08-21--t4-sparse-s11-mdp-repair-plan.md`（S11/S11b 配方保留为对照 lineage，本文件成为梅花桩轨道现行计划）
> Branch: `t4-train`
> Planning surface: docs plan
> GPU: nubot 四卡开 S12 老师热启；学生 waiter 等 `model_23999.pt`；zhuoqun 翻箱不动

## Objective

修掉踏石/圆桩「走完最后一块就掉进洞、却被记成 OOB 成功」的几何合同，并略微增加轻转，让机器人在桩上多踩几脚、学会小转弯，而不是 80% 直线冲出界。

老师过固定 evaluator 门之后，再开 **sparse 深度学生**：在 **同一套 S12 地形/命令/终止** 上蒸馏，网络按 LightLP §VI 加上 GRU 与 height-scan 重建辅助损失。旧 Stage E 学生 `stage_s_head35` 不覆盖。

## Active slice

**阶段 2（当前）**：nubot **从 S11b `model_19000` 热启** S12 老师（用户 2026-08-22 改口，不再冷启动）。worktree `TienKung-Lab-s12-from-s11b-5k`，run `t_sparse_lightlp_s12_from_s11b_5k`，`--reset_optimizer`，目标 19000→24000。阶段 1 边框+轻转合同与阶段 3 GRU 学生代码已落地；学生 waiter 等 `model_23999.pt`，不 resume `stage_s_head35`。

## 冻结数值（本切片不再讨论）

| 项 | 取值 | 理由 |
| --- | --- | --- |
| 边框宽度 `T4_SPARSE_RIM_WIDTH` | **0.75 m** | 硬踏石最后外沿离中心约 3.37 m；0.50 m 边框内沿在 3.50 m，中间还会留缝。0.75 m 内沿在 3.25 m，easy/hard 踏石和圆桩都会压到边框上 |
| 出界 | 仍 Chebyshev **4.25 m** | 邻砖 0.75 m 边框覆盖这 0.25 m |
| 晋级 | 仍 path > **4.0 m** 且 tracking ≥ 0.5 | 不改 LightLP 半砖条 |
| 轻转 | **40%** 抽 `wz∈[-0.3,0.3]`，**60%** `wz=0` | 「稍微加大」；多数仍直行，避免绕圈刷 path 晋级 |
| `vx` 踏石/圆桩 | 仍 `[0.6, 2.0]` | 不加慢走来凑 1000 步 |
| 终止阈值 | accel 40、Trunk 1 N、horizon 20 s、pit_fall 只记账 | 本切片不靠改终止掩饰掉坑 |
| 跨栏命令 | **不改** | 仍全向 + 20% 站住 + heading |

直线 0.6 m/s 出界上限仍约 350 步。**不要把 mean episode length → 1000 当成功。**

## 几何合同（边框）

出生在砖中心。边框是贴着四边、高 z=0 的一圈实地，宽度 0.75 m（砖内沿到砖边）。中间仍是洞 + 出生台 1.6 m + 踏石/圆桩格子。

石头可以和边框重叠，**不要**为了避让边框删掉外圈石头，否则硬踏石和边框之间又会留缝。

邻砖关系：踏石列 6–8 的 +x 仍是踏石，列 9 接圆桩；圆桩 10–12 的 +x 仍是圆桩。每块都有边框后，走出本砖 0.25 m（OOB 点）踩的是邻砖边框，不是邻砖的洞。

代数验收（Isaac-free）：

- `d ∈ {0, 0.39, 1}` 的踏石和圆桩：最后一块支撑外沿 ≥ 边框内沿（重叠或相接，缝 ≤ 0）。
- 沿 +x 的点 `origin + (4.25, 0)` 落在「邻砖边框」的 support 上（用同一 `point_on_support` / 新 `point_on_rim`）。
- 出生台外沿到边框内沿仍 ≥ 2 m，不能沿着边框抄近道绕过梅花桩。
- `illegal_footstep` 把边框当合法支撑。

网格实现落在 `stepping_stone_layout.py` + `terrain_generator_cfg.py::_sparse_base_meshes`（边框 mesh + 现有 pit/platform/stones）。

## 命令合同（轻转）

只覆盖 `stepping_stones` / `raised_pillars`：

- `SPARSE_FOOTHOLD_STRAIGHT_YAW_PROB = 0.60`
- `cfg.sparse_command_straight_yaw_prob = 0.60`
- 采样、overlay、每步 enforce、reset 后 resample 的路径不变
- 跨栏 / flat / 楼梯 / rough 仍走默认全向命令
- play / eval 钉死命令时仍关闭 overlay

## 学生阶段（阶段 3–4，老师门之后才开工）

契合第一阶段 = 学生 **继承 S12 老师 MDP**，不要拿 Stage E 学生环境去蒸梅花桩老师。

现有缺口：`T4LocoDepthStudentEnvCfg` 继承的是 `T4LocoTeacherEnvCfg`（Stage E），FT 只换了 sparse 地形和奖励，**没有** LightLP 终止、地形感知命令、边框。S12 学生必须 `T4LocoSparseDepthStudentEnvCfg(T4LocoSparseTeacherEnvCfg)`。

在此基础上按 LightLP §IV.A / §VI 加学生内容（论文只参考，不搬翻箱/技能组）：

| 老师（已有，保持 MLP） | 学生（要加） |
| --- | --- |
| HeightScan ×5 + 接触 2，无 RNN | 深度 CNN（沿用 T4 `DepthStudentTeacher` 编码器）+ 本体史 + 命令 + 上一动作 |
| 不给 GRU | 融合后再进 **GRU**（`rnn_type=gru`，默认 1 层 hidden 256，老师 `teacher_recurrent=False`） |
| 无重建头 | 训练期解码器 `D(h_t) → teacher scan`，`L_recon = MSE`；**导出时丢掉** |
| PPO+AMP | DAgger（动作模仿）**同时**开 PPO（论文 DAgger+PPO；现成 `Distillation` 的 `pg_coef` 可复用） |
| 干净 scan | 深度噪声：`σ=0.005+0.02d`、全局尺度 ±5%、块 dropout、30–60 ms 延迟、约 30 Hz 持帧；然后再短 FT |

明确 **不** 做（留给翻箱 G3）：多专家技能组合、vault transition 组、skill 标签。

导出合同仍冻结：深度 + 本体史 + 命令 + 上一动作 + GRU 隐状态；无 HeightScan、无接触真值、无重建头、无 Critic。

旧 `pi_loco` / `stage_s_head35` / `model_25746` 只作走跑部署对照，不 resume 进梅花桩学生。

## Non-goals

- 不改 OOB 4.25 / 晋级 4.0 / accel 40 / Trunk 1 N / 20 s 重采样。
- 不开 pit_fall 硬终止，不把摔倒误当出界来「修 length」。
- 不恢复踏石站住、倒退、侧移、`heading_command` 随机大朝向。
- 不改跨栏命令，不改 Stage E `t4_loco_teacher` 1155D。
- 不在 S11b logdir 上原地热补；S12 用**新 worktree / 新 logdir**，从 `model_19000` 加载并 reset optimizer，再训 5k。
- 老师未到 `model_23999.pt` 前不蒸学生；不 resume Stage E `stage_s_head35`。
- 不把 LightLP 的 vault / climb / transition 组搬进梅花桩学生。
- 不动 zhuoqun 翻箱。
- 不用 TB、reward、ckpt 存在、episode length 宣称能力。
- 不把 length 回到 1000 写成目标。

## Success criteria

**老师 S12**

1. Isaac-free 测试：边框几何、`point_on_rim`、hard 无缝、OOB 点在邻砖边框上、轻转 40%±1%、eval/play 钉死命令不被覆盖。
2. Stage E 老师命令/终止/奖励不变。
3. S12 热启：新 worktree/logdir，从 S11b `model_19000` 加载、reset optimizer，训到 24000。
4. 到 24000 前后用同一套 fixed evaluator 对照 S10 / S11b：踏石/圆桩 `d=0/0.39/1.0`，`vx=0.7/1.0`。不要把 length→1000 当成功。
5. play 回放：走完踏石/圆桩后应踩在边框上再出界，而不是先掉进洞。`Reset/oob` 可以高；`Reset/pit_fall` 与 `Reset/fall_over` 不应随出界一起涨。

**学生（阶段 3–4）**

6. 学生 env 继承 S12 sparse 老师 MDP；Actor 无特权泄漏。
7. 合同：GRU 前向、reset 清隐状态、`L_recon` 只训练、export 无解码器。
8. 同一冻结 evaluator + 深度消融（深度全零/打乱应掉能力）+ MuJoCo 深度 Sim2Sim 回放。
9. 蒸馏成功 = evaluator 接近老师阈值，不是 BC loss 下降。

## Verification path

```text
阶段 1
python -m pytest tests/test_t4_sparse_command_contract.py tests/test_t4_terrain_column_map.py tests/test_t4_sparse_reward_contracts.py tests/test_t4_sparse_monitor_contract.py tests/test_t4_sparse_evaluator_contract.py -q
  + 新增/扩展踏石布局测试（rim 宽度、hard 无缝、OOB 点 support、illegal_footstep 认边框）
  -> review
  -> 阶段 2 nubot 隔离 worktree 从 S11b `model_19000` 热启 5k
  -> ~24000 同一 fixed evaluator + play 边框收尾
  -> `model_23999.pt` 后 waiter 开阶段 4 蒸馏（阶段 3 代码已绿）
```

### Verification path status

`runnable`：阶段 1 本机 pytest 已过。阶段 2 热启已在 nubot 跑。阶段 4 蒸馏在 `model_23999.pt` 前 **gated**。

## Required capabilities

- 本机 pytest（无 Isaac）做几何/命令合同。
- nubot 4×GPU + `scripts/nubot_run.sh` + tmux（阶段 2、4）。
- S10 / S11b ckpt 只读对照。
- 现成 `rsl_rl`：`Memory`(GRU)、`StudentTeacherRecurrent`、`DepthStudentTeacher`、`Distillation`(behavior + pg_coef)。
- 学生阶段需要 T4 深度相机合同（已有 D455 cfg）和 `sim2sim_t4_depth_student.py` 扩展 GRU export。

## Fallback evidence

- 本机无 Isaac：边框/轻转用纯 Python 布局 + 命令代数。GPU smoke 看 nubot 开训 log 不即崩。
- 边框观感：10k 前可用 1-env play 短回放，不替代 evaluator。
- 若 10k hard `d=1` 相对 S11b 无改善：停长训，先查边框 mesh 是否没进碰撞 / 扫描是否仍把边框当洞，不先松 accel。
- 学生 GRU 若 DAgger 不收敛：先做「老师开车 BC」信息瓶颈（与翻箱 G2 同一纪律），再开学生开车 DAgger。

## Final integration claim

`final_integration_claim`: S12 是梅花桩老师的几何/命令修复 lineage（0.75 m 收尾边框 + 40% 轻转，终止阈值不变）。后续 sparse 深度学生必须吃这套 MDP，并在现有深度 CNN 学生上加 GRU 与 scan 重建辅助，用 DAgger+PPO 再短 FT。本计划不声明 hard 梅花桩已会走，也不声明学生已可上机。能力仍要 evaluator JSON + lineage + 连续回放。

## 工作项

- [x] 阶段 0：S11b 出界/掉落诊断基线
  - acceptance_criteria: 已确认 length~300 来自 4.25 m OOB；走完踏石掉落主要是 OOB 不是 fall_over；最后支撑 3.37–3.78 m，晋级/OOB 在空洞里
  - verification_commands: S11b TB `Reset/oob≈0.52`、`Reset/horizon≈0.06`、`Reset/pit_fall≈0.016`、`Reset/fall_over≈0.01`；布局脚本 last-edge vs 4.25
  - success_definition: 根因是几何合同，不是 length 计数 bug

- [x] 阶段 1：S12 边框 + 轻转合同与实现
  - acceptance_criteria: rim=0.75 m 写入布局真值；hard 踏石/圆桩无支撑缝；OOB 点在邻砖边框；straight_prob=0.60；vy=0、无站住/倒退；play/eval 钉死命令；相邻 sparse pytest 绿；S11 计划标 superseded
  - verification_commands: `python -m pytest tests/test_t4_sparse_command_contract.py tests/test_t4_terrain_column_map.py tests/test_t4_sparse_reward_contracts.py tests/test_t4_sparse_monitor_contract.py tests/test_t4_sparse_evaluator_contract.py tests/test_t4_sparse_evaluator_contract.py -q` 以及新增 rim 布局测试
  - success_definition: 代码合同与上表一致，训练行为只改边框和轻转比例

- [ ] 阶段 2：nubot 从 S11b `model_19000` 热启 5k 并对照（当前）
  - acceptance_criteria: S11b 已停；worktree `TienKung-Lab-s12-from-s11b-5k`；run `t_sparse_lightlp_s12_from_s11b_5k`；reset optimizer；19000→24000；fixed evaluator + play 显示边框收尾而非掉洞
  - verification_commands: nubot `tmux ls` / TB `:8016`；`artifacts/eval/` 同口径 JSON；短 play 回放
  - success_definition: 新 lineage 在跑，且 24000 有与 S10/S11b 同口径的固定评估 + 边框观感，而不是 TB success 或 length→1000

- [x] 阶段 3：sparse GRU 深度学生合同与实现（代码已落地，开训等阶段 2 门）
  - acceptance_criteria: 新 env cfg 继承 `T4LocoSparseTeacherEnvCfg`；`DepthStudentTeacher`+GRU；训练期 scan recon；export 无 recon；DAgger+PPO 系数可配；Isaac-free 模块测试；不改 Stage E 学生任务名
  - verification_commands: 新增 `tests/test_t4_sparse_depth_student_gru_contract.py` 以及现有 distillation/obs 合同
  - success_definition: 学生观测/网络/损失合同可蒸馏 S12 老师，且无特权泄漏

- [ ] 阶段 4：学生蒸馏训练与验收（阶段 3 之后）
  - acceptance_criteria: 冻结 S12 teacher ckpt 作 expert；DAgger+PPO 再噪声 FT；同一 fixed evaluator；深度消融；MuJoCo 深度 Sim2Sim 回放；lineage manifest
  - verification_commands: `artifacts/eval/s12_student_*`；`sim2sim_t4_depth_student`（GRU export）；消融 JSON
  - success_definition: 学生在冻结桶上接近老师，且打乱深度会掉能力；不靠 loss 宣称成功

## Commit units

1. `sparse-s12-rim-yaw-contract`：阶段 1 布局/命令/测试 + 本计划落盘 + S11 superseded + recovery 同步。
2. 训练产物不进 git。
3. `sparse-s12-depth-student-gru`：阶段 3 代码与测试（老师门后另开）。

提交前置：该 unit 实现完成 + review 无 Critical + 对应 pytest 绿。

## Known risks / blockers

- nubot 四卡现由 S12 热启占用。S11b ckpt 只读对照，不要第二份四卡训练。
- 边框若只加视觉 mesh、没进碰撞，观感仍会掉。测试必须覆盖 support 代数；开训后用 play 看脚是否踩在边上。
- 扫描若仍把边框当洞，illegal_footstep 会罚合法收尾。边框必须进入 `point_on_support`。
- 40% 轻转仍可能用 path_length 绕圈晋级。若 `promotion_rate` 升而 `reach_4m` 不升，下一刀是把晋级改回径向，而不是再加大 `wz`。
- 老师 Actor 已有 scan×5；**不要**给老师加 GRU，否则和学生的信息差反过来。
- 现成 `StudentTeacherRecurrent` 默认 `rnn_type=lstm`。学生必须显式 `gru`。
- 现成深度学生 `pg_coef=0` 纯 BC。梅花桩学生按论文要 DAgger+PPO，否则踩空无法恢复。
- 深度 CNN 已有，不要无故改成论文的 depth-MLP；GRU 加在 CNN+本体融合之后。
- 翻箱 G2 未过箱，禁止把 G3 多专家过渡塞进本学生。

## Next skill

`review`

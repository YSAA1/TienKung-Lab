# Executable Plan - T4 特权专家到深度学生 Locomotion

> Status: active
> Date: 2026-08-12
> Spec: `docs/specs/2026-08-12--t4-unified-depth-locomotion.md`
> Branch: `t4-train`
> Planning surface: docs plan
> Training route: privileged teacher + adaptive curriculum -> depth student distillation

## Objective

基于 TienKung-Lab 的 velocity tracking + PPO + AMP 主干，交付一个统一的 T4 27DoF
深度感知 locomotion Actor `pi_loco`。训练路线改为：

```text
Stage E: privileged pi_teacher
    proprio history + local velocity command + previous action + local terrain privilege
    -> PPO + AMP + adaptive curriculum
    -> learns walk / jog / strong rough / up-down stairs / local composition

Stage S: depth pi_loco
    depth history + proprio history + local velocity command + previous action
    -> distillation / DAgger / optional short PPO fine-tune
    -> final exported deployment Actor
```

最终部署 Actor 的冻结接口为：

```text
depth history + proprio history + local velocity command + previous action
    -> depth encoder + locomotion actor
    -> 27DoF action in T4_JOINT_NAMES order
```

`pi_teacher` 可以在训练期使用局部 HeightScan / elevation scan 等可由深度近似推断的地形
特权；`pi_loco`、导出接口和 MuJoCo Actor 输入不得包含 HeightScan、高程图、接触真值、
地形真值、route progress 真值或 teacher-only modules。

## Active Slice

当前只执行训练前合同闭环 M0：在真实 IsaacLab 环境验证 T4 spawn、body/joint/camera
frame 和全部 motion playback，生成结构化审计结果。M0 未通过前不实现正式 AMP expert、
不启动 teacher 长训练，也不启动 student 蒸馏。

## Non-goals

- 不在本计划内完成真机部署或真机安全验收。
- 不保留 walk/run 多策略，不使用 MoE、技能 ID 或运行时 checkpoint 切换。
- 不把 `pi_teacher` 当部署策略；最终只部署 `pi_loco`。
- 不让最终 `pi_loco` Actor 接收 HeightScan、高程图、接触真值、地形参数或 route progress
  真值。
- 不让 locomotion Actor 承担全局路线规划。
- 不用五次正式长训分别训练 walk、jog、rough、stairs 和 route；它们是 curriculum /
  evaluator milestones。
- 不用兼容层保留旧 20DoF/52D AMP 或现有 20DoF MuJoCo runner 合同。
- 不把训练预算跑完、reward 上升、episode 变长、distillation loss 下降或 checkpoint 存在
  定义为阶段成功。

## Success Criteria

最终完成必须同时满足：

1. 只有一个导出的 T4 部署 Actor `pi_loco`，输入输出合同在 student 训练、导出、IsaacLab
   和 MuJoCo 中保持兼容。
2. `pi_teacher` 的 local terrain privilege 有明确 schema 和 cheating audit；`pi_loco` 无
   特权信息泄漏。
3. 66D AMP expert/runtime state 使用同一生成定义，所有纳入 motion 已通过目标 T4 资产
   playback 审核。
4. `pi_teacher` 通过基础 walk、jog ramp、强 rough、上楼、下楼和局部组合 evaluator。
5. `pi_loco` 在相同冻结 buckets 中达到最终阈值，并通过 depth-zero/shuffle/freeze/delay
   消融证明感知依赖。
6. 每次课程晋级后，所有已通过旧 buckets 仍达到冻结回归阈值。
7. command collapse、reward gaming、绕障、滑落、撞击后补偿、teacher cheating 和 hard
   violation 能被 evaluator 明确识别为失败。
8. 同一最终导出的 `pi_loco` 在 IsaacLab 与 MuJoCo 对应 buckets 均通过。
9. 每项能力声明同时具有结构化 evaluator JSON、lineage manifest 和连续行为回放证据。

## Verification Path

验证链按以下顺序执行，任一硬 Gate 失败即停止后续正式训练：

```text
静态合同测试
  -> IsaacLab T4 spawn/camera smoke
  -> 18 motion playback 审计
  -> AMP expert/runtime parity
  -> teacher privileged observation schema + cheating audit
  -> depth preprocessing/student Actor/export parity
  -> evaluator RED/negative cases
  -> T4 MuJoCo observation/depth golden parity
  -> teacher 数值健康 probe
  -> Stage E privileged teacher adaptive curriculum
  -> teacher IsaacLab fixed evaluator + 旧能力回归
  -> teacher rollout dataset 生成与 schema 验证
  -> Stage S depth student distillation / DAgger / optional fine-tune
  -> student IsaacLab fixed evaluator + depth ablation
  -> 同 checkpoint MuJoCo fixed evaluator
  -> route acceptance + 全量回归
```

所有训练、GPU probe、批量 playback、evaluator、teacher rollout 采集和其他长运行命令必须
通过 tmux 启动。

### Verification Path Status

`runnable`

代码、测试和仿真入口都可在本仓库内建设。当前 GPU/IsaacLab runtime、motion 人工视觉
审核、teacher 收敛、student 蒸馏和最终 Sim2Sim 尚未通过，但它们是计划内的显式 Gate，
不构成无法规划的外部阻塞。

## Required Capabilities

- 可运行 Isaac Sim 4.5 / IsaacLab 2.1 的 GPU 环境。
- tmux，用于所有训练和长运行任务。
- 人工查看 IsaacLab 与 MuJoCo 连续回放，判断穿地、滑步、动作不连续和提前抬脚。
- T4 local terrain privileged observation：HeightScan / elevation scan 或等价局部地形表征。
- MuJoCo 离屏 depth rendering 与 T4 MJCF。
- PyTorch/torchvision，用于轻量 depth CNN、distillation loader 和导出。
- teacher rollout dataset 存储空间，结构化 evaluator JSON 与视频/回放 artifact 存储空间。

## Fallback Evidence

无可替代最终行为 Gate 的 fallback。

- 若当前机器暂时不能运行 IsaacLab/GPU，可先完成纯 Python schema/unit tests，但不得将
  对应工作项标记完成。
- 若暂时不能人工观看 motion，可生成逐帧限位、足底高度和速度审计，但不得替代最终
  playback 通过结论。
- Teacher IsaacLab 通过不能替代最终 student depth Sim2Sim。
- MuJoCo depth runner 不可用时，不得用 IsaacLab-only student 结果宣称 Sim2Sim 通过。

## Final Integration Claim

`final_integration_claim`: 单个 T4 depth-aware `pi_loco` 在冻结的 Actor 合同下完成站立、
前后/侧向行走、转向、连续 jog、强 rough、上下楼梯和受 corridor/gates 约束的路线穿越；
该策略由 `pi_teacher` 通过 adaptive curriculum 训练得到的专家行为蒸馏而来。所有新旧
buckets 在固定 evaluator 中通过，depth 消融显示感知依赖，且同一导出策略在 IsaacLab 与
MuJoCo 中均有结构化结果和连续回放证据。

## 冻结合同

### 机器人与动作合同

- 动作维度：27。
- 关节顺序：`legged_lab/assets/t4/constants.py::T4_JOINT_NAMES`。
- policy/control 目标频率：50 Hz，physics 200 Hz，最终以实现后 manifest 为准。
- teacher 与 student 共享 action order、action scale、PD/control contract 和 command
  semantics。
- checkpoint 兼容键至少包含：policy role、joint order、action scale、policy observation
  schema、teacher privileged schema、depth preprocessing schema、depth history、proprio
  history、network class、AMP schema 和 distillation dataset schema。
- 任一兼容键变化时拒绝完整 optimizer resume；部分加载必须生成新 lineage 并报告
  loaded/skipped keys。

### Teacher / Student / Critic 合同

- `pi_teacher` actor：proprio history、local velocity command、previous action、local terrain
  privilege。
- `pi_loco` actor：短 depth history、proprio history、local velocity command、previous action。
- Critic：Actor 输入外可增加 base velocity、contact、terrain state、terrain params 等训练期
  特权信息。
- 导出文件只包含 `pi_loco` 所需模块和 normalization/preprocessing 常量。
- `pi_loco` 输入中不得出现 HeightScan、地形高度真值、接触真值、teacher latent、terrain
  id 或 route progress 真值。
- Teacher privileged observation 只能描述局部可见/可由深度近似推断的几何，不得包含全局
  路线真值、未来 gate、成功标签或完整地图。

### AMP 合同

- 单帧 AMP state：`q27 + dq27 + hands_root6 + feet_root6 = 66D`。
- discriminator 输入：相邻 transition，132D。
- expert 与 runtime 必须调用相同的 feature builder 或同一可数值对照的实现。
- AMP 不接收 velocity command，不负责任务成功。
- motion 类别权重显式配置，不能由文件数隐式决定。
- AMP expert 全部为平地动作，而楼梯/强 rough 需要偏离该分布的步态；AMP reward 系数
  必须支持按 terrain difficulty 显式调度（衰减或按地形分组配置），该旋钮在 Stage E
  开训前冻结进配置，不作为训中临时补丁。仅当调度旋钮不足以解决可复现风格压制时，
  才升级讨论 conditioning / 多 discriminator。

### Depth / Terrain 合同

- Teacher local terrain privilege 初始候选：base-frame height/elevation scan，分辨率、范围、
  height offset、invalid value 和历史顺序进入 versioned schema。
- Teacher scan 不得沿用原框架 pelvis 居中 `(1.6, 1.0)` 默认形状；必须使用前向不对称窗口
  （前向延伸约 `1.0-1.5 m`，覆盖提前落脚决策区），且覆盖范围与 depth 相机可视区域大致
  对齐，否则 Stage S 蒸馏时 student 无法从深度观测到 teacher 的决策依据。
- 相机位姿从 T4 `forward_camera` site 起步，在 IsaacLab 和 MuJoCo 中显式校准。
- 原始 D455 `480 x 270` 只作为传感器参考，不直接进入 MLP。
- clipping、invalid value、normalization、resize、history order 和 update cadence 进入 versioned
  preprocessing schema。
- 初始容量 probe 候选：`64 x 48` 或 `80 x 45`、约 15 Hz、3 帧历史；probe 后冻结正式值。
- policy 中间 step 复用最新 depth；student 蒸馏和 fine-tune 覆盖 frame hold、有限 delay、
  noise 和 dropout。

### Distillation 合同

- teacher rollout dataset 必须记录 teacher checkpoint、student input、teacher action、
  command bucket、terrain bucket、seed、schema version 和 simulator。
- student 训练默认使用 teacher action mean 作为主要监督目标；是否使用 action sample、
  value、feature matching 或短 horizon statistics 必须显式记录。
- DAgger 数据必须标记 collection policy、teacher relabel checkpoint 和 student checkpoint。
- 蒸馏 loss 仅作为训练诊断，不作为能力通过标准。

## Evaluator 设计与阈值冻结

### 两阶段冻结规则

1. 在查看正式训练结果前冻结 evaluator version、seed 集、command/terrain/task buckets、
   episode 时长、指标定义、禁止项和 strict success 逻辑。
2. 运行零策略、随机策略、首个可运行 teacher baseline 和首个 student baseline 后，冻结
   成功率、tracking、滑移、跌倒、student-teacher gap 与 latency 的数值阈值。之后若修改
   阈值，必须升级 evaluator version 并重新评估全部历史对照，不得静默迁就某个 checkpoint。

### 通用指标

- `policy_role`: zero/random/teacher/student。
- `requested_command`: evaluator 固定请求。
- `generated_command`: 实际进入 reward/policy 的最终 command。
- `achieved_motion`: base-frame 实际线速度/角速度。
- `actual_progress`: 世界坐标连续位移或 route progress。
- tracking error、standing drift、fall、feet slide、undesired contact、joint soft/hard limit、
  action saturation、nonfinite、timeout 和 strict success。
- teacher privilege presence、student privilege leakage check、student-teacher action gap。
- depth freshness、frame age、dropout fraction 和 inference latency。
- checkpoint、git commit、config、seed、simulator、camera/preprocess schema、teacher terrain
  schema、distillation dataset schema 和 evaluator version。

### 基础 command buckets

初始范围继承原框架作为候选，不代表预先声明 T4 全部可达：

```text
stand: vx=0, vy=0, wz=0
forward: vx in {0.2, 0.4, 0.6, 0.8, 1.0}, vy=0, wz=0
backward: vx in {-0.2, -0.4, -0.6}, vy=0, wz=0
lateral: vy in {-0.5, -0.3, 0.3, 0.5}, vx=0, wz=0
turn-in-place: wz in {-1.57, -1.0, 1.0, 1.57}, vx=vy=0
curved: selected nonzero vx + wz pairs
jog-ramp: selected low->high vx ramps after jog motions pass audit
```

若首个 baseline 证明边缘 bucket 不可达，应在正式训练前基于数据缩小并冻结范围；不得在
正式结果出来后删除失败 bucket。

### 每阶段固定评估协议

- 每个 bucket 至少使用固定 seed 集和足以观察连续动作的 episode；具体 episode 数在
  evaluator baseline 后冻结。
- 分别报告 bucket 结果，不以宏平均掩盖单方向失败。
- 每个正式 evaluation 同时生成 JSON 与连续 replay/video。
- evaluator 运行期间关闭训练随机 command，使用明确固定请求；保留规定的测试扰动。
- Teacher 晋级前执行当前 curriculum evaluator 和所有旧 bucket 回归。
- Student 通过前执行同一 bucket evaluator、depth ablation、privilege leakage audit 和 MuJoCo
  对应 evaluator。

## 完整训练计划

训练预算是上限与资源计划，不是成功标准。每条正式 lineage 使用固定 seed、代码 commit、
配置快照、evaluator version 和 preprocessing schema；MDP、Actor、teacher privileged schema
或 depth 合同实质变化后必须新建 lineage。

### 训练启动通用 Gate

每次正式训练前必须满足：

- 对应工作项的自动测试、smoke、`git diff --check` 全部通过。
- GPU、显存、磁盘、tmux session、绝对 checkout 和 commit 已记录。
- 1-env smoke、128-env 数值 probe 和容量 probe 均无 NaN/Inf/CUDA/OOM。
- evaluator 能消费未训练 checkpoint 并生成完整 JSON/replay。
- teacher/student observation schema 和 export manifest 能被机器检查。
- 日志至少包含 command provenance、progress、fall/slide/hard violation、AMP/task reward、
  privilege leakage audit 和 distillation 指标。

### 训练节奏

- 优化器 rollout 频率沿用原框架 50 Hz 控制与 24 steps/env 起点。
- 每个正式阶段先做 `200-500` iteration 单卡因果 probe，再决定是否进入正式预算。
- Stage E 每 `500` iterations 保存 checkpoint；每 `2,000` iterations 运行轻量固定 evaluator；
  在 `10k/20k/40k/final` 或课程相应节点运行完整 teacher evaluator。
- Stage S 每个 distillation epoch / DAgger round 记录 dataset version 和 student checkpoint；
  每个主要 round 后运行 student fixed evaluator，正式通过前运行 MuJoCo evaluator。
- 若连续两个完整 Gate checkpoint 均未改善关键行为指标，暂停 lineage 进入 `diagnose`，不以
  追加预算代替根因分析。
- 任一 nonfinite、hard-limit、command-collapse、teacher cheating、student privilege leakage
  或 evaluator schema drift 立即停止训练。

### Stage E：Privileged Teacher PPO+AMP

- 初始化：fresh teacher policy，不从旧 MjLab/HIW 或原 TienKung checkpoint 完整恢复。
- Actor 输入：proprio history + local velocity command + previous action + local terrain privilege。
- Critic 输入：Actor 输入外可加入 base velocity、contact、terrain params/state 等训练期特权。
- 地形课程：flat/轻 rough -> jog command/ramp -> strong rough/boxes/wave/slope -> up/down
  stairs -> short local composition；旧 buckets 持续采样。
- 课程两层语义：env 级 terrain level 按行进距离自动升降（IsaacLab 训练内机制，照常允许）；
  课程里程碑声明（如宣称 stairs 能力、解锁 local composition 段）必须由固定 evaluator +
  行为回放决定，不由 terrain level 或训练 reward 单独决定。
- gait 表示在正式开训前由短 probe 定案并冻结：固定 clock、command-conditioned 相位或
  楼梯段放松周期约束三者择一写入 Stage E 配置；gait reward 属于 MDP 实质部分，训中
  变更即新建 lineage。
- command：显式 stand + forward/backward/lateral/turn/curved/jog-ramp buckets。
- AMP：使用通过审核的 stand/walk/backward/lateral/turn/jog motions；`t4_run` hold out，除非
  独立审核通过。
- 任务信号：task MDP 负责速度、转向、progress、traversal、失败和 strict success；AMP 不
  替代任务成功。
- 预算上限：`80,000-120,000` iterations；首个正式 lineage 建议 1024 env 起步，容量 Gate
  允许后再扩到 2048/4096，不预设必须 4096。
- 晋级条件：teacher 基础 walk、jog ramp、strong rough、上楼、下楼和局部组合 buckets 达到
  冻结阈值；无 command collapse/hard violation；连续 replay 行为自然且无持续滑步、撞击
  后补偿或下楼滑落。
- 失败分类：不能站稳/控制合同；能站不走/reward specification；速度跟踪但滑步/gait 与
  AMP；rough 过而 stairs 不过/task MDP；teacher 用作弊字段/privilege schema。

### Stage S：Depth Student Distillation

- 起点：Stage E 通过 teacher checkpoint；student Actor/observation/action/depth schema 冻结。
- 数据：生成 teacher rollout dataset，覆盖全部 teacher 通过 buckets、边界命令、rough、up/down
  stairs、局部组合和失败近邻场景。
- Student 输入：depth history + proprio history + local velocity command + previous action。
- 蒸馏：先 behavior cloning / policy distillation，再 DAgger 或在线数据聚合；teacher 对 student
  自身状态分布重新标注动作。
- Fine-tune：只有 student evaluator 出现可定位 gap 时才做短 PPO fine-tune；Actor 不增加特权
  输入，Critic 可使用训练期特权。
- Depth：使用冻结 preprocessing schema；训练覆盖 frame hold、delay、noise、dropout 和
  invalid value；正式 evaluator 使用冻结扰动档位。
- 预算上限：BC/DAgger 以 dataset rounds 和 evaluator 为准；可选 PPO fine-tune 上限
  `10,000-30,000` iterations。
- 晋级条件：student 在基础/jog/rough/up/down stairs/局部组合 buckets 达到冻结阈值；
  正常 depth 显著优于 zero/shuffle/freeze；student export 无特权泄漏；同一导出 policy 通过
  MuJoCo 对应 buckets。

### Route Acceptance：最终组合验收

- 起点：Stage S 通过 student checkpoint。
- route manager：只输出局部 velocity command，不输出关节动作或技能 ID。
- 任务合同：corridor + ordered gates/waypoints + 越界失败 + strict exit success。
- 验收：单一已会地形段 -> 两段组合 -> 转弯加楼梯/rough -> 完整路线。
- 默认不作为第三次正式长训；若 locomotion buckets 全通过但 route 系统性失败，允许
  `<=15,000` iteration 短 fine-tune，新建 lineage 并重新跑全量回归。
- 通过条件：全部 required gates 顺序通过；越界、跳 gate、逆序和侧绕成功率为零；旧
  locomotion buckets 无退化；IsaacLab 与 MuJoCo route replay 均通过。

## 工作项

- [ ] M0：T4 资产、相机与 motion 事实闭环（当前）
  - scope: 扩展 spawn smoke，发现并冻结 joint/body/foot/hand/camera frame；逐条播放 18 个
    motion，生成机器审计与人工判定清单。
  - progress: 已新增离线机器审计脚本并生成 `artifacts/eval/t4_motion_audit.json`；静态
    MJCF facts 中 `forward_camera`、左右 foot/palm sites 均存在；18 条 motion 中机器审计
    accept 17 条，`t4_run` 因 hard joint limit violations 与 holdout 规则 reject。所有 motion
    的 `human_playback_status` 仍为 `pending`；已新增 IsaacLab headless playback 审计入口
    `legged_lab/scripts/playback_t4_motions.py`，待 nubot 目标环境运行后才能生成仿真播放证据。
  - acceptance_criteria: 27DoF 顺序精确匹配；相机 frame 在 IsaacLab 与 MJCF 语义明确；
    每条 motion 有 accept/reject 和原因；穿地、限位、root height、方向/速度均有记录。
  - verification_commands: `pytest -q tests/test_t4_asset_migration.py`; `tmux new-session -d -s t4-m0-spawn 'cd /home/nubot/phn_ws/t4_train/TienKung-Lab && python legged_lab/scripts/smoke_t4_asset.py 2>&1 | tee /tmp/t4-m0-spawn.log'`; `tmux new-session -d -s t4-m0-motion-audit 'cd /home/nubot/phn_ws/t4_train/TienKung-Lab && python legged_lab/scripts/audit_t4_motions.py --task t4_loco_teacher --output artifacts/eval/t4_motion_audit.json 2>&1 | tee /tmp/t4-m0-motion-audit.log'`; `tmux new-session -d -s t4-m0-playback 'cd /home/nubot/phn_ws/t4_train/TienKung-Lab && python legged_lab/scripts/playback_t4_motions.py --task t4_loco_teacher --output artifacts/eval/t4_motion_playback.json --sim-device cuda:0 2>&1 | tee /tmp/t4-m0-playback.log'`
  - success_definition: T4 资产与 motion 不再依赖文件名或静态 shape 推断，teacher/student
    训练输入集合有可复核的目标仿真证据。

- [ ] M1：66D AMP schema、loader 与 expert 生成
  - scope: 建立共享 feature builder，泛化 loader，使用已通过 M0 的 motion 生成 expert。
  - acceptance_criteria: expert/runtime 66D 字段、顺序、坐标系一致；132D transition 正确；
    loader 无 20DoF/52D 常量；坏维度、错 joint order、nonfinite fail-fast；motion 类别权重
    显式配置。
  - verification_commands: `pytest -q tests/test_t4_amp_contract.py tests/test_amp_loader_schema.py`; `python -m compileall -q legged_lab/envs/t4 rsl_rl/rsl_rl/utils`; `git diff --check`
  - success_definition: AMP 可以读取真实 T4 expert，并与 teacher/runtime feature 做数值对照。

- [ ] M2：Teacher privilege、Depth preprocessing、Student Actor 与导出合同
  - scope: 实现 versioned teacher local terrain schema、student depth schema、短 depth history、
    轻量 CNN student Actor、teacher/student policy role 和导出签名。
  - acceptance_criteria: teacher privilege 只含局部可迁移几何；teacher scan 为前向不对称
    窗口（前向约 `1.0-1.5 m`）且与 depth 相机可视区域大致对齐；不直接 flatten 480x270；
    student Actor 无特权输入；JIT/ONNX 只导出 student encoder+actor；preprocessing golden
    tests 通过；输入 shape/顺序 manifest 可机器检查。
  - verification_commands: `pytest -q tests/test_t4_teacher_privilege.py tests/test_t4_depth_preprocessing.py tests/test_t4_depth_actor.py tests/test_t4_policy_export.py`; `python -m compileall -q rsl_rl/rsl_rl/modules legged_lab/envs/t4`; `git diff --check`
  - success_definition: teacher 可用局部特权地形学习，student 可训练、可导出且合同冻结。

- [ ] M3：最小 teacher/student 任务、固定 evaluator、distillation dataset 与 T4 depth Sim2Sim
  - scope: 注册唯一 T4 locomotion 任务族；实现 command provenance、teacher/student evaluator、
    负向 RED cases、teacher rollout dataset writer/loader、T4 27DoF MuJoCo depth runner 和
    observation golden parity。
  - acceptance_criteria: 未训练 teacher/student 也能生成完整 evaluator JSON/replay；绕行、错
    gate、command collapse、teacher cheating、student privilege leakage fixture 被判失败；
    IsaacLab/MuJoCo depth+proprio observation golden parity 达到冻结容差；dataset schema
    可追踪 teacher checkpoint。
  - verification_commands: `pytest -q tests/test_t4_loco_registration.py tests/test_t4_evaluator_contract.py tests/test_t4_distillation_dataset.py tests/test_t4_sim2sim_parity.py`; `tmux new-session -d -s t4-m3-smoke 'cd /home/ssy/桌面/TienKung-Lab && python legged_lab/scripts/train.py --task=t4_loco_teacher --headless --num_envs=1 --max_iterations=2 2>&1 | tee /tmp/t4-m3-smoke.log'`; `git diff --check`
  - success_definition: teacher 训练、student 蒸馏、评估、导出和 MuJoCo 验证形成最小端到端闭环。

- [ ] M4：Stage E privileged teacher adaptive curriculum
  - scope: 1/128 env health gates、teacher capacity probe、Stage E 正式 lineage、基础/jog/
    rough/up-down stairs/local composition buckets；stairs 地形同时包含上行与下行两个方向。
    按 IsaacLab origin 约定，机器人生成在中心平台上，因此 `MeshInvertedPyramidStairsTerrainCfg`
    对应上行起步，`MeshPyramidStairsTerrainCfg` 对应下行起步，两类必须同时配置。
  - acceptance_criteria: finite 全路径；正式 env 数有显存证据；teacher 在全部冻结 buckets
    通过；无 command collapse、持续滑步、撞击后补偿、滑落和 hard violation；teacher
    privilege schema 无作弊字段；AMP terrain-difficulty 调度旋钮与 gait 表示已按开训前
    冻结合同配置；楼梯 traversal 的 episode 时长在 evaluator 冻结前随任务设计确定。
  - verification_commands: `tmux new-session -d -s t4-teacher-capacity 'cd /home/ssy/桌面/TienKung-Lab && bash scripts/probe_t4_teacher_capacity.sh 2>&1 | tee /tmp/t4-teacher-capacity.log'`; `tmux new-session -d -s t4-teacher-formal 'cd /home/ssy/桌面/TienKung-Lab && bash scripts/train_t4_loco_stage.sh teacher 2>&1 | tee /tmp/t4-teacher-formal.log'`; `python legged_lab/scripts/eval_t4_loco.py --role teacher --stage all --checkpoint <teacher-checkpoint> --output <json>`
  - success_definition: `pi_teacher` 已在统一 adaptive curriculum 中学出可蒸馏的 T4 locomotion
    专家行为。

- [ ] M5：Stage S depth student distillation / DAgger
  - scope: 用 M4 teacher 生成覆盖全 buckets 的 rollout dataset，训练 depth student，执行 DAgger
    或在线数据聚合，必要时短 PPO fine-tune。
  - acceptance_criteria: dataset lineage 完整；student action scale/normalizer/export 正确；
    student 在基础/jog/rough/up-down stairs/local composition buckets 达到冻结阈值；正常 depth
    显著优于 zero/shuffle/freeze；student export 无特权泄漏。
  - verification_commands: `tmux new-session -d -s t4-teacher-rollouts 'cd /home/ssy/桌面/TienKung-Lab && bash scripts/collect_t4_teacher_rollouts.sh --checkpoint <teacher-checkpoint> 2>&1 | tee /tmp/t4-teacher-rollouts.log'`; `tmux new-session -d -s t4-student-distill 'cd /home/ssy/桌面/TienKung-Lab && bash scripts/distill_t4_depth_student.sh --teacher <teacher-checkpoint> 2>&1 | tee /tmp/t4-student-distill.log'`; `python legged_lab/scripts/eval_t4_loco.py --role student --stage all --checkpoint <student-checkpoint> --depth-ablation all --output <json>`
  - success_definition: `pi_loco` 通过深度输入复现 teacher 的主要 locomotion 能力，并具备可导出
    的部署合同。

- [ ] M6：最终 route acceptance、MuJoCo Sim2Sim 与 artifact handoff
  - scope: route manager、corridor/gates、组合验收、全量回归、MuJoCo final evaluator 和
    artifact handoff。
  - acceptance_criteria: strict route success 通过；绕行/错序/越界成功率为零；M4 teacher
    证据、M5 student 证据和 final student MuJoCo 证据 lineage 完整；同一最终 `pi_loco` 在
    IsaacLab/MuJoCo 通过；JSON、replay、manifest、teacher dataset 和 export 完整。
  - verification_commands: `python legged_lab/scripts/eval_t4_loco.py --role student --stage route --checkpoint <student-checkpoint> --simulators isaaclab mujoco --output artifacts/eval/t4_route.json`; `python legged_lab/scripts/eval_t4_loco.py --role student --stage all --checkpoint <student-checkpoint> --simulators isaaclab mujoco --output artifacts/eval/t4_final.json`; `git diff --check`
  - success_definition: `final_integration_claim` 有完整、同 lineage、跨仿真的最终 student 行为证据支持。

- [x] M-1：目标与 Spec 收敛
  - acceptance_criteria: purpose、scope、teacher/student、Actor/Critic/AMP/depth/route 边界、
    成功标准与验证策略已由用户确认。
  - verification_commands: `test "$(sed -n 's/^> 状态 \/ Status: //p' docs/specs/2026-08-12--t4-unified-depth-locomotion.md)" = user-approved`
  - success_definition: 已批准 Spec 可作为本 Plan 的稳定输入。

## Commit Units

每个 commit unit 只有在对应 work item 实现完成、review 无 Critical、verify PASS 后提交。

1. `docs(t4): 修订特权专家到深度学生训练路线`
   - work items: M-1 与 planning/recovery 文档。
   - scope: Spec、Plan、PROJECT_CONTEXT 路线同步、最小 `.harness` 指针。
2. `feat(t4): 验证资产和动作合同`
   - work items: M0。
3. `feat(amp): 增加 T4 schema 驱动 expert pipeline`
   - work items: M1。
4. `feat(t4): 增加 teacher privilege 和 depth student 合同`
   - work items: M2。
5. `feat(t4): 增加 teacher/student evaluator 和 depth sim2sim`
   - work items: M3。
6. `train(t4): 建立 privileged teacher curriculum`
   - work items: M4 的配置、脚本、evaluator artifacts；大 checkpoint 是否入库按仓库 artifact
     规范决定，不默认提交。
7. `train(t4): 蒸馏 depth student locomotion policy`
   - work items: M5 的 dataset schema、distillation artifacts、student evaluator。
8. `feat(t4): 增加 route final acceptance`
   - work items: M6。

## Known Risks / Blockers

- T4 URDF 转 USD 不会自动保留 MJCF camera site；相机 frame 可能需在两个资产中分别显式
  定义并做 golden scene 校准。
- motion 文件名和原始 root velocity 方向可能不等同于 T4 command frame，必须以 playback
  和测量结果决定纳入类别。
- Teacher local terrain privilege 若过强，会产生 student 无法蒸馏的行为；必须做 cheating
  audit 和 student-teacher gap 分析。
- 单 discriminator 可能在 walk/jog/stairs 风格间冲突；只有固定 evaluator 复现后才升级。
- 固定 gait clock 可能限制连续走跑或楼梯；先建立 teacher baseline，再由 transition probe
  决定是否做连续 command-conditioned schedule。
- Depth rendering 会降低 student 并行 capacity；正式 env 数由容量 probe 决定，不以 4096
  为目标。
- MuJoCo 和 IsaacLab depth 的 range、坐标、遮挡和 invalid value 可能不同，需要合成场景
  数值 parity 而非仅 shape parity。
- Student 蒸馏可能出现 covariate shift；DAgger 与短 fine-tune 是修复路径，不得降低 Gate
  或扩大预算掩盖问题。
- 最终收敛无法由代码完成预先保证；预算用尽而 evaluator 未通过时转 `diagnose`，不得降低
  Gate 或扩大预算掩盖问题。

## Recovery Protocol

每次恢复工作先执行：

```bash
cd /home/ssy/桌面/TienKung-Lab
git status --short --branch
sed -n '1,120p' .harness/state.md
rg -n '^[-] \[[ x]\]' docs/plans/2026-08-12--t4-unified-depth-locomotion-plan.md
tmux ls
```

训练阶段还需重新核对当前 commit、绝对 checkout、GPU ownership、tmux session、log、最近
teacher checkpoint、student checkpoint、dataset version、evaluator version 和错误扫描。
不能从旧对话中的进度推断当前训练状态。

## Next Skill

`implement`

Reason: M0 active slice、文件面和验证路径已经清楚，可从资产/motion 事实闭环开始最小实现；
若首次 IsaacLab smoke 失败且根因不明，则切换 `diagnose`。

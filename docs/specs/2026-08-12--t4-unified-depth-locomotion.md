# Spec - T4 特权专家到深度学生 Locomotion

> 状态 / Status: user-approved（走跑 / 深度学生阶段 **done**；部署合同仍以本文为准）
> 货架: living 架构。当前执行队列不在本文，见 `docs/README.md`。
> Owner: user / agent
> Date: 2026-08-12
> Revision: teacher-student route
> 来源请求 / Source request: 基于 TienKung-Lab 训练 T4 高质量统一 locomotion policy，并改为“特权专家 + adaptive curriculum + 深度蒸馏”的正式路线

## 背景

目标机器人是 T4 27DoF 人形机器人。现有 T4 工作已经迁入机器人资产、原始动作数据和
motion visualization 中间数据，但尚未注册可训练任务、生成正式 AMP expert、建立深度
Actor，也没有可证明的训练行为。

TienKung-Lab 已提供可复用的 velocity tracking、PPO、AMP、训练/播放/导出和基础
MuJoCo Sim2Sim 链路，但原始实现的 `walk` 与 `run` 是两个独立任务和两个独立策略，
并未解决单策略连续走跑。现有 sensor 任务也只是将原始深度图直接拼入 MLP，不能视为
已经验证的感知 locomotion 架构；现有 Sim2Sim 仍绑定原机器人 20DoF 且没有深度输入。

本 Spec 采用更成熟的感知 locomotion 组织方式：先训练一个可使用局部特权地形观测的
统一 locomotion 专家 `pi_teacher`，用 adaptive curriculum 在一个专家 lineage 中学习
站立、走、慢跑、强 rough、上下楼梯和局部组合能力；再把该专家蒸馏为最终部署的
深度学生 `pi_loco`。最终导出策略仍然只能使用深度相机、本体历史、局部速度命令和
上一动作，不携带 HeightScan、高程图、接触真值或路线进度真值。

本文取代此前“从第一阶段开始直接训练 depth Actor 的累积式 PPO 路线”。`docs/README.md`
中关于第一阶段/第二阶段的旧表述应在执行阶段同步更新。

## 目标

- 交付一个统一的 T4 27DoF 部署 Actor `pi_loco`，而不是站立、行走、跑步、楼梯或障碍
  specialist 的集合。
- `pi_loco` 以深度相机、本体历史、局部速度命令和上一动作作为唯一部署输入，输出
  27DoF 动作。
- 第一阶段训练 `pi_teacher`：允许 teacher actor 使用局部 HeightScan / elevation scan
  等可由深度近似推断的特权地形观测，配合 PPO+AMP 和 adaptive curriculum 学出统一
  locomotion 能力。
- 第二阶段训练 `pi_loco`：通过 depth student distillation / DAgger / 必要的短 PPO
  fine-tune，把 teacher 行为迁移到 depth-only Actor。
- 保留 TienKung-Lab 已验证的 PPO+AMP 主干：task MDP 决定运动目标和任务成功，AMP
  只提供自然运动 prior。
- 为 teacher 和 student 分别建立固定 evaluator；最终能力声明只以 student 的
  IsaacLab/MuJoCo evaluator 和连续回放为准。

## 非目标（Non-goals）

- 本 Spec 不要求完成真机部署或真机安全验收。
- 不把 HeightScan、高程图、接触真值、地形参数或 route progress 真值放入最终导出
  `pi_loco` Actor。
- 不把 `pi_teacher` 当作部署策略；teacher 只是训练期专家和监督信号来源。
- 不训练多个运行时 specialist，不使用 MoE 或技能 ID 选择站立、行走、跑步或楼梯策略。
- 不让 locomotion Actor 同时承担全局路线规划。
- 不用五次正式长训分别堆出 walk、jog、rough、stairs 和 route；这些是课程/evaluator
  milestones，不是默认独立 lineage。
- 不把总奖励、episode length、训练存活、loss 下降或 checkpoint 存在当作行为成功。
- 不承诺未经仿真播放和性能 probe 验证的相机频率、图像分辨率、速度上限或楼梯参数。

## 用户 / 调用者（Users / Callers）

- 训练人员通过 TienKung-Lab 训练入口启动 IsaacLab PPO+AMP teacher 训练，以及后续
  depth student distillation / fine-tune。
- 评估人员通过固定 evaluator 和 IsaacLab playback 检查 teacher 与 student 的各
  command/terrain/task bucket。
- MuJoCo Sim2Sim runner 加载最终导出的 `pi_loco`，提供深度、本体状态和局部速度命令。
- 路线任务的 route/gate manager 仅向 `pi_loco` 提供局部速度命令。
- 后续真机部署可以复用同一 `pi_loco` 输入合同，但不属于本 Spec 验收范围。

## 行为规格（Behavior Spec）

### 策略拓扑

训练和部署使用两个明确区分的策略角色：

```text
pi_teacher:
    privileged local terrain + proprio history + local velocity command + previous action
        -> privileged locomotion actor
        -> 27DoF action

pi_loco:
    depth history + proprio history + local velocity command + previous action
        -> depth encoder + locomotion actor
        -> 27DoF action
```

`pi_teacher` 只存在于训练和评估中；`pi_loco` 是唯一允许导出的部署 Actor。二者共享
27DoF 动作顺序、action scale、control frequency、command 语义和基础 safety limits。

### Actor / Critic / Privilege 边界

`pi_loco` 的固定部署输入为：

```text
短深度历史 + proprio history + local velocity command + previous action
```

Actor 输出固定 27DoF 动作，关节顺序以
`legged_lab/assets/t4/constants.py::T4_JOINT_NAMES` 为唯一真值。

`pi_teacher` actor 可以使用局部特权地形观测，例如以机器人为中心的 HeightScan /
elevation scan、台阶局部几何和地形类别 curriculum id。此类输入必须满足两个限制：

- 只描述局部可见/可由深度近似推断的几何，不包含未来 gate 真值、完整路线进度、全局地图、
  成功标签或人为答案。
- 字段、坐标系、范围、分辨率和历史顺序进入 versioned teacher observation schema。
- scan 窗口必须前向不对称（前向延伸约 `1.0-1.5 m`，覆盖提前落脚决策区），并与 depth
  相机可视区域大致对齐；不得沿用原框架 pelvis 居中 `(1.6, 1.0)` 默认形状，否则 teacher
  的决策依据超出 student 深度可见范围，蒸馏不可达。

训练期 Critic 可以额外使用真实 base velocity、接触、terrain state、terrain params 等
特权信息。Critic 特权不得进入 `pi_loco` Actor、导出签名或 MuJoCo Actor 输入。

### AMP 职责

AMP state 第一版采用共享函数生成：

```text
q(27) + dq(27)
+ left/right hand position relative to root(6)
+ left/right foot position relative to root(6)
= 66 dimensions
```

expert transition 和 policy transition 均由同一字段定义和坐标系生成，discriminator
transition 输入为 132 维。AMP 只判断运动是否属于自然 T4 运动分布，不读取 velocity
command，也不负责决定去向、速度、楼梯成功或路线成功。

第一阶段使用经 IsaacLab T4 播放核验通过的站立、行走、后退、侧移、转向和慢跑 motion。
`t4_run` 在关节限位和动作质量审核前保持 hold out。动作文件数量不得隐式决定行为类别
权重；AMP 数据采样需要按实际纳入的运动类别显式记录。

AMP expert 全部来自平地动作，而楼梯与强 rough 要求策略偏离该分布（高抬腿、躯干前倾、
非常规步幅）。为避免 discriminator 在课程后段持续惩罚正确的地形步态，AMP reward 系数
必须支持按 terrain difficulty 显式调度（随难度衰减或按地形分组配置）；该调度旋钮属于
Stage E 冻结合同，在正式开训前定案，不作为训中临时补丁。

### Stage E：特权专家训练

`pi_teacher` 通过一个正式 PPO+AMP lineage 训练。能力顺序用 adaptive curriculum 和
evaluator buckets 表达，不拆成五次长训：

1. 基础 locomotion：零速站立、前后行走、左右侧移、原地转向、移动转弯。
2. jog / 走跑过渡：加入经核验的慢跑 motion、高速 forward 和 command ramp。
3. 强 rough：提高不平整度、摩擦变化、box、wave、斜坡和扰动。
4. 上下楼梯：同一 traversal task 覆盖上楼和下楼，要求提前调整落脚，而不是碰撞后补偿、
   滑落或坠落。
5. 局部组合：转弯接 rough / stairs、短 corridor 和 ordered local gates，用于验证
   locomotion 能力组合，不让 actor 承担全局规划。

curriculum 分两层语义：env 级 terrain level 按行进距离自动升降，是 IsaacLab 训练内
机制，照常允许，不需要 evaluator 介入；课程里程碑声明（宣称某能力通过、解锁下一
课程段）必须由固定 evaluator 和行为回放决定，不由训练 reward 或地形 level 单独决定。
旧 buckets 需持续采样并回归；旧能力发生不可接受退化时不得晋级。

第一阶段 command 范围以原框架范围作为初始候选：

```text
vx: [-0.6, 1.0] m/s
vy: [-0.5, 0.5] m/s
yaw rate: [-1.57, 1.57] rad/s
```

这些范围不是预先宣称的 T4 可达能力。首个可运行 baseline 必须按固定 buckets 测量真实
可达范围并冻结后续数值阈值。零速站立必须作为显式 bucket 采样，不能依赖连续分布偶然
采到。

### Stage S：深度学生蒸馏

`pi_loco` 从 teacher rollout 中学习。训练数据至少包含：

- student 输入：depth history、proprio history、local velocity command、previous action；
- teacher 目标：teacher action mean / action sample、必要时包含 teacher value 或短 horizon
  行为统计；
- 对齐元数据：teacher checkpoint、terrain bucket、command bucket、depth preprocessing
  schema、seed 和 simulator。

默认训练顺序：

1. teacher rollout 数据集生成，覆盖全部已通过 teacher buckets 和边界场景。
2. supervised behavior cloning / policy distillation，让 `pi_loco` 复现 teacher action。
3. DAgger 或在线数据聚合：student 进入自己状态分布，teacher 对同状态给动作监督。
4. 必要时进行短 PPO fine-tune；Actor 仍只看 student 输入，Critic 可继续使用训练期特权。

蒸馏成功不是 loss 下降，而是 student 在冻结 evaluator 中达到 teacher 行为阈值，并在
depth 消融中表现出合理的感知依赖。

### Depth 时间合同

- control/policy 频率沿用原框架 50 Hz 作为目标合同。
- 深度允许以较低固定频率更新，policy 中间步骤复用最新帧。
- Actor 使用短深度历史以抵抗遮挡、dropout 和单帧歧义。
- 约 15 Hz、最近 3 帧仅作为性能 probe 起点，不是冻结参数。
- 训练和蒸馏必须覆盖帧保持、有限随机延迟、噪声和 dropout；具体范围在相机与并行性能
  baseline 后冻结。
- IsaacLab、MuJoCo 和最终导出运行时必须共享相机内参、安装位姿、裁剪范围、预处理、
  下采样、帧顺序和无效值编码。

### Route traversal 边界

路线层与 locomotion 层边界为：

```text
route/gate manager
    -> local velocity command
pi_loco
    -> 27DoF action
```

route manager 不输出关节动作，不切换 locomotion specialist。路线型任务必须同时使用
几何 corridor、ordered gates/waypoints、越界失败和 strict success。成功要求按顺序通过
全部 required gates、保持在 corridor 内、无跌倒或禁止接触，并通过出口；从侧面绕开
障碍不得计为成功。

route 默认是最终验收层，不默认作为第三次正式长训。只有当 `pi_loco` 已通过 locomotion
buckets、但局部组合 route 系统性失败时，才允许短 fine-tune，并必须创建新 lineage。

### 边界情况（Edge Cases）

- 零速命令下冻结或正确处理步态相位，不能强迫机器人原地倒脚，也不能靠长时间存活刷
  主要奖励。
- velocity command 变化时不得出现由离散 walk/run 模式切换造成的明显动作跳变。
- reward 或 tracking 上升但 requested/generated/achieved velocity、实际位移或任务成功下降
  时，必须判定为 command collapse 或 reward gaming，而非训练进步。
- `pi_teacher` 不能依赖 student 无法从深度推断的作弊字段；否则必须修改 teacher
  observation 或降低对 student 的可迁移声明。
- 深度全零、打乱、冻结、延迟或大面积 dropout 时，student evaluator 必须能观察并记录
  能力变化；`pi_loco` 不得依赖未声明的特权输入蒙混通过感知任务。
- 观测、动作、深度预处理、teacher privileged schema、网络结构或 AMP state 合同改变后，
  不得无条件续训旧 checkpoint。
- 若仅部分加载旧权重，必须记录 loaded/skipped 范围并创建真实的新 lineage；不得宣称
  完整无损 resume。
- 楼梯到达终点后应立即结束 episode 或进入下一有效段，不能停在终点持续积累主要奖励。

### 接口 / 状态（Interfaces / State）

- T4 资产合同：`legged_lab/assets/t4/`。
- T4 motion 源与生成数据：`legged_lab/envs/t4/datasets/`。
- 训练任务应提供唯一 T4 locomotion 任务族；teacher/student 是训练角色，不是多个运行时
  specialist。
- teacher checkpoint、student checkpoint、teacher rollout dataset 和 final export manifest
  必须记录 lineage 关系。
- 导出 policy 只包含 depth encoder、student locomotion Actor 和 normalization/preprocessing
  常量，并带有可机器检查的输入输出 shape/顺序说明。
- evaluator 输出结构化结果，至少包含 lineage、seed、checkpoint、policy_role、command
  bucket、terrain bucket、requested/generated/achieved motion、实际 progress、tracking
  error、跌倒、滑移、禁止接触、关节限位和 strict success。
- MuJoCo runner 必须使用 T4 MJCF、27DoF 映射和深度 renderer，不得复用现有 20DoF
  `sim2sim.py` 合同冒充 T4 验证。

## 约束（Constraints）

- 训练和长时间运行必须在 tmux 中执行。
- 第一版优先复用现有 PPO+AMP runner、reward/terrain 基础设施、导出链路和成熟依赖，只做
  T4、teacher privileged observation、depth student 和 evaluator 所必需的改动。
- 不保留旧机器人 20DoF/52D 的兼容层；AMP loader 应直接泛化为显式 schema。
- 不为未来可能的传感器或多策略需求预建插件、MoE、第二 discriminator 或复杂配置层。
- AMP 与地形步态的冲突优先用开训前冻结的 terrain-difficulty 系数调度解决；只有调度
  旋钮不足以消除可复现风格压制时，才升级讨论 command/style conditioning 或多个
  discriminator。
- 训练启动前必须验证 T4 spawn、27DoF/body 顺序、质量/碰撞/站姿、动作播放和 AMP
  expert/runtime 一致性。
- T4 MJCF 的 `forward_camera` site 可作为相机位姿起点，但 IsaacLab URDF 转换不会自动
  保留该 site；最终相机 frame 必须在两个仿真器中显式核对。
- observation/action/AMP/teacher privileged/depth preprocessing 合同变化时禁止延续不兼容
  optimizer state。
- 正式训练只能在短 probe 通过数值健康、资源容量和关键行为信号后启动。

## 选定方案（Chosen Approach）

采用“TienKung-Lab PPO+AMP 主干 + T4 原生合同 + 特权专家 + 深度学生蒸馏”：

- 先建立 T4 资产、动作、AMP、teacher privileged observation、depth preprocessing、
  evaluator 和 MuJoCo depth parity 的合同闭环。
- Stage E 训练 `pi_teacher`：使用局部 HeightScan/elevation scan 和 adaptive curriculum，
  在一个专家 lineage 中学出 walk、jog、strong rough、up/down stairs 和局部组合能力。
- Stage S 训练 `pi_loco`：用 teacher rollout 做 depth student distillation / DAgger，必要时
  进行短 PPO fine-tune。
- 最终部署只导出 `pi_loco`，不导出 teacher，不导出 HeightScan 或其他特权输入。
- route/gate manager 保持在 Actor 外，以局部 velocity command 连接唯一 locomotion policy。
- 同步建设 T4 depth MuJoCo runner，使最终 student 的 Sim2Sim 成为硬证据，而不是最终补做项。

此方案把“先学会运动”和“再用深度复现运动”分开，降低直接 depth PPO 中感知噪声、地形
课程和动力学 reward 同时耦合造成的失败归因难度。

## 拒绝方案（Rejected Options）

- **原样保留 walk/run 两个 policy**：与最终统一 Actor 目标冲突，运行时需要隐藏的策略
  切换，也无法自然扩展为统一楼梯与 route policy。
- **五次正式长训 walk -> jog -> rough -> stairs -> route**：把 evaluator milestones 当成
  lineage 边界，流程成本高，且与成熟 teacher-student 感知 locomotion 路线不匹配。
- **直接从第一阶段训练 depth Actor 作为默认主线**：可行但失败归因更难，depth 噪声、
  遮挡、延迟、地形课程和 locomotion reward 会相互掩盖；当前将其降为 fallback。
- **原始深度图直接 flatten 进入 MLP**：参数量和并行渲染成本过高，现有实现也没有提供
  已验证的感知 locomotion 证据。
- **Teacher 使用不可迁移的作弊特权**：例如全局路线真值、未来 gate、成功标签或完整地形
  地图；这会让 student 蒸馏不可达。
- **Actor 同时负责全局路线规划**：扩大问题边界，混淆规划、感知和关节控制失败原因。
- **多个 specialist 最后蒸馏**：增加 lineage、训练与验证复杂度，当前没有证据表明统一
  teacher 不可行。

## 验证策略（Verification Strategy）

### 基线证据（Baseline Evidence）

- 记录当前 git HEAD、dirty paths、T4 资产路径和动作数据清单。
- 运行现有 T4 静态资产测试，确认当前仅有迁移/格式证据，没有训练任务或行为能力。
- 在实际 IsaacLab 环境执行 T4 spawn smoke，记录 joint/body 顺序、质量、碰撞、初始站姿
  和相机 frame。
- 逐条播放候选 motion，记录脚底穿透、滑移、关节错序、root 高度、不连续和限位风险。
- 对现有原框架 walk/run 记录配置差异，作为继承边界，而不是 T4 行为 baseline。

### 自动检查（Automated Checks）

- T4 27DoF 名称、顺序、动作 shape、关节限制和 IsaacLab/MuJoCo 映射测试。
- AMP schema 测试：66D shape、字段顺序、坐标系、expert/runtime 共用生成函数、transition
  shape 132D。
- motion loader 测试：多文件/类别采样、维度不匹配 fail-fast、无旧 20DoF 常量依赖。
- teacher privileged observation 测试：local terrain scan shape、坐标、范围、无 route/global
  cheating 字段。
- depth 预处理测试：裁剪、无效值、归一化、下采样、历史顺序、IsaacLab/MuJoCo 数值一致。
- Actor/Critic 边界测试：student Actor 不包含 privileged state；导出 signature 与训练
  signature 一致。
- distillation dataset 测试：teacher/student obs-action 对齐、schema 版本、seed、bucket 和
  checkpoint lineage 可追踪。
- checkpoint lineage 测试：合同不兼容时拒绝完整 resume，部分加载显式报告。
- evaluator 合同测试：固定 buckets、strict success、ordered gates、绕行失败和结构化输出。
- targeted lint/format、相关 pytest 和 `git diff --check`。

### Smoke / E2E 检查

- 1-env T4 spawn 和 motion playback smoke。
- 最小 AMP expert 生成与 runtime observation 对照 smoke。
- teacher privileged observation smoke：height/elevation scan 随地形变化，且坐标与机器人
  base frame 一致。
- 小环境数、少 iteration 的 teacher PPO 数值健康 probe：finite observation/action/loss/
  gradient/parameter，无 CUDA/OOM/NaN，reward 分量和 command 指标可读。
- depth pipeline 容量 probe，用于冻结图像尺寸、历史长度、更新频率和最大并行环境数。
- teacher rollout dataset 生成 smoke，确认可被 student loader 消费。
- student BC / DAgger 小批量 smoke，确认 loss、action scale、normalizer 和导出路径可用。
- 固定 command bucket 的 IsaacLab playback：站立、前进、后退、侧移、转向、jog ramp。
- teacher IsaacLab rough/stairs playback。
- 同一最终 student export 的 T4 MuJoCo depth Sim2Sim bucket playback。

### 负向 / 边界检查（Negative / Boundary Checks）

- 零策略、随机策略、早期 teacher checkpoint 和早期 student checkpoint 作为 evaluator 负基线。
- teacher cheating audit：移除或打乱不可迁移特权字段时，确认这些字段没有出现在 student
  export 或 student observation schema 中。
- depth-zero、depth-shuffle、depth-freeze、延迟和 dropout 消融。
- requested command 固定但 generated/achieved velocity 下降的 command-collapse 场景。
- 高 tracking reward 但低实际 progress、站立不动或滑步的 reward-gaming 场景。
- 楼梯撞击后抬脚、下楼滑落/坠落、禁止接触、joint hard-limit 场景。
- route 跳 gate、逆序 gate、越界和侧绕障碍场景，全部必须失败。
- 新课程阶段对所有已通过旧 buckets 的回归检查。

### 文档 / 状态检查（Documentation / State Checks）

- Spec 批准后同步 `docs/README.md`，删除旧的 direct-depth-first 训练路线表述。
- `README.md` 只在真实 T4 训练/评估入口可运行后增加命令，不提前宣称能力。
- `AGENTS.md` 保持为稳定规则与入口，不写训练进度和临时阈值。
- git status 中仅处理本任务显式路径，不覆盖用户现有 dirty 文件。

### 完成前所需 fresh evidence

- 重新运行相关自动测试、lint 和 `git diff --check`。
- 重新核对 teacher checkpoint、student checkpoint、distillation dataset、配置、导出 policy、
  evaluator 结果和 MuJoCo run 的 lineage。
- 每一项能力声明必须产生最新固定 evaluator 结构化结果和连续行为回放；训练日志和
  checkpoint 只能作为辅助证据。
- 声明阶段通过前，必须重新扫描 NaN/Inf、CUDA/OOM、hard-limit、禁止接触、teacher
  cheating audit 和 evaluator 失败项。

## 能力缺口（Capability Gaps）

- 当前机器的 IsaacLab/GPU 可用性、相机并行性能和最大安全环境数尚未在本轮实测；执行时
  需在 tmux 中运行 spawn、渲染和容量 probe。
- （已关闭，2026-08-12）18 条 T4 motion 的逐条视觉核验已完成：nubot IsaacLab headless
  playback 0 reject，人工复核基于 `artifacts/motion_review/` 的三视角回放通过，accept 17 条，
  `t4_run` 继续 held out。已接受的缺陷是 clip 不对齐地面、接触时序不可用；66D AMP feature
  不含绝对 root 高度，故 expert 不受影响。
- 现有 RSL-RL policy 没有 teacher/student 双角色模块、depth CNN Actor 或 distillation
  dataset/loader，需要新增最小模块并验证导出能力。
- 现有 terrain observation 需要明确 teacher privileged schema，避免把不可迁移 route/global
  信息喂给 teacher actor。
- 现有 MuJoCo runner 只有原机器人 20DoF 本体观测，需要 T4 27DoF、相机 renderer、
  预处理与 observation parity 实现。
- 当前没有 T4 行为 baseline，因此绝对成功率、tracking error、滑移和延迟阈值尚不能可靠
  冻结。
- 真机相机内参、安装标定、延迟和噪声不在本 Spec 验收中；未来进入真机阶段时必须重新
  建立硬件证据，不能把仿真 D455 参数当作硬件事实。

## 成功标准（Success Criteria）

### 合同与训练启动 Gate

- T4 在 IsaacLab 正确 spawn，27DoF/body/质量/碰撞/初始站姿和相机 frame 已核对。
- 纳入 AMP 的所有 motion 已通过目标 T4 回放审核；不合格 motion 被排除并记录原因。
- expert/runtime AMP observation 在 shape、字段顺序、坐标系和数值定义上完全一致。
- teacher privileged schema、student depth schema、Actor/Critic 边界、导出 policy 和 MuJoCo
  输入合同一致，student Actor 无特权泄漏。
- 短 probe 数值健康且容量可承受后，才允许启动正式 teacher 训练或 student 蒸馏。

### Teacher 行为 Gate

- `pi_teacher` 在固定 evaluator 中通过基础 walk、jog ramp、强 rough、上楼、下楼和局部
  组合 buckets。
- evaluator、command/terrain buckets、禁止项和输出 schema 在查看正式训练结果前冻结。
- 每个 bucket 独立报告，不能用全局均值掩盖某个方向或地形失败。
- requested、generated、achieved motion 和实际 progress 一致可追踪；不存在 command
  collapse。
- nonfinite action、joint hard-limit violation 和禁止接触为零。
- 上楼和下楼分别统计并通过；提前调整落脚，不以撞击后补偿、滑落或坠落计为成功。

### Student / 感知 / Sim2Sim Gate

- `pi_loco` 在相同冻结 buckets 中达到 teacher-derived 阈值；允许指标略低于 teacher，但
  必须超过冻结成功阈值。
- depth 消融造成与地形相关且可解释的能力下降，证明策略在感知任务中实际使用深度；
  正常 depth 输入下能力达到冻结阈值。
- student export 不包含 teacher privileged modules、HeightScan 输入或 Critic。
- 同一最终导出 `pi_loco` 在 IsaacLab 与 MuJoCo 对应 buckets 均达到冻结阈值。
- route 必须按顺序通过全部 required gates；越界、跳 gate、逆序或侧绕的成功率为零。
- 每一项能力声明都有结构化 evaluator 结果和连续行为回放，不以训练曲线代替。

## 残余风险（Residual Risks）

- T4 motion 的速度方向或坐标定义可能与 command frame 不一致；先播放和测量，再冻结
  command bucket，避免仅凭文件名配置训练。
- Teacher privileged observation 若过强，会产生 student 无法蒸馏的行为；必须持续做
  cheating audit 和 teacher/student gap 分析。
- 一个未做 command conditioning 的 discriminator 可能在 walk/jog 或楼梯风格之间发生
  冲突。默认先用单 discriminator；只有固定 evaluator 复现风格压制后才升级设计。
- 固定 gait clock 可能限制连续走跑或楼梯适应。gait 表示（固定 clock、command-conditioned
  相位或楼梯段放松周期约束）必须在 Stage E 正式开训前由短 probe 定案并冻结进配置；
  gait reward 属于 MDP 实质部分，训中变更即新建 lineage。
- depth 渲染可能显著降低并行环境数。通过下采样、较低传感器更新频率和容量 probe 控制，
  不以牺牲输入合同为代价盲目维持 4096 env。
- IsaacLab 与 MuJoCo renderer 存在深度定义和图像坐标差异；必须通过合成场景 golden
  parity 检查，而不是只比较 shape。
- Student 蒸馏可能出现 covariate shift；DAgger 和短 fine-tune 是默认修复路径，但不能
  通过降低 evaluator Gate 掩盖问题。

## Plan 交接（Plan Handoff）

- 当前切片 / Active slice: 建立训练前 T4 合同闭环：spawn/body discovery、共享 66D AMP
  observation、schema 化 loader、motion playback/expert 生成，以及 teacher privileged
  observation 与 student depth 输入合同；在这些 Gate 通过前不启动长训练。
- 建议下一 skill / Suggested next skill: implement
- 计划提示 / Planning notes: 先验证资产与数据，再实现共享 AMP schema；随后完成
  teacher privileged observation、depth preprocessing、teacher/student policy contracts、
  最小 `t4_loco` 任务、固定 evaluator、T4 depth MuJoCo parity 和 distillation dataset
  loader；所有训练和长运行进入 tmux。
- 建议里程碑 / Suggested milestones: M0 资产与 motion 事实闭环；M1 AMP schema/expert；
  M2 teacher/student 观测与导出合同；M3 evaluator + T4 depth Sim2Sim；M4 privileged
  teacher adaptive curriculum；M5 depth student distillation；M6 final student route acceptance。
- 里程碑验收提示 / Per-milestone acceptance hints: M0/M1 以合同和 playback 为准；M2/M3
  以数值、shape、parity、cheating audit 和负向测试为准；M4 以 teacher evaluator 为准；
  M5/M6 以最终 student 的固定 buckets、depth 消融、IsaacLab 与 MuJoCo 同 lineage evaluator
  及连续行为证据为准。

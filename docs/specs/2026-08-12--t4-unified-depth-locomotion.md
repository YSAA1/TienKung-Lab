# Spec - T4 统一深度感知 Locomotion

> 状态 / Status: user-approved
> Owner: user / agent
> Date: 2026-08-12
> 来源请求 / Source request: 基于 TienKung-Lab 训练 T4 高质量统一 locomotion policy 的讨论

## 背景

目标机器人是 T4 27DoF 人形机器人。现有 T4 工作已经迁入机器人资产、原始动作数据和
motion visualization 中间数据，但尚未注册可训练任务、生成正式 AMP expert、建立深度
Actor，也没有可证明的训练行为。

TienKung-Lab 已提供可复用的 velocity tracking、PPO、AMP、训练/播放/导出和基础
MuJoCo Sim2Sim 链路，但原始实现的 `walk` 与 `run` 是两个独立任务和两个独立策略，
并未解决单策略连续走跑。现有 sensor 任务也只是将原始深度图直接拼入 MLP，不能视为
已经验证的感知 locomotion 架构；现有 Sim2Sim 仍绑定原机器人 20DoF 且没有深度输入。

本 Spec 定义从 TienKung-Lab 主干出发构建一个 T4 原生、深度感知、可累积扩展的统一
locomotion policy，以及证明该策略有效所需的行为 Gate。本文在 T4 感知与训练路线方面
取代 `PROJECT_CONTEXT.md` 中“HeightScan 或深度感知”“第一阶段 Actor 仅本体观测”等旧
表述；该背景文档应在 Spec 批准后的执行阶段同步更新。

## 目标

- 交付一个统一的 T4 27DoF locomotion Actor，而不是站立、行走、跑步、楼梯或障碍
  specialist 的集合。
- Actor 以深度相机、本体历史、局部速度命令和上一动作作为唯一部署输入，输出 27DoF
  动作。
- 先复现并验证高质量 T4 基础行走，再在同一 Actor 合同下累积扩展慢跑、强 rough、
  上下楼梯和路线型穿越能力。
- 保留 TienKung-Lab 已验证的 PPO+AMP 主干：task MDP 决定运动目标和任务成功，AMP
  只提供自然运动 prior。
- 为每一阶段建立固定、可复现、能识别 command collapse、滑步和绕障等伪成功的
  evaluator。
- 同一导出策略必须在 IsaacLab 和 MuJoCo 中通过相应行为 Gate。

## 非目标（Non-goals）

- 本 Spec 不要求完成真机部署或真机安全验收。
- 不使用 HeightScan、高程图、接触真值或地形真值作为部署 Actor 输入。
- 不训练多个运行时 specialist，不使用 MoE 或技能 ID 选择站立、行走、跑步或楼梯策略。
- 不让 locomotion Actor 同时承担全局路线规划。
- 不从第一轮训练开始同时加入全速跑、完整楼梯状态机和路线型障碍。
- 不把总奖励、episode length、训练存活、loss 下降或 checkpoint 存在当作行为成功。
- 不承诺未经仿真播放和性能 probe 验证的相机频率、图像分辨率或速度上限。

## 用户 / 调用者（Users / Callers）

- 训练人员通过 TienKung-Lab 训练入口启动 IsaacLab PPO+AMP 训练。
- 评估人员通过固定 evaluator 和 IsaacLab playback 检查各 command/terrain bucket。
- MuJoCo Sim2Sim runner 加载同一导出 policy，提供深度、本体状态和局部速度命令。
- 路线任务的 route/gate manager 仅向 locomotion Actor 提供局部速度命令。
- 后续真机部署可以复用同一 Actor 输入合同，但不属于本 Spec 验收范围。

## 行为规格（Behavior Spec）

### Actor 与 Critic 边界

部署 Actor 的固定输入为：

```text
短深度历史 + proprio history + local velocity command + previous action
```

Actor 输出固定 27DoF 动作，关节顺序以
`legged_lab/assets/t4/constants.py::T4_JOINT_NAMES` 为唯一真值。

深度图必须经过确定性的裁剪、无效值处理、归一化和下采样，再由轻量 CNN 编码。CNN
与 locomotion policy 端到端训练并包含在导出 policy 中。不得把原始 `480 x 270` 深度图
直接 flatten 后拼入普通 MLP。

训练期 Critic 可以额外使用真实 base velocity、接触、地形参数或地形状态。此类特权
信息不得进入 Actor、导出签名或 MuJoCo Actor 输入。

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

第一阶段使用经 IsaacLab T4 播放核验通过的站立、行走、后退、侧移和转向 motion。
`t4_run` 在关节限位和动作质量审核前保持 hold out。动作文件数量不得隐式决定行为类别
权重；AMP 数据采样需要按实际纳入的运动类别显式记录。

### 累积式训练课程

训练始终维护同一个 Actor 合同，按以下能力顺序累积扩展：

1. 基础 depth-aware walk：零速站立、前后行走、左右侧移、原地转向、移动转弯，地形为
   flat + 轻 rough。
2. jog / 走跑过渡：在基础 checkpoint 上加入经核验的慢跑 motion 和连续速度能力。
3. 强 rough：提高不平整度、摩擦变化和扰动，要求策略使用深度保持稳定与低滑移。
4. 上下楼梯：同一 traversal task 同时覆盖上楼和下楼，要求提前调整落脚，而不是碰撞后
   补偿、滑落或坠落。
5. route composition：在 corridor 和 ordered gates 中组合既有 locomotion 能力。

进入新阶段时仍需采样已通过的旧 command/terrain buckets，并运行固定回归 evaluator。
旧能力发生不可接受退化时不得晋级。

第一阶段 command 范围以原框架范围作为初始候选：

```text
vx: [-0.6, 1.0] m/s
vy: [-0.5, 0.5] m/s
yaw rate: [-1.57, 1.57] rad/s
```

这些范围不是预先宣称的 T4 可达能力。首个可运行 baseline 必须按固定 buckets 测量真实
可达范围并冻结后续数值阈值。零速站立必须作为显式 bucket 采样，不能依赖连续分布偶然
采到。

### Depth 时间合同

- control/policy 频率沿用原框架 50 Hz 作为目标合同。
- 深度允许以较低固定频率更新，policy 中间步骤复用最新帧。
- Actor 使用短深度历史以抵抗遮挡、dropout 和单帧歧义。
- 约 15 Hz、最近 3 帧仅作为性能 probe 起点，不是冻结参数。
- 训练必须覆盖帧保持、有限随机延迟、噪声和 dropout；具体范围在相机与并行性能
  baseline 后冻结。
- IsaacLab、MuJoCo 和最终导出运行时必须共享相机内参、安装位姿、裁剪范围、预处理、
  下采样、帧顺序和无效值编码。

### Route traversal 边界

路线层与 locomotion 层边界为：

```text
route/gate manager
    -> local velocity command
depth-aware T4 locomotion Actor
    -> 27DoF action
```

route manager 不输出关节动作，不切换 locomotion specialist。路线型任务必须同时使用
几何 corridor、ordered gates/waypoints、越界失败和 strict success。成功要求按顺序通过
全部 required gates、保持在 corridor 内、无跌倒或禁止接触，并通过出口；从侧面绕开
障碍不得计为成功。

### 边界情况（Edge Cases）

- 零速命令下冻结或正确处理步态相位，不能强迫机器人原地倒脚，也不能靠长时间存活刷
  主要奖励。
- velocity command 变化时不得出现由离散 walk/run 模式切换造成的明显动作跳变。
- reward 或 tracking 上升但 requested/generated/achieved velocity、实际位移或任务成功下降
  时，必须判定为 command collapse 或 reward gaming，而非训练进步。
- 深度全零、打乱、冻结、延迟或大面积 dropout 时，evaluator 必须能观察并记录能力变化；
  策略不得依赖未声明的特权输入蒙混通过感知任务。
- 观测、动作、深度预处理、网络结构或 AMP state 合同改变后，不得无条件续训旧
  checkpoint。
- 若仅部分加载旧权重，必须记录 loaded/skipped 范围并创建真实的新 lineage；不得宣称
  完整无损 resume。
- 楼梯到达终点后应立即结束 episode 或进入下一有效段，不能停在终点持续积累主要奖励。

### 接口 / 状态（Interfaces / State）

- T4 资产合同：`legged_lab/assets/t4/`。
- T4 motion 源与生成数据：`legged_lab/envs/t4/datasets/`。
- 训练任务应提供唯一的 T4 locomotion 注册入口；课程阶段是该任务的训练状态，不是多个
  部署 task/policy。
- 导出 policy 必须包含 depth encoder 和 locomotion Actor，并带有可机器检查的输入输出
  shape/顺序说明。
- evaluator 输出结构化结果，至少包含 lineage、seed、checkpoint、command bucket、terrain
  bucket、requested/generated/achieved motion、实际 progress、tracking error、跌倒、滑移、
  禁止接触、关节限位和 strict success。
- MuJoCo runner 必须使用 T4 MJCF、27DoF 映射和深度 renderer，不得复用现有 20DoF
  `sim2sim.py` 合同冒充 T4 验证。

## 约束（Constraints）

- 训练和长时间运行必须在 tmux 中执行。
- 第一版优先复用现有 PPO+AMP runner、reward/terrain 基础设施、导出链路和成熟依赖，只做
  T4 与统一深度 Actor 所必需的改动。
- 不保留旧机器人 20DoF/52D 的兼容层；AMP loader 应直接泛化为显式 schema。
- 不为未来可能的传感器或多策略需求预建插件、MoE、第二 discriminator 或复杂配置层。
- 只有出现可复现的 AMP 风格冲突时，才讨论 command/style conditioning 或多个
  discriminator。
- 训练启动前必须验证 T4 spawn、27DoF/body 顺序、质量/碰撞/站姿、动作播放和 AMP
  expert/runtime 一致性。
- T4 MJCF 的 `forward_camera` site 可作为相机位姿起点，但 IsaacLab URDF 转换不会自动
  保留该 site；最终相机 frame 必须在两个仿真器中显式核对。
- observation/action/AMP 合同变化时禁止延续不兼容 optimizer state。
- 正式训练只能在短 probe 通过数值健康、资源容量和关键行为信号后启动。

## 选定方案（Chosen Approach）

采用“原框架主干 + T4 原生合同 + 累积式单策略课程”：

- 复用 TienKung-Lab 的 velocity tracking、PPO+AMP、训练循环和导出结构。
- 先忠实建立 T4 walk 闭环，而不是把原项目独立 walk/run 两套配置直接拼接。
- 使用一个 depth CNN Actor 和 asymmetric Critic，从第一阶段起冻结最终 Actor 输入/输出
  接口。
- 通过累积课程扩展能力，每次保留旧 buckets 和回归 Gate。
- 路线规划保持在 Actor 外，以局部 velocity command 连接唯一 locomotion policy。
- 同步建设 T4 depth MuJoCo runner，使 Sim2Sim 成为每阶段的硬证据，而不是最终补做项。

此方案最大化复用原框架已验证部分，同时明确隔离原框架尚未解决的统一走跑、深度编码和
T4 Sim2Sim 问题。

## 拒绝方案（Rejected Options）

- **原样保留 walk/run 两个 policy**：与最终统一 Actor 目标冲突，运行时需要隐藏的策略
  切换，也无法自然扩展为统一楼梯与 route policy。
- **一次性联合训练 walk、run、rough、stairs 和 route**：失败归因困难，AMP、动作合同、
  感知和任务 MDP 会相互掩盖，无法快速获得可信行为 baseline。
- **原始深度图直接 flatten 进入 MLP**：参数量和并行渲染成本过高，现有实现也没有提供
  已验证的感知 locomotion 证据。
- **HeightScan Actor 或 HeightScan teacher-to-student 作为默认路线**：违反用户明确选择的
  depth-only 感知边界，并增加当前目标不需要的训练系统。
- **Actor 同时负责全局路线规划**：扩大问题边界，混淆规划、感知和关节控制失败原因。
- **多个 specialist 最后蒸馏**：增加 lineage、训练与验证复杂度，当前没有证据表明直接
  累积单策略路线不可行。

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
- depth 预处理测试：裁剪、无效值、归一化、下采样、历史顺序、IsaacLab/MuJoCo 数值一致。
- Actor/Critic 边界测试：Actor 不包含 privileged state；导出 signature 与训练 signature
  一致。
- checkpoint lineage 测试：合同不兼容时拒绝完整 resume，部分加载显式报告。
- evaluator 合同测试：固定 buckets、strict success、ordered gates、绕行失败和结构化输出。
- targeted lint/format、相关 pytest 和 `git diff --check`。

### Smoke / E2E 检查

- 1-env T4 spawn 和 motion playback smoke。
- 最小 AMP expert 生成与 runtime observation 对照 smoke。
- 小环境数、少 iteration 的数值健康 probe：finite observation/action/loss/gradient/parameter，
  无 CUDA/OOM/NaN，reward 分量和 command 指标可读。
- depth pipeline 容量 probe，用于冻结图像尺寸、历史长度、更新频率和最大并行环境数。
- 固定 command bucket 的 IsaacLab playback：站立、前进、后退、侧移、转向。
- 同一导出 policy 的 T4 MuJoCo depth Sim2Sim bucket playback。
- 后续阶段分别执行 rough、上楼、下楼和 route 连续回放。

### 负向 / 边界检查（Negative / Boundary Checks）

- 零策略、随机策略和训练早期 checkpoint 作为 evaluator 负基线。
- depth-zero、depth-shuffle、depth-freeze、延迟和 dropout 消融。
- requested command 固定但 generated/achieved velocity 下降的 command-collapse 场景。
- 高 tracking reward 但低实际 progress、站立不动或滑步的 reward-gaming 场景。
- 楼梯撞击后抬脚、下楼滑落/坠落、禁止接触、joint hard-limit 场景。
- route 跳 gate、逆序 gate、越界和侧绕障碍场景，全部必须失败。
- 新课程阶段对所有已通过旧 buckets 的回归检查。

### 文档 / 状态检查（Documentation / State Checks）

- Spec 批准后同步 `PROJECT_CONTEXT.md`，删除 HeightScan/第一阶段 proprio-only 等冲突表述。
- `README.md` 只在真实 T4 训练/评估入口可运行后增加命令，不提前宣称能力。
- `AGENTS.md` 保持为稳定规则与入口，不写训练进度和临时阈值。
- git status 中仅处理本任务显式路径，不覆盖用户现有 `an.txt` 或其他 dirty 文件。

### 完成前所需 fresh evidence

- 重新运行相关自动测试、lint 和 `git diff --check`。
- 重新核对 checkpoint、配置、导出 policy、evaluator 结果和 MuJoCo run 使用同一 lineage。
- 每一阶段必须产生最新固定 evaluator 结构化结果和连续行为回放；训练日志和 checkpoint
  只能作为辅助证据。
- 声明阶段通过前，必须重新扫描 NaN/Inf、CUDA/OOM、hard-limit、禁止接触和 evaluator
  失败项。

## 能力缺口（Capability Gaps）

- 当前机器的 IsaacLab/GPU 可用性、相机并行性能和最大安全环境数尚未在本轮实测；执行时
  需在 tmux 中运行 spawn、渲染和容量 probe。
- 18 条 T4 motion 尚未在目标 IsaacLab 资产上逐条视觉核验；该步骤需要人工查看回放，
  自动 shape 检查不能替代。
- 现有 RSL-RL policy 没有 depth CNN Actor，需要新增最小模块并验证导出能力。
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
- depth Actor、asymmetric Critic、导出 policy 和 MuJoCo 输入合同一致，Actor 无特权泄漏。
- 短 probe 数值健康且容量可承受后，才允许启动正式训练。

### 每阶段行为 Gate

- evaluator、command/terrain buckets、禁止项和输出 schema 在查看正式训练结果前冻结。
- 首个可运行 baseline 后冻结数值阈值；阈值修改必须创建新的 evaluator 版本并说明原因，
  不得静默迁就结果。
- 每个 bucket 独立报告，不能用全局均值掩盖某个方向或地形失败。
- requested、generated、achieved motion 和实际 progress 一致可追踪；不存在 command
  collapse。
- nonfinite action、joint hard-limit violation 和禁止接触为零。
- 跌倒、脚滑、tracking error 和 strict success 相对零策略/早期 baseline 达到冻结阈值。
- 新阶段通过时，所有已通过旧 buckets 仍满足其冻结回归阈值。
- IsaacLab 通过后，同一导出 policy 在 MuJoCo 对应 buckets 也达到冻结阈值。

### 感知、楼梯与路线 Gate

- depth 消融造成与地形相关且可解释的能力下降，证明策略在感知任务中实际使用深度；
  正常 depth 输入下能力达到冻结阈值。
- 上楼和下楼分别统计并通过；提前调整落脚，不以撞击后补偿、滑落或坠落计为成功。
- route 必须按顺序通过全部 required gates；越界、跳 gate、逆序或侧绕的成功率为零。
- 每一项能力声明都有结构化 evaluator 结果和连续行为回放，不以训练曲线代替。

## 残余风险（Residual Risks）

- T4 motion 的速度方向或坐标定义可能与 command frame 不一致；先播放和测量，再冻结
  command bucket，避免仅凭文件名配置训练。
- 一个未做 command conditioning 的 discriminator 可能在 walk/jog 或楼梯风格之间发生
  冲突。默认先用单 discriminator；只有固定 evaluator 复现风格压制后才升级设计。
- 固定 gait clock 可能限制连续走跑或楼梯适应。第一阶段先对齐 walk baseline；后续是否
  使用 command-conditioned gait schedule 必须由走跑过渡 probe 决定。
- depth 渲染可能显著降低并行环境数。通过下采样、较低传感器更新频率和容量 probe 控制，
  不以牺牲输入合同为代价盲目维持 4096 env。
- IsaacLab 与 MuJoCo renderer 存在深度定义和图像坐标差异；必须通过合成场景 golden
  parity 检查，而不是只比较 shape。
- 累积课程仍可能发生灾难性遗忘；旧 bucket 回归 Gate 是晋级硬条件，但不能保证一次训练
  即收敛。

## Plan 交接（Plan Handoff）

- 当前切片 / Active slice: 建立训练前 T4 合同闭环：spawn/body discovery、共享 66D AMP
  observation、schema 化 loader、motion playback/expert 生成，以及最小 depth Actor 输入
  合同；在这些 Gate 通过前不启动长训练。
- 建议下一 skill / Suggested next skill: plan
- 计划提示 / Planning notes: 先验证资产与数据，再实现共享 AMP schema；随后完成 depth
  预处理/Actor 和最小 `t4_loco` 注册；最后补 evaluator 与 T4 MuJoCo parity，所有训练和
  长运行进入 tmux。
- 建议里程碑 / Suggested milestones: M0 资产与 motion 事实闭环；M1 AMP schema/expert；
  M2 depth Actor 与最小训练任务；M3 固定 evaluator + T4 depth Sim2Sim；M4 基础 walk；
  M5 jog 过渡；M6 强 rough/stairs；M7 route composition。
- 里程碑验收提示 / Per-milestone acceptance hints: M0/M1 以合同和 playback 为准；M2/M3
  以数值、shape、parity 和负向测试为准；M4-M7 均需固定 buckets、旧能力回归、IsaacLab
  与 MuJoCo 同 lineage evaluator 及连续行为证据。

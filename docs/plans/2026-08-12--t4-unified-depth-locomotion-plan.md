# Executable Plan - T4 统一深度感知 Locomotion

> Status: active
> Date: 2026-08-12
> Spec: `docs/specs/2026-08-12--t4-unified-depth-locomotion.md`
> Branch: `t4-train`
> Planning surface: docs plan

## Objective

基于 TienKung-Lab 的 velocity tracking + PPO + AMP 主干，交付一个统一的 T4 27DoF
深度感知 locomotion Actor。该 Actor 从基础行走逐步累积扩展到 jog、强 rough、上下楼梯
和 route traversal，并由固定 evaluator、连续行为回放及同一导出策略的 IsaacLab/MuJoCo
结果证明，而不是由训练 reward 或 checkpoint 数量证明。

最终部署 Actor 的冻结接口为：

```text
depth history + proprio history + local velocity command + previous action
    -> depth encoder + locomotion actor
    -> 27DoF action in T4_JOINT_NAMES order
```

Critic 可在训练期使用特权状态；HeightScan、高程图、接触真值和地形真值不得进入 Actor
或导出接口。

## Active slice

当前只执行训练前合同闭环 M0：在真实 IsaacLab 环境验证 T4 spawn、body/joint/camera
frame 和全部 motion playback，生成结构化审计结果。M0 未通过前不实现正式 AMP expert，
也不启动任何长训练。

## Non-goals

- 不在本计划内完成真机部署或真机安全验收。
- 不保留 walk/run 多策略，不使用 MoE、技能 ID 或运行时 checkpoint 切换。
- 不使用 HeightScan Actor，也不预建 HeightScan teacher-to-depth student 路线。
- 不让 locomotion Actor 承担全局路线规划。
- 不一次性联合训练 walk、run、stairs 和 route。
- 不用兼容层保留旧 20DoF/52D AMP 或现有 20DoF MuJoCo runner 合同。
- 不把训练预算跑完、reward 上升、episode 变长或 loss 下降定义为阶段成功。

## Success criteria

最终完成必须同时满足：

1. 只有一个导出的 T4 Actor，输入输出合同在全部训练阶段保持兼容。
2. Actor 无特权信息泄漏，depth 预处理和 observation contract 在 IsaacLab、导出模型与
   MuJoCo 中一致。
3. 66D AMP expert/runtime state 使用同一生成定义，所有纳入 motion 已通过目标 T4 资产
   playback 审核。
4. 基础 walk、jog 过渡、强 rough、上楼、下楼和 route 均通过各自冻结 evaluator。
5. 每次课程晋级后，所有已通过旧 buckets 仍达到冻结回归阈值。
6. command collapse、reward gaming、绕障、滑落、撞击后补偿和 hard violation 能被
   evaluator 明确识别为失败。
7. 同一 checkpoint 导出的同一 policy 在 IsaacLab 与 MuJoCo 对应 buckets 均通过。
8. 每项能力声明同时具有结构化 evaluator JSON 和连续行为回放证据。

## Verification path

验证链按以下顺序执行，任一硬 Gate 失败即停止后续正式训练：

```text
静态合同测试
  -> IsaacLab T4 spawn/camera smoke
  -> 18 motion playback 审计
  -> AMP expert/runtime parity
  -> depth preprocessing/Actor/export parity
  -> evaluator RED/negative cases
  -> T4 MuJoCo observation/depth golden parity
  -> 数值健康和容量 probe
  -> 分阶段正式训练
  -> 每阶段 IsaacLab fixed evaluator
  -> 同 checkpoint MuJoCo fixed evaluator
  -> 旧能力回归
```

所有训练、GPU probe、批量 playback、evaluator 和其他长运行命令必须通过 tmux 启动。

### Verification path status

`runnable`

代码、测试和仿真入口都可在本仓库内建设。当前 GPU/IsaacLab runtime、motion 人工视觉
审核和最终收敛尚未通过，但它们是计划内的显式 Gate，不构成无法规划的外部阻塞。

## Required capabilities

- 可运行 Isaac Sim 4.5 / IsaacLab 2.1 的 GPU 环境。
- tmux，用于所有训练和长运行任务。
- 人工查看 IsaacLab 与 MuJoCo 连续回放，判断穿地、滑步、动作不连续和提前抬脚。
- MuJoCo 离屏 depth rendering 与 T4 MJCF。
- PyTorch/torchvision，用于轻量 depth CNN 和导出。
- 结构化 evaluator JSON 与视频/回放 artifact 存储空间。

## Fallback evidence

无可替代最终行为 Gate 的 fallback。

- 若当前机器暂时不能运行 IsaacLab/GPU，可先完成纯 Python schema/unit tests，但不得将
  对应工作项标记完成。
- 若暂时不能人工观看 motion，可生成逐帧限位、足底高度和速度审计，但不得替代最终
  playback 通过结论。
- MuJoCo depth runner 不可用时，不得用 IsaacLab-only 结果宣称 Sim2Sim 通过。

## Final integration claim

`final_integration_claim`: 单个 T4 depth-aware policy 在冻结的 Actor 合同下完成站立、
前后/侧向行走、转向、连续 jog、强 rough、上下楼梯和受 corridor/gates 约束的路线穿越；
所有新旧 buckets 在固定 evaluator 中通过，且同一导出策略在 IsaacLab 与 MuJoCo 中均有
结构化结果和连续回放证据。

## 冻结合同

### 机器人与动作合同

- 动作维度：27。
- 关节顺序：`legged_lab/assets/t4/constants.py::T4_JOINT_NAMES`。
- policy/control 目标频率：50 Hz，physics 200 Hz，最终以实现后 manifest 为准。
- checkpoint 兼容键至少包含：joint order、action scale、policy observation schema、depth
  preprocessing schema、depth history、proprio history、network class、AMP schema。
- 任一兼容键变化时拒绝完整 optimizer resume；部分加载必须生成新 lineage 并报告
  loaded/skipped keys。

### Actor/Critic 合同

- Actor：短 depth history、proprio history、local velocity command、previous action。
- Critic：Actor 输入外可增加 base velocity、contact、terrain state 等训练期特权信息。
- 导出文件只包含 Actor 所需模块和 normalization/preprocessing 常量。
- Actor 输入中不得出现 HeightScan、地形高度真值、接触真值或 route progress 真值。

### AMP 合同

- 单帧 AMP state：`q27 + dq27 + hands_root6 + feet_root6 = 66D`。
- discriminator 输入：相邻 transition，132D。
- expert 与 runtime 必须调用相同的 feature builder 或同一可数值对照的实现。
- AMP 不接收 velocity command，不负责任务成功。
- motion 类别权重显式配置，不能由文件数隐式决定。

### Depth 合同

- 相机位姿从 T4 `forward_camera` site 起步，在 IsaacLab 和 MuJoCo 中显式校准。
- 原始 D455 `480 x 270` 只作为传感器参考，不直接进入 MLP。
- clipping、invalid value、normalization、resize、history order 和 update cadence 进入 versioned
  preprocessing schema。
- 初始容量 probe 候选：`64 x 48` 或 `80 x 45`、约 15 Hz、3 帧历史；probe 后冻结正式值。
- policy 中间 step 复用最新 depth；训练覆盖 frame hold、有限 delay、noise 和 dropout。

## Evaluator 设计与阈值冻结

### 两阶段冻结规则

1. 在查看正式训练结果前冻结 evaluator version、seed 集、command/terrain buckets、episode
   时长、指标定义、禁止项和 strict success 逻辑。
2. 运行零策略、随机策略和首个可运行 baseline 后，冻结成功率、tracking、滑移、跌倒与
   latency 的数值阈值。之后若修改阈值，必须升级 evaluator version 并重新评估全部历史
   对照，不得静默迁就某个 checkpoint。

### 通用指标

- `requested_command`: evaluator 固定请求。
- `generated_command`: 实际进入 reward/policy 的最终 command。
- `achieved_motion`: base-frame 实际线速度/角速度。
- `actual_progress`: 世界坐标连续位移或 route progress。
- tracking error、standing drift、fall、feet slide、undesired contact、joint soft/hard limit、
  action saturation、nonfinite、timeout 和 strict success。
- depth freshness、frame age、dropout fraction 和 inference latency。
- checkpoint、git commit、config、seed、simulator、camera/preprocess schema 和 evaluator version。

### 基础 command buckets

初始范围继承原框架作为候选，不代表预先声明 T4 全部可达：

```text
stand: vx=0, vy=0, wz=0
forward: vx in {0.2, 0.4, 0.6, 0.8, 1.0}, vy=0, wz=0
backward: vx in {-0.2, -0.4, -0.6}, vy=0, wz=0
lateral: vy in {-0.5, -0.3, 0.3, 0.5}, vx=0, wz=0
turn-in-place: wz in {-1.57, -1.0, 1.0, 1.57}, vx=vy=0
curved: selected nonzero vx + wz pairs
```

若首个 baseline 证明边缘 bucket 不可达，应在正式训练前基于数据缩小并冻结范围；不得在
正式结果出来后删除失败 bucket。

### 每阶段固定评估协议

- 每个 bucket 至少使用固定 seed 集和足以观察连续动作的 episode；具体 episode 数在
  evaluator baseline 后冻结。
- 分别报告 bucket 结果，不以宏平均掩盖单方向失败。
- 每个正式 evaluation 同时生成 JSON 与连续 replay/video。
- evaluator 运行期间关闭训练随机 command，使用明确固定请求；保留规定的测试扰动。
- 晋级前执行当前阶段 evaluator、所有旧阶段回归 evaluator 和 MuJoCo 对应 evaluator。

## 完整训练计划

训练预算是上限与资源计划，不是成功标准。每条正式 lineage 使用固定 seed、代码 commit、
配置快照、evaluator version 和 preprocessing schema；MDP 或 Actor 合同实质变化后必须新建
lineage。

### 训练启动通用 Gate

每次正式训练前必须满足：

- 对应工作项的自动测试、smoke、`git diff --check` 全部通过。
- GPU、显存、磁盘、tmux session、绝对 checkout 和 commit 已记录。
- 1-env smoke、128-env 数值 probe 和容量 probe 均无 NaN/Inf/CUDA/OOM。
- evaluator 能消费未训练 checkpoint 并生成完整 JSON/replay。
- 日志至少包含 command provenance、progress、fall/slide/hard violation 和 AMP/task reward
  分量。

### 训练节奏

- 优化器 rollout 频率沿用原框架 50 Hz 控制与 24 steps/env 起点。
- 每个阶段先做 `200-500` iteration 单卡因果 probe，再决定是否进入正式预算。
- 正式训练每 `500` iterations 保存 checkpoint；每 `2,000` iterations 运行轻量固定 evaluator；
  在 `10k/20k/final` 或阶段相应节点运行完整 IsaacLab + MuJoCo evaluator。
- 若连续两个完整 Gate checkpoint 均未改善关键行为指标，暂停 lineage 进入 `diagnose`，不以
  追加预算代替根因分析。
- 任一 nonfinite、hard-limit、command-collapse 或 evaluator schema drift 立即停止训练。

### Stage T1：基础 depth-aware walk

- 初始化：fresh policy，不从旧 MjLab/HIW 或原 TienKung checkpoint 恢复。
- 地形：flat 为主，混入低幅 rough、摩擦/质量随机化和有限 push。
- command：显式 stand + forward/backward/lateral/turn/curved buckets。
- AMP：仅使用通过审核的 stand/walk/backward/lateral/turn motions；`t4_run` hold out。
- depth：最终冻结 preprocessing schema；flat/轻 rough 均提供真实 depth，不使用零占位。
- 预算上限：`50,000` iterations；首个正式 lineage 建议 1024 env 起步，容量 Gate 允许后再
  扩到 2048/4096，不预设必须 4096。
- 晋级条件：所有基础 buckets 达到冻结阈值、无 command collapse/hard violation，同一导出
  policy 通过 MuJoCo 基础 buckets，连续 replay 行为自然且无持续滑步。
- 失败分类：不能站稳/控制合同、能站不走/reward specification、速度跟踪但滑步/gait 与
  AMP、IsaacLab 好而 MuJoCo 坏/dynamics 或 observation parity。

### Stage T2：jog 与连续走跑过渡

- 起点：T1 通过 checkpoint；Actor/observation/action/depth schema 不变。
- 数据：加入审核通过的 jog motions，`t4_run` 仍需独立审核后才可加入。
- 课程：保留全部 T1 buckets，逐步增加高速 forward 和 transition episodes。
- gait：先以 T1 walk clock 为 baseline；只有 transition evaluator 复现固定 clock 限制后，
  才实现 command-conditioned 连续 gait schedule，不使用离散 walk/run 技能开关。
- 预算上限：在 T1 checkpoint 上追加 `30,000` iterations；任何 gait/AMP schema 实质变化新建
  lineage，并明确是否只迁移 Actor 权重。
- 晋级条件：低速 walk 不退化，走到 jog 的 command ramp 无明显动作跳变，高速 buckets 和
  transition buckets 通过 IsaacLab/MuJoCo evaluator。

### Stage T3：强 rough

- 起点：T2 通过 checkpoint。
- 地形：逐步增加不平整度、摩擦变化、斜坡和扰动；继续保留 flat/轻 rough。
- 感知：加入 depth delay/dropout/noise curriculum，但正式 evaluator 使用冻结扰动档位。
- 预算上限：追加 `40,000` iterations。
- 晋级条件：各 rough bucket 的 success/slide/fall 达到阈值；depth-zero/shuffle 消融产生
  与地形相关的显著能力下降；正常 depth 下旧 flat/jog buckets 不退化。
- 禁止解释：仅 episode length 上升、速度命令变小或站立时间变长不算 rough 改善。

### Stage T4：上下楼梯 traversal

- 起点：T3 通过 checkpoint。
- 任务：episode 同时覆盖上楼与下楼，成功后结束或进入下一有效段，不在终点刷奖励。
- 课程：低台阶单向 -> 多阶上楼 -> 多阶下楼 -> 同 episode 上下楼；保留 flat/rough/jog
  回归环境。
- 奖励/终止：task MDP 负责连续 progress、有效 traversal、失败和 strict success；rough
  reward 或 AMP 不替代楼梯信号。
- 预算上限：追加 `50,000` iterations。
- 晋级条件：上楼/下楼分别通过；提前抬脚而不是撞击后补偿；下楼无滑落/坠落；depth
  消融验证感知依赖；全部旧 buckets 回归通过；同 policy MuJoCo stairs 通过。

### Stage T5：route composition

- 起点：T4 通过 checkpoint。
- route manager：只输出局部 velocity command，不输出关节动作或技能 ID。
- 任务合同：corridor + ordered gates/waypoints + 越界失败 + strict exit success。
- 课程：单一已会地形段 -> 两段组合 -> 转弯加楼梯/rough -> 完整路线。
- 预算上限：追加 `30,000` iterations。
- 晋级条件：全部 required gates 顺序通过；越界、跳 gate、逆序和侧绕成功率为零；旧
  locomotion buckets 无退化；IsaacLab 与 MuJoCo route replay 均通过。

## 工作项

- [ ] M0：T4 资产、相机与 motion 事实闭环（当前）
  - scope: 扩展 spawn smoke，发现并冻结 joint/body/foot/hand/camera frame；逐条播放 18 个
    motion，生成机器审计与人工判定清单。
  - acceptance_criteria: 27DoF 顺序精确匹配；相机 frame 在 IsaacLab 与 MJCF 语义明确；
    每条 motion 有 accept/reject 和原因；穿地、限位、root height、方向/速度均有记录。
  - verification_commands: `pytest -q tests/test_t4_asset_migration.py`; `tmux new-session -d -s t4-m0-spawn 'cd /home/ssy/桌面/TienKung-Lab && python legged_lab/scripts/smoke_t4_asset.py 2>&1 | tee /tmp/t4-m0-spawn.log'`; `tmux new-session -d -s t4-m0-motion-audit 'cd /home/ssy/桌面/TienKung-Lab && python legged_lab/scripts/audit_t4_motions.py --task t4_loco --output artifacts/eval/t4_motion_audit.json 2>&1 | tee /tmp/t4-m0-motion-audit.log'`
  - success_definition: T4 资产与 motion 不再依赖文件名或静态 shape 推断，训练输入集合有
    可复核的目标仿真证据。

- [ ] M1：66D AMP schema、loader 与 expert 生成
  - scope: 建立共享 feature builder，泛化 loader，使用已通过 M0 的 motion 生成 expert。
  - acceptance_criteria: expert/runtime 66D 字段、顺序、坐标系一致；132D transition 正确；
    loader 无 20DoF/52D 常量；坏维度、错 joint order、nonfinite fail-fast。
  - verification_commands: `pytest -q tests/test_t4_amp_contract.py tests/test_amp_loader_schema.py`; `python -m compileall -q legged_lab/envs/t4 rsl_rl/rsl_rl/utils`; `git diff --check`
  - success_definition: AMP 可以读取真实 T4 expert，并与 policy runtime feature 做数值对照。

- [ ] M2：Depth preprocessing、CNN Actor 与 asymmetric Critic
  - scope: 实现 versioned depth schema、短历史、轻量 CNN Actor、特权 Critic 和导出签名。
  - acceptance_criteria: 不直接 flatten 480x270；Actor 无特权输入；JIT/ONNX 包含 encoder；
    preprocessing golden tests 通过；输入 shape/顺序 manifest 可机器检查。
  - verification_commands: `pytest -q tests/test_t4_depth_preprocessing.py tests/test_t4_depth_actor.py tests/test_t4_policy_export.py`; `python -m compileall -q rsl_rl/rsl_rl/modules legged_lab/envs/t4`; `git diff --check`
  - success_definition: 一个可训练、可导出且合同冻结的 depth-aware T4 Actor 可被 runner 使用。

- [ ] M3：最小 `t4_loco` 任务、固定 evaluator 与 T4 depth Sim2Sim
  - scope: 注册唯一 T4 task；实现 command provenance、结构化 evaluator、负向 RED cases、
    T4 27DoF MuJoCo depth runner 和 observation golden parity。
  - acceptance_criteria: 未训练 policy 也能生成完整 evaluator JSON/replay；绕行、错 gate、
    command collapse fixture 被判失败；IsaacLab/MuJoCo depth+proprio observation golden parity
    达到冻结容差。
  - verification_commands: `pytest -q tests/test_t4_loco_registration.py tests/test_t4_evaluator_contract.py tests/test_t4_sim2sim_parity.py`; `tmux new-session -d -s t4-m3-smoke 'cd /home/ssy/桌面/TienKung-Lab && python legged_lab/scripts/train.py --task=t4_loco --headless --num_envs=1 --max_iterations=2 2>&1 | tee /tmp/t4-m3-smoke.log'`; `git diff --check`
  - success_definition: 训练、评估、导出和 MuJoCo 验证形成最小端到端闭环。

- [ ] M4：数值健康、容量 probe 与基础 walk 正式训练
  - scope: 1/128 env health gates、depth 容量测试、T1 正式 lineage 和全部基础 buckets。
  - acceptance_criteria: finite 全路径；正式 env 数有显存证据；T1 IsaacLab/MuJoCo fixed
    evaluator 通过；无 command collapse、持续滑步和 hard violation。
  - verification_commands: `tmux new-session -d -s t4-m4-capacity 'cd /home/ssy/桌面/TienKung-Lab && bash scripts/probe_t4_depth_capacity.sh 2>&1 | tee /tmp/t4-m4-capacity.log'`; `tmux new-session -d -s t4-walk-formal 'cd /home/ssy/桌面/TienKung-Lab && bash scripts/train_t4_loco_stage.sh walk 2>&1 | tee /tmp/t4-walk-formal.log'`; `python legged_lab/scripts/eval_t4_loco.py --stage walk --checkpoint <checkpoint> --output <json>`
  - success_definition: T4 基础 depth-aware walk 有跨仿真行为证据，可作为后续唯一合法起点。

- [ ] M5：jog 与连续走跑过渡
  - scope: 纳入审核后的 jog motion，训练高速和 command ramp，同时保留 M4 buckets。
  - acceptance_criteria: 低速 walk 不退化；走跑过渡连续；高速/transition buckets 在
    IsaacLab 与 MuJoCo 均通过；无离散技能开关。
  - verification_commands: `tmux new-session -d -s t4-jog-formal 'cd /home/ssy/桌面/TienKung-Lab && bash scripts/train_t4_loco_stage.sh jog --resume <m4-checkpoint> 2>&1 | tee /tmp/t4-jog-formal.log'`; `python legged_lab/scripts/eval_t4_loco.py --stage jog --checkpoint <checkpoint> --output <json>`
  - success_definition: 同一 Actor 从 walk 连续扩展到 jog，旧能力保持。

- [ ] M6：强 rough 与 depth 依赖验证
  - scope: 累积强 rough、扰动和 depth degradation curriculum，执行 depth 消融。
  - acceptance_criteria: rough buckets 通过；正常 depth 显著优于 zero/shuffle/freeze；M4/M5
    全部旧 buckets 回归通过；MuJoCo rough 通过。
  - verification_commands: `tmux new-session -d -s t4-rough-formal 'cd /home/ssy/桌面/TienKung-Lab && bash scripts/train_t4_loco_stage.sh rough --resume <m5-checkpoint> 2>&1 | tee /tmp/t4-rough-formal.log'`; `python legged_lab/scripts/eval_t4_loco.py --stage rough --checkpoint <checkpoint> --depth-ablation all --output <json>`
  - success_definition: 策略在强 rough 上真实使用 depth，且不牺牲基础 locomotion。

- [ ] M7：上下楼梯 traversal
  - scope: 建立上/下楼课程、提前落脚指标和统一 traversal task。
  - acceptance_criteria: 上楼和下楼分别通过；撞击后补偿、滑落、坠落均失败；depth 消融
    与 MuJoCo stairs 通过；全部旧 buckets 回归通过。
  - verification_commands: `tmux new-session -d -s t4-stairs-formal 'cd /home/ssy/桌面/TienKung-Lab && bash scripts/train_t4_loco_stage.sh stairs --resume <m6-checkpoint> 2>&1 | tee /tmp/t4-stairs-formal.log'`; `python legged_lab/scripts/eval_t4_loco.py --stage stairs --checkpoint <checkpoint> --output <json>`
  - success_definition: 同一 Actor 稳定完成可感知的上楼与下楼，而非碰撞或滑落过关。

- [ ] M8：Route composition 与最终集成验收
  - scope: route manager、corridor/gates、组合课程、全量回归和最终 artifact handoff。
  - acceptance_criteria: strict route success 通过；绕行/错序/越界成功率为零；M4-M7 全回归；
    同一最终 policy 在 IsaacLab/MuJoCo 通过；JSON、replay、manifest 和 lineage 完整。
  - verification_commands: `tmux new-session -d -s t4-route-formal 'cd /home/ssy/桌面/TienKung-Lab && bash scripts/train_t4_loco_stage.sh route --resume <m7-checkpoint> 2>&1 | tee /tmp/t4-route-formal.log'`; `python legged_lab/scripts/eval_t4_loco.py --stage all --checkpoint <checkpoint> --simulators isaaclab mujoco --output artifacts/eval/t4_final.json`; `git diff --check`
  - success_definition: `final_integration_claim` 有完整、同 lineage、跨仿真的行为证据支持。

- [x] M-1：目标与 Spec 收敛
  - acceptance_criteria: purpose、scope、Actor/Critic/AMP/depth/route 边界、成功标准与验证策略
    已由用户确认。
  - verification_commands: `test "$(sed -n 's/^> 状态 \/ Status: //p' docs/specs/2026-08-12--t4-unified-depth-locomotion.md)" = user-approved`
  - success_definition: 已批准 Spec 可作为本 Plan 的稳定输入。

## Commit units

每个 commit unit 只有在对应 work item 实现完成、review 无 Critical、verify PASS 后提交。

1. `docs(t4): approve depth locomotion spec and add executable plan`
   - work items: M-1 与 planning/recovery 文档。
   - scope: Spec、Plan、PROJECT_CONTEXT 路线同步、最小 `.harness` 指针。
2. `feat(t4): validate asset and motion contracts`
   - work items: M0。
3. `feat(amp): add schema-driven T4 expert pipeline`
   - work items: M1。
4. `feat(t4): add depth actor and export contract`
   - work items: M2。
5. `feat(t4): add loco evaluator and depth sim2sim`
   - work items: M3。
6. `train(t4): establish unified walk baseline`
   - work items: M4 的配置、脚本、evaluator artifacts；大 checkpoint 是否入库按仓库 artifact
     规范决定，不默认提交。
7. `train(t4): extend locomotion curriculum`
   - work items: M5-M7，按阶段独立 commit，不将多个未验证阶段合并。
8. `feat(t4): add route composition and final evaluator`
   - work items: M8。

## Known risks / blockers

- T4 URDF 转 USD 不会自动保留 MJCF camera site；相机 frame 可能需在两个资产中分别显式
  定义并做 golden scene 校准。
- motion 文件名和原始 root velocity 方向可能不等同于 T4 command frame，必须以 playback
  和测量结果决定纳入类别。
- 单 discriminator 可能在 walk/jog/stairs 风格间冲突；只有固定 evaluator 复现后才升级。
- 固定 gait clock 可能限制连续走跑或楼梯；先建立 walk baseline，再由 transition probe
  决定是否做连续 command-conditioned schedule。
- depth rendering 会降低 env capacity；正式 env 数由容量 probe 决定，不以 4096 为目标。
- MuJoCo 和 IsaacLab depth 的 range、坐标、遮挡和 invalid value 可能不同，需要合成场景
  数值 parity 而非仅 shape parity。
- 最终收敛无法由代码完成预先保证；预算用尽而 evaluator 未通过时转 `diagnose`，不得降低
  Gate 或扩大预算掩盖问题。

## Recovery protocol

每次恢复工作先执行：

```bash
cd /home/ssy/桌面/TienKung-Lab
git status --short --branch
sed -n '1,120p' .harness/state.md
rg -n '^[-] \[[ x]\]' docs/plans/2026-08-12--t4-unified-depth-locomotion-plan.md
tmux ls
```

训练阶段还需重新核对当前 commit、绝对 checkout、GPU ownership、tmux session、log、最近
checkpoint、evaluator version 和错误扫描。不能从旧对话中的进度推断当前训练状态。

## Next skill

`implement`

Reason: M0 active slice、文件面和验证路径已经清楚，可从资产/motion 事实闭环开始最小实现；
若首次 IsaacLab smoke 失败且根因不明，则切换 `diagnose`。

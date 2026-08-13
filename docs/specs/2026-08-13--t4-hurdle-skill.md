# Spec - T4 连续跨栏 skill（100m 障碍赛障碍 #2，reward-only heightscan 技能）

> 状态 / Status: user-approved (2026-08-13)
> Owner: user
> Date: 2026-08-13
> 来源请求 / Source request: 用 RL 训练 100m 障碍赛的连续跨栏并集成进现有 plans
> （brainstorm 会话 2026-08-13：两轮 frontier + assumption batch 全部按推荐确认，
> 用户指示直接落 Spec 与 Plan）。

## 背景

规则契约 `docs/research/100m-obstacle-rule-contract.md` 障碍 #2「连续跨栏」：多道软质
泡沫栏，间距 1.1 m；杆长 2.4 m、宽 0.05 m、高 0.3 m；跨越或跳跃通过；明文失败 =
单脚到达障碍终点前摔倒，或碰倒栏杆。规则歧义 #7：栏杆数量未写明、「碰倒」未定义
（触碰未倒是否失败不明）——工程上取保守解释并保持可配置。

任务定性：这是**纯脚上任务**。0.3 m 对 ~1.4 m 站高的 T4 约 0.21H 摆动腿 clearance，
属「高抬腿跨步」量级，无需手接触。`docs/research/2026-08-13--t4-box-climbing.md` 中
「纯 reward 学不出」的结论只针对手上箱子；脚上障碍（Parkour in the Wild、Humanoid
Parkour Learning 0.42 m 跳台）reward-only + heightscan teacher 是文献已证路线。因此
跨栏不走 vault 的 mimic 参考动作流水线。

仓库现状与工程事实：

- Stage E teacher `stage_e_prov5` 在 nubot 4 卡训练中，本 checkout 不改训练 MDP；
  跨栏不能进当前 lineage。
- teacher scan 为 0.1 m 网格（15x13=195 维，前向 0.2–1.6 m）：1.1 m 间距的下一道栏
  在视野内，但 0.05 m 宽的杆在 0.1 m 网格上会**间歇性 aliasing**（任意时刻约 50%
  概率落在采样点之间）。
- `T4_STAGE_E_TERRAINS_CFG` 无跨栏类地形；需要新建细杆子地形。
- zhuoqun 4 卡无 IsaacLab（Route W 已证），部署与 preflight 是 vault plan V3 的共享
  硬前置。

## 目标

- 在 Stage E teacher 1155D 观测合同上训出**独立的 T4 连续跨栏技能策略**：
  reward-only（无参考动作），从 `prov5` 稳定行走 checkpoint warm-start，新 lineage。
- strict zero-contact 语义下通过固定 evaluator gate（10 栏 @1.1 m/0.3 m，
  成功率 ≥90% @500 trials）。
- 从成功 rollout 提取 66D skill AMP clips 入库（含 manifest），保证与未来多专家
  merge（vault plan V7 开工时裁定 2 或 3 技能合并）合同兼容。
- 证据三件套：evaluator JSON + 连续回放视频 + lineage manifest。

## 非目标（Non-goals）

- Unitree G1 机器人（不在本仓库；规则契约的 G1 → T4 顺序属项目级排期，不属本 slice）。
- 其余 9 个障碍、赛道串联、折返重试状态机（多个障碍技能成熟后另起 Spec）。
- depth 蒸馏与部署观测（归 depth slice，沿用已批 teacher→depth 合同）。
- 修改 1155D schema、`legged_lab/assets/t4/schemas.py`、plant 或冻结部署合同。
- 跳跃专项奖励（允许策略自发选择跳，不专门 shaping）。
- 动态可倒杆进训练分布（仅 evaluator 非 gate 报告项）。
- merge 本身（G3/V7 属 vault plan；本 Spec 只保证合同兼容）。
- 真机部署与安全验收。

## 用户 / 调用者（Users / Callers）

- 使用者为项目所有者本人的训练管线；入口是 `legged_lab` 下新增的跨栏训练 /
  evaluator 脚本，长任务一律 tmux。
- 产物消费者：未来多专家 merge（loco + vault + hurdle 三技能候选，V7 开工时裁定）
  与后续 depth 蒸馏 slice。

## 行为规格（Behavior Spec）

### 正常路径（Happy Path）

- 前向速度命令（0.4–1.0 m/s）下接近连续栏：策略基于 scan 提前调整步态与落脚，
  摆动腿在 0.3 m 杆上方通过，双脚与全身不触杆，过完全部栏后恢复正常行走至出口。

### 边界情况（Edge Cases）

- 栏前发后退命令：策略退出，不进入栏区（负例 0/100）。
- 原地冻结 / 超时：判失败，计入成功率分母。
- corridor 宽 2.4 m 且栏杆全宽贴 corridor，物理上无侧绕空间；仍保留 ordered gates
  判定，侧绕 / 跳 gate 一律判失败。
- 训练间距随机化（0.9–1.4 m）防「看到一根就盲目周期抬腿」的假感知策略。

### 接口 / 状态（Interfaces / State）

- 观测 / 网络 = Stage E teacher actor 合同：1155D（proprio 96x10 + 前向 195D scan +
  命令 + prev action），MLP 512/256/128；27D action，scale 0.25。
- 两条硬要求（实现方式 plan/implement 裁定）：**栏杆必须出现在 teacher scan 中**；
  **碰栏必须可被可靠计数**（地形 mesh 烘焙 vs 独立静态 prim + raycast 列表二选一）。
- reset = loco 一致站姿，不用 RSI（与 vault G2 同理，为 merge 铺路）。
- AMP：沿用 walk AMP + 既有 per-difficulty 衰减旋钮，跨栏地形映射到高 difficulty 使
  系数 ≈0；不新建 hurdle 判别器；gait clock 与 symmetry mirror loss 保持（交替领腿
  的镜像仍是合法跨栏）。开训前冻结进配置。
- skill AMP clips：成功 rollout 提取 66D 特征入库 + manifest，与 vault G1 clips 路径
  对称，只服务未来 merge 的 skill prior。

## 约束（Constraints）

- plant 唯一真值为 `legged_lab/assets/t4/t4.py` 现值；`prov5` 若失败并再改 plant/MDP，
  跨栏 warm-start 基线与已训部分作废重训（用户已接受，与 vault 侧同一风险条款）。
- 算力（用户裁定 2026-08-13）：zhuoqun 4 卡对半分——vault 2 卡 + 跨栏 2 卡并行；
  正式 lineage 2 卡可接受（放宽 vault plan「4 卡级 runtime」条款，两侧同待遇）；
  nubot 4 卡 Stage E 专属；本机 1 卡仅渲染 / 调试。
- 开训 gate：zhuoqun IsaacLab preflight 通过（与 vault V3 共享同一 gate）+ `prov5`
  出现稳定行走信号的 checkpoint 可用（nubot→zhuoqun 搬运，bundle/scp 惯例）。
- warm-start 语义：加载 policy/critic 权重，optimizer 与 AMP 判别器状态重置，
  开新 lineage；partial load 报告 loaded/skipped keys（沿用 checkpoint 兼容键合同）。
- 训练预算：probe 200–500 iter；正式 lineage 上限 ~15000 iter（fine-tune 性质）；
  预算是上限，不是成功标准。
- 验收不得以 reward 曲线、loss、checkpoint 存在替代；一律 evaluator + 回放。
- 代码遵循仓库 Black 120 / pre-commit；改动小而准，不动无关资产。

## 选定方案（Chosen Approach）

1. **跨栏地形**：新建细杆子 sub-terrain——训练随机化栏数 3–8、间距 0.9–1.4 m、
   高度随 terrain difficulty 从 0.10 m 爬到 0.35 m（自然课程）、宽度固定 0.05 m、
   前导距离 1–3 m；杆全宽贴 corridor。
2. **reward-only fine-tune**：从 `prov5` 行走 checkpoint warm-start（新 lineage）；
   碰栏惩罚 / 终止 + 摆动腿 clearance 与前向 task 奖励（细节 plan/implement 定）；
   从零训练保留为 warm-start 行为不良时的 fallback。
3. **scan aliasing**：先接受闪烁（50 Hz 下杆相对 scan 网格连续扫过，接近全程有大量
   命中帧 + proprio 历史兜底）；probe 阶段观察「首杆碰撞率 / 是否预抬腿」，不达标
   再切 raycast-only 加宽 proxy（0.12 m 隐形杆，进 privilege schema 文档，属 schema
   变更需开新 lineage）。
4. **strict zero-contact**：栏杆静态化，机器人任何 body 触栏即判失败（训练惩罚 /
   终止 + evaluator 碰栏计数为零才过）；规则歧义 #7 的保守工程解释。
5. **evaluator**：corridor 2.4 m + ordered gates，固定 10 栏 @1.1 m / 高 0.3 m /
   宽 0.05 m，前导 3 m + 出口 3 m，固定前向命令 0.7 m/s。

## 拒绝方案（Rejected Options）

- **vault 式 mimic 流水线**（找 / 做跨栏参考动作 → tracking → 蒸馏）：脚上任务文献
  证明 reward-only 可学，参考动作投入过重。
- **并入下一条 Stage E 主 lineage**：拖延 student 蒸馏时间表，把 loco 回归与跨栏
  能力搅在一起；Stage E 主线是一切的 backbone，不做实验场。
- **动态可倒杆作为训练 / gate 判定**：泡沫倒伏物理不可信、per-env 动态体代价高；
  降级为报告项。
- **从零训练为主路线**：warm-start 省掉基础行走的重复学习；从零只作 fallback。

## 验证策略（Verification Strategy）

### 基线证据（Baseline Evidence）

- `prov5` 稳定行走 checkpoint 的行为（warm-start 输入与对照）。
- 零策略 RED case：evaluator 产出完整 JSON 且成功率为 0、失败原因分类正确。
- 现有合同测试保持绿：`python -m pytest tests/test_t4_asset_migration.py`、
  `tests/test_t4_observation_contracts.py`。

### 自动检查（Automated Checks）

- 新增纯 Python 合同测试：栏杆布局真值（数量 / 间距 / 高度随 difficulty 映射 /
  全宽贴 corridor）、碰栏判定函数（给定接触位置 vs 杆位真值表）、evaluator schema
  字段、1155D 观测合同回归。
- `pre-commit run --all-files`。

### Smoke / E2E 检查

- 任务注册后 2-env smoke（无 NaN/Inf/OOM，栏杆在 scan 中可见、碰栏计数生效）。
- probe（200–500 iter）观察数值健康 + 首杆碰撞率 / 预抬腿信号，再进正式 lineage
  （2 卡，tmux）。
- gate 用固定 evaluator 跑成功率 + 渲染连续回放视频供人工复核。

### 负向 / 边界检查（Negative / Boundary Checks）

- 后退命令负例 0/100 进入栏区。
- 超时 / 冻结判失败；ordered gates 判定跳 gate / 逆序失败。
- gate 通过前重扫 nonfinite action、joint hard-limit、碰栏计数为零。

### 文档 / 状态检查（Documentation / State Checks）

- lineage manifest（warm-start 来源 checkpoint、任务名、commit、数据 SHA）随 gate
  产出；`.harness/state.md` 与 work index 同步。

### 完成前所需 fresh evidence

- 跨栏 evaluator JSON + 连续回放视频 + lineage manifest，checkpoint 固定（禁止
  latest 自动选择），全部在 IsaacLab 产出。本 slice 不做 MuJoCo cross-sim（归
  depth slice）。

## 能力缺口（Capability Gaps）

- **跨栏地形生成与碰栏计数**：新代码；栏杆布局真值模块必须纯 Python 可测。
- **corridor + ordered gates evaluator**：语义沿用 vault plan V2；先完成者建骨架，
  后者复用，不建硬依赖。
- **zhuoqun IsaacLab 部署**：与 vault V3 共享硬前置；跨栏 1-env spawn 加入 preflight
  清单。
- **checkpoint 搬运**：nubot→zhuoqun（bundle/scp 惯例）。
- **人工复核**：回放视频行为裁定需用户目视确认。

## 成功标准（Success Criteria）

- **Gate**：固定 evaluator（10 栏 @1.1 m、高 0.3 m、宽 0.05 m、corridor 2.4 m、
  前向 0.7 m/s）成功率 **≥90% @500 trials**；成功 = 顺序过全部栏 + 全程零碰栏 +
  不摔 + 到达出口；超时 / 冻结计失败。
- **负例**：后退命令 0/100 进入栏区。
- **报告项（不 gate）**：动态可倒杆场景（轻质自由刚体近似泡沫，~100 trials）、
  0.35 m 高度、0.9 m 密间距、平均通过时间。
- 三证齐全（evaluator JSON + 连续回放 + lineage manifest）。

## 残余风险（Residual Risks）

- 0.05 m 杆 scan aliasing 可能导致感知不稳；缓解：probe 信号裁定 + raycast proxy
  fallback（schema 变更 + 新 lineage）。
- walk AMP 衰减不足以放开高抬腿；缓解：AMP 系数旋钮开训前冻结进配置，probe 观察。
- warm-start 的 walk 步态先验过强（不抬腿贴地走）；fallback 从零训练。
- `prov5` 失败并再改 plant/MDP → warm-start 基线与已训部分作废重训（用户已接受）。
- 2 卡吞吐约减半，墙钟拉长；预算按 iter 计不变。
- zhuoqun IsaacLab 部署工期不确定（与 vault 共享该风险）。

## Plan 交接（Plan Handoff）

- 当前切片 / Active slice: H0 跨栏地形布局真值与碰栏判定纯 Python 合同（本机可验证）。
- 建议下一 skill / Suggested next skill: plan（本会话内完成）。
- 计划提示 / Planning notes: H0–H2 零训练代码面与 vault V0–V2 并行；训练侧共享
  vault V3 preflight gate；GPU 划分与 tmux 纪律要写进 plan。
- 建议里程碑 / Suggested milestones: H0 合同 → H1 任务注册 + smoke → H2 evaluator +
  RED → H3 preflight + checkpoint 搬运 → H4 probe + 正式训练 + gate + clips 入库。

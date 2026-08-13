# Spec - T4 走跑与 1m 翻箱合并：heightscan 统一策略（LightLP V-B/V-C 复现）

> 状态 / Status: user-approved (2026-08-13)
> Owner: user
> Date: 2026-08-13
> 来源请求 / Source request: 复现 arXiv:2608.02653（Light-Loco-Parkour）第二阶段蒸馏，
> 使已有 Stage E 高程图走跑 teacher 与 1m 翻箱 tracking 模型自然衔接（brainstorm 会话 2026-08-13）。

## 背景

仓库已按 `docs/specs/2026-08-12--t4-unified-depth-locomotion.md` 训练 Stage E 特权 loco
teacher（`stage_e_prov5`，nubot 4 卡进行中，evaluator 未通过）。另有 PHP 栈训出的 1m
翻箱 tracking teacher（`t4_vault_student_minimal_20260812.zip` 内 `model_29999.pt`：150D
actor obs / 276D critic / 27D action / uniform action scale 0.5，参考
`overbox_1m_t4_mjcf_fps50.npz`，346 帧 @50Hz，固定 1.0m³ 箱）。两者是孤立能力：loco
teacher 遇 1m 箱只能停或撞；tracking teacher 离开参考与固定箱无法自主决策。

LightLP 的答案是三段：V-B1 把技能 expert 蒸馏到 heightscan+速度命令观测、V-B2 丢
expert 做技能泛化、V-C 多专家 DAgger 合并 + transition 组 RL 微调（稀疏过箱奖励 +
AMP 判别器在触发点从 loco prior 切到 skill prior）。论文消融：无 transition 组 0–33%，
有 98%；去 AMP 51%。

两个已核实的阻塞事实：

1. **plant 不一致**：vault teacher 训练所用 plant（MJCF spherehand、踝 10/0.5、arm 01-04
   力矩 25 N·m、scale 0.5、全零默认位）与 Stage E plant（URDF `t4_std.urdf`、踝 pitch
   80/4 roll 20/1、arm 36 N·m、scale 0.25、屈膝站姿）在踝刚度上差 8 倍。`model_29999`
   冻结复用不可行。
2. **zip 内 depth student 证据无效**：`evidence/model6500/evaluation.json` 成功率 0%
  （16/16 在 step 0 因 `wrist_body_pos` 终止），不得作为已有能力引用。

## 目标

- 在 Stage E plant 上从零重训 1m 翻箱 mimic teacher（G1）。
- 把 mimic teacher 蒸馏为与 Stage E 同观测合同的 heightscan 技能策略并泛化（G2）。
- 用多专家 DAgger + transition 组产出**单一 heightscan 策略**：会走、会翻 1m 箱、凭
  scan 与速度命令自主切换并恢复行走，无 reference / skill label / 运行时状态机（G3）。
- 三个 gate 各自产出 evaluator JSON + 连续回放 + lineage manifest。

## 非目标（Non-goals）

- depth 蒸馏与部署观测（论文 §VI；留给下一 slice，沿用已批 Spec 的 teacher→depth 合同）。
- 第二个技能（speed-vault / reverse-vault）、技能链、高度泛化验收（验收只看 1m）。
- 真机部署与安全验收。
- 修改已冻结的部署 Actor 合同或 `legged_lab/assets/t4/schemas.py` 中的观测 schema。
- PHP 栈（`whole_body_tracking`）的继续开发；它退役为配方与参考数据来源。

## 用户 / 调用者（Users / Callers）

- 使用者为项目所有者本人的训练管线；入口是 `legged_lab` 下新增的训练 / evaluator
  脚本，长任务一律 tmux（nubot 现有惯例同样适用于 zhuoqun）。
- 产物消费者是下一个 depth 蒸馏 slice：G3 checkpoint 将作为其唯一 teacher。

## 行为规格（Behavior Spec）

### 正常路径（Happy Path）

- 前进速度命令（量级 0.6–1.0 m/s）+ 前方 corridor 内 1m 箱：策略接近时自主减速，
  以手/前臂/躯干接触箱体完成 climb-and-over（爬上—通过—落到另一侧），落地后恢复
  正常行走并继续跟踪命令。
- Stage E 各 loco 地形（平地、rough、楼梯上下等）：行为与 Stage E teacher 一致，
  不因合并显著回退。

### 边界情况（Edge Cases）

- 箱前 1m 发后退命令：策略后退离开，不得被箱子"吸上去"（论文 command adherence 负例）。
- corridor 内侧向绕开箱子到达出口：判失败（继承已批 Spec 的 corridor + ordered gates 裁定）。
- 翻越中途失稳跌落：允许终止；不要求中途恢复能力（首版）。
- 0.8m 箱与梯形未见外形障碍：仅报告成功率，不作 gate。

### 接口 / 状态（Interfaces / State）

- G1 新增 tracking 任务（LeggedLab 内移植 PHP mimic 配方：box 场景、RSI、全局 anchor
  与 wrist 奖励、`anchor/foot/wrist` 偏差终止语义）；其 150D tracking 观测合同只存在于
  G1 内部，不外泄。
- G2 / G3 策略观测与网络 = Stage E teacher actor 合同：1155D（proprio 96×10 + 前向
  195D scan + 命令 + prev action），MLP 512/256/128。
- G3 合并环境按组划分（loco 组 / skill 组 / transition 组）：per-group reward 与
  termination，critic 观测加组别 one-hot，actor 无任何标签。
- AMP：G1/G2 阶段系数为 0；G3 transition 组双 prior——loco 侧用现有 66D expert，
  skill 侧从 G1 成功 rollout 提取 66D 特征，判别器在箱前缘固定偏移触发点切换。
- 数据：参考动作 `overbox_1m_t4_mjcf_fps50.npz`（按关节名重排进 LeggedLab 顺序）；
  skill AMP clips 由 G1 rollout 生成并入库（含 manifest）。
- 证据输出：每 gate 固定 evaluator（JSON 至少含 lineage、seed、checkpoint、bucket、
  成功判定、失败原因、禁止接触与 hard-limit 计数）+ 连续回放视频。

## 约束（Constraints）

- plant 唯一真值为 `legged_lab/assets/t4/t4.py` 现值；若 Stage E 后续再改 plant，
  技能侧接受重训（用户已裁定此风险）。
- G1 从零训练（预算 ~20000 iter，多卡），不做 `model_29999` warm-start（用户裁定：
  合同复刻成本高于多卡从零）。
- G3 开工硬前置：Stage E teacher evaluator 通过（继承现有 stop gate"teacher evaluator
  通过前不启动 student 蒸馏"）。G1/G2 代码与数据准备即刻开工，`prov5` 出现稳定行走
  信号即可开训。
- 算力分配：nubot 4 卡专属 Stage E；zhuoqun 4 卡跑技能侧（开训前必须通过 runtime
  preflight）；本机 1 卡仅渲染/调试。
- 验收不得以 reward 曲线、loss、checkpoint 存在替代；一律 evaluator + 回放。
- 代码遵循仓库 Black 120 / pre-commit；改动小而准，不动无关资产。

## 选定方案（Chosen Approach）

按 LightLP 三段复现，全部在 TienKung-Lab（LeggedLab 栈）内进行：

1. **G1 mimic teacher（从零）**：移植 PHP mimic 配方到 LeggedLab 新任务，plant 换为
   Stage E 现值，参考与箱体沿用 1m 原件。产出：能在新 plant 上稳定翻越 1m 固定箱的
   特权 tracking teacher + 成功 rollout（后者兼作 skill AMP 数据与 G2 的 per-config
   mimic 参考）。
2. **G2 heightscan 技能策略**：DAgger(+PPO) 把 G1 expert 蒸馏到 1155D 合同（蒸馏前
   拿掉 expert 的障碍尺寸/距离特权），随后丢 expert，在箱宽 x∈[0.7,1.0] m、深
   y∈[0.5,1.5] m、位置随机下加"箱后目标" task reward 做技能泛化；reset 从 loco 一致
   站姿开始（不用 RSI），为 G3 衔接铺路。
3. **G3 合并 + transition**：一个 student、三组环境——loco 组跑冻结 Stage E teacher
   的地形与 DAgger 监督、skill 组跑 G2 策略的箱体地形与 DAgger 监督、transition 组
   （箱嵌入可走地形，三区域 reset：箱前 / 技能区 / 箱后）无逐帧监督，用稀疏过箱位置
   奖励 + 密集接近奖励 + 相位切换 AMP 做 RL 微调。

## 拒绝方案（Rejected Options）

- **冻结 `model_29999` 直接当 DAgger expert**：plant 不一致（踝刚度差 8 倍），闭环行为
  大概率失效；已被证据否决。
- **`model_29999` warm-start 微调**：需精确复刻 150D obs 布局 + 0.5 scale + 零位 offset
  才有意义，收益不确定；用户裁定多卡从零更省心。
- **直接在 depth 观测上合并**：把"技能/衔接失败"与"感知失败"搅在一起，违反已批
  Spec 的 teacher→depth 分层。
- **V-B 留在 PHP 栈训练**：plant 对齐要维护两份，且 V-C 时 expert 必须在合并环境内
  逐状态查询，跨栈迟早要合。
- **无 expert 的 reward-only 联训**：论文消融（Teacher w/o ED）不收敛；研究笔记
  `2026-08-13--t4-box-climbing.md` 已裁定不走。

## 验证策略（Verification Strategy）

### 基线证据（Baseline Evidence）

- `stage_e_prov5` 的 TensorBoard 与（通过后的）Stage E evaluator 结果，作为 G3 loco
  回退对照基线。
- `model_29999` 在原 PHP 环境的行为仅作参考对照，不作为任何 gate 证据。
- 现有合同测试通过状态：`python -m pytest tests/test_t4_asset_migration.py`、
  `tests/test_t4_observation_contracts.py`。

### 自动检查（Automated Checks）

- 上述两个既有 pytest 合同测试保持绿。
- 新增纯 Python 合同测试：参考 npz 重排（关节名双射、帧数/频率）、G2/G3 观测维度
  与 1155D schema 一致、skill AMP clip 66D 特征形状与 manifest。
- `pre-commit run --all-files`。

### Smoke / E2E 检查

- 每个训练阶段先短 probe（数值健康：NaN/Inf、OOM、吞吐；关键行为信号）再进正式
  lineage（继承已批 Spec 规则）。
- 每 gate 用固定 evaluator 跑成功率 + 渲染连续回放视频供人工复核。

### 负向 / 边界检查（Negative / Boundary Checks）

- 后退命令负例 0/100 上箱。
- corridor 侧绕、跳 gate、逆序 gate 一律判失败。
- 每 gate 通过前重扫 nonfinite action、joint hard-limit、组外禁止接触计数为零。

### 文档 / 状态检查（Documentation / State Checks）

- lineage manifest（teacher/skill/student checkpoint 与数据 SHA、任务名、commit）随
  每 gate 产出；`.harness/state.md` 更新 active slice 与 stop gate。

### 完成前所需 fresh evidence

- G1/G2/G3 三份 evaluator JSON + 对应连续回放视频，且 checkpoint 固定（禁止 latest
  自动选择），全部在 IsaacLab 产出。本 slice 不做 MuJoCo cross-sim（heightscan 无
  对应观测基建；cross-sim 属于 depth slice）。

## 能力缺口（Capability Gaps）

- **zhuoqun runtime 未验证**：Isaac Sim / IsaacLab 安装状态未知。缓解：准备阶段先跑
  preflight（可参考 zip 内 `tools/preflight.py` 思路），失败则先做环境部署。
- **rsl_rl 扩展**：现库无多组 DAgger runner、per-group reward/termination、critic 组别
  one-hot、AMP 双判别器相位切换。需在仓库内置 `rsl_rl` 上实现，属本 slice 代码工作。
- **LeggedLab 地形/评估扩展**：1m 箱 corridor 地形、transition 三区域 reset、strict
  evaluator（crossing sustained + stable landing 语义可沿用 zip evaluator 定义）。
- **人工复核**：回放视频的行为裁定需要用户目视确认（agent 无法独立替代）。

## 成功标准（Success Criteria）

- **G1**：新 plant 上 1m 固定箱 strict evaluator（持续翻越 + 稳定落地 + 无 hard
  violation）成功率 ≥95%，≥100 trials。
- **G2**：1m 箱、宽深/位置随机，成功率 ≥90%，500 trials；观测中无参考、无障碍特权。
- **G3**（同一 checkpoint 四项全过）：
  1. Stage E evaluator 全 bucket 相对 loco teacher 成功率回退 ≤5 个百分点；
  2. 1m 箱 corridor 成功率 ≥90%；
  3. transition course（走 ≥3m → 翻箱 → 走 ≥3m 至出口，corridor + ordered gates）
     成功率 ≥90%，100 trials；
  4. 后退命令负例 0/100 上箱。
- 报告项（不 gate）：0.8m 箱成功率、梯形未见外形成功率。

## 残余风险（Residual Risks）

- 1m ≈ 0.71H（T4 站高 ~1.4m）接近论文 student 有效上限 0.77H；G1 若卡在 95% 以下，
  备选路径是把参考降 5–10cm 做一轮 LightLP 式增强再抬回 1m——属范围变更，需回来
  重新裁定。
- `prov5` 若失败并再改 plant/MDP，技能侧已训部分作废重训（用户已接受）。
- per-group reward/termination 与双判别器改造是本 slice 最大代码面，可能引入训练
  不稳定；缓解：每步先 smoke probe，合同测试先行。
- transition 稀疏奖励探索难；缓解：论文三区域 reset 照抄。
- 爬箱中段 scan 可观测性在 teacher 层无遮挡问题（scan 是特权俯视采样），遮挡属
  depth slice 风险，此处不处理。

## Plan 交接（Plan Handoff）

- 当前切片 / Active slice: G1 准备——mimic 任务移植 + 1m 箱场景 + strict evaluator +
  zhuoqun preflight（零训练代码面先行）。
- 建议下一 skill / Suggested next skill: plan
- 计划提示 / Planning notes: G1/G2 与 Stage E 并行；G3 被 Stage E evaluator 硬 gate；
  skill AMP clips 在 G1 通过后立即入库；rsl_rl 多组扩展可与 G1 训练并行开发。
- 建议里程碑 / Suggested milestones: M-prep（移植/地形/evaluator/preflight）→ M-G1 →
  M-G2 → M-G3。
- 里程碑验收提示 / Per-milestone acceptance hints: M-prep 以合同测试 + preflight 通过
  为准；M-G1/G2/G3 以上述成功标准与 fresh evidence 为准。

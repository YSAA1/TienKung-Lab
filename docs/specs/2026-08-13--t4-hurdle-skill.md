# Spec - T4 连续跨栏（100m 障碍赛障碍 #2，Stage E 课程地形 bucket）

> 状态 / Status: user-approved (2026-08-13)；**2026-08-14 路线修订 user-approved**：
> 跨栏不再作为独立 skill lineage，改为并入 Stage E locomotion 课程的地形 bucket。
> 货架: half-living。课程已并入 Stage E 并随走跑阶段关闭；100m `rule` 零碰杆 evaluator 仍未做，不要当当前执行队列。
> Owner: user
> Date: 2026-08-13（修订 2026-08-14）
> 来源请求 / Source request: 用 RL 训练 100m 障碍赛的连续跨栏并集成进现有 plans。
> 修订依据：用户裁定「跨栏是纯脚上地形障碍，和翻箱子（手上接触技能）不是一类，
> 单独训练与现有 Stage E 割裂；可以放进正常 locomotion 课程一起训练」。

## 背景

规则契约 `docs/research/100m-obstacle-rule-contract.md` 障碍 #2「连续跨栏」：多道软质
泡沫栏，间距 1.1 m；杆长 2.4 m、宽 0.05 m、高 0.3 m；跨越或跳跃通过；明文失败 =
单脚到达障碍终点前摔倒，或碰倒栏杆。规则歧义 #7：栏杆数量未写明、「碰倒」未定义
（触碰未倒是否失败不明）——工程上取保守解释（strict zero-contact）并保持可配置。

任务定性：**纯脚上任务**。0.3 m 对 ~1.4 m 站高的 T4 约 0.21H 摆动腿 clearance，属
「高抬腿跨步」量级，无需手接触，与楼梯 / 乱石同类，而非 vault 类手上技能。因此它
的正确位置是 Stage E 地形课程里的一个 bucket（与 stairs、rough 并列），而不是一条
独立 skill lineage——后者会把 loco 能力复制一份出去、再逼未来做 3 技能 merge。

工程事实：

- 地形 bucket 方式下杆子是静态 tile mesh 的一部分：height scan 天然可见（0.1 m 网格
  对细杆有间歇 aliasing，见风险）；碰杆走既有 terrain 接触语义——脚绊杆触发
  `feet_stumble`，小腿触杆触发 shank 接触惩罚，躯干/手臂触杆触发 undesired contact
  与终止，绊倒触发姿态终止（-200）。**训练期 MDP 零新增项。**
- Stage E 课程按 terrain difficulty 行升降级（promote = 径向位移 ≥4 m），AMP 系数
  按 difficulty 线性衰减：hurdle bucket 高难度行自动放开非专家步态。
- `stage_e_prov6_local`（无跨栏地形）已证明修复后 MDP 数值健康；地形分布变更 =
  新 lineage 从零。

## 目标

- 在**不改 1155D 观测合同、不加 MDP 项**的前提下，把连续跨栏地形并入
  `T4_STAGE_E_TERRAINS_CFG`，训练单一 Stage E teacher 同时掌握
  平地 / 乱石 / 楼梯 / **连续跨栏**。
- 新 lineage `stage_e_prov7_hurdle` 从零训练（nubot 2 卡 × 2048 env，25000 iter
  预算，2026-08-14 用户指示），取代 `stage_e_prov6_local` 成为 Stage E 主线。
- 跨栏布局几何有纯 Python 真值模块 + 合同测试，供地形生成与未来 strict-contact
  evaluator 共用。
- 最终验收仍走固定 evaluator gate（见成功标准），证据三件套不变。

## 非目标（Non-goals）

- **独立跨栏 skill lineage、warm-start fine-tune、66D skill AMP clips 提取**
  （2026-08-14 路线修订废弃；merge 候选回到 loco + vault 两技能）。
- Unitree G1、其余 9 个障碍、赛道串联、折返重试状态机。
- depth 蒸馏与部署观测（归 depth slice，沿用已批 teacher→depth 合同）。
- 修改 1155D schema、`legged_lab/assets/t4/schemas.py`、plant 或冻结部署合同。
- 跳跃专项奖励、碰杆专用奖励/终止项（既有接触语义已覆盖；probe 发现不够再议，
  那属 MDP 变更 = 再开 lineage）。
- 动态可倒杆进训练分布（仅 evaluator 非 gate 报告项）。
- 真机部署与安全验收。

## 行为规格（Behavior Spec）

### 正常路径（Happy Path）

- 机器人在 hurdle tile 上收到任意 heading 速度命令：基于 scan 预判杆位、调整步态，
  摆动腿从杆上方通过、不触杆，连续跨过 2–3 道环形杆后继续正常行走；课程将其
  推向更高杆（0.05→0.35 m）。

### 边界情况（Edge Cases）

- 杆高 difficulty 低端（0.05 m）近似平地，保证课程入口平滑。
- 碰杆绊脚：`feet_stumble` 惩罚 + 摔倒姿态终止给出负梯度；不会静默。
- 环形布局（四向对称方环）使任意命令方向都会遇杆，不存在「绕开杆」的假解；
  tile 间距/杆高 per-tile 随机防「看到一根就盲目周期抬腿」。
- 站立命令（零速）在 hurdle tile 上合法：站在环间平地即可。

### 接口 / 状态（Interfaces / State）

- 观测 / 网络 / action 完全不变：1155D teacher actor（proprio 96×10 + 前向 195D
  scan + 命令 + prev action），MLP 512/256/128，27D action scale 0.25。
- 地形接口：`legged_lab/terrains/hurdle_layout.py` 纯 Python 布局真值
  （环半宽序列、difficulty→杆高、杆 AABB、碰杆点判定）；
  `MeshHurdleRingsTerrainCfg`（`terrain_generator_cfg.py`）消费同一真值生成 trimesh。
- 课程 / AMP / 终止 / 奖励：全部沿用 Stage E 现值，零改动。

## 约束（Constraints）

- plant 唯一真值 `legged_lab/assets/t4/t4.py` 现值（commit `859fe24` 姿态终止修复
  之后的 MDP 冻结面）。
- 算力（2026-08-14 用户指示）：prov7 用 **nubot 2 张卡、每卡 2048 env（合计
  4096），25000 iter 预算**；nubot 其余 2 卡留空闲；zhuoqun 4 卡不再为跨栏保留
  配额（原 2+2 裁定中跨栏侧取消，vault 是否回收 4 卡由 vault surface 裁定）；
  本机 1 卡继续跑 prov6_local 对照组。
- 地形分布变更 = 新 lineage 从零；`stage_e_prov6_local`（无跨栏地形）**继续在
  本机运行作为对照组**（2026-08-14 用户裁定，不因 prov7 启动停止），产物保留。
- 训练预算是上限不是成功标准；验收不得以 reward 曲线替代行为证据。
- 代码遵循仓库 Black 120 / pre-commit；改动小而准。

## 选定方案（Chosen Approach）

1. **环形跨栏地形** `hurdles` bucket：平地 tile + 以出生平台为中心的同心方环细杆
   （仿 pyramid stairs 的四向对称惯例，任意 heading 皆遇杆）。参数（真值常量在
   `hurdle_layout.py`）：tile 8×8 m、平台宽 1.6 m、border 0.25 m、环间距 per-tile
   随机 0.9–1.3 m（每 tile 2–3 环）、杆高随 difficulty 0.05→0.35 m、杆厚 0.07 m
   （近实物栏板 7 cm，兼顾 0.1 m scan 网格可见性）。
2. **占比重配**：`hurdles` 0.10；flat 0.10→0.08、random_rough 0.20→0.16、
   wave 0.10→0.06；stairs 4×0.0875、boxes 0.15、slopes 2×0.05 不动；总和 1.0。
3. **MDP 零新增**：杆合入静态 tile mesh，碰杆由既有 stumble / shank / undesired
   contact / 姿态终止语义惩罚；AMP per-difficulty 衰减自动覆盖 hurdle 高难度行。
4. **strict zero-contact 归 evaluator**：训练期不做逐杆碰撞计数；专用 corridor
   evaluator（10 栏 @1.1 m/0.3 m、corridor 2.4 m）用 `hurdle_layout` 的 AABB 真值
   做零碰杆判定，作为后续工作项，不阻塞训练启动。

## 拒绝方案（Rejected Options）

- **独立跨栏 skill lineage（原 2026-08-13 路线）**：与 Stage E 割裂、复制 loco
  能力、逼 3 技能 merge；2026-08-14 用户裁定废弃。
- **vault 式 mimic 流水线**：脚上任务文献已证 reward-only 可学，参考动作投入过重。
- **一排直线横杆 sub-terrain**：机器人任意 heading 可绕开，产生假解；改环形。
- **碰杆专用奖励 / 终止项**：既有接触语义已覆盖，加项 = MDP 变更 + 调参面扩大。
- **动态可倒杆进训练**：泡沫倒伏物理不可信、per-env 动态体代价高；留报告项。

## 验证策略（Verification Strategy）

### 基线证据（Baseline Evidence）

- `stage_e_prov6_local`（无跨栏）行为视频与 TB 曲线（对照组）。
- 既有合同测试保持绿：`tests/test_t4_asset_migration.py`、
  `tests/test_t4_observation_contracts.py`、`tests/test_t4_terrain_curriculum.py`。

### 自动检查（Automated Checks）

- `tests/test_t4_hurdle_contracts.py`：环半宽序列（≥2 环 / 等间距 / border 内）、
  difficulty→杆高单调 + 端点 + clamp、杆 AABB 正反例、常量自洽（杆厚 < 最小间距、
  平台 + 首环 ≤ border 内）。纯 Python，任意机器可跑。
- `pre-commit run --all-files`。

### Smoke / E2E 检查

- 本机 64-env 3-iter smoke：hurdle 地形生成不炸、无 NaN/Inf、reward 有限。
- prov7 启动后监控：显存 / steps/s / episode length / terrain_levels；hurdle 行
  是否随课程 promote（Curriculum/terrain_levels 上行 = 策略在杆上取得径向进展）。

### 负向 / 边界检查（Negative / Boundary Checks）

- difficulty 0 行杆高 0.05 m（近平地）不应显著拉低 episode length。
- 中期渲染回放（本机 headless 录制惯例）人工确认「预抬腿过杆」而非「贴地犁杆」。

### 文档 / 状态检查（Documentation / State Checks）

- lineage 记录（commit、env 数、卡数、日志路径、tmux）进 `.harness/state.md`；
  work index 跨栏行改为 Stage E 课程内工作面。

### 完成前所需 fresh evidence

- 固定 checkpoint 的跨栏 corridor evaluator JSON + 连续回放视频 + lineage
  manifest（三证齐全，IsaacLab 产出）——归后续 evaluator 工作项。

## 能力缺口（Capability Gaps）

- ~~跨栏地形生成与布局真值~~（本 slice 已交付：`hurdle_layout.py` +
  `MeshHurdleRingsTerrainCfg` + 合同测试）。
- corridor + ordered gates evaluator：语义沿用 vault plan V2 骨架，后续工作项。
- 人工复核：中期与 gate 回放视频需用户目视确认。

## 成功标准（Success Criteria）

- **训练侧观察项（不 gate）**：prov7 在 25000 iter 预算内，hurdle bucket 的
  terrain_levels 持续上行且中期回放可见明确抬腿过杆行为。
- **Gate（不变，归 evaluator 工作项）**：固定 evaluator（10 栏 @1.1 m、高 0.3 m、
  杆厚 0.05 m、corridor 2.4 m、前向 0.7 m/s）成功率 **≥90% @500 trials**；成功 =
  顺序过全部栏 + 全程零碰栏 + 不摔 + 到达出口；超时 / 冻结计失败。
- **负例**：后退命令 0/100 进入栏区。
- **报告项（不 gate）**：动态可倒杆（~100 trials）、0.35 m 高度、0.9 m 密间距、
  平均通过时间。
- 三证齐全（evaluator JSON + 连续回放 + lineage manifest）。

## 残余风险（Residual Risks）

- 0.07 m 杆在 0.1 m scan 网格仍有间歇 aliasing；缓解：proprio 历史 + 连续扫过 +
  中期回放裁定；不达标再议 raycast proxy（schema 变更 = 新 lineage）。
- 高杆（0.35 m）+ 姿态终止（-200）可能让课程在 hurdle 高难度行卡住；观察
  terrain_levels，卡住属课程调参议题（fine-tune 不改 MDP 合同）。
- AMP 衰减不足以放开高抬腿步态；观察中期回放；旋钮已冻结进配置。
- 8×8 tile 只放得下 2–3 环，「连续 10 栏节奏」由 evaluator corridor 场景专门验收，
  训练 tile 不模拟全赛道。
- 2 卡 × 2048 env 吞吐推算 ~23–30 h 墙钟（25000 iter 预算上限）。

## Plan 交接（Plan Handoff）

- 当前切片 / Active slice: H2 prov7 lineage 启动与稳定性监控（H0 布局合同、
  H1 地形接入已随本修订交付）。
- 建议下一 skill / Suggested next skill: implement（本会话内完成启动）。
- 计划提示 / Planning notes: evaluator（H3/H4）与 vault V2 共享 corridor 骨架；
  prov7 期间本机 GPU 空闲可做渲染复核。

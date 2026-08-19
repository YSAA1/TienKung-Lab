# Spec - T4 梅花桩 Stage E 课表与跨栏 0.30 稳定

> 状态 / Status: user-approved (2026-08-15)；2026-08-17 / 08-18 修订执行面
> 货架: living 目标规格，但文中「T-compat / T-paper 双老师」已不是执行路线。现行执行：`docs/plans/2026-08-19--t4-sparse-easy-geometry-s6-plan.md`（s6 从零，sparse Actor 1937D，Stage E 1155D 不动）。
> Owner: user
> Date: 2026-08-15
> 来源请求 / Source request: 走跑已可用；补 LightLP 稀疏落脚（梅花桩）并把跨栏稳定在 0.30–0.35，不继续加高。
> 2026-08-17 修订: sparse teacher 从零开训、收窄踏石、允许打破该任务 1155D；不做旧学生短 FT。默认 Stage E `TEACHER_ACTOR_OBS_DIM=1155` 仍冻结。

## 背景

走跑 / 深度学生阶段已关闭。用户在 MuJoCo `loco` 上看到 **0.25 / 0.28 m 可过、0.30 m 经常过不去且常碰杆**。Teacher 同高度也不稳。课表 `terrain_levels` 均值约 5–6（10 行，`d = level / 9`）对应跨栏杆高约 **0.22–0.25 m**；0.30 m 约 level 7.5。H2 83/100 只是前进 3 m，犁杆也能过。现有 `boxes` 是连续 0.45 m 网格、高差 0–15 cm，不是稀疏落脚。

100m 规则没有「梅花桩」。本 Spec 的梅花桩 = Light-Loco-Parkour Fig. 4 **(5) raised pillars + (6) stepping stones**（离散桩顶/石面，间隙不可踩）。不是 #4 绕桩，不是 (9) 窄板。

LightLP §IV：稀疏落脚在 **同一个 loco teacher** 里和楼梯/rough 一起训；学生只是后来把 scan 策略迁到深度。Table I 为这类地形多了 **illegal-footstep** 和 **velocity-slack**；足底向下扫描只算奖励 / 给 critic，不上实机。导出仍是深度学生（已批 `docs/specs/2026-08-12--t4-unified-depth-locomotion.md`）。

## 目标

- 新开 **两条从零 teacher lineage**（不续 `stage_e_prov7_hurdle`），课表相同：旧 Stage E 地形 + 桩阵 + 踏石 + LightLP 落脚奖励 + 跨栏碰杆/高杆暴露。
  - **T-compat**：Actor 仍 1155D，可蒸现有学生。
  - **T-paper**：Actor 只多当前帧左右脚接触 2 维（1157D）。足底 scan 不进 Actor。现有学生不能蒸这条；以后另开学生。
- **保留** `stage_s_head35`。本机对它做 0.30 跨栏 PPO 微调（碰杆惩罚，不蒸会犁杆的旧老师）。
- 实机合同不变：导出只有深度 + 本体历史 + 速度命令 + 上一动作。T-paper 不直接上机。

## 非目标（Non-goals）

- 把课表杆高推过 0.35 m。
- 作废或覆盖 `stage_s_head35`。
- 改冻结的默认 1155D schema（T-paper 用独立任务 / 独立 dim，不改 `TEACHER_ACTOR_OBS_DIM`）。
- 无老师的深度学生从零踩梅花桩。
- 窄板 / S 弯桥 / 绕桩 / 翻箱 / climb-vault LightLP 全身流水线。
- 把 HeightScan、足底 scan、接触真值写入导出 `pi_loco`。
- 用 reward、episode length、checkpoint 存在代替行为验收。
- 打断 zhuoqun 翻箱 G1/G2。

## 用户 / 调用者（Users / Callers）

- 训练：nubot 2+2 开两条 teacher；本机 1 卡微调旧学生。
- 观看：用户用 `sim2sim_t4_depth_student --course loco` 看 0.30。
- 评估：teacher play / 现有 hurdle progress；strict 零碰栏 evaluator 仍是后续项，不挡开训。

## 行为规格（Behavior Spec）

### 正常路径（Happy Path）

1. T-compat / T-paper 在 Isaac 课表上从零训。机器人在踏石/桩阵上选有效落脚前进；跨栏高行走样增加，碰杆受罚。
2. 本机旧学生在学生观测上微调，目标是 MuJoCo `loco` 第三根 0.30 能抬腿过、少碰、不摔。
3. T-compat 以后可蒸新学生；T-paper 以后另蒸。本机 0.30 不会自动长到新学生上。

### 边界情况（Edge Cases）

- 只跟踪、不加 slack：LightLP 写明会停在稀疏落脚前。本 Spec 两条老师都加 slack。
- T-paper 误蒸进 1155D 学生：形状对不上，必须硬失败。
- 旧学生微调若仍犁 0.30：记为探针失败，不自动改 T-compat 合同。
- 梅花桩石顶 < 0.25 m：0.1 m scan 会看丢；第一版石顶 0.30–0.45 m。

### 接口 / 状态（Interfaces / State）

- 布局真值：`legged_lab/terrains/stepping_stone_layout.py`（纯 Python）。
- 地形：`T4_STAGE_E_SPARSE_TERRAINS_CFG`（或等价），在 Stage E 基础上加 `stepping_stones`、`raised_pillars`，从 flat/rough/wave 挪占比；hurdles 保留并提高高行走样。
- 奖励（两条老师 + 旧学生 FT）：`illegal_footstep`、`velocity_slack`、`hurdle_bar_contact`。足底射线只进 reward/critic。
- 任务名：
  - `t4_loco_teacher_sparse`（T-compat，1155D）
  - `t4_loco_teacher_sparse_paper`（T-paper，1157D）
  - 旧 `t4_loco_teacher` 不改默认，避免翻箱/旧脚本误用新课表。
- 旧学生 FT：独立入口，加载 `stage_s_head35`，PPO，不加载旧 teacher 标签。
- 冻结：`legged_lab/assets/t4/schemas.py` 的 `TEACHER_ACTOR_OBS_DIM` / 禁止字段不改。T-paper dim 另写常量。

## 约束（Constraints）

- 卡位：**nubot 2 + nubot 2 + 本机 1**。zhuoqun 留给翻箱。
- 地形或 MDP 变更 = 从零 lineage。
- 代码 Black 120；长训 / evaluator 必须 tmux。
- 实机方便：T-compat + 现有学生 = 当前上机合同。illegal-footstep / slack / 碰杆都不是实机传感器。

## 选定方案（Chosen Approach）

**三线并行，双 teacher 对照，旧学生保留。**

| 线 | 观测 | 卡 | 作用 |
|---|---|---|---|
| T-compat | 1155D | nubot 2 | 可部署蒸馏源 |
| T-paper | 1157D（+接触） | nubot 2 | 对齐 LightLP Actor 接触标志的试验 |
| 旧学生 0.30 FT | 现有 depth 学生 | 本机 1 | 当前观看面的短探针 |

两条老师共享地形与奖励。T-paper 只多 Actor 接触；足底 scan 两边都不进 Actor。

拒绝「只加地形不加 slack/illegal-footstep」：论文 §IV-C1 已说明会停在桩前。

## 拒绝方案（Rejected Options）

- **只微调旧学生、不再开 teacher**：0.30 探针可以，梅花桩从零用深度学生，论文无记忆是 0%。
- **先学生 0.30 再蒸梅花桩老师进同一学生**：现有 Distillation 会把更好的 0.30 拉回老师。
- **给导出学生加接触/足底 scan**：破部署合同，实机要新硬件。
- **一条老师兼容 + 改 Actor**：1155D 与 1157D 不能混在一个任务里。
- **独立梅花桩 skill + AMP clips**。

## 验证策略（Verification Strategy）

### 基线证据（Baseline Evidence）

- 现有 `tests/test_t4_hurdle_contracts.py`、`tests/test_t4_observation_contracts.py`、`tests/test_t4_terrain_curriculum.py`。
- 用户目视：MuJoCo `loco` 0.25/0.28 可过、0.30 不稳；teacher 同样不稳。

### 自动检查（Automated Checks）

- `tests/test_t4_stepping_stone_contracts.py`：石/桩几何、难度单调、石顶 ≥ 0.30、间隙与脚/scan 自洽。
- `tests/test_t4_sparse_reward_contracts.py`：slack 区间、illegal-footstep δ=0.1、碰杆点判定。
- `tests/test_t4_observation_contracts.py`：默认 teacher 仍 1155D；paper 任务 1157D；禁止字段未进 T-compat Actor。
- `python -m pytest tests/test_t4_asset_migration.py tests/test_t4_hurdle_contracts.py tests/test_t4_observation_contracts.py tests/test_t4_stepping_stone_contracts.py tests/test_t4_sparse_reward_contracts.py -q`

### Smoke / E2E 检查

- 本机 64-env 3-iter：两个新 teacher 任务不炸、无 NaN。
- nubot / 本机训练启动后：无 OOM/NaN，日志持续写。

### 负向 / 边界检查（Negative / Boundary Checks）

- 用 1155D loader 加载 T-paper checkpoint 必须失败。
- 零速命令不因 slack 被逼着往前走。
- 脚在石面中心且接触：illegal-footstep ≈ 0。

### 文档 / 状态检查（Documentation / State Checks）

- README 训练任务名；`.harness/work_index.md` 增加本工作面；翻箱行保留。

### 完成前所需 fresh evidence

- 合同测试绿。
- 三条训练在 tmux 里健康启动（本切片不要求训完）。
- 行为验收（`loco` 0.30、桩上回放）是后续切片，不在开训切片宣称。

## 能力缺口（Capability Gaps）

- 0.30 零碰栏 / 梅花桩 corridor evaluator 尚未交付；开训不依赖它们。
- 人工目视：MuJoCo `loco`、teacher 回放。
- nubot 同步仍可能走 git bundle。

## 成功标准（Success Criteria）

**开训切片（本 Spec 第一刀）：**

- 上述 pytest 绿。
- `t4_loco_teacher_sparse` actor 1155D；`t4_loco_teacher_sparse_paper` actor 1157D。
- nubot 两线 + 本机 FT 已在 tmux 启动，日志无即刻崩溃。

**行为切片（后续，不挡开训）：**

- 旧学生：`loco` 第三根 0.30 连续抬腿过、不摔；碰杆明显少于开训前（目视 + 以后 JSON）。
- T-compat：踏石/桩阵上能前进，不长期停在出生平台；跨栏少犁杆。
- `rule` 10×0.30 只报告。
- T-paper 不挡旧学生；只作对照。

## 残余风险（Residual Risks）

- 0.1 m scan 对细杆仍 alias；训练杆可加厚到 0.10 m，验收仍 0.30 / 厚 0.05。
- slack 可能鼓励莽撞；靠碰杆/摔倒/illegal-footstep 压。
- 本机 FT 改善的 0.30 不会进入新蒸学生。
- T-paper 更好也不能上机，除非再蒸不含接触的学生。
- 双卡 + 新地形可能降 env 数；OOM 则每卡 1024。

## Plan 交接（Plan Handoff）

- 当前切片 / Active slice: 布局合同 + 奖励合同 + 双任务注册 + 2+2+1 开训。
- 建议下一 skill / Suggested next skill: plan
- 计划提示 / Planning notes: 先纯 Python 合同再接 Isaac 地形；先开训再写 strict evaluator。
- 建议里程碑 / Suggested milestones: M0 合同与任务；M1 开训；M2 中期回放；M3 行为 evaluator。
- 里程碑验收提示 / Per-milestone acceptance hints: M0 pytest；M1 tmux 健康；M2 目视抬腿/落脚；M3 JSON + 回放。

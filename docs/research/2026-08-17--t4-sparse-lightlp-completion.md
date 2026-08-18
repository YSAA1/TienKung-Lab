# Research: LightLP 对照下的 T4 梅花桩完整补全方案

> **非权威。** 文中「先软后硬」已撤回。现行执行：`docs/plans/2026-08-17--t4-sparse-lightlp-complete-plan.md`。
> Date: 2026-08-17
> Status: 历史对照；执行面已落到 plan
> Question: 踏石、圆桩现在都不算会；上一版「只给圆桩填坑」太保守。对照 Light-Loco-Parkour §IV 与当前代码，完整该补什么？
> 主源: `docs/research/2608.02653v1/auto/2608.02653v1.md`
> 取代建议: `docs/research/2026-08-16--t4-precise-foothold-literature.md` 里「先别做两阶段 / 先关步态」的顺序
> 现行列: `docs/plans/2026-08-17--t4-sparse-lightlp-complete-plan.md`
> 2026-08-17 用户裁定: 从零、不短 FT、收窄踏石、sparse Actor 1155D 可打破。下文「续训 / 保 1155」作废。

## 1. 一句话

LightLP 的**感知走跑老师**（§IV，不是翻箱/攀爬那半本）已经是一份完整梅花桩配方。我们只搬了 Table I 的 `slack` + `illegal`，其余缺件还在，又用真坑把探索掐死。

完整方案不是再叠一条奖励，也不是另开 skill。是**一次新 lineage 把 LightLP §IV 缺件和 BeamDojo 软地形打齐**，同一条 1155D 老师上先软后硬，再收几何，最后才蒸带 GRU 的学生。

S1d 失败不能拿来否这份配方：S1d 是在**真坑 + 双 Critic**上关步态。这里是**软地形 + LightLP 原表，不装双 Critic**。

## 2. 当前水位（续训 `t_compat_sparse_v2_resume`，约 iter 24670）

| 地形 | reach_2m | 严格过关 | 摔倒 | 前进 |
|---|---:|---:|---:|---:|
| 平地 | 0.76 | 0.66 | 0.003 | 6.9 m |
| 踏石 | 0.80 | 0.45 | 0.24（hard 0.50） | 6.0 m |
| 圆桩 | 0.004 | 0 | 0.30 | 0.95 m |
| 上楼梯 30 | 0.05 | 0.002 | 0.16 | 1.2 m |

`Episode_Reward/illegal_footstep ≈ -0.001`，没有在教瞄准。  
踏石 = 大方砖上能蹦、不准、hard 一半摔。圆桩 = 第一跨没有正样本。都不是「会了」。

## 3. LightLP §IV 老师实际有什么

感知走跑是**一个混合课表老师**（Fig. 4：坡、Perlin、平地、栏、圆桩、踏石、箱子、楼梯、窄板），不是独立梅花桩 skill。翻箱/攀爬是另一套 mimic 老师，跟本切片无关。

### 3.1 观测（§IV-A/B）

| 项 | LightLP | T4 现在 | 差在哪 |
|---|---|---|---|
| Actor 高程 | 无噪声 height scan | 前向 195D @0.1 m，history=1 | 他们把最近 5 帧整段观测叠进去；我们本体 10 帧，scan 只有当前帧 |
| Actor 接触 | **特权接触标志** | T-compat 无；T-paper 有，真坑上已否 | 1155D 冻结，主线不进 Actor |
| Critic | 干净输入 + **脚下 height scan** + 冲击免疫 flag | 本体史 + 线速度 + 接触 + 同一张前向 scan | **脚下 30D 没进 Critic** |
| 网络 | 全 MLP，无 GRU | 全 MLP | 老师侧对齐；GRU 是学生的 |
| 学生 | 深度 + GRU；无 GRU 时踏石 **0%**（Table V） | 现有学生无梅花桩记忆设计 | 蒸馏阶段必须上 GRU + 末段 FT |

### 3.2 奖励（Table I）——感知走跑表里**没有**周期步态，也**没有** AMP

| 项 | 权重 | T4 sparse 现在 |
|---|---:|---|
| 线速度跟踪 | +2.0 | 有 +2.0 |
| 角速度跟踪 | +2.0 | 有，但是 +1.0 |
| 直立 | +1.0 | 有，但是 L2 罚 −2.0，形式不同 |
| velocity-slack | +1.5 | 有 +1.5 |
| 非法接触 | −2.0 | 有，Trunk/臂 −1.0、小腿 −0.3 |
| 关节限位 | −10.0 | 有，但是 −2.0 |
| illegal-footstep | −1.0 | 有 −1.0，均值却是 −0.001 |
| heading error | −1.0 | **无独立项**（只有 heading 命令） |
| opposite direction | −1.0 | **无** |
| action rate | −0.1 | 有 −0.01 |
| foot acceleration（EMA，τ=0.06 s） | −0.01 | **无**。现有 `foot_touchdown_impact −0.08` 只在触地瞬间看下落速度，没有落地前减速记忆 |

LightLP 明文：只靠指数跟踪，策略会停在踏石/平衡木前；slack 是为了跨出去。illegal 用足底向下网格，δ=0.1 m，和我们公式相同。  
**Table I 没有 `gait_*`，没有 AMP，没有 `feet_stumble`，没有 `legal_foothold`，没有双 Critic。**

AMP 出现在 §V-C 过渡组，用来在走跑和翻箱技能之间切先验，**不是**感知走跑老师的落足项。

### 3.3 终止（§IV-C2）

- 超时、出界：当 timeout，bootstrap，不记失败。
- 躯干接触 >1 N：reset。
- 根加速度 >40 m/s²：reset。
- 倾斜 >63°：**每步 1% 概率**才终止，留大约 100 步自救。
- **10% 环境**对躯干接触和冲击免疫，每 200 步重采样；免疫 flag 给 Critic。
- 没有「掉坑立刻 −200」这种项。

### 3.4 课表（§IV-C3）

- 10 行难度 × 32 列地形。稀疏相关至少是圆桩 + 踏石 + 窄板（Fig. 4 第 5/6/9 列）。
- 晋级：沿命令走出半格 **且跟踪好**；用路径长度，不用径向位移。
- 10% reset 随机难度，避免低难度被抽空。
- 几何厘米数**论文没写**。不能把我们的 48 cm 方砖说成 LightLP 原配。

### 3.5 Table V 能引用的数

- Teacher 踏石 99.9%，带高度差的石头 98.6%。
- 学生无 FT：踏石 34.6%；再去掉 GRU：**0**。
- BeamDojo 在他们表里踏石 91.7%、平衡木 94.3%。
- 这些是**学生深度**消融，不能拿来要求现在的老师加 GRU。

## 4. 当前代码相对 LightLP 的缺口

`t4_loco_teacher_sparse` = Stage E 老师 + 40% 稀疏 + slack/illegal/碰杆。

多出来、会和落足打架的：

- `fixed_clock` 0.85 s 周期步态，三项合计 **+2.6**，只被速度跟踪门控，**不按地形关**。
- Actor 本体里还有 `gait_phase_sin/cos`、`gait_air_ratio`，时钟一直在。
- 平地 AMP，`decay_start_difficulty=0.3`：easy 稀疏行 AMP 仍是满权。专家全是平地走。
- `feet_stumble −2.0`：水平力 > 5×垂向力就罚。圆盘帽檐正好是这种擦边。
- `termination_penalty −200` + 掉坑 0.5 m 立刻 reset。
- 踏石 easy 48 cm 方砖，脚约 21×8 cm，偏十几厘米仍整脚在砖上 → illegal 沉默。

缺了、论文里有的：

- Critic 脚下 scan（射线已经打了，只进了 reward）。
- 滤波足加速度（落地前减速）。
- heading / 反向走。
- 10% 冲击免疫 + 倾斜随机终止。
- 软地形（这是 BeamDojo 的，LightLP 没写；没有它，圆桩到不了「illegal 能干活」的那一步）。

已经证伪、不要加回：

- Actor 接触（T-paper，真坑上圆桩仍 0）。
- `legal_foothold`（几乎从不发火）。
- 双 Critic（S1d 对照组同迭代 reward 更差，圆桩仍 0）。
- 改 1155D、独立梅花桩 skill、规划器。

## 5. 完整补全方案

一条 1155D 老师，三个阶段。阶段 A 把缺件打成**一个新配方**，不再 5k 拧一颗螺丝。归因靠「软地形 + LightLP 原表」整包对比当前续训，不靠包内再拆。

```text
当前续训（不停，只当骨干对照）
    │
    ▼
阶段 A  软稀疏 + LightLP §IV 缺件（新 lineage，可加载现 Actor）
    │   踏石/圆桩填缝，scan 仍是真图
    │   稀疏上关掉步态/AMP/stumble
    │   Critic 脚下 scan；补足加速度 / heading / 反向
    │   掉坑不终止，illegal 仍算
    │
    ▼
阶段 B  同一权重开真坑
    │   恢复 pit_fall；illegal 应明显离开 0
    │
    ▼
阶段 C  收几何
    │   方砖收窄，圆盘保持圆
    │   这时才叫梅花桩，而不是大方砖蹦过去
    │
    ▼
阶段 D  蒸 depth 学生（GRU + 末段 FT）
        LightLP Table V：无 GRU 踏石 = 0
```

### 阶段 A — 一次打齐（新 lineage）

**地形 / 探索**

- 只改 `stepping_stones` 和 `raised_pillars`：物理缝填实（或与支撑面同高的实心底），**前向 scan 和足底射线仍打在「未填」的真图上**。人走平地，眼看桩。
- 这两种地上关闭 `pit_fall` 终止；`illegal_footstep` 继续按真图算。踩偏扣分，回合不结束。
- 楼梯、跨栏、箱子保持真几何。
- 稀疏占比保持 40%，或提到约 50%（对齐 Fig. 4 里稀疏相关列的份量）。不要 100% 摘出去。
- 可选、建议一起做：10% 环境对躯干冲击免疫，flag 只进 Critic（§IV-C2）。

**奖励 / 先验（只在稀疏 tile 上）**

- 保留 `velocity_slack +1.5`、`illegal_footstep −1.0`。
- `gait_feet_*` 三项 ×0。连续地形仍用现在的钟。
- AMP 系数 ×0。连续地形仍按现在的 difficulty 衰减。
- `feet_stumble` ×0。
- 不加 `legal_foothold`，不装双 Critic。
- 新增 LightLP `foot acceleration`：足加速度超 30 m/s² 的部分做 τ=0.06 s 的 EMA，权重 −0.01。现有 `foot_touchdown_impact` 可留可并，不要两套各打各的。
- 新增 `opposite_direction −1.0`。`heading error` 若命令里的 heading 跟踪已经在干活，不必再叠一条同义项。

**观测（不改 Actor 1155D）**

- 左右足底 scan 拼进 **Critic**（约 15+15=30 D）。Actor 不看脚下。
- Critic 因此变宽：Actor 权从当前续训加载，**Critic 重开**。不要从 S1d 双头 ckpt 迁。
- 不加 Actor 接触。T-paper 若要再试，必须在阶段 B 之后单独开，不挡主线。

**终止**

- 稀疏软阶段：不掉坑、不因掉坑吃 −200。姿态摔仍可留，但不要在软阶段把探索抽干。
- 连续地形：保持现在的终止。

**从哪起步**

- 新 run：`--resume` 当前 `t_compat_sparse_v2_resume` 的 Actor；Critic 因 30D 对不上，随机初始化。
- 现续训不停，当对照，不宣称能力。

### 阶段 B — 开真坑

- 触发：软阶段上踏石、圆桩的 `reach_2m` 都稳定离开 0，且 `illegal` 均值明显负于现在的 −0.001（脚开始「挨边」）。
- 同一权重：去掉填缝，恢复 `pit_fall`。
- 稀疏上的步态/AMP/stumble 仍然关。这时才是「瞄点 + 真的会掉」。

### 阶段 C — 收几何

- 触发：真坑上两边都能走出一段，不是只到 2 m。
- 踏石从 (0.48, 0.32) 往更窄收；圆桩保持圆盘。illegal 这时必须能咬到偏脚。
- 现在就收 = 把踏石变成第二个圆桩，阶段 A 还没过会一起死。

### 阶段 D — 学生

- 老师阶段 C 过了再蒸。
- 深度学生加 GRU，蒸馏后短 FT。LightLP：无 GRU 踏石学生是 0，不是「老师再加记忆」。

## 6. 明确不做

- 独立梅花桩 skill / 新 AMP clips / 规划器。
- 改冻结 1155D，或把脚下 scan / 接触写进导出学生。
- 双 Critic、`legal_foothold`、再给 Actor 加接触当主线。
- 把阶段 A 拆回「先只填坑、5k 后再关步态」。那是上一版保守切法；软地形上这些项是同一套 LightLP 配方。
- 打断 zhuoqun 翻箱。
- 用 reward、`terrain_levels`、ckpt 存在宣称梅花桩能力。

## 7. 怎么算够（仍要回放，不靠单条 TB）

- **A：** 踏石和圆桩 `reach_2m` 都离开 0；圆桩 progress 离开 ~1 m；`illegal` 不再是 −0.001。
- **B：** 真坑上两边都能连续走，hard 摔倒明显低于现在踏石的 0.50。
- **C：** 收窄后回放脚落在面上，不是擦边碰运气。
- **D：** 学生固定课能过，才谈部署。

## 8. 和旧结论怎么相处

| 旧结论 | 现在 |
|---|---|
| 2026-08-16 研究：先别两阶段，先关步态 | 顺序反了。关步态必须放在软地形上；真坑上关 = S1d |
| 回退计划阶段 4：只做圆桩软地形 | 太窄。踏石也不算会；软地形覆盖两种稀疏地，并一次补齐 §IV 缺件 |
| S1d 证明双 Critic / 关步态无效 | 只证明**真坑厨房水槽**无效。双 Critic 仍不加；关步态改到软地形上做 |

## 9. 未核

- LightLP 踏石/圆桩厘米几何仍未写。阶段 C 的目标宽度要另对 Isaac 演示和脚尺寸，不能冒充论文原配。
- 足底 0.04 m 网格对收窄后圆盘的命中率未在 Isaac 里实测。
- 软地形「填缝但 scan 打真图」的实现落点（heightfield 填平 vs mesh 实心）还没选型，属于 plan 阶段。

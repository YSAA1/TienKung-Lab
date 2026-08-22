# Research: 精准落足文献对照 T4 当前 MDP

> Date: 2026-08-16
> Status: findings（非权威，不替代 plan / Spec）
> Question: 踏石能学、圆桩学不会，对照精准落足主源，当前 MDP 还缺什么？要不要再加项？
> 非权威研究笔记。现行梅花桩执行面：`docs/plans/2026-08-22--t4-sparse-s12-rim-yaw-student-plan.md`。

## 1. Verdict

**现在不该再往 Actor 里塞感知，也不该上落足规划器 / 双 Critic / 两阶段软地形。**
v3 已经对准了文献里「先让第一跨可迈、再用足底射线塑造落点」这条最小路径。
圆桩若 5k 仍 `reach_2m=0`，下一刀应是**关掉和落足打架的平地步态/AMP/stumble**，而不是再加一套奖励。

一句话对照：

| 文献共识 | T4 现在 | 要不要加 |
|---|---|---|
| 稀疏落足 from-scratch 会停在障碍前 | 旧 25k / 圆桩 v2 符合 | 已有 slack |
| 方砖能原谅误差，圆盘/窄梁不能 | v2 踏石 `reach_2m≈0.79`，圆桩≈0 | v3 只改圆桩几何 |
| 足底向下射线评多边形脚，不上实机 | 已有 illegal；v3 加 legal | 先观察，不再叠项 |
| Critic 可以看脚下，Actor 不必 | 足底 scan 只进 reward | 5k 仍挂再考虑进 Critic |
| 周期步态 / 平地 AMP 和精确落足冲突 | `fixed_clock` 步态全开；easy AMP 不衰减 | **最可疑的剩余项** |
| 踩空即终止会抽干探索 | 掉坑 / 姿态立刻 reset | 先别做两阶段 |
| 规划器 + CLF 比纯 RL 稳 | 未做 | 过重，违背现 Spec |
| 学生要记忆（GRU） | teacher scan history=1 | 学生阶段再做 |

## 2. 主源

| 源 | 打开了什么 | 用来证什么 |
|---|---|---|
| LightLP 本地 `docs/research/2608.02653v1/auto/2608.02653v1.md` §IV-B/C、Table I/V | 原文 | teacher 观测、slack、illegal、足底 scan、停住、学生 GRU |
| [BeamDojo arXiv:2502.10363](https://arxiv.org/abs/2502.10363) / [项目页](https://why618188.github.io/beamdojo/) | HTML + 式 (2)–(6) + Table II / App. VI-B | Naive 0.33%；两阶段 42%→92%；easy 0 间隙 |
| [Walk the PLANC arXiv:2601.06286](https://arxiv.org/html/2601.06286v1) | HTML | 纯 RL 踏石 56% vs 规划器引导 100%（4096 env） |
| Duan et al. Cassie 踏石 [NSF PAR](https://par.nsf.gov/servlets/purl/10396740) | 摘要 | 先学「跟落足命令」，再外挂视觉选石 |
| Ames/Nguyen CBF 踏石 [CDC 2016](https://hybrid-robotics.berkeley.edu/publications/CDC2016_3DWalking_SteppingStones.pdf) | PDF | 经典方法把落足当硬约束，不是 reward shaping |
| Isaac Lab `hf_terrains.py` + `check_height_field_subterrains.py` | 本机源码 | 官方演示间隙 0.05–0.10 m |
| 本仓 `teacher_cfg.py` / `t4_env.py` / `stepping_stone_layout.py` / `rewards.py` | 源码 | 当前 MDP 事实 |
| B 站 `BV1bAAtexEoB`（张伟楠SJTU《BeamDojo：人形机器人走梅花桩啦！》） | agent-reach bili-cli | 中文社区把 BeamDojo 直接叫梅花桩 |

Twitter 本次 404；Reddit 关键词检索未落到相关帖。未把二手综述当证据。
`lucidrains/light-loco-parkour` 是非官方重写，**不当** LightLP 实现真值。

LightLP **没有**写出踏石边长/间隙厘米数。几何数字来自 Isaac 官方演示和本仓脚/scan 尺寸，不是论文原文。

## 3. 文献实际写了什么

### 3.1 LightLP（最直接的对照）

观测（§IV-B）：

- Actor：无噪声 height scan + 轻度 DR 的本体；稀疏地形**另外给 privileged contact flag**。
- Critic：干净输入 + **脚下 height scan**。
- 足底 scan 画在 Fig. 3，用于算 illegal，不是部署观测。

奖励（Table I，感知 loco teacher）：

| 项 | 权重 |
|---|---|
| 线速度跟踪 | +2.0 |
| 角速度跟踪 | +2.0 |
| 直立 | +1.0 |
| velocity-slack `[0.3, 1.5]×cmd` | +1.5 |
| 非法接触 | −2.0 |
| 关节限位 | −10.0 |
| illegal-footstep（δ=0.1 m） | −1.0 |
| heading / 反向走 | 各 −1.0 |
| action rate | −0.1 |
| 滤波足加速度（落地预减速） | −0.01 |

**Table I 没有周期步态项。** 停住问题写在 §IV-C1：只有指数跟踪时，策略会停在踏石/平衡木前，永远不跨。slack 是为了这个，不是为了更准。

Table V：踏石学生 99.9%；去掉最后 FT 剩 34.6%；再去掉 GRU 为 **0**。有高度变化的石头 95.6 / 24.4 / 0。
这是**学生深度观测**的消融，不能用来要求 teacher 现在加 GRU。
BeamDojo 在他们表里踏石 91.7%、平衡木 94.3%。

### 3.2 BeamDojo（专门打梅花桩/稀疏落足）

问题陈述（原文）：人形脚是**多边形**，四足那种点脚奖励不适用；落足奖励稀疏；踩空早停导致探索崩掉。

做法：

1. 脚底采 n 点，接触时统计有多少点低于阈值 ε，作为 `r_foothold`（式 2）。语义上就是 LightLP illegal 的多边形版。
2. **双 Critic**：密步态奖励和稀疏落足奖励分开估价值，再加权合成优势。
3. **两阶段**：Stage 1 人在平地上走，但看的是真实踏石高程图，踩空只罚不终止；Stage 2 再到真坑上微调。硬石成功率 **42% → 92%**。
4. 消融：只加落足奖励、单阶段单 Critic 的 Naive，硬石成功率 **0.33%**（Table II）。说明「多一条落足项」本身几乎不够。
5. App. VI-B 几何：Stones Everywhere 的 ℓ=0 是 **1.5 m 石 / 0 间隙**；走廊踏石从 **0.8 m / 0.10 m** 起。真机硬石约 20 cm（G1 身高 1.32 m）。

B 站官方视频标题直接叫「走梅花桩」。

ALLSTEPS / Cassie：前两块石头就放在脚下，前三块免费，避免「第一跨就是跳」。Extreme Parkour 把脚踩在边 5 cm 内当罚（比 LightLP 的 10 cm 掉深更狠）。LightLP Fig. 4(5)/8 的圆桩在图上是密铺圆柱，不是孤岛。

### 3.3 PLANC / Cassie / CBF

- PLANC：纯 RL 在踏石上 56%，加 LIP 落足规划 + CLF 奖励到 100%（4096 env）。作者结论是「只靠 reward shaping 发现精确落足很慢」。
- Cassie：先训「跟踪落足命令」的底层，再外挂相机选石。把「踩得稳」和「选哪块石」拆开。
- CBF 2016：落足是冲击瞬间的硬约束，不是稠密奖励。

这些说明规划器有效，但和本仓「一个 Stage E teacher、不改 1155D、不加规划器」的 Spec 冲突。

### 3.4 Isaac 官方踏石

演示参数：石宽 0.25–1.575 m，**间隙 0.05–0.10 m**，石高最大 0.2 m。
难度↑则石变窄、缝变宽。easy 几乎是「能迈过去的缝」，不是 30 cm 以上的第一跨。

## 4. T4 当前 MDP（代码事实）

观测：

- T-compat Actor 1155D = 本体史 + 前向 scan（0.1 m，前方 0.2–1.6 m，**history=1**）。
- T-paper 再加 2 维接触；v2 @2.1k 圆桩 `reach_2m≈0`，这条已经否掉。
- Critic：本体 + 线速度 + 接触 + 同一张前向 scan。**脚下 scan 不算进 Critic。**
- 足底 RayCaster 0.04 m、0.16×0.08 m，只服务 reward。

奖励（稀疏老师 = 平地老师 + 三项）：

- 继承：跟踪 2.0、周期步态 1.0+1.0+0.6、`feet_stumble −2.0`、滑移、AMP。
- 新增：slack +1.5、illegal −1.0、legal_foothold +1.5（抬高支撑 × slack）、碰杆 −2.0。
- `gait.mode = fixed_clock`：步态只被速度跟踪门控，**不按地形关掉**。
- AMP 从 difficulty 0.3 才开始衰减；easy 行（≤0.22）AMP 仍是满权重，专家全是平地走。

几何（v2 训练中的踏石 / v3 探针的圆桩）：

- 踏石第一跨 easy 32 cm、d=0.5 为 46 cm；方顶 48→32 cm。
- v2 圆桩第一跨 35–60 cm，直径 42→32 cm。
- v3 圆桩间距 (0.55, 0.58)，直径 (0.50, 0.38)，easy 第一跨约 5 cm。

终止：掉坑 0.5 m、|pitch|>1.0、|roll|>0.8，立刻 −200。没有 BeamDojo 那种「踩空只罚不停」。

## 5. 对照后还存在的问题

按证据强度：

1. **圆桩几何（已在改）**  
   文献 easy 间隙 5–10 cm；v2 圆桩第一跨 35 cm + 圆盘。踏石能过是因为 48 cm 方砖原谅偏脚，不是已经学会瞄准。v3 只动圆桩，方向对。

2. **周期步态和 AMP 在抢落足（高嫌疑，还没改）**  
   LightLP 感知 loco 表里没有周期步态。BeamDojo 专门把步态密奖励和落足稀疏奖励拆开，否则后者被淹没。
   T4 easy 稀疏上：步态三项合计 +2.6，AMP 满权，而 illegal 在日志里只有 −1e-4。
   圆桩上要拉长步、踩偏、单脚晃，和 0.85 s 对称时钟 + 平地 AMP 是反的。

3. **`feet_stumble −2.0` 会打圆盘边缘**  
   实现是水平力 > 5×|垂向力| 就罚。圆盘帽檐正是这种擦边接触。LightLP 没有这项。踏石直边较少触发，圆桩很容易。

4. **Critic 看不见脚下（中等）**  
   LightLP 明文给 Critic 脚下 scan。T4 已经打了射线，只是没拼进 Critic。不改 1155D。
   不是现在的第一刀：几何不对时，Critic 多几维也学不会。

5. **踩空即死（中等，圆桩更痛）**  
   BeamDojo 认为这是探索崩掉的主因。T4 圆桩 fall≈0.65、progress≈1 m，符合「第一脚死后没有正样本」。
   v3 把第一跨收到 5 cm 后，正样本应自己出现。先看 5k，不要先上两阶段。

6. **不必现在加的**  
   - Actor 接触：T-paper 已否。  
   - 落足规划器 / CLF：PLANC 有效，但改 teacher 架构，违 Spec。  
   - 双 Critic：先把步态在稀疏上衰减，比改 PPO 轻。  
   - Teacher GRU / scan history：Table V 的 0% 是学生深度，不是 teacher scan。  
   - 再叠一层「脚心对准圆心」：legal 已覆盖「抬高支撑上的站立」；圆心项在第一跨还迈不出时是 0。
   - 再叠同类正奖励：`legal_foothold` 不在 LightLP Table I / BeamDojo Table VII 里。BeamDojo Naive=0.33% 说明只加落足项几乎不够。

## 6. 建议顺序（先看门，再改）

```text
现在（v2 踏石 + v3 圆桩）
  -> 5k：圆桩 reach_2m 是否离开 0，legal_foothold 是否抬头
  -> 若起来：不要再加项；10k 固定 evaluator
  -> 若仍为 0：只做一件 —
       稀疏 tile 上衰减周期步态和 AMP，并减弱 feet_stumble
  -> 还不行：Critic 拼脚下 scan（仍不进 Actor）
  -> 再不行：圆桩-only 的「踩空只罚不终止」短探针
不要：Actor 接触、规划器、双 Critic、改 1155D
```

## 7. 未核缺口

1. LightLP 踏石/桩的厘米几何未写。
2. BeamDojo 附录完整奖励表未逐项打开。
3. PLANC 56%/100% 的石头尺寸、是否圆桩，HTML 中未逐条核对。
4. 未在 Isaac 里实测 0.1 m scan 对 5 cm 圆桩缝的命中率。
5. Twitter 本次不可用；Reddit 检索未命中相关讨论。

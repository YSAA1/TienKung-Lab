# Research: 当前 sparse MDP 对真机踏石/圆桩够不够泛化

> Date: 2026-08-22
> Status: findings（非权威，不替代 plan / Spec）
> Question: 当前训练 MDP（踏石+圆桩约 40%、规则格子、混合楼梯）训出的策略，对真实部署、尤其是塔石和圆桩，是否够泛化？
> 主源: 本仓 `stepping_stone_layout.py` / `teacher_cfg.py` / `t4_env.py`；交接 `t4-s12-sim2sim-and-recovery-handoff-2026-08-22.md`；LightLP `docs/research/2608.02653v1`；BeamDojo arXiv:2502.10363；ALLSTEPS arXiv:2005.04323；Walk the PLANC arXiv:2601.06286；capturability (Koolen et al.)
> 现行执行面: `docs/plans/2026-08-22--t4-sparse-s12-rim-yaw-student-plan.md`
> 不中断 S12；MuJoCo 摔倒不能当 Isaac 能力判决。

## 1. Verdict

**不够。当前 MDP 不能支撑「踏石/圆桩已可上真机」的泛化声明。**

这不是因为课表只有 40% 稀疏、也不是因为楼梯「只会往上走」。对真机梅花桩，当前配方缺的是三件不能互相替代的东西：

1. **可部署观测还没训出来。** 真机吃深度+本体+命令，不吃 HeightScan。现在只有特权老师；sparse GRU 学生还没开训。LightLP Table V：同一任务上学生去掉 GRU 是 0%，去掉末段 FT 剩 34.6%。老师会走 ≠ 能上机。
2. **规则格子上的「对准走」≠ 偏脚后还能救。** 真机第一脚几乎总会偏。当前 easy 第一跨约 0 缝、hard 才 15–17 cm；`vy=0` + 速度 slack 奖励继续往前冲；真洞上踩偏就容易被 accel/torso 截断。策略学到的是「别进坏状态」，不是「进了坏状态下一脚收回支撑」。
3. **Isaac 上老师自己也还不是硬能力。** S11b `model_19000` 在 d=0.8 固定直行是 26/32 strict、6/32 早停。S10 hard d=1 踏石只有个位数 strict。MuJoCo 第一步不稳就倒，同时混了接触失配和这个覆盖缺口，不能单独当真机判决。

把稀疏占比调到 50%、把楼梯改成更多下楼、或把格子再「随机一点」，都不会单独补上这三件。

## 2. 当前 MDP 事实（代码）

`T4_SPARSE_TERRAIN_PROPORTIONS`（和为 1.0）：

| 桶 | 占比 |
| --- | ---: |
| 踏石 + 圆桩 | 0.20 + 0.20 = **0.40**（不是 50%） |
| 上楼梯 30/34 | 0.11 |
| 下楼梯 30/34 | 0.11 |
| 上下坡 | 0.08 |
| 跨栏 | 0.08 |
| 平地/rough/box/wave | 0.22 |

楼梯不是「只会前向上楼」：`stairs_up_*` 是 inverted pyramid（出生在底部往上走），`stairs_down_*` 是 pyramid（出生在顶部往下走），两边都有。

几何是 **难度插值的规则方格**，没有单块 XY 抖动；踏石高度抖动 0–4 cm。第一跨：

| d | 踏石顶 / 缝 / 高 / 第一跨 | 圆桩直径 / 缝 / 高 / 第一跨 |
| --- | --- | --- |
| 0.00 | 40 / 10 / 9 / **0 cm** | 50 / 5 / 8 / **5 cm** |
| 0.39 | 34.5 / 17 / 15 / 6 cm | 45 / 11 / 16 / 10 cm |
| 0.80 | 29 / 24 / 21 / 12 cm | 40 / 17 / 24 / 15 cm |
| 1.00 | 26 / 28 / 24 / **15 cm** | 38 / 20 / 28 / **17 cm** |

合同测试写明：easy 40 cm 方顶原谅 6 cm 偏脚；hard 窄顶同样 6 cm 会触发 illegal。圆盘更不原谅偏脚。

稀疏命令：`vx∈[0.6,2.0]`、`vy=0`、60% `wz=0`、40% `[-0.3,0.3]`。出生仍继承 Stage E：`xy±0.5 m`、`yaw±π`、六维速度 ±0.5。AMP 和周期步态在稀疏砖上乘 0。`body_orientation_l2` 在稀疏奖励里权重是 0，只留 LightLP `upright=+1`。

## 3. 真机泛化要过的三层，现在过了哪层

| 层 | 要什么 | 现在 |
| --- | --- | --- |
| Isaac 老师 | 规则格子上能走完、偏脚/轻转还能活 | 中等难度有实质能力；hard 和边沿恢复未过门 |
| 深度学生 | 同一 MDP 上 GRU + 干净深度蒸馏 + 噪声 FT | 代码在，开训等 S12 老师 10k 门 |
| 真机几何/接触 | 不规则石、摩擦、延迟、第一步必偏 | 未开始。MuJoCo 只是接触失配探针 |

LightLP 感知走跑老师也是混合课表（坡、Perlin、平地、栏、圆桩、踏石、箱子、楼梯、窄板），所以「和楼梯混训」本身不是错，也不是真机失败的主因。他们真机踏石靠的是 **深度学生 + GRU + 末段 FT**，不是老师 HeightScan。

## 4. 文献怎么说「第一步 / 偏脚 / 真机」

- **Capturability**（Koolen / Pratt）：落地后瞬时捕获点必须落在下一脚能踩到的支撑上。支撑越小，捕获域越小。硬踏石 26 cm、硬圆盘 38 cm，第一步偏 4–8 cm 就可能从 1-step capturable 掉到「必须立刻侧向改落点」，否则下一脚走进洞。
- **ALLSTEPS**：一上来均匀抽难题，策略会学会 **站在第一块上不动**。有效配方是先 easy 再难，不是加地形种类。
- **BeamDojo**（G1 真机梅花桩）：Naive（只加落足项、单阶段真洞早停）hard 踏石成功率 **0.33%**。两阶段「平地走、看真洞图、踩空只罚不停」才把 hard 拉到 ~92%。真机单独测了 **misstep recovery**（脚下没扫到、第一步踩偏仍能救）。他们 Stage 1 用 Stones Everywhere（散石），Stage 2 才收成走廊。课表占比不是他们的旋钮；探索会不会被第一步摔死才是。
- **Walk the PLANC**：纯 RL 踏石 56%，加落足规划+CLF 到 100%。作者结论：只靠 reward shaping 发现精确落足很慢。
- **Extreme Parkour**：脚踩在边 5 cm 内要罚，否则仿真会走边、真机不稳。T4 illegal 是 10 cm 掉深，比这个松。
- **B 站中文场景**（张伟楠 SJTU `BV1bAAtexEoB` 等）：社区把 BeamDojo 直接叫梅花桩，强调正走/倒走和踩偏恢复，不是「课表再加 10% 石头」。

## 5. 用户容易混在一起的两件事

「课表不够杂」和「第一步不稳就倒」不是同一个缺口。

- **课表杂**（楼梯上下、rough、跨栏、40% 稀疏）教的是：同一套走跑先验能切到不同连续地形。对踏石/圆桩的 **离散支撑恢复** 帮助很小。楼梯是连续台阶，踩偏还能踩在下一级；梅花桩踩偏就是真空。
- **格子固定** 会让策略背「每 50–54 cm 一格」的开环步长。真机石不会对齐这个格子。这伤害的是 **布局泛化**，不是第一步动力学本身。要补布局泛化，需要石心 XY 抖动 / Stones Everywhere，而不是把 40% 改成 50%。
- **MuJoCo 第一步就倒** 同时有：PhysX 盒接触 vs MuJoCo 伺服接触（交接的工作假设），以及 7 车道走廊比训练 8 m 方格更窄（`play_t4_sparse_teacher_mujoco.py` 已写明为了容纳老师的 sim2sim 侧漂）。不能用它证明「课表不够杂所以真机不行」，但可以当「第一步接触一旦偏、策略不会救」的观感。

## 6. 对踏石 vs 圆桩

圆桩比踏石更接近真机失败模式：同样 6 cm 偏脚，方顶 easy 能原谅，圆盘 hard 不能。文献（BeamDojo、先前仓内笔记）一致：方砖先会走，圆盘/窄梁才暴露瞄准和恢复。当前 easy 圆桩第一跨只有 5 cm，hard 才把「真正的第一跨」打开——所以训练分布里 **稳定的偏脚-再救** 正样本仍然稀。

## 7. 下一步（不改 S12）

跟交接文档同一条决策链，不要用本笔记开新 lineage。

2026-08-22 已跑完最小 4 桶（S11b `model_19000`，d=0.8，`vx=0.8`，yaw=0，32 局，出生钉在台上）：

| 地形 | 侧偏 | reach_4m | legal first | 早停 |
| --- | ---: | ---: | ---: | ---: |
| 踏石 | 0 cm | 32/32 | 1.00 | 0 |
| 踏石 | 8 cm | 32/32 | 1.00 | 0 |
| 圆桩 | 0 cm | 28/32 | 1.00 | 4（accel 3 + torso 1） |
| 圆桩 | 8 cm | 32/32 | 1.00 | 0 |

JSON：`artifacts/eval/s12_firststep_pin/`。8 cm 没有比 0 cm 更差。Isaac 老师在 d=0.8 第一步侧偏不是主洞；MuJoCo 第一步就倒按接触失配查。不因此改课表。S12 仍走原计划 10k/23999 固定评估门后再蒸学生。

未测：d=1.0、偏航 8°、S12 `model_23999`。不是本切片。

## 8. Sources

- `legged_lab/terrains/stepping_stone_layout.py`（占比、第一跨、illegal 6 cm）
- `legged_lab/envs/t4/teacher_cfg.py`（稀疏命令、DR、`body_orientation_l2=0`）
- `legged_lab/envs/t4/t4_env.py`（AMP/gait 稀疏置零、terrain-aware 命令）
- 交接：Isaac S11b 26/32 @ d=0.8；MuJoCo 不作能力
- LightLP auto-md §IV / Table V
- BeamDojo HTML v3 Table II / Fig. 9 misstep recovery
- ALLSTEPS：均匀难题 → 站在第一块
- Pratt/Koolen capturability；Walk the PLANC 56% vs 100%

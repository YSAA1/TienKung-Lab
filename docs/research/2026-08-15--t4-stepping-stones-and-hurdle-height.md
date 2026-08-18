# Research: T4 梅花桩与跨栏加高

> **非权威。** 现行工作面见 `docs/README.md`（梅花桩 + 翻箱两条，不是「唯一翻箱」）。
> Date: 2026-08-15
> Question: 本项目里「梅花桩」最可能指什么？当前跨栏到底训到哪、验到哪？梅花桩该怎么训才不违反合同？跨栏再加高该先诊断还是改 range？和工作面怎么排？
> Status: 历史 findings
> Paper: `docs/research/2608.02653v1/auto/2608.02653v1.md`（arXiv:2608.02653，Light-Loco-Parkour）

## 1. Verdict

**梅花桩不是 100m 障碍 #4 绕桩；它是稀疏落脚的脚上任务（LightLP 的 stepping stones / raised pillars），应按 2026-08-14 跨栏裁定并入 Stage E bucket，不要开独立 skill。跨栏「再加高」更可能是把 MuJoCo `loco` 回放从 0.22/0.28 拉到规则/Spec 的 0.30–0.35，而不是把课表推过 0.35。**

| 任务 | 它是什么 | 当前仓库距离 | 最小建议 |
|------|----------|--------------|----------|
| **A. 梅花桩 / stepping stones** | 在离散桩顶/石面上落脚，间隙是坑或不可踩地面；纯脚上 | Stage E **没有**该 bucket；`boxes` 是连续 0.45 m 网格、高差 0–15 cm，中间没有洞 | 新 Stage E terrain bucket + 新 lineage 从零；1155D 不动；足底 scan 只做 **reward**，不进 Actor |
| **B. 绕桩（100m #4）** | 5 根 φ≈3 cm、高≈1.5 m 木桩，**绕行不碰** | 未做 | **不是梅花桩**；以后单独做 corridor/slalom |
| **C. 跨栏加高** | 摆动腿 clearance，不是跳跃 skill | 训练 range 已到 0.35；人工回放是 `loco` 的 0.22/0.28；gate 0.30 的 H3/H4 **不存在** | 先改回放几何并做零碰栏诊断；不要先改 `T4_HURDLE_BAR_HEIGHT_RANGE` |

三条硬结论：

1. **100m 障碍 1–10 没有梅花桩。** #4 绕桩是绕行细杆；#7 S 弯桥才接近 LightLP 的 plank bridge。梅花桩对应的是 LightLP Fig. 4 第 5–6 列（raised pillars / stepping stones），以及 Isaac Lab 官方 `HfSteppingStonesTerrainCfg`。
2. **现有跨栏完成声明 ≠ 0.30 m 零碰栏能力。** 训练课表 0.05→0.35，H2 只是 progress diagnostic（83/100 @ difficulty 0.85 ≈ 0.305 m），人工复核走的是 `loco` 课 0.22/0.28。H3/H4 corridor + zero-contact 仍是未开工工作项。
3. **梅花桩从零训会停在稀疏落脚前**（LightLP §IV-C1）。现有 1155D 没有足底 scan、Actor 禁止 `contact_truth`。最小可行是新 bucket + 可选 illegal-footstep **奖励**（足底射线只进 MDP）；独立 skill、LightLP 全流水线、改 1155D 都过重。地形分布变更 = 新 lineage 从零。

## 2. Primary sources consulted

### 本仓库

| Source | What it establishes |
|--------|---------------------|
| `docs/research/100m-obstacle-rule-contract.md` 表 2 / 图 3 | 100m 十障碍顺序与几何；**无「梅花桩」条目**；#2 跨栏 0.3 m；#4 绕桩 φ≈0.03 m / 高≈1.5 m / 绕行不碰；#7 S 弯桥宽 0.5 m / 离地 0.1 m |
| `docs/specs/2026-08-13--t4-hurdle-skill.md` | 2026-08-14 裁定：纯脚上任务并入 Stage E bucket，不另开 skill；训 0.05→0.35；gate 0.30；报告项 0.35；地形分布变更 = 新 lineage；H3/H4 未交付 |
| `docs/archive/plans/2026-08-13--t4-hurdle-skill-plan.md` | 跨栏 surface 已归档为 done；H3/H4 checkbox 仍空；closure 明确不含 100m `rule` strict-contact |
| `legged_lab/terrains/hurdle_layout.py` | `T4_HURDLE_BAR_HEIGHT_RANGE = (0.05, 0.35)`；间距 0.9–1.3；杆厚 0.07 |
| `legged_lab/terrains/terrain_generator_cfg.py` | Stage E：`boxes` 0.45 m / 0–15 cm；`hurdles` 0.10；`MeshGap` 只在旧 `ROUGH` 里注释掉；**无 stepping-stones bucket** |
| `legged_lab/assets/t4/schemas.py` | Teacher 前向 195D scan：1.4×1.2 @0.1，offset x=0.9，clip ±1，invalid=1.0；Actor 禁 `contact_truth` / height map；teacher actor 1155D |
| `legged_lab/envs/t4/t4_env.py` | Actor = proprio 史 + 前向 scan；`feet_contact` 只进 Critic；姿态终止 \|pitch\|>1.0 或 \|roll\|>0.8 |
| `legged_lab/envs/t4/teacher_cfg.py` | `termination_penalty` −200；AMP 按 difficulty 线性衰减到 0.3；无 illegal-footstep、无足底 scan |
| `legged_lab/assets/t4/urdf/t4_std.urdf` / `mujoco_sim2sim.py` | 脚三 box 包络约 0.209 × 0.077 m |
| `legged_lab/assets/t4/mjcf/t4_std.xml` | Trunk `z=0.9`；`forward_camera` 相对 Trunk +0.42 → 约 1.32 m；`mid360_link` +0.535 → 约 1.435 m |
| `legged_lab/scripts/sim2sim_t4_depth_student.py` | `loco` 课两根杆 **0.22 / 0.28**；`rule` 课 10 栏半高 0.15 → **全高 0.30** |
| `legged_lab/scripts/eval_t4_hurdle.py` | `t4_hurdle_progress_v1`；`final_zero_contact_gate: False` |
| `.harness/state.md` / `.harness/work_index.md` | 唯一 active surface 是翻箱 G1/G2 recovery；走跑已关闭；H2 83/100 ≠ 零碰栏 |
| `docs/specs/2026-08-12--t4-unified-depth-locomotion.md` | 单一 policy；Actor 无 HeightScan/接触真值；防绕行必须 corridor + ordered gates |
| `docs/plans/2026-08-15--t4-vault-g1-g2-recovery-plan.md` | 当前唯一执行面；不改 1155D / Stage E plant |

### 邻仓 / 官方实现（本机可读）

| Source | What it establishes |
|--------|---------------------|
| `/home/ssy/桌面/ame_mjlab_height_scan-master/docs/plans/2026-08-11--t4-hiw-three-hiking-routes.md` | AME HIW 计划把 stepping stones 列为覆盖目标；子计划 foothold-geometry **文件不存在** |
| 同仓 `src/mjlab/tasks/velocity/config/t4_hiw_parkour/terrains.py` | **已实现的** mixed terrain 只有 flat / rough / stairs / slope / wave，**没有** stepping stones |
| 同仓 `InstinctMJ-main/.../hf_terrains.py::perlin_stepping_stones_terrain` | Isaac 式踏石 + 可选 Perlin；难度↑则石面变窄、间隙变宽 |
| 同仓 `InstinctMJ-main/.../parkour_env_cfg.py` | HIW parkour 混合用 `square_gaps`（间隙 0.1–0.7 m、深 0.4–0.6 m）和 discrete boxes，**未挂** stepping-stones 生成器 |
| `/home/ssy/ssy_files/IsaacLab/.../hf_terrains.py::stepping_stones_terrain` | 官方踏石：四周深坑 + 中心平台 + 随机高度石块 |
| `/home/ssy/ssy_files/IsaacLab/.../check_height_field_subterrains.py` | 官方演示参数：石宽 0.25–1.575 m，间隙 0.05–0.1 m，石高最大 0.2 m，坑深 −2.0 m |
| `/home/ssy/ssy_files/IsaacLab/.../mesh_terrains.py::random_grid_terrain` | `MeshRandomGrid` 是连续单元格；`holes=True` 也只是十字走廊，不是稀疏桩阵 |

### 外部主源

| Source | Relevant claim |
|--------|----------------|
| LightLP §IV-B / §IV-C1 / Table I / Fig. 3–4 / Table V | 稀疏落脚 from-scratch 会停住；illegal-footstep + 足底向下 scan（δ=0.1 m）是 stepping stones / plank / pillar 的核心惩罚；Actor 另给 privileged contact flag；Critic 有 under-foot scan。Table V：stepping stones student 99.9%，w/o FT 34.6%，w/o GRU 0；Stones(+height) 95.6 / 24.4 / 0 |
| LightLP §VII-B | 实机 stepping stones 是 perceptive loco，不是 whole-body skill；高平台 30 cm ≈ 0.33H 是单腿高台阶，也不是梅花桩 |

规则 mineru 原文路径 `docs/research/sources/2026-world-humanoid-games/...` 在本工作区 **不存在**；100m 条目以项目已整理的 rule contract 为准。LightLP **未写出** 踏石顶面边长 / 间隙厘米数。

## 3. 「梅花桩」在本项目里最可能指什么

### 3.1 100m 障碍 1–10 没有梅花桩

`docs/research/100m-obstacle-rule-contract.md` 表 2（与图 3 顺序一致）：

| # | 名称 | 几何与通过 | 失败 |
|---|------|------------|------|
| 1 | 连续斜坡 | 7.4×2.4 m，15° | 摔倒 |
| 2 | 连续跨栏 | 间距 1.1 m；杆 2.4×0.05×**0.3** m；跨越或跳跃 | 摔倒或碰倒栏杆 |
| 3 | 交叉斜坡 | 6×3 m，15° | 摔倒 |
| 4 | **绕桩** | 5 个木桩，间距 1.1 m；高约 **1.5 m**、直径约 **0.03 m**；**按序从桩间绕行** | 摔倒或**触碰木桩** |
| 5 | 对称斜坡 | 3×3.5 m，20° | 摔倒 |
| 6 | 螺旋台阶 | 每级 0.15 m，上下各 6 | 摔倒 |
| 7 | “S”弯桥 | 长 10 m、宽 **0.5 m**，桥面离地 **0.1 m** | 摔倒或跌落桥外 |
| 8 | 跳台 | 高 1 m；可手撑 | 摔倒或侧面落地 |
| 9 | 匍匐通道 | 净高 0.5 m | 摔倒 |
| 10 | 限宽 L 弯 | 宽 0.6 m | 摔倒或跌落 |

**梅花桩 ≠ 绕桩。** #4 的桩是几乎看不见的细杆，目标是 *weave between without contact*，不是 *step on top*。把它当梅花桩训，会学出错误接触语义（碰桩失败 vs 必须踩桩顶）。

#7 S 弯桥更接近 LightLP 的 plank bridge（窄连续支撑面），也不是梅花桩。

### 3.2 LightLP / AME 里和「梅花桩」同类的东西

LightLP Fig. 4 把脚式 loco 地形拆成 9 列，其中稀疏落脚三列必须分开：

| Fig. 4 列 | 论文用语 | 行为 | 像不像梅花桩 |
|-----------|----------|------|--------------|
| 5 | raised pillars | 随机正负高度的柱状凸起/坑（InstinctMJ `perlin_discrete_obstacles_terrain` 文档写的就是 pillars） | 部分像：有可踩柱顶，但地面往往仍连续 |
| 6 | stepping stones | 离散石面 + 石间空隙；必须选有效落脚 | **最像** |
| 7 | box fields | 连续 box 网格 | 像本仓 `boxes`，**不是**梅花桩 |
| 9 | plank bridges | 窄连续梁 | 像 #7 S 弯桥 |

LightLP §IV-C1 / §VII-B 对 stepping stones 的定义是：间隙里没有连续支撑；成功 = 选到并踩上有效 foothold；胸前相机看不清脚底，要靠记忆。Table V 把它和 balance beam 一起算 perceptive locomotion，**不是** climb/vault 那种 whole-body skill。

AME 侧：

- `2026-08-11--t4-hiw-three-hiking-routes.md` 路线 A/C 把 square gap / stepping stones / low box 写成覆盖目标。
- 实际 `t4_hiw_parkour/terrains.py` **没有**挂上踏石。
- InstinctMJ 生成器 `perlin_stepping_stones_terrain` 在仓库里，但 HIW parkour `ROUGH_TERRAINS_CFG` 用的是 `square_gaps`（间隙 0.1–0.7 m、深 0.4–0.6 m）和 discrete boxes，不是梅花桩场。
- 子计划 `t4-hiw-foothold-geometry-plan.md` **未落地**。

所以：用户说的「梅花桩」按本项目用语，应读成 **LightLP stepping stones（必要时加矮 raised pillars）**，不是 100m 赛道项，也不是 AME 已经训过的现成课表。

### 3.3 本仓 / Isaac Lab 可复用的生成器

本仓 Stage E（`T4_STAGE_E_TERRAINS_CFG`）现有：flat、random_rough、boxes、wave、hurdles、slope_up/down、stairs_up/down。**没有** stepping stones / gap / pit。

旧 `ROUGH_TERRAINS_CFG` 注释掉了：

```text
# "gap": MeshGapTerrainCfg(gap_width_range=(0.1, 0.4), platform_width=2.0)
```

以及 `high_platform` = `MeshPitTerrainCfg(pit_depth_range=(0.0, 0.3), double_pit=True)`——这是中心坑/高台，不是桩阵。

Isaac Lab 官方（本机 `/home/ssy/ssy_files/IsaacLab`）可直接复用：

| 生成器 | 几何 | 能不能当梅花桩 |
|--------|------|----------------|
| `HfSteppingStonesTerrainCfg` / `stepping_stones_terrain` | 全场先填 `holes_depth`（默认 −10 m，演示 −2 m），再铺随机高度石块；中心平台修平 | **首选** |
| `MeshGapTerrainCfg` | 中心平台四周一条环缝 | 只能练一次跨缝，不是桩阵 |
| `MeshPitTerrainCfg` | 往下的坑 | 高台/出坑，不是梅花桩 |
| `MeshRandomGridTerrainCfg`（本仓 `boxes`） | 固定宽单元格，高度随机；`holes=True` 只保留十字 | **不能**。默认无洞，脚随时有支撑 |
| InstinctMJ `perlin_stepping_stones_terrain` | 上面踏石 + 可选 Perlin | 算法可抄，本仓是 IsaacLab 不是 mjlab |
| InstinctMJ `perlin_cross_stone_terrain` | 从中心平台十字伸出一串石；默认 `ground_depth=-0.5` | 有序梅花桩走廊，适合 evaluator |

官方演示数值（`check_height_field_subterrains.py`，**不是** LightLP 论文数）：

```text
stone_width_range = (0.25, 1.575)   # difficulty↑ → 取下限 0.25
stone_distance_range = (0.05, 0.1)  # difficulty↑ → 取上限 0.10
stone_height_max = 0.2
holes_depth = -2.0
platform_width = 1.5
horizontal_scale = 0.1
```

难度公式（Isaac / InstinctMJ 相同）：石宽 = `max − d·(max−min)`，间隙 = `min + d·(max−min)`。容易档是大石、窄缝。

### 3.4 相对 T4 脚和 0.1 m scan，几何建议范围

脚包络（Isaac 训练用的 URDF 三 box，`t4_std.urdf` / `ISAAC_FOOT_BOXES`）：

| box | origin xyz | full size xyz |
|-----|------------|---------------|
| 前掌 | (0.089, 0, −0.027) | 0.119 × 0.077 × 0.015 |
| 后跟 | (−0.0355, 0, −0.027) | 0.050 × 0.069 × 0.015 |
| 中段 | (0.0095, 0, −0.027) | 0.040 × 0.066 × 0.015 |

包络：x ∈ [−0.0605, 0.1485] → **长 0.209 m**；y 半宽 0.0385 → **宽 0.077 m**。MJCF 默认 7 根 capsule（`fromto` x −0.056…0.143，y ±0.030，半径 0.01）大约 0.20 × 0.08 m，同一量级。

Teacher scan（`schemas.py`）：

```text
resolution = 0.1 m
size = (1.4, 1.2) → 15 × 13 = 195
offset = (0.9, 0) → 前向 0.2–1.6 m，侧向 ±0.6 m
clip = (−1, 1)；射线 miss → 1.0
history = 1（只有当前帧）
挂在 Trunk，不是脚
```

可见性（由上面常数推出，不是论文数）：

| 物体 | 约占网格 | 前向 scan 能不能当决策依据 |
|------|----------|----------------------------|
| 跨栏杆厚 0.07 m | < 1 cell | Spec 已写「间歇 aliasing」 |
| 绕桩 φ 0.03 m | << 1 cell | 几乎不可见 |
| 石顶 0.20 m | ~2 cell | 贴着 Nyquist，易丢 |
| 石顶 0.25 m（Isaac 最难档） | 2–3 cell | 勉强 |
| 石顶 0.30–0.40 m | 3–4 cell | 第一版应取这一档 |
| 间隙 0.05 m | 半格 | 可能被扫成连续地面 |
| 间隙 0.10 m | 1 cell | 临界 |
| 间隙 0.20 m | 2 cell | 作为「洞」更稳 |
| 坑深 −2…−10 m | miss → 1.0 | 对比极强，scan 看得见「没地」 |
| 桩顶相对邻石 ±5–10 cm | 0.05–0.10 | 和现有 `boxes` 高差同类 |

**第一版几何建议（工程范围，不是赛场明文）：**

```text
桩顶 / 石面 : 0.30–0.45 m（≥ 脚长 0.21、≥ 3 个 0.1 m 格）
间隙        : 课程 0.05–0.10 → 0.12–0.20 m（先小于一步，再逼精确落脚）
高度差      : 先 0，再 ±0.05，最后 ±0.10–0.15
坑深        : ≤ −1 m 或官方 −2 m，让 miss 打到 invalid=1.0
中心平台    : 1.5–2.0 m，保证 reset 站得住
```

不要一上来用 InstinctMJ `square_gaps` 的 0.7 m 间隙——那已经接近跳跃/大跨，不是梅花桩入门。桩顶小于 0.25 m 时，0.1 m 前向 scan 会像 0.07 m 栏杆一样 alias；再小就必须改 schema 或加足底 scan 进观测，那是新合同。

## 4. 当前跨栏高度：训到哪、验到哪

### 4.1 四套高度不要混

| 层 | 高度 | 出处 |
|----|------|------|
| 训练课表 | difficulty 0→1 线性 **0.05→0.35** | `T4_HURDLE_BAR_HEIGHT_RANGE`；`hurdle_bar_height(d)=0.05+d·0.30` |
| H2 默认行 | d=0.85 → **0.305 m** | `eval_t4_hurdle.py --difficulty 0.85` |
| Spec gate | **0.30 m**，杆厚 0.05，间距 1.1，corridor 2.4，0.7 m/s，≥90% @500，零碰栏 | `docs/specs/2026-08-13--t4-hurdle-skill.md` 成功标准 |
| Spec 报告项 | **0.35 m**、0.9 m 密间距、可倒杆 | 同上，不 gate |
| 100m #2 | **0.30 m**，厚 0.05，间距 1.1 | rule contract 表 2 |
| MuJoCo `loco` 回放 | 两根杆 **0.22 和 0.28** | `build_loco_course()`：`enumerate((0.22, 0.28))`，`size_z=0.5*height` |
| MuJoCo `rule` 课 | 10 栏，body z=0.15、`size_z=0.15` → **全高 0.30** | `obstacle_2_hurdle_*` |
| `--course hurdles` | 同样半高 0.15 → **0.30** | `build_model_xml` |

`h(d)` 几个锚点：d=0 → 0.05；d=0.57 → 0.22；d=0.77 → 0.28；d=0.83 → 0.30；d=1 → 0.35。

### 4.2 验到哪：progress ≠ zero-contact

H0/H1（布局 + 地形接入）已交付。H2 是 **progress diagnostic**，不是 gate。

`.harness/state.md`（2026-08-14 修正）：

- 初版在 `T4LocoEnv.step()` 自动 reset **之后**读 `root_pos_w`，且用 world-x 而非初始 yaw 的 body-forward；**旧数字作废**。
- 修正后 `stage_e_prov7_hurdle/model_14000.pt`：flat 10/10 达 3 m，均 progress 12.73 m；hurdle **83/100**，均 10.78 m。
- 原文 caveat：**不等价于零碰栏 / 10 栏 gate**，仍需 H3/H4。

`eval_t4_hurdle.py` 自己写：`final_zero_contact_gate: False`；成功 = 不中断 timeout **且** body-forward ≥ 3 m。环形 2–3 杆 tile 上犁杆前进也能过。

`eval_t4_hurdle_mujoco.py` 会数 `hurdle_*` 接触，但是 **MuJoCo 重建几何的诊断**，episodes 默认 10，不是 IsaacLab 500-trial gate。

走跑关闭依据（`.harness/work_index.md`）：用户用 `model_24000.pt` 在 MuJoCo **`loco` 交互**里目视通过。`loco` 只有 0.22/0.28 两根杆，不是 0.30×10 零碰栏。

### 4.3 H3/H4 不存在

计划里 H3/H4 checkbox 仍是空的。仓库里 **没有** corridor + ordered gates + AABB 零碰栏的 IsaacLab evaluator。`eval_t4_vault.py` 的 corridor 骨架在翻箱侧，跨栏还没复用。

因此：**不存在 H3/H4 strict evaluator。** 不能把 83/100 或 `loco` 目视说成 0.30 m 过关。

### 4.4 「提高跨栏高度」更可能指什么

按证据，用户更可能是 **A，不是 B**：

| | A. 回放对齐规则/Spec | B. 课表推过 0.35 |
|--|----------------------|------------------|
| 改什么 | `build_loco_course()` 的 `(0.22, 0.28)` → 0.30 或 0.30+0.35 | `T4_HURDLE_BAR_HEIGHT_RANGE` 上端 |
| 要不要新 lineage | 否（只改诊断场景） | **要**（地形分布变更 = 从零） |
| 和已批准合同 | 对齐 gate 0.30 / 报告 0.35 / 规则 0.30 | Spec 明确 0.35 只是报告项 |
| 先做什么 | 同一 ckpt 看 0.30 是否还抬腿、是否碰杆 | 先有 H3/H4 或至少零碰栏诊断 |

A 半天内能做完。B 是新训练面，且会和翻箱抢决策带宽。

## 5. 梅花桩该怎么训才符合本仓库合同

### 5.1 2026-08-14 裁定直接适用

跨栏 Spec 用户原话：*「跨栏是纯脚上地形障碍，和翻箱子（手上接触技能）不是一类，单独训练与现有 Stage E 割裂；可以放进正常 locomotion 课程一起训练」。*

梅花桩同样是脚上 perceptive loco（LightLP 自己也把它放在 locomotion 列，不是 climb/vault）。因此：

- **并入 Stage E bucket**，与 stairs / rough / hurdles 并列。
- **不**开独立 skill lineage，**不**抽 66D skill AMP，**不**做 3 技能 merge。
- **不**改 1155D，**不**把 height map / `contact_truth` 放进 Actor。
- 地形分布一变 = **新 lineage 从零**（跨栏 Spec 约束条；MDP 变更同样从零，见 `.harness/state.md` 2026-08-13 四项变更先例）。

### 5.2 LightLP 为什么说 from-scratch 会停住；illegal-footstep 是不是必须

LightLP §IV-C1：

> On the hardest terrains, such as stepping stones and balance beams, a policy trained from scratch tends to halt in front of an obstacle to avoid falling. Under the exponential tracking reward of Eq. (1) alone, this timidity is never overcome.

他们的补丁是 **velocity-slack**（式 3，权重 +1.5）：前向速度落在命令的 [0.3, 1.5] 倍就给奖，逼机器人往前走。

另一组核心项是 **illegal-footstep**（式 4，权重 −1.0）：每只接触脚向下打一小格射线，命中点比脚低超过 **δ=0.1 m** 的比例当惩罚。Fig. 3：这是足底 scan，用来算 reward，**不是**部署观测。论文另外写：稀疏地形上 Actor **额外**拿 privileged contact flag；Critic 拿 under-foot height scans。

Table V：stepping stones 上 depth student 去掉 GRU 成功率 **0**；去掉最后 RL fine-tune 掉到 34.6%（等高变体 24.4%）。脚式稀疏落脚对感知误差极敏感。

对 T4 的映射：

| LightLP 件 | T4 现在有没有 | 能不能原样搬 |
|------------|---------------|--------------|
| 前向 HeightScan teacher | 有，195D @0.1 | 保持；石顶 ≥0.30 m 才稳 |
| 指数速度跟踪 | 有，`track_lin_vel_xy_exp` 权重 2.0 | 单独不够，LightLP 已消融 |
| velocity-slack | **无** | 可加，但是 MDP 变更 = 新 lineage |
| illegal-footstep + 足底向下 scan | **无** | **只做 reward / critic** 不改 1155D；进 Actor = `contact_truth` / 新 scan，违禁 |
| Actor contact flag | 禁（`TEACHER_FORBIDDEN_PRIVILEGE_FIELDS` 含 `contact_truth`） | 不要给 Actor |
| Critic `feet_contact` | 已有 2D | 可留 |
| 足底 scan 进 Actor | 无；scan history=1 | 不要。Depth student 也没有脚相机 |
| GRU + scan 重建 | student 是 3 帧 depth MLP | 第一版 teacher 不要改网络 |

**illegal-footstep 不是「改观测合同」的必须项，但是「稀疏落脚可学」的强证据。** 它可以是训练期惩罚（足底射线只在 Isaac 里算），导出 Actor 仍然只有 depth+proprio+cmd。这和 LightLP Fig. 3 的用法一致，也和 InstinctMJ HIW 把 `left/right_height_scanner` 只用于 `feet_at_plane` reward、不进 policy 一致。

T4 没有 slack 时，课程 promote（径向 ≥4 m）理论上会降级停住的 env，但指数 tracking 仍可能在出生平台上「跟得很像但不往前」。梅花桩第一版至少要准备：要么 slack / 显式 progress，要么像跨栏那样用几何逼你迈出去（环形/走廊 + 远处 gate）。

### 5.3 为什么现有 `boxes` 不能当梅花桩

`T4_STAGE_E_TERRAINS_CFG["boxes"]`：

```text
MeshRandomGridTerrainCfg(
    proportion=0.15, grid_width=0.45, grid_height_range=(0.0, 0.15), platform_width=2.0
)
```

`random_grid_terrain`：正方形单元格铺满，只随机 **顶面高度**；默认 `holes=False`，邻格贴在一起。脚始终有支撑，学的是高低差，不是选 foothold。`holes=True` 也只留十字走廊，空区是整块缺口，不是桩阵。

格子 0.45 m 远大于脚 0.21×0.08，高差最大 15 cm，和 LightLP「box fields」同类，和 stepping stones 不是一类。把 `grid_height_range` 加大只会变成高台阶/boxes Path A，仍然没有「踩空即掉」。

### 5.4 最小可行 vs 过重

**最小可行（建议写进未来 Spec 的上限）：**

1. 新 sub-terrain：官方 `HfSteppingStonesTerrainCfg`（或薄封装，课程参数见 §3.4）。
2. 并入 `T4_STAGE_E_TERRAINS_CFG`，占比从 flat/rough/wave 里挪，**不要**动 vault plant。
3. 观测 / 动作 / AMP 合同不动。
4. 可选但强烈建议：illegal-footstep 作 **RewTerm**（足底向下射线，δ=0.1 m 可先照抄 LightLP）；**不要**把射线拼进 Actor。
5. 可选：velocity-slack 或更硬的径向 progress（已有 `episode_max_radial_dist`）。
6. Evaluator：corridor + 有序石块/桩 + 脚落在桩顶 AABB 内 + 掉坑失败；progress 不能当成功。
7. **新 lineage 从零**。不要从 `stage_e_prov7_hurdle` 续。

**过重（第一版拒绝）：**

- 独立梅花桩 skill + AMP clips + 将来 3-way merge。
- LightLP 全流水线（per-skill teacher → DAgger → transition group → GRU depth + scan 重建）。那是 90 cm 机器人在已有 perceptive loco **之后**扩全身技能的系统。
- 改 1155D：足底 scan、contact flag、加密网格、加 history。Student 不可达，且冻结合同禁止。
- 把 Actor 换成 HIW 的 2012-D / 32×20×3。
- 用 AME `square_gaps` 0.7 m 当入门。
- 把 #4 绕桩和梅花桩合成一个 bucket。

### 5.5 要不要新 lineage

要。跨栏 Spec：「地形分布变更 = 新 lineage 从零」。加 illegal-footstep / slack 还是 MDP 变更，同样从零（`.harness/state.md` 对 `4d4e088` 的四项变更已有先例）。

只改 MuJoCo 回放场景、不加训练地形，**不必**新 lineage。

## 6. 跨栏再加高怎么搞

### 6.1 0.30 / 0.35 / >0.40 对 ~1.4 m T4 的意义

站高线索（无独立 FK 头顶测量，用 MJCF 站点）：

- Trunk `pos z=0.9`
- `forward_camera` Trunk+0.42 → **1.32 m**
- `head_pitch` 0.3155 + `head_yaw` 0.0795 + geom 0.025 → 头顶约 **1.32 m**
- `mid360_link` Trunk+0.535 → **1.435 m**

沿用爬箱笔记的 **H≈1.4 m** 标尺（范围 1.32–1.44）：

| 杆高 | h/H @1.32 | h/H @1.40 | 行为归类 |
|------|-----------|-----------|----------|
| 0.22（`loco`） | 0.17 | 0.16 | 略抬腿 |
| 0.28（`loco`） | 0.21 | 0.20 | 接近规则杆 |
| **0.30（规则 / gate）** | 0.23 | 0.21 | Spec：高抬腿跨步，不是跳 |
| **0.35（训练顶 / 报告项）** | 0.27 | 0.25 | 仍是脚上跨步 |
| 0.40 | 0.30 | 0.29 | 仍低于 LightLP 高平台 0.33H（那是 **上台阶**，不是跨栏） |
| 0.50 | 0.38 | 0.36 | 接近 LightLP 高楼梯 35 cm / 0.39H；开始像「必须有飞步」 |

0.30 对 T4 是 **0.21H 摆动腿 clearance**（跨栏 Spec 原文）。0.35 仍在同一类。**>0.40 才需要讨论要不要当成跳跃 skill**；在那之前不要另开 lineage 名。

### 6.2 Spec 已写明的高杆风险

`docs/specs/2026-08-13--t4-hurdle-skill.md` 残余风险：

1. **0.07 m 杆 × 0.1 m scan aliasing** — 更高的杆在 *高度* 上更可见，但 *厚度* 仍可能间歇丢。
2. **0.35 m + 姿态终止 −200** — 高抬腿容易 |pitch|>1.0 rad 或 |roll|>0.8，课程在高难度行卡住。
3. **AMP 衰减不够** — 专家全是平地；`T4AmpTerrainScheduleCfg` 从 difficulty 0.3 线性降到 floor 0.3。高抬腿仍可能被 discriminator 打。
4. 8×8 tile 只有 2–3 环，10 栏节奏不在训练分布里。

这些在 **还没通过 0.30 零碰栏** 之前就会发作。把 range 改成 (0.05, 0.45) 只会让课程更早撞 −200。

### 6.3 先诊断，不要先改 range

建议顺序（仍不替用户排优先级）：

1. **零训练：** `loco` 杆改为 0.30，必要时再加一根 0.35；同一 student/teacher ckpt 目视是否预抬腿、是否碰杆。
2. **便宜诊断：** 现有 `eval_t4_hurdle_mujoco.py` 扫 d=0.83（0.30）和 d=1.0（0.35），看 contact count，不要只看 progress。
3. **若要宣称能力：** 先做 H3/H4（10 栏 @0.30 零碰栏）。这是已批准、未做的 gate。
4. **仅当 0.30 零碰栏过了、0.35 报告项明确失败**，再讨论微调 AMP floor / 课程，或 **新 lineage** 把训练顶从 0.35 略抬。
5. **不要**为了「看起来更高」加 clearance 奖励——Spec 非目标写明碰杆专用项 = MDP 变更 = 再开 lineage；现有 stumble / shank / undesired / 姿态终止已覆盖。
6. **>0.35 或 >0.40：** 新 lineage；先复用 H3 几何只改高度当报告项；只有出现稳定飞步需求才讨论跳跃 skill。那时梅花桩式落脚（起跳/落地精度）会和加高跨栏耦合，更不能两件事一起改。

用户「提高跨栏高度」若没有同时要求改 `hurdle_layout.py`，默认走第 1–3 步。

## 7. 和工作面冲突

当前事实（`.harness/work_index.md` / `state.md` / vault recovery plan）：

| Surface | 状态 | 含义 |
|---------|------|------|
| 翻箱 G1/G2 recovery | **唯一 active** | R3 教师开车 5000；R4 waiter 挂在 zhuoqun |
| 走跑 / 深度走跑 | **已关闭** | 不要把 loco ckpt 当翻箱 expert |
| 跨栏 Stage E bucket | 作为 loco 一部分关闭 | H3/H4、100m `rule` 零碰栏 **不在**完成声明里 |
| 梅花桩 | 无 surface | 本文不创建 |

依赖，不是替用户拍板：

```text
翻箱 R3/R4（正在跑，zhuoqun）
    ⟂ 不要在同一卡上新开 Stage E / 梅花桩长训

跨栏「加高」诊断（改 loco XML + 可选 MuJoCo contact 计数）
    不依赖翻箱，不占训练卡
    不产生能力声明

H3/H4 0.30 零碰栏 evaluator
    依赖：走廊骨架（可抄 vault V2）+ 空闲 GPU 做 500 trial
    与翻箱抢 nubot/zhuoqun 时，应等用户切 surface

梅花桩 Stage E bucket
    依赖：用户批准新 lineage；1155D 冻结；最好先有跨栏零碰栏诊断
         （证明「新 bucket + 旧 MDP」这条路还成立）
    不依赖：翻箱 G2 成功（脚上任务不必等手上技能）
    若与翻箱并行：必须独立 worktree / 独立卡，且不改 schemas.py
```

**不要**在翻箱 recovery 进行中改 `T4_STAGE_E_TERRAINS_CFG` 或 `hurdle_layout.py` 上端——那会模糊 loco 主线已关闭的事实，并逼出一次与 vault 无关的从零长训。

## 8. 建议的下一步（仍等用户选）

若只做一件便宜的事：**把 `loco` 两根杆改成 0.30（再选一根 0.35），用现 ckpt 回放。** 这回答「现在到底能不能跨规则高度」。

若要宣称跨栏能力：开工已写在计划里的 **H3/H4**，不要先加高训练 range。

若确认要梅花桩：另开 slice，Spec 写成「Stage E stepping-stones bucket」，几何用 §3.4，MDP 最多加 illegal-footstep（reward-only），**新 lineage 从零**。

## 9. 未核到的缺口

1. 规则 mineru 原文不在本工作区；100m 十条以 `100m-obstacle-rule-contract.md` 为准，未回链到 PDF 页码。
2. LightLP **没有**踏石边长 / 间隙 / 柱顶厘米数；§3.4 是脚尺寸 + scan 网格 + Isaac 官方演示推出来的。
3. T4 头顶高度没有独立 stand-pose FK 测量；H 用 1.32–1.44 m 站点包络。
4. AME `t4-hiw-foothold-geometry-plan.md` 被引用但文件不存在；HIW mixed terrain 实际不含踏石。
5. H2 83/100 的 Isaac 原始 JSON 未在本笔记中打开；数字引自 `.harness/state.md`。
6. `stage_e_prov7_hurdle` 最终 ckpt 在 hurdle 行上的 `terrain_levels` / 是否犁杆，本笔记未重跑 TB。
7. 未在 Isaac 里实测 0.25 m 石顶在 0.1 m RayCaster 上的命中率；可见性表是网格算术。
8. 未核 LightLP 前向 scan 的分辨率 / 窗口（论文只写 height scan + 足底 δ=0.1 m）。
9. 未核 BeamDojo 踏石几何（Table V 只引了 91.7% 成功率）。
10. `loco` 里 box `size=` 按 MuJoCo 半长解释；与 Isaac 全尺寸对照未做运行时 assert。

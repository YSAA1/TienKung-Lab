# Research: T4 爬箱子方案

> Date: 2026-08-13
> Question: 在已批准的 T4 统一 depth locomotion 合同下，如何做「爬箱子」？Light-Loco-Parkour 与 2025–2026 相关工作里，哪些必须借、哪些对当前未完成的 Stage E 过重？
> Status: findings for a future slice (does not amend Spec/Plan by itself)
> Paper: `docs/research/2608.02653v1/auto/2608.02653v1.md`（arXiv:2608.02653）

## 1. Verdict

**先把「箱子」拆成两个任务，不要混成一次训练。**

| 任务 | 行为 | 当前仓库距离 | 建议 |
|------|------|--------------|------|
| **A. 脚上箱子** | 只用脚踏上/跨过离散 box，无手撑 | Stage E 已有 `boxes`，但高度只到 **15 cm** | 并入现有 teacher 课程，作为 stairs 之后的高台阶 bucket |
| **B. 手上箱子** | 手/膝/胸接触，把身体抬上高于腿长舒适范围的箱子（LightLP 的 climb-and-step） | 当前 MDP **明确禁止** Trunk/上臂接触；无可用接触动作 | Stage E evaluator 通过后再开独立 skill lineage；只做 **一个** climb-and-step，不一次上 vault/翻越 |

文献共识与 T4 现状对齐后的结论：

1. **纯 reward 学不出用手爬箱子（depth + 速度命令设定下）。** LightLP 去掉 expert distillation 则 climb/vault 不收敛；PHP 自己的速度跟踪基线在 G1 上只到 **36 cm、且始终是脚式**。例外是 [APEX, arXiv:2602.11143](https://arxiv.org/abs/2602.11143)：无 mocap、6 个 specialist + LiDAR 高程图，G1 实机爬 **0.8 m**。APEX 的部署观测与 T4 Spec（depth CNN、Actor 无 HeightScan/地图）冲突，不当作第一版路线。
2. **现有 18 条 T4 AMP clip 不能当爬箱子参考。** 接触时序已被裁定不可用；它们是平地走跑，且 AMP 会惩罚爬箱所需的前倾/撑手姿态。
3. **不要复制 LightLP 全流水线（每技能 teacher → DAgger 合并 → transition group → depth GRU → 多技能 parkour）。** 那是在已有 perceptive loco backbone 之后、为 90 cm 低力矩机器人扩技能集的系统。T4 的 Stage E teacher 还没有行为验收。
4. **T4 第一版手上箱子应借 LightLP 的数据配方，不借 PHP/MGMT 的运行时 motion graph / diffusion generator。** 与 Spec「单一导出 Actor、无 reference/skill label、PPO+AMP 主干」一致。

推荐顺序：**A 作为 Stage E 的高 box/高台阶扩展 → B 用一条 seed 做出物理可行的 climb 参考 → 特权 skill teacher → 再与 loco teacher 做 DAgger 合并。** Vault / reverse-vault / 连续课程链是 B 成功之后的事。

## 2. Primary sources consulted

### 本仓库

| Source | What it establishes |
|--------|---------------------|
| `docs/specs/2026-08-12--t4-unified-depth-locomotion.md` | 部署 Actor 只有 depth+proprio+cmd+prev action；teacher 可用局部 HeightScan；禁止一次性联训全部能力 |
| `PROJECT_CONTEXT.md` §7–§9 | 两阶段骨架是基础 loco → rough+楼梯；「不做不必要的多策略、MoE、多阶段复杂系统」 |
| `docs/research/2026-08-12--t4-two-stage-training-scheme.md` | 正式训练次数应压成两次；route 不是第三次 PPO |
| `.harness/state.md` | 当前工作面是 Stage E teacher（`stage_e_prov4`）；evaluator 尚未通过；无行为能力声明 |
| `legged_lab/terrains/terrain_generator_cfg.py` | Stage E `boxes`：`grid_height_range=(0.0, 0.15)`；楼梯台阶最高约 0.20 m |
| `legged_lab/envs/t4/teacher_cfg.py` | `undesired_contacts` 与 `terminate_contacts_body_names` 含 `Trunk`, `A[LR]2`, `A[LR]4`；另有 `joint_deviation_arms`、`body_orientation_l2` |
| `legged_lab/assets/t4/mjcf/t4_std.xml` | Trunk `z=0.9`；头链约 1.30 m；相机注释约 1.32 m → T4 是 **~1.4 m 级**，不是 Lightbot 0 的 0.90 m |
| `.harness/state.md` M0 缺陷 | 现有 motion 接触时序不可用；66D AMP 不含绝对 root 高度 |

### 外部主源

| Source | Relevant claim |
|--------|----------------|
| [Light-Loco-Parkour, arXiv:2608.02653](https://arxiv.org/abs/2608.02653) / [project](https://light-loco-parkour.github.io/) | 见 §3。90 cm / 21 DoF Lightbot 0；climb 50–75 cm（0.55–0.83H）；seed→物理 mimic→每轮抬 5–10 cm；DAgger+PPO；transition group 必需 |
| [PHP, arXiv:2602.15827](https://arxiv.org/abs/2602.15827) | G1 1.3 m / 29 DoF；OmniRetarget + **离线** motion matching；每技能 privileged tracking teacher → DAgger+PPO depth student。Student 观测是 proprio + depth + **离散 2D 速度**（2 档×5 朝向），**论文未写 skill one-hot**；LightLP「one-hot / 被障碍吸走」是 LightLP 对 PHP 的批评，不是 PHP 自述。实机爬墙 **1.25 m / 3.63 s**；vault 0.4×0.5 m、峰值 3.41 m/s。Reward-only 基线停在 **36 cm** |
| [Deep Whole-body Parkour, arXiv:2601.07701](https://arxiv.org/abs/2601.07701) / [project](https://project-instinct.github.io/deep-whole-body-parkour/) | G1；光学 mocap + iPad LiDAR 对齐场景；GMR 重定向；depth 进 tracking；训练箱 **0.5×0.6×0.4 m**；技能是 kneel-climb / roll-vault 等。**部署仍跟踪参考**，不是 T4 的 velocity-command 合同 |
| [HIL, arXiv:2505.12619](https://arxiv.org/abs/2505.12619) | Actor **不看** per-frame reference，只看任务条件；track env + general RL env 混合。LightLP 写明：HIL 式、无 expert distill 时，他们的 parkour teacher **不收敛** |
| [Parkour in the Wild, arXiv:2505.11164](https://arxiv.org/abs/2505.11164) | 四足：每地形 expert（高程图）→ DAgger 成 depth 基础策略 → RL fine-tune。LightLP transition 明确 follow 这篇。蒸馏后必须 RL fine-tune，否则跨技能/未见地形差 |
| [MGMT, arXiv:2604.17335](https://arxiv.org/abs/2604.17335) | G1；diffusion 生成器 + RL tracker；部署用 Livox 高程图，不是 depth-only。种子 climb 50 cm / vault 35 cm，增强到箱 35–75 cm；fine-tune 到 85 cm。Sim Tracker+Gen 80 cm 成功率 0.962（tracker-only 0.230）；实机 **75 cm** 用手+膝。与 T4 单 Actor / 无地图冲突 |
| [OmniRetarget, arXiv:2509.26633](https://arxiv.org/abs/2509.26633) / [project](https://omniretarget.github.io/) | 交互网格重定向；G1 本体跟踪实机爬 **0.9 m（约 0.69H）**。LightLP Table IV：climb 漂、vault 瞬移 — 可当 retargeter，不能当 perceptive policy 或唯一 expert |
| [GMR, arXiv:2510.02252](https://arxiv.org/abs/2510.02252) | 运动重定向；LightLP：忽略障碍 → 手/膝穿透 |
| [APEX, arXiv:2602.11143](https://arxiv.org/abs/2602.11143) / [project](https://apex-humanoid.github.io/) | G1；**无 mocap**，ratchet progress + 6 skill teacher → distill；部署 **LiDAR elevation map**。实机全身手爬 **0.8 m**（约 114% 腿长），sim 1000 trial 95.4%。是 mocap-free 手爬的存在证明，但观测合同不兼容 T4 |
| [Humanoid Parkour Learning, arXiv:2406.10759](https://arxiv.org/abs/2406.10759) | Unitree H1 19 DoF；reward-only + scandots teacher → depth student。实机跳上 **0.42 m** / plyo **0.4 m**；手臂只平衡、不承力。属于 Path A，不是手上箱子 |
| [AMP, TOG 2021](https://arxiv.org/abs/2104.02180) / DeepMimic RSI | LightLP transition 用 **相位切换 AMP**（触发点前 loco AMP，之后 skill AMP）；mimic 阶段 RSI 是动态技能可学的关键 |
| [BeyondMimic, arXiv:2508.08241](https://arxiv.org/abs/2508.08241) | LightLP object-interaction mimic 的底；从自由空间 tracking 扩到接触 |
| Unofficial HIL climb code ([Hybrid-Motion-Imitation](https://github.com/jiashunwang/Hybrid-Motion-Imitation)) | G1 29DoF `g1-29dof-wbt-hybrid-climb`：Stage1 只 track `climbox1`，Stage2 50/50 track+gen。是图形/跟踪栈，不是 T4 的 PPO+AMP loco 栈 |

检索后端：agent-reach Exa + GitHub `gh search` + Jina 读 arXiv HTML / 项目页。

## 3. LightLP 里和爬箱子真正相关的部分

论文把「箱子」用在两处，不要弄混：

- **Locomotion 地形 column「box fields」**：脚式 perceptive loco，和现在 Stage E 的 `MeshRandomGrid` 同类。
- **Whole-body skill「climb-and-step」**：双手撑到约胸高，把身体抬上平台，跪到站起再跳下。这才是「用手爬箱子」。另有 speed-vault / reverse-vault **翻过**箱子，不是爬上去。

流水线（Fig. 2 / §IV–VI），对 T4 的含意：

```text
1. 特权 perceptive loco teacher（HeightScan + 速度命令，无 reference）
2. 每技能一条视频 seed → GVHMR+GMR → 人手把虚拟箱子对齐接触点
3. Object-interaction mimic（跟踪全局 root；RSI；接触部位全局位置奖励）
   → 物理下不再穿透、可执行
4. 成功 rollout 当作下一条参考，箱子和运动一起抬 5–10 cm，重复
   论文例子：一条 45 cm climb 扩到 75 cm
5. 把「最难那条」expert 蒸馏到 HeightScan+cmd 观测（DAgger + PPO）
6. 丢掉 expert，在增强得到的高度范围内做技能泛化；task reward 是箱子后方目标
7. 多专家 DAgger：loco 组 + 每技能组，Actor 无 skill label
8. 再加 transition 组：无逐帧参考；稀疏过箱奖励 + 相位切换 AMP
9. 最后才把 HeightScan 策略蒸馏成 onboard depth（GRU + 重建 scan + RealSense 噪声）
```

关键数字（论文 Table V，500 trial；机器人站高 H=90 cm）：

| Skill | 高度 | Student 成功率 | Teacher | 无 expert distill | 无参考增强 |
|-------|------|----------------|---------|-------------------|------------|
| climb-and-step | 60 cm / 0.66H | 99.2% | 99.9% | ✗ 不收敛 | 99.9%（只在种子高度） |
| climb-and-step | 70 cm / 0.77H | 90.0% | 99.2% | ✗ | ✗ |
| climb-and-step | 75 cm / 0.83H | 33.4% | 98.6% | ✗ | ✗ |
| speed-vault | 40–50 cm | 99–93% | ~100% | ✗ | 高处失败 |

Transition（Table VII，100 trial）：无 transition 组 = 0%（平地+孤立技能）或 33%（加上 rough loco）；有 = 98%；去掉 AMP = 51% 且动作不自然。

实机：climb 在未见过的 50 cm 鞍马形障碍上仍成功；high platform **30 cm ≈ 0.33H** 是单腿高台阶，属于脚式 loco，不是用手爬。

作者自己写的上限：seed 仍要人把视频和障碍对齐；技能集小、障碍挨太近会退化；胸前深度相机在 vault/爬的中段会看天。

## 4. 2025–2026 其它路线（只保留对 T4 的差）

三条主路线都承认：**没有「和障碍对齐的参考」，用手爬/撑箱学不会。**

| 路线 | 参考从哪来 | 部署时 Actor 看什么 | 和 T4 Spec |
|------|------------|---------------------|------------|
| LightLP | 一条视频 + 物理 self-aug | 只有 depth + cmd，无 reference / one-hot | **最接近**。Teacher 用 scan 与当前 Stage E 同构 |
| PHP | 稀缺原子技能 + 离线 motion matching | depth + 离散 2D 速度（非 skill one-hot） | 运动库过重；1.25 m 是能力上限参考。Reward-only 只到 36 cm |
| APEX | 无 mocap，ratchet + 6 specialist | LiDAR 高程图 | 证明无参考也能手爬，但部署观测与 Spec 冲突 |
| DWBP | mocap×LiDAR 场景对 | **仍跟踪参考**，depth 用来对初始位置/手点 | 违反「无 reference 导出 Actor」 |
| MGMT | 在线 diffusion 生成参考 | 生成器 + tracker 两网络 | Spec 禁止运行时第二网络 |
| HIL / hybrid-climb | 少量 climb clip；Actor 不看 reference | 任务 goal，不是 T4 的 velocity command | 思想可借（obs 不放 reference）；LightLP 消融表明 **缺特权 expert 时人形 parkour 不收敛** |
| Parkour in the Wild | 无整身参考，四足脚式 | depth；expert→distill→RL FT | 只借「合并后再 RL fine-tune」；不能替代人手接触数据 |

高度标尺（用 h/H，避免和 90 cm / 1.3 m 机器人比绝对厘米）：

- 脚式高台阶：LightLP 实机 0.33H；H1 2024 跳上 0.42 m；T4 现楼梯 ≤0.20 m ≈ 0.14H。
- 手撑爬上：LightLP 学生有效到 ~0.77H（0.83H 掉到 33%）；MGMT 实机 75 cm；OmniRetarget 跟踪 0.9 m；APEX 地图策略 0.8 m；PHP 爬墙 0.96H。
- T4 若按站高 ~1.4 m：0.33H≈46 cm，0.55H≈77 cm，0.77H≈1.08 m。**第一版手上箱子不要冲 1 m。**

## 5. 对 T4 的硬约束（方案必须遵守）

1. **Stage E 未验收。** 爬箱子不能替换当前 teacher；LightLP 自己也是先训 perceptive loco。
2. **当前 contact MDP 与爬箱子相反。** `Trunk` / `A[LR]2` / `A[LR]4` 接触会终止；臂偏差、躯干竖直都被惩罚。任务 B 必须改允许接触集合，不能只加一个箱子地形。
3. **现 AMP expert 是平地分布。** Spec 已在楼梯上衰减 AMP；爬箱需要 **单独的 skill AMP**（或阶段切换），否则 discriminator 会打掉撑手姿态。现 clip 接触不可用，不能从它们榨 climb prior。
4. **Teacher 可用 HeightScan，导出 Actor 不能。** 与 LightLP §VI 一致。爬箱中段相机会被躯干挡住，学生侧需要比现在 3 帧 depth 更强的短时记忆（LightLP 用 GRU + scan 重建）。
5. **防绕行。** 箱子放在 tile 中间 + 向前 progress，会被绕开。沿用 Spec 的 corridor + ordered gates。
6. **T4 有 7+7 臂，比 Lightbot 21 DoF 更适合撑箱**；但力矩/质量未知，seed 必须在 **T4 物理**下 mimic，不能直接播 G1/人类轨迹。

## 6. Recommended scheme

### 6.1 Path A — 脚上箱子（先做，可进现 lineage）

目标：把 Stage E 的 box/platform 从 15 cm 抬到 **一次高台阶**，仍禁止手撑。

- 新 bucket：单级 box / `high_platform`，高度课程约 15 → 25 → 35→（若站高实测允许）~0.30H。
- 观测/动作/AMP 合同不变；`undesired_contacts` 保持现状。
- 验收：固定朝向、corridor 防绕；成功率分高度；必须提前抬脚，不能撞上再爬。
- 这是 LightLP 的「high platform 30 cm」，不是 climb-and-step。

**不要**在 Path A 里放开手接触「看看会不会自己爬」：LightLP Teacher w/o ED 已经给出否定答案，且会污染现有 stairs 奖励。

### 6.2 Path B — 手上箱子（Stage E evaluator 通过后的新 slice）

WIP=1：只做 **climb-and-step 上一个箱子**。非目标：vault、reverse-vault、技能链、室外课程、改部署 Actor 合同。

**B0. 测站高与可达高度（半天，无训练）**

用 T4 stand pose FK 量头顶高度 H。第一目标箱高取 **h ≈ 0.40–0.50H**（约 55–70 cm 量级），低于 PHP 的 0.96H，接近 LightLP 种子 45 cm / 0.50H。高于当前楼梯 20 cm，迫使用手。

**B1. 一条物理可行的 seed（这是数据瓶颈）**

1. 拍或找一段「双手撑箱、膝上沿、站上箱顶」的短视频（或 mocap）。
2. GVHMR（或已有 retarget）→ GMR 到 T4 27DoF 顺序（`T4_JOINT_NAMES`）。
3. 人手把箱子对齐到手/脚暗示的接触点（LightLP 也承认这一步是人工的）。
4. 在 IsaacLab T4 上做 **object-interaction mimic**（BeyondMimic + 全局 root + 接触部位位置奖励 + RSI）。允许手/前臂/膝接触，**不要** terminate on Trunk 直到 mimic 稳定。
5. 成功轨迹存成下一条参考；箱子和运动一起抬 5–10 cm，重复到目标高度。

不要用 OmniRetarget/GMR 的运动当最终 expert（LightLP Table IV）。不要把现有 `motion_source` 当 climb 数据。

**B2. Skill teacher（特权，HeightScan 合同与 Stage E 对齐）**

- Actor：Stage E 同构（proprio + 前向 195D scan + velocity command + prev action）。Mimic 期可以暂时给障碍尺寸/距离，**蒸馏前必须拿掉**，否则 student 不可达（LightLP §IV-B）。
- 先 DAgger 最难 expert，再丢掉 expert、在增强高度范围内加「箱子后方目标」task reward。
- Reset：技能泛化阶段从 loco 站姿开始，不用 RSI（LightLP §V-B2），否则以后接不上走路。
- AMP：climb 专用 discriminator，或系数先降到接近 0，避免平地 AMP 打姿态。
- 验收：各高度 固定 evaluator；成功 = 双脚在箱顶、未倒、未绕行。不能用 reward 曲线代替。

**B3. 与 loco 合并（仅当 B2 过 evaluator）**

抄 LightLP §V-C / Parkour in the Wild 的最小版：

- 一组环境跑冻结的 Stage E loco teacher，一组跑 climb teacher；一个 student，**无 skill one-hot**。
- 再加 transition 组：箱子嵌在可走地形里；过箱后稀疏位置奖励；AMP 在触发点从 loco prior 切到 climb prior。
- 无 transition 组时不要声称「会自己决定爬」。

**B4. Depth student**

沿用已批准的 teacher→depth 蒸馏。爬箱额外需要：中段遮挡（加 history/GRU 或 scan 重建辅助 loss），以及 depth 噪声。这是 Spec 第二阶段的扩展，不是第三条部署合同。

### 6.3 明确不采用

- 把 vault/爬/走一次联训进 Stage E。
- 部署 Actor 加 HeightScan、参考帧、skill label。
- PHP motion matching 或 MGMT 运行时生成器。
- 用现 AMP clip 当 climb prior。
- 只改地形高度、不改接触奖励，指望 PPO 自己用手。

## 7. 建议的验收（Path B）

固定场景，至少：

- 训练分布内：箱高 h、箱深/宽随机（LightLP 用 x∈[0.7,1.0] m，y∈[0.5,1.5] m 量级，T4 按腿长缩小）。
- 未见外形：一个斜面/圆角箱（LightLP 鞍马测试）。
- 负例：命令后退时不得被箱子吸上去（LightLP Table VII command adherence）。
- 证据：evaluator JSON + 连续回放；IsaacLab 与 MuJoCo 同 checkpoint。

## 8. 和当前工作面的关系

现在不该开 Path B 的训练代码。Active slice 仍是 Stage E teacher evaluator。本笔记只回答「以后爬箱子走哪条」：

1. 先把 Path A 的高度写进 Stage E / Stage B 的 terrain+evaluator bucket。
2. Path B 作为 **evaluator 通过之后** 的新 Spec/Plan slice，数据工作（B1）可以与 evaluator 并行准备，但不并入 `t4_loco_teacher` 的接触终止。

# G1 Isaac↔MuJoCo sim2sim gap 配对定位（2026-09-10）

Checkpoint: `g1_vital_motion_v1/model_29999.pt`（Isaac 踏石 d=0 63/64、d=1.0 vx=0.5 71.9%，
`artifacts/eval/g1_vital_v1_m29999/`、`artifacts/eval/g1_vital_v1_hard_slow/`）。

## 方法

同 checkpoint、同命令、同名义初态（无噪声、无 DR、确定性 actor）逐步回放：

- Isaac: `probe_isaac_paired.py`（nubot GPU0，flat、命令钉死、reset 扰动全部钉到名义值）。
- MuJoCo: `probe_mujoco_paired.py` / `probe_mujoco_sparse.py`（本机，复用 play runner，
  对齐 Isaac reset 约定：历史预填首帧、首帧步态相位 [0,0]）。
- 逐步 diff: `diff_paired.py`（1997D obs 通道、动作、接触、解析 PD 力矩与饱和）。

## 结论（按证据强度）

1. **部署管线无 bug**：t0 obs/action 完全一致（max|Δ|≈1.9e-6）。PD/增益/力矩限、
   action scale 0.25、dt 0.005×4、观测顺序与扫描数值全部对齐。
   管线侧仅需一个无害修饰：play 首帧行走相位 [0.38,0.88] vs Isaac 训练首帧 [0,0]（Isaac env 的
   gait_phase 张量 reset 后为 0，_update_gait 在 step 后才加偏移；t≥1 两边完全一致）。
2. **发散从第 1 个物理步开始，在踝/肘/腕**：同目标同初态，20ms 后 ankle Δq≈0.04–0.06 rad、
   某关节 Δdq 峰值 ~7 rad/s；接触时序错开 1–3 步（vx=0.7 首次接触标志错位在 t=1–2，
   vx=1.5 在 t=6）。平地上这些差异是有界的混沌（8s 内 Δz≤1.3cm，两侧行走都稳），策略能吸收。
3. **脚部接触几何两边不同**：Isaac URDF 每脚 4×r5mm 点球 + ankle_pitch STL 网格碰撞；
   MJCF 每脚 7×r10mm 胶囊、无踝网格。换成 URDF 球脚后 hard 踏石摔倒从 122 步推迟到 174 步，
   但不救——是因素之一，不是根因。
4. **扫描约定不是根因**：IsaacLab RayCaster `ray_alignment="yaw"`（射线垂直向下、起点绕
   torso 体 yaw 含腰 yaw）。把 play 的扫描改成 torso 锚定 yaw-only 或全姿态，
   hard 踏石仍摔（133/98 步）。腰 yaw ±15° 造成的扫描远缘 0.2–0.47m 横移真实存在，
   但不是决定性失败原因。
5. **真正的机制是裕度**：d=1.0 石头 pitch 0.54m、宽 0.26m → 落点预算 ±0.09m。
   Isaac 自己也只有 71.9%（64 集里 9 摔 1 掉坑）。MuJoCo 名义出生点的确定性轨迹
   被第 1–2 条里的植物级差异（求解器、脚几何、执行器响应）推出成功盆地：
   先踩上第一排，再带着 ±15° 腰 yaw / 持续漂移的偏航跨第二排时坠入缝坑。
   d≤0.5 预算 ±0.2m 以上，差异被吸收（MuJoCo d=0 10/10 零摔、d=0.5 过 3m 后程摔）。
6. **踝关节常饱和**：平地每步都有 3–8 个关节解析力矩顶限，固定包含双侧 ankle
   pitch/roll（kp=40、effort 25）、shoulder、waist_yaw、wrist。最弱、最影响落点的
   执行器恰是两侧差异被力矩削顶放大的地方。
7. 附带：同一策略 Isaac 平地实际速度 0.51/0.89 m/s（命令 0.7/1.5），MuJoCo 0.63/1.48 —
   跟踪行为本身也有可见差异。

## 基于证据的 DR 配方（热启、窄区间、有门槛）

vital_v2 的教训：冷启动整套 DR（delay 0–2 + 摩擦 0.4–1.2 + 增益 ±10% + 复位速度 ±0.3）
把 tracking_mean 压到 0.44 < 0.5 晋级门 → 等级卡 3.2、踏石归零。所以：

- **热启自 v1 `model_29999`**（能力锚点），不要冷启动。
- **阶段 A（5k）**：仅执行器增益/阻尼缩放 (0.9, 1.1)。
  依据：踝/肩常饱和 + 首步踝瞬态差异 → 练力矩裕度最对症。
- **阶段 B（5k）**：+ 摩擦 static (0.6, 1.2) / dynamic (0.5, 1.0)（MuJoCo μ=1 在区间内，
  比 v2 的 0.4–1.2 收窄以保护 tracking）。
- **阶段 C（可选 5k）**：+ action delay (0, 1)（0–20ms，v2 的一半）。
  delay 是冷启包里最伤的成分，仅在 A/B 守门时加。
- **不加**复位速度 DR（±0.3 那条）：无证据指向初速敏感性，且直接对抗晋级 tracking。
- **每阶段门槛**：Curriculum/episode_tracking_mean ≥ 0.5；踏石 reach_2m ≥ 0.9；
  阶段末跑 MuJoCo `probe_mujoco_sparse.py`（hard 踏石 vx=0.5）作为 KPI：
  目标 reach_2m ≥ 5/10（Isaac 同条件 71.9%），且 d≤0.5 与平地不回归。
- **诚实预期**：求解器级差异（PhysX trimesh 碰撞 vs MuJoCo 凸体）DR 无法覆盖，
  d=1.0 在 MuJoCo 难以完全追平 Isaac。若硬石头是硬指标，走植物对齐路线：
  MJCF 脚改 URDF 球 + 踝网格碰撞代理（已证 122→174 步），叠加阶段 A/B；
  MuJoCo 演示难度建议 d≤0.7。

## 工件清单

- 探针脚本：`probe_isaac_paired.py`、`probe_mujoco_paired.py`、`probe_mujoco_sparse.py`、
  `probe_foot_ablation.py`、`forensic_stones.py`、`diff_paired.py`
- 平地配对：`{isaac,mujoco}_vx{07,15}.npz` + `diff_vx{07,15}/`
- 稀疏行为：`mujoco_sparse_*.json`（d=0/0.5/1.0 踏石与圆桩）
- 消融：`ablation_urdf_feet_stones_d10_vx05.json`；扫描约定消融结果在对话记录
  （torso yaw-only 133 步 / 全姿态 98 步，均未通过）

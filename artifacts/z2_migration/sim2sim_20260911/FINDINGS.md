# Z2 reset_aligned_v1 `model_29999` 评测结论（2026-09-11）

Checkpoint：`TienKung-Lab-z2-reset-20260909/logs/z2_loco_teacher_sparse/2026-09-09_10-33-53_z2_reset_aligned_v1/model_29999.pt`
SHA256 `491ff4c830694f5614da6e407581feafb6d48cf17335c14348f752dd8d0a76b9`（本地与远端核对一致）。
训练：源码 `105164f`，30000 iter 冷启动已完成（1000/2000/3000 节点监控在 `artifacts/z2_migration/formal_v1/monitor/`）。

## Isaac 终评（nubot GPU0，evaluator `locomotion_terrain_perception_v4`，32env/64ep，vx=0.7，确定性 actor）

| 地形 | 难度 | strict 3m | reach2m | 摔/早终止 | 实际前进速度 |
| --- | --- | --- | --- | --- | --- |
| flat | d=0.0 | 64/64 | 64/64 | 0 | 0.559 m/s |
| stepping_stones | d=0.0 | 64/64 | 64/64 | 0 | 0.673 m/s |
| raised_pillars | d=0.0 | 63/64 | 64/64 | 0 | 0.680 m/s |
| stepping_stones | d=0.85 | 60/64 | 62/64 | 3 | 0.710 m/s |
| raised_pillars | d=0.85 | 64/64 | 64/64 | 0 | 0.710 m/s |

- 3000 → 29999 的跃迁：3000 时踏石 reach2m 仅 4/64、15/64 早终止；29999 全地形接近满分、d=0.85 也基本过关。
- 64ep 评测中 0 个回合终态倒退（flat 终态前进 ≥3.89 m）；但 1-env 确定性 20s 视频（seed 42）6 次 reset 多数从侧/后边界 OOB——方向锚定仍是最弱一环（见数据体检）。
- 证据：远端 `artifacts/z2_migration/formal_v1/evaluation/z2/{29999,29999_hard}/`；本地副本 `isaac_29999/`、`isaac_29999_hard/`（JSON+lineage+20s MP4×3，stdout 日志留远端）。

## MuJoCo sim2sim（本机，mujoco 3.11.0，同 checkpoint，PD=`Z2_29DOF_WALK_POSE_DAMPED_PD_CFG`）

| 条件 | 结果 |
| --- | --- |
| flat d=0 vx=0.7，10ep×20s | **10/10 零摔**，全部 reach2m，20s 前进 11.53 m，实际 0.578 m/s |
| stepping_stones d=0 vx=0.7，名义出生点，10ep | 10/10 全同（确定性）：都在 ~4.7 s、x≈2.72 m（约第 3 排石墩）摔倒 |
| stepping_stones d=0 vx=0.7，出生扰动 y±6cm/yaw±6°，10ep | 8/10 摔、9/10 reach2m、2/10 走满 20 s（最远 10.64 m） |

- 平地 Isaac↔MuJoCo 行为一致（0.56 vs 0.58 m/s，均零摔）——部署管线（观测序、PD、action scale、步态相位）没有 bug 级差异。
- 踏石存在明显 Isaac→MuJoCo gap（Isaac 64/64 vs MuJoCo 2/10 存活）。与 G1 2026-09-10 gap 定位（`artifacts/g1_gap_20260910/FINDINGS.md`）同性质：脚几何（Isaac USD convex hull vs MJCF 圆柱脚）、接触时序、执行器瞬态在精确落点任务上被放大。平地差异被吸收，踏石落点预算不够。
- 证据：本目录 JSON（`mujoco_*.json`）+ MP4（`z2_m29999_mujoco_{flat,stones}_vx07.mp4`）。

## Z2 动作数据体检（用户疑虑：T4→Z2 数据是否有问题）

来源：上游 `nubot-zhixing/z2-lab-stable-AMP@c78eb1f` 的 `walk/walk_l/run` 三个 PKL，转 CSV（36D `root_xyz+quat_xyzw+q29`）后在 Isaac 用原版 USD FK 生成 70D 专家。上游仓库未标注 T4 出处；无论源头为何，数值合同与内容体检结果如下。

数值合同（全部干净，无问题）：
- CSV 第 0 帧 == Isaac 专家第 0 帧（差 5.5e-17）→ **CSV 即策略序**，专家头 `JointOrder` == `Z2_29DOF_JOINT_NAMES`。
- 全部关节角在 Z2 MJCF 限位内（源与专家均零超限）；四元数归一（|q|∈[1±2e-16]）；dq 全有限（|dq|≤9.56，p99 3.4–5.6）。
- 独立 URDF FK 对齐 < 1.5e-6 m（生成时已验收）。

内容体检（实质限制，非 bug）：
- **`walk`（2.5 s）与 `run`（1.35 s）是原地片段**：净位移 0.01–0.02 m。AMP 特征是 root 相对（q/dq/hands/feet），不携带全局位移，所以它们只提供"踏步样式"，不提供前进样式；这与上游 64D AMP 无根位移的设计一致。
- **`walk_l`（9.4 s）是唯一真实前进片段**：0.75 m/s，但骨盆偏低（0.66–0.71 vs 训练站立 0.75）且 root yaw ~90°（身体朝 +y 行进，体前方向仍是前进）。
- 对训练的印证：实际速度 0.56–0.68 m/s @ 命令 0.7（样式来源单一）；3000 节点出现倒走 OOB 的 episode（AMP 不惩罚方向，方向只靠速度命令 reward）。
- 结论：**数据没有数值/合同问题；限制是内容性的**——真实前进样式只有 walk_l 一条、无带位移的跑步样式、总时长仅 13.2 s。若后续 Z2 要提速度/跑动，建议补 LAFAN/AMASS → Z2 重定向或采集带位移的 walk/run。

## 验收入口（本机）

```bash
# 交互 viewer（平地/踏石/圆桩；--vx 改命令）
D:/anaconda/envs/pytorch/python.exe legged_lab/scripts/play_t4_sparse_teacher_mujoco.py \
  --robot z2 --checkpoint artifacts/checkpoints/nubot/z2_reset_aligned_v1/model_29999.pt \
  --terrain flat --difficulty 0.0 --vx 0.7

# headless 指标复跑
D:/anaconda/envs/pytorch/python.exe artifacts/z2_migration/sim2sim_20260911/eval_z2_mujoco.py \
  --terrain stepping_stones --episodes 10 --spawn-jitter 7
```

依赖：`D:\anaconda\envs\pytorch`（mujoco 3.11.0 / torch 2.13.0）。29-DoF checkpoint 与 G1 同为 1997D，**必须显式 `--robot z2`**。

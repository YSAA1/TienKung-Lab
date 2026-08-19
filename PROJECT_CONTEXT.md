# T4 工作背景与现状

状态：T4 路线的 living context。入口与货架见 `docs/README.md`。
已批准合同以 `docs/specs/` 为准；当前执行以 `docs/plans/` 里 **仅有的两份** active 计划为准。
若本文与 Spec 冲突，以 Spec 为准。

## 这是什么项目

在 TienKung-Lab 的 PPO + AMP 主干上，给 **T4 27DoF** 做一条统一感知 locomotion：
先特权老师（局部 HeightScan + 速度命令），再蒸馏成只吃机载深度的学生。
走跑老师已经训完。剩下两件事：梅花桩补进老师课表，以及 1m 翻箱技能并最终和走跑合成一条策略。

T4 约 1.4 m 站高，Trunk 在 0.9 m。资产在 `legged_lab/assets/t4/`，关节顺序唯一真值是
`constants.py::T4_JOINT_NAMES`（左臂 7 + 右臂 7 + 腰 yaw + 左腿 6 + 右腿 6）。

迁到本仓库是为了复用已验证的 velocity tracking + PPO + AMP，而不是换一层仿真包装。
总奖励、episode length、checkpoint 存在都不能当能力证明。

## 已经交付

- **Stage E 老师** `t4_loco_teacher`：1155D。站立 / 走 / 慢跑方向、rough、上下楼梯、跨栏地形 bucket。主线 `stage_e_prov7_hurdle`。
- **深度学生** `pi_loco`：`stage_s_head35/model_24999.pt`，楼梯对齐后的部署候选 `model_25746.pt`。
- **AMP 66D** 与人工 motion 复核：17 条 accept，`t4_run` hold out。Provisional expert 仍可用；升 formal 不是当前阻塞。
- **本机 Isaac play**：`t4-isaac-jammy:v2` + Isaac Sim 5.1 + IsaacLab 2.1.0，见 `docs/runbooks/local-isaac-docker.md`。
- **ZL / direct 楼梯 gap**：`vx=0.8` 三次冷启动 3/3；不是 100m 或真机验收。

跨栏是脚上地形，不是手上技能，因此并进 Stage E，不再单独开 lineage。

## 还在推进

1. **梅花桩**（nubot）  
   LightLP §IV 单阶段真洞老师 `t4_loco_teacher_sparse`。s5 列映射已对，但 easy 第一脚踩不上；现行从零 `t_sparse_lightlp_s6`（放宽 easy 顶面/缝/上台高）。
   Actor 1937D（scan 史 ×5 + 接触）；足底 scan 只给 Critic。Stage E 1155D 不改。  
   本切片 **不** 训梅花桩学生。  
   计划：`docs/plans/2026-08-19--t4-sparse-easy-geometry-s6-plan.md`。

2. **翻箱**（zhuoqun）  
   G1 mimic 专家 → G2 同 1155D 观测的 heightscan 技能。  
   G2 学生蒸馏、以及和走跑 / 梅花桩合并成一条策略（Spec 里的 G3 / transition），
   **等 G2 固定 1m 箱能过之后**再开，不在当前 G1/G2 计划里并行发明配方。  
   计划：`docs/plans/2026-08-15--t4-vault-g1-g2-recovery-plan.md`。  
   规格：`docs/specs/2026-08-13--t4-vault-loco-merge.md`。

两条轨道可以同时跑；权威各只有一份计划。

## 明确不做（现在）

- 用遥控 high 档 3 m/s 或 VITAL 草稿 3.5 m/s 冒充已训速度。现行命令 `vx ∈ [-0.6, 1.0]`。
- 覆盖 Stage E 1155D，或把足底 scan / 接触真值写进导出学生。
- 把失败的 S1d、软硬二阶段、旧 25k sparse ckpt 当梅花桩能力。
- 用 TB 曲线代替 evaluator 与回放。
- 100m `rule` 全路线零碰杆验收（独立未开工范围）。

## 数据与 AMP

原始 CSV 在 `legged_lab/envs/t4/datasets/motion_source/`，schema 为
`root_xyz(3) + root_quat_xyzw(4) + q27`，按 30 Hz。
`motion_visualization/` 只供回放。训练 AMP 用 runtime 同定义的 66D
（`q27+dq27+双手相对 root 6+双脚相对 root 6`）。
clip 普遍未对齐地面、接触时序不可用；66D 不含绝对高度，所以不影响当前 expert 数值。
任何消费绝对 root 高度或 motion 接触相位的逻辑都不得引用这些 clip。

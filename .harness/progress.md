# Progress

2026-08-22 以前的 TB 逐窗流水已从本文件删掉，仍在 git 历史。本文件只留能接住当前切片的证据。

## 2026-09-06：同配方重开已停

- 用户指出：探针已证无漏杀、配方没改，再冷启动不会过桩。已杀 tmux `g1-teacher`，GPU1+3 已空。`2026-09-06_10-08-08_g1_sparse_teacher_g1term` 作废，不续。对照仍用 40k `model_39999`。

## 2026-09-06：回放 + 随机策略探针：无 MDP 漏杀；冷启动新 g1term

- 回放：`artifacts/replay/g1_g1term_m39999/`。策略踏石 d=0 16 s：5 次 reset 全是 `collapsed`（tilt 0.85–0.95 rad < 63°，胸/膝 0 N），每段路径 1.1–2.0 m，从未 reach 2 m。
- 单环境 `U(-1,1)` 随机动作：踏石 20 s 19 次 reset（17 collapsed / 1 fall_over / 1 accel+collapsed）；flat 子地形 20 s 20 次 reset。三份 trace 都是 `low_clearance_no_reset_frames=0`，clearance 一掉到 0.20 的**同一帧**就 reset。蹲姿 0.20–0.35 最长只撑 0.08 s。
- 第一次 `--terrain` 省略走 Isaac `plane` 会在 `GetPrimAtPath(None)` 崩；不是 MDP 洞。改用 `--terrain --terrain_types flat`。
- 40k 进程打满后挂在 PhysX「no suitable CUDA GPU」，仍占 GPU1+3。已杀。新冷启动 `2026-09-06_10-08-08_g1_sparse_teacher_g1term`，`resume: false`，不加载 `model_39999`。配方未改：同一 0.20 m collapse。梅花桩 `reach_2m≈0` 是策略/形态，再跑同样 40k 不会自动过桩。

## 2026-09-06：g1term 40k 跑满；连续地形会走，梅花桩不会

- 进程没崩。`max_iterations=40000` 打满，ckpt `model_39999.pt`（logdir `2026-09-04_16-28-27_g1_sparse_teacher_g1term`）。
- 不是 plantfix：`Reset/oob` 0→0.14，`timeout` 不是 0.88，`ep_len` 44→231。也不是 v8：`collapsed` 0.91→0.73，不是钉死 99%。
- 连续地形 `reach_2m` 约 0.73–0.82（flat/boxes/hurdles/stairs/slope/rough/wave）。踏石/圆桩 `reach_2m≈0`，progress 卡在 ~1.5 m，晋级 0。
- 主死因仍是 `collapsed`（73%）。G1 腰可折叠是形态差，不是这趟又把资产训坏。不续 40k 空转。

## 2026-09-04：开 `g1_sparse_teacher_g1term`

- 合同：LightLP 任务终止全留；只加 G1 `collapse_reset_pelvis_above_feet_m=0.20`（相对最低支撑脚）。不换 63°，不清空 `torso_link` 接触。不续 plantfix，不用 v8 的 0.40。
- Composer 2.5 对抗审查 CONDITIONAL（无 Critical）。能力仍要等 evaluator，不以 ep_len 当会走。
- nubot 已冷启动：tmux `g1-teacher`，GPU1+3×2048，logdir `2026-09-04_16-28-27_g1_sparse_teacher_g1term`。

## 2026-09-04：plantfix 混时长；torso≠Trunk

- TB `g1_sparse_teacher_plantfix` @step 6359：`ep_len=877`，`Reset/timeout=horizon=0.88`，`oob=0` 全程，`reach_2m≈0`，`promotion=0`，`timeout_success=0`，`tracking=0.18`，path 1.66 m / radial 0.77 m。`Reset/torso` 从 ~0.91 降到 0.05。
- 现行 ckpt `model_6000` Isaac 踏石 d=0 `vx=0.7` 16 s：零 reset；z 0.766→0.183 @1 s，停在 0.17；path 0.49 m；torso/knee 0 N；tilt mean 22.5° max 42.5°。帧：直立 → 蹲坐在 spawn 格，胸不着地。
- 几何：G1 `torso_collision` cylinder r=0.09 L=0.12 at (0.01,0,0.14)；T4 Trunk box 0.10×0.16×0.34 at (0,0,0.12) 下沿 trunk z−0.05。G1 坐下骨盆 0.17 时胸胶囊最低仍约 0.29 m。
- 回放：`artifacts/replay/g1_plantfix_m6000/`。v6 同姿态对照仍在 `artifacts/replay/g1_v6_m15000/`。

## 2026-09-04：21500 血统核实；G1 回 T4 配方冷启动

- nubot `2026-08-22_14-54-33_t_sparse_lightlp_s12_from_s11b_5k/params/agent.yaml`：`resume: true`，`load_run: s11b_warmstart_src`，`load_checkpoint: model_19000.pt`，`max_iterations: 5000`。ckpt 从 `model_19000.pt` 起到 `21500`。
- S6/S10/S11 计划是 `--resume` 关的混合 LightLP 冷启动。Stage E 1155D 不是 21500 的 parent。
- G1 正式配方撤回 collapse / flat-only / vx 上限 1.0。plant 对表：URDF 脚踝 effort=35 已与 `g1.py` 一致（MJCF 写 50，跟 URDF 不跟 MJCF 力矩）。碰撞仍是 MJCF 同步、站姿 MIMIC。开 `g1_sparse_teacher_plantfix`。

## 2026-09-03 晚：v8 实况 + 终止审核

- nubot v8 @iter 1376 仍在跑（GPU1+3，~2.3 h）。`Reset/collapsed≈99%`，`Reset/torso=0`（pelvis 接触没咬到坐腿上），`ep_len≈77`，平地/踏石/圆桩 `reach_2m=0`，晋级 0。不是会走。
- 终止审核：LightLP 40 m/s²、63°、1 N **不是**按机器人质量标定的；G1 失败漏杀来自接触刚体（T4 `Trunk` ≠ G1 `torso_link`）和蹲坐包络（28°/0 N/accel 24）。v6 回放 12 s 无 reset 仍成立。不要松 40/63。
- v8 的 0.40 m 相对脚高抓住了 v6 那种 t=1.12 s 坐下。5k 前若 collapsed 仍钉死且 tracking 不涨，下一刀是课表（先平地），不是再拧阈值。
- 翻箱：本机连 zhuoqun 超时，无新证据。

## 2026-09-03 G1 蹲坐漏终止；开 v8

- v6 回放：倾角中位 ~28°、max 49°，torso/knee 0 N，12 s 无 reset。LightLP 要 63° 且每步 1% 抽签。
- 修复：G1 `collapse_reset_pelvis_above_feet_m=0.40`（相对脚高，下楼梯不误杀）+ `pelvis` 接触终止。TB `Reset/collapsed`。
- 已杀 v7（~2500，delay 已关但终止未补）。nubot 冷启动 `g1_sparse_teacher_v8`：`logs/g1_loco_teacher_sparse/2026-09-03_18-23-14_g1_sparse_teacher_v8`。TB http://100.100.188.39:8031/#scalars

## 2026-09-03 G1 v6 不会走；冷启动 v7 对齐 T4 plant

- 回放：`artifacts/replay/g1_v6_m15000/`（d=0 踏石 `vx=0.7`，12 s）。t=0 站住，~1.2 s 蹲到 z≈0.17，之后趴着；`mean_speed=0.09`，无 reset。课表 `level_0_frac=64%`、`tracking=0.20`、踏石/圆桩晋级全程 0。
- 对照：T4 稀疏老师 delay **关**、无执行器 DR，且 S12 从 S11b `19000` 热启。G1 v2 把 delay+执行器 DR 绑进冷启动，v6 修碰撞后这两项还在。
- 已杀 v6。nubot 开 `g1_sparse_teacher_v7`：同一 MJCF 碰撞 / MIMIC / AMP / mirror / 稀疏 MDP，`action_delay=False`，无 actuator gain/armature/effort 抖动。

## 2026-09-02 G1 plant v6：MJCF 简化碰撞 + mirror + 重训

- 根因确认：Isaac 用的 `g1_29dof_mode_15.urdf` 原 URDF **24 处 mesh 碰撞** + 脚 **4×5 mm 球**；同仓库 `xmls/g1_actuated.xml` 才是宇树 RL 简化碰撞（visual mesh `contype=0`，collision 胶囊/球，每脚 7 capsule）。
- 修复：`legged_lab/assets/unitree_g1/sync_urdf_collision_from_mjcf.py` 从 `g1_actuated.xml` 写回 URDF（0 mesh / 31 cylinder + 2 sphere）。合同测试 `test_g1_asset_contract.py` 14 passed。
- 配方：`g1_sparse_teacher_v6` = MIMIC 站立姿 + LAFAN AMP + mirror symmetry（`legged_lab/envs/g1/symmetry.py`）。
- nubot 已杀 v4/v5，tmux `g1-teacher` 开 `logs/g1_loco_teacher_sparse/2026-09-02_18-11-09_g1_sparse_teacher_v6`。
- 早期 @iter32（非能力门）：`Reset/accel` **26%**（v4 @1861 仍 44%）；`Reset/torso` **75%**（新主因，早期随机策略）；`ep_len≈53`。accel 门明显改善， locomotion 能力未过。

## 2026-09-02 G1 plant：用宇树 29DoF MIMIC 站立姿，不用 Isaac Lab 自带 G1

- Isaac Lab `G1_CFG` / `G1_MINIMAL_CFG` 是另一台旧 G1（关节名不同），不能接 LAFAN 70D。宇树官方 `g1_29dof_rev_1_0.urdf` 脚碰撞同样是 4×5 mm 球，换官方 URDF 不解穿地。
- 穿地根因是站立角抄了 Isaac Lab 浅蹲（hip=-0.20），配宇树 URDF `z=0.76`。已改成宇树 MIMIC / `g1_actuated.xml` keyframe：`hip=-0.312, knee=0.669, ankle=-0.363`，FK 脚底约 +3 mm。
- 已杀 v3。新 run `g1_sparse_teacher_v4`。不手改碰撞网格。

## 2026-09-02 G1 老师训不起来：不是自碰撞指标，是 plant 穿地 + accel 1s 门

- 现象：`g1_sparse_teacher_v3` @358 仍 `ep_len≈50–55`（= `dt=0.02` × 1.0 s）、`Reset/accel` 53–62%、`Reset/torso` 41–48%、`track_lin_vel`≈0.02。v2 @1190 同样钉在 53。reward 从 −7 收到 −0.4 是短 episode 少积惩罚，不是会走。
- 已证伪：v2/v3 `enabled_self_collisions: false`；`Reset/torso` 不是自碰，是 `torso_link` 对地形 `net_force>1N`（摔倒后胸部着地）。v1 才是自碰：`self_col=true` + `effort_limit_sim=300` → `Reset/torso→0.998`、`ep_len→2.5`。
- 根因：Isaac 用了 URDF **视觉 STL 当碰撞** + 每脚 **4×r=5 mm 球**，站立角 `hip=-0.20/knee=0.42/ankle=-0.23`、骨盆 `z=0.76`。FK：脚球底 **z≈−18 mm（穿地）**。同仓库 MJCF 是胶囊脚、`g1_actuated.xml` 蹲姿 keyframe 脚底 **+3 mm**。穿地弹跳 → LightLP `accel>40` 在 warmup 1 s 后集体 reset。不要靠放开 `LIGHTLP_ACCEL_LIMIT` 假装变长。
- 证据：nubot TB `2026-09-02_15-45-44_g1_sparse_teacher_v3`；本地 URDF FK + STL AABB。v3 不必续到修好 plant。

## 2026-09-02 G1 换上 LAFAN1 走跑 AMP

- 公开源：`lvhaidong/LAFAN1_Retargeting_Dataset`（官方 Unitree HF 已下架）。只用 walk1–4 / run1–2，hold-out sprint。CSV 宽 36 = `xyz + quat_xyzw + q29`，关节序与 `G1_29DOF_JOINT_NAMES` 一致。
- 合同：G1 AMP 70D（q29+dq29+hands6+feet6），T4 仍 66D。专家由 Isaac `G1AmpFeatureBuilder` 生成，不能拿 T4 `motion_amp_expert` 喂 G1。
- 配方：`AmpOnPolicyRunner` + `AMPPPO`，`amp_reward_coef=0.3`，稀疏 tile 上 AMP 仍乘 0（与 T4 LightLP 相同）。新 run `g1_sparse_teacher_v3`，不续 v2。
- nubot 已用 Isaac 生成 6 条 70D 专家（各 899 帧，FK 响应 0.12 m）。已杀 v2，开训 `logs/g1_loco_teacher_sparse/2026-09-02_15-45-44_g1_sparse_teacher_v3`，tmux `g1-teacher`，GPU1+3。不是过桩。

## 2026-09-02 G1 越障老师（不是 walk）

- 用户纠正：要的是现行梅花桩/圆桩越障（`t4_loco_teacher_sparse` MDP），不是 `walk` 平地砂石。
- 任务仍叫 `g1_loco_teacher`，但 env 换成 `T4LocoEnv`，cfg 继承 `T4LocoSparseTeacherEnvCfg`，PPO / 无 AMP，2048×2。
- 错开的 walk run `2026-09-02_13-48-30_g1_teacher_29dof` 已杀，不当老师。
- 现行 run：nubot tmux `g1-teacher`，GPU1+3。第一趟 `2026-09-02_14-05-38_g1_sparse_teacher` 已崩（`Reset/torso`≈1、ep_len≈2），不续。重开 `logs/g1_loco_teacher_sparse/2026-09-02_14-41-32_g1_sparse_teacher_v2`：URDF 力矩、关自碰撞、delay 0–2、执行器 DR。前几 iter 仍是 ep_len≈50，不是立刻 2 步摔。

## 2026-09-02 老师 plant 重训立项

- 对齐回顾：`apply_isaac_pd` + μ=1 足底 box 已在；Stage E 楼梯 MuJoCo 过；稀疏老师 21500 仍 8.5 s / 7.5 s 摔。
- 新计划：`docs/plans/2026-09-02--t4-s12-teacher-plant-retrain-plan.md`。从 21500 热启 5k，只加 delay 0–2 与执行器缩放。

## 2026-09-02 Spec 收口：deploy-only + Isaac 回放

- `model_13999` Isaac hard 过 Spec 门；MuJoCo 稀疏仍 8 s 内摔，用户确认不上真机。
- 本机写出 `artifacts/checkpoints/nubot/s12_repr_first/model_13999_deploy.pt`（4.3 MB，prefix 仅 depth_encoder/memory_s/std/student）。dummy：3168-D → 27-D finite，GRU reset 可复现首步，第二步用记忆。
- 清单：`artifacts/checkpoints/nubot/s12_repr_first/delivery_manifest.json`。
- Isaac 连续回放已落 `artifacts/replay/s12_repr_first_m13999/`（GIF，因 nubot imageio mp4 失败回退）。hard 踏石/圆桩 reset 全是站立 `timeout,oob`，peak ~5 m；easy 圆桩有一次 `accel`。未宣称人工看过。

## 2026-09-01 表示先行 plan+implement

- Spec 按用户「按推荐落 Spec / 赶紧规划实施审核」视为批准。
- 阶段 1 代码合同 review READY。开训时修了两处运行时洞：`log()` 把 builtin `str` 盖掉导致第一 iter 崩；探针写在 `transition.clear()` 之后导致 Recon tag 全空。已重启。
- 正式 run：nubot GPU2×256，tmux `t4-s12-repr-first`，logdir `2026-09-01_09-39-29_s12_repr_first`。`Distill/phase=representation`、iter 100 分层 recon 已出。不是过桩。

## 2026-09-01 knowledge cleanup

- 清根目录 planning-with-files 草稿 `findings.md` / `progress.md` / `task_plan.md`（内容与已否决的「一键监督器」冲突）。
- 清 `artifacts/work/` 一次性 nubot 编排脚本（poll/SSH/tmux 一次性工具）。证据 JSON 仍在 `artifacts/eval/`、`artifacts/diagnostics/`。
- `docs/specs/2026-09-01--t4-s12-repr-first-distill.md` 标 draft，不是 work surface。
- 训练侧多代 AlgCfg（DAgger / Joint / DeployFt / TargetedFt / ResidualFt / PlantFt）未删：测试与 FT CLI 仍钉死；行为清扫交给后续 implement。

## 2026-08-31 合同 A 诊断：13500 评测关深度噪声

- 同一 ckpt、同一 32-ep。噪声开 vs `--disable_student_depth_noise`。
- 踏石 easy 32→31；hard **18→16**。圆桩 easy 30→32；hard **23→25**。
- 未达「关噪声 hard 抬 +8/32」。不是评测期噪声盖住技能。
- JSON：`artifacts/eval/s12_final_main_m13500_noise_off/`。

## 2026-08-31 分层：13500 sim2sim + plant_ft 代码

- `--mode plant_ft` 落地，未开训。focused pytest 当时 `70 passed, 1 skipped`。
- 13500 MuJoCo：踏石 ~3.9 s 摔；圆桩 ~6.2 s 摔。JSON：`artifacts/eval/s12_final_main_m13500_sim2sim/sim2sim_summary.json`。

## 2026-08-31 停 10k 续训与 residual FT

- 杀 `t4-s12-final-main-from14999-10k`。`model_16500.pt` 不作候选。产物锁 `model_13500.pt`。
- 13500 residual FT `model_999` / `model_500` 相对 parent 掉超过 4/32，弃用。
- 14999 residual FT 同样弃用。

## 2026-08-29 S12 最终两命令方案

- 正式 plan：`docs/plans/2026-08-29--t4-s12-student-distill-ft-final-plan.md`。
- `s12_final_main` 训出 `model_13500` / `model_14999`；15k 终点硬稀疏不如 13500。
- Phase B `model_5999.pt` 只作对照，不再当默认 parent。

# Progress

2026-08-22 以前的 TB 逐窗流水已从本文件删掉，仍在 git 历史。本文件只留能接住当前切片的证据。

## 2026-09-11/12：Z2 AMP curated v2 + z2_vital_v31 + B2 G1 学生环境（三任务轮）

- **Z2 AMP 复查不合格**：v1 三段中 `run`/`walk` 为原地踏步（源 root 净位移 0.9–1.1 cm、速度 ~0.01 m/s），仅 `walk_l` 前进（0.76 m/s）；T-pose/哈希/dq 本身合格。证据 `artifacts/z2_amp_recheck_20260912/z2_amp_quantitative.json`。
- **curated v2**：从 pinned 上游 fixture（c78eb1f8）晋升 `run2`（2.99 m/s）/`run_l`（1.07）/`run_140_l`（1.28），加 `walk_l` 组成 4 段（walk:run 类权重 2:1 与 v1 一致）；v1 lineage 与 `z2_loco_teacher` 默认 `amp_expert_files()` 零变化。expert 在 nubot 新树 Isaac FK 生成（4×70D），本地验收哈希/冻结臂/有限性/权重全 PASS（`motion_amp_expert_z2_v2/`，commit `6a3446d`）。
- **subagent 对抗审查修复轮**（PASS with issues→修）：P1 v2 manifest 曾记绝对路径+回退剥 `liyang/` 子目录（远端生成必炸）→ 改仓库相对路径+后缀保留回退+可移植性测试；P2：write_csv 固定 LF 行尾（CSV 哈希机器无关，nubot 首次生成即栽在这上面）、curate 上游候选位置、G1 学生相机外参 DR 对齐 T4、教师门 fail-fast+拒学生 ckpt、测试桩 torso 值、G1/T4 四元数等价测试。506 passed 6 skipped。
- **z2_vital_v31**：plant DR 抽共享层 `locomotion/plant_dr.py`（G1 vital_v3/v31 逐字等价，别名保持），Z2 入口只叠 DR 不动 reset-aligned 奖励/终止；`train.py --z2_motion_experiment`。启动脚本 GPU 参数化（默认 0/2）。
- **nubot 链**：GitHub 直连慢（~50KB/s）→ 全量 git bundle 442M 走 VPN scp（md5 双端一致）+ 增量 bundle×3；新树 `TienKung-Lab-z2-vital-v31-20260912`@9bc1930。GPU 1/3 被他人任务占用 → `z2-v31-watcher` tmux 自动接力：v3（GPU0/2，~19k/30k）结束后 +2min 启动 Z2 v31 30k。
- **B2 G1 深度学生**：`depth_student_contract.py`（纯合同 102/1997 维、相机四元数与 T4 逐位等价、扫描区间）+ `depth_student_env.py`（TiledD455 原生 48×64、torso_link 挂载、外参 DR 同 T4）+ `depth_student_cfg.py`（repr-first 配方镜像 T4 数值）+ `train_g1_sparse_depth_student.py`（1997D 教师门+manifest 对账）；4 项合同测试。Isaac 实启待 B4（教师=v3/v31 优胜者）。

## 2026-09-11：G1 视觉学生主线立项 + vital_v31 三项 DR 落地

- 计划：`docs/plans/2026-09-11--g1-vision-student-and-teacher-v31-plan.md`（brainstorm 收敛：不走盲走档，主轴=G1 深度学生；v3@9500/30000 在训，A1 验收顺延至跑满）。
- B1 done：`docs/runbooks/depth-student-line.md`——S12 线判定（`train_t4_sparse_depth_student{,_ft}` 为现行，旧 Stage E 线只标注）、关键代码地图、换机器人接入清单；README 挂链接。澄清：学生线相机**外参 DR 已存在**（`_apply_camera_extrinsic_jitter`），内参 per-env 不可行（渲染器限制）挂 backlog。
- A2 done：`vital_v31` profile = v3 三事件 + 编码器偏置 ±0.015（`EncoderBiasCfg`，ramp 与 v3 同窗）+ torso COM ±0.05（IsaacLab `randomize_rigid_body_com` startup，NS fallback 双路径）+ 教师特权扫描 RPL 式侧向带遮挡（`legged_lab/locomotion/mdp/scan_occlusion.py`，仅 actor 流，critic 保持特权干净）；`train.py` choices 同步。
- B3 部分：dropout 地形调档（`central_band_column_draw` + `student_depth_spare_lateral_for_sparse`，默认关）——踏石 env dropout 块避开两侧 1/4 边距；ResidualFt-G1/辅助头归 B2/B4。
- 验证：本地 495 passed 6 skipped（含新增遮挡 5 项 + v31 profile 合同 + central band）；black/flake8 干净。默认路径零行为变化（全部旋钮默认关）。
- 审查：用户指定 KIMI K3 外部对抗审查未达成（kimi CLI 缺失 + MOONSHOT_API_KEY 未设；codex 限额至 9/15、claude 认证失败、gemini 未装、grok 挂起终止）；改为内部对抗审查（self，10 审查点全过，无 Critical/Important；2 项 P2：encoder bias env 级路径待 Isaac 启动即验、`legged_lab.mdp` 是否重导出 `randomize_rigid_body_com` 同为启动即验）。任务包保留 `work/review/vital-v31-review-task.md`（gitignored）可补跑。
- 审查修复轮（subagent 只读对抗审查，commit `1e58fc6`）：**P0** IsaacLab 2.1.0 正式版无 `randomize_rigid_body_com`（nubot 树有属版本差异）→ `legged_lab/mdp/events.py` 重导出 `motion_tracking` 本地实现 + 来源静态合同测试；**P1** 遮挡掩码展平方向反了（真实布局 y 外 x 内 `iy*nx+ix`，原 view 成斜条纹）→ `scan_occlusion.py` 重写 + 测试按物理布局断言；P2×4：runbook S12 线渲染尺寸澄清（原生 48×64，270×480 是旧线）、reset 防御 getattr 改直连+补 stub、偏置全程恒定语义入 docstring、计划 A2 草案残留清理 + B4 蒸馏输入一致性备注。修复后 496 passed 6 skipped。教训：teacher 观测几何类改动必须对照 `tests/test_t4_observation_contracts.py` 的扫描布局权威。

## 2026-09-11：项目整合收敛（main+develop 单工作树）

- 测试：本地双环境 488 passed（.gitattributes 强制 LF；manifest 哈希按 LF 刷新；`tests` 真包防 IsaacLab `tests` 遮蔽；mujoco/warp/zl 部署/上游 fixture 缺失主机显式 skip；conftest 引导 sys.path + basetemp 回落）；nubot Isaac python 416 passed。
- 代码：registry `get_cfgs` 深拷贝防引用污染（`tests/test_task_registry_isolation.py` 合同）；恢复稀疏课程地形 depth 组 group=3（缺口分析会话误改 0）。
- 分支：删除已合并 t4-train / g1-portability / z2-teacher / backup；t4-walk 与 origin/dev 以 `archive/*` 标签归档后删除；本地+远端仅剩 `main`+`develop` 同指 `f05b297`；单工作树 `D:/TienKung-Lab`。
- 保全：旧 worktree 未提交/未跟踪 → `artifacts/consolidation/worktree-preserved/`；全引用 bundle → `artifacts/consolidation/refs-backup-20260911.bundle`（均不入 git）。
- 遗留：真实 Isaac sim 级旧新对照 probe 顺延（4 GPU 被 v2/v3 训练占用）；nubot 训练树为冻结源码未动。计划 `docs/plans/2026-09-09--project-consolidation.md` 已全部勾选。

## 2026-09-11：Z2 reset_aligned_v1 训练完成与双仿真器验收

- 训练：30k 跑满，`model_29999` SHA256 `491ff4c830694f5614da6e407581feafb6d48cf17335c14348f752dd8d0a76b9`；本机 `artifacts/checkpoints/nubot/z2_reset_aligned_v1/`（ckpt+agent/env params，与远端哈希一致）。
- Isaac 终评（nubot，32env/64ep，vx=0.7，确定性）：d=0 flat/踏石/圆桩 64/64、64/64、63/64（全 0 摔）；d=0.85 踏石 60/64（3 摔）、圆桩 64/64。3000→29999 踏石 reach2m 4/64→62/64。证据远端 `formal_v1/evaluation/z2/{29999,29999_hard}/`，本地副本 `artifacts/z2_migration/sim2sim_20260911/isaac_29999*/`。
- MuJoCo sim2sim 落地：新增 `legged_lab/assets/z2/mujoco_sim2sim.py`（WALK_POSE_DAMPED PD/摩擦合同，ankle 150Nm/50/4 等）与 `play_t4_sparse_teacher_mujoco.py` 的 `--robot z2` 分支（Z2 MJCF 根体 `Z2_0_Lite_description_0429_1`、扫描体 `waist_roll_link`、按 body 映射脚接触）。29DoF 与 G1 同 1997D，必须显式 `--robot z2`。回归：robot-neutral 14 项、G1 asset 16 项通过；Z2 asset 合同需 `work/upstream-z2` 镜像本机缺（nubot 启动时 42 passed）。
- MuJoCo 结果：平地 d=0 vx=0.7 10/10 零摔、0.578 m/s（与 Isaac 0.559 一致→部署管线无 bug 级差异）；踏石 d=0 名义出生点 10/10 同点摔在第 3 排（x≈2.72m），扰动（y±6cm/yaw±6°）8/10 摔、2/10 走满 20s——Isaac 64/64 vs MuJoCo 2/10 存在踏石 sim2sim gap，与 G1 `artifacts/g1_gap_20260910/` 同性质（脚几何/接触时序/执行器瞬态）。MP4×2 + JSON 见 `artifacts/z2_migration/sim2sim_20260911/`。
- 数据体检（用户疑虑 T4→Z2）：来源为上游 `z2-lab-stable-AMP@c78eb1f` 三个 PKL。数值合同干净（CSV==策略序、expert 帧 0 差 5.5e-17、零超限、四元数归一、dq 有限）；实质限制：`walk`(2.5s)/`run`(1.35s) 原地片段，仅 `walk_l`(9.4s) 真实前进 0.75 m/s（骨盆 0.66–0.71、yaw≈90°）。AMP 特征无全局位移→方向只靠命令 reward，1-env 视频偶发侧/后 OOB。报告 `artifacts/z2_migration/data_check_20260911/motion_data_report.json`。

## 2026-09-11：G1 vital_v3 开训（curated AMP + 端到端 ramp DR）

- 数据：v5 前 30s 含 T-pose 起始（前 60 帧肩外展>0.6rad 占 92–100%）、walk1/3 滑步（proxy p95 1.85/1.06 m/s）；用 `curate_g1_amp.py` 生成 `unitree_v6`（8 段干净步态周期，最早起始帧 2254，proxy p95≤0.36）。视觉 MCP + 本地/远端 pytest + nubot Isaac validate（误差 4.9e-7）三重验收。证据：`artifacts/g1_amp_acceptance_20260911/`（含 v6_all.mp4）。
- 配方：`vital_v3` = vital_v1 奖励/终止 + DR 从第 0 步生效、36000 策略步线性 ramp（摩擦 (0.6,1.0)/(0.4,0.8)→(0.6,1.2)/(0.5,1.0) interval；增益锚 1.0→(0.9,1.1) reset；延迟 P(1步)→50% reset，上限 1 步；无 reset 初速度）。依据：配对回放 FINDINGS + BeamDojo/mjlab/humanoid-gym/IsaacLab 维护者调研。
- 开训：nubot 工作树 `~/phn_ws/t4_train/TienKung-Lab-g1-vital-v3-20260911`，tmux `g1-vital-v3`（GPU0/2，2×2048 env，run `2026-09-11_01-08_vital_motion_v3` 附近），TB `g1-vital-v3-tb` 端口 8051；v2 仍占 GPU1/3（26.5k+/30k）。
- 两次启动失败已修复：set_time_lag int32；`body_names=".*"` resolve 为列表的 covers_all_bodies 守卫。开训核验：env.yaml 三 ramp 事件/delay (0,1)/velocity{}，agent.yaml 8 段 v6，model_0，interval 首触发通过。
- 对抗审查（只读 subagent）：无 blocking；非阻塞建议已落实（docstring、死区测试、脚本工作树断言）。中止线：~3000 iter tracking<0.45 或地形等级落后 v1 轨迹≥2 级。
- Spec `docs/specs/2026-09-11--g1-vital-v3-curated-amp-ramped-dr.md`；Plan `docs/plans/2026-09-11--g1-vital-v3-curated-amp-ramped-dr-plan.md`。

## 2026-09-10：G1 Isaac↔MuJoCo gap 配对定位

- 同 checkpoint（v1 `model_29999`）同命令配对回放：t0 obs/action 对齐（1.9e-6），部署管线无 bug；发散从第 1 物理步的踝/肘瞬态开始，接触时序错 1–3 步；平地差异有界（8s Δz≤1.3cm）。
- 行为：MuJoCo 踏石 d=0 10/10 零摔、d=1.0 名义出生确定性摔（第一排，±0.09m 落点预算被植物差异吃掉）；Isaac 同条件本身 71.9%。脚型（URDF 球替换胶囊）与扫描约定（torso yaw）消融均只能推迟不能救。
- 配方结论：热启自 v1、窄区间分阶段 DR（增益→摩擦→半量 delay），门槛 tracking≥0.5 + 踏石 reach_2m≥0.9 + MuJoCo hard KPI。证据与工件：`artifacts/g1_gap_20260910/FINDINGS.md`。

## 2026-09-10：G1 vital_v2 配方

- Spec `docs/specs/2026-09-10--g1-vital-v2-lafan-dr.md`；计划 `docs/plans/2026-09-10--g1-vital-v2-lafan-dr-plan.md`。
- `vital_v2` = VITAL 终止/步态/action_rate + delay 0–2、摩擦加宽、reset xy/yaw ±0.3、kp/kd 0.9–1.1；AMP 钉 `unitree_v5`。
- 开训：nubot `TienKung-Lab-g1-vital-v2-20260910`，tmux `g1-vital-v2`，run `2026-09-10_01-05-50_vital_motion_v2`，GPU1/3 约 8.5GB，model_0 已保存。保存的 env.yaml 含 delay 0–2、摩擦 0.4–1.2、reset xy/yaw ±0.3、actuator_gains 0.9–1.1、action_rate -0.01；agent.yaml 为 6 段 `unitree_v5`，不是 rob2rob 17 段。

## 2026-09-07：教师算法从机器人配置中分离

- 新工作面：`docs/archive/plans/2026-09-07--robot-neutral-locomotion.md`（已归档）。下面 portable_v1 属于已停止的对照，不能作为当前开训配置。
- 恢复随机重置 0.10/None/None，原 AMP 曲线与稀疏清零。通用算法迁入 `legged_lab/locomotion`，T4/G1 独立 spec 和配方，无跨机器人继承。
- 实际 T4/G1 probe 均采样到 0～9，240 个 finite step，镜像与 reset 前 AMP 快照通过。Actor/Critic/AMP 分别 1937/2016/66 和 1997/2076/70。
- 新机器人接入检查和 21 关节、不同命名的替身通过；核心回归149项、相邻回归90项通过（两组有重叠）。独立审查旧新镜像逐元素一致。
- `g1_lightlp_amp_full_levels_v2` 已在代码 `4fd6ec2` 上启动，GPU 1、3；训练证据在 `artifacts/portability/v2/`。后续本地整理不覆盖正在运行的远端源码。

## 2026-09-07：学生共享运行时与导入边界

- 深度处理与观测组装移入 `locomotion/depth_env.py`；T4 相机/配方留在机器人目录。27/29/21 关节的 FF 与 GRU 拼接及特权隔离测试通过。
- 独立 reviewer 逐方法检查与 cold verifier 的 48 次新旧逐元素对照一致。真实 4-env/24-step RTX probe 保持 3168/1937/96/27，处理后深度去预热后 11 次变化；不代表新机器人的学生能力。
- 真实冷导入复现了共享配置反向触发任务注册的循环。共同配置原样移至 `legged_lab/config.py`，保留旧导出；公共环境与教师配置首次独立导入均通过，且不加载任务注册包。
- 最终 RTX/退出验证通过，tmux `neutral-depth-final` 已结束，GPU 0 回到 47 MiB。Kit 关闭在本机镜像会卡住，探针沿用现有 portability probe 的 30 秒退出兜底；主验证失败码保留。证据在 `artifacts/portability/depth/`。独立结构和冷验证均通过。

## 2026-09-07：动作跟踪与全库收尾

- 共享 motion_tracking：MDP 与 RSL 接口抽离，T4 原始动作逐字段一致，21/27/29 关节重排通过；真实 2-env/51-step 接口验证与退出通过。证据在 `artifacts/portability/tracking/`。
- 当前树 243 个 Python 文件，边界检查 0 违规；原根 313 文件只读清单保留，不覆盖用户源码。评估出生点改用各机器人显式站距/足尺寸，相关 60 项测试通过。
- 全量本机 375 passed、22 failed、1 skipped：失败均依赖缺失的 Linux/ZL 部署环境，冷审查已复现。当前代码整理与新训练启动完成，能力待训后评估。
- 收尾训练 iteration 1042，最近 20 轮随机重置 10.29%，loss 有限，已保存 model_1000；GPU 1、3 持续训练，远端保持 4fd6ec2。

## 2026-09-07：G1 通用性修复，启动有预算的 portable_v1（已停止）

- 工作树/配方：`docs/archive/plans/2026-09-07--g1-teacher-portability.md`。源码提交 `82599f1`；原始根工作区代码未覆盖。
- 旧 `model_39999` 固定 32-env 评估：flat 32/32；d=0 踏石/圆桩均 0/32、全部 collapsed。三份 JSON 在 `artifacts/portability/`。
- 两个可复现代码错误：AMP 终止 transition 混入 reset 后状态；G1 evaluator scan 起点误用 960（实际 1020）、关节索引误用 T4。已修。
- 本次配方试验：按 effort/Kp 标定动作；稀疏速度随 level 从 0.3–1.0 增至 0.6–2.0 m/s；level 0 开始，随机回访限制 0–2，正常晋级仍全覆盖。
- 168 个本机合同通过。Isaac probe：1997-D actor / 2076-D critic / 70-D AMP；116 次终止快照逐项与 reset 前状态一致，且均不同于 reset 后状态。单纯零动作站立未作为通过门。
- 远端 2026-09-07 00:17:43 开 tmux `g1-portable-train`，GPU1+3，各 2048 env，首轮 10000 iterations。之后自动固定评估 5 条件并产出两类 easy 连续回放。当前不宣称过桩。
- `artifacts/portability/lineage.json` 核对 202 个实际部署源码文件及 6 个 AMP 专家的 SHA256，绑定旧基线 checkpoint。新旧动作语义不同，不交叉加载。

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

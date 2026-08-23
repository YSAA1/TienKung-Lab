# Progress

## 2026-08-23 student collapse root repair

- `fixed-v3-nanguard` 已判失败：3305 更新指标先突变，环境回报随后坠落；不是课程、命令、teacher mix、NaN/OOM 或 TensorBoard 断流触发。
- 本地新增 `SafeRecurrentDistillation`：critic/GAE、recurrent sequence minibatch、全局 advantage、KL/behavior transaction gate、policy+Adam rollback、LR 降档、目标梯度诊断与 recon-only conflict projection。
- 非妥协 handoff：PPO 保持 `pg_coef=0.5`；behavior imitation 随 accepted update 从 1→0，teacher mix 也只随 accepted update 衰减。拒绝更新不会提前撤掉教师。
- 新开训支持 `--student_warmstart_checkpoint <pre-collapse model_3000.pt>`：只迁移 CNN/GRU/actor/recon/std；teacher 重新加载，critic/Adam/counters 重置，并写 `student_lineage.json`。
- 提交前深审补强：initial hidden 主动 detach；safe runner 未加载 teacher 时拒绝开训；manifest 采用 exact checkpoint path，并把 teacher/warm-start SHA256 写入 lineage。最终本机 9 文件回归 `83 passed, 2 deselected`，后两项仅缺 ZL vendor MJCF。
- 外部报告复核后新增两条硬合同：每次 Adam step 与 legacy warm-start 后投影 raw std，S12 上限 0.20；recon 进入共享 CNN/GRU 的加权梯度范数最多为 control 的 1.0 倍。focused RED→GREEN 为 `44 passed`。
- nubot 只读刷新：旧 v3 仍到 8134+，四卡各约 18 GiB、20–22% 利用率；TB 8017 存活。`model_3000.pt` SHA256=`df50a89a...332ad5`，raw std min/mean/max=`0.242/0.421/0.515`。尚未停止旧进程、同步或启动新 lineage。

## 2026-08-22 student deploy gate + v3 lineage（后续坍塌，继续训练判断已推翻）

- 当时判断：`fixed-v3-nanguard` 可继续训；该判断已被 2026-08-23 iteration 3305 坍塌证据推翻。
- 落实：Sim2Sim / `build_depth_student_policy` 加载前剥离 teacher 与 scan decoder；`write_deployable_checkpoint` 写瘦身包。学生开训入口强制 1937D sparse teacher，并要求 `--teacher_eval_manifest`，现行 21500 只能走 `--allow_ungated_teacher`。
- `random_level_reset_max_level=None` 写入 S12 学生 lineage：这是继承的老师课表，不是 v3 对照泄漏。
- 根目录 `findings.md` 迁到 `docs/research/2026-08-22--s12-mdp-findings.md`。

## 2026-08-22 knowledge cleanup

- `docs/plans/` 只留 S12 + 翻箱 G1/G2。S6/S11 迁到 `docs/archive/plans/`。根目录 S10 诊断草稿与 `artifacts/tmp_*` scratch 删除。`HANDOFF.md` 改成 S12 快照。

## 2026-08-22 S12 收尾边框 + 40% 轻转

- 诊断：S11b length~300 是 4.25 m OOB，走完踏石掉落是出界不是 fall_over。最后支撑 3.37–3.78 m，晋级/OOB 落在空洞里。
- 现行计划：`docs/plans/2026-08-22--t4-sparse-s12-rim-yaw-student-plan.md`。S11 标 superseded。
- 阶段 1 代码：`T4_SPARSE_RIM_WIDTH=0.75` 进入布局真值、碰撞 mesh 和 algebraic support；轻转 `straight_prob=0.60`；run_name `t_sparse_lightlp_s12_rim_yaw40`。Isaac-free `68 passed`（踏石/命令/列映射/奖励/监控/evaluator）。
- 阶段 3 代码：`T4LocoSparseDepthStudentEnvCfg` 继承 S12 老师 MDP；`DepthStudentTeacherRecurrent`（CNN+GRU、scan recon 训练期、export 丢掉解码器）；DAgger+PPO 的 logπ 记在实际执行动作上。任务 `t4_loco_sparse_depth_student`。pytorch 环境 `test_t4_sparse_depth_student_gru_contract.py` + vault distillation `23 passed`。开训脚本 `legged_lab/scripts/train_t4_sparse_depth_student.py`，等阶段 2 的 10k 门。
- 2026-08-22 14:54 CST：停 S11b，热启 S12。加载 `model_19000.pt`，`--reset_optimizer`，`Learning iteration 19002/24000` 已见。学生不并行，waiter 等 `model_23999.pt`。不把 length→1000 当成功。
- 2026-08-22 第一步侧偏诊断：评估器加 `--spawn_y_offset_m` / `--spawn_yaw_deg`（台上钉死，不出洞）。nubot GPU3 与 S12 并行，ckpt=`S11b model_19000`，d=0.8，`vx=0.8`，32 局。踏石 0/8 cm 均为 32/32 reach_4m、legal first=1.0；圆桩 0 cm 28/32（accel 3 + torso 1）、8 cm 32/32。8 cm 没有更差。JSON：`artifacts/eval/s12_firststep_pin/`。不改课表、不热补 S12。

## 2026-08-21 S11 MDP 修复包

- 交接否决“只改 random level cap”。S11 计划现已归档：`docs/archive/plans/2026-08-21--t4-sparse-s11-mdp-repair-plan.md`。S10 不热补。
- 阶段 1：terrain-aware sparse command、full-level 10% reset、orientation −2、TB promotion/跨 rank 归约。本机合同测试是本阶段证据；开训是阶段 2。
- 2026-08-21 用户覆盖：不加 `body_orientation_l2` / `upright_orientation`（权重 0）；sparse `vx max=2.0`。本机合同 `58 passed`。
- 2026-08-22：S11 vx2 在 ~1.8k 踏石/圆桩仍 ~0.9 m、torso~80%，用户要求停掉重开。新冷启动 `t_sparse_lightlp_s11b_upright_tbslim`：恢复 S10 `upright=+1`（ori 仍 0），砍 TerrainCol/逐地形 reset 原因，TB 控制台只打关键 tag，跨 rank 一次 all_reduce。TB `http://100.100.188.39:8015/`。vx2 日志保留。

## 2026-08-20 S9 capsulefix 正确物理 lineage

- nubot 隔离 worktree `/home/nubot/phn_ws/t4_train/TienKung-Lab-s9-capsulefix` 使用 HEAD `fd84ee3`，worktree clean。主 run 为 `2026-08-20_16-07-04_t_sparse_lightlp_s9_capsulefix`，tmux `t4-sparse-lightlp-s9-capsulefix`，TensorBoard `http://100.100.188.39:8012/`。
- S9 从 s6 `model_31000.pt` warm-start，checkpoint SHA256 `55a459b96665594b9a09f98ac6eb270ed8ebe39c553e066e84a03ee6cfc92d33`；使用 fresh optimizer，不加载 S7/S8。保留 tile-filling grid、真实 `Trunk`/双 `Shank` collision 和 self-collision；`termination_penalty.weight=0.0`，`lin_vel_x=(-0.6, 1.0)`。
- 2026-08-20 16:18 CST 刷新：训练和 TensorBoard 均存活，四卡显存约 7.3–7.6 GiB，日志到 `31143/40000` 附近，未见 Traceback/CUDA/NCCL/OOM/NaN。`Perf/total_fps` 最近 20 点约 43.3k。
- 修复后的关键早期反证：`Episode_Reward/shank_contacts=0`，`undesired_contacts` 最近 20 点约 `-0.015`，远低于污染 lineage 的约 `-1.4〜-1.8`；`Reset/torso≈0.066`，`Reset/accel≈0.556`，`Reset/pit_fall≈0.029`。踏石 easy `reach_2m≈0.463/progress≈2.27 m`，圆桩 easy `reach_2m≈0.400/progress≈2.06 m`。这证明假 Shank-foot 自碰修复已进入实际训练 plant，但仍只是早期 TB 信号。
- 250 iter gate（event step `31322`）已过：`shank_contacts=0`，`undesired_contacts` 最近 20 点约 `-0.014`，`Reset/torso≈0.071`，`Reset/accel≈0.576`；踏石/圆桩 easy `reach_2m` 约 `0.475/0.460`。不停训、不改速度/奖励/AMP，继续到 500 iter；`model_31500.pt` 落盘后用 fixed evaluator + 连续回放判断真实行为。

## 2026-08-20 S8 model_33000 本地复评与 Shank 自碰修复

- S8 已冻结在 `model_33000.pt`，训练 tmux 已停止。checkpoint 已拉到本地，SHA256 为 `472b482924ee82a47045dd9978a87dfc84cf68728982e5e70dcff158ba909831`。
- 修复前 fixed evaluator：flat strict 13/16、reach2 16/16；踏石 strict 0/16、reach2 0/16；圆桩 strict 0/16、reach2 1/16。两类 sparse 均 accel 16/16，torso 为 13/16、11/16。
- 本地正式 Isaac 回放发现 flat 正常姿态左右 Shank 210/210 步持续接触，均值约 20.7/21.8 kN；关闭 self-collision 后全部归零。几何核算锁定为下段 Shank capsule 与同侧脚 collider 重叠约 3 mm。
- 修复：保留 self-collision 和外部 Shank collision，把 URDF 下段 cylinder segment 从 0.28 m 缩为 0.20 m。Isaac capsule 最终外包络恢复为 `z=-0.30..-0.02 m`，匹配 MJCF；资产测试先红后绿。
- 修复后同 checkpoint evaluator：flat/踏石/圆桩 strict 均 0/16；flat reach2 7/16、踏石/圆桩 0/16，三类均 early 16/16。连续回放显示旧策略在去掉假自碰后前扑或失稳，证明 S8 已适应污染 plant，不能续训。
- `termination_penalty=-200` 属于对错误物理的补偿性误诊，已恢复 LightLP sparse 配方的 `weight=0.0`。下一正确 lineage 从 s6 `model_31000.pt` + fresh optimizer 重开，不加载 S7/S8，不先把速度上限改到 2.0。
- 完整记录：`artifacts/diagnostics/s8_model33000_local_replay/summary.md`；修复后 evaluator：`artifacts/eval/s8_model33000_capsulefix_local/`。

## 2026-08-20 s8 termination cost 监控

- S7 grid+collision 训练已停止，保留 checkpoint 和 evaluator 证据。失败模式不是出生台/碰撞盒重叠或速度命令特异误触，而是新增 `Trunk` collision 后 torso reset 变成可达，且 sparse `termination_penalty=0` 让策略在难地形学会提前结束。
- S8 已在 nubot `TienKung-Lab-grid-probe` 启动：tmux `t4-sparse-lightlp-s8-termcost`，logdir `logs/t4_loco_teacher_sparse/2026-08-20_13-15-08_t_sparse_lightlp_s8_grid_collision_termcost`，TensorBoard `http://100.100.188.39:8011/`。从 s6 `model_31000.pt` warm-start，使用 `--reset_optimizer`，`termination_penalty.weight=-200.0`，继续 9000 iter 到累计 40000。
- 2026-08-20 13:26 CST 刷新：四个训练 rank 和 TB 均存活，GPU 0–3 约 42/100/82/42% 利用率、显存约 7.3–7.7 GB；日志到 `Learning iteration 31250/40000`，event 到 step 31260，未见 runtime/NaN 硬错误。当前 checkpoint 仍只有初始 `model_31000.pt`，正常。
- 250 iter gate：不是 S7 的“稀疏地形第 6 步 16/16 torso reset”崩法。`Reset/torso` 最近 20 点均值约 0.348，末值约 0.365；但 `Reset/accel` 最近均值约 0.782，仍很高。踏石 easy `progress≈1.25 m`、`reach_2m≈0`；圆桩 easy `progress≈1.29 m`、`reach_2m≈0`。flat/连续地形已能推进 2–3 m。判定：S8 不该现在停，继续到 500 iter / `model_31500.pt`，但不能把 TB progress 当 sparse 能力声明。
- 500 iter gate：`model_31500.pt` 已落盘。TB event 到 step 31525：`Reset/torso` 最近 20 点约 0.239，`Reset/accel` 约 0.743；踏石 easy `progress≈1.27 m`、`reach_2m≈1.0%`；圆桩 easy `progress≈1.23 m`、`reach_2m≈0.1%`；flat `progress≈2.92 m`、`reach_4m≈43.7%`。判定：termination cost 消除了 S7 的早期 torso 逃逸趋势，但 sparse 仍未学会过。
- `model_31500.pt` fixed evaluator（tmux `t4-s8-eval-m31500`，已退出）产物：`artifacts/eval/s8_model31500_fixed/`。flat d=0.85：strict 16/16，reach2 16/16，reach4 13/16，pit 0，early 0。踏石 d=0：strict 0/16，reach2 0/16，pit 0/16，early 16/16，reset records 为 accel 16/16、torso 8/16，平均 139.3 steps、平均进度 1.49 m。圆桩 d=0：strict 0/16，reach2 1/16，pit 0/16，early 16/16，accel 16/16、torso 8/16，平均 139.2 steps、平均进度 1.48 m。结论：不再是 S7 的第 6 步几乎零进度终止逃逸；当前失败主因转为新碰撞下的高速/姿态稳定性（accel，半数 torso），继续到 1k 门点 `model_32000.pt`。
- 1k 门点：`model_32000.pt` 已落盘，训练继续到约 `32200/40000`，无 runtime/NaN 硬错误。最新 TB 窗口不支持“已经不动 timeout”判断：踏石 easy timeout 约 9.6%、fall 约 90.4%、progress 约 1.34 m、reach2 约 3.6%；圆桩 easy timeout 约 8.7%、fall 约 91.3%、progress 约 1.25 m、reach2 约 1.0%。全局 `Reset/timeout` 从起始约 0.09 升到约 0.28，但 sparse easy 仍主要是 fall/accel，不是站着等 timeout。`Reset/accel` 从起始约 0.835 降到约 0.681，`Reset/torso` 从约 0.428 降到约 0.223。当前命令范围仍是 `lin_vel_x=(-0.6, 1.0)`、`rel_standing_envs=0.2`，不是最大速度 2.0。

## 2026-08-20 s7 grid + collision 组合修复

- 用户批准一次性修两个物理合同：旧 9×9 格点改为按 tile/border/pitch 动态铺满；URDF 为 `Trunk` 添加 0.10×0.16×0.34 m box，为左右 `Shank` 各添加膝部 r=0.05/l=0.10 m 和胫骨 primitive。胫骨最初直接写成 l=0.28 m，后来确认 capsule 转换会扩张外包络，最终合同改为 cylinder segment l=0.20 m，使 capsule 外包络与 MJCF 的 0.28 m 圆柱一致。
- 资产回归先红后绿：修复前 `Trunk` active collision 数为 0；修复后本机相关套件 `108 passed`，`py_compile` 通过。URDF 保留 CRLF，因此 whitespace 检查使用 `git -c core.whitespace=cr-at-eol diff --check`。
- nubot 隔离 asset smoke 已证明强制重转 USD 生效：27 joints / 32 bodies；`Trunk` 实际生成 1 个 box prim，`Shank_Left/Right` 各生成 2 个 capsule prim。
- evaluator 另外暴露了非 0 卡 device bug：App 在 `cuda:3`但 env 仍在 `cuda:0`，reset 跨设备崩溃。已显式对齐 app/env/sim/runner device，并加 `--command_vx` 用于静站 collision smoke。
- `model_31000` 新物理短评：`vx=0` flat 4/4 完整跑满 20 s，`torso/accel/fall=0`，排除默认姿态自碰撞。`vx=0.7` flat 2/4 clean，另 2 局在 281/787 step 因 torso+accel 终止。easy 踏石/圆桩均 0/4 strict、reach2=0，主要是 135–470 step 的 torso/accel，pit=0；说明几何无启动爆炸，但旧策略必须适应新接触。
- 同 `model_31000` 的 grid-only 旧物理矩阵作配对 baseline：easy 踏石 strict 15/32、reach4 27/32、pit 4/32；easy 圆桩 strict 8/32、reach4 28/32、pit 14/32。后续 s7 只能声称组合修复效果，不单独归因 collision。
- 旧 s6 tmux 已停，`model_31000.pt` 用于配对 warm-start，`model_32000.pt` 也已保留。新 tmux `t4-sparse-lightlp-s7-grid-collision`从 31000 追加 9000 iter；主 logdir `2026-08-20_10-02-30_t_sparse_lightlp_s7_grid_collision`，TB `:8010`。分布式 rank 跨秒产生的 `10-02-31` 目录只有 params，主目录持有 events/checkpoint。已看到 `31000–31030/40000`，吞吐约 39–41k steps/s，无 runtime/NaN；新 `undesired_contacts` 不再为假 0，初期约 -1.4〜-1.8。

## 2026-08-19 s6 easy 几何

s5 @17.5k 踏石/圆桩 `reach_2m` 仍 3%/2%、进度 1.1 m。回放第一脚 accel。代码改 easy 踏石 40/10/9 cm、圆桩高 8 cm、随机 level cap=4。pytest 50 passed。nubot 停 s5，从零 `2026-08-19_12-51-05_t_sparse_lightlp_s6`，`resume: false`，TB `:8009`。14:20 CST 刷新：训练/TB 存活，ckpt 到 `model_2000.pt`，step≈2171；两类 easy `reach_2m mean20` 仍约 1.1% / 2.3%，进度约 1.27 m，未到 3k 健康门，不提前停。

14:56 CST 3k 健康门：踏石 easy 通过（`reach_2m mean20≈12.2%`、`progress≈1.40 m`），圆桩 easy 未通过（`reach_2m≈3.1%`、`progress≈1.22 m`）。S6 整体标高风险，但按计划继续到 5k 决策门；未授权不改配方。

16:09 CST 5k 前：step≈4870。踏石 easy `reach_2m mean20≈29.3%`、`progress≈1.61 m`；圆桩 easy `reach_2m≈14.7%`、`progress≈1.33 m`，还没过 20%/1.40，但从 3k 持续上升，不是平台期。用户确认不要机械停；若 5k 仍上升且未触发硬停线，继续到 6k/7k 再判。

16:17 CST 5k 后：踏石 easy 过门（`reach_2m mean20≈29.7%`、`progress≈1.60 m`）；圆桩 easy 未过 reach 门但接近 progress 门（`reach_2m≈15.9%`、`progress≈1.38 m`），仍在上升，未触发硬停线，继续训。hard 曲线全 0 因踏石/圆桩 `hard_episodes=0`；mid 样本很少，只作参考。

16:54 CST 6k：`model_6000.pt` 已出。踏石 easy `reach_2m mean20≈34.7%`、`progress≈1.70 m`；圆桩 easy `reach_2m≈19.0%`、`progress≈1.42 m`，继续上升且未平台。继续到 7k，看圆桩是否稳定越过 20%。

17:37 CST 7k：`model_7000.pt` 已出。踏石 easy `reach_2m mean20≈36.5%`、`progress≈1.72 m`；圆桩 easy `reach_2m≈24.1%`、`progress≈1.49 m`，已稳定过 easy 健康门。继续长训；下一步关注 8k/10k 趋势，并准备固定 evaluator / 回放，不能仅凭 TB 宣称能力。

17:48 CST `model_6000.pt` 固定 evaluator 出结果：flat d=0.85 strict 32/32；踏石 easy reach2 29/32 deterministic、27/32 stochastic，但 strict 0/32；圆桩 easy reach2 23/32，strict 2/32。说明能推进过 2m，但仍会早停/掉洞；只能算行为诊断，不能当能力声明。

18:17 CST 8k：`model_8000.pt` 已出。踏石 easy `reach_2m mean20≈37.0%`、`progress≈1.75 m`；圆桩 easy `reach_2m≈28.3%`、`progress≈1.57 m`，稳定高于 7k。继续到 10k；下一阶段关注 fixed evaluator / 回放里的早停和掉洞是否下降。

17:48 CST evaluator 修复完成并复评 `model_6000.pt`。修复后分开输出 episode 内最大指令方向位移、terminal radial progress 和最终前向位移，不再用 reset 后状态或单一 progress 字段混用；本地 `py_compile` + `tests/test_t4_sparse_evaluator_contract.py` 通过。nubot fixed JSON：flat d=0.85 32/32 strict；踏石 d=0 deterministic 0/32 strict、reach_2m 29/32、坑/掉落 23/32、提前终止 32/32；圆桩 d=0 deterministic 2/32 strict、reach_2m 23/32、坑/掉落 21/32、提前终止 30/32；踏石 stochastic 0/32 strict、reach_2m 27/32、坑/掉落 17/32、提前终止 32/32。结论：evaluator bug 已修，S6 6k 不是不会前进，而是还不会稳定踩顶面/避坑，不能过行为验收。

19:38 CST 深层诊断改写上述结论：

- 修正为“首次离出生垫 touchdown 对应 swing”后，`model_6000` easy 踏石摆高均值 22.1 cm（范围 14.7–33.4 cm），touchdown 8.65 cm，与 9 cm 石顶一致。机器人会主动抬脚；旧 4.8 cm 指标是垫上首次小摆腿。
- 32-seed scan 配对消融 `suite.exit=0`：normal 比 zero/permuted 在踏石平均多 0.428/0.386 m，圆桩多 0.428/0.694 m，95% bootstrap CI 均不跨 0。scan 链路没断，策略在使用空间排列。
- 主根因是旧 9×9 有限格点与 4 m gate 不一致：easy 踏石/圆桩物理支撑只到前向 2.20/2.45 m，之后到 reach4/OOB 是 1.55–2.05 m 深坑。
- 同 `model_6000`、同 seeds、只换 tile-filling grid：踏石 strict 0→13/32、reach4 2→23/32、pit 24→2/32、进度 +1.17 m；圆桩 strict 0→5/32、reach4 4→17/32、pit 23→6/32、进度 +0.62 m。隔离 probe `suite.exit=0`。
- `model_8000` 旧 grid 平衡 evaluator `suite.exit=0`：踏石/ 圆桩 reach2 为 32/32、29/32，摆高均值 23.4/21.4 cm，但 strict 仍 0/32。迭代在改善第一步，不可能修复物理格点提前结束。
- URDF 活跃 collision 只有双脚和球手。s6 @10k 的 `Episode_Reward/undesired_contacts`、`Episode_Reward/shank_contacts`、`Reset/torso` 仍为 0；跨栏的“能过”不能等价于严格无碰撞抬腿。旧 MuJoCo 对照在 difficulty 0.85 / Level 5 均为 strict 0/10，每局 99/45 次栏杆接触。
- AMP 不是本次主因：论文§IV locomotion teacher 是 reward-only PPO，当前 sparse tile 也是显式置零 AMP，不是 runner 漏调。次级差异是 scale=0 后 task reward 仍只有 `0.7×`，以及当前不是论文的完整 5-frame observation / 条件 map encoder；必须等 grid 单变量 lineage 后再分开验证。

处置：S6 不停、不热同步地形，保留为旧 grid baseline。本地 tile-filling grid + evaluator 修复属于下一 lineage；先从 s6 checkpoint warm-start 1k–2k，不同时开 AMP、改奖励、放松 accel 或加碰撞。

19:44 CST 10k 后：`model_10000.pt` 已出，四卡训练和 TB 存活，日志尾部无 runtime 异常。step≈10.2k；踏石 easy `reach_2m mean20≈37.7%`、`progress≈1.77 m`，圆桩 easy `reach_2m≈27.6%`、`progress≈1.58 m`，都仍过健康线。mid 已有少量样本：踏石 mid `reach_2m≈18.2%`、圆桩 mid `≈11.8%`；hard 仍 `episodes=0`，不能当 hard 能力失败。level 4–9 population 合计约 32%，但整体均值仍约 level 2.8，说明课程正在上移但 sparse hard 样本还不足。继续训；下一门建议 12k/15k 刷新，并用 10k 或更高 ckpt 跑 fixed evaluator / 回放确认 pit_fall 和 early termination 是否下降。

19:50 CST 10.3k：S6 旧 grid baseline 继续跑，不停、不热同步 grid。远端到 `10328/40000`，四卡训练 / TB 存活，最新 ckpt `model_10000.pt`，日志尾部无异常。踏石 easy `reach_2m mean20≈39.8–40.4%`、`progress≈1.80–1.83 m`；圆桩 easy `reach_2m≈28.8–30.8%`、`progress≈1.60–1.64 m`，无平台期证据。mid 样本继续增加：踏石 mid `episodes≈779`、`reach_2m≈17.7%`；圆桩 mid `episodes≈490`、`reach_2m≈14.5%`。hard 仍 `episodes=0`，不能判 hard 失败。level 4–9 合计约 32%，整体均值 level≈2.83。鉴于主根因已定位到旧 9×9 grid 与 4m gate 不一致，本轮不启动新的旧 grid evaluator；下一实质动作仍是 tile-filling grid 单变量 lineage。

19:57 CST 10.5k：`model_10500.pt` 已出，日志到 `Learning iteration 10500/40000`，训练/TB/GPU 正常。踏石 easy `reach_2m mean20≈37.9%`、`progress≈1.77 m`，较上一窗口回落但仍高于健康线；圆桩 easy `reach_2m≈31.7%`、`progress≈1.66 m`，继续上行。mid 样本继续增加但仍弱：踏石 mid `episodes≈817`、`reach_2m≈15.7%`；圆桩 mid `episodes≈524`、`reach_2m≈14.7%`。hard 仍 0 样本。`Reset/accel≈63.8%` 略高，`pit_fall≈11.3%` 没明显恶化。继续跑到 12k 门点；S6 仍只当旧 grid baseline，不追加旧 grid evaluator。

20:00 CST 10.6k：尚未到 12k。远端到 `10593/40000`，`model_10500.pt` 存在，四卡训练 / TB / GPU 正常，日志尾部无异常。踏石 easy `reach_2m mean20≈39.6%`、`progress≈1.80 m`；圆桩 easy `reach_2m≈31.6%`、`progress≈1.65 m`，两者健康。mid：踏石 `episodes≈845`、`reach_2m≈15.5%`；圆桩 `episodes≈537`、`reach_2m≈15.7%`。hard 仍无样本。level 4–9 合计约 32%，`Reset/accel≈62.9%` 持平偏高，`pit_fall≈10.7%` 未恶化。12k ETA 约 20:57；下一轮按 12k baseline 快照决定是否继续只监控，或停止旧 grid 训练转 tile-filling lineage。

20:02 CST 本地收口：tile-filling grid、terminal/evaluator 快照和回放终态合同已对齐；修复 `play.py` 仍读取已删除 `last_step_*` 字段的回归。`py_compile` 通过，稀疏/课程/evaluator/MuJoCo 相关 61 测试通过，观测/跨栏 29 测试通过，`git diff --check` 通过。远端 S6 与 TB 仍存活，无 evaluator 残留进程。

20:05 CST 10.7k：TB HTTP 短暂 502/timeout，后续恢复；远端 tmux probe 到 `10710/40000`，四卡训练 / TB 正常，最新 ckpt 仍 `model_10500.pt`，日志尾部无异常。踏石 easy `reach_2m≈41.1%`、`progress≈1.80 m`；圆桩 easy `reach_2m≈29.7%`，仍过健康线。mid 明显改善：踏石 mid `episodes≈896`、`reach_2m≈18.5%`、`progress≈1.43 m`；圆桩 mid `episodes≈566`、`reach_2m≈18.9%`、`progress≈1.41 m`。hard 仍 0 样本。level 4–9 合计约 32.6%。继续到 12k 门点；旧 grid baseline 未平台，但下一实质路线仍是 tile-filling grid。

20:09 CST 10.8k：远端到 `10825/40000`，四卡训练 / TB 正常，最新 ckpt 仍 `model_10500.pt`，日志尾部无异常。踏石 easy `reach_2m≈38.8%`、`progress≈1.74 m`，回落但仍过健康线；圆桩 easy `reach_2m≈29.8%`、`progress≈1.61 m`，保持健康。mid 继续改善：踏石 mid `episodes≈932`、`reach_2m≈21.4%`、`progress≈1.47 m`；圆桩 mid `episodes≈587`、`reach_2m≈19.5%`、`progress≈1.41 m`。hard 仍 0 样本。level 4–9 合计约 33.8%，课程没有塌。`accel≈60.7%`、`pit_fall≈9–10%` 不恶化。继续到 12k；12k 仍作为旧 baseline 停止/转 tile-filling lineage 决策点。

20:14 CST 10.95k：远端到 `10940/40000`，四卡训练 / TB 正常，最新 ckpt 仍 `model_10500.pt`，日志尾部无异常。踏石 easy `reach_2m≈38.4%`、`progress≈1.75 m`，仍过健康线；圆桩 easy `reach_2m≈31.6%`、`progress≈1.62 m`，保持健康。mid 稳住：踏石 `episodes≈959`、`reach_2m≈22.3%`、`progress≈1.47 m`；圆桩 `episodes≈605`、`reach_2m≈19.3%`、`progress≈1.42 m`。hard 仍 0 样本。level 4–9 合计约 33.3%，均值 level≈2.89。无 runtime/optimization 硬停信号；12k 预计约 20:55，届时决策是否停旧 baseline 转 tile-filling lineage。

20:18 CST 11.0k：远端到 `11049/40000`，`model_11000.pt` 已落盘，四卡训练 / TB 正常，日志尾部无异常。踏石 easy `reach_2m≈38.7%`、`progress≈1.78 m`；圆桩 easy `reach_2m≈29.7%`、`progress≈1.59 m`，仍过健康线。mid 稳定：踏石 `episodes≈998`、`reach_2m≈22.4%`、`progress≈1.46 m`；圆桩 `episodes≈620`、`reach_2m≈18.9%`、`progress≈1.43 m`。hard 仍 0 样本。level 4–9 合计约 31.3%，均值 level≈2.82。无硬停信号；继续到 12k 门点。

20:22 CST 11.15k：远端到 `11145/40000`，`model_11000.pt` 存在，四卡训练 / TB 正常，日志尾部无异常。踏石 easy `reach_2m≈38.8%`、`progress≈1.77 m`；圆桩 easy `reach_2m≈29.0%`、`progress≈1.59 m`，仍过健康线。mid：踏石 `episodes≈1017`、`reach_2m≈22.8%`、`progress≈1.45 m`；圆桩 `episodes≈625`、`reach_2m≈18.5%`、`progress≈1.43 m`。hard 仍 0 样本。level 4–9 合计约 31.2%。无硬停信号；12k 后建议不要用旧 grid 做能力验收，主线切到 tile-filling grid。

20:28 CST 11.3k：远端到 `11292/40000`，`model_11000.pt` 仍最新，四卡训练 / TB 正常，日志尾部无异常。踏石 easy `reach_2m≈39.4%`、`progress≈1.78 m`；圆桩 easy `reach_2m≈28.7%`、`progress≈1.60 m`，仍过健康线。mid：踏石 `episodes≈1044`、`reach_2m≈23.4%`、`progress≈1.46 m`；圆桩 `episodes≈638`、`reach_2m≈17.5%`、`progress≈1.40 m`。hard 仍 0 样本。level 4–9 合计约 32.9%，课程没有塌回低难度；10% random reset 固定抽 level 0–3，不随当前课程上浮。无 runtime / optimization 硬停信号；12k ETA 约 20:55。

20:56 CST 12k：`model_12000.pt` 已落盘，远端到 `12022/40000`，四卡训练 / TB 正常，日志尾部无异常。踏石 easy `reach_2m≈41.2%`、`progress≈1.81 m`，较前窗上升；圆桩 easy `reach_2m≈30.0%`、`progress≈1.63 m`，较前窗略回落但仍健康。mid：踏石 `episodes≈1272`、`reach_2m≈19.9%`、`progress≈1.45 m`；圆桩 `episodes≈798`、`reach_2m≈13.3%`、`progress≈1.27 m`，圆桩 mid 偏弱。hard 仍 0 样本。level 4–9 合计约 33.3%，课程整体仍上移。`accel≈61.5%`、`pit_fall≈10.5%`，value loss finite，entropy/noise 未塌。结论：S6 没有硬停信号，但旧 grid baseline 信息增量下降；建议授权后冻结 `model_12000.pt`、停旧 baseline，转 tile-filling grid 单变量 lineage。

21:16 CST 12.5k：`model_12500.pt` 已落盘，远端到 `12533/40000`，四卡训练 / TB 正常，日志尾部无异常。踏石 easy `reach_2m≈42.2%`、`progress≈1.85 m`；圆桩 easy `reach_2m≈33.8%`、`progress≈1.69 m`，两者均高于 12k。mid：踏石 `episodes≈1427`、`reach_2m≈20.7%`、`progress≈1.44 m`，基本横盘；圆桩 `episodes≈1002`、`reach_2m≈20.7%`、`progress≈1.49 m`，12k 低点恢复。hard：踏石仍 0 样本；圆桩 hard 仅约 3 个 episode，不能判能力。level 4–9 合计约 32.1%，课程没有塌。`accel≈60.4%`、`pit_fall≈13.2%`，value loss finite，entropy/noise 未塌。结论：无硬停，但旧 grid baseline 的下一步仍应是冻结后转 tile-filling grid。

21:36 CST 13k：`model_13000.pt` 已落盘，远端到 `13033/40000`，四卡训练 / TB 正常，日志尾部无异常。踏石 easy `reach_2m≈42.6%`、`progress≈1.84 m`；圆桩 easy `reach_2m≈34.7%`、`progress≈1.74 m`，easy 继续无平台。mid：踏石 `episodes≈1654`、`reach_2m≈24.2%`、`progress≈1.40 m`，reach2 上升但 progress 未同步上升；圆桩 `episodes≈1168`、`reach_2m≈18.9%`、`progress≈1.45 m`。hard：踏石仍 0 样本；圆桩 hard 仍约 3 个 episode，不能判能力。level 4–9 合计约 34.5%，课程略上移。`accel≈58.4%`、`pit_fall≈10.8%`，value loss finite，entropy/noise 未塌。结论不变：旧 grid baseline 继续跑无硬停，但下一实质路线仍是 tile-filling grid。

21:55 CST 13.5k：`model_13500.pt` 已落盘，远端到 `13528/40000`，四卡训练 / TB 正常，日志尾部无异常。踏石 easy `reach_2m≈43.9%`、`progress≈1.85 m`；圆桩 easy `reach_2m≈33.1%`、`progress≈1.70 m`，较 13k 回落但仍健康。mid：踏石 `episodes≈1837`、`reach_2m≈22.6%`、`progress≈1.42 m`，从高点回落；圆桩 `episodes≈1309`、`reach_2m≈24.6%`、`progress≈1.46 m`，继续改善。hard：踏石仍 0 样本；圆桩 hard 仍约 3 个 episode，不能判能力。level 4–9 合计约 32.4%，课程回落但未塌。`accel≈61.8%`、`pit_fall≈12.7%`，value loss finite，entropy/noise 未塌。结论：无硬停；旧 grid baseline 边际信息下降，下一步仍应转 tile-filling grid。

22:54 CST 15k：14k 附近本地访问链路多次 TB 502 / SSH timeout，Tailscale 经 DERP(hkg) 中继；22:16 后恢复，训练未中断。`model_15000.pt` 已落盘，远端到 `15034/40000`，四卡训练 / TB 正常，日志尾部无异常。踏石 easy `reach_2m≈43.6%`、`progress≈1.87 m`，高位横盘；圆桩 easy `reach_2m≈35.3%`、`progress≈1.74 m`。mid：踏石 `episodes≈2472`、`reach_2m≈25.5%`、`progress≈1.58 m`；圆桩 `episodes≈1821`、`reach_2m≈26.3%`、`progress≈1.52 m`，比 13.5k 更稳。hard：踏石仍 0 样本；圆桩 hard 仍约 3 个 episode，不能判能力。level 4–9 合计约 34.4%，课程略上移。`accel≈59.6%`、`pit_fall≈12.9%`，value loss finite，entropy/noise 未塌。结论：无硬停、mid 信号增强，但旧 grid 合同 bug 仍是下一步主线。

## 2026-08-19 cleanup

删除根目录审计草稿 `findings.md` / `progress.md` / `task_plan.md`（内容已进现行计划与 `.harness`）。s4 计划迁到 `docs/archive/plans/`。TB 抓数 scratch 不入库。

## 2026-08-19 开训 s5

停 s4 tmux，ckpt 留到 `model_12500.pt`。nubot 工作区覆盖了列映射/MDP 源码后从零启动 `t_sparse_lightlp_s5`。logdir `logs/t4_loco_teacher_sparse/2026-08-19_00-20-19_t_sparse_lightlp_s5`，无 load ckpt。TB `:8008`。

## 2026-08-18 阶段 1–2 代码落地

列映射 helper `terrain_columns.py`；`t4_env` / `rewards` 走列号；TB 按真名聚合并打 `TerrainCol/*/occupancy` 与 `Curriculum/level_*_frac`。LightLP：Eq.4 求和、Eq.5 泄漏积分、opposite 点积、10% 随机 level、路径长度晋级。审查后又修了两处：play 单类型 `curriculum=False` 时全列绑定；horizon 超时重抽命令不再翻 standing/moving。相关 pytest 74 passed / 1 skipped。未开 s5、未停 s4。

## 2026-08-18 s4 列号审计与 TB 重读

IsaacLab 2.1.0 `terrain_types` 是 20 列列号。s4 代码用 13 个 `sub_terrains` 下标，TB 名整体错位。nubot EventAccumulator @ iter 12018：连续地形 reach_2m 0.70–0.77、progress 3.3–3.5 m；四列踏石与三列可见圆桩 reach_2m 0.02–0.05、progress 1.17–1.30 m。他人「踏石 71%」实为跨栏列。执行面当时改到列号修复计划（现已归档 `docs/archive/plans/2026-08-18--t4-sparse-terrain-index-fix-plan.md`）。

## 2026-08-18 文档收口与 s4

入口改为 `docs/README.md`。S1d / rollback / 软硬 v4 计划已进 `docs/archive/plans/`。现行训练 `t_sparse_lightlp_s4`（nubot 四卡，真洞从零）。翻箱仍走 G1/G2 recovery。

## 2026-08-17 梅花桩完整补全改从零

用户裁定：不续训；不做短 FT；踏石方砖太大必须改；sparse Actor 1155D 可打破。执行面改为 `docs/plans/2026-08-17--t4-sparse-lightlp-complete-plan.md`。默认 Stage E 1155D 不动。


## 2026-08-17 S1d 失败抽检

原始表：`artifacts/eval/t4_sparse_ab_tb_dump.json`。nubot A/B 当时仍在跑，ckpt ≥ `model_11000.pt`。

3.5k 对齐：v2 reward 57.0 / 踏石 0.75；稳定组 48.1 / 0.76；对照组 43.6 / 0.75。圆桩三条 success 都是 0、progress ≈1.1 m。

末值：稳定组 11629 reward 56.4 踏石 0.81/success 0.45 圆桩 0.019/success 0；对照组 11795 reward 57.5 踏石 0.79 圆桩 ≈0。`legal_foothold` 稳定组 4e-6。

## 2026-08-17 S1d 回退与续训

- 代码已去掉 stable/allin、双 Critic、脚下 scan、legal_foothold、稀疏 gait/AMP/stumble 缩放。
- 本机合同 `42 passed`。
- nubot A/B tmux 已停，ckpt 保留。
- 续训 tmux `t4-sparse-v2-resume`，CUDA 0,1，任务 `t4_loco_teacher_sparse`，从 `2026-08-16_13-18-07_t_compat_sparse_v2/model_3500.pt` 加载，run `2026-08-17_01-50-07_t_compat_sparse_v2_resume`，日志 `logs/t4-sparse-v2-resume.log`。已见 PPO 从 3500+ 继续。

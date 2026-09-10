# Z2 29DoF 教师迁移与早期训练验收

Status: active。唯一执行工作树 `D:/TienKung-Lab-z2-teacher-20260909`，分支 `z2-teacher-20260909`。
基线 `97b70e7`（包含 G1 核心修复）；来源 `https://github.com/nubot-zhixing/z2-lab-stable-AMP`，固定提交 `c78eb1f8e31b7f7872733110c10276b7b2159414`。

## 目标与分工

完整迁入上游 Z2 29DoF 物理资产、参数与可追溯原配置，并通过机器人独立 spec 接入本项目共享 LightLP/AMP 教师。
规划与最终审核由主代理负责；实施由外部 Grok `grok-4.6`、`xhigh` 执行；另派只读对抗审查 agent。
用户指定的 harness-workflow:plan 在已安装技能路径未找到，沿用项目规格/计划结构由主代理直接规划，不安装或冒称使用该技能。
不修改原工作树和 G1 运行源，不中止其他训练，不静默替换用户指定模型。

## 1. 来源与模型识别

- 从固定上游提交识别真正 29 个受控自由度的 Z2 版本；检查可动/固定/辅助关节和实际加载模型，而非按文件名判断。
- 保存原始 URDF/USD/MJCF、全部引用 meshes/materials、许可证、资产路径和 SHA256 清单。检测 LFS 指针、缺失依赖。
- 建立上游到目标逐项对照：质量/惯量/碰撞、关节轴和顺序/限位/速度/力矩、站姿/根高度、驱动/Kp/Kd/armature/friction、动作单位/缩放/延迟/控制步长、自碰撞/solver/import flags。
- 保留上游任务和训练配置原件；适配 LightLP 的算法差异必须显式记录，不把 G1 身体参数复制为 Z2 参数。

## 2. 实施范围

- `legged_lab/assets/z2/` 定义资产、唯一关节序、schema 与 `LocomotionRobotSpec`；`legged_lab/envs/z2/` 独立组合共享教师；注册 `z2_loco_teacher`。
- 从真实几何/FK确定脚/手 AMP site、足底尺寸、站距、根/躯干与碰撞语义；镜像用轴与 FK 验证，不能只按名称猜符号。
- 核对上游专家 schema、关节序、四元数约定、坐标系、fps/qvel、site 和特征宽度。不能把同为 29DoF 的 G1 motion/70D 文件直接当 Z2 expert。
- 必要时从上游 Z2 motion 通过共享 AmpFeatureBuilder 生成目标专家，保存原始数据、转换程序、权重、帧数和哈希 lineage。用相同姿态真实 Isaac FK 对齐运行时特征。
- 共享代码只为已证实的机器人中立缺口作最小修正，保持 G1/T4 核心修复，不通过放宽终止或改进度口径掩盖失败。

## 3. 开训前证据

- 针对性测试覆盖参数与来源一致性、29D 控制映射、专家重排/有限值/帧时、镜像 FK 和错误模型/数据拒绝；运行受影响 G1/T4/共享合同回归及边界扫描。
- 真实 Isaac 5.1 + IsaacLab 2.1 环境、tmux 下验证 articulation、每关节控制脉冲、站姿足底/自碰撞、重置、Actor/Critic/AMP 形状和有限值、专家 FK、左右镜像、课程/终止与恢复。
- 主代理审核实际 diff/源文件和证据，独立 agent 对抗检查同一提交与上游。重要问题由原 Grok 会话修复后复核；静态全绿不能替代真实仿真。
- 生成包含代码/资产/专家/运行环境/完整命令的 lineage；验证通过的相关改动以简洁中文提交。

## 4. 正式训练与完成标准

- nubot 独立远端工作目录，使用 `scripts/nubot_run.sh`，tmux 启动。启动前再次确认 GPU；当前 GPU0/2 可用，1/3 有 G1，不能把此次快照视为资源保留。
- 冷启动独立 Z2 30k lineage（沿用共享教师规模，最终 GPU/环境数由真实探针资源结果确定）；不直接续 G1 checkpoint。保存参数、日志、PID/tmux 和初始 checkpoint。
- 每 1000 iter 监控一次，计划检查点 1000/2000/3000。按实际速度安排等待，不高频重复抓 TensorBoard，不因为观察超时重启进程。
- 每节点保存 loss/finite、速度跟踪、OOB 与其他终止计数、reach2m/路线进展、地形等级/课程；核对分母和指标语义，OOB 非零本身不代表成功。
- 相同固定条件下对照当前 G1（固定 checkpoint、命令/terrain/seed/episode budget），保存 evaluator JSON 与连续回放，并与 lineage 绑定。至少实证 reach2m 非零、OOB 非零且无站桩/持续坍塌/数值爆炸，说明量化差距，不宣称完全等同 G1。
- 如果几千 iter 后仍失败，依据行为与物理/配置证据定位迁移缺陷或学习差异，不能以“已开训”缩减完成标准。

## 恢复入口

实施材料和执行日志：`work/z2-migration/`；正式验收/来源清单：`artifacts/z2_migration/`。
当前：迁移代码、原始USD物理参数、393帧完整AMP专家与对抗审核通过，已冷启动正式训练。

- 训练源码：`933e08a03f32beec85b8b968cb42ec56354475c9`。启动前618个运行文件和全部专家哈希通过。
- 原始USD为30body/29DoF；重新导入URDF会出现额外固定neck刚体，故正式任务直接使用原始USD。质量/COM/惯量/PD/限幅/armature/硬限位/friction对照通过，见`artifacts/z2_migration/migration_review.md`。
- nubot独立目录：`/home/nubot/phn_ws/t4_train/TienKung-Lab-z2-teacher-20260909`。tmux `z2-teacher-v1-20260909`，GPU0/2，2048env/rank、4096全局，seed42、30k、resume=false。
- 主run：`logs/z2_loco_teacher_sparse/2026-09-09_03-45-12_z2_source_usd_teacher_v1`；`03-45-13`为另一rank的元数据目录，不用它找checkpoint。
- 启动检查：iter151全部已查loss有限、实际配置和4个相关进程已记录，见`artifacts/z2_migration/formal_v1/startup_verified.json`。这不是行为验收。
- 监控tmux：`z2-early-monitor-20260909`。只在1000/2000/3000checkpoint稳定且标量到达后写`formal_v1/monitor/iteration_<n>.json`，包含最近100iter窗口及checkpoint SHA。1000已完成：OOB/reach2m为0；同条件evaluator与20秒连续MP4确认倾倒，见`artifacts/z2_migration/formal_v1/iteration_1000_review.md`。保持原训练，自动评估批次`z2-milestone-evidence-20260909`等待2000/3000；尚未达到完成标准。
- 不再改动正在训练的远端源码。后续若发现需要改配方/代码的问题，先据证据定位，并保存新lineage；不能把“已开训”当作完整目标完成。

### 2000节点（基线历史）

见`artifacts/z2_migration/formal_v1/iteration_2000_review.md`：Z2训练OOB/reach2m仍为0；确定性平地64回合均到时限但最大前向进度均值仅0.116m，踏石0/64。4个G1/Z2同条件JSON及lineage全部完成并验证。保持原训练到3000，同协议视频完成后决定下一lineage。若仍失败，优先仅验证动作变化权重-0.1到-0.01，保持资产/PD/专家/终止等合同；目前未实施，不将候选当已验证修复。

### 历史：动作变化代价单变量候选（用户已停止）

基线933e08a的1k/2k/3k监控、全部固定评估和4个20秒视频已完成，均未达到行走标准，已主动停训并保留全部工件，见`artifacts/z2_migration/formal_v1/iteration_3000_review.md`及停止记录。

- 候选源码7ca0c7678c08ec45dfd79d249626779a3b7bd9dd，仅Z2动作变化权重-0.1→-0.01。Grok4.6/xhigh实施，31项合同测试/格式检查和独立差异审核通过；不声称因果修复已验证。
- 新远端`/home/nubot/phn_ws/t4_train/TienKung-Lab-z2-actionrate-20260909`。从冻结基线复制618个运行文件；仅teacher_cfg.py不同，完整专家哈希通过。旧目录未改动。
- 30k冷启动，GPU0/2，2048env/rank，seed42，resume=false；tmux`z2-actionrate-v1-20260909`。
- canonical run为`logs/z2_loco_teacher_sparse/2026-09-09_07-30-59_z2_actionrate_001_v1`。启动实际配置确认-0.01、加速度终止true、gait gate true；完整env.yaml只差此权重，agent.yaml只差run_name（仅归一化目录前缀）。startup loss81步有限仅证明数值启动。
- 本地证据`artifacts/z2_migration/action_rate_v1/`；远端沿用新目录内`artifacts/z2_migration/formal_v1/`，不要与旧基线目录混淆。
- 监控tmux`z2-actionrate-monitor-20260909`，评估tmux`z2-actionrate-evidence-20260909`；仍在1000/2000/3000节点固定评估，3000连续回放。复用已冻结G1参考JSON/视频，不重复运行相同G1条件。
- 完成标准不变：实际非零OOB/reach2m、有持续行走而非站桩/持续倾倒，全部由节点JSON、lineage和连续回放证明。当前只是已启动候选，总体目标未完成。

### 历史：三项VITAL训练（已按用户要求停止并补齐配置）

用户要求直接采用三项G1 VITAL设置：动作变化权重-0.01、加速度终止false、gait跟踪门控false。已停止单变量训练，保留旧数据；源码d6bef33在nubot GPU0/2正式冷启动30000 iter，4096环境，tmux `z2-vital-formal-20260909`。实际保存配置三项均验证，model_0已生成；按1000/2000/3000监控，不再执行逐项对比实验。行为尚未验收。

恢复以 `artifacts/z2_migration/vital_formal_v1/lineage.json` 为准；旧单变量和基线均不得重启。

### 当前执行：reset与接触奖励补齐后重启（已完成30000轮并验收）

当前已补齐G1倾倒/塌陷重置与接触奖励：roll/pitch=(0.8,1.0)rad、塌陷高度0.20m/持续0.20s/遵守碰撞免疫、接触权重-1；保留action rate -0.01、加速度终止false、gait gate false。源码105164f，nubot GPU0/2冷启动30000 iter、4096环境，tmux `z2-reset-formal-20260909`。八项实际配置与G1对照一致，model_0已生成。1000/2000/3000监控，无逐项对比实验；行为待验收。

原三项运行1000迭代确定性平地64回合均站桩；当前修改经用户明确授权。旧运行数据保留。阈值核验、启动配置和运行信息见 `artifacts/z2_migration/reset_aligned_v1/`。

### 2026-09-11：训练完成与双仿真器终评（现行状态）

训练已跑满30000轮，`model_29999.pt` SHA256 `491ff4c830694f5614da6e407581feafb6d48cf17335c14348f752dd8d0a76b9`；本机副本 `artifacts/checkpoints/nubot/z2_reset_aligned_v1/`（含params）。

- Isaac终评（远端`formal_v1/evaluation/z2/{29999,29999_hard}/`）：d=0 flat/踏石/圆桩 64/64、64/64、63/64全部0摔；d=0.85踏石60/64（3摔）、圆桩64/64。3000→29999踏石从4/64 reach2m跃迁到62/64。
- MuJoCo sim2sim（本机`artifacts/z2_migration/sim2sim_20260911/`）：平地10/10零摔（0.58m/s）；踏石d=0名义出生点确定性摔在第3排、带扰动2/10存活——存在Isaac→MuJoCo踏石gap，与G1 gap定位同性质。验收入口与结论见该目录`FINDINGS.md`。
- 本机sim2sim代码合同：`legged_lab/assets/z2/mujoco_sim2sim.py`（WALK_POSE_DAMPED PD）+ `play_t4_sparse_teacher_mujoco.py --robot z2`。29DoF与G1同1997D，必须显式`--robot z2`。
- 动作数据体检（`artifacts/z2_migration/data_check_20260911/`）：数值合同全部干净（CSV即策略序、零超限、四元数归一）；限制是`walk`/`run`为原地片段、仅`walk_l`真实前进（0.75m/s）——AMP无全局位移故方向/速度样式受限，3000节点曾出现倒走OOB。若要提速度需补带位移数据。
- 训练tmux已结束；nubot GPU0/2已被G1 vital_v3接管。

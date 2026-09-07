# 机器人无关的运动训练算法整理

Status: v4严重崩塌，已停止并保留model_0/model_500；正在改为宇树官方纯29DoF无灵巧手资产并验证训练。目标未完成。

## 当前修复：纯29DoF与训练有效性（2026-09-07）

- 用户目标：G1能正常训练，训练及行为表现至少与同类型T4接近；有限loss、短probe、checkpoint存在均不能证明完成。
- 用户最新明确：只用官方无灵巧手29DoF配置，不叠加多变量补偿实验。此前准备的body/scaled/scaled_rate训练未启动，停止推进该方案。
- 实时失败证据：v4第485轮最近20轮平均回合5.84步，collapsed约96.10%；model_500在64个平地环境中反复每5步收腿触发collapsed。零动作约54步、标准随机动作约44步，表明已学到提前终止行为。
- 已确认错配：官方IsaacLab locomanipulation资产实际43关节，14个手指不进策略却进入限位惩罚，其中10个默认关节角在收缩软限位外；上半身刚度3000/5000与统一0.5探索导致巨大运动惩罚。局部修正探针仅用于归因，不作为新训练配方。
- 正确来源：Unitree `unitree_rl_lab`提交`4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`的`UNITREE_G1_29DOF_CFG`和`Unitree-G1-29dof-Velocity`；配套`unitree_ros`提交`7d6075f7f58588b189b940130e3edab3c839b2df`的`g1_29dof_rev_1_0.urdf`。29个可动关节、0手指关节；35个引用网格与上游Git blob哈希相同。
- 使用官方文档支持的URDF分支、官方PD/力矩/速度/默认姿态、统一动作尺度0.25。现有LightLP地形、10%全等级重置、AMP规则和奖励不添加补偿。重新生成该模型的AMP FK数据；验证实际关节、动作响应和回报后开启独立冷启动。
- 验收仍需足够训练曲线与匹配预算的T4对照，以及固定平地/稀疏地形evaluator JSON、lineage和连续回放。尚无修复后训练成功证据。

## 官方 G1 v4 替换（2026-09-07）

用户已要求替换最新官方 G1 29DoF 移动操作资产配置并重新训练。官方源码固定到 IsaacLab `b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8`，原始配置保存在 `legged_lab/assets/unitree_g1/official_g1.py`；canonical AST 与上游一致，溯源见 `artifacts/portability/v4/official_source.json`。

- 模型采用官方 Isaac 5.1 G1 USD。旧服务器资产常量指向缺失的本地4.5目录，因此仅在任务资产包装层显式使用官方5.1 URL，并开启接触传感器。电机模型、PD、限值、惯量参数、默认姿态保持官方配置；不升级服务器共享IsaacLab。
- 官方USD实测为29身体关节+14手指关节。策略仍控制29身体关节；14手指显式列为辅助关节，保持默认目标。Actor/critic/AMP仍为1997/2076/70。辅助关节仍属于整机动力学及整机接触、速度/限位约束；不宣称这是只含29关节的USD。
- 使用官方默认关节姿态重置，关闭独立角度乘0.5～1.5与初始根速度扰动，避免改写新资产默认出生姿态。地形、10%全0～9随机等级、AMP原曲线、统一action_scale=0.5、奖励权重及30k预算保持。
- 在新USD上重新生成6个LAFAN1专家，路径为 `legged_lab/envs/g1/datasets/motion_amp_expert_official_v4`，每份899帧，70D；旧URDF与旧专家保留。`legacy_mode15.py`只用于旧资产复现。
- 候选远端：`/home/nubot/phn_ws/t4_train/TienKung-Lab-g1-official-v4-20260907`。新run名 `g1_official_teacher_v4`，计划GPU1/3、各2048env，resume=false；旧v3在新运行验证后已停止，7个checkpoint保留。
- 当前验证：104项相关检查通过（其中一项最初因Windows pytest默认临时目录权限在setup失败，指定新临时目录后通过）。官方USD加载成功，32env/240步接口probe有限，终止AMP快照160项一致。双卡更新与正式开训待完成。

## 工作面和不可缩减的验收

执行代码仍在 `D:\TienKung-Lab-g1-portability-20260906`，分支 `g1-portability-20260906`。原根工作区的用户改动保留。此前 portable_v1 的随机上限 3 已被用户否决，不能把它当现行配方。

1. G1 训练的随机重置比例 0.10、等级范围 0～9；保存配置和运行时采样都要验证覆盖。
2. 用户再次明确 **AMP 恢复原规则**：普通地形 difficulty≤0.3 时倍率 1，随后线性衰减至 difficulty=1 时倍率 0.3；踏石/圆桩始终为 0。撤回全地形从等级 0 衰减的新方案。权重属于当前 transition，不能在自动 reset 后读新 episode 等级。
3. 扫描全部 Python 源码、入口和测试，区分通用算法、机器人定义、兼容入口以及历史实验文件。通用实现迁入算法模块；不能只改名而仍由 G1 继承 T4。
4. T4、G1 以各自机器人配置接入同一环境、AMP、观测和镜像实现；机器人接入必须显式提供身体/关节语义，不得按名字分支或默认套 T4 参数。固定 T4 观测与旧 checkpoint 合同保留。
5. 启动前验证关节顺序、左右镜像、身体选择、AMP 特征与专家宽度、动作尺度和观测维度；接入错误提前失败并给出具体原因。用另一组关节命名/数量证明算法无 T4/G1 特判。
6. 更新训练、评估、回放、学生和运动跟踪中受影响的共享调用，以及当前文档。确属机器人资产、标定、已命名任务和历史产物的名称保留，并记录理由。
7. 窄测试、相邻回归、T4/G1 真实 Isaac probe、独立审查通过后中文里程碑提交；同步并核对远端源码，开启修正后的 G1 新训练和 TensorBoard。训练健康与行为能力分开验收。

## 完成范围与后续

- 已完成：课程与 AMP transition 修复，教师/深度学生/动作跟踪共享实现，独立机器人配置和接入检查，全库调用清单及独立审查。
- 已完成：中文里程碑提交、远端 SHA 核验、新 lineage 开训与保存配置检查。代码收尾不暂停已运行的训练。
- 后续实验：旧 v2 已按用户要求停止，保留 checkpoint 对照。修复后的 v3 已另开 lineage 冷启动，预算 30k，完成后执行固定评估与回放。

## 下一轮正式训练预算

- 用户最新指定：G1 统一 `action_scale=0.5`，关闭逐关节 effort/Kp 缩放，恢复原动作惩罚。尺度补偿候选已撤回；200 轮结果不能证明原始根因或否定统一缩放。排查记录见 `docs/reports/2026-09-07-g1-teacher-slow-convergence.md`。
- 用户已澄清：30000 轮用于排查后的新训练，不部署旧 v2 接续任务。新 v3 启动后，用户又授权关闭旧 v2，已执行。
- 新一轮已使用冻结代码 `d1f7f52` 与独立 lineage，启动命令和保存配置均为 30000，resume=false；不会从旧 v2 的 model_9999 接续。
- 短对照均已退出；200 轮的动作尺度/奖励补偿候选不计入正式预算，也不作为确定收敛根因的证据。

## v3 正式运行与验证

- 本地执行面不变；远端新目录 `/home/nubot/phn_ws/t4_train/TienKung-Lab-g1-amp-fix-20260907`，tmux `g1-teacher-v3`，GPU 0/2 各 2048 env。旧 v2 已退出，GPU 1/3 释放，20 个 checkpoint 保留。
- run：`2026-09-07_10-08-58_g1_lightlp_teacher_v3`；TensorBoard `http://100.100.188.39:8032/#scalars`。
- 已验证 283 个源码/资产文件及 6 个专家文件 SHA；启动前实测双 GPU 3 次更新后 policy、discriminator、AMP normalizer 的跨卡差值均为 0。
- AMP 原始统计量输入、梯度惩罚坐标的失败均已复现；修复及相邻检查 33 passed。原尺度补偿已撤回，统一 0.5 基线相关 46 项通过。Black 对新增 AMP 实现和测试通过。
- 启动证据 iteration 18：保存预算 30000，resume=false，scale=0.5，model_0 已生成，所查 loss 有限；随机重置比例约 9.94%，抽中等级均值 4.69。只证明训练在更新，不证明学习改善或越障能力。
- 证据：`artifacts/portability/v3/`；排查详情：`docs/reports/2026-09-07-g1-teacher-slow-convergence.md`。

## 本轮已修复的问题（修复前）

- `G1LocoTeacherEnvCfg`、奖励和 agent 均继承 T4 类；核心环境按 T4/G1 关节名字分支选择 AMP builder。
- 同一 AMP 代数在两个 builder 复制；G1 镜像从 T4 schema 导入公共感知尺寸；环境内存在 `t4_joint_ids`、`action_t4` 和默认 T4 身体名。
- AMP runner 在 `env.step()` 自动重置后计算难度倍率，终止 transition 可能使用新等级。需在 step 前取快照。
- AMP 保持原规则；恢复后须逐等级验证普通地形曲线和两类稀疏地形清零。基础系数仍为 0.3，task reward lerp 仍为 0.7。
- 已核对 portable_v1 训练进程树并发送 Ctrl-C；随后 tmux 训练会话和 GPU 计算进程均消失。远端 `artifacts/portability/stopped_v1.json` 保存停训原因和原进程树。TensorBoard 留存。

## 验证记录

- 教师开训审查：独立结构 review 与 cold verification 已通过；记录 `artifacts/portability/teacher_ready_review.json`。
- 两个实际 Isaac JSON：T4 1937/2016/66、G1 1997/2076/70，240 步接口验证，实际随机采样全 0～9；AMP 原规则。此 probe 为 flat，不是越障能力证明。
- 149 项核心检查、90 项相邻检查通过；评估入口调整后相关89项通过。旧新镜像逐元素相同。
- `g1_lightlp_amp_full_levels_v2` 已在最终提交和远端 SHA 核验后启动，GPU 1、3，各 2048 env，首轮 10k。
- 原始 probe 源码哈希在 `neutral_probe_source.json`；最终训练另立 manifest。学生、运动跟踪与全库边界整理已完成，详见下方记录；行为验收待训练产出。

## v2 已启动

- 训练源码：`4fd6ec262dde7e0e3dd83dce73a71e8bb6916d93`。已逐项核对 211 个代码文件和 6 个 AMP 专家，见 `artifacts/portability/v2/lineage.json`。
- 训练目录：`/home/nubot/phn_ws/t4_train/TienKung-Lab-g1-portability-20260906/logs/g1_loco_teacher_sparse/2026-09-07_01-35-42_g1_lightlp_amp_full_levels_v2`。tmux `g1-full-levels-v2`；TensorBoard `http://100.100.188.39:8031/#scalars`。
- 启动证据保存于 iteration 43：最近 20 轮随机重置比例 10.12%，loss 为有限值，`model_0.pt` 已保存。随后在线确认推进至 iteration 138；保存配置是 0.10/None/None、10 行、AMP start/min 均为 0.3。这只证明训练在更新，不证明越障能力。
- 训练所用远端源码保持冻结。后续学生/运动跟踪代码整理已在本地工作树完成；10k 后按训练脚本执行固定评估与连续回放。

## 学生共享运行时已验证

- `locomotion/depth_env.py` 接管 FF 与 LightLP 深度学生观测、历史、延迟、噪声和相机扰动；T4 相机与训练配方保留。27/29/21 关节测试覆盖动态维度和特权隔离。
- 独立新旧对照：14 个方法 AST 不变，48 次含 clean/noise、裁剪、延迟、重置和等待渲染状态的逐元素对照一致。对应 T4 旧 checkpoint 的观测排列保持不变。
- 实际独立 RTX 验证：4 env/24 步，3168/1937/96/27；处理深度全有限，去预热后 11 次变化。当前源码清单 214 文件，见 `artifacts/portability/depth/final_source_manifest.json`。探针已退出，GPU 0 释放。
- 发现并修复首次独立导入的循环注册：共同配置原样迁至 `legged_lab/config.py`，公共环境与教师配置冷导入不加载 `legged_lab.envs`。RED/GREEN JSON、源码清单与 review 记录在 `artifacts/portability/depth/`。
- 本切片独立审查通过；随后完成运动跟踪和全库边界整理，正式 G1 训练持续运行。

## 跟踪共享实现已验证

- 命令/采样/奖励/终止/观测/事件迁入 `motion_tracking/mdp`，命名轴与离线加载迁入 `motion_tracking/schema.py`、`loader.py`；RSL 接口迁入 `utils/rsl_rl_compat.py`。旧 T4 路径为兼容导出，T4 文件与任务配方保留。
- 64 项相关检查通过；独立对照真实 T4 NPZ 的 12 个字段完全一致。生产 `MotionLoader` 的 21/27/29 关节重排和错误名称拒绝均通过。
- 真实 Isaac 2 env/原任务 50 步 + wrapper 1 步，150/276/27 保持有限值，实际使用公共类；公共 MDP 冷导入不加载任务注册。225 个源码文件 SHA 已核对，tmux `neutral-tracking-probe` 已退出。证据在 `artifacts/portability/tracking/`。
- 结构 review 与 cold verification 均通过。

## 全库核对与收尾

- `scripts/audit_robot_boundaries.py` 扫描当前工作树 243 个 Python 文件，记录角色、导入和逐文件 SHA，0 项边界违规；清单在 `artifacts/portability/robot_coupling_after.json`。扫描覆盖整个树，排除依赖缓存和 pytest 临时目录；静态检查不替代机器人参数与行为验证。
- 原根目录只读清单为 313 个 Python 文件，包含 109 个诊断产物；确认原有 3 处跨机器人依赖均已在新工作树消除。原根目录用户代码不覆盖，证据为 `original_checkout_inventory.json`。
- 最终冷审查发现评估出生点仍默认 T4 站距；现已改为必须显式提供站距与足底尺寸，通用评估从机器人 spec/足底扫描配置读取。T4 原数值保持，宽站距与大足底越界均被拒绝；相关 60 项检查通过。
- 全量本机测试记录为 375 passed、22 failed、1 skipped（最后站距修复前）；22 项均为现有 Linux/ZL 部署环境不齐，包括缺失 `zl_deploy`、固定 Linux 场景包及 `/usr/bin/python3`。相关测试与迁移前一致，独立审查复现；没有跳过这些失败或宣称全库测试全绿。Isaac 合同在实际服务器探针验证。
- 收尾实时证据：iteration 1042，最近 20 轮随机重置均值 10.29%，所查 loss 有限；另核实 `model_1000.pt` 已保存。`artifacts/portability/v2/closure_health.json` 只证明训练健康，不证明越障能力。
- 正式训练冻结于 `4fd6ec2`；后续学生/跟踪和评估边界收尾不热覆盖该源码。新增机器人仍需自己的物理资产、动作数据、相机标定和行为评估，当前没有 G2 实机模型或学习成功证据。

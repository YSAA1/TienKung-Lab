# 机器人无关的运动训练算法整理

Status: 2026-09-08，按用户最新授权实施中。F1–F6/R1已落地，173项CPU回归通过，三组单卡语义验收通过；专门reset增强验收及双GPU更新进行中。旧A/B源码和历史保留，尚未停止或启动新正式训练。

## 实施中：明确缺陷修复与真实环境验收

**目标与范围**：修复两轮审计已经复现的计算/重置/接线错误，并在服务器真实 Isaac 环境中验证。执行工作树为 `D:\TienKung-Lab-g1-portability-20260906`；修改不得热覆盖现有远端 `TienKung-Lab-g1-progress-ab-20260908`。下方 A/B 记录继续描述既有运行，不代表本修复已经启动。

**Actor 合同保持**：真实根线速度只进入 Critic 是合理的非对称 Actor–Critic 设计。此前将其列为优先修复方向的建议撤回；本计划不向 Actor 添加真实速度，也不安排该变量对照。G1 Actor/Critic/AMP 保持 **1997/2076/70**；T4 sparse 保持 **1937/2016/66**，其余已有观测合同不变。

### 已确认的修复清单

| 编号 / 优先级 | 缺陷与修改位置 | 修复决定 | 必须通过的验收 |
| --- | --- | --- | --- |
| F1 / P1 | 足速度历史跨 reset：`locomotion/env.py::reset`、`update_foot_accel_penalty` | 在 reset 写入和 forward 完成后，以**已可靠刷新的本回合实际 link velocity**重新播种对应 env 的足速度历史；清该 env 的 EMA。不得读到旧 ArticulationData 缓存，不得给所有机器人统一置零，不额外推进物理时间来掩盖时序。 | 重跑已失败的8env G1反例：root/qdot/PhysX足速度为0时，首步仅重力运动，EMA从113.93/64.22变为0/0；第二步真实加速度仍生效。再覆盖T4非零初速度、连续/部分reset，未reset环境历史不变。 |
| F2 / P1 | `extras["time_outs"]`跨步残留：`env.py::step/reset` | 每个控制步无条件导出当前timeout mask，不依赖是否有人reset；明确与本步done、有效物理终止原因的关系。 | 实际horizon结束后的连续三个无reset步骤返回全false；部分reset、全reset及混合原因均正确。 |
| F3 / P2 | 截断bootstrap使用动作前价值：`env.py`、`amp_on_policy_runner.py`、`amp_ppo.py` | reset前构造真实终点Critic观测，只对纯截断补 `gamma * V(terminal)`。使用临时历史副本，不推进真实历史，不取reset后状态，不改变输入字段/维度。 | CPU反例r=1、gamma=.99、V(start)=10、V(terminal)=2得到2.98而非10.90；真实终点观测对齐、历史只推进一次；物理失败与timeout重叠不bootstrap。 |
| F4 / P1 | `.05`探索下限配置无效：`amp_on_policy_runner.py`、`amp_ppo.py`、策略std参数化 | 读取真实配置值，在初始化、加载、optimizer更新后落实下限；覆盖scalar/log std。明确 `.05`是policy action标准差，本G1尺度下对应目标角标准差 `.0125 rad`，不是关节范围比例。 | 下限以下初始化/加载/更新均被约束；正常高于下限时分布不变；两rank一致。保留已有`.05`数值，不扩大探索强度。 |
| F5 / P2 | 合法站立受隐藏航向目标惩罚：`mdp/rewards.py::heading_error` | 仅对 `is_heading_env & ~is_standing_env` 启用heading误差；没有heading任务时该项为0，运动heading原公式保留。 | 相同站立输入下，隐藏target=0或pi均罚0；运动heading仍正确；稀疏yaw-rate桶沿用既有语义。无需增加heading观测。 |
| F6 / P2 | AMP插值和起点采样有偏：`rsl_rl/utils/motion_loader.py` | 按 `t/frame_dt` 索引；起点均匀采样 `[0, length-step_dt]`，明确边界与过短clip报错。 | 4帧线性例在一帧时间取到1而非1.333；scalar/batch结果一致，终点不越界，无多余首帧点质量。保持17个专家文件、SHA、逐文件权重和70D格式。 |

**F1 实现约束**：单纯把赋值挪到 `sim.forward()` 后不算完成，必须与当前运行时直接 PhysX `get_link_velocities()` 的结果对照，确认body顺序、坐标系和缓存刷新；零与非零reset速度都要测试。若运行时确实无法可靠获得reset瞬间速度，备选是每env历史有效标记：首个真实物理更新只播种、跳过该20ms差分，随后正常计算；这有明确首步语义变化，必须在实施记录中说明，并对T4/G1验证，不可悄悄叠加两种方案。

**F3 实现约束**：

- 仅为待bootstrap的env构造Critic只读预览：当前本体帧和前向scan分别按既有oldest→newest顺序加入各自临时历史；足scan与immunity保持单帧，按原顺序在末尾拼接，再使用原尺度/clip。覆盖history=1、多帧和未填满历史；不调用会append的 `compute_observations()`，不采Actor噪声，不更新normalizer统计。
- 保留现行 `timeout_causes = horizon | OOB | joint_guard`、所有阈值、免疫规则与reset事件。有效物理 `terminated` 必须来自真实终止原因，不能用 `done & ~time_out` 反推；使用 `bootstrap_mask = timeout_causes & done & ~terminated`，使同时发生的真实失败优先。
- GAE在所有done处截断。正常reset及其后观测生成仍只有一次。检查共享环境/runner调用方，不要求未迁移环境默默使用错误的旧值；对本次支持的合同显式接线，兼容范围及相邻调用须验证。

### 两项运行保障

| 编号 | 内容 | 验收 |
| --- | --- | --- |
| R1 | 必需foot scanner缺失时显式失败：`locomotion/env.py::compute_foot_scan_privilege`及初始化校验 | 配置要求足scan却缺任意一只传感器时给出明确错误；不再用全0伪装有效观测。正常ray miss、洞、全平地常量和显式关闭的可选配置按原语义处理。 |
| R2 | 精确运行依赖与资源溯源 | manifest同时记录项目commit、IsaacLab `3d5ea25ddbcba05bef4c9acd1dacb9fa728b289b`、唯一`assets.py`内容补丁、Isaac Sim/Python/Torch/NumPy版本、实际导入路径及URDF/专家SHA。保持已实测的运行时内容，不顺手升级、回滚至精确v2.1.0、改4.5视觉资源根或重装服务器环境。 |

当前342文件及真实导入检查未发现业务代码漏同步；IsaacLab的1534个Git modified中1533项仅权限变化，唯一内容改动是资产根。实际G1的29关节PD/力矩、URDF来源和最终自碰撞属性已核对。历史T4缺少精确依赖commit，保留这一对照限制，不能宣称历史环境已完整重建。

### 执行顺序与验证

- [x] **计划与修复前证据**：完成两轮审计及真实Isaac检查，确认Actor速度设计正常。证据持久化到 `artifacts/portability/g1_core_fixes_20260908/evidence/`，SHA见该目录 `manifest.json`。这只表示证据已保存，不表示修复通过。
- [x] **阶段0，固定候选环境**：记录实施时本地HEAD和dirty状态；由冻结源码建立独立候选运行面，明确允许的变更清单。记录上述精确依赖与`assets.py`补丁，保留旧A/B的代码、checkpoint、配置和日志。不得用旧运行目录承接修改。
- [x] **阶段1，修reset及transition边界**：优先F1、F2、F3，先把现有反例写成有意义的行为测试，再修实现。F2/F3共享同一组终止原因、terminal观测和history检查，避免两个实现互相覆盖。提取可复用的小型Isaac验收入口，不把旧probe的 `ok=true` 当作合同通过。
- [x] **阶段2，修配置与采样接线**：完成F4、F5、F6及R1；互不影响的实现/审查可并行，相关验证通过后按范围提交。R2贯穿两个阶段。保持原专家内容与其权重，原奖励数值、动作尺度/PD、命令、地形、终止阈值不调参。
- [x] **阶段3，CPU回归**：运行新增反例及受影响的AMP更新/地形transition、T4/G1观测与机器人映射、稀疏奖励/终止/部分reset回归。实现触及共享深度环境或其他runner时，仅补相应相邻检查；无需重训学生或机械跑全库部署测试。证据记录实际命令、解释器和结果，不能沿用修复前通过数。
- [ ] **阶段4，真实单卡Isaac集成**：nubot独立tmux中串行跑G1 mixed、T4 mixed、G1 flat各32env约500步，加关节脉冲、部分reset、horizon后无reset、真实物理失败与截断重叠；重跑8env高速足速度reset反例和T4非零初速度版本。逐项断言F1/F2/F3语义，而非仅检查finite/shape；核对最新scan/历史、terminal AMP以及固定Actor/Critic/AMP维度。仍保留诊断注入与自然rollout的区别。
- [ ] **阶段5，真实双GPU集成**：同一候选，每rank32～64env、原24steps/update、2次PPO/AMP更新，使用当前完整17段数据；在不同rank制造不同reset/截断分布。确认第二次采样正常、没有因某rank无截断而遗漏collective、policy/discriminator和AMP normalizer一致、std下限生效、checkpoint与新metadata可保存加载、进程正常退出。只在tmux运行，不与其他probe并行初始化USD。
- [ ] **阶段6，独立审查与交付**：审查最终差异、所有反例的修复前后结果、运行模块/资产SHA和保存配置；仅提交任务相关修改，使用中文里程碑commit。产出修复后manifest、原始JSON及日志，更新本计划状态。最终行为验收未完成时，不将G1学习问题标为已解决。

CPU入口沿用现有测试组合，具体按实际修改选择：`tests/test_amp_update_contract.py`、`tests/test_amp_curriculum_transition.py`、`tests/test_t4_observation_contracts.py`、`tests/test_robot_neutral_locomotion.py`、`tests/test_t4_sparse_reward_contracts.py`、`tests/test_collapse_recovery_contract.py`及新增reset/bootstrap/loader反例。本机使用已有 `D:\anaconda\envs\pytorch\python.exe` 做CPU检查；Isaac验收使用nubot的 `scripts/nubot_run.sh`，不安装本机Docker来替代它。

**后续训练与能力边界**：2026-09-08用户已明确授权实施本计划、使用指定code-review独立审核；全部门槛通过后停止旧A/B并保留全部checkpoint和日志。新lineage单组冷启动，B径向净进展晋级，30000更新，GPU0/2各2048env，24steps/update、原17段专家；不加载旧optimizer、不增加消融组。能力验收继续使用同预算checkpoint、固定vx=.7、零根初速、flat/easy stones/easy pillars各32env，报告实际速度/净位移/reach2m/终止原因、evaluator JSON、lineage与连续回放。无需重新发明或降低现有门槛。

**明确不纳入本轮**：Actor/Critic新增特权量、AMP70→76D或降权、净位移窗口奖励、PD/动作尺度/探索强度调参、放宽终止、改镜像PPO、启用empirical normalization、重定向/裁剪专家数据。这些不属于本轮已确认缺陷的必要修复。脚速度和timeout问题虽已在G1真实复现，仍位于共享逻辑，不能预先承诺修完就追平历史T4。

`final_integration_claim`：完成时必须满足F1～F6及R1/R2均完成、修复前失败的真实反例通过、正常差分/终止/历史/采样合同及既有观测宽度保持、单卡和双卡实际更新验证通过、候选来源可追溯且旧运行未被污染。此声明只覆盖核心缺陷和开训合同修复；G1持续行走、越障及追平同期T4必须另外由行为证据支持。

修复前证据索引：[证据说明](../../artifacts/portability/g1_core_fixes_20260908/README.md)。

### 本次实施记录（2026-09-08）

- 修复前基线 `c2cb0440519cb6a35b54c5554da2a6774398d577`；原有未跟踪工件未纳入修改，快照为 `artifacts/portability/g1_core_fixes_20260908/stage0.json`。候选运行面 `/home/nubot/phn_ws/t4_train/TienKung-Lab-g1-core-fixes-20260908`，旧A/B的342文件SHA仍匹配冻结清单。
- F1采用reset写入与forward后的直接PhysX COM世界速度，沿用实际feet body索引；没有额外物理步，也没有采用首步跳过方案。G1零初速首步EMA已实测0/0；T4非零初速、部分和连续reset通过。进一步用第二步速度注入明确覆盖非零真实差分。
- F2每个控制步写当前timeout快照；F3从有效torso/accel/fall_over/collapsed原因独立计算terminated。终点Critic用临时oldest→newest历史，保留scan/foot scan/immunity排列；reset后的真实历史仅追加一次。AMP只给纯截断加终点价值，所有done截断GAE。
- F4读取原`.05`配置并约束scalar/log参数的初始化、加载、optimizer更新；这是policy action标准差，对G1 `.25`动作尺度为`.0125 rad`。F5站立/非heading任务为0；F6按frame duration插值、在有效连续起点区间均匀采样，短clip显式拒绝。未修改奖励系数、动作尺度、PD、地形、命令、终止阈值或专家。
- 兼容范围：共享T4/G1前馈AMP教师。未提供终点Critic的旧AMP环境和循环Critic在初始化明确拒绝；深度蒸馏环境输出的是teacher targets，保持其原step接口且不构造PPO终点Critic。相邻深度观测测试通过，未重训学生。
- CPU：指定`D:/anaconda/envs/pytorch/python.exe`运行173项受影响检查通过；8个最初失败用例及额外timeout基线反例已保存。flake8在3.12运行，本次无新增问题；3条既有告警已用基线源码独立复现。其余适用pre-commit hooks通过，旧pyupgrade用3.12单独运行以避开3.14兼容问题。
- 真实单卡G1 mixed/T4 mixed/G1 flat各32env、500步加pulse、部分reset、horizon后3个无reset步骤、真实加速度失败与截断重叠均通过；Actor/Critic/AMP仍1997/2076/70与1937/2016/66。首轮末尾的根位置下移注入没有触发物理失败，明确保留为失败记录；修正为实际根速度+物理步后重跑通过，不改变正式阈值。
- 工件位于 `artifacts/portability/g1_core_fixes_20260908/`。单卡/双卡是计算与开训合同验收，不是行走或越障能力证明。阶段5、独立review与正式开训完成后继续更新本记录。

## 既有运行：恢复 T4 速度与真实进展晋级 A/B

- 原因：同 11000 更新，recovery OOB 0.23% 对 T4 54.35%，简单石头/圆桩 reach2m 为 0。固定 model_11000 平地 6 秒累计路径 1.558 m、净位移仅 3.45 cm，证实原地往复晃动。旧晋级用累计路径；G1 最低速度缩放 0.5 又使静止在 0.3 m/s 命令下达到 tracking 0.698，不能继续把 terrain level 或回合增长当能力。
- 用户明确取消降速：`sparse_command_min_speed_scale=1.0`，踏石/圆桩各难度都是 0.6–2.0 m/s；其余地形命令与合法站立合同不变。最低速度 0.3 是未经独立验证的迁移假设，不是 G1 物理要求。
- A：GPU0/2，`--g1_progress_ab A`，晋级沿用累计路径。B：GPU1/3，`--g1_progress_ab B`，晋级用相对 tile 中心最大径向距离 >4 m。两者保持 tracking>=0.5、moving 和 pit 等现有约束，OOB 4.25 m 不变。径向距离可以在对角方向先于 OOB 达标，不能把晋级视作 OOB。
- 两组共用同一冻结代码、recovery 终止（塌低 .20 m/.20 s、impact immunity）、全17段 AMP、线速度权重2、统一动作尺度.25、官方资产/PD、零根初速、固定默认关节姿态、10%全等级随机回访。每组双卡各2048env、24steps、seed42+rank、98304样本/更新、预算30000、save interval500，不加载旧checkpoint。A/B 唯一训练行为差异是晋级距离。
- 共用只读监控：实际出生点净/最大位移、路径、净位移/路径比、命令方向进度和完成度、.4秒窗口平均速度；按地形和命令桶记录。Progress/moving 是回合曾有移动命令，standing 是全程站立；window_net_speed 是窗口速度模长的回合均值，不是最终净位移/时长。RewardMix 为 transition 开始地形/命令的每步 task/style贡献，使用 __n 做跨rank加权；原有末步命令重采样语义保留。
- 先通过CPU回归、独立review、真实Isaac命令采样/部分reset/晋级接线probe，再保存旧checkpoint与配置清单、停止旧训练，串行初始化 A、B 避免Isaac临时USD冲突。新根 `TienKung-Lab-g1-progress-ab-20260908`；入口 `scripts/train_g1_progress_ab.sh A|B`，每组独立tmux。
- 500/1000/2000/4000更新按相同预算复查真实位移和固定vx=.7、零初速、32env的flat/easy stones/easy pillars评估与连续回放。没有自动调参、自动切组或无证据能力声明。
- 本轮不改瞬时速度奖励、AMP权重或探索下限。若全速两组仍晃动，先离线核查奖励排序，再单独验证窗口速度奖励；AMP .3→.1为后续独立候选，不混入本次对照。
- 工件：`artifacts/portability/g1_progress_ab_20260908/`。完成声明只覆盖代码合同与正确开训；行走/越障能力仍须 evaluator JSON、lineage 和连续回放。

### 本次验证与启动记录

- 120项相关CPU检查通过；独立review无阻断问题。适用pre-commit hooks通过；缓存flake8/pyupgrade与默认Python3.14不兼容，使用已有Python3.12运行同插件。flake8仅有3个HEAD已存在的告警（env R506、runner log C901/E126），没有本次新增告警。
- 真实Isaac32env/100步：level0/4/9每级260次稀疏命令采样完全相同，范围.60055–1.98255；部分reset清空窗口并重新记录出生点；注入相同晃动历史后A晋级率1、B为0。该注入只验证接线，不是物理行走成功率。
- 两rank各64env/2更新smoke生成model_0/model_1，Progress与RewardMix标量存在且有限；已退出。headless图形插件/退出阶段有环境日志警告，未发现训练Traceback/OOM/NCCL超时。
- 旧full17/recovery四PID已退出，38+26个checkpoint及历史TB保留，最近为model_18500/model_12500，SHA清单在old_stop_confirmed.json。
- 同一候选冻结342个源码/资产/入口原始字节SHA与17段专家。前身manifest使用LF归一化哈希；本次改为原始字节并逐文件比对，未改写二进制资产。运行时相对前身只覆盖声明的6个已有源码文件，加新增monitor/启动/probe入口。
- A run `logs/g1_progress_A/2026-09-08_10-16-38_g1_progress_A`，PID1107854/1107855；B run `logs/g1_progress_B/2026-09-08_10-18-18_g1_progress_B`，PID1109813/1109814。tmux分别 `g1-progress-A` / `g1-progress-B`；TensorBoard `http://100.100.188.39:8041/#scalars`，会话 `g1-progress-tb`，API已确认两个run。
- 10:19快照A37/B13更新，两组保存model_0，最近全部标量有限；实际env/agent配置只差晋级模式与两个实验名称。10:20再次核实4rank的CUDA_VISIBLE_DEVICES、LOCAL_RANK以及342文件SHA；没有加载旧模型的日志或训练致命错误。
- 恢复查询：远端系统Python运行 `artifacts/portability/g1_progress_ab_20260908/status.py`，写latest_health.json；启动核验在launch_verification.json。A先初始化，因此按同更新/样本量比较，不按同一墙钟时刻比较。

`final_integration_claim`：按用户授权完成全速、同预算、晋级判据单变量的四卡A/B开训，代码与运行配置可追溯；后续500/1000/2000/4000的行为评估待完成，未宣称G1已学会行走或越障。

## 已替换实验：G1 终止与恢复合同修复

### 研究结论与方案

- 全面对照证据：`docs/reports/2026-09-07-g1-t4-contract-comparison.md`。不能再声称历史T4与G1完全对齐。官方29DoF、PD、关节顺序与70D专家为必要适配；AMP原始统计/同步/终止帧修复保留。
- 官方固定提交 `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3` 的 G1 velocity `TerminationsCfg` 使用 root height .2m、orientation .8rad和timeout，无torso净接触硬终止。说明躯干接触gate不是G1必须照搬的接口；其高度是根高度，不能把本项目脚相对高度称为官方同一判据。
- 当前PhysX对地形mesh的GPU contact filter实测不支持。拒绝全零过滤力，也不重新启用已复现内部自碰撞误判的torso净力终止。**修正上一轮“必须外部接触判据可用才能做任何协调”的过强前提**：第一组实验适配几何塌低与恢复语义，明确不宣称恢复了T4外部触地事件。
- 实施可配置的塌低持续时间和碰撞豁免：保留.20m几何判据，G1需连续低姿态.20s才因collapsed结束，服从既有impact immunity；恢复高度即清零计时，每次reset也清零。其他机器人默认保持原行为。T4式63°/每步1%倾倒、加速度/时限/出界/关节速度保护保留；本次仅新增对collapsed的豁免，其他事件沿用既有规则（immunity原本就屏蔽接触/加速度冲击，但不屏蔽随机倾倒和timeout）。
- .20s是待验证的短暂恢复窗口（当前20ms控制周期的10步），不是文献最优值或已证明能够治愈停步。记录原始塌低占比、被豁免的持续塌低、实际塌低终止，避免曲线归零制造假改善。
- 首个新实验只改变“塌低与恢复”这一合同组，线速度回到2，其他奖励、全17段AMP、重置/命令/地形/官方资产保持基线。固定默认关节姿态与零根初速暂保留，因为这是起步能力的清晰验收起点；不把未经接触检验的T4姿态倍率直接套给G1。
- 后续待因果证据：若恢复合同正常但仍站立，优先验证AMP相对强度/上肢速度判别，再验证初始化分布；不同时叠加奖励、PD、命令和地形修改。软接触选择器差异、稀疏速度缩放和实体边框保留为明示偏离。共享std下限失效不是本次单变量实验的修复项，另列问题，不偷偷混入。

### 执行与验收

- [x] 核实冻结历史配置、当前训练与官方G1终止设计；两组23:06分别有model_5000/model_2000，tmux存活。
- [x] 实现塌低持续时间/豁免合同和原始事件诊断；56项新增/相关检查通过。旧flake8插件在Python3.13崩溃，改用既有3.12运行；R506/SIM901两处告警在HEAD原文件复现，未引入新告警。pyupgrade在3.12通过，其余适用hooks通过。
- [x] 独立候选目录 `TienKung-Lab-g1-recovery-20260907` 的真实Isaac probe：部分episode计时清零通过；32env/1000步未发现持续时间或豁免违规；flat32/32站满6s、终止AMP快照验证通过。新组2 collapsed/28 accel、定义内恢复0，属于终止原因迁移，未证明冷启动学习改善。证据 `artifacts/portability/g1_recovery_20260907/probe_summary.json`；两组先执行部分reset校验，不能将旧审计的28例当成本轮legacy的27例。
- [x] review-agent 独立只读审查完整diff，再复核原始JSON/SHA、摘要和开训脚本；No findings。修正计划中原加速度豁免的文字错误。启动状态待单独核验。
- [x] 保留权重2基线；确认权重4五个进程退出，6个checkpoint保留（最后model_2500）。GPU1/3新lineage已启动：双卡各2048env、24steps、seed42、全17段、权重2、冷启动30000轮。339份源码/资产SHA复核，新旧保存env配置除rank/device差异外仅新增两项塌低恢复字段；17份专家启动逐项校验。23:28已36轮且所查loss有限，两个rank为920838/920839。
- [x] cleanup：本轨道README/索引/状态与本计划已同步；原始轨迹无损压缩并验证压缩前后SHA，审计输入及脚本快照保留；仅删除本任务传输tar和lint基线临时副本。23:34再次核验训练推进至132轮，双rank和GPU0/2基线仍在，17段专家与基线SHA完全相同。相关变更与证据纳入本次中文里程碑提交。

`final_integration_claim`：G1恢复合同实现、回归和真实物理验证通过；独立review无未解决阻断发现；新对照按冻结合同正确开训、基线未被污染、证据与恢复入口可追溯。**此声明不包含G1已学会行走/越障**。训练能力后续必须用同预算checkpoint的固定命令速度/净位移、恢复率、evaluator JSON、lineage及连续回放验收，不能由collapsed计数下降替代。

参考：[官方固定提交G1速度配置](https://github.com/unitreerobotics/unitree_rl_lab/blob/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py)、[IsaacLab接触传感器](https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/contact_sensor.html)。运行时限制以本机实际IsaacLab源码和contact_probe日志为准。


## 已替换：线速度权重 2 / 4 对照

- 历史配置：基线 GPU0/2、tmux `g1-full17-30k`、权重2；原实验 GPU1/3、tmux `g1-full17-lin4`、权重4。两组均全17段AMP、seed42、每卡2048env、24steps、双卡、全局4096env、冷启动30000轮，每轮98304样本。权重4现已停止；TB8039和checkpoint作为历史对照保留。
- 合并 TensorBoard：`http://100.100.188.39:8039/#scalars`，会话 `g1-full17-lin4-tb`，weight2/weight4 对应两个日志目录。原 TB8038 保留。
- 实验 run `2026-09-07_21-11-47_g1_full17_lin4`，日志 `logs/g1_full17_lin4/`。21:13核验20轮、双rank存活、model_0保存、所查loss有限；基线继续至2730轮。旧v6的4个训练进程退出，18个历史checkpoint保留。
- 保存的 env.yaml 唯一差异为 `reward.track_lin_vel_xy_exp.weight: 2.0 -> 4.0`；agent.yaml只差experiment_name/run_name。独立train入口从冻结基线快照派生，在构造env前设置该权重，不改共享运行源码。SHA、停止记录、配置差异及健康证据在 `artifacts/portability/g1_full17_lin4_20260907/`。
- T4 S11b历史训练日志第0/1轮累计98304/196608样本，确认每轮采样量相同。2400～2600轮窗口：G1/T4未加权tracking均值0.250/0.302，terrain level 1.374/1.544，平地reach2m 0%/13.37%；晋级率3.52%/1.36%，路径长度1.956/1.405m。G1数据为TB抽样79点，T4为201点。证据支持平地进展及平均等级落后，但不是所有指标都慢，也不能归因为机器人本身或量化成慢几倍。
- 两组权重不同，不直接按加权奖励判胜；路径长度也不是净前进距离。按相同轮数比较未加权tracking、等级、实际速度/净位移，最终仍以同预算checkpoint的evaluator JSON、lineage及连续回放验收。本次只确认实验正确开训，未宣称权重4更好。

## 保留基线：完整 17 段、双卡正式训练 30000 轮

- 18:51阶段用户取消当时的数据对照，启动双卡正式训练 30000 轮；后续新增的恢复对照以本页顶部为准。基线运行 `g1-full17-30k`，物理 GPU0/2，DDP world_size=2，每卡2048env、全局4096env、24steps/update、seed42、不续旧模型。正式日志根目录 `logs/g1_full17_30k/`；TensorBoard `http://100.100.188.39:8038/#scalars`，会话 `g1-full17-30k-tb`。
- 数据清单由 T4 S11b 保存的 agent.yaml 和实际17个 expert 文件逐项核对；远端源CSV关节角与当时expert一致，本地源CSV的LF哈希与远端一致。证据 `artifacts/portability/t4_rob2rob_full17_v1/t4_runtime_manifest.json`、`t4_source_check.json`。
- 全部17段：站立1、前进走1、后退走1、侧移类3、转向7、jog4；保留每个原始姿态，N帧CSV产生N-1帧前向差分AMP，不按前进速度裁剪、不跨片段拼接。总3303个70D expert帧、110.1秒。`t4_run`原本未用于T4训练，继续保持原有held-out状态。`walk_left`及`jog_left/right`包含曲线运动，文件名/类别沿用实际T4训练合同。
- 保持T4实际逐文件MotionWeight；归一化类别概率：stand20.833%、walk_forward20.833%、walk_backward12.5%、walk_lateral16.667%、turn16.667%、jog12.5%。不再沿用两段版62.5%/37.5%的配比。
- 修复全量检查暴露的后退跑IK分支错误：膝关节接近伸直时，允许的小幅超伸会把暖启动吸入反向屈膝分支。求解按T4正向屈膝约束G1膝角非负，保留官方训练关节限位不变；脚部误差较大时用源解剖姿态重试有界IK，不裁剪输出、不删除后退跑。回归检查覆盖后退跑完整轨迹。
- 全量最大脚掌位置误差4.30mm，关节硬限位超出0，最高关节速度为官方限值82.43%。Isaac同一G1 articulation / AMP builder逐帧重算全部3303帧，最大70D特征误差小于1e-6；CPU AMPLoader实际预采样100000条transition有限且权重一致；19项数据/资产/IK回归检查通过。
- 数据在 `legged_lab/envs/g1/datasets/motion_amp_expert_t4_rob2rob_full17_v1/`，完整根位姿/q29在 `motion_source_t4_rob2rob_full17_v1/`。连续回放 `artifacts/portability/t4_rob2rob_full17_v1/all_clips.mp4`，17段均包含，110.4秒/15fps；3页审阅图共68帧覆盖所有动作。
- 限制：运动学转换不代表动力学可跟踪或训练已成功。侧移最大足部方向拟合差约0.408rad；一次刚性根高度平移后脚底几何最低约-20.95mm，未逐帧抬根掩盖接触误差。足部近地运动只是几何代理，不能称为物理滑移验收。
- 正式训练复用冻结c125f74运行代码（204个文件哈希核验），仅使用支持 `--amp_expert_manifest` 的正式train入口快照选择新数据；原配置、奖励、动作尺度0.25、官方纯29DoF资产不改。启动脚本、训练入口快照、数据SHA和旧任务停止记录：`artifacts/portability/g1_full17_30k_20260907/`。维护版入口 `legged_lab/scripts/train.py`；实际远端入口为该工件目录下 `train.py`，避免改写仍运行的v6源码。
- 启动核验18:53:38：正式run `2026-09-07_18-52-02_g1_full17_30k` 已到27轮，model_0已保存，两个rank存活，已保存配置确认17文件/30000轮/不resume。所查PPO/AMP数值有限，早期能力未验收。恢复查询可在远端系统Python执行工件目录的 `status_snapshot.py.txt`；最新启动证据 `latest_health.json`。
- 原两段组已停止；全17段4000轮组只完成初始化，未放行学习，随后按用户最新指令停止；原旧数据零速control也停止并保留checkpoint。此后的正式目标是30000轮，不再等待4000轮对照或自动切换阶段。最终能力按同checkpoint的evaluator JSON、lineage和连续策略回放验收，不能用奖励或启动验证代替。

## 历史：两段前进走跑数据对照（已停止）

- 原始问题已量化：旧6段G1专家均只截取前30秒，按既有采样权重约26.3%帧水平速度低于0.1m/s；未加载dance文件，但低速与风格化动作不能靠walk/run文件名排除。该现象不单独证明训练失败的唯一根因。
- 全序列筛选曾得到21段，再按脚部运动筛到8段；回放发现后者偏向弯腰、手扶腰的walk4，未用于训练。候选统计及否决证据在 `artifacts/portability/t4_rob2rob_v1/rejected_candidates.json` 与对应contact sheet；草稿原件归档在 `artifacts/diagnostics/g1_curated_ablation/rejected_drafts/`，不属于正式数据。
- 最终采用用户允许的rob2rob：从现有T4 `t4_walk_forward.csv` / `t4_jog_forward.csv` 求解G1有界IK，保持按腿长/髋宽缩放的脚掌位置和方向，以及按臂长缩放的肩部相对手腕轨迹。腿长比例0.91938、髋宽1.01722、臂长0.73515；腰yaw映射，G1额外腰roll/pitch为0。不是把27维关节角直接填成29维。
- 每条转换轨迹仅作一次刚性竖直平移以对齐地面，无逐帧抬根、时间缩放、跨片段拼接或关节硬裁剪。全源CSV保留在 `legged_lab/envs/g1/datasets/motion_source_t4_rob2rob_v1/`；带源帧范围及SHA的70D专家在 `legged_lab/envs/g1/datasets/motion_amp_expert_t4_rob2rob_v1/`。
- 走路197帧/6.57秒/6个stride周期，均速0.896m/s；慢跑90帧/3秒/4个周期，均速1.861m/s。低于0.1m/s帧占0，硬关节限位超出0，最大关节速度分别为官方限值的54.1%/76.0%。脚掌位置最大拟合误差1.184mm，方向0.0932rad；手腕最大3.98cm。脚底几何接近地面的运动代理p95为0.423/0.318m/s，不等同真实接触滑移验收。
- 官方IsaacLab同一G1 articulation、同一AMP builder对全部287帧70维逐帧重算通过；独立17项数据/资产检查通过。连续专家回放9.6秒，144帧，SHA及12帧审阅图在 `artifacts/portability/t4_rob2rob_v1/`。这是运动学专家验收，不是动力学跟踪或已训练策略能力验收；只有两条短源动作，需通过新实验检验泛化与学习效果。
- 数据对照：GPU0/tmux `g1-reset-control` 使用原6段数据；GPU2/tmux `g1-amp-rob2rob` 使用新2段rob2rob数据。均seed42、单GPU2048env、24steps/update、4000更新、冷启动、零根速度、官方固定初姿/纯29DoF/scale0.25。walk/run采样总概率仍为62.5%/37.5%，奖励与AMP算法不变。
- 已核对两组环境配置完全一致，agent配置仅 `amp_expert_dir` / `amp_motion_files` 不同；初始policy、discriminator、q、root pose、terrain levels/origins的6组SHA完全一致。新组18:18:01放行，专家预采样SHA按预期不同。基线启动更早，必须按相同更新/样本量比较，不按同一墙钟时刻判优劣。
- TensorBoard `http://100.100.188.39:8037/#scalars`，tmux `g1-amp-data-tb`，工件 `/home/nubot/phn_ws/t4_train/TienKung-Lab-g1-unitree-v6-20260907/artifacts/portability/rob2rob_amp_ablation_20260907/`。`control` 为原基线目录的软链接；`rob2rob` 为新组。初始化、停止旧随机初速组、源码与数据哈希均有JSON记录。恢复查询脚本 `artifacts/diagnostics/g1_curated_ablation/status.py`。
- 每组4000更新后自动做同条件零初速平地/踏石/圆桩评估及平地/踏石连续回放。最终checkpoint为model_3999，结果只按自主起步、持续行走、速度与reach2m评判；不把奖励上涨作为解决证明。

## 历史对照：根平面初速度（随机组已停止，零速组复用）

- 触发证据：v6 `model_3000`，vx=0.7、32回合固定平地，实际速度0.00333m/s、最大前进距离均值0.02472m、reach2m=0/32，全部到时限。见 `artifacts/diagnostics/plateau_3000/flat.json`。专家回放和数值健康不能替代策略验收。
- 2026-09-07用户批准开始。两组均冷启动，seed42、2048env、24steps/update、4000更新（每组196,608,000样本），单GPU；最终checkpoint为 `model_3999.pt`。不加载v6旧策略。
- 唯一任务差异：`control` 根速度为零；`random_xy` 只将 `reset_base.params.velocity_range` 的x/y设为[-0.5,0.5]m/s。z与角速度仍为零，官方纯29DoF资产、PD、0.25动作尺度、固定关节姿态、奖励、AMP、混合地形及10%全等级重置保持一致。
- 原运行源码c125f74保持冻结，204个源码/专家文件LF哈希再次核对通过。独立训练入口为 `artifacts/diagnostics/g1_reset_ablation/train.py`；远端目录 `/home/nubot/phn_ws/t4_train/TienKung-Lab-g1-unitree-v6-20260907/artifacts/portability/reset_xy_ablation_20260907/`。
- tmux：`g1-reset-control`（GPU0）、`g1-reset-random_xy`（GPU2）；TensorBoard：`http://100.100.188.39:8036/#scalars`，会话 `g1-reset-ab-tb`。原v6继续在GPU1/3运行，未热改源码。
- 16:44:45开训前，两组保存配置除根x/y速度范围外完全相同；初始policy、discriminator、专家采样、关节姿态、根位姿、地形等级与地形原点的7组SHA完全一致。实际control速度全0；random_xy的x/y标准差0.2873/0.2890m/s，其余四维为0。证据 `artifacts/portability/reset_xy_ablation_20260907/pair_verification.json`。
- 两组完成后分别自动运行同一冻结G1配置的平地、简单踏石、简单圆桩评估（32env/32episodes，vx=0.7）；评估均从零根速度及官方关节姿态开始。随后各录16秒平地/踏石回放，并保存checkpoint、视频SHA。`play.py`保留默认材质/质量随机化，连续回放用于辅助行为检查。
- 验收看自主起步、持续行走、速度跟踪、reach2m及失败类型；不以总奖励判胜，不自动把实验组部署为正式新配置。单seed对照只能检验该候选解释，不能证明唯一根因。单GPU对照与历史双GPUv6仅作背景比较。
- 当前启动核验：16:50两组分别更新到109/99轮，均已保存model_0，所查PPO/AMP loss有限；早期塌低率仍约99%，尚无改善结论。正式行为结果待4000轮后的JSON及回放。恢复查询可在远端系统Python运行 `artifacts/diagnostics/g1_reset_ablation/status.py`；本地可用既有SSH工具发送执行。
- 启动原始脚本按字节保存在同一证据目录的 `train_snapshot.py.txt`，SHA与lineage一致；本地维护入口后续格式化不改变运行中的源码。该证据目录同时保存两组配置、启动健康和TensorBoard双run核验。
- 本次脚本与文档的适用pre-commit检查通过；旧flake8-return、pyupgrade与本机默认Python不兼容，改用现有Python3.12执行同版本插件通过，其余hooks正常通过。维护版train与启动快照的AST一致，远端恢复查询入口已实跑验证；16:55两组为195/177轮，原v6仍存活。

以下为历史修复与实验记录；当前执行以本页上方的单变量对照为准。

## 当前修复：纯29DoF与训练有效性（2026-09-07）

- 用户目标：G1能正常训练，训练及行为表现至少与同类型T4接近；有限loss、短probe、checkpoint存在均不能证明完成。
- 用户最新明确：只用官方无灵巧手29DoF配置，不叠加多变量补偿实验。此前准备的body/scaled/scaled_rate训练未启动，停止推进该方案。
- 实时失败证据：v4第485轮最近20轮平均回合5.84步，collapsed约96.10%；model_500在64个平地环境中反复每5步收腿触发collapsed。零动作约54步、标准随机动作约44步，表明已学到提前终止行为。
- 已确认错配：官方IsaacLab locomanipulation资产实际43关节，14个手指不进策略却进入限位惩罚，其中10个默认关节角在收缩软限位外；上半身刚度3000/5000与统一0.5探索导致巨大运动惩罚。局部修正探针仅用于归因，不作为新训练配方。
- 正确来源：Unitree `unitree_rl_lab`提交`4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`的`UNITREE_G1_29DOF_CFG`和`Unitree-G1-29dof-Velocity`；配套`unitree_ros`提交`7d6075f7f58588b189b940130e3edab3c839b2df`的`g1_29dof_rev_1_0.urdf`。29个可动关节、0手指关节；35个引用网格与上游Git blob哈希相同。
- 使用官方文档支持的URDF分支、官方PD/力矩/速度/默认姿态、统一动作尺度0.25。现有LightLP地形、10%全等级重置、AMP规则和奖励不添加补偿。重新生成该模型的AMP FK数据；验证实际关节、动作响应和回报后开启独立冷启动。
- 验收仍需足够训练曲线与匹配预算的T4对照，以及固定平地/稀疏地形evaluator JSON、lineage和连续回放。尚无修复后训练成功证据。
- v5配置提交`5b3402b`，80项相关检查通过，双GPU各128env/3更新后policy、discriminator、AMP normalizer差值均0；实际29关节、默认姿态软限位违规0。204个源码和专家文件远端LF哈希一致。单纯资产接入通过未被视为训练成功。
- v5在13:35冷启动，run=`2026-09-07_13-36-29_g1_unitree_29dof_teacher_v5`，GPU1/3、各2048env。到51轮回合3.57步、torso终止99.04%，随后停止并保留model_0及TB8034。
- **已复现的MDP错误**：官方纯29DoF开启自碰撞，当前躯干净接触力硬终止会将内部碰撞判成摔倒。16环境悬空且禁用重力的定动作探针，第3步躯干仍高1.786m、接触力2000.49N，触发2个torso终止。同物理状态单独清空G1的硬终止接触body名单后，完整50步状态/接触力相同，但所有torso误终止为0。普通手臂小幅摆动首个探针未触发，未把该阴性结果丢弃。
- **v6唯一行为改动**：`G1LocoTeacherEnvCfg.robot.terminate_contacts_body_names=[]`；不改官方资产、自碰撞、动作尺度、奖励、地形或AMP。接触软惩罚保留，原LightLP倾倒/冲击/速度及G1塌低判定保留。官方Unitree速度任务本来也不使用躯干净接触力硬终止。
- v6远端：`/home/nubot/phn_ws/t4_train/TienKung-Lab-g1-unitree-v6-20260907`；沿用v5生成的同一纯29DoF AMP专家。下一步：验证并启动单一v6，按实际采样量对照T4冷启动S11b（T4每rank1024env，G1每rank2048env，均24steps/iter）；G1第500轮与T4第1000轮才近似匹配采样量。T4资产Python一致，URDF仅换行不同，归一化后相同。
- v6已启动：tmux `g1-teacher-v6`，run `2026-09-07_13-55-07_g1_unitree_29dof_teacher_v6`，TB `http://100.100.188.39:8035/#scalars`；204个源码/专家文件LF哈希一致，model_0已保存，保存配置29关节/0.25/2048env/resume=false/30000核对通过。13:56第14轮回合49.57步、回报-4.53；仍是冷启动早期，不能作为成功证据。
- T4 S11b model_1000的32回合平地/简单踏石评估已完成，tmux `g1-t4-reference-eval`已退出，结果在v6/t4_reference。平地31/32到时限但平均前进仅0.075m；踏石平均0.964m，两者reach2m均0，因此不能仅追平此早期弱基线就宣称G1正常走路。G1到model_500后用相同条件评估并采集连续回放，并继续检查后续训练是否持续改善。
- 14:13实时：v6第349轮回合76.69步、回报3.52，collapsed约95.38%；尚未再现3～5步退化，但仍未证明正常学习。当前训练保持运行，不改参数、不重启。此前准备的多变量训练脚本已删除，未执行。
- 初始model_0的平地/简单踏石各32回合固定评估已产出JSON：全部在第69步collapsed，reach2m均0；约0.83m前进和0.54m/s平均速度来自倒下前运动，不能当成正常行走。证据位于`artifacts/portability/v6/early_evaluation/model_0/`。
- 第500轮验收已完成且未通过：平地32/32 collapsed，平均94.97步、前进0.891m；简单踏石32/32 collapsed，平均78步、前进0.774m，两者reach2m均0。同采样量T4第1000轮平地多数站至时限但几乎不前进，踏石约103步倒下；不能把追平该弱基线作为正常步态证明。
- 连续回放各16秒已逐段抽帧检查并核对800步诊断：短暂支撑后前倾下沉，没有持续交替步态。平地首次终止在第105步，骨盆高0.312m、相对最低脚部0.196m、倾斜0.832rad；符合塌低判定，没有再次出现躯干自碰撞误终止。JSON、MP4哈希及分析在`artifacts/portability/v6/early_evaluation/behavior_review.json`。回放保留出生点/材质/质量随机化，不是固定evaluator中的相同回合。
- 回放兼容处理仅涉及入口和编码：默认plane加载失败，改用已验证的`--terrain --terrain_types flat --difficulty 0`；MP4后端失败时产生GIF，随后用系统ffmpeg将全部200帧按原采样12.5Hz封装成16秒MP4。原始GIF留远端，800步诊断与视频manifest留存。`completed.txt`不单独作为成功证据，已核实录制进程退出、实际视频及帧数。
- 14:36实时：第797轮，最近100轮平均回合125.63步、collapsed约93.73%；训练仍在改善，保持官方配置及当前MDP不变。`g1-v6-early-eval`已退出。新tmux `g1-v6-eval-1000`在GPU0先评估T4 model_2000，再等待G1 model_1000并运行相同32回合平地/踏石评估及各16秒回放。脚本与lineage在`artifacts/portability/v6/evaluation_1000/`；下一步检查这些实际结果、进程状态和训练趋势，不重复启动评估或重启训练。

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

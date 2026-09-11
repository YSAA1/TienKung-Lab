# G1 教师收敛迟缓排查

2026-09-07。代码工作面：`g1-portability-20260906`。当前修复与运行入口见[执行计划](../archive/plans/2026-09-07--robot-neutral-locomotion.md)（已归档，后续由 vital 系列计划接管）。下文v2/v3结论是历史记录。

## v4/v5崩塌与纯29DoF修复

v4误选了带14个灵巧手指的IsaacLab locomanipulation资产：实际43关节，手指不进策略却参与限位惩罚，其中10个默认角度违反软限位。model_500在平地每5步收腿终止，训练回报从约-40升到-1.8同时回合骤缩，复现了提前终止的退化行为。

按用户最新要求改为宇树`unitree_rl_lab`的纯29DoF速度资产（4960b847）及`unitree_ros`无手URDF（7d6075f7），官方PD、姿态与统一0.25动作尺度保持。35个网格哈希一致，真实Isaac为29关节，默认软限位违规0，AMP在新资产上重建。未部署逐关节补偿或奖励重配方案。

v5进一步揭示整个MDP仍有错配：官方资产开启自碰撞，但G1继承的躯干净接触力硬终止把内部碰撞也当成摔倒。第51轮99.04%的终止为torso，平均仅3.57步。悬空、无重力的反例在躯干离地1.786m时由2000.49N自碰撞触发torso终止，排除了触地解释。仅清空G1硬接触终止名单后，50步物理轨迹及接触力不变，torso误终止消失。接触软惩罚、真实倾倒/塌低/冲击等原判据保留。

v4/v5均已停止并保留失败证据。v6采用同一官方无手资产与同一AMP，仅修复上述硬终止错配；尚待真实训练曲线、固定评估及T4对照，不宣称根因已穷尽或任务已完成。

## 结论与证据边界

G1 的低课程等级对应真实运动失败。排查进一步复现了 AMP 统计量输入错误、梯度惩罚坐标不一致、多卡判别器同步遗漏，已修复并开启新训练。它们是否解释完整的 T4/G1 学习差距仍需训练验证。逐关节 effort/Kp 缩放是后续新增设计，不能当成原始失败的既定根因。`d2a27cf` 的奖励补偿已撤回，未启动该方案的正式训练。

用户最新指定 G1 统一 `action_scale=0.5`，不使用逐关节缩放，动作惩罚恢复原函数。200 轮对照不能否定统一缩放，也不能证明补偿奖励有利。新的 v3 已按要求冷启动 30000 轮；随后用户要求关闭旧 v2，已在 9662 轮停止，未做接续训练。

## 已复现并修复的 AMP 实现错误

提交 `d1f7f52`，修改限于 AMP PPO 的输入与分布式一致性，未更改 AMP 难度曲线或奖励权重。

1. **统计量输入被重复归一化。** `AMPPPO.update()` 覆盖了 `policy_state/expert_state` 后，把标准化值交给 `Normalizer.update()`，但后者维护的是原始数据均值与方差。真实 update 的可重复输入 `[10,-20,30]`，实际累计成约 `[4,-5.67,6.5]`。现在统计更新使用原始采样；单卡合同已由失败变为通过。
2. **判别器梯度惩罚使用了不同坐标。** 分类损失使用归一化输入，`compute_grad_pen()` 却直接把原始样本送入同一个网络，约束的不是分类损失使用的位置。现在两者使用一致的归一化输入；原失败已消除。
3. **多卡同步遗漏 AMP。** 原 `broadcast_parameters()` 和 `reduce_parameters()` 只有 policy/RND。真实双进程 Gloo 检查中，广播后 policy 差值为 0、discriminator 差值为 0.605；梯度归约后 policy 差值为 0、discriminator 差值仍为 1。现在同步判别器参数和梯度，并按各 rank 样本数合并原始统计矩，保持 normalizer 一致。

修复后，同一双进程检查的所有跨卡差值为 0；不同样本数情况下，全局方差与直接合并样本的参考误差约 4.22e-15。真实双 GPU Isaac 训练 3 次更新后，policy、discriminator、AMP normalizer 差值也全部为 0，实际 action_scale=0.5，actor 1997D。

这些问题在 T4 沿用的 AMP 代码中也存在。T4 曾成功训练不能证明其处理正确，但当前证据也不足以将某一个问题认定为 G1 慢收敛的唯一原因。固定评估、原始/修复后输入以及分布式结果都保存在本报告对应的诊断目录。

## 比较的是冷启动 T4

T4 使用 `2026-08-22_01-12-08_t_sparse_lightlp_s11b_upright_tbslim`。保存配置 `resume=false`，TensorBoard 从 step 0 开始。S12 从 S11b 的 19000 轮加载，未将该热启动实验作为此处的冷启动基线。

以下为各自 8000 轮之前最近 100 轮的训练统计；它们不是固定评估成功率：

| 指标 | T4 S11b | G1 v2 |
| --- | ---: | ---: |
| 平均地形等级 | 4.682 | 1.602 |
| episode path length / m | 4.106 | 1.131 |
| episode tracking mean | 0.536 | 0.419 |
| 简单踏石 reach 2m | 89.27% | 0.00424% |
| 简单圆桩 reach 2m | 91.50% | 0.00556% |
| 平均回合步数 | 307.76 | 153.68 |
| 自适应学习率 | 6.33e-5 | 1.22e-5 |

原始数据和保存配置：`artifacts/diagnostics/g1_teacher_slow_20260907/comparison.json`。S11b 目录没有可读的 `.git`；其保存配置、事件文件与目录源码已留存，不伪造历史 commit。

## 固定评估复现了失败

均为 model_8000、32 env / 32 episodes、固定 vx=0.7、difficulty=0。保留种子控制的 reset 扰动；关闭观测噪声、质量/材质随机化和 push；踏石固定出生方向与横向位置。

- G1 平地：reach 2m 为 0/32，平均前进速度 0.01289 m/s，25 次 horizon、6 次 collapsed、1 次 fall_over。
- G1 踏石：reach 2m 为 0/32，31 次 collapsed；平均第一步抬脚约 3.44 cm。预 reset 状态显示机器人从出生平台滑出并下落，确实跌倒。
- T4 平地：reach 2m 为 32/32，平均前进速度 0.59879 m/s，平均进展 4.666 m。
- T4 踏石：reach 2m 为 32/32，平均前进速度 0.68825 m/s，平均进展 4.202 m；31 次走出地形、1 次 accel。

G1 的采样只读记录了动作、关节、脚和躯干状态，没有更改 reset 决策。body 选择对应真实 ankle-roll / torso_link，未发现“错把正常站姿判为坍塌”。不能靠放松终止阈值修复当前行为。

## 已撤回的尺度补偿尝试

旧 T4 的动作目标是 `q_default + 0.25 * a`；当前 G1 为 `q_default + s_j * a_j`，其中 `s_j = 0.25 * effort_j / Kp_j`。原平滑项为 `sum((a_t-a_prev)^2)`，因此同样的目标角度变化被罚 `1/s_j^2`。

| G1 关节 | 当前 s_j / rad | 同角度变化的惩罚 / 原 0.25 尺度惩罚 |
| --- | ---: | ---: |
| 髋 pitch / 膝 | 0.17375 | 2.07 倍 |
| 踝 | 0.4375 | 0.327 倍 |
| 腰 yaw | 0.11 | 5.17 倍 |
| 腰 pitch / roll | 0.05833 | 18.37 倍 |

最后一项对比的是 G1 原统一动作尺度，不是不存在的 T4 腰 pitch 对应关节。数据由 `compare_action_units.py` 从真实 runtime 尺度与 checkpoint 重算。

已撤回的方案曾使用 `sum((s_j*(a_t-a_prev)/0.25)^2)`，并补偿踝动作惩罚。数学尺度一致并不证明学习更好；这套补偿及专属测试已撤回，不进入下一轮正式配方。当前 G1 使用统一 0.5，原动作惩罚权重及函数保留。

已通过 GitHub API 核验 Isaac Lab v2.1.0 的[公共动作配置](https://github.com/isaac-sim/IsaacLab/blob/v2.1.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py#L101)为统一 scale=0.5，[G1 rough 配置](https://github.com/isaac-sim/IsaacLab/blob/v2.1.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_env_cfg.py#L99)未覆盖该尺度。官方任务使用 G1_MINIMAL_CFG，与本项目 29DoF 资产不同；此处按用户要求对齐动作尺度。

## 对照及排除项

- 单独把已训练 G1 的踝探索噪声缩到原角度幅度：64 个平地 episode 中 collapsed 从 15 次到 12 次，但两组 reach 2m 都是 0，未救活停走策略。
- 两组 200 轮 / 512 env / seed 42 冷启动，仅改变动作缩放：当前 effort 尺度最后 20 轮回合均值 36.39 步，统一 0.25 为 12.43 步。样本预算不足，不能据此否定统一缩放基线。
- 10% 随机重置仍覆盖 0～9；运行时随机等级均值约 4.5。课程升难门槛依旧为运动命令、path>4m、tracking>=0.5，低难度并非采样功能丢失。
- T4 S11b 与 G1 v2 保存的 PPO 参数一致，`amp_ppo.py` 相同。较低学习率是已观测的训练状态，尚不是已证明的独立原因。
- 两者保存配置都是 upright=1、body_orientation_l2=0；没有依据把历史建议的 -2 误当成当时实际配置。
- AMP 难度规则原样保留：普通地形 d<=0.3 为 1，之后线性降至 d=1 的 0.3；踏石/圆桩为 0。基础系数 0.3，task lerp 0.7。没有新增 AMP 课程。

## 验证与后续训练

尺度补偿候选曾通过 61 项检查，但这些只证明接口和数值合同，不证明它是有效学习修复。撤回补偿、改用统一 0.5 后，G1、跨机器人和稀疏奖励相关 46 项通过。

AMP 修复及相邻合同检查 33 passed；CPU 双进程和真实双 GPU Isaac 检查通过。Black 检查所需模块在训练 Python 中未安装，使用本机已有 Black 完成格式化；最终与 GPU 探针之间仅空白格式差异，部署前逐文件验证 Python AST 相同。

被否决的补偿候选快照：`/home/nubot/phn_ws/t4_train/TienKung-Lab-g1-action-units-20260907`。只运行了 200 轮探针，不能从该目录启动正式训练。

正式入口为 `bash scripts/train_g1_portable.sh`。远端 `TienKung-Lab-g1-amp-fix-20260907`，冻结代码 `d1f7f52`，283 个源码/资产文件和 6 个专家文件已核验 SHA。run `2026-09-07_10-08-58_g1_lightlp_teacher_v3`，GPU 0/2 各 2048 env，预算 30000、resume=false；已推进至 iteration 18，model_0 已保存，所查 loss 有限。TensorBoard 为 `http://100.100.188.39:8032/#scalars`。

旧 v2 随后按用户要求在 9662 轮停止，原训练进程全部退出、GPU 1/3 释放，20 个 checkpoint 保留；未部署任何续到 30000 的脚本。v3 正式完成后执行 5 组固定评估和 2 段连续回放，不能用训练曲线代替行为验收。

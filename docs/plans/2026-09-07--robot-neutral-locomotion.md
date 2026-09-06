# 机器人无关的运动训练算法整理

Status: active。用户要求恢复 10% 全等级随机重置、按难度衰减 AMP，并扫描整个代码库，消除新机器人继承旧机器人实现的结构。

## 工作面和不可缩减的验收

执行代码仍在 `D:\TienKung-Lab-g1-portability-20260906`，分支 `g1-portability-20260906`。原根工作区的用户改动保留。此前 portable_v1 的随机上限 3 已被用户否决，不能把它当现行配方。

1. G1 训练的随机重置比例 0.10、等级范围 0～9；保存配置和运行时采样都要验证覆盖。
2. 用户再次明确 **AMP 恢复原规则**：普通地形 difficulty≤0.3 时倍率 1，随后线性衰减至 difficulty=1 时倍率 0.3；踏石/圆桩始终为 0。撤回全地形从等级 0 衰减的新方案。权重属于当前 transition，不能在自动 reset 后读新 episode 等级。
3. 扫描全部 Python 源码、入口和测试，区分通用算法、机器人定义、兼容入口以及历史实验文件。通用实现迁入算法模块；不能只改名而仍由 G1 继承 T4。
4. T4、G1 以各自机器人配置接入同一环境、AMP、观测和镜像实现；机器人接入必须显式提供身体/关节语义，不得按名字分支或默认套 T4 参数。固定 T4 观测与旧 checkpoint 合同保留。
5. 启动前验证关节顺序、左右镜像、身体选择、AMP 特征与专家宽度、动作尺度和观测维度；接入错误提前失败并给出具体原因。用另一组关节命名/数量证明算法无 T4/G1 特判。
6. 更新训练、评估、回放、学生和运动跟踪中受影响的共享调用，以及当前文档。确属机器人资产、标定、已命名任务和历史产物的名称保留，并记录理由。
7. 窄测试、相邻回归、T4/G1 真实 Isaac probe、独立审查通过后中文里程碑提交；同步并核对远端源码，开启修正后的 G1 新训练和 TensorBoard。训练健康与行为能力分开验收。

## 顺序

- 当前切片：纠正课程合同与 AMP transition 时序；停止已否决的 portable_v1，保留证据。
- 接着：算法/机器人边界迁移，配置组合、观测/AMP/镜像共用实现，配套接入检查。
- 接着：全库调用收敛、文档、行为对照与独立审查。
- 最后：新 lineage 的实际训练、保存配置检查、固定评估和连续回放。

## 已核实的问题

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
- 原始 probe 源码哈希在 `neutral_probe_source.json`；最终训练另立 manifest。全库剩余学生/运动跟踪整理与行为验收继续推进，目标尚未完成。

## v2 已启动

- 训练源码：`4fd6ec262dde7e0e3dd83dce73a71e8bb6916d93`。已逐项核对 211 个代码文件和 6 个 AMP 专家，见 `artifacts/portability/v2/lineage.json`。
- 训练目录：`/home/nubot/phn_ws/t4_train/TienKung-Lab-g1-portability-20260906/logs/g1_loco_teacher_sparse/2026-09-07_01-35-42_g1_lightlp_amp_full_levels_v2`。tmux `g1-full-levels-v2`；TensorBoard `http://100.100.188.39:8031/#scalars`。
- 启动证据保存于 iteration 43：最近 20 轮随机重置比例 10.12%，loss 为有限值，`model_0.pt` 已保存。随后在线确认推进至 iteration 138；保存配置是 0.10/None/None、10 行、AMP start/min 均为 0.3。这只证明训练在更新，不证明越障能力。
- 训练所用远端源码保持冻结。后续学生/运动跟踪代码整理在本地工作树继续；10k 后按训练脚本执行固定评估与连续回放。

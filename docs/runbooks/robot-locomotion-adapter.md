# 接入一个运动训练机器人

公共算法在 `legged_lab/locomotion/`。T4 和 G1 的任务配置分别组合同一 `AmpLocomotionEnvCfg` / `LightLPLocomotionEnvCfg`，新机器人不继承另一个机器人的任务类。

| 内容 | 位置 | 新机器人需要提供什么 |
| --- | --- | --- |
| 物理模型 | `legged_lab/assets/<robot>/` | USD/URDF、碰撞、默认站姿、真实关节限位、PD 增益和力矩限制 |
| 身体语义 | `assets/<robot>/locomotion.py` | `LocomotionRobotSpec`：唯一关节序、左右脚/手、躯干、六项腿关节语义、镜像置换和符号、AMP site 偏移、奖励身体/关节组 |
| 训练配方 | `envs/<robot>/teacher_cfg.py` | 组合共同算法与该资产/spec，声明 AMP 专家路径和宽度、动作单位，以及有依据的机器人参数差异 |
| 注册 | `envs/__init__.py` | 任务名绑定 `LocomotionEnv` 和自己的配置 |
| 公共训练/评估 | `scripts/train.py`、`scripts/eval_locomotion.py`、`scripts/play.py` | 使用 `--task` 选择机器人，不能加入机器人名字分支 |

公共代码按运行时 spec 和 `ObservationLayout` 推导本体、scan、AMP 和镜像索引。控制量按声明关节序进入策略，再映射到模拟器关节序。原 T4 模块只保留兼容入口或机器人配方。已有 task ID、专家文件名、资产名和历史 checkpoint 名称不需要改写。

## 开训前必查

1. `spec.validate()` 检查唯一关节、左右腿/脚、镜像为合法置换且两次镜像还原。启动后用真实 articulation 再检查关节全集、必需身体和命名顺序。
2. `validate_training_contract` 检查真实 Actor/Critic/AMP 宽度、AMP 专家声明关节序、动作标准差数量。AMPLoader 继续核验专家文件元数据和帧宽。专家生成与运行时共用 `AmpFeatureBuilder`。
3. 用真实 Isaac 启动 probe：`bash scripts/nubot_run.sh legged_lab/scripts/probe_locomotion_portability.py --task <task> --output <json> --headless`，必须放在 tmux。它验证随机等级覆盖、AMP 曲线、镜像和自动重置快照，不验证走路能力。
4. 动作单位要明确。原 T4 为 0.25 rad；当前 G1 采用 `0.25 * effort_limit / Kp`。它表示静态误差下的名义 PD 请求比例，不是实际力矩保证。不同语义的 checkpoint 不直接续训。
5. 固定命令 evaluator JSON、绑定源码/模型/专家的 lineage、连续回放共同证明行为。换机器人会改变动力学与可达落脚区域；公共代码和合同只能减少迁移错误，不能保证不同形态得到同样学习曲线。

## 本轮共同课程要求

- 随机重置比例 0.10，10 行地形的范围是 0～9，min/max 均为 `None`。初始出生等级与随机重置范围是两个独立配置。
- AMP 保持原规则：普通地形在 difficulty 0.3 前倍率 1，之后线性降到 difficulty 1 时倍率 0.3；踏石和圆桩恒为 0。
- `amp_reward_coef=0.3`、`amp_task_reward_lerp=0.7` 保持原值；AMP 难度系数读取发生在自动重置之前。

## 深度学生边界

`locomotion/depth_env.py` 提供 `DepthDistillationEnv`（本体史与深度史）和 `LightLPDepthDistillationEnv`（最新本体帧与策略深度）。关节相关尺寸来自教师的 `ObservationLayout`；48×64 深度处理、延迟、噪声与重置时序为共同算法。机器人配置负责相机安装位置、朝向和自己的教师配方；新机器人不能直接复制 T4 的相机标定。

T4 的 10176 维旧学生、3168 维 GRU 学生、1937 维稀疏教师及既有训练配方保留。`rsl_rl` 的学生网络和蒸馏算法已经按传入维度工作；另一个机器人接入时须设置其 `proprio_obs_dim`、教师 scan 偏移等策略参数，再通过自己的相机实测与行为评估。当前只有 T4 有真实学生 RTX 验证；G1/21 关节的学生证据只覆盖观测组装。

公共配置字段在 `legged_lab/config.py`，可以首次独立导入公共环境/教师配置，无需先加载 T4 任务。旧 `envs/base/base_config.py` 只是兼容导出。运动跟踪实现的边界整理仍在进行。

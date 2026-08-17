# T4 depth student `model_24999` 速度上限 lineage 调研

日期：2026-08-17  
工作树：`/home/ssy/桌面/TienKung-Lab`  
目标：解释当前代码为何显示 `lin_vel_x` 上限 `1.0`，而记忆中出现 `3.0`，并区分训练命令、课程最终范围、部署遥控档位和实际行为速度。

## 结论先行

当前 TienKung-Lab Stage S depth student 的训练命令合同是：

```text
lin_vel_x = (-0.6, 1.0) m/s
lin_vel_y = (-0.5, 0.5) m/s
ang_vel_z = (-1.57, 1.57) rad/s
```

`model_24999.pt` 文件本身只保存 `model_state_dict`、`optimizer_state_dict`、`iter=24999` 和 `infos=None`，不含环境配置或 Git commit。因此不能从 checkpoint 字节单独读出 command range；但它的保存路径、时间、网络键和 Stage S 代码 lineage 与上述合同一致，且后续 model_25746 FT run 的真实 `params/env.yaml` 再次明确记录同一 `(-0.6, 1.0)`。在当前可获得证据下，应把 `1.0` 判为 `model_24999` 的训练上限。

明确写成 `3.0 m/s` 的来源不是 VITAL T4 训练配置，而是 `zl_deploy` 的通用遥控器 high speed 档：标准档前向上限 `1.0`，高速档前向上限 `3.0`。该数值是遥控输入的可发布上限，不是 `model_24999` 已训练到的范围，也不是机器人实际达到的速度。

VITAL_Lab 当前工作树里的 T4 文件则是另一套合同：`lin_vel_x=(-0.6, 3.5)`，每个 terrain 的 `max_command` 默认也是 `3.5`。但这些 T4 文件在 VITAL_Lab 的当前 Git `HEAD=8b1a371b523a7d5024f7743c8581a7c4e8ae58a3` 下均为未跟踪文件；该 commit 只初始化仓库，不能作为已提交、可复现的历史训练 lineage。故“3.0 来自 VITAL 原始配置”的说法不成立；可复核的一手 VITAL 源码数值是 `3.5`，不是 `3.0`。

## 四层合同对照

| 层 | 一手证据 | 数值/语义 | 对 `model_24999` 的判断 |
|---|---|---|---|
| 训练 command range | `legged_lab/envs/t4/teacher_cfg.py:279-291`；`legged_lab/envs/t4/depth_student_env.py:37` 的学生环境继承 teacher cfg | `vx=(-0.6,1.0)`，`vy=(-0.5,0.5)`，`wz=(-1.57,1.57)` | 训练上限为 `1.0 m/s` |
| 课程/terrain 最终范围 | TienKung Stage E teacher 只有上述全局 ranges；VITAL 另有 per-terrain `max_command` | TienKung 当前没有把 student 课程扩到 `3.0` 的证据；VITAL 当前源码全局和 terrain 默认均为 `3.5` | 不能把 VITAL `3.5` 或遥控 `3.0` 回写成 student 已训练范围 |
| 部署遥控档位 | `zl_deploy/install/robot_bringup/share/robot_bringup/config/teleop_params.yaml:3-9`；`teleop_joy.py:19-28,94-111,379-395` | std `1.0 m/s`；high `3.0 m/s`；菜单 speed=`std|high` | `3.0` 的确定来源；是输入档位 |
| 实际行为速度 | `artifacts/eval/t4_depth_student_sim2sim/sim2sim_summary.json`；ZL live JSON | model_24999 flat forward command `0.6` 时实际均速 `0.5255 m/s`；loco nav command 由 `0.55` cruise 生成，实际均速 `0.1254 m/s` 且摔倒；ZL model_25746 `vx=0.8` 到楼梯终点，约 `4.59 m/9.46 s=0.49 m/s`（抵达后命令归零） | 没有 `3.0 m/s` 实际行为证据 |

## 1. TienKung-Lab 当前 Stage S 训练合同

### 1.1 当前源码的 command range

`legged_lab/envs/t4/teacher_cfg.py:279-291` 定义 `T4LocoTeacherEnvCfg.commands`：

```python
ranges=CommandRangesCfg(
    lin_vel_x=(-0.6, 1.0), lin_vel_y=(-0.5, 0.5),
    ang_vel_z=(-1.57, 1.57), heading=(-math.pi, math.pi)
)
```

`legged_lab/envs/t4/depth_student_env.py:37` 的 `T4LocoDepthStudentEnvCfg` 直接继承 `T4LocoTeacherEnvCfg`；`legged_lab/scripts/train_t4_depth_student.py:45-68` 创建该环境并交给 `OnPolicyRunner`，没有覆盖 `commands.ranges`。因此 Stage S 学生训练实际消费 teacher env 的这组 ranges，而不是部署遥控器的上限。

Git 证据：

- 该 range 最初由 `5a539b56326e54fe1a61e9079b855243e322abcd`（2026-08-12，`feat(t4): 冻结T4观测合同并落地Stage E特权teacher任务`）引入；`git blame -L 279,291 legged_lab/envs/t4/teacher_cfg.py` 显示 range 行仍归属于该 commit。
- depth student 训练入口和配置由 `890e159ff288e787c5a5f5856ebb46301745f656`（2026-08-14，`feat(t4): add depth-only student distillation training`）引入。
- `b702157756f25f9be772cde81dc11471e578d82f`（2026-08-15 13:49）只把 run name 从 `stage_s_depth_distill` 改成 `stage_s_head35`，没有改 command range；该 commit 的 diff 只涉及 run name、相机和 sim2sim 课程。

### 1.2 `model_24999.pt` 可获得的 checkpoint/lineage 证据

文件：`artifacts/checkpoints/nubot/t4_loco_depth_student_stage_s_head35/model_24999.pt`

只读核验命令：

```text
sha256sum .../model_24999.pt
8040c4ee87b5b0bc3a83c96c654c5afc269cb9354a4b473e2438d350d88d1599
stat: size=13693399, mtime=2026-08-15 15:51:31 +0800
torch.load(...): keys=['model_state_dict','optimizer_state_dict','iter','infos']; iter=24999; infos=None
model_state_dict keys include student.*, teacher.*, depth_encoder.*
```

该 mtime 晚于 `b702157` 的 run-name/相机 commit，文件名和网络结构与 Stage S distillation 配置相符。但 checkpoint 没有 `env.yaml`、`agent.yaml`、`git_sha` 或 command range metadata；因此“从 checkpoint 内部直接证明 `1.0`”是不可能的，必须把代码 lineage 和外部 run artifact 一起使用。

可获得的后续 FT run artifact 提供了同一环境合同的直接序列化证据：

- `artifacts/checkpoints/t4_depth_student_hurdle030_seqref05/params/env.yaml:404-422`：`commands.ranges.lin_vel_x=(-0.6,1.0)`、`lin_vel_y=(-0.5,0.5)`、`ang_vel_z=(-1.57,1.57)`。
- 该 artifact 的 `params/agent.yaml` 显示它是加载已有 depth student 的 `t4_loco_depth_student_ft`，不是把命令范围改成 3.0 的新任务。
- `artifacts/eval/zl_deploy/t4_tienkung_depth_deploy_manifest.json` 表明当前 ZL ONNX 实际 lineage 已是 `artifacts/checkpoints/t4_depth_student_hurdle030_seqref05/model_25746.pt`（`iteration=25746`），而不是原始 `model_24999.pt`；manifest 记录了 checkpoint、params 和 ONNX hash。部署配置/adapter 中的“正式 model_24999 lineage”是 lineage 注释，不是当前实际 checkpoint 路径。

因此更精确的表述是：`model_24999` 是 Stage S 基线 checkpoint，训练代码合同上限为 `1.0`；当前部署制品已是从该 lineage FT 得到的 `model_25746`，其保存的 env params 仍为 `1.0`。

## 2. VITAL_Lab 原始配置核验

### 2.1 当前 VITAL T4 源码写的是 3.5，不是 3.0

当前文件 `VITAL_Lab/vital_lab/envs/t4_27/t4_27_vital_cfg.py:364-376` 写明：

```python
ranges=CommandRangesCfg(
    lin_vel_x=(-0.6, 3.5), lin_vel_y=(-0.5, 0.5),
    ang_vel_z=(-1.57, 1.57), heading=(-math.pi, math.pi),
)
```

`__post_init__`（同文件 `:480-495`）使用 `DIFFICULT_TERRAINS_CFG`。`VITAL_Lab/vital_lab/terrains/terrain_generator_cfg.py:37-43` 定义 `DEFAULT_MAX_COMMAND=(3.5,0.5,1.57)`，`with_max_command` 给 terrain 写入该上限；`DIFFICULT_TERRAINS_CFG` 在 `:232-242` 开启 `curriculum=True`。`VITAL_Lab/vital_lab/envs/base/vital_env.py:120-141,270-275` 的 `TerrainLimitedUniformVelocityCommand` 会按课程列读取 per-terrain max command 并 clamp，因而该当前配置的最终 per-terrain 前向上限仍是 `3.5`，不是 `3.0`。

### 2.2 VITAL Git 历史边界

```text
git -C VITAL_Lab rev-parse HEAD
8b1a371b523a7d5024f7743c8581a7c4e8ae58a3
git -C VITAL_Lab status --short
M README.md, docs/tasks.md, vital_lab/envs/__init__.py, ...
?? vital_lab/assets/t4_27.py
?? vital_lab/assets/t4_27_constants.py
?? vital_lab/envs/t4_27/
?? vital_lab/...
```

`git -C VITAL_Lab show --stat 8b1a371` 只有 `Initialize VITAL_Lab`；`git ls-tree -r 8b1a371 --name-only | grep t4_27` 不包含当前 `vital_lab/envs/t4_27`。也就是说，当前看到的 T4 VITAL 配置是 dirty worktree 的未跟踪源码，不能声称它在 `8b1a371` 历史中已经被提交或被某个已知 checkpoint 使用。即便把当前源码作为一手参考，它提供的是 `3.5`，仍不能解释 `3.0`。

## 3. `3.0` 的确定来源：部署遥控 high mode

### 3.1 配置直接写出标准档和高速档

`zl_deploy/install/robot_bringup/share/robot_bringup/config/teleop_params.yaml:3-9`：

```yaml
max_linear_x_mps: 1.0
max_linear_y_mps: 0.5
max_linear_x_high_mps: 3.0
max_linear_y_high_mps: 1.0
max_angular_z_radps: 1.5
```

`zl_deploy/install/robot_teleop/.../teleop_node.py:26-55` 将这些参数传入 `JoyTeleopConfig`；`teleop_joy.py:93-111` 定义 speed 菜单选项 `("std", "high")` 和 `_speed_mode`；`teleop_joy.py:379-395` 在 `_speed_mode == "high"` 时选用 `max_linear_x_high_mps`，然后把摇杆归一化值缩放成 `RobotIntent.linear_x_mps`。因此 `3.0` 是明确的“遥控器高速档前向输入上限”。这些安装树文件目前是工作树未跟踪文件，不属于 TienKung Git commit；但它们是当前部署栈实际读取的第一方配置/代码。

### 3.2 该 high mode 没有被 T4 depth adapter 限幅到 1.0

`zl_deploy/install/robot_rl_manager/.../t4_tienkung_depth.py:155-166` 直接把 `context.cmd_vel.linear.x/y` 和 `angular.z` 写入学生 proprio frame；该 adapter 没有把输入 clamp 到 `(-0.6,1.0)`。因此 generic teleop high mode 可以把最多 `3.0` 的 command 送进深度学生观测，形成“部署输入范围大于训练范围”的 OOD 风险；这不等于模型已经学会 3.0 m/s。

与之相对，当前 T4 导航脚本使用训练侧合同：

- 未跟踪 `legged_lab/assets/t4/navigation.py:10`：`COMMAND_RANGES={"vx":(-0.6,1.0), ...}`。
- `heading_velocity_command` 同文件 `:35-40` 将 `vx` clamp 到 `COMMAND_RANGES["vx"][1]`。
- `zl_deploy/scripts/run_t4_depth_course_nav.py:45-47,235-251,275-303` 默认 `--vx=0.8`，通过 `CourseNavigator` 生成并发布命令；不是 3.0 high mode。

补充：旧 `t4_sprint.py:241-248` 中的 `3.0` 只是步态周期选择阈值（`command_norm <=1.5`、`<=3.0`、否则），不是 command range；`zl_deploy/config/humanoid/robot_joints.yaml` 中大量 `max_vel: 3.0` 是关节 rad/s 上限，也不是机器人 base `vx`。这两类 `3.0` 都不能作为 depth student 速度训练证据。

## 4. 实际行为速度证据（不能用 command 上限替代）

### 4.1 `model_24999` MuJoCo 固定评估

`artifacts/eval/t4_depth_student_sim2sim/sim2sim_summary.json` 明确指向 `artifacts/checkpoints/nubot/t4_loco_depth_student_stage_s_head35/model_24999.pt`，由 `legged_lab/scripts/eval_t4_depth_student_sim2sim.py:40-58` 定义固定脚本：

- flat forward：固定 command `(0.6,0,0)`，20 s，`actual_forward_mean_mps=0.525537`，无摔倒。
- flat turn：前 8 s `(0.4,0,0)`，之后 yaw `0.6`，全程实际前向均值 `0.130681`。
- flat slalom：前向基准 `0.5`，实际前向均值 `0.428016`。
- loco nav：navigator cruise 默认 `0.55`，实际均值 `0.125373`，在 step 1666 摔倒；不能称为 0.55 稳定行走，更不能外推 3.0。
- hurdles nav：实际均值 `0.364156`，到达 22 m 目标；这仍是 `0.55` cruise 生成的导航命令。
- rule nav：实际均值 `0.027331`，未到达 100 m，说明复杂课程中“命令上限”与“实际行为速度”差异很大。

这些是 model_24999 在 MuJoCo 的连续行为证据；没有任何一项显示 `3.0 m/s`。

### 4.2 当前 ZL 部署制品是 model_25746，实测 command 为 0.8

`artifacts/eval/zl_deploy/t4_tienkung_depth_deploy_manifest.json` 的 behavior evidence 使用 `live_zl_original_policy_depth_vx080_final_run1/2/3.json`。三次均为 `arrived=true`、`final_upright=true`、`final_on_platform=true`，约 9.46–9.69 s 到达楼梯终点；例如 run1 `delta_x_m=4.5902`、最后样本 `t_s=9.4599`，平均路径速度约 `0.485 m/s`。每个 JSON 的最后 command 都是 `[0.0,0.0,0.0]`，因为到达后 evaluator 刹停；不能把终点距离除以总命令档位解释成“3.0 m/s 实际跑速”。

## 5. 已验证事实、未证实假设与剩余缺口

### 已验证事实

1. 当前 TienKung Stage S teacher/student 环境的 `lin_vel_x` 最大值是 `1.0`；后续 FT 的序列化 `env.yaml` 也明确是 `1.0`。
2. `model_24999.pt` 的 checkpoint iteration 是 `24999`，但文件内无 config/commit metadata；SHA256 为 `8040c4ee87b5b0bc3a83c96c654c5afc269cb9354a4b473e2438d350d88d1599`。
3. 当前 VITAL T4 源码（未跟踪）写的是 global `3.5`，terrain default max command 也是 `3.5`；VITAL Git `8b1a371` 没有这些 T4 文件。
4. `3.0 m/s` 明确出现在部署遥控器 high speed 档；std 档是 `1.0 m/s`。
5. 当前 depth adapter 不对 `cmd_vel` 做训练范围 clamp；generic high mode 可以产生 OOD command，但没有 3.0 行为验收。
6. model_24999 固定 MuJoCo 评估实际速度在 `0.027–0.526 m/s`；当前部署 manifest 对应 model_25746，实测楼梯 probe 使用 `vx=0.8`，不是 3.0。

### 未证实/不应继续假设的内容

- 没有找到 model_24999 原始远端 run 目录中的 `env.yaml`、`agent.yaml`、TensorBoard event 或训练启动日志；因此不能给出“该 checkpoint 的远端命令行原文”或训练机器上的独立 config hash。
- 不能声称 model_24999 曾训练过 VITAL_Lab `(-0.6,3.5)` 配置；当前 checkpoint 的代码 lineage 是 TienKung `890e159` Stage S，当前 VITAL T4 源码也未在其 Git 历史提交。
- 不能把 `3.0` 解释为 model_24999 的 curriculum 最终范围、模型能力上限或实际 base velocity；现有一手证据只支持“遥控 high mode 输入上限”。

### 结论性判定

用户记忆中的 `3.0` 最可能且已被一手部署代码直接证实的来源是 `teleop_params.yaml` 的 high mode，而不是 `model_24999` 训练 config，也不是 VITAL T4 原始配置（后者当前源码为 `3.5`）。如果后续要让 T4 depth student 正式覆盖 `3.0 m/s`，必须建立新的、可追溯的训练 lineage（明确 command range、terrain per-command clamp、checkpoint params 和固定速度桶 evaluator）；不能仅把遥控器 high 档或 adapter 输入放大到 `3.0` 就宣称已支持。

## 复核命令摘要

```text
git log --oneline --all -- legged_lab/envs/t4/teacher_cfg.py legged_lab/envs/t4/depth_student_cfg.py
git blame -L 279,291 legged_lab/envs/t4/teacher_cfg.py
git show 890e159ff288e787c5a5f5856ebb46301745f656 --stat
git show b702157756f25f9be772cde81dc11471e578d82f -- legged_lab/envs/t4/depth_student_cfg.py
git -C VITAL_Lab rev-parse HEAD
git -C VITAL_Lab status --short
sha256sum artifacts/checkpoints/nubot/t4_loco_depth_student_stage_s_head35/model_24999.pt
python - <<'PY'  # torch.load 只读核验
...
PY
```


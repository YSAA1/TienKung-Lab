# T4 高质量 Locomotion 迁移背景与当前任务

状态：T4 路线的 living context。已批准的行为与架构合同见
`docs/specs/2026-08-12--t4-unified-depth-locomotion.md`，当前可执行工作面见
`.harness/work_index.md` 中的唯一 active work surface。本文保留迁移背景与现状，
若与已批准 Spec 冲突，以 Spec 为准。

## 1. 项目背景

当前目标机器人是 T4 27DoF 人形机器人。此前主要在
`/home/ssy/桌面/ame_mjlab_height_scan-master` 中基于 MjLab 开发视觉地形行走、
rough terrain 和楼梯任务。原路线已经具备 HeightScan、高程图观测、课程学习、
路线进度和成功率统计，但训练暴露了几个核心问题：

- 任务曾把“到目标点后停住并持续站稳”作为重要目标，容易让到点站立/生存信号
  压过真正的楼梯穿越信号；
- 普通速度跟踪和宽泛 rough terrain 并不能保证学会上楼梯，尤其不能保证提前抬脚；
- 仅用终点距离或向前 progress，机器人可能绕过障碍、滑着下楼或通过 reward gaming
  获得看似不错的训练曲线；
- 总奖励、episode length 或 checkpoint 存在都不能证明已经稳定上下楼梯，必须有
  固定 evaluator 和行为播放证据。

迁移到 TienKung-Lab 的原因不是简单换一个训练框架，而是复用其已验证的
`velocity tracking + PPO + AMP` locomotion 主干，先快速得到高质量、自然、稳定的
T4 基础走跑策略，再在该基础上加入 rough、楼梯和路线型越障 MDP。

## 2. 最终诉求

需要一个统一 T4 loco policy，而不是站立、走路、跑步、上下楼梯分别训练多个策略。
最终策略应覆盖：

- 近零速度稳定站立，但不能靠长时间原地存活刷主要奖励；
- 前进、后退、侧移、原地转向和转弯；
- 从走路连续过渡到慢跑/跑步；
- rough terrain 上保持稳定和低滑移；
- 使用深度相机提前调整落脚，稳定上楼和下楼；
- 在路线型障碍任务中不能从障碍侧面绕开。

优先级是快速得到真实行为结果，不做不必要的多策略、MoE、多阶段复杂系统或大规模
抽象。第一版应尽量沿用 TienKung-Lab 的成熟 PPO+AMP 训练链，只做 T4 合同和任务
MDP 所必需的改动。

## 3. AMP 与任务 MDP 的职责边界

AMP 只回答“动作像不像示范中的自然运动”：

```text
expert transition (s_t, s_t+1) -> discriminator
policy transition (s_t, s_t+1) -> discriminator
discriminator style reward + task reward -> PPO
```

AMP 不负责决定去哪里、是否穿过楼梯、是否绕过障碍，也不与某一条 expert motion
逐帧同步。速度、方向、地形穿越和成功结束必须由 task command/reward/termination
定义。

第一版采用一个 discriminator 混合站立、走路、慢跑、转向数据。只有出现可复现的
风格冲突，例如平地 AMP 明显压制上楼抬腿时，才考虑 command/style conditioning
或第二个 discriminator。不要预先过度设计。

## 4. 当前已迁移资产

T4 机器人资产位于：

```text
legged_lab/assets/t4/
  constants.py          # 27DoF 唯一关节顺序
  t4.py                 # IsaacLab ArticulationCfg
  urdf/t4_std.urdf
  mjcf/t4_std.xml
  meshes/
```

资产来源是 AME/MjLab 已使用的 T4 27DoF 模型。控制合同沿用该模型的关节名称、
关节限制、effort/velocity limit、KP/KD 和 armature。当前资产配置使用 URDF 由
IsaacLab 转换为 USD；正式训练前必须在实际 IsaacLab 环境完成一次 spawn smoke，核对
27 个关节、body 名称、质量、碰撞和初始站姿。

冻结的 27DoF 顺序是：

```text
左臂7 + 右臂7 + 腰yaw1 + 左腿6 + 右腿6
```

代码中的唯一真值是 `legged_lab/assets/t4/constants.py::T4_JOINT_NAMES`。

## 5. 当前动作数据

用户提供的数据源是仓库根目录下的 `steering_t4_27.zip`。已迁移 18 条动作到：

```text
legged_lab/envs/t4/datasets/motion_source/
```

动作包含站立、前进/后退、侧移、原地左右转、45/90/180 度转向、慢跑和跑步。
原始 CSV 每行严格为：

```text
root_xyz(3) + root_quat_xyzw(4) + T4 joint position(27)
```

当前按照 `30 Hz` 解释。转换脚本：

```bash
python legged_lab/scripts/t4_csv_motion_conversion.py \
  --input legged_lab/envs/t4/datasets/motion_source \
  --output-dir legged_lab/envs/t4/datasets/motion_visualization \
  --fps 30
```

已生成 18 个 `motion_visualization/*.txt` 中间文件，每帧 66 维：

```text
root_xyz(3) + root_euler_xyz(3) + q(27)
+ root_linear_velocity(3) + root_angular_velocity(3) + dq(27)
```

历史审计认为 16 条 locomotion clip 可作为首轮候选；`t4_stand` 只用于站立/reset，
`t4_run` 因过激且存在关节限位风险，首轮训练先 hold out。所有数据仍需在 IsaacLab
T4 资产上播放核验，检查脚底穿透、滑移、关节错序和 root 高度。

## 6. AMP 数据尚未完成的部分

当前 `motion_visualization` 不是最终 AMP expert。下一步必须让 T4 在 IsaacLab 中按
这些轨迹播放，然后通过目标 T4 模型计算 runtime 同定义的 AMP state。

推荐第一版 T4 AMP state：

```text
selected/full q(27) + corresponding dq(27)
+ left/right hand position relative to root(6)
+ left/right foot position relative to root(6)
= 66 dimensions
```

判别器 transition 输入为 132 维。最终字段顺序不是建议文本决定的，必须由一个共享
函数同时生成 expert 和 runtime AMP observation，并由测试验证完全一致。不能复用
TienKung 的硬编码 20DoF/52D loader，也不能离线复制其他机器人末端位置。

## 7. 训练任务设计

### 第一阶段：统一 depth-aware 基础 loco

- 使用一个 policy 覆盖站立、前后侧移、转向和基础行走，慢跑在该 checkpoint 通过后累积
  加入；
- Actor 从第一阶段起固定使用深度历史、本体历史、速度 command 和上一动作；深度图经
  轻量 CNN 编码，不直接 flatten 进入 MLP；
- task reward 负责速度/角速度跟踪、姿态、安全、低滑移和能耗；
- AMP 首轮使用审核通过的站立、行走、后退、侧移和转向数据提供较弱的动作自然度 prior；
- `t4_run` 首轮不加入，稳定后再单独评估是否纳入；
- 第一轮先在 flat + 轻量 rough 上训练，避免同时引入完整楼梯状态机。

### 第二阶段：rough + 上下楼梯

- 保持同一个 Actor observation/action/network 合同；
- 提高深度感知地形难度，让 Actor 看见前方台阶；HeightScan、高程图和接触真值不得进入
  Actor 或导出接口；
- rough 用于摩擦、落脚和扰动鲁棒性，不替代楼梯专用任务信号；
- 上下楼梯作为一个统一 traversal task，episode 同时包含上楼和下楼；
- 成功后立即结束 episode 或进入下一段，不在终点持续站着刷高奖励。

### 防绕障最小合同

路线型任务必须组合使用：

```text
几何 corridor + ordered gates/waypoints + 越界失败 + strict success
```

成功条件应是按顺序穿过所有 required gate、保持在 corridor 内、未跌倒/未触发禁止
接触并通过出口。仅有 centerline reward、终点距离、向前 progress 或把障碍放在 tile
中间都不足以防绕行。课程晋级必须按固定 evaluator 的严格成功率，而不是总奖励。

## 8. 质量与验收标准

训练启动前：

- T4 URDF 能在 IsaacLab 成功 spawn，27DoF 和 body 顺序确认；
- 所有候选动作能播放，无明显穿地、关节错序和不连续；
- AMP expert/runtime shape、字段顺序、坐标系完全一致；
- 观测、动作或 AMP 合同改变时不续训旧 checkpoint。

基础 loco 验收不能只看 reward。至少按固定 command buckets 检查：

- 零速站立成功率和漂移；
- 前进、后退、侧移、转向 tracking error；
- 走跑切换是否连续；
- 跌倒、脚滑、禁止接触和关节硬限位；
- MuJoCo playback/sim2sim 行为。

楼梯与 rough 验收至少检查：

- 上楼、下楼分别统计成功率；
- 是否提前抬脚，而不是撞台阶后补偿；
- 下楼是否滑步或直接坠落；
- route gate 是否按顺序通过，绕行成功率必须为零；
- rough/stairs 每个 terrain bucket 的成功率、滑移、跌倒和 hard violation。

## 9. 当前边界与下一步

### 2026-08-16 状态快照

走跑阶段已关闭：Stage E teacher 主线已把连续跨栏作为 terrain bucket 并入课程
（`stage_e_prov7_hurdle`）；深度 student `stage_s_head35/model_24999.pt` 已完成训练预算，
用户通过仓库原始 `sim2sim_t4_depth_student.py --course loco` MuJoCo 交互环境人工复核。
该声明限定为 T4 loco/深度 loco 阶段，不等同于 100m `rule` 全路线、真机部署或翻箱合并完成。

当前双工作面：翻箱仍在 zhuoqun，计划
`docs/plans/2026-08-15--t4-vault-g1-g2-recovery-plan.md`；稀疏落足执行面是
`docs/plans/2026-08-17--t4-sparse-ab-rollback-plan.md`。旧 25k 与 S1d A/B 都已否决。
T-compat v2 @3874 踏石 `reach_2m=0.80`，圆桩≈0。S1d 厨房水槽 10k 未救圆桩，还把踏石从零砸掉。
能力声明仍由 fixed evaluator 和连续回放决定。
本机 Docker 已补齐（`t4-isaac-jammy:v2` + Isaac Sim 5.1 + IsaacLab 2.1.0），旧学生
0.30 FT 在 tmux `t4-student-ft`（32 env，`logs/t4-student-hurdle030-ft.log`）。
G3 保持 blocked。旧 merge plan 在 `docs/archive/plans/`。runtime 以
`.harness/state.md` 和 `.harness/decisions.md` 为准。

`model_25746.pt` 的 ZL stair sim2sim gap 已闭环到可接受范围。真机模型名为
`t4_tienkung_depth`，ZL parity alias 为 `t4_tienkung_depth_sim`，两者复用同一 ONNX。
deterministic sim 不再使用 vendor 29DoF plant，而是直接加载原始 27DoF `t4_std.xml`，
复用 direct position servo。最终 gap 根因是进入 `RL_ACTIVE` 前已到达的首个 policy seed
action 被 driver 丢弃，driver 使用了带 noise/cutoff 的 MJCF gyro 而 direct 读取
`qvel[3:6]`，以及原生 `270×480` float 深度图经 CycloneDDS 传输后断流。当前 sim 仍按
`270×480` 渲染，在 driver 内执行训练等价的 clip/area resize 后只传 `48×64` policy 深度；
三项对齐后，`vx=0.8` 独立冷启动楼梯严格评估 3/3 通过。ZL 三次均值相对 direct 的
最低/最高高度、最终 x、最终高度绝对 gap 分别为
`0.0091/0.0026/0.0498/0.0011 m`；最终 y gap 为 `0.0884 m`。

M0 资产、相机与 motion 事实闭环已于 2026-08-12 全部通过：nubot IsaacLab 上完成
joint/body/foot/hand/camera discovery 与 18 条 motion headless playback（0 reject），
人工视觉复核基于 `artifacts/motion_review/` 的三视角回放裁定通过，accept 17 条，
`t4_run` 因 hard joint limit violations 继续 held out。已接受的缺陷是 clip 不对齐地面、
接触时序不可用；66D AMP feature 只含关节量与 root 相对量，不受该缺陷影响。

已注册 teacher 任务 `t4_loco_teacher`（`legged_lab/envs/t4/`）。历史 `prov1`–`prov6`
lineage 已被后续修订接替；当前主线是含跨栏 bucket 的 `stage_e_prov7_hurdle`，并保留
`stage_e_prov6_local` 作为无跨栏对照。teacher fixed evaluator 仍未通过；深度 student
当前只有 `model_25746.pt` 在指定 stair XML、`vx=0.8` 下的 direct/ZL 行为验收，不能外推
到完整 100m 路线、其他复杂地形或真机能力。

下一步：

1. 把 provisional AMP expert 提升为 formal 并补 lineage 记录；
2. 建立 teacher 固定 evaluator，跑通基础 walk/jog/rough/上下楼 buckets；
3. 进入翻箱 G1/R3→R4 recovery；走跑阶段产物只作为独立 loco teacher/对照，不直接替代翻箱 expert。

# T4 Teacher 与深度学生部署手册

状态：当前实现合同（2026-08-16）。本文说明如何把现有模型接到实机传感器和低层关节控制器；它**不是**真机安全验收，也不证明可直接上人形机器人跑障碍。

## 先选部署路线

| 路线 | 部署的模型 | 需要的感知 | 适用条件 |
| --- | --- | --- | --- |
| 高程图 teacher | 1155D teacher actor | 肚子模块输出的局部高程图 + 本体状态 | 已有可靠的局部 elevation map，并能按下文重采样 |
| 深度学生 | `model_25746.pt` | 头部前向下视深度相机 + 本体状态 | 希望只靠机载相机感知地形 |

两条路线的关节顺序、动作缩放和控制频率相同，但**输入不能混用**。teacher 不能直接喂深度图；深度学生不能直接喂 195 格高程图。

当前深度学生进度 gate 的候选是：

```text
/home/ssy/桌面/TienKung-Lab.worktrees/t4-student-ft-fix/
logs/t4_loco_depth_student_ft/
2026-08-15_21-29-41_hurdle030_sequence_source_ref05_probe150/model_25746.pt
```

在删除该 worktree 前，应把 checkpoint、同目录 `params/` 与
`artifacts/eval/t4_depth_student_hurdle030/source_ref05_25746/lineage_manifest.json`
一起复制到受管的部署制品目录。它通过的是 MuJoCo `loco` **progress** gate；不等价于零碰杆、100m rule 或真机验收。

## 共用的控制侧合同

1. 控制器以 `50 Hz` 调用网络。
2. 网络给出 27 维原始动作 `a`，按下式变成位置控制目标：

   ```text
   q_target = q_default + 0.25 * a
   ```

3. `q_default`、27 个关节的顺序和 PD 参数必须从 T4 资产/部署控制器统一读取；不可按电机编号猜测顺序。
4. 最先只允许站立和低速人工速度命令；急停、关节限位、力矩/速度限幅、失联 hold 都必须由网络之外的安全层处理。

## 路线 A：部署高程图 teacher

### 肚子模块要提供什么

传感器可以装在肚子，只要其软件能持续产出**以机器人当前朝向为前方**的局部地形高度。传感器本身可以是下视深度相机、激光雷达或已有 elevation-map 模块；teacher 不关心原始点云来自哪里。

每个控制周期，把局部地图重采样为下面的 195 个点：

```text
前方 x：0.2, 0.3, ..., 1.6 m       （15 列）
左右 y：-0.6, -0.5, ..., +0.6 m    （13 行）
```

`x` 是肚子朝向的前方，`y` 是左右；不是世界坐标。数组顺序固定为：先取最左一行 `y=-0.6`，在该行内按 `x=0.2 -> 1.6` 排列，再取下一行。即 `y` 为外层循环、`x` 为内层循环。

每格不要发送绝对海拔，而是发送：

```text
scan[y, x] = clip((z_base - 0.5) - z_terrain[y, x], -1.0, +1.0)
```

- `z_base`：机器人躯干/浮动基座离地高度，由状态估计器给出。
- `z_terrain`：同一世界高度基准下该格地面或障碍顶部的高度。
- 平地站立时该值约为 `0.35`；30 cm 高栏处会更小；坑会更大。
- 若你的地图直接给的是 `z_terrain - z_base`，则直接使用 `clip(-0.5 - map_value, -1, 1)`。

训练里从高处向下采样只是仿真取得地面高度的手段，不要求在机器人上方再装一个传感器。实机只要让肚子模块最终输出上述 15×13 数组即可。

### teacher 的其余输入

teacher 一次输入为 `1155` 维：最近 `10` 帧、每帧 `96` 维本体状态，加上当前 `195` 维高程图。

本体状态包含：机身角速度、重力在机身坐标下的方向、局部速度命令、27 个关节角度偏差、27 个关节速度、上一动作以及步态时钟。部署适配器必须沿用训练的字段顺序与归一化；不要仅把高程图接上就调用 actor。

### 实机接入顺序

1. 静止时打印 15×13 图：平地应近似为常数；在前方放 30 cm 箱/栏时，对应格子必须变小。
2. 机身原地转向：同一个障碍必须始终出现在“机身前方”对应格子，而不能固定在世界坐标格子。
3. 离地悬空或地图失效时，停止向 actor 送数据并进入安全 hold；不能拿全零图替代。
4. 先吊装、低速平地，再逐步加入单个低障碍。每一步记录高程图、1155D 输入、动作和实际关节目标，便于回放。

## 路线 B：部署头部深度学生

### 相机安装位置

当前学生的真实训练契约是**头部高度**，不是肚子中部。把相机固定在头壳前方或上胸最高前缘的正中，镜头中心相对机器人躯干中心为：

```text
向前 0.085 m
左右 0 m
向上 0.42 m
镜头中心光轴：向前下俯 35°
```

相机离地高度会随机器人实际站姿变化，因此不要按一个固定离地数字钻孔：先按相对躯干的 `前 8.5 cm、上 42 cm` 固定，再实测镜头中心离地高度 `h`。相机保持正中、无左右偏航和侧滚；先对齐机器人正前方，再绕左右轴向下转 35°。可用 35° 楔形支架，或在水平地面上用角度仪对准镜头光轴校验。水平地面上的中心光轴落点应在 `h / tan(35°)` 前方；例如 `h=1.32 m` 时约为 `1.9 m`。

### 图像预处理

相机输出必须按以下次序处理：

```text
输入：480×270 的 Z-depth（沿光轴的距离，单位米）
无效/NaN/Inf：填 1.0 m
裁剪：限制到 0.2～3.0 m
归一化：(depth - 0.2) / 2.8
缩放：area resize 到 48×64
历史：保存最近 3 帧
刷新：每 3 个 50 Hz 控制周期更新一次，其余周期保持上一帧
```

这里要使用相机的 Z-depth，不是点到相机的欧氏距离，也不能把未经训练的点云高度图替代深度图。若相机原生分辨率不同，应先保证同样的视场覆盖，再得到 480×270 或等价 16:9 图像后执行上述缩放。

### 学生输入与接入顺序

学生输入是最近 10 帧本体状态加 3 帧 `48×64` 深度图。深度编码器在模型内部处理图像，因此部署程序应输入完整的原始 observation，不应自行只拼一个 1088D 的 MLP 输入。

1. 机器人直立、平地：确认图像下半部分连续看到地面。
2. 在 `h / tan(35°)` 前放地面标记：它应接近画面中心；若偏差大，先改相机外参，而不是直接上策略。
3. 分别放 25/28/30 cm 栏杆：三者都应在进入前保留于图像内，且预处理后的深度值连续。
4. 录制真实相机流，离线复现预处理，和仿真同场景深度截图人工比对后，再进行吊装低速测试。

### 本机 zl_deploy MuJoCo 启动

所有长运行必须放在 tmux。使用下面的专用入口：

```bash
tmux new-session -s zl-depth-bringup \
  'bash /home/ssy/桌面/TienKung-Lab/zl_deploy/scripts/run_t4_depth_sim2sim.sh'
```

该入口会自动完成：

1. 启动 `zl_robot.launch.py` 和 T4 MuJoCo。
2. 在 ready 插值早期自动 Reset，避免默认零姿态在控制接管前倒下；不需要人工点击 Reset。
3. 等待高度 `0.82–0.88m`、roll/pitch `<2°`、最大关节速度 `<0.1rad/s` 连续稳定 2 秒。
4. 在 READY 状态预选 sim-only alias `t4_tienkung_depth_sim`，收到首个真实导航命令后才进入 RL_ACTIVE。

bootstrap 结果写到 `/tmp/t4_depth_sim_bootstrap.log`。成功标志类似：

```text
[BOOTSTRAP] ready altitude=0.845m ... max_joint_speed=0.000rad/s model=t4_tienkung_depth_sim
```

当前部署修复要求：`32FC1` 深度按米处理，`16UC1` 按毫米处理；T4 控制栈的
`ankle_in_series` 必须为 `true`。这两项错误都会让 checkpoint 在数秒内失稳。

真机模型 `t4_tienkung_depth` 与 sim alias `t4_tienkung_depth_sim` 使用同一份
`t4_tienkung_depth.onnx`。alias 只用于抵消 ZL simulator 与 direct MuJoCo 的执行器差异：

- vendor simulator 会把左右 ankle pitch 的命令 Kp/Kd 硬乘 4；sim profile 使用 Kp=20，
  进入 simulator 后等效为训练/direct 的 80，真机配置保持 80/4；
- 训练 Kd 由 `prepare_t4_depth_sim_xml.py` 写入 MuJoCo joint damping，sim profile 的命令 Kd
  全部为 0，避免显式 PD 与 implicit damping 重复；
- vendor 二进制存在 direct 没有的 torque-speed envelope，膝关节实测速度可超过其
  `14.66rad/s` 阈值；`prepare_t4_depth_sim_binary.py` 只为 parity run 生成临时二进制，
  原 vendor 二进制和真机均不修改。

模型切换时，depth policy 必须自己生成 activation seed，旧模型 output smoother 不得覆盖
第一帧。hardware feedback 使用 Best Effort QoS。200Hz camera patch 和 feedback step-sync
inference 都只得到 1/3 通过，已经撤销，不得恢复。

ZL MuJoCo 的场景也必须与 direct evaluator 对齐。旧
`T4_std_add_head/xml/t4_std_add_head.xml` 在 `x=1.31 m` 起放置了两组半宽 `5 m`
的楼梯条带，其边界正好贴住中心线 `y=0`；机器人轻微横漂就会提前撞上侧边楼梯，
造成“速度偏小、持续转向、高速翻倒”的假 sim2sim gap。当前 flat 部署场景已删除该
terrain gallery，并把地面改为 `condim=3, friction="1 0.005 0.0001"`，与 direct
MuJoCo 接触合同一致。回归测试要求 world-body 碰撞几何只有 `ground`。

同一 `model_25746.pt` 的 15 秒 flat A/B 结果：

| 命令 | direct MuJoCo 前进 | 完整 ROS/ZL 前进 | direct / ZL 最低高度 |
| --- | ---: | ---: | ---: |
| `vx=0.3` | `3.35 m` | `2.97 m` | `0.842 / 0.842 m` |
| `vx=0.8` | `9.33 m` | `9.41 m` | `0.819 / 0.822 m` |

因此 flat plant 与闭环部署已经基本对齐；不要再使用旧 terrain gallery 的表现判断
policy 或部署速度。楼梯测试必须让 direct 与 ZL 加载同一份 stair/loco 几何后再比较。

### 0.8 m/s 楼梯对照

训练和 direct MuJoCo 默认使用 `heading_command=True`：策略输入不是恒定
`[vx, 0, 0]`，而是 centerline/目标航向闭环生成的 `[vx, 0, wz]`。T4 本体存在轻微
横漂，固定 `wz=0` 时 direct 和 ZL 都可能走出 3 m 宽楼梯；这不是 ZL plant 独有失败。
共享命令真值在 `legged_lab/assets/t4/navigation.py`，direct 与 ZL evaluator 必须复用它。

deterministic ZL sim 默认直接加载原始 `legged_lab/assets/t4/mjcf/t4_std.xml`，并复用
direct runner 的 position servo、Kp/Kd 和 effort limit。已确认的 gap 根因是首个 policy
seed action 会比 `RL_ACTIVE` 状态早到一个 ROS callback；旧 driver 丢掉该 action，实际从
下一帧才推进物理，因此在第一阶接触前已经偏离 direct。当前 driver 会缓存并在进入
`RL_ACTIVE` 时执行这个首帧。另一个冷启动波动来源是旧 driver 使用带 noise/cutoff 的
MJCF gyro sensor，而 direct 直接观察 `qvel[3:6]`；当前已统一为后者。每个 policy action
固定推进 4 个 `0.005s` physics step。

原生 `270×480 32FC1` 深度图若按 50Hz 直接经过 CycloneDDS，运行约十几到几十个 action
后会停止到达 RL manager；`depth_image` 超过 `0.25s` 后变为 missing，状态机退出
`RL_ACTIVE`。这就是改成原生分辨率后“站着不走”的根因。当前 deterministic driver 仍在
原始 `270×480` 相机上渲染，但在进 ROS 前执行与训练一致的无效值替换、`0.2–3.0m`
clip 和 adaptive area resize，只传 policy 实际消费的 `48×64` 米制深度。该 NumPy resize
与 PyTorch `area` 的随机输入 max diff 为 `7.2e-7`；真实相机的原生 `270×480` 路径仍由
adapter 直接处理，不受 sim transport 优化影响。

同一 stair XML、同一 `model_25746.pt`、`vx=0.8 m/s` 的结果：

| 运行时 | 最低/最高高度 | 最终位置 `(x, y, z)` | 严格结果 |
| --- | ---: | ---: | --- |
| direct MuJoCo | `0.819 / 1.930m` | `(4.646, -0.169, 1.924)m` | 稳定停在楼梯顶 |
| ROS/ZL 原始 plant 冷启动 1 | `0.808 / 1.933m` | `(4.590, -0.076, 1.922)m` | `success=true` |
| ROS/ZL 原始 plant 冷启动 2 | `0.809 / 1.933m` | `(4.620, -0.119, 1.914)m` | `success=true` |
| ROS/ZL 原始 plant 冷启动 3 | `0.811 / 1.933m` | `(4.581, -0.046, 1.932)m` | `success=true` |

三次 ZL 均值相对 direct 的绝对 gap：最低高度 `0.0091m`、最高高度 `0.0026m`、
最终 x `0.0498m`、最终高度 `0.0011m`、最终 y `0.0884m`。严格成功要求到目标距离
`≤0.4m`、平台高度 `≥1.85m`、倾角 `<10°`、连续保持 1 秒、全程最低高度 `>0.70m`、
峰值高度 `>1.90m`，并且 RL_ACTIVE 的 desired→accepted→hardware→feedback 无中间丢失。

正式 ZL 楼梯入口：

```bash
tmux new-session -s zl-depth-bringup \
  "T4_DEPTH_SIM_DRIVER=deterministic T4_DEPTH_SIM_PLANT=original \
   T4_DEPTH_NAV_GOAL_X=4.6 T4_DEPTH_NAV_GOAL_Y=0 \
   T4_DEPTH_NAV_VX=0.8 T4_DEPTH_NAV_DURATION=15 \
   T4_DEPTH_NAV_OUTPUT=/home/ssy/桌面/TienKung-Lab/artifacts/eval/zl_deploy/live_zl_stairs_course_nav_vx080.json \
   bash /home/ssy/桌面/TienKung-Lab/zl_deploy/scripts/run_t4_depth_sim2sim.sh"
```

最终行为证据为 `artifacts/eval/zl_deploy/direct_stairs_nav_vx080_seed_action.json` 与
`live_zl_original_policy_depth_vx080_final_run1/2/3.json`。对应 driver 日志中的所有 policy trace 均为
`physics_steps=4`。

机器核验：`/usr/bin/python3 zl_deploy/scripts/verify_t4_depth_deployment.py`。

本机 deterministic ZL 的相机渲染保持 `270×480`，ROS transport 为处理后的
`48×64 32FC1` 米制深度；不再使用旧 vendor C++ 硬编码的 `36×64` 相机图。

## 上机前的共同门槛

- checkpoint、观测 schema、关节顺序、默认站姿、动作缩放必须来自同一 lineage。
- 高程图或相机时间戳若落后于控制循环，丢弃该帧并进入安全 hold；不要静默复用过久的感知。
- 先做 output-only 回放，再做无脚离地的关节目标检查，再做吊装测试；不允许首次接入就自主跨栏。
- 当前仓库没有这两条路线的真机 HIL/安全验收记录。实机部署必须另建记录，至少保存输入感知、网络输入、动作、关节状态、IMU、命令与急停事件。

## 代码真值

- teacher 高程图范围、维度、编码：`legged_lab/assets/t4/schemas.py` 的 `TEACHER_SCAN_*`。
- teacher 高程图生成与输入拼接：`legged_lab/envs/t4/t4_env.py::compute_teacher_terrain_privilege`。
- 深度相机位置、俯角、预处理：`legged_lab/assets/t4/schemas.py` 的 `DEPTH_*`，以及 `legged_lab/envs/t4/depth_student_env.py`。
- 头部安装点的资产证据：`legged_lab/assets/t4/mjcf/t4_std.xml::forward_camera`。

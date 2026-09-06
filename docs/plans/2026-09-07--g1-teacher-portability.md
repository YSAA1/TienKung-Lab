# G1 教师通用性修复与重训

Status: active。用户已授权排查、修改和重新训练。

## 工作面与验收

- 本机工作树：`D:\TienKung-Lab-g1-portability-20260906`，分支 `g1-portability-20260906`。
- nubot：`/home/nubot/phn_ws/t4_train/TienKung-Lab-g1-portability-20260906`。
- 原始 `D:\TienKung-Lab` 的未提交改动保留；本次基于快照提交 `7e6029a` 单独记录差异。
- 实现检查：真实 reset 前 AMP 快照、关节动作单位、G1/T4 观测和配置隔离；pytest 合同。
- 能力检查：固定 `vx=0.7`、32 环境/32 episode，flat、stepping_stones、raised_pillars；稀疏地形中心出生，真实洞。
- 训练只是开始试验。通过 evaluator、lineage 和连续回放后才能声称改善了过桩能力。

## 已复现的失败

旧 checkpoint：`TienKung-Lab-s12-gru-ppo/logs/g1_loco_teacher_sparse/2026-09-04_16-28-27_g1_sparse_teacher_g1term/model_39999.pt`。

| 固定评估，d=0 | strict success | reach 2m | 平均前进 | 终止 |
| --- | --- | --- | --- | --- |
| flat | 32/32 | 32/32 | 4.653 m | 32 oob |
| stepping_stones | 0/32 | 0/32 | 1.464 m | 32 collapsed |
| raised_pillars | 0/32 | 0/32 | 1.439 m | 32 collapsed |

证据：远端 `artifacts/portability/baseline_*.json`。评估时仍使用旧动作尺度和旧训练配置，仅修正评估器的观测/关节索引。
旧逐帧回放 `D:\TienKung-Lab\artifacts\replay\g1_g1term_m39999` 也有相同摔倒，0.20 m 的 collapse 没有漏杀。
40k TB 末窗：整体 tracking 0.376，collapsed 0.733；踏石/圆桩进度约 1.5 m、晋级 0。TB 仅作诊断，不替代上述固定评估。

## 差异与确定性

1. **确定的训练数据错误**：`T4LocoEnv.step` 已自动 reset；AMP runner 再读一次 getter 得到新 episode 状态，错误连接成前一局 transition。两种 AMP 宽度 66/70 的回归检查均先失败。现在 reset 前保存紧凑 `terminal_amp_obs`，只覆盖终止行；下一步策略继续使用重置后的观测。旧非 T4LocoEnv runner 用户保留兼容路径，其终止合同不在本次范围内。
2. **确定的评估器不通用**：扫描从 960 起、诊断动作按 T4 关节索引切片；G1 本体历史实际是 1020。改为读取实际 buffer 宽度和运行时关节映射。G1 scan `[1020,1995)`，T4 仍 `[960,1935)`。
3. **已量化的控制尺度差异，训练作用待验证**：同样 `0.25 rad` 的 unit action，G1 脚踝约请求 14.3% 名义力矩，腰 roll/pitch 为 107.1%，髋 pitch 为 36.0%；T4 ankle pitch 为 27.8%。不能把相同弧度输入当成相同探索强度。加入可选 `scale_j = 0.25 * effort_limit_j / Kp_j`。这是静止误差下的 PD 请求比例，不保证实际力矩比例，也不替代硬件限幅。增益、力矩限制、碰撞和站姿均沿用当前资产。
4. **课表难度对不同形态不等价**：G1 继承 T4 的 0.6–2.0 m/s 稀疏命令、8 m 砖、9–24 cm 踏石高度、10% 全难度随机重置；初次学会离散落脚时没有速度渐进。T4 S12 还继承了 S11b 的 19000 checkpoint；不能只比较最后一段训练时长。新配方在 level 0 开始，随机回访限定 0–2，正常成功晋级不受此限制；稀疏速度随 level 从原范围的 0.5 倍线性增长到 1 倍。
5. **尚存的机器人适配项**：关节和脚名、默认姿态、PD 参数、AMP 专家/特征、镜像符号仍须各机器人提供。足底扫描区域和多项物理惩罚仍是绝对单位，不宣称本次已实现任意机器人零配置。G1 AMP 和镜像已有专项合同检查，未发现关节顺序错误。扫描非零敏感性也不等于策略会正确落脚。

## 本次试验

- `g1_sparse_teacher_portable_v1`，冷启动，PPO/AMP/mirror 网络与权重保持原配置。
- G1 动作尺度按关节名义力矩比例换算；T4 配置默认仍为原 `0.25 rad`。
- 同一混合地形、真实洞，20% 踏石 + 20% 圆桩。速度渐进属于单次训练的课表，不拆多个正式阶段。
- 63°、40 m/s²、torso 接触、0.20 m collapse 等终止保留。
- 首轮预算 10000 iterations，GPU 1+3，各 2048 env，save interval 500。
- 改了动作语义，旧 `model_39999` 只作为基线，不能直接按新动作尺度续训。新 checkpoint 必须绑定本次动作尺度/源码。
- 不用数值稳定、reward 或 checkpoint 存在宣称成功。首轮完成后评估 flat/easy sparse；若仍为 0/32，应复盘落脚与摆腿控制，不重复同配方 40k。

## 验证记录

- 本机 `D:\anaconda\envs\pytorch\python.exe`：当前 167 passed，1 skipped（G1 可选 MuJoCo 检查）；Windows 全局 pytest 临时目录权限错误通过专用 `--basetemp` 重跑该项解决，没有改环境。
- 运行时 probe、正式启动和首个 checkpoint 信息在同目录的执行记录与 `artifacts/portability/lineage.json` 中补齐。

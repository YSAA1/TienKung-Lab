# Executable Plan - s6 baseline 与 s7 格点/碰撞组合修复

> Status: active
> Date: 2026-08-20
> Spec: `docs/specs/2026-08-15--t4-stepping-stones-and-hurdle-stable.md`（目标规格；双老师路线已废）
> 对照: `docs/research/2608.02653v1/auto/2608.02653v1.md` §IV（研究笔记非权威）
> 取代: `docs/archive/plans/2026-08-18--t4-sparse-terrain-index-fix-plan.md`（s5 列映射/MDP 已落地；梅花桩能力未起）
> Branch: `t4-train`
> Planning surface: docs plan
> GPU: nubot 四卡从零训 `t_sparse_lightlp_s6`。zhuoqun 翻箱不动。Stage E 1155D 不覆盖。

## Objective

s6 从零验证放宽 easy 几何是否能让稀疏地形起势。后续配对 evaluator 已推翻“策略不会抬脚”这个前提，并确认旧 9×9 格点与 4 m gate 不一致。S7/S8 随后同时铺满格点并激活躯干/小腿 primitive collision，但本地正式回放又确认第三个物理 bug：Isaac 将 0.28 m Shank cylinder 转成 capsule 后与脚 collider 持续自重叠。S8 已冻结，额外 `termination_penalty=-200` 撤回；修复后的 S9 已从未经历假自碰的 s6 `model_31000.pt` 重开。

## Active Slice

阶段 3 健康门和深层诊断已完成。s6 已停，`model_31000.pt` 作为下一正确 lineage 的 warm-start；它属于旧 9×9 / 非足碰撞缺失 baseline，但没有经历 Shank 假自碰。S7/S8 已停止并保留为污染 plant 的失败证据。当前 active slice 是完成 0.20 m capsule-source 资产验证，然后从 s6 `model_31000.pt` + fresh optimizer 重开。

## 已核实的 s5 卡点（iter ≈17500，勿当梅花桩证据）

TB `:8008`，任务 `2026-08-19_00-20-19_t_sparse_lightlp_s5`。列 occupancy 每列 5%、稀疏合计 40%，连续地形 `reach_2m` 0.64–0.71。踏石/圆桩：

| | 踏石 | 圆桩 |
| --- | ---: | ---: |
| `reach_1m` | 0.62 | 0.61 |
| `reach_2m` | 0.038 | 0.017 |
| 进度 | 1.15 m | 1.06 m |
| easy 进度 | 1.16 m | 1.09 m |

从 iter 18 到 17.5k 无上升。`model_14500` 回放 easy 全是 `accel`，peak 1.1–1.4 m，无 `pit`。s5 easy 踏石顶 26 cm、石间净空 24 cm、上台 16 cm；圆桩缝 5 cm 但上台 14 cm。脚约 21×8 cm，只有脚底碰撞，小腿穿石头。

`illegal_footstep≈-0.004` 是没落台样本，不是项写错。不要拧全局 length、accel=40、OOB。

## 独立诊断结论

交接文档只用于找线索；以下判断来自当前代码、固定 checkpoint、配对 seeds 和单变量地形对照。

| 层 | 已验证事实 | 当前判定 |
| --- | --- | --- |
| Runtime / 观测 | scan 数值 finite，前向覆盖 0.2–1.6 m。`model_6000` 配对消融中，normal 相对 zero/permuted 在踏石多 0.43/0.39 m，在圆桩多 0.43/0.69 m。 | scan 链路没断，策略确实使用空间排列；“有高程图但完全不用”被排除。 |
| 摆脚行为 | 修正指标为“首次离开出生垫的 touchdown 对应 swing”后，`model_6000` 踏石摆高均值 0.221 m，范围 0.147–0.334 m；touchdown 高 0.0865 m，与 0.09 m 石顶一致。`model_8000` 踏石/圆桩摆高均值达 0.234/0.214 m。 | “机器人不知道抬脚”是错误归因。真实剩余问题是全脚掌落点、后续连续支撑和稳定终止。 |
| 地形 / gate | 旧格点固定 9×9。easy 踏石最后支撑边只在前向 2.20 m，圆桩在 2.45 m；课程 promotion 是累计 path >4 m，evaluator `reach4` 是指令方向位移 4 m，OOB 4.25 m。 | 这是主要深层 bug：evaluator 的前进门槛在旧地形上物理不可达；课程则可被支撑区内的往返 path 污染，都不再测量真实穿越。 |
| 因果对照 | 同 `model_6000`、同 seeds，只把格点延伸到 tile 边界。踏石 strict 0→13/32、reach4 2→23/32、pit 24→2/32，平均进度 +1.17 m（95% CI 0.86–1.45）；圆桩 strict 0→5/32、reach4 4→17/32、pit 23→6/32，平均进度 +0.62 m（95% CI 0.24–1.03）。 | 大量 `accel` / pit 发生在物理格点结束后，不能解释为第一脚不抬或不会连续走。 |
| 资产碰撞 | URDF 活跃 collision 只有双脚和球手；`Trunk`、`Shank_*`、`A[LR]2/4` 无 collision。s6 @10k 的 `undesired_contacts`、`shank_contacts`、`Reset/torso` 仍全 0。 | 论文的非足接触惩罚与 torso reset 实际失效；这会破坏 sim2real / 跨栏解读，但不是“脚抬不起”的解释。 |

根因优先级：

1. **已确证主 bug：有限格点与 4 m 合同不一致。** 这直接污染 strict、pit、accel，并使课程 path promotion 不再等价于向前穿越。
2. **真实剩余行为问题：落点和稳定性仍弱。** 动态 grid 上仍有圆桩 27/32、踏石 19/32 非 clean completion，说明修完课程合同不等于能力已验收。
3. **evaluator 曾有口径和采样 bug。** 现在保留 terminal 快照、episode 内最大指令方向进度、每 env 固定配额和 per-env records；早摔 env 不再重复占据样本。
4. **AMP 缺失不是主因。** 论文§IV 的 perceptive-locomotion teacher 是 Table I reward-only PPO；AMP 主要出现在后续技能过渡。当前 sparse tile 是代码主动置零 AMP，且共享策略仍从连续地形获得步态先验。
5. **两个次级实现差异待单变量验证。** sparse 的 AMP scale=0 后，当前 runner 仍把 task reward 留在 `0.7×`；论文说堆叠最后 5 个完整 observation frame，当前是 proprio×10 + scan×5 + 当前 contact×1，且 Actor 是 flat MLP 而不是条件 map encoder。这些可能影响样本效率，但不能在 grid 修复前同时改。

这里的 `0.1 m` 是 scan 点在水平面上的网格间距，不是垂直高度量化。不能用“9 cm 上台只差一个 scan 格”判断石顶是否可见；是否可见、是否被策略利用必须分别由观测检查和消融验证。

## 配方（s6，不在 s5 ckpt 上热补）

1. **踏石 easy**：顶 0.40 m，间距 0.50 m → 石间净空 0.10 m，上台 0.09 m。出生垫边沿接到第一块砖（`first_gap≈0`）。hard：顶 0.26 m、净空 0.28 m、上台 0.24 m。
2. **圆桩**：直径/间距仍是 (0.50, 0.38) / (0.55, 0.58)。easy 高 0.14→0.08 m，hard 0.28 m。
3. **10% 随机 level** 仍开，但 `random_level_reset_max_level=4`（exclusive → 行 0–3）。
4. MDP / 观测 / 真洞 / 1937D 与 s6 相同；不填缝。资产合同变为：`Trunk` 使用 MJCF 对齐的 0.10×0.16×0.34 m box；每侧 `Shank` 使用膝部 r=0.05/l=0.10 m cylinder 和胫骨 r=0.04/l=0.20 m cylinder segment。Isaac 转成 capsule 后，胫骨最终外包络仍为 0.28 m，与 MJCF 一致且不与脚 collider 重叠。

## Non-goals

- 续训 s5 / 在 s5 ckpt 上热补几何。
- 改 `LIGHTLP_ACCEL_LIMIT`、免疫 10%、OOB 4.25 m、20 s horizon、algebraic / 软填。
- 同时扩到手臂、大腿或 mesh collision；本次只修用户指定的躯干和双小腿。
- 本切片蒸学生、改 Stage E 1155D、动 zhuoqun。
- 用 s5 的 `reach_2m` 或全局 reward 宣称梅花桩能力。

## Success Criteria

1. 格点不再固定 9×9，而是按 tile / border / pitch 动态铺满；地形生成与 `t4_env` 代数支撑判定使用同一 `max_ring` 合同。
2. URDF 中 `Trunk` 恰有 1 个 box collision，`Shank_Left/Right` 各有 2 个左右对称 cylinder，尺寸与 MJCF primitive 对齐，不使用 mesh collision。
3. Isaac 强制重转 USD 后必须找到 `Trunk` 1 个 box prim 与每侧 `Shank` 2 个 capsule prim；关节/刚体数仍为 27/32。
4. evaluator 在 `--device cuda:N` 下必须同步设置 App、env、sim 和 runner device，不允许 reset 期间出现 `cuda:N` / `cuda:0` 张量混用。
5. 旧 s6 checkpoint 在新物理上的 flat 短评估必须能完成固定配额，无启动崩溃、NaN 或大量 step-0 torso/self-contact reset；这只是资产回归，不要求旧策略直接适应所有新碰撞。
6. `difficulty=0` 踏石/圆桩短评估不得因碰撞几何而全部立即失败；必须单独报告 torso、accel、pit 和 clean completion。
7. `s7_grid_collision` 从冻结的 s6 checkpoint 开新 logdir；1k–2k 先看 flat 不回归、踏石/圆桩 pit/accel 下降和 clean completion 上升，然后才决定长训。
8. 最终仍需 easy/mid/hard evaluator JSON、lineage manifest 和连续回放证据；由于同时改了 grid 与 collision，s7 只能声明组合修复效果。

## Verification Path

```text
本机 pytest（asset + grid + sparse + evaluator）
  -> nubot 隔离 worktree 强制重转 USD 并枚举 collision prim
  -> 旧 s6 checkpoint 在新物理上跑 flat / easy sparse 短回归
  -> 冻结 s6 checkpoint，开 t_sparse_lightlp_s7_grid_collision
  -> 1k–2k TB + fixed evaluator + 连续回放决策是否长训
```

### Verification Path Status

`running`。本机组合同同测试已绿；nubot 隔离 USD collision smoke、静站/动态短回归已完成。s7 已加载 `model_31000.pt` 并进入 PPO，当前等 1k–2k 适应门。

阶段 3 监控入口：

- 训练 tmux：nubot `t4-sparse-lightlp-s6`。
- TensorBoard：`http://100.100.188.39:8009`。
- 本地抓数脚本：`artifacts/monitoring/t4_s6_tb_monitor.py`，输出两类 easy 地形的 `reach_2m` / `progress` 最近值、最近 20 点窗口和前一 20 点窗口。
- 本地后台日志：`artifacts/monitoring/t4-s6-tb-monitor.log`。
- checkpoint：`logs/t4_loco_teacher_sparse/2026-08-19_12-51-05_t_sparse_lightlp_s6/model_*.pt`。

2026-08-19 14:20 CST 快照：远端四卡训练和 TB 均存活，最新 ckpt 为 `model_2000.pt`，无 `Traceback` / `RuntimeError` / `CUDA error` / `NaN` 证据。TB 到 step≈2171，仍未到 3k 健康门；踏石 easy `reach_2m mean20≈0.011`、`progress mean20≈1.267 m`，圆桩 easy `reach_2m mean20≈0.023`、`progress mean20≈1.273 m`。这说明 runtime / optimization 暂正常，behavior 仍低；未到 3k 前不提前判负。

2026-08-19 14:56 CST 3k 健康门：**未整体通过**。踏石 easy 已过门：`reach_2m mean20≈0.122`、`progress mean20≈1.400 m`，且较前一窗口上升。圆桩 easy 未过门：`reach_2m mean20≈0.031`、`progress mean20≈1.223 m`，低于 `reach_2m>0.10` 和 `progress>1.25 m`。结论：s6 相对 s5 证明 easy 几何能帮踏石起势，但圆桩仍卡第一落点 / 抬脚 / 观测利用链路；按计划继续到 5k 决策门，不现在停训，也不改配方。

2026-08-19 16:09 CST 5k 前人工判断：不要把 5k 门机械解释为自动 kill。当前 step≈4870，踏石 easy 已明显过门（`reach_2m mean20≈0.293`、`progress mean20≈1.615 m`）；圆桩 easy 未过 5k 目标但仍在爬坡（3k `reach_2m≈0.031` → 4.8k `≈0.147`，`progress≈1.33 m`），没有平台期证据，也未触发 `reach_2m<=0.05` 或 `progress<=1.20 m` 硬停线。若 5k 正式点仍呈上升趋势，应继续观察到 6k/7k；5k 的含义改为“人工复核并禁止无脑跑满 40k”，不是趋势未尽时强制停止。

2026-08-19 16:17 CST 5k 后快照：继续训，不停。踏石 easy 已过 5k 门（`reach_2m mean20≈0.297`、`progress mean20≈1.597 m`）。圆桩 easy 仍未过 `reach_2m>=0.20`，但 `reach_2m mean20≈0.159`、`progress mean20≈1.383 m`，相比 3k 仍在上升，未触发硬停线。mid/hard 曲线解释：`easy/mid/hard` 是按 curriculum row 分桶的监控 band，不是 10% 随机难度的三个模式；当前踏石/圆桩 `hard_episodes=0`，hard 全 0 表示没有有效 hard episode 样本，不能当 hard 能力失败证据。mid episodes 很少（踏石约 87、圆桩约 50，对比 easy 约 8.3–8.6 万），波动大，只用于早期参考。

2026-08-19 16:54 CST 6k 快照：继续训到 7k。踏石 easy 稳定过门（`reach_2m mean20≈0.347`、`progress mean20≈1.698 m`）。圆桩 easy 基本贴近 5k reach 门且 progress 已过门（`reach_2m mean20≈0.190`、`progress mean20≈1.419 m`），较前一窗口继续上升，未出现平台期。远端 `model_6000.pt` 已写出，四卡训练 / TB 存活，日志尾部无 runtime 异常。

2026-08-19 17:37 CST 7k 快照：s6 easy 几何健康门可判通过，继续长训并准备固定 evaluator / 回放验证。踏石 easy 稳定过门（`reach_2m mean20≈0.365`、`progress mean20≈1.721 m`）。圆桩 easy 也稳定过门（`reach_2m mean20≈0.241`、`progress mean20≈1.494 m`），且较前一窗口继续上升。远端 `model_7000.pt` 已写出，四卡训练 / TB 存活，日志尾部无 runtime 异常。mid 仍只作参考，hard 仍无有效 episode 样本；能力声明仍需固定 difficulty=0 evaluator JSON 和视频确认脚踩顶面。

2026-08-19 17:48 CST `model_6000.pt` 固定 evaluator（tmux `t4-s6-eval-fixed-m6000`）已产出 JSON，作为早期行为诊断而非能力声明：flat d=0.85 为 32/32 strict；踏石 easy deterministic `reach_2m=29/32`、stochastic `27/32`，但 strict 均为 0/32，主要因早停 / 掉洞；圆桩 easy deterministic `reach_2m=23/32`，strict 为 2/32。结论：策略已经能推进过 2 m，TB 的 easy 起势是真信号；但仍不是“稳定走完、不摔、脚踩顶面”的能力证据，后续必须用更新 ckpt 做 evaluator + 连续回放确认。

2026-08-19 18:17 CST 8k 快照：继续训到 10k。踏石 easy 维持健康（`reach_2m mean20≈0.370`、`progress mean20≈1.751 m`）。圆桩 easy 继续高于 7k，且稳定过门（`reach_2m mean20≈0.283`、`progress mean20≈1.569 m`）。远端 `model_8000.pt` 已写出，四卡训练 / TB 存活，日志尾部无 runtime 异常。当前重点从“easy 是否起势”转为“是否形成稳定不摔的行为证据”；下一道建议在 10k 或更高 ckpt 重跑 fixed evaluator / 回放。

2026-08-19 17:48 CST evaluator 修复与 `model_6000.pt` 复评：旧 evaluator 的 progress / terminal 状态口径与训练 monitor 不一致。现在同时输出 episode 内最大指令方向进度、terminal radial progress 和最终前向位移，不再用单一字段混为 progress。flat d=0.85 为 32/32 strict，证明基本口径可用。

2026-08-19 19:38 CST 深层诊断收口：

- 平衡采样修复：每 env 固定 episode 配额，避免早摔 env 重复入样、长存活 env 被排除。`episode_records` 可按 env/seed 配对。
- scan 消融：`artifacts/eval/s6_model6000_scan_ablation_v4_paired/`，`suite.exit=0`。normal 在两类地形均显著优于 zero/permuted。
- 动态 grid 因果 probe：隔离 worktree `TienKung-Lab-grid-probe`，产物 `artifacts/eval/s6_model6000_extended_grid_probe/`，`suite.exit=0`。踏石平均进度 3.137→4.307 m，pit 24→2；圆桩 3.154→3.776 m，pit 23→6。
- 摆脚 probe：踏石首次离垫 touchdown 对应 swing 均值 22.1 cm，touchdown 8.65 cm。旧 4.8 cm 指标测到的是出生垫上首次小摆腿，已废弃。
- `model_8000.pt` 旧 grid 复评 `suite.exit=0`：踏石 reach2 32/32、圆桩 29/32，两者 strict 仍 0/32；摆脚均值 23.4/21.4 cm。更多迭代在改善第一步，但无法修复地形物理结束后的深坑。

2026-08-19 19:44 CST 10k 后监控：继续训，不改配方。远端四卡训练和 TB 均存活，最新 ckpt 为 `model_10000.pt`，日志尾部未见 `Traceback` / `RuntimeError` / `CUDA error` / `NaN`。TB 到 step≈10.2k：踏石 easy `reach_2m mean20≈0.377`、`progress mean20≈1.77 m`；圆桩 easy `reach_2m mean20≈0.276`、`progress mean20≈1.58 m`，仍在健康门以上。mid 开始有少量有效样本：踏石 mid `episodes≈727`、`reach_2m mean20≈0.182`、`progress≈1.37 m`；圆桩 mid `episodes≈461`、`reach_2m≈0.118`、`progress≈1.23 m`，只能作趋势参考。踏石/圆桩 hard 仍 `episodes=0`，不能解释为 hard 能力失败。全局 level 分布已不只低难度：level 4–9 合计约 32%，但 sparse 两类还没被足量推到 hard 桶；`Curriculum/terrain_levels mean20≈2.80`、`terrain_difficulty≈0.31`。Reset 侧 `accel mean20≈0.593`、`fall_over≈0.034`、`pit_fall≈0.120`，仍提示行为验收重点是踩顶面和减少掉洞，不是 runtime/优化崩溃。

2026-08-19 19:50 CST 10.3k 监控：S6 作为旧 grid baseline 继续跑，不停、不热同步 grid 修复。远端训练到 `Learning iteration 10328/40000`，四卡 worker / TB 存活，最新 ckpt 仍为 `model_10000.pt`（下一个预期 `model_10500.pt`），日志尾部干净。TB easy 最近窗口没有平台证据：踏石 easy `reach_2m mean20≈0.398–0.404`、`progress≈1.80–1.83 m`；圆桩 easy `reach_2m≈0.288–0.308`、`progress≈1.60–1.64 m`。mid 样本继续增加但仍少于 easy 两个数量级：踏石 mid `episodes≈779`、`reach_2m≈0.177`、`progress≈1.36 m`；圆桩 mid `episodes≈490`、`reach_2m≈0.145`、`progress≈1.29 m`。踏石/圆桩 hard 仍 `episodes=0`，继续按“无 hard 样本”处理。level 4–9 合计约 32%，`terrain_levels mean20≈2.83`、`terrain_difficulty≈0.315`；12k ETA 约 21:00 CST，15k ETA 约 23:06 CST。因为本地诊断已把主因定位到旧 9×9 grid 与 4 m gate 不一致，本轮不建议再对旧 grid S6 启动新的 10k fixed evaluator；下一实质动作应是 tile-filling grid 单变量 lineage。

2026-08-19 19:57 CST 10.5k 监控：`model_10500.pt` 已落盘。训练 / TB 仍正常，日志确认到 `Learning iteration 10500/40000`，四卡 GPU 使用约 39–44%。TB 最近窗口：踏石 easy `reach_2m mean20≈0.379`、`progress≈1.77 m`，较上一窗口略低但仍明显高于健康线；圆桩 easy `reach_2m≈0.317`、`progress≈1.66 m`，继续上行。mid：踏石 `episodes≈817`、`reach_2m≈0.157`、`progress≈1.35 m`，圆桩 `episodes≈524`、`reach_2m≈0.147`、`progress≈1.30 m`。hard 仍 `episodes=0`。`Reset/accel mean20≈0.638` 较前窗略高，`fall_over≈0.027`、`pit_fall≈0.113` 未恶化成硬停信号。`terrain_levels mean20≈2.83`，level 4–9 合计约 32%。判定：旧 grid baseline 没有 runtime/optimization 崩溃，也没有 easy 平台期；但由于有限 grid 根因已明确，继续训练只作为 baseline 曲线，不应追加旧 grid evaluator 消耗 GPU。

2026-08-19 20:00 CST 10.6k 监控：尚未到 12k 门点，继续跑。远端训练到 `Learning iteration 10593/40000`，`model_10500.pt` 已存在，四卡 worker / TB 正常，日志尾部无 `Traceback` / `RuntimeError` / `CUDA error` / `NaN`。TB：踏石 easy `reach_2m mean20≈0.396`、`progress≈1.80 m`；圆桩 easy `reach_2m≈0.316`、`progress≈1.65 m`，两者仍健康。mid 样本继续增长：踏石 mid `episodes≈845`、`reach_2m≈0.155`、`progress≈1.35 m`；圆桩 mid `episodes≈537`、`reach_2m≈0.157`、`progress≈1.31 m`。hard 仍 `episodes=0`，不能判 hard 失败。`terrain_levels mean20≈2.79–2.83`，level 4–9 合计约 32%。`Reset/accel mean20≈0.629` 持平偏高，`pit_fall≈0.107` 未恶化。12k ETA 约 20:57 CST；下一轮按 12k baseline 快照判定是否继续仅监控或停止旧 grid 训练并转入 tile-filling lineage。

2026-08-19 20:05 CST 10.7k 监控：继续跑到 12k 门点。TensorBoard HTTP 查询短暂出现 502 / timeout，但后续恢复；远端 tmux probe 显示训练到 `Learning iteration 10710/40000`，四卡 worker / TB 进程正常，`model_10500.pt` 仍为最新 ckpt，日志尾部无 runtime 异常。TB easy：踏石 `reach_2m mean20≈0.411`、`progress≈1.80 m`；圆桩 `reach_2m≈0.297`，当前点低于上一窗口但仍过健康线，缺失的一次 progress 查询不影响 runtime 判断。mid 明显比上一轮更有信号：踏石 mid `episodes≈896`、`reach_2m≈0.185`、`progress≈1.43 m`；圆桩 mid `episodes≈566`、`reach_2m≈0.189`、`progress≈1.41 m`。hard 仍 `episodes=0`。level 4–9 合计约 32.6%，`Perf/total_fps mean20≈44k`，`Loss/value_function` finite，`entropy≈19.21`、`noise≈0.505`。`Reset/accel` 仍偏高但近窗持平，`pit_fall≈0.109` 未恶化。判定：S6 baseline 的 easy/mid 曲线仍有学习信号，不构成平台；但已知旧 grid 物理合同错误，12k 后应优先决策是否停止旧 baseline 并转入 tile-filling lineage。

2026-08-19 20:09 CST 10.8k 监控：继续跑，尚未到 12k。远端训练到 `Learning iteration 10825/40000`，`model_10500.pt` 仍为最新 ckpt（下一个预期 `model_11000.pt`），四卡 worker / TB 存活，日志尾部无 runtime 异常。TB easy：踏石 `reach_2m mean20≈0.388`、`progress≈1.74 m`，较前窗回落但仍高于健康线；圆桩 `reach_2m≈0.298`、`progress≈1.61 m`，保持健康。mid 继续上移：踏石 mid `episodes≈932`、`reach_2m≈0.214`、`progress≈1.47 m`；圆桩 mid `episodes≈587`、`reach_2m≈0.195`、`progress≈1.41 m`。hard 仍 `episodes=0`。`terrain_levels mean20≈2.86`、`terrain_difficulty≈0.317`，level 4–9 合计约 33.8%，课程没有塌回低难度。`Reset/accel≈0.607`、`pit_fall≈0.094–0.103`，较前窗不恶化。判定：旧 grid baseline 仍有 mid 学习信号，不构成平台；但由于物理 grid 合同错误已明确，12k 门点仍应作为是否停止 baseline、转 tile-filling lineage 的决策点。

2026-08-19 20:14 CST 10.95k 监控：继续跑，尚未到 12k。远端训练到 `Learning iteration 10940/40000`，`model_10500.pt` 仍为最新 ckpt，四卡 GPU / worker / TB 正常，日志尾部无 `Traceback` / `RuntimeError` / `CUDA error` / `NaN`。TB easy：踏石 `reach_2m mean20≈0.384`、`progress≈1.75 m`，较 20:05 高点回落但仍明显过健康线；圆桩 `reach_2m≈0.316`、`progress≈1.62 m`，保持健康。mid：踏石 `episodes≈959`、`reach_2m≈0.223`、`progress≈1.47 m`；圆桩 `episodes≈605`、`reach_2m≈0.193`、`progress≈1.42 m`，基本稳住上一轮改善。hard 仍 `episodes=0`。`terrain_levels mean20≈2.89`、level 4–9 合计约 33.3%，课程均值继续略上移。`Perf/total_fps≈44k`，value loss finite，`entropy≈19.26`、`noise≈0.505`。`Reset/accel≈0.622`、`pit_fall≈0.116`，无硬停信号。判定：S6 旧 grid baseline 仍未平台；12k 门点约 20:55 CST，届时重点不是证明能力，而是决定是否停止旧 baseline 并转入 tile-filling grid 单变量 lineage。

2026-08-19 20:18 CST 11.0k 监控：继续跑，尚未到 12k。远端训练到 `Learning iteration 11049/40000`，`model_11000.pt` 已落盘，四卡 worker / TB 正常，日志尾部无 runtime 异常。TB easy：踏石 `reach_2m mean20≈0.387`、`progress≈1.78 m`；圆桩 `reach_2m≈0.297`、`progress≈1.59 m`，仍在健康线以上。mid 基本稳定：踏石 mid `episodes≈998`、`reach_2m≈0.224`、`progress≈1.46 m`；圆桩 mid `episodes≈620`、`reach_2m≈0.189`、`progress≈1.43 m`。hard 仍 `episodes=0`。`terrain_levels mean20≈2.82`、`terrain_difficulty≈0.313`，level 4–9 合计约 31.3%；没有课程塌陷，但 sparse hard 仍无有效样本。`Reset/accel≈0.595`、`pit_fall≈0.112`，value loss finite，`entropy≈19.30`、`noise≈0.506`。判定：旧 baseline 仍没有 runtime/optimization 硬停信号，easy/mid 也未平台；继续到 12k 门点再决定是否停止旧 grid、转 tile-filling lineage。

2026-08-19 20:22 CST 11.15k 监控：继续跑，尚未到 12k。远端训练到 `Learning iteration 11145/40000`，`model_11000.pt` 已存在，四卡 worker / TB 正常，日志尾部无 runtime 异常。TB easy：踏石 `reach_2m mean20≈0.388`、`progress≈1.77 m`；圆桩 `reach_2m≈0.290`、`progress≈1.59 m`，仍过健康线。mid：踏石 `episodes≈1017`、`reach_2m≈0.228`、`progress≈1.45 m`；圆桩 `episodes≈625`、`reach_2m≈0.185`、`progress≈1.43 m`。hard 仍 `episodes=0`。`terrain_levels mean20≈2.79`、`terrain_difficulty≈0.310`，level 4–9 合计约 31.2%。`Reset/accel≈0.599`、`pit_fall≈0.101`，value loss finite，`entropy≈19.30`、`noise≈0.506`。判定：旧 baseline 继续稳定，未平台；12k ETA 约 20:55 CST，届时建议不要再用旧 grid 做能力验收，而是选择停止 baseline 或保留后台继续跑、同时把主线切到 tile-filling grid。

2026-08-19 20:28 CST 11.3k 监控：继续跑，尚未到 12k。远端训练到 `Learning iteration 11292/40000`，`model_11000.pt` 仍为最新 ckpt，四卡 worker / TB 正常，日志尾部无 `Traceback` / `RuntimeError` / `CUDA error` / `NaN`。TB easy：踏石 `reach_2m mean20≈0.394`、`progress≈1.78 m`；圆桩 `reach_2m≈0.287`、`progress≈1.60 m`，仍过健康线。mid：踏石 `episodes≈1044`、`reach_2m≈0.234`、`progress≈1.46 m`；圆桩 `episodes≈638`、`reach_2m≈0.175`、`progress≈1.40 m`。hard 仍 `episodes=0`，不能判 hard 失败。level 4–9 合计约 32.9%，`terrain_levels mean20≈2.86`、`terrain_difficulty≈0.318`，课程没有塌回低难度；但 10% random reset 仍固定抽 level 0–3，不随当前课程上浮。`Reset/accel≈0.604`、`pit_fall≈0.104`，value loss finite，`entropy≈19.22`、`noise≈0.505`，`Perf/total_fps≈44.3k`。12k ETA 约 20:55 CST；12k 决策仍应围绕“旧 grid baseline 是否停止并转 tile-filling lineage”，不是用 S6 证明最终梅花桩能力。

2026-08-19 20:56 CST 12k 门点：`model_12000.pt` 已落盘，远端训练继续到 `Learning iteration 12022/40000`，四卡 worker / TB 正常，日志尾部无 `Traceback` / `RuntimeError` / `CUDA error` / `NaN`。TB easy 仍健康：踏石 `reach_2m mean20≈0.412`、`progress≈1.81 m`，较前窗上升；圆桩 `reach_2m≈0.300`、`progress≈1.63 m`，较前窗略回落但仍高于健康线。mid：踏石 `episodes≈1272`、`reach_2m≈0.199`、`progress≈1.45 m`；圆桩 `episodes≈798`、`reach_2m≈0.133`、`progress≈1.27 m`，圆桩 mid 明显偏弱，且不能用作稳定能力声明。hard 仍 `episodes=0`。level 4–9 mean20 合计约 33.3%，`terrain_levels mean20≈2.92`、`terrain_difficulty≈0.324`，课程整体仍在上移；10% random reset 仍是固定低难度 level 0–3 入口。`Reset/accel≈0.615`、`pit_fall≈0.105`，value loss finite，`entropy≈19.16`、`noise≈0.503`，`Perf/total_fps≈44.4k`。判定：没有 runtime / optimization 硬停信号，也不能说 easy 已平台；但旧 9×9 grid 与 4 m gate 的合同 bug 已确证，继续跑 S6 的信息增量下降。建议在用户授权后冻结 `model_12000.pt`，停止旧 baseline，主线切到 tile-filling grid 单变量 warm-start 1k–2k；若暂不授权停止，可让 S6 后台继续，但不要追加旧 grid evaluator 或用其 strict 结果做能力验收。

2026-08-19 21:16 CST 12.5k 监控：`model_12500.pt` 已落盘，远端训练到 `Learning iteration 12533/40000`，四卡 worker / TB 正常，日志尾部无 runtime 异常。TB easy 没有平台：踏石 `reach_2m mean20≈0.422`、`progress≈1.85 m`，圆桩 `reach_2m≈0.338`、`progress≈1.69 m`，两者均高于 12k。mid：踏石 `episodes≈1427`、`reach_2m≈0.207`、`progress≈1.44 m`，基本横盘；圆桩 `episodes≈1002`、`reach_2m≈0.207`、`progress≈1.49 m`，12k 的低点恢复。hard：踏石仍 `episodes=0`；圆桩 hard 仅约 3 个 episode，`reach_2m=0`，样本量仍不足，不能判 hard 能力。level 4–9 mean20 合计约 32.1%，`terrain_levels mean20≈2.83`、`terrain_difficulty≈0.315`，课程没有塌。`Reset/accel≈0.604`、`pit_fall≈0.132`，value loss finite，`entropy≈19.09`、`noise≈0.503`，`Perf/total_fps≈44.3k`。判定：S6 仍无 runtime / optimization 硬停；behavior 继续支持“easy 起势 + mid 有弱信号”，但掉坑/accel 仍高，且旧 grid 合同 bug 仍使 strict/evaluator 不可作为最终能力验收。建议不再为旧 grid 追加 evaluator，下一实质动作仍是 tile-filling grid 单变量 lineage。

2026-08-19 21:36 CST 13k 监控：`model_13000.pt` 已落盘，远端训练到 `Learning iteration 13033/40000`，四卡 worker / TB 正常，日志尾部无 runtime 异常。TB easy 继续无平台：踏石 `reach_2m mean20≈0.426`、`progress≈1.84 m`；圆桩 `reach_2m≈0.347`、`progress≈1.74 m`。mid：踏石 `episodes≈1654`、`reach_2m≈0.242`、`progress≈1.40 m`，reach2 上升但 progress 未同步上升；圆桩 `episodes≈1168`、`reach_2m≈0.189`、`progress≈1.45 m`，基本维持。hard：踏石仍 `episodes=0`；圆桩 hard 仍约 3 个 episode，统计无效。level 4–9 mean20 合计约 34.5%，`terrain_levels mean20≈2.95`、`terrain_difficulty≈0.328`，课程略上移。`Reset/accel≈0.584`、`pit_fall≈0.108`，value loss finite，`entropy≈19.11`、`noise≈0.503`，`Perf/total_fps≈44.0k`。判定：runtime/optimization 正常，behavior 没有平台；但 mid 仍只是弱趋势，hard 无样本。继续跑只增加旧 grid baseline 曲线，不改变“下一步应转 tile-filling grid 单变量 lineage”的结论。

2026-08-19 21:55 CST 13.5k 监控：`model_13500.pt` 已落盘，远端训练到 `Learning iteration 13528/40000`，四卡 worker / TB 正常，日志尾部无 runtime 异常。TB easy：踏石 `reach_2m mean20≈0.439`、`progress≈1.85 m`，继续上升；圆桩 `reach_2m≈0.331`、`progress≈1.70 m`，较 13k 回落但仍高于健康线。mid：踏石 `episodes≈1837`、`reach_2m≈0.226`、`progress≈1.42 m`，从 13k 高点回落；圆桩 `episodes≈1309`、`reach_2m≈0.246`、`progress≈1.46 m`，继续改善。hard：踏石仍 `episodes=0`；圆桩 hard 仍约 3 个 episode，统计无效。level 4–9 mean20 合计约 32.4%，`terrain_levels mean20≈2.86`、`terrain_difficulty≈0.317`，课程分布回落但没有塌。`Reset/accel≈0.618`、`pit_fall≈0.127`，value loss finite，`entropy≈19.22`、`noise≈0.505`，`Perf/total_fps≈44.1k`。判定：S6 仍无 runtime / optimization 硬停；behavior 是 easy 高位波动、mid 有信号但抖动，hard 仍不可解读。继续跑旧 grid 的边际信息继续下降；下一步仍应冻结 ckpt 后转 tile-filling grid 单变量 lineage，而不是用旧 grid strict/evaluator 做能力结论。

2026-08-19 22:54 CST 15k 监控：14k 附近本地访问链路出现多次 TB 502 / SSH timeout，Tailscale 只能经 DERP(hkg) 中继；22:16 后通道恢复，远端训练未中断。`model_15000.pt` 已落盘，远端训练到 `Learning iteration 15034/40000`，四卡 worker / TB 正常，日志尾部无 runtime 异常。TB easy：踏石 `reach_2m mean20≈0.436`、`progress≈1.87 m`，高位横盘；圆桩 `reach_2m≈0.353`、`progress≈1.74 m`，较 13.5k 恢复。mid：踏石 `episodes≈2472`、`reach_2m≈0.255`、`progress≈1.58 m`，progress 较 13.5k 改善但 reach2 基本横盘；圆桩 `episodes≈1821`、`reach_2m≈0.263`、`progress≈1.52 m`，比 13.5k 更稳。hard：踏石仍 `episodes=0`；圆桩 hard 仍约 3 个 episode，统计无效。level 4–9 mean20 合计约 34.4%，`terrain_levels mean20≈2.99`、`terrain_difficulty≈0.332`，课程略上移。`Reset/accel≈0.596`、`pit_fall≈0.129`，value loss finite，`entropy≈19.32`、`noise≈0.507`，`Perf/total_fps≈44.3k`。判定：runtime / optimization 正常；behavior 未平台，mid 信号比 13.5k 更强，但掉坑/accel 仍高、hard 仍无有效样本。S6 继续跑可作为旧 grid baseline 曲线；能力验收和下一步修复仍应切到 tile-filling grid 单变量 lineage。

## Required Capabilities

- 本机 pytest（无 Isaac）。
- nubot 4×4090 + `scripts/nubot_run.sh` + tmux。
- s5 logdir 作负结果对照。

## Fallback Evidence

- 本机无 Isaac：几何合同 + 公式 cap。64-env smoke 放到 nubot 开训后看 log 无即崩。
- s6 未过阶段门时先冻结对应 checkpoint、固定种子、两类地形 evaluator JSON 和第一脚连续回放，后续每个探针都开新 lineage，不在同一训练上混改。

## s6 诊断与 s7 组合修复分支

前三个分支是已完成的单变量诊断。用户在 2026-08-20 明确选择一次性修正 grid 和 collision，因此第 4 步起是组合物理 lineage：

1. **已完成：scan 因果消融。** normal 明显优于 zero/permuted，观测断链被排除。
2. **已完成：首次离垫 swing / touchdown instrumentation。** 脚会主动抬高；“软支撑用来教抬脚”这条分支不再需要。
3. **已完成：旧 grid vs tile-filling grid 单变量对照。** 结果确认地形长度是当前第一修复项。
4. **当前 lineage S9：同步 tile-filling grid + 修复后的 Trunk/双 Shank collision，从 s6 checkpoint warm-start。** USD asset smoke 与 flat/sparse 短回归已通过，正在进行 1k–2k 适应。
5. **S9 仍失败才进入落点控制分支。** 优先验证完整 5-frame contact history、条件 map encoder 或 soft-support foothold curriculum；从这里开始恢复每次只改一个变量。

## s8 termination cost 分支

S7/S8 当时被解释为终止代价问题，但该解释已被更强的本地物理证据推翻。flat 正常姿态下双 Shank 每步约 20 kN，关闭 self-collision 后归零；根因是 Shank capsule 与脚 collider 持续重叠。早期 torso/accel 不能再作为“策略主动结束”证据。

S8 已冻结在 `model_33000.pt`。保留 tile-filling grid、真实 `Trunk`/双 `Shank` collision 和机器人 self-collision；胫骨 cylinder segment 改为 0.20 m。sparse `termination_penalty.weight` 恢复为 0.0。后续 S9 不加载 S7/S8，而从 s6 `model_31000.pt` 加载模型并重置 optimizer。

250 iter gate（event step 31260）：`Reset/torso` 最近 20 点约 0.348，没有复现 S7 接近 1 的终止逃逸；但 `Reset/accel` 最近约 0.782，踏石/圆桩 easy progress 仅约 1.25–1.29 m，`reach_2m` 基本为 0。结论是继续到 500 iter / `model_31500.pt`，再跑 flat、easy 踏石、easy 圆桩 fixed evaluator；现在不能宣称 sparse 行为能力。

500 iter gate：`model_31500.pt` fixed evaluator 产物在 `artifacts/eval/s8_model31500_fixed/`。flat d=0.85 strict 16/16、reach4 13/16；踏石 d=0 strict 0/16、reach2 0/16、pit 0、early 16/16；圆桩 d=0 strict 0/16、reach2 1/16、pit 0、early 16/16。两类 sparse 都是 accel 16/16，torso 8/16，平均约 139 steps、进度约 1.48–1.49 m。当时把改善归因于 `termination_penalty=-200`，但 `model_33000` 本地回放随后证明 S7/S8 均受 Shank-foot 假自碰污染，因此这条因果解释已撤回。

1k gate：`model_32000.pt` 已落盘。当时窗口显示 S8 仍未过 sparse，但不是“机器人不动 timeout”：踏石/圆桩 easy 约 90% 为 fall，`Reset/accel` 和 `Reset/torso` 虽下降但仍高。后续根因更新后，不再做 termination-cost 或 2.0 m/s 速度 ablation；先在修复 Shank capsule 的 plant 上用原速度范围和 `termination_penalty=0` 重训，才能重新判断奖励和速度课程。

## s9 capsulefix 分支

S9 已在 nubot 隔离 worktree `/home/nubot/phn_ws/t4_train/TienKung-Lab-s9-capsulefix` 启动，HEAD `fd84ee3`，主 run `2026-08-20_16-07-04_t_sparse_lightlp_s9_capsulefix`。它从 s6 `model_31000.pt` 加载策略并重置 optimizer，不加载 S7/S8；`termination_penalty=0`，速度范围保持 `[-0.6, 1.0]`。

启动后 event step `31143` 早期窗口中，`Episode_Reward/shank_contacts=0`，`undesired_contacts` 最近 20 点约 `-0.015`，`Reset/torso≈0.066`、`Reset/accel≈0.556`。踏石 easy `reach_2m≈0.463/progress≈2.27 m`，圆桩 easy `reach_2m≈0.400/progress≈2.06 m`。这组信号已经证伪“修复没进训练 plant”，但不替代 checkpoint evaluator。S9 保持配方不变到 250/500 iter，`model_31500.pt` 落盘后做 flat/踏石/圆桩 fixed evaluator 和连续回放。
6. **AMP reward blend 仍分开立项。** sparse task reward 从 0.7× 恢复到 1.0× 是否提升样本效率，只能在 S9 物理合同和行为基线稳定后单独验证。

## 干预分类

| 干预 | 分类 | 用途与边界 |
| --- | --- | --- |
| s6 easy 几何放宽 | `repair_proposal` | 给第一落点制造足够正样本；是否成功由 3k/5k 与固定 evaluator 决定。 |
| 第一摆动脚、合法首次接触、action 饱和日志 | `instrumentation` | 提高可观测性，不改变任务。 |
| normal/zero/permuted scan、真洞观测下的短程软支撑 | `causal_probe` | 区分观测利用、探索断层与控制能力；结果不能直接作为最终策略证据。 |
| tile-filling grid + Trunk/双 Shank collision | `approved_combined_repair` | 同时修正可穿越长度与非足接触物理；由于是组合干预，不做单独碰撞因果声明。 |
| 放松 accel/OOB | `symptom_suppression` | 仍无证据支持，不混进 s7。 |

## Final integration claim

`final_integration_claim`: s6 证明策略会抬脚、踩上顶面并使用 scan；它的 strict 失败同时被有限 9×9 格点和缺失躯干/小腿碰撞污染。S7/S8 又被 Shank-foot capsule 假自碰污染；S9 才是 tile-filling grid + 正确 primitive collision 的当前 lineage。本切片不蒸学生、不热改 AMP/奖励/accel，能力声明等 fixed evaluator 和连续回放。

## 工作项

- [x] 阶段 1：easy 几何 + 随机 level 0–3
  - acceptance_criteria: 上文 Success 1–2；相关 pytest 绿
  - verification_commands: `python -m pytest tests/test_t4_stepping_stone_contracts.py tests/test_t4_sparse_reward_contracts.py tests/test_t4_sparse_monitor_contract.py tests/test_t4_terrain_column_map.py tests/test_t4_terrain_curriculum.py tests/test_t4_sparse_teacher_mujoco.py -q`
  - success_definition: easy 第一脚在合同上是迈不是跳
- [x] 阶段 2：停 s5，从零开 s6
  - acceptance_criteria: s5 tmux 停、ckpt 保留；s6 `--resume` 关；run name `t_sparse_lightlp_s6`；任务仍 `t4_loco_teacher_sparse`
  - verification_commands: nubot `tmux ls`；`logs/t4-sparse-lightlp-s6.log` 无 load ckpt；`params/agent.yaml` `resume: false`
  - success_definition: 新 lineage 在放宽的 easy 几何上冷启动
- [x] 阶段 3：s6 健康门 + 深层诊断（非能力验收）
  - acceptance_criteria: 3k 时两类 easy 都达到 `reach_2m>0.10`、进度 `>1.25 m` 且趋势上升；5k 时都达到 `reach_2m≥0.20`、进度 `≥1.40 m`
  - verification_commands: TB `Terrain/{stepping_stones,raised_pillars}/{easy_reach_2m_rate,progress_m}`；固定 difficulty=0 各 16 局 evaluator + 连续视频
  - success_definition: 两类地形 reach2 健康门已过；scan / 摆脚 / grid 因果分支已关闭，定位到有限格点合同 bug
- [ ] 阶段 4：`s7_grid_collision` 组合物理 lineage（当前 active slice）
  - acceptance_criteria: grid/asset/evaluator 合同绿；USD prim 实际存在；flat/easy sparse 旧 ckpt 回归无启动崩溃；从冻结 s6 checkpoint 开新 logdir 适应 1k–2k
  - verification_commands: 本地 106+ pytest；nubot tmux asset smoke + flat/踏石/圆桩 evaluator；s7 TB + 连续回放
  - success_definition: 在足够长的真洞课程和有效的躯干/小腿接触物理上评估落点与稳定性

## Commit units

1. `sparse-easy-geometry-s6`：阶段 1。
2. `sparse-grid-collision-contract-and-evaluator`：tile-filling grid、Trunk/双 Shank collision、terminal/evaluator 仪器、回归测试与 living docs。
3. 训练产物不进 git。

提交前置：实现完成 + review 无 Critical + 对应 pytest 绿。

## Known risks / blockers

- scan 的 `0.1 m` 是水平采样间距，不是高度分辨率。normal/zero/permuted 已确认策略使用 scan；后续不再重复这个 probe。
- 当前 first-contact 合法率使用 root yaw 近似 foot yaw，未加真实 scanner `x=0.04 m` offset，同步 touchdown 取最小 illegal 值；它是诊断指标，不是最终 exact-footprint gate。
- URDF 非足碰撞缺失已修，因此所有既有 checkpoint 都只能作为新物理上的 warm-start/诊断，不能继承旧 lineage 的动力学能力声明。
- 共享 URDF 会同时影响 Stage E、depth student 和 vault task；本切片只训 sparse s7，其他旧 checkpoint 在新 plant 上的能力重评是后续独立 gate。
- AMP scale=0 时 task reward 仍被 `amp_task_reward_lerp=0.7` 缩放，以及当前非论文式完整 5-frame observation / map encoder，都是 grid 修复后的次级可证伪假设。
- nubot 工作树脏；只 rsync 本切片源码，禁止 `git reset`。
- 翻箱并行，禁止改 zhuoqun checkout。

## Next skill

`verify/train`：阶段 4 `s7_grid_collision` 组合物理 lineage。s6 保留为旧 9×9 / 非足碰撞缺失 baseline；TB `:8009`。

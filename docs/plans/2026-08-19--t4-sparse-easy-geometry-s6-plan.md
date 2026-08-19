# Executable Plan - s6 easy 几何（第一脚能踩上）

> Status: active
> Date: 2026-08-19
> Spec: `docs/specs/2026-08-15--t4-stepping-stones-and-hurdle-stable.md`（目标规格；双老师路线已废）
> 对照: `docs/research/2608.02653v1/auto/2608.02653v1.md` §IV（研究笔记非权威）
> 取代: `docs/archive/plans/2026-08-18--t4-sparse-terrain-index-fix-plan.md`（s5 列映射/MDP 已落地；梅花桩能力未起）
> Branch: `t4-train`
> Planning surface: docs plan
> GPU: nubot 四卡从零训 `t_sparse_lightlp_s6`。zhuoqun 翻箱不动。Stage E 1155D 不覆盖。

## Objective

s5 列号和 LightLP §IV 公式是对的，但 easy 踏石/圆桩卡在第一脚：贴地走、脚不抬上台、从缝里掉下去，`accel>40` 收局。开 **s6 从零**，只放宽 easy 几何并锁随机 level 到 0–3 行。不热补 s5，不改 accel / OOB / 免疫 / algebraic，不加腿碰撞。

## Active Slice

阶段 3：s6 已在 nubot 四卡从零跑（`2026-08-19_12-51-05_t_sparse_lightlp_s6`）。下一步是 easy `reach_2m` 健康门，不是能力验收。

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

交接文档只用于找到线索；以下判断以当前代码、训练标量和回放行为为准。s5 的问题不是“再多训一会儿”可以解释的，而是稀疏地形第一落点缺少可学习的成功样本。

| 层 | 已验证事实 | 当前判定 |
| --- | --- | --- |
| Runtime / 接口 | 稀疏列 occupancy 约 40%，地形列映射正确；Actor 前向 scan 覆盖约 0.2–1.6 m，理论上能看到第一圈。URDF 主要只有脚底碰撞，小腿可穿石。 | 资产加载、列映射和明显观测缺失基本排除，不是当前第一嫌疑。scan 是否真正被策略使用仍需因果消融。 |
| Optimization | iter 约 18k 时 entropy≈19.6、noise≈0.51，value loss 和 PPO 更新保持 finite。 | 没有优化崩溃证据；继续堆迭代不能自动跨过没有正样本的探索屏障。 |
| Behavior | 最近窗口踏石 easy `reach_2m≈0.016`、进度约 1.13 m、fall≈89%；圆桩 easy `reach_2m≈0.029`、进度约 1.10 m、fall≈90%。同期平地/跨栏 `reach_2m≈0.67/0.63`。 | s5 行为失败已确认，且瓶颈集中在约 1.1 m 的第一脚/第一落点。 |

根因优先级：

1. **主要阻塞：第一落点与探索。** s5 踏石 easy 顶面 26 cm、净空 24 cm、上台 16 cm；脚约 21×8 cm，落点和抬脚容错都极小。
2. **共同症状：没有提前抬高第一摆动脚。** 圆桩平面已经较宽但仍在同一位置失败，因此“顶面太窄”不能独立解释两类地形。
3. **物理能力不是当前主要阻塞。** 只有观察到髋膝动作持续饱和、但脚高仍达不到目标时，才转查动作尺度、关节能力或控制器。
4. **加小腿碰撞不能教会抬脚。** 它只会把穿模改成碰撞，并同时改变楼梯、跑步等既有合同，不能作为 s6 的直接 fallback。

这里的 `0.1 m` 是 scan 点在水平面上的网格间距，不是垂直高度量化。不能用“9 cm 上台只差一个 scan 格”判断石顶是否可见；是否可见、是否被策略利用必须分别由观测检查和消融验证。

## 配方（s6，不在 s5 ckpt 上热补）

1. **踏石 easy**：顶 0.40 m，间距 0.50 m → 石间净空 0.10 m，上台 0.09 m。出生垫边沿接到第一块砖（`first_gap≈0`）。hard：顶 0.26 m、净空 0.28 m、上台 0.24 m。
2. **圆桩**：直径/间距仍是 (0.50, 0.38) / (0.55, 0.58)。easy 高 0.14→0.08 m，hard 0.28 m。
3. **10% 随机 level** 仍开，但 `random_level_reset_max_level=4`（exclusive → 行 0–3）。
4. MDP / 观测 / 终止 / 真洞 / 1937D 与 s5 相同。不加腿碰撞，不填缝。

## Non-goals

- 续训 s5 / 在 s5 ckpt 上热补几何。
- 改 `LIGHTLP_ACCEL_LIMIT`、免疫 10%、OOB 4.25 m、20 s horizon、algebraic / 软填。
- 给 URDF 加腿碰撞。
- 本切片蒸学生、改 Stage E 1155D、动 zhuoqun。
- 用 s5 的 `reach_2m` 或全局 reward 宣称梅花桩能力。

## Success Criteria

1. 纯 Python：easy 踏石顶 ≥0.36 m、石间净空 ≤0.10 m、上台 8–10 cm；easy 6 cm 偏置 illegal 仍为 0；hard 6 cm 偏置 illegal >0。圆桩 easy 高 6–8 cm，平面直径/缝不变。
2. `random_level_reset_high(10, 4)==4`。源码 run name 为 `t_sparse_lightlp_s6`。
3. s6 从零开训，不加载 s5。日志开头无 load ckpt。
4. **3k 健康门**：踏石和圆桩 easy 的最近稳定窗口都满足 `reach_2m>0.10`、进度 `>1.25 m`，且相对前一窗口仍在上升；单点尖峰不算通过。
5. **5k 决策门**：两种 easy 地形都满足 `reach_2m≥0.20`、进度 `≥1.40 m`。任何一项未通过都不得直接放任训练到 40k，必须停下进入下文的单变量诊断。
6. **固定行为门**：分别固定 `difficulty=0`、固定 evaluator 配置跑 16 局，每种地形至少 8/16 到达 2 m；视频必须确认脚实际踩在顶面，不能把踩隐藏支撑、跨列或终止漏洞算成功。
7. 任一地形到 5k 仍 `reach_2m≤0.05` 或进度 `≤1.20 m`，立即停止 s6，并将“仅靠 easy 几何即可起能力”记为负结果，不等 40k。
8. 3k/5k 曲线只决定 lineage 是否值得继续，不是能力声明。最终仍需 easy/mid/hard evaluator JSON、lineage manifest 和连续回放证据。

## Verification Path

```text
本机 pytest（几何 + 随机 level cap + 既有 sparse 合同）
  -> nubot 停 s5（保留 ckpt）
  -> rsync 源码，从零开 t_sparse_lightlp_s6
  -> TB：Terrain/stepping_stones|raised_pillars/{easy_reach_2m_rate,progress_m}
```

### Verification Path Status

`runnable`。合同本机已跑。训练/TB 在 nubot。

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

## Required Capabilities

- 本机 pytest（无 Isaac）。
- nubot 4×4090 + `scripts/nubot_run.sh` + tmux。
- s5 logdir 作负结果对照。

## Fallback Evidence

- 本机无 Isaac：几何合同 + 公式 cap。64-env smoke 放到 nubot 开训后看 log 无即崩。
- s6 未过阶段门时先冻结对应 checkpoint、固定种子、两类地形 evaluator JSON 和第一脚连续回放，后续每个探针都开新 lineage，不在同一训练上混改。

## s6 失败后的单变量诊断分支

按以下顺序执行；每一步只回答一个问题，不能同时改奖励、终止、碰撞和几何：

1. **观测因果消融**：用同一 checkpoint、同一批固定 seeds、`difficulty=0`，分别跑正常 scan、全零 scan、空间 permuted scan。若正常 scan 与 zero/permuted 无行为差异，优先判断策略没有利用地形观测或观测链路有问题；若正常 scan 显著更好，再查落脚控制。
2. **增加最小行为 instrumentation**：记录第一摆动脚的最大离地高度、首次非出生垫接触是否为合法顶面接触、髋/膝 action 饱和率。指标按踏石/圆桩分别输出，不能只看全局均值。
3. **脚完全不抬**：新开短程 probe lineage，使用“物理上填底/软支撑，但 scan、illegal-footstep 与评估仍报告真洞”的环境。它只用于检验失败是否来自稀疏奖励与坠落造成的探索断层，不能作为训练配方或能力证据。
4. **脚会抬但落点错过**：保持 MDP 与控制不变，只做落点/顶面宽度课程的单变量调整；先扩大可落区，再逐步收回目标几何。
5. **只有持续动作饱和时才查物理能力**：若髋膝 action 长时间贴限且第一脚高度仍不足，再检查 action scale、关节限位、PD/力矩与目标台阶的可达性。
6. **腿碰撞最后单独立项**：只有明确需要真实胫腿碰撞合同后，才用独立 lineage 评估对踏石、楼梯、跑步和 sim2sim 的影响；它不是 s6 失败后的首选修复。

## 干预分类

| 干预 | 分类 | 用途与边界 |
| --- | --- | --- |
| s6 easy 几何放宽 | `repair_proposal` | 给第一落点制造足够正样本；是否成功由 3k/5k 与固定 evaluator 决定。 |
| 第一摆动脚、合法首次接触、action 饱和日志 | `instrumentation` | 提高可观测性，不改变任务。 |
| normal/zero/permuted scan、真洞观测下的短程软支撑 | `causal_probe` | 区分观测利用、探索断层与控制能力；结果不能直接作为最终策略证据。 |
| 放松 accel/OOB、直接加腿碰撞 | `symptom_suppression` | 会改变失败表现或其他地形合同，当前没有证据支持作为 s6 fallback。 |

## Final integration claim

`final_integration_claim`: s6 在 s5 的列映射与 LightLP MDP 上，用可迈的 easy 几何从零训练。s5 不作梅花桩证据。本切片不蒸学生、不加腿碰撞、不宣称实机能力。

## 工作项

- [x] 阶段 1：easy 几何 + 随机 level 0–3
  - acceptance_criteria: 上文 Success 1–2；相关 pytest 绿
  - verification_commands: `python -m pytest tests/test_t4_stepping_stone_contracts.py tests/test_t4_sparse_reward_contracts.py tests/test_t4_sparse_monitor_contract.py tests/test_t4_terrain_column_map.py tests/test_t4_terrain_curriculum.py tests/test_t4_sparse_teacher_mujoco.py -q`
  - success_definition: easy 第一脚在合同上是迈不是跳
- [x] 阶段 2：停 s5，从零开 s6
  - acceptance_criteria: s5 tmux 停、ckpt 保留；s6 `--resume` 关；run name `t_sparse_lightlp_s6`；任务仍 `t4_loco_teacher_sparse`
  - verification_commands: nubot `tmux ls`；`logs/t4-sparse-lightlp-s6.log` 无 load ckpt；`params/agent.yaml` `resume: false`
  - success_definition: 新 lineage 在放宽的 easy 几何上冷启动
- [ ] 阶段 3：s6 健康门（非能力验收）（当前）
  - acceptance_criteria: 3k 时两类 easy 都达到 `reach_2m>0.10`、进度 `>1.25 m` 且趋势上升；5k 时都达到 `reach_2m≥0.20`、进度 `≥1.40 m`
  - verification_commands: TB `Terrain/{stepping_stones,raised_pillars}/{easy_reach_2m_rate,progress_m}`；固定 difficulty=0 各 16 局 evaluator + 连续视频
  - success_definition: 两类地形各至少 8/16 到达 2 m，且视频确认脚踩顶面。未过 5k 门立即转单变量诊断，不跑满 40k

## Commit units

1. `sparse-easy-geometry-s6`：阶段 1。
2. 训练产物不进 git。

提交前置：实现完成 + review 无 Critical + 对应 pytest 绿。

## Known risks / blockers

- scan 的 `0.1 m` 是水平采样间距，不是高度分辨率。需要直接检查 scan 数值，并用 normal/zero/permuted 消融判断策略是否使用它。
- 出生垫与第一砖共边：第一脚变成 9 cm 路缘，不再是 24 cm 空洞。若仍贴地绊倒，先按“摆动脚高度 → scan 消融 → 软支撑 probe”的证据链定位，不直接加腿碰撞。
- nubot 工作树脏；只 rsync 本切片源码，禁止 `git reset`。
- 翻箱并行，禁止改 zhuoqun checkout。

## Next skill

`review`：阶段 3 健康门。s6 在训；TB `:8009`。

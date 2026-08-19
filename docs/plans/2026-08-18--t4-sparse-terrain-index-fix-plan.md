# Executable Plan - 地形列号修复 + LightLP MDP 对齐 + s5 从零

> Status: active
> Date: 2026-08-18
> Spec: `docs/specs/2026-08-15--t4-stepping-stones-and-hurdle-stable.md`（目标规格；双老师路线已废）
> 对照: `docs/research/2608.02653v1/auto/2608.02653v1.md` §IV（研究笔记非权威）
> 取代: `docs/archive/plans/2026-08-17--t4-sparse-lightlp-complete-plan.md`（s4 配方仅作历史，训练验收作废）
> Branch: `t4-train`
> Planning surface: docs plan
> GPU: 阶段 1–2 本机改代码。阶段 3 起 nubot 四卡从零训 `t_sparse_lightlp_s5`。zhuoqun 翻箱不动。Stage E 1155D 不覆盖。

## Objective

先修 IsaacLab `terrain_types`（列号）被当成 `sub_terrains` 下标的 P0，再按 LightLP §IV 补上 10% 随机 level reset 与 Eq.4/5 / opposite / 路径长度晋级。修完后 **从零** 开 `t_sparse_lightlp_s5`。s4 ckpt 只当错误实现基线，不续训、不宣称梅花桩能力。

## Active Slice

阶段 3 已开：nubot 四卡从零 `t_sparse_lightlp_s5`（`2026-08-19_00-20-19_t_sparse_lightlp_s5`）。s4 ckpt 保留。下一步是健康门，不是能力验收。

## 已核实的 s4 现状（iter 12018，勿当梅花桩证据）

来源：`logs/t4_loco_teacher_sparse/2026-08-18_16-01-39_t_sparse_lightlp_s4`，nubot TB `:8007`，EventAccumulator 全量标量（s4 已停，ckpt `model_12500.pt`）。

真实列（IsaacLab 2.1.0：`index/num_cols+0.001 < cumsum(proportion)`）：

```text
col 5     hurdles
col 6-9   stepping_stones
col 10-13 raised_pillars
```

代码用 `names.index`，所以 TB 名是错的。按下表读：

| TB 名 | 实际列 | reach_2m | progress | fall | pit_fall | 说明 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `flat` | col0 flat | 0.74 | 3.55 m | 0.34 | 0 | 连续地形会走 |
| `random_rough` / `boxes` | col1–2 rough | 0.72–0.77 | 3.42 m | 0.37–0.42 | 0 | 同上 |
| `wave` | col3 boxes | 0.76 | 3.50 m | 0.36 | 0 | 同上 |
| `hurdles` | col4 wave | 0.70 | 3.29 m | 0.42 | 0 | 同上 |
| **`stepping_stones`** | **col5 hurdles** | **0.78** | **2.96 m** | 0.68 | ≈0 | 不是踏石；solid floor |
| **`raised_pillars`** | **col6 stones[0]** | **0.024** | **1.17 m** | 0.83 | 0.36 | 唯一进 sparse mask 的踏石列 |
| `slope_up` / `slope_down` / `stairs_up_30` | col7–9 stones[1–3] | 0.03–0.05 | 1.23–1.27 m | 0.84–0.88 | **0** | 踏石，但没进 mask |
| `stairs_up_34` / `stairs_down_30` / `stairs_down_34` | col10–12 pillars[0–2] | 0.016–0.038 | 1.21–1.30 m | 0.84–0.91 | **0** | 圆桩；col13 无日志 |

全局：reward 28.3（5k 时约 32）、length 498 step ≈ 10.0 s、levels 3.44、slack 0.27–0.35、`illegal` −6e-4、noise 0.49、AMP loss 0.23。PPO 没崩。

读法：

1. 全局 `episode_length≈500`（10.0 s）**不是** 20 s 上限，也不是第一优先级要拧的旋钮。`step_dt=0.02`，满局是 1000 step。LightLP OOB 在 Chebyshev 4.25 m（半格 4 m + 0.25 m，好让轴对齐走完还能晋级）。跟令 0.4–0.5 m/s 走到出界正好 8–11 s。连续地形 `timeout_rate` 0.58–0.66 且 `progress` 3.3–3.5 m，是出界超时，不是 horizon。稀疏列 85% 摔在 1.2 m，把均值往下拉；站立局若没摔能撑满 1000，往上拉。两头一混，均值就钉在 ~500。s5 即使踏石学会走，成功局仍会在 4.25 m 出界，全局 length 仍可能停在 500–700。要看的是分地形 `timeout` / `progress` / `reach_4m`，不要把全局 length 拉到 1000。
2. 连续地形 / 跨栏能走到 3 m 附近；不是 accel 闸把所有人掐死。accel / 躯干 / 倾倒记在 `fall`（`~timeout`）里，连续地形大约 35–40%。
3. **四列踏石和三列可见圆桩都卡在约 1.2 m、reach_2m 2–5%、几乎零 success。** 他人报告的「踏石 71% / 圆桩 5%」是跨栏 vs 一列踏石。
4. 只有 col5+col6 被当成 sparse：illegal / AMP 关闭 / pit-fall 统计都打偏。其余踏石和全部圆桩仍吃 AMP/步态，pit_fall 记成 0。
5. `raised_pillars/hard_episodes=0` 是因为踏石走不出 4 m、又没有 10% 随机 level。`stepping_stones/hard_episodes=2693` 是跨栏在晋级。
6. 不要用放开 `LIGHTLP_ACCEL_LIMIT`、推迟 OOB、或 `use_algebraic_sparse_scan=True` 去「拉高 episode length」。40 与 10% 免疫是论文原值；algebraic 是软填洞图，且仍走错 type id。

## 配方（一次打进 s5，不在 s4 ckpt 上热补）

1. **列映射**：保留 Isaac 的 `terrain_types` 为列号。用与 `TerrainGenerator._generate_curriculum_terrains` 相同的公式建 `column_to_name`。sparse / illegal / AMP / 步态 / hurdle_bar / pit-fall / TB 全部走列集合。
2. **监控**：按真名聚合所有列；另打每列 occupancy 与 level 直方图。hard 带与 runtime 一致（现在是 `(level * 3) // 10`，level 7–9）。
3. **`random_level_reset_fraction=0.10`**：每次 reset 独立抽样，level 覆盖 0–9。与 `LIGHTLP_IMMUNITY_FRAC` 无关。
4. **Eq.4**：双脚求和，不再 `.mean()`。
5. **Eq.5**：`ẽ = αẽ + Σ_i max(|a_i|-30, 0)`，`α=e^{-Δt/τ}`，罚 `ẽ`。不要先滤 `|a|` 再减阈值。
6. **opposite**：`max(0, -v · v̂)`，连续点积，不是 0/1。
7. **晋级**：累计路径长度 + 命令跟踪良好；不再用峰值径向位移单独决定。
8. 真洞、1937D Actor、足底 scan 只进 Critic、无双 Critic / `legal_foothold`：保持 s4 已落地部分。

## Non-goals

- 续训 s4 / v4 / v2 任何 ckpt。
- 改 `LIGHTLP_ACCEL_LIMIT`、免疫比例、OOB 4.25 m、20 s horizon，或打开 algebraic / 软填。全局 length≈500 不当作要修的失败。
- 本切片蒸学生、改 Stage E 1155D、动 zhuoqun 翻箱。
- 用 s4 的 `Terrain/stepping_stones|raised_pillars` 曲线宣称能力。
- 独立梅花桩 skill、新 AMP clips。

## Success Criteria

1. 纯 Python：20 列映射与上表一致；`stepping_stones` 列集合 = {6,7,8,9}，`raised_pillars` = {10,11,12,13}，`hurdles` = {5}。
2. 源码不再用 `names.index(...)` 去比 `terrain_types`。
3. Eq.4 单脚接触 = 全额；Eq.5 / opposite / 10% reset 有代数测试。
4. s5 从零开训，不加载 s4。TB 真名下踏石+圆桩 occupancy 合计约 0.40；`hard_episodes` 在 10% reset 后早期非零。
5. 能力声明仍要等 evaluator + 回放。s5 的 `reach_2m` 离开 0 只是健康门，不是完成。

## Verification Path

```text
本机 pytest（列映射 + 奖励代数 + 既有 sparse 合同）
  -> 确认不再 names.index 比列号
  -> nubot 停 s4（保留 ckpt）
  -> 从零开 t_sparse_lightlp_s5
  -> TB：按真名的 stones/pillars reach_2m、illegal、hard_episodes、occupancy
  -> 之后才回放 / evaluator
```

### Verification Path Status

`runnable`。映射与公式合同本机可跑。s5 训练/TB 在 nubot；SSH 不稳时开训 deferred，不改配方范围。

## Required Capabilities

- 本机 pytest（无 Isaac）。
- nubot 4×4090 + `scripts/nubot_run.sh` + tmux（阶段 3）。
- 已有 s4 logdir 作错误基线对照。

## Fallback Evidence

- 本机无 Isaac：用官方 proportion 公式钉死 20 列合同；64-env smoke 放到 nubot 开训前。
- nubot SSH 挂：阶段 1–2 先合入；s5 deferred。
- s5 开训后真名踏石/圆桩 `reach_2m` 仍长期为 0：先查 occupancy 与 sparse mask，不要先拧 accel。

## Final integration claim

`final_integration_claim`: s5 在正确列映射 + LightLP 10% 随机 level 与 Eq.4/5/opposite/路径晋级上从零训练。s4 不作梅花桩证据。本切片不蒸学生、不宣称实机能力。

## 工作项

- [x] 阶段 0：审计 + 按真列重读 s4 TB
  - acceptance_criteria: P0 列号错位已用 IsaacLab 2.1.0 源码与 20 列公式钉死；s4 @12018 的错位表已写入本计划
  - verification_commands: IsaacLab `terrain_importer.py` `_compute_env_origins_curriculum`；本仓库 `T4_SPARSE_TERRAIN_PROPORTIONS`；nubot EventAccumulator dump
  - success_definition: 不再把 `Terrain/raised_pillars` 当圆桩
- [x] 阶段 1：列号映射、监控、合同
  - acceptance_criteria: Isaac-free helper 复现官方列分配；`sparse_foothold` / `hurdle` / TB 走列集合；有每列名、occupancy、level hist；pytest 覆盖 20 列真值；无 `names.index` 比 `terrain_types`
  - verification_commands: `python -m pytest tests/test_t4_terrain_column_map.py tests/test_t4_sparse_monitor_contract.py tests/test_t4_sparse_reward_contracts.py -q`
  - success_definition: 再跑同样课表时，踏石/圆桩标签不会再打到跨栏/单列上
- [x] 阶段 2：LightLP MDP 对齐
  - acceptance_criteria: 10% 随机 level；Eq.4 求和；Eq.5 先超限再泄漏积分再双脚求和；opposite 为连续点积；晋级用累计路径 + 跟令质量；不改 accel 40 / 免疫 10% / algebraic
  - verification_commands: `python -m pytest tests/test_t4_sparse_reward_contracts.py tests/test_t4_terrain_curriculum.py -q`
  - success_definition: 论文第 140 行免疫和第 150 行随机 level 是两套开关；公式与 Table I / Eq.4–5 同形
- [x] 阶段 3：停 s4，从零开 s5
  - acceptance_criteria: s4 tmux 停、ckpt 保留；s5 `--resume` 关；run name `t_sparse_lightlp_s5`；任务仍 `t4_loco_teacher_sparse`
  - verification_commands: nubot `tmux ls`；`logs/t4-sparse-lightlp-s5.log` 开头无 load ckpt
  - success_definition: 新 lineage 在修过的 MDP 上冷启动
- [ ] 阶段 4：s5 健康门（非能力验收）（当前）
  - acceptance_criteria: 真名踏石+圆桩 occupancy≈0.40；两侧 `hard_episodes` 早期非零；`illegal` 明显负于 s4 的 −0.001；连续地形 length/reward 不崩
  - verification_commands: TB `Terrain/stepping_stones|raised_pillars/{reach_2m_rate,hard_episodes,episodes}` 与 per-column occupancy
  - success_definition: 监控不再撒谎，稀疏列能进 hard。能力仍要 evaluator + 回放

## Commit units

1. `sparse-column-map`：阶段 1。
2. `sparse-lightlp-mdp`：阶段 2。
3. 训练产物不进 git。s5 健康门的 TB 摘录可进 `artifacts/eval/`。

提交前置：实现完成 + review 无 Critical + 对应 pytest 绿。

## Known risks / blockers

- 映射和公式一起改会难归因：必须先合阶段 1 测绿，再合阶段 2，然后才开 s5。
- 本机可能无 Isaac：阶段 1 以公式合同为主，runtime smoke 放 nubot。
- 路径长度晋级会让连续地形更快涨 level；10% 随机保证稀疏 hard 仍有人。
- nubot s4 仍占四卡。阶段 3 再停，阶段 1–2 不需要 GPU。
- 翻箱并行，禁止改 zhuoqun checkout。

## Next skill

`review`：阶段 4 健康门。s5 在训；Reset/* 日志已写入代码，当前进程未重载所以 TB 还没有这些曲线。

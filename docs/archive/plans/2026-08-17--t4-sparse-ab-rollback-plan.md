# Executable Plan - 回退 S1d 厨房水槽实验，恢复 T-compat v2 配方

> **Status: archived（非权威）** — 续训 v2 已被从零 LightLP 单阶段取代。living 梅花桩: `docs/plans/2026-08-22--t4-sparse-s12-rim-yaw-student-plan.md`。入口：`docs/README.md`。

> Status: superseded
> Date: 2026-08-17
> Spec: `docs/specs/2026-08-15--t4-stepping-stones-and-hurdle-stable.md`（user-approved）
> 取代: `docs/archive/plans/2026-08-15--t4-stepping-stones-and-hurdle-stable-plan.md` 的 S1d 执行面
> 被取代: `docs/archive/plans/2026-08-17--t4-sparse-lightlp-complete-plan.md`
> Branch: `t4-train`
> Planning surface: docs plan
> GPU: nubot 2 卡恢复 T-compat；另 2 卡空出。zhuoqun 翻箱不动

## Objective

把稳定组 / 对照组（S1d）判定为失败实验：停训、回退其代码，回到最后一次真正学会踏石的 T-compat v2 配方，再只开一刀文献里真正有效的圆桩杠杆。

```text
停 A/B
  -> 回退 Dual Critic / 脚下 Critic scan / 稀疏关步态·AMP·stumble / 掉坑免 −200 / legal_foothold
  -> 保留 v2 踏石几何 + v3 圆桩第一跨 + illegal/slack + 掉坑终止 + 分地形 TB
  -> 从 v2 model_3500 续 T-compat
  -> 踏石恢复后再单独做圆桩两阶段软地形
```

## Active Slice

阶段 2/3：停 A/B，从 v2 `model_3500` 续 `t4_loco_teacher_sparse`。

## 失败分析（本切片证据）

原始事件流已落到 `artifacts/eval/t4_sparse_ab_tb_dump.json`。同迭代对齐如下。

**3.5k 对齐（v2 停训点附近）：**

| run | reward | length | 踏石 `reach_2m` | 踏石 success | 圆桩 `reach_2m` | 圆桩 progress |
|---|---:|---:|---:|---:|---:|---:|
| v2 | **57.0** | 857 | 0.75 | 0.35 | ≈0 | 1.02 m |
| 稳定组 | **48.1** | 893 | 0.76 | 0.43 | ≈0 | 1.13 m |
| 对照组 | **43.6** | 807 | 0.75 | 0.37 | ≈0 | 1.08 m |

**各 run 末值：**

| run | iter | reward | length | 踏石 `reach_2m` / success | 圆桩 `reach_2m` / success / progress |
|---|---:|---:|---:|---|---|
| Stage E `prov7` | ~18k | ~70.6 | ~995 | n/a | n/a |
| v2 | 3874 | 56.8 | 838 | 0.80 / 0.39 | ≈0 / **0** / 1.03 m |
| v3 | 1593 | 35.6 | 654 | 0.62 / 0.27 | 0.009 / **0** / 1.15 m |
| 稳定组 | 11629 | 56.4 | 934 | 0.81 / 0.45 | 0.019 / **0** / 1.13 m |
| 对照组 | 11795 | 57.5 | 881 | 0.79 / 0.41 | ≈0 / **0** / 1.02 m |

读数要点：

- **踏石没有死。** 3.5k 时三条 `reach_2m` 都在 0.75–0.76。S1d 11k 末值 0.80，只是把 v2 已经走到的曲线又走了一遍。严格 4 m success 仍只有 0.39–0.45，回放仍会摔，所以目视会觉得「没成功」。
- **圆桩全程失败。** success 全是 0，progress 钉在出生台到第一桩（≈1.0–1.2 m）。`reach_1m` 有 0.5–0.7，说明走到台边就掉。`legal_foothold` 稳定组末值 4e-6，对照组 4e-4，等于没发过奖。
- **同迭代全局 reward 确实掉了。** 3.5k：v2 57 vs 稳定组 48 vs 对照组 44。对照组 1k 只有 15.8 / length 400，开局明显更烂。这就是「性能下降很明显」的 TB 对应物。相对 Stage E（70 / 995）更差，因为课表 40% 是坑。
- 对照组（双 Critic + 掉坑免 −200 + AMP=0）圆桩比稳定组更差，楼梯 `reach_2m` 也更差（0.037 vs 0.062）。双 Critic 这刀可以判负。
- S1d 一次叠了：稀疏关步态/stumble、AMP×0.1 或 0、Critic +30D 脚下 scan、对照组再加双 Critic 与掉坑免 −200。无法归因，ckpt 还不能迁回单头。
- BeamDojo Naive（只加落足项、单阶段）= 0.33%；拉开差距的是两阶段软地形 42%→92%。S1d 没走那条路。

根因判断：

1. **圆桩是第一落点/探索问题，不是再缺一条奖励。** 方砖 48 cm 原谅偏脚；圆盘+坑在第一跨吃掉几乎全部 episode。
2. **S1d 没救圆桩，还把全局 reward 拉低，对照组更差。** 失败实验该回退。
3. **踏石曲线本身可保留为 v2 配方的证据，不要再从零赌厨房水槽。**
4. **不是感知不够。** T-paper 接触已否。不要改 1155D。

## Non-goals

- 续训稳定组 / 对照组，或把它们的 ckpt 迁到单头 Critic。
- 再加落足奖励、Actor 接触、规划器、GRU、改 1155D。
- 回退 v2 踏石几何、illegal-footstep、velocity-slack、掉坑终止、分地形 TB。
- 回退 v3 圆桩间距（easy 第一跨 5 cm 仍对齐 Isaac / BeamDojo easy 间隙；单独保留）。
- 打断 zhuoqun 翻箱。
- 用 reward / terrain_levels / ckpt 存在宣称梅花桩能力。

## Success Criteria

1. 仓库不再注册 `t4_loco_teacher_sparse_stable` / `_allin`；`DualValueActorCritic`、`sparse_split`、critic 脚下 scan、`legal_foothold` RewTerm 从训练路径消失。
2. `t4_loco_teacher_sparse` 仍是 1155D；合同 pytest 绿。
3. nubot A/B 训练已停，日志和 `model_11000.pt` 留作失败证据，不删。
4. 回退后从 v2 `model_3500.pt` 续 `t4_loco_teacher_sparse`；2k 内踏石 `reach_2m` 回到 ≥0.6（v2 在 3.8k 是 0.80）。
5. 圆桩下一刀只能是「两阶段软地形」，不得和步态/AMP/双 Critic 一起开。

## Verification Path

```text
本机合同 pytest
  -> 确认任务注册只剩 sparse / sparse_paper
  -> nubot 停 A/B，补齐 stable/allin 分地形 TB JSON
  -> 回退后从 v2 model_3500 续 T-compat
  -> 2k：踏石 reach_2m ≥0.6，圆桩仍允许接近 0
  -> 5k：踏石不回落；才允许单独开圆桩软地形探针
```

### Verification Path Status

`runnable`。合同测试本机可跑。nubot 训练/TB 依赖 SSH；2026-08-17 00:15 后出现过超时，implement 时先确认连通。

## Required Capabilities

- 本机 pytest（纯 Python 合同）。
- nubot SSH + 2 张空闲 4090 续 T-compat。
- 已有失败证据：v2 `2026-08-16_13-18-07_t_compat_sparse_v2/model_3500.pt`；A/B run dir 与 1.2 GB 日志。

## Fallback Evidence

- nubot SSH 不稳：先完成本机回退与 pytest；停训/续训记 deferred，不改回退范围。
- v2 ckpt 与回退后 obs 宽度不一致：禁止硬续，改为同任务从零，并写明差在哪一维。
- 续训 2k 踏石仍 <0.5：停，先查是否误把 S1d 残留留在 MDP，而不是再加项。

## Final integration claim

`final_integration_claim`: S1d 厨房水槽被判定失败并移出训练路径；T-compat 回到可续 v2 的 1155D 配方。不宣称圆桩或梅花桩能力。

## 工作项

- [x] 阶段 0：失败数据分析
  - acceptance_criteria: 四条 lineage 的 iter / reward / length / 踏石与圆桩 `reach_2m` 对齐；S1d 旋钮清单写清
  - verification_commands: nubot 日志 `Learning iteration` + `/tmp/dump_sparse_tb.py` 事件流
  - success_definition: 回退范围由数据而不是直觉决定
- [x] 阶段 1：回退 S1d 实验代码
  - acceptance_criteria: 删除任务 `t4_loco_teacher_sparse_stable` / `_allin`、`T4SparseSplitCfg`、`DualValueActorCritic` 与 rsl_rl 双头 GAE、`compute_critic_foot_scan` 拼进 Critic、稀疏 gait/AMP/stumble 缩放、`penalize_pit_fall=False`、`legal_foothold` RewTerm；保留 `t4_loco_teacher_sparse`、v2 踏石、v3 圆桩几何、illegal/slack、掉坑终止、分地形 TB、足底射线只服务 reward
  - verification_commands: `python -m pytest tests/test_t4_observation_contracts.py tests/test_t4_sparse_reward_contracts.py tests/test_t4_sparse_monitor_contract.py tests/test_t4_stepping_stone_contracts.py tests/test_t4_sparse_evaluator_contract.py -q`
  - success_definition: 训练路径回到单头 1155D T-compat，S1d 符号搜不到
- [x] 阶段 2：停 A/B 并固化证据
  - acceptance_criteria: tmux `t4-sparse-stable` / `t4-sparse-allin` 已停；ckpt 与日志保留。分地形 JSON 已在 `artifacts/eval/t4_sparse_ab_tb_dump.json`
  - verification_commands: nubot `tmux ls`；`nvidia-smi`
  - success_definition: 失败实验可复核，GPU 已释放
- [ ] 阶段 3：从 v2 `model_3500` 续 T-compat（当前）
  - acceptance_criteria: 任务 `t4_loco_teacher_sparse`，run name 标明 `t_compat_sparse_v2_resume`；2k 内踏石 `reach_2m≥0.6`；圆桩允许仍接近 0
  - verification_commands: nubot tmux 日志 + TB `Terrain/stepping_stones/reach_2m_rate`
  - success_definition: 踏石学习曲线回到 v2 轨迹，而不是再从零赌厨房水槽
- [ ] 阶段 4：圆桩只做两阶段软地形（下一步，不与阶段 3 同时开）
  - acceptance_criteria: 仅改圆桩生成：Stage 1 坑填实、踩空只罚不终止；Stage 2 再打开真坑。不改 Actor、不加奖励项、不关步态/AMP
  - verification_commands: 几何合同 + 单独 2 卡探针；5k 看圆桩 `reach_2m` 是否离开 0
  - success_definition: 用 BeamDojo 已验证的那一刀，而不是再叠 MDP

## Commit units

1. `S1d-rollback`：阶段 1 代码与合同测试（review 后）。
2. 训练产物不进 git。阶段 2 的 TB JSON 可进 `artifacts/eval/`。

## Known risks / blockers

- A/B ckpt critic 宽度 / 双头与回退后不兼容，续训只能用 v2 `model_3500`。
- v3 圆桩几何仍在课表里，续 v2 权重时圆桩域有偏移；踏石几何未变。
- 40% 稀疏混合会继续拖累全局 reward，这是课表代价，不是再加项的理由。
- nubot SSH 不稳定；长命令必须 tmux。
- 翻箱与本工作面并行，禁止改 zhuoqun checkout。

## Next skill

监控 `t_compat_sparse_v2_resume`：2k 内踏石 `reach_2m≥0.6`。未到 gate 前不要开阶段 4。

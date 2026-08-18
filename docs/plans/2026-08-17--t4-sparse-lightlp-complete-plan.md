# Executable Plan - LightLP §IV 梅花桩完整补全（从零、收窄踏石、新观测）

> Status: active
> Date: 2026-08-17
> Updated: 2026-08-18 — 撤回 BeamDojo 软/硬二阶段；现行 lineage 为单阶段真洞 `t_sparse_lightlp_s4`
> Spec: `docs/specs/2026-08-15--t4-stepping-stones-and-hurdle-stable.md`（批准规格；执行以本文件为准，不再开 T-compat/T-paper 双老师）
> 对照: `docs/research/2608.02653v1/auto/2608.02653v1.md` §IV（研究笔记非权威）
> 取代: `docs/archive/plans/2026-08-17--t4-sparse-ab-rollback-plan.md`
> Branch: `t4-train`
> Planning surface: docs plan
> GPU: nubot 4×4090 从零训 `t_sparse_lightlp_s4`。zhuoqun 翻箱不动。Stage E `t4_loco_teacher` 1155D 不覆盖。

## Objective

按 LightLP 感知走跑老师（§IV）补全梅花桩配方，**从零**训一条新 sparse teacher。踏石收窄到 `illegal_footstep` 能咬到偏脚；同一课表上开真洞，**不再**做软填再切真坑。本 lineage **可以打破 1155D**。不做旧学生短 FT，不续任何旧 ckpt。本切片不蒸学生；梅花桩学生与走跑/翻箱合并是后续工作面。

用户 2026-08-17 裁定：

1. 可以重头训练，不续 `t_compat_sparse_v2_resume` / v2 `model_3500`。
2. 不要短 FT。
3. 现在的方砖太大，illegal 沉默、落足项等于没干活；必须改观。Actor 1155D 不必保留。

## Active Slice

阶段已改：撤回软/硬二阶段。现行 lineage 为单阶段真洞 `t_sparse_lightlp_s4`（任务仍是 `t4_loco_teacher_sparse`）。旧 v4 hard 已停，ckpt 仅作对照。

## 配方（一次打齐，不是再拧一颗螺丝）

同一条老师、同一个任务名 `t4_loco_teacher_sparse`。默认 Stage E `t4_loco_teacher` 仍是 1155D，给翻箱/旧学生用。

### 几何

脚约 21×8 cm，足底 scan 16×8 cm。现行踏石 easy 48 cm、第一跨 32 cm，偏十几厘米仍整脚在砖上 → `illegal≈-0.001`。

新踏石必须同时满足：

- easy 顶面边长 ∈ **[0.24, 0.30] m**（略大于脚，3–5 cm 偏置就开始把 scan 打出砖外）
- hard 顶面边长 ∈ **[0.20, 0.24] m**，且 hard < easy
- easy 第一跨（出生台边到第一块砖边）∈ **[0.03, 0.10] m**，是迈不是跳
- hard 第一跨 > easy，且 ≤ **0.18 m**
- 中心距随难度单调增、顶面单调减；9×9 阵仍装进 8 m tile
- 石顶高度保持现范围（easy ≥0.16 m），0.1 m 前向 scan 仍能看见

圆桩保持 v3：间距 (0.55, 0.58)、直径 (0.50, 0.38)、easy 第一跨 5 cm 左右。

合同里加一条 **illegal 不再沉默**：中心站立分数≈0；沿边偏置 6 cm 后分数 >0。

### 观测（新常量，不改 `TEACHER_ACTOR_OBS_DIM=1155`）

| 通道 | 新 sparse teacher | 为什么 |
|---|---|---|
| 本体史 10 帧 | 保留 | 已有 |
| 前向 scan | history **1→5**（LightLP 叠 5 帧） | 现在 scan 只有当前帧 |
| 左右脚接触 | **进 Actor**（2D） | LightLP §IV-B；旧 T-paper 是真坑上试的，这里跟软地形一起重做 |
| 足底 scan 30D | **只进 Critic** | LightLP Critic 特权；实机没有朝下足雷达，不进 Actor |
| 冲击免疫 flag | 只进 Critic（若做 10% 免疫） | §IV-C2 |

名义维：Actor `96*10 + 195*5 + 2 = 1937`；Critic 再加脚下 30（及可选 1D flag）。以 `schemas.py` 算出的常量 tes t 为准。

禁止字段仍禁止：`terrain_id` / 全局地图 / 成功标签。打破 1155 ≠ 把不能蒸馏的真值塞进 Actor。

删除任务 `t4_loco_teacher_sparse_paper`：接触已经并进主任务。

### MDP（稀疏 tile 上）

- 保留 `velocity_slack +1.5`、`illegal_footstep −1.0`。
- `gait_feet_*` ×0、AMP ×0、`feet_stumble` ×0。连续地形保持现配方。
- 新增 LightLP 滤波足加速度（τ=0.06 s，超 30 m/s² 的 EMA，−0.01）。与现有 `foot_touchdown_impact` 合并成一套，不要两套各打各的。
- 新增 `opposite_direction −1.0`。
- 不加 `legal_foothold`，不装双 Critic。
- **单阶段真洞**：踏石/圆桩一开始就是真间隙。`pit_fall −200` 不是稀疏失败模式（LightLP 无此项）；掉坑由躯干接触 / 姿态 / 超时等终止捕获。
- 下文若仍出现「软阶段 / 硬阶段 / `t_sparse_lightlp_v4`」，视为 2026-08-17 草稿，**已撤回**。

### 训练

- 旧 v4 软/硬与更早 lineage 的日志/ckpt 留作对照，**不加载**。
- `bash scripts/nubot_run.sh` + 四卡 `torch.distributed.run`，任务 `t4_loco_teacher_sparse`，`--resume` 关闭，每 rank 1024 env，AMP 仍走 `artifacts/amp_expert_provisional`。
- run name：`t_sparse_lightlp_s4`。
- 验收看分地形 `reach_2m` / `illegal` / 回放，不要切软硬。

## Non-goals

- 续训 v2 / v2_resume / S1d / T-paper 旧 ckpt。
- 旧 depth 学生短 FT，或本切片蒸新学生。
- 独立梅花桩 skill、新 AMP clips、落足规划器。
- 双 Critic、`legal_foothold`。
- 改默认 `TEACHER_ACTOR_OBS_DIM=1155`（Stage E / 翻箱 / 旧学生仍用它）。
- 把足底 scan 写进导出 `pi_loco`。
- 打断 zhuoqun 翻箱。
- 用 reward / `terrain_levels` / ckpt 存在宣称梅花桩能力。

## Success Criteria

1. 合同：新踏石几何、illegal 偏置 >0、软地形「站得住仍报洞」、新 Actor/Critic 维、paper 任务已删。pytest 绿。
2. 默认 teacher 仍 1155D；sparse 任务是新维，用 1155 loader 硬失败。
3. nubot 从零健康开训，不加载旧 ckpt。
4. 真洞课表上踏石与圆桩 `reach_2m` 都离开 0；`illegal` 不再是 −0.001。
5. 两边都能连续走，不是只活过第一跨。能力声明必须有分地形 TB + 回放 + evaluator，不要单靠 reward。

## Verification Path

```text
本机合同 pytest
  -> 确认 1155 默认未改、sparse 新维、paper 任务消失
  -> nubot 停旧续训，从零开 t_sparse_lightlp_s4
  -> TB：踏石/圆桩 reach_2m、illegal
  -> 回放 / evaluator（无学生 FT）
```

### Verification Path Status

`runnable`。合同测试本机可跑。训练/TB 在 nubot；SSH 不稳时训练记 deferred，不改配方范围。

## Required Capabilities

- 本机 pytest（无 Isaac）。
- nubot 4×4090 + `scripts/nubot_run.sh` + tmux。
- 已有对照：`t_compat_sparse_v2_resume` 踏石 hard 摔倒 0.50、圆桩 progress 0.95 m、`illegal≈-0.001`。

## Fallback Evidence

- 双 prim 软地形在 Isaac 里走不通：改用布局代数写 scan/illegal，合同仍要「站得住 + 该点算洞」。
- nubot SSH 挂：先交阶段 1–2 代码与 pytest；开训 deferred。
- 软阶段 8k 圆桩 `reach_2m` 仍 <0.1：停，查 scan 是否真的在看洞、填缝是否真的托住脚，不要加回双 Critic。

## Final integration claim

`final_integration_claim`: 新的从零 sparse teacher 按 LightLP §IV 单阶段真洞 + 能咬偏脚的踏石几何训练。默认 1155D Stage E 不动。本切片不蒸学生、不宣称实机梅花桩。

## 工作项

- [x] 阶段 0：失败诊断与 LightLP 对照
  - acceptance_criteria: 踏石/圆桩都不算会；S1d 真坑厨房水槽已否；§IV 缺件清单已写
  - verification_commands: `artifacts/eval/t4_sparse_ab_tb_dump.json`；TB `t_compat_sparse_v2_resume`；`docs/research/2026-08-17--t4-sparse-lightlp-completion.md`
  - success_definition: 回退续训不再是执行面，完整补全范围由论文+数据决定
- [x] 阶段 1：几何 / 观测 / 软地形合同
  - acceptance_criteria: 踏石边长与第一跨落在上文区间；中心站立 illegal≈0、偏 6 cm>0；新 `TEACHER_SPARSE_*_DIM` 有测试；`TEACHER_ACTOR_OBS_DIM==1155` 仍成立；`t4_loco_teacher_sparse_paper` 不再注册
  - verification_commands: `python -m pytest tests/test_t4_stepping_stone_contracts.py tests/test_t4_observation_contracts.py tests/test_t4_sparse_reward_contracts.py tests/test_t4_sparse_monitor_contract.py -q`
  - success_definition: implement 不能再「先大方砖再看」；illegal 沉默被测试钉死
- [x] 阶段 2：环境与奖励实现
  - acceptance_criteria: 软/硬地形可切换；稀疏 tile 步态/AMP/stumble=0；Actor 接触+scan×5；Critic 脚下 scan；足加速度与反向走已挂；无 `legal_foothold` / 双 Critic
  - verification_commands: 同上 pytest + 有 Isaac 时 64-env 3-iter smoke（nubot tmux）
  - success_definition: 新配方在代码里是一条任务，不是注释掉的开关堆
- [ ] 阶段 3：nubot 从零单阶段真洞开训（`t_sparse_lightlp_s4`）
  - acceptance_criteria: 不加载旧 ckpt；tmux 健康；两边 `reach_2m` 离开 0；`illegal` 明显负于 −0.001
  - verification_commands: nubot `tmux ls`；`logs/t4-sparse-lightlp-s4.log` + TB `Terrain/stepping_stones|raised_pillars/reach_2m_rate`、`Episode_Reward/illegal_footstep`
  - success_definition: 圆桩不再死在 1 m，踏石 illegal 开始干活
- [ ] 阶段 4：本 lineage 切真坑
  - acceptance_criteria: 加载阶段 3 的本 run ckpt；真坑打开；两边仍能走，不是立刻回到 progress≈1 m
  - verification_commands: 同一 TB 硬阶段曲线 + 回放
  - success_definition: 软地上学会的瞄点迁到真坑
- [ ] 阶段 5：行为验收（无学生）
  - acceptance_criteria: teacher 回放踏石/圆桩连续走；固定 evaluator JSON；不跑旧学生 FT
  - verification_commands: `play_t4_sparse_teacher_mujoco.py` / `eval_t4_hurdle.py --terrain stepping_stones|raised_pillars`（以当时入口为准）
  - success_definition: 能力来自回放和 JSON，不是 reward 曲线

## Commit units

1. `sparse-v4-contracts`：阶段 1 几何/维/测试。
2. `sparse-v4-mdp`：阶段 2 环境与奖励。
3. 训练产物不进 git。阶段 5 的 evaluator JSON 可进 `artifacts/eval/`。

提交前置：实现完成 + review 无 Critical + 对应 pytest 绿。

## Known risks / blockers

- 软地形「脚踩实、眼看洞」是实现主风险；先写合同再选双 prim 或布局代数。
- 收窄踏石 + 从零会暂时丢掉现行「大方砖蹦过去」的曲线，这是预期。
- Actor 1937D / scan×5 让旧 1155 学生无法蒸这条老师；本切片接受，以后另开学生。
- 接触进 Actor 在旧 T-paper 真坑上没救过圆桩；这里绑定软地形重做，不再当单独对照。
- 稀疏关步态后 Actor 里仍有 gait_phase 通道；先留着，不在本切片改本体 schema。
- nubot SSH 不稳；长命令必须 tmux。
- 翻箱并行，禁止改 zhuoqun checkout。

## Next skill

`implement`：阶段 1 合同（几何 + 新维 + illegal 偏置 + 删 paper 任务）。未测绿前不开训。

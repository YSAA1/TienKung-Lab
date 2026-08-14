# Executable Plan - T4 连续跨栏并入 Stage E 课程（terrain bucket 路线）

> Status: active
> Date: 2026-08-13（**2026-08-14 路线修订**：独立 skill lineage 废弃，改为 Stage E
> 课程地形 bucket；修订后的 Spec 见下）
> Spec: `docs/specs/2026-08-13--t4-hurdle-skill.md`（user-approved，含 2026-08-14 修订）
> Branch: `t4-train`
> Planning surface: docs plan
> 关系: 本计划并入 Stage E surface（`docs/plans/2026-08-12--t4-unified-depth-locomotion-plan.md`
> 的 teacher 训练线）；与 vault plan 并行但不再共享 zhuoqun 配额；未来 merge 候选
> 回到 loco + vault 两技能。

## Objective

把连续跨栏（100m 障碍赛障碍 #2）作为地形 bucket 并入 Stage E teacher 课程，
单一 1155D policy 同时掌握平地 / 乱石 / 楼梯 / 连续跨栏：

```text
hurdle-ring sub-terrain（环形细杆，间距/杆高随机，占比 0.10）
    + Stage E 现有 MDP 零新增（碰杆走既有接触/终止语义，AMP 按 difficulty 衰减）
    -> 新 lineage stage_e_prov7_hurdle 从零训练
       （nubot 2 卡 x 2048 env = 4096，25000 iter 预算，tmux）
    -> 训练侧观察 hurdle terrain_levels 上行 + 中期回放人工复核
    -> 最终 gate（后续工作项）：corridor evaluator 10 栏 ≥90% @500，零碰栏
```

## Active Slice

H2 已完成（2026-08-14 01:10 prov7 于 nubot 启动，起步健康）；下一片 H3
corridor evaluator + RED case，待开工指令。本机 `stage_e_prov6_local`（无跨栏）
继续运行作为对照组（2026-08-14 用户裁定，不停止）。

## Non-goals

- 独立跨栏 skill lineage、warm-start fine-tune、66D skill AMP clips（2026-08-14 废弃）。
- Unitree G1、其余 9 个障碍、赛道串联与折返重试状态机。
- depth 蒸馏与部署观测（归 depth slice）。
- 修改 1155D schema、plant、冻结部署合同；新增碰杆专用 MDP 项。
- 动态可倒杆进训练分布（仅 evaluator 报告项）。
- 不以 reward、loss、checkpoint 存在替代行为验收。

## Success Criteria

1. **训练侧观察项（不 gate）**：prov7 数值健康（无 NaN/OOM），hurdle bucket
   terrain_levels 持续上行，中期回放可见预抬腿过杆。
2. **Gate（归 H3/H4）**：固定 evaluator（10 栏 @1.1 m、高 0.3 m、杆厚 0.05 m、
   corridor 2.4 m、前向 0.7 m/s）成功率 ≥90% @500 trials；成功 = 顺序过全部栏 +
   全程零碰栏 + 不摔 + 到达出口；超时 / 冻结计失败。
3. **负例**：后退命令 0/100 进入栏区。
4. 证据三件套：evaluator JSON + 连续回放视频 + lineage manifest。

## Verification Path

```text
H0 布局真值纯 Python 合同（本机 pytest，已绿）
  -> H1 hurdle sub-terrain 接入 Stage E 地形 + 本机 64-env smoke（已交付）
  -> H2 prov7 nubot 双卡启动 + 稳定性监控（当前；prov6_local 留作对照组不停）
  -> H3 corridor evaluator + RED case（零策略完整 JSON 判失败；复用 vault V2 骨架）
  -> H4 gate 评估（≥90% @500 + 负例 + 报告项）+ 连续回放人工复核
```

所有训练、GPU probe、批量 playback、evaluator 一律 tmux。课程卡住 / 行为异常先
诊断根因（aliasing、AMP 衰减、终止压力），不以追加预算代替根因分析。

### Verification Path Status

`runnable`

H0/H1 本机已验证；H2 依赖 nubot 4 卡空闲（2026-08-14 已核实：prov5 tmux 已亡，
4 卡全空）+ git bundle 同步惯例；H3/H4 为计划内后续工作项。

## Required Capabilities

- nubot 2 张卡（CUDA 0,1）跑 prov7；其余 2 卡留空；本机 1 卡跑 prov6_local
  对照组（渲染 / 调试与其共存，显存允许时）；tmux。
- zhuoqun 不再为跨栏保留 2 卡配额（原 2026-08-13 的 2+2 裁定中跨栏侧取消；
  vault 是否回收 4 卡由 vault surface 裁定）。
- nubot 代码同步：GitHub fetch 不可靠时走 git bundle over SSH 惯例。
- corridor + ordered gates evaluator（H3）：复用 vault V2 骨架，后完成者复用。
- 人工复核：中期与 gate 回放视频需用户目视确认。

## Fallback Evidence

无可替代最终行为 gate 的 fallback。

- nubot 不可用时 fallback 本机单卡 4096 env（显存风险，OOM 降 2048；需用户裁定
  是否让出 prov6 对照组的卡）；lineage 语义不变。
- 课程在 hurdle 高难度行卡住：先渲染回放诊断（贴地犁杆 vs 不敢进杆区），再裁定
  课程 / AMP 旋钮调整；任何 MDP 合同变更 = 再开 lineage。

## Final Integration Claim

`final_integration_claim`: 单一 Stage E teacher（1155D，无 skill label）在固定
checkpoint 上通过跨栏 corridor gate（10 栏 ≥90% @500、零碰栏、后退负例 0/100），
同时楼梯 / 乱石能力不回归（Stage E 既有 evaluator 语义），三证齐全。

## 关键合同事实（实现输入）

- 布局真值：`legged_lab/terrains/hurdle_layout.py` 常量——tile 8×8 m、平台 1.6 m、
  border 0.25 m、间距 0.9–1.3 m（per-tile 随机，2–3 环）、杆高 0.05→0.35 m
  （difficulty 插值）、杆厚 0.07 m；AABB / 点判定函数供 evaluator 共用。
- 地形接入：`MeshHurdleRingsTerrainCfg`（`terrain_generator_cfg.py`），
  `T4_STAGE_E_TERRAINS_CFG` 占比 hurdles 0.10 / flat 0.08 / random_rough 0.16 /
  wave 0.06（其余不动，总和 1.0）。
- MDP：零新增项；碰杆 = 既有 stumble / shank / undesired contact / 姿态终止；
  AMP per-difficulty 衰减自动覆盖。
- lineage：`stage_e_prov7_hurdle`，从零，nubot 2 卡 × 2048 env，25000 iter 预算，
  torchrun `--distributed` 惯例（prov5 同款），日志
  `logs/t4-stage-e-teacher.stage_e_prov7_hurdle.log`。
- evaluator 固定值（H3/H4）：10 栏 @1.1 m / 0.3 m / 杆厚 0.05 m、corridor 2.4 m、
  前导 3 m + 出口 3 m、前向 0.7 m/s、500 trials。

## 工作项

- [x] H0：布局真值与碰杆判定纯 Python 合同（2026-08-14 完成）
  - scope: `legged_lab/terrains/hurdle_layout.py`（环半宽、difficulty→杆高、AABB、
    点判定，全常量真值）；`tests/test_t4_hurdle_contracts.py` 5 项合同。
  - verification_commands: `python -m pytest tests/test_t4_hurdle_contracts.py -q`
  - evidence: 5 passed；既有 24 项纯测试不回归。

- [x] H1：hurdle sub-terrain 接入 Stage E 地形（2026-08-14 完成）
  - scope: `MeshHurdleRingsTerrainCfg` + `hurdle_rings_terrain`（trimesh 环形杆，
    消费 H0 真值）；`T4_STAGE_E_TERRAINS_CFG` 占比重配（hurdles 0.10）。
  - acceptance_criteria: 本机 64-env 3-iter smoke 无异常（地形生成成功、无 NaN）。
  - verification_commands: 本机 tmux `t4-hurdle-smoke`，日志 `/tmp/t4_hurdle_smoke.log`。

- [x] H2：prov7 lineage 启动与稳定性监控（2026-08-14 完成）
  - scope: 提交代码（lineage 锚定 commit）→ git bundle 同步 nubot → tmux
    `t4-stage-e` 双卡启动 `stage_e_prov7_hurdle`（CUDA_VISIBLE_DEVICES=0,1，
    torchrun nproc=2，每进程 2048 env，25000 iter）→ 监控起步（显存 / steps/s /
    episode length / 无 NaN）→ `.harness/state.md` 记 lineage。本机
    `stage_e_prov6_local` 继续运行作对照组（2026-08-14 用户裁定，不停止）。
  - acceptance_criteria: prov7 跑过起步阶段无 OOM/NaN，日志与 TB 就位；prov6
    对照组不受影响。
  - verification_commands: `tmux ls`（nubot）；`nvidia-smi`；
    `tail -f logs/t4-stage-e-teacher.stage_e_prov7_hurdle.log`
  - success_definition: Stage E 主线切换到含跨栏课程的 prov7。
  - evidence（2026-08-14 01:14）: commit `e9f46fb` 同步 nubot（bundle ff）；
    iter ~100 时 iteration time ~2.27s、显存 ~7.2GB/卡、mean reward 3.6→4.1、
    episode length ~190→198 上行，无 NaN/OOM；prov6_local 对照组未受影响。

- [ ] H3：corridor evaluator + RED case
  - scope: corridor 2.4 m + ordered gates 跨栏评估场景（gates 与零碰杆判定消费
    `hurdle_layout` AABB 真值）；evaluator JSON 全字段（lineage/seed/checkpoint/
    成功判定/失败原因/碰杆与 hard-limit 计数）；负例（后退命令、超时冻结）；
    动态可倒杆报告场景（非 gate）；RED case：零策略产完整 JSON 且判失败。
    与 vault V2 共享 corridor/gates 骨架，先完成者建骨架，后者复用。
  - acceptance_criteria: 零策略成功率 0 且失败原因正确；跳 gate / 逆序判失败；
    碰杆计数进 JSON；schema 合同测试通过。
  - verification_commands: `python -m pytest tests/test_t4_hurdle_contracts.py -q`；
    evaluator RED 命令定型后回填。

- [ ] H4：gate 评估 + 人工复核
  - scope: prov7 固定 checkpoint 跑 H3 evaluator（≥90% @500 + 负例 + 报告项）+
    连续回放视频人工复核 + lineage manifest；同时抽查楼梯 / 乱石不回归。
  - acceptance_criteria: Success Criteria 2–4 全达成；三证齐全。
  - success_definition: `final_integration_claim` 成立。

## Commit Units

1. `feat(t4): 跨栏环形地形并入 Stage E 课程`——H0+H1 代码 + 合同测试 + Spec/Plan
   修订 + harness 同步（本 session，先于 prov7 启动提交，lineage 锚定该 commit）。
2. `chore(harness): prov7 lineage 启动记录`——H2 完成后 state 更新。
3. `feat(t4): 跨栏 corridor evaluator 与 RED case`——H3。
4. `docs(t4): 跨栏 gate 证据与收尾`——H4。

## Known Risks / Blockers

- 0.07 m 杆在 0.1 m scan 网格间歇 aliasing；缓解：proprio 历史 + 连续扫过 + 中期
  回放裁定；不达标再议 raycast proxy（schema 变更 = 新 lineage）。
- 高杆 + 姿态终止（-200）可能卡课程；观察 terrain_levels、渲染回放诊断。
- AMP 衰减不足压制高抬腿；旋钮已冻结，观察回放。
- nubot GitHub fetch 不可靠；bundle over SSH 惯例。
- 8×8 tile 只有 2–3 环，连续 10 栏节奏由 evaluator corridor 场景验收。
- 双 GPU NCCL / kit 多进程偶发初始化失败；重启 tmux 重试一次再诊断。

## Recovery Protocol

```bash
git status --short --branch
sed -n '1,60p' .harness/state.md
rg -n '^[-] \[[ x]\]' docs/plans/2026-08-13--t4-hurdle-skill-plan.md
tmux ls  # 本机
sshpass -p ' ' ssh nubot@100.100.188.39 'tmux ls; nvidia-smi --query-gpu=index,memory.used --format=csv'
```

训练阶段恢复时核对：当前 commit、nubot checkout commit、GPU ownership
（prov7=nubot CUDA 0,1）、tmux session、日志、最近 checkpoint；不得从旧对话推断
训练状态。

## Next Skill

`implement`

Reason: H2 active slice 命令面清楚（同步 → 启动 → 监控 → 切换），今晚完成。

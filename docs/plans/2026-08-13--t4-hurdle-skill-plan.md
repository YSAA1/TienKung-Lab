# Executable Plan - T4 连续跨栏 skill（reward-only heightscan 技能）

> Status: active
> Date: 2026-08-13
> Spec: `docs/specs/2026-08-13--t4-hurdle-skill.md`（user-approved）
> Branch: `t4-train`
> Planning surface: docs plan
> 关系: 与 `docs/plans/2026-08-13--t4-vault-loco-merge-plan.md`（vault，V0 active）并行；
> 共享 vault V3 zhuoqun preflight gate；zhuoqun 4 卡对半分（vault 2 卡 + 跨栏 2 卡）。

## Objective

在 Stage E teacher 1155D 合同上交付独立的 T4 连续跨栏技能策略（100m 障碍赛障碍 #2）：

```text
跨栏地形（栏数/间距/高度随机，strict zero-contact）
    + reward-only fine-tune（warm-start 自 prov5 行走 checkpoint，新 lineage）
    -> 固定 evaluator：10 栏 @1.1m/0.3m，corridor 2.4m，前向 0.7 m/s
       成功率 ≥90% @500 trials，零碰栏
    -> 66D skill AMP clips 入库（服务未来 merge 的 skill prior）
```

无参考动作、无 mimic 阶段；merge（2 或 3 技能）留给 vault plan V7 开工时裁定，
本计划只保证合同兼容（1155D、同 plant、clips 可提取）。

## Active Slice

H0 跨栏地形布局真值与碰栏判定纯 Python 合同：栏杆布局（数量/间距/高度/前导距离
随 difficulty 与 rng 的映射）做成不依赖 IsaacLab 的纯 Python 模块，供地形生成、
碰栏判定、evaluator gates 三处共用；合同测试本机可跑。零训练代码面。

## Non-goals

- Unitree G1、其余 9 个障碍、赛道串联与折返重试状态机。
- depth 蒸馏与部署观测（归 depth slice）。
- 修改 1155D schema、`legged_lab/assets/t4/schemas.py`、plant 或冻结部署合同。
- 跳跃专项奖励；动态可倒杆进训练分布（仅 evaluator 报告项）。
- merge 本身（G3/V7 属 vault plan）。
- 不以 reward、loss、checkpoint 存在替代行为验收。

## Success Criteria

1. **Gate**：固定 evaluator（10 栏 @1.1 m、高 0.3 m、宽 0.05 m、corridor 2.4 m、
   前导 3 m + 出口 3 m、前向 0.7 m/s）成功率 ≥90%，500 trials；成功 = 顺序过全部
   栏 + 全程零碰栏 + 不摔 + 到达出口；超时/冻结计失败。
2. **负例**：后退命令 0/100 进入栏区。
3. **报告项（不 gate）**：动态可倒杆 ~100 trials、0.35 m 高度、0.9 m 密间距、
   平均通过时间。
4. 证据三件套：evaluator JSON（lineage、seed、固定 checkpoint、bucket、成功判定、
   失败原因、碰栏与 hard-limit 计数）+ 连续回放视频 + lineage manifest。

## Verification Path

```text
H0 布局真值与碰栏判定纯 Python 合同（本机 pytest）
  -> H1 任务注册 + scan 集成 + 2-env smoke（nubot tmux，短占用）
  -> H2 corridor evaluator + RED case（零策略完整 JSON 判失败）
  -> H3 zhuoqun preflight（共享 vault V3）+ prov5 checkpoint 搬运
  -> H4 probe（首杆碰撞率/预抬腿/数值健康）-> scan aliasing 裁定
     -> 正式 lineage（zhuoqun 2 卡，≤15000 iter）-> gate + clips 入库
```

所有训练、GPU probe、批量 playback、evaluator 一律 tmux。probe 不达标先裁定
aliasing 缓解方案（切 raycast proxy = privilege schema 变更 = 新 lineage），
不以追加预算代替根因分析。

### Verification Path Status

`runnable`

H0 本机即可跑；H1 smoke 在 nubot 已验证的 runtime 短暂执行（不占训练卡时长跑）；
H3 是计划内显式 gate（zhuoqun 部署工期不确定，与 vault 共享）；H4 依赖 `prov5`
稳定行走 checkpoint（计划内 stop gate，非外部阻塞）。

## Required Capabilities

- zhuoqun 4 卡对半分：vault 用 2 卡、跨栏用 2 卡（2026-08-13 用户裁定；建议固定
  `CUDA_VISIBLE_DEVICES`：vault=0,1、跨栏=2,3，写进各自训练脚本）；nubot 4 卡
  Stage E 专属（仅借短 smoke）；本机 1 卡渲染/调试；tmux。
- 跨栏地形生成（IsaacLab sub-terrain 或独立静态 prim）+ 栏杆进 scan + 碰栏计数
  （实现方式 H1 裁定）。
- corridor + ordered gates evaluator：语义沿用 vault V2；先完成者建骨架，后者复用。
- nubot→zhuoqun checkpoint 搬运（bundle/scp 惯例）。
- 人工复核：回放视频行为裁定需用户目视确认。

## Fallback Evidence

无可替代最终行为 gate 的 fallback。

- zhuoqun 部署未就绪时，H0–H2 合同与 evaluator 工作先行；H4 可临时用 nubot 空闲
  窗口做小规模 probe，但正式 lineage 与 gate 证据必须在 zhuoqun 2 卡 runtime 产出
  （2026-08-13 用户裁定的 2+2 分配）。
- warm-start 行为不良（不抬腿贴地走且 probe 无改善信号）时 fallback 从零训练，
  开新 lineage 并记录裁定。

## Final Integration Claim

`final_integration_claim`: 单一 1155D heightscan 跨栏技能策略（Stage E teacher actor
合同，无 reference / skill label）在固定 checkpoint 上通过跨栏 gate（10 栏 ≥90%
@500 trials、零碰栏、后退负例 0/100），三证齐全；66D skill AMP clips 已入库并可被
未来 merge 消费。

## 关键合同事实（实现输入）

- 规则几何：杆长 2.4 m、宽 0.05 m、高 0.3 m、间距 1.1 m（歧义 #7：栏数未写明、
  「碰倒」未定义 → strict zero-contact 保守解释，全部参数可配置）。
- 观测合同：1155D（proprio 96x10 + 前向 195D scan + 命令 + prev action），scan
  0.1 m 网格 15x13、前向 0.2–1.6 m；MLP 512/256/128；27D action，scale 0.25。
- plant 真值：`legged_lab/assets/t4/t4.py` 现值；prov5 再改 plant/MDP 则跨栏侧
  作废重训。
- warm-start：加载 policy/critic 权重，optimizer 与 AMP 判别器重置，新 lineage；
  partial load 报告 loaded/skipped keys。
- AMP：沿用 walk AMP + per-difficulty 衰减旋钮，跨栏地形映射高 difficulty 使系数
  ≈0；gait clock 与 symmetry mirror loss 保持；开训前冻结进配置。
- 训练随机化：栏数 3–8、间距 0.9–1.4 m、高度随 difficulty 0.10→0.35 m、宽 0.05 m
  固定、前导 1–3 m；命令前向 0.4–1.0 m/s + yaw ±0.3 rad/s，不采样侧移/后退。
- reset：loco 一致站姿，无 RSI。
- evaluator 固定值：10 栏 @1.1 m/0.3 m/0.05 m、corridor 2.4 m、前导 3 m + 出口
  3 m、前向 0.7 m/s、500 trials。

## 工作项

- [ ] H0：跨栏地形布局真值与碰栏判定纯 Python 合同（当前）
  - scope: 新增纯 Python 布局模块（建议 `legged_lab/terrains/hurdle_layout.py`）：
    给定 cfg（栏数/间距/高度/宽度/前导范围）+ difficulty + rng → 杆位真值表
    （AABB 列表）；碰栏判定纯函数（接触点 vs 真值表）；difficulty→高度映射单调；
    合同测试进 `tests/test_t4_hurdle_contracts.py`（布局边界、全宽贴 corridor、
    判定函数正反例、evaluator gate 序生成）。不 import IsaacLab。
  - acceptance_criteria: 布局真值表数值正确（间距/数量/高度界内）；碰栏判定对
    构造正反例分类正确；difficulty 单调映射有测试；既有合同测试不回归。
  - verification_commands: `python -m pytest tests/test_t4_hurdle_contracts.py -q`;
    `python -m pytest tests/test_t4_asset_migration.py tests/test_t4_observation_contracts.py -q`
  - success_definition: 地形生成、碰栏判定、evaluator gates 三处消费同一布局真值，
    数据合同本机可复核。

- [ ] H1：跨栏任务注册 + scan 集成 + 2-env smoke
  - scope: 注册任务（建议名 `t4_hurdle_skill`）：跨栏 sub-terrain（消费 H0 布局）、
    栏杆进 ray caster（mesh 烘焙 vs 静态 prim + raycast 列表在此裁定并记录）、
    碰栏计数接 contact 判定、碰栏惩罚/终止 + clearance/task 奖励、loco 站姿 reset、
    AMP difficulty 映射与命令采样按合同事实配置；1155D 合同测试。
  - acceptance_criteria: 任务可被 train 入口解析；观测/动作维度合同测试通过；
    nubot 2-env 2-iter smoke 无 NaN/Inf/OOM，smoke 中栏杆位置的 scan 值非平地、
    人为踩杆能触发碰栏计数。
  - verification_commands: `python -m pytest tests/test_t4_hurdle_contracts.py -q`;
    `tmux new-session -d -s t4-hurdle-smoke 'cd <nubot-checkout> && bash scripts/nubot_run.sh legged_lab/scripts/train.py --task=t4_hurdle_skill --headless --num_envs=2 --max_iterations=2 2>&1 | tee /tmp/t4-hurdle-smoke.log'`
  - success_definition: 训练环境端到端可跑，栏杆对策略可见、碰栏对判定可见。

- [ ] H2：corridor evaluator + RED case
  - scope: corridor 2.4 m + ordered gates 跨栏评估场景（gates 由 H0 布局真值生成）；
    evaluator JSON 全字段（lineage/seed/checkpoint/bucket/成功判定/失败原因/碰栏与
    hard-limit 计数）；负例场景（后退命令、超时冻结）；动态可倒杆报告场景（轻质
    自由刚体，非 gate）；RED case：零策略产完整 JSON 且判失败、失败原因分类正确。
    与 vault V2 共享 corridor/gates 骨架，先完成者建骨架，后者复用。
  - acceptance_criteria: 零策略成功率 0 且失败原因正确；跳 gate/逆序判失败；
    碰栏计数进 JSON；schema 合同测试通过。
  - verification_commands: `python -m pytest tests/test_t4_hurdle_contracts.py -q`;
    `tmux new-session -d -s t4-hurdle-red 'cd <nubot-checkout> && bash scripts/nubot_run.sh legged_lab/scripts/eval_t4_hurdle.py --policy zero --output artifacts/eval/t4_hurdle_red.json 2>&1 | tee /tmp/t4-hurdle-red.log'`
  - success_definition: gate 证据机器可产出、失败可分类，评估语义先于训练冻结。

- [ ] H3：zhuoqun preflight（共享 vault V3）+ checkpoint 搬运
  - scope: 跨栏 1-env spawn 加入 vault V3 的 preflight 清单（不重复建 preflight）；
    `prov5` 稳定行走 checkpoint 从 nubot 搬运到 zhuoqun（bundle/scp），记录 SHA 与
    来源 iter；GPU 划分（vault=0,1 / 跨栏=2,3）写进训练脚本。
  - acceptance_criteria: preflight 在 zhuoqun 全 PASS 且含 `t4_hurdle_skill` 1-env
    spawn；checkpoint SHA 与来源记录进 `.harness/state.md`。
  - verification_commands: 沿用 vault V3 preflight 命令（tmux）；
    `ssh zhuoqun@100.95.109.48 'sha256sum <checkpoint-path>'`
  - success_definition: 跨栏训练资源与 warm-start 输入就绪。
  - blocker 备注: zhuoqun 当前无 IsaacLab；部署分支与 vault V3 完全共享。

- [ ] H4：probe + 正式训练 + gate + skill AMP clips 入库
  - scope: 短 probe（200–500 iter，2 卡）看数值健康 + 首杆碰撞率 + 预抬腿信号 →
    scan aliasing 裁定（不达标切 raycast proxy = schema 变更 = 新 lineage）→
    正式 lineage（≤15000 iter，zhuoqun 2 卡，tmux）→ 固定 evaluator gate（≥90%
    @500）+ 负例 + 报告项 → 成功 rollout 提取 66D skill AMP clips 入库（新目录 +
    manifest）→ 连续回放视频人工复核。
  - acceptance_criteria: Success Criteria 1–4 全达成；clips 66D 形状与 manifest
    合同测试通过；nonfinite/hard-limit/碰栏计数为零。
  - verification_commands: `tmux new-session -d -s t4-hurdle-train 'ssh zhuoqun ... CUDA_VISIBLE_DEVICES=2,3 bash scripts/train_t4_hurdle.sh 2>&1 | tee logs/t4-hurdle-train.log'`;
    evaluator 命令由 H2 产物固定；`python -m pytest tests/test_t4_hurdle_contracts.py -q`
  - success_definition: `final_integration_claim` 成立，clips 可被未来 merge 消费。

## Commit Units

每个 commit unit 在对应工作项实现完成、review 无 Critical、verify PASS 后提交：

1. `docs(t4): 跨栏 skill Spec 与计划`——Spec + 本计划 + harness 同步（本 session）。
2. `feat(t4): 跨栏地形布局与碰栏判定合同`——H0。
3. `feat(t4): 注册跨栏技能任务`——H1。
4. `feat(t4): 跨栏 corridor evaluator 与 RED case`——H2。
5. `chore(t4): 跨栏 preflight 与 checkpoint 搬运记录`——H3（与 vault V3 提交协同，
   避免重复改动）。
6. `train(t4): 跨栏技能 lineage 与 skill AMP clips`——H4（大 checkpoint 是否入库按
   仓库 artifact 规范，不默认提交）。

## Known Risks / Blockers

- 0.05 m 杆在 0.1 m scan 网格上 aliasing；缓解：probe 双信号裁定 + raycast proxy
  fallback（schema 变更 + 新 lineage）。
- warm-start walk 先验过强不抬腿；fallback 从零训练。
- walk AMP 衰减不足压制高抬腿；AMP 旋钮开训前冻结，probe 观察。
- `prov5` 失败再改 plant/MDP ⇒ 跨栏侧作废重训（用户已接受）。
- zhuoqun 无 IsaacLab，部署工期不确定（与 vault V3 完全共享）。
- 2 卡吞吐减半墙钟拉长；同机双 lineage 需 GPU 划分与 tmux 纪律，防互相挤显存。
- 碰栏计数实现（mesh 烘焙 vs 静态 prim）各有陷阱：烘焙需几何判定兜底、prim 需
  raycast 列表支持；H1 裁定并记录。

## Recovery Protocol

```bash
git status --short --branch
sed -n '1,60p' .harness/state.md
rg -n '^[-] \[[ x]\]' docs/plans/2026-08-13--t4-hurdle-skill-plan.md
tmux ls
```

训练阶段恢复时还需核对：当前 commit、绝对 checkout、GPU ownership（跨栏=2,3）、
tmux session、日志、最近 checkpoint、warm-start 来源 SHA、evaluator version；
不得从旧对话推断训练状态。

## Next Skill

`implement`

Reason: H0 active slice、文件面与验证路径清楚且本机可验证；H1 起需要 nubot/zhuoqun
runtime 时再按各自 gate 推进。

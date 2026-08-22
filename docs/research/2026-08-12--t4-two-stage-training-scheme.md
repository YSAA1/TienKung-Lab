# Research: T4 最多两阶段正式训练方案

> **非权威。** 走跑阶段已按老师→学生两条正式训练交付。入口：`docs/README.md`。
> Date: 2026-08-12
> Question: 当前 Plan 的五段正式训练（walk → jog → rough → stairs → route）是否过度设计？在本框架约束下，最多两次正式训练的可执行方案是什么？
> Status: 历史 findings

## 1. Verdict

**最多两次正式 PPO+AMP 训练是合理默认，且与仓库既有意图一致。**

五段正式 lineage 把「能力累积顺序」误写成了「五次独立长训」。能力顺序仍应保留；正式训练次数应压成：

1. **Stage A — depth-aware 基础 loco**（含 jog 可选并入）
2. **Stage B — 强 rough + 上下楼梯**（同 Actor 合同 resume）
3. **Route — 默认不做第三次长训**；用 route manager → local velocity command 做固定验收；仅在回归失败时才考虑短 fine-tune

这与 `PROJECT_CONTEXT.md` §7 的两阶段骨架一致，也满足 Spec「不一次性联合训练全部能力」的硬约束，同时砍掉 Plan 中 T2/T5 作为独立正式阶段的流程税。

## 2. Primary sources consulted

### 本仓库（最高优先）

| Source | What it establishes |
|--------|---------------------|
| `PROJECT_CONTEXT.md` §7 | 已写明两阶段：①统一 depth-aware 基础 loco；② rough + 上下楼梯；route 是防绕障合同，不是第三段 PPO |
| `docs/specs/2026-08-12--t4-unified-depth-locomotion.md` §累积式训练课程 | 能力顺序 1–5 是累积扩展，不是必须五次独立训练；拒绝一次性联训；拒绝 HeightScan teacher→student 默认路线 |
| `docs/archive/plans/2026-08-12--t4-unified-depth-locomotion-plan.md` §Stage T1–T5 | 把课程拆成五段正式预算（50k/30k/40k/50k/30k）与五次 Gate |
| `legged_lab/envs/tienkung/walk_cfg.py` | 现有 walk：`GRAVEL_TERRAINS_CFG`，`max_iterations=50000`，`num_steps_per_env=24`，`resume=False`，AMP walk expert |
| `legged_lab/envs/tienkung/run_cfg.py` | 现有 run：同结构另一任务/另一 AMP 文件，`max_iterations=50000` — 历史是 **walk/run 两套 policy**，不是五段课程 |
| `legged_lab/terrains/terrain_generator_cfg.py` | `GRAVEL_TERRAINS_CFG`（轻 rough，`curriculum=False`）；`ROUGH_TERRAINS_CFG`（stairs+boxes+rough+wave+pit，`curriculum=True`） |
| `legged_lab/scripts/train.py` | 已支持 `agent_cfg.resume` → `runner.load(resume_path)` |

### 外部主源（训练分期惯例）

注意：文献里的 “two-stage” 常指 **特权 HeightScan teacher → depth student 蒸馏**，与本 Spec 拒绝的默认路线不同。下面只借用「正式训练次数少、课程在环境内混合」的证据，不建议改走 HeightScan distill。

| Source | Relevant claim |
|--------|----------------|
| [PRIOR, arXiv:2603.18979](https://arxiv.org/abs/2603.18979) / [project](https://prior-iros2026.github.io/) | Isaac Lab 上强调避免 multi-stage 流水线；用 **adaptive terrain curriculum** 在同一训练中同时覆盖 Plane / Boxes / Pyramid Stairs / Inverted Stairs |
| [Extreme Parkour, arXiv:2309.14341](https://arxiv.org/abs/2309.14341) | 典型两阶段是 privileged RL → depth distill；地形难度用 **行进距离晋级/降级** 在单阶段内调度，而不是五次独立 PPO lineage |
| [Now You See That, arXiv:2602.06382](https://arxiv.org/abs/2602.06382) | 显式两阶段：特权多地形 RL → depth 蒸馏；地形类别在 **同一特权阶段内分区 reward**，不是每地形一次完整训练 |
| [RPL, arXiv:2602.03002](https://arxiv.org/abs/2602.03002) | 两阶段 = 多地形专家 → 多深度相机 transformer 蒸馏 |
| [ANYmal Parkour, arXiv:2306.14874](https://arxiv.org/abs/2306.14874) | 导航与 locomotion **分层**：loco 训完冻结，上层只发局部命令 — 支持「route 不是第三次 loco 长训」 |

## 3. Framework evidence: what already fits two stages

1. **原框架事实上就是 ~2 次训练心智模型**：`walk_cfg` / `run_cfg` 各 `50k` iterations，不是五段课程。统一 Actor 后，应把 run 能力并入同一合同的课程，而不是再拆成 T2 独立正式阶段。
2. **地形已按「轻 / 重」二分**：`GRAVEL_TERRAINS_CFG` ↔ Stage A；`ROUGH_TERRAINS_CFG`（含 stairs）↔ Stage B。IsaacLab 原生 `curriculum=True` 可在 **一次训练内** 抬难度。
3. **Resume 已存在**：Stage B 从 Stage A checkpoint 加载，不必新写训练基础设施。
4. **Spec 的 route 边界**已是「manager → local velocity command → Actor」；ANYmal Parkour 同类分层说明上层不应默认再开一轮全量 PPO。
5. **Spec 禁止的是「一次联训全部」与「HeightScan 默认蒸馏」**，不是禁止把 jog 并入 A、把 rough+stairs 并入 B。

## 4. Spec vs Plan: what must stay vs what to cut

### Must stay（合同与证据）

- 单一导出 Actor：`depth history + proprio history + local velocity command + previous action → 27DoF`
- Actor 无 HeightScan / 地形真值 / 接触真值
- 66D AMP；expert/runtime 同 builder
- 固定 evaluator + 连续回放；IsaacLab 与 MuJoCo 同 checkpoint
- 能力累积顺序与旧 bucket 回归
- 训练前 Gate（M0–M3 合同闭环、数值/容量 probe）

### Cut / demote（流程，非能力）

| Plan item | Action |
|-----------|--------|
| T2 独立正式 jog 训练 | **并入 Stage A**（motion 审核通过后提高 `vx` 与加入 jog AMP）；仅当 gait/AMP 冲突可复现时再拆 |
| T3 与 T4 两次正式训练 | **合并为 Stage B**（ROUGH curriculum + stairs traversal MDP） |
| T5 正式 route 长训 | **降为验收**；失败才短 fine-tune |
| 每阶段都跑满独立预算上限 | 改为两次预算上限；阶段内用 probe + 定期 evaluator，而非五次完整 Gate 仪式 |

能力列表仍可按 Spec 的 1→5 做 **evaluator buckets**，但 buckets ≠ 正式训练次数。

## 5. Recommended two-stage scheme

### 0. 训练前（不算正式训练）

与现 Plan M0–M3 / 健康 probe 相同，不可省：

- T4 spawn / joint / camera / 18 motion playback 审计
- 66D AMP expert/runtime parity
- depth preprocessing + Actor export 合同
- 固定 evaluator RED cases + MuJoCo depth golden
- 1-env smoke、128-env 数值 probe、depth 容量 probe
- 每阶段正式开训前 `200–500` iter 因果 probe

### Stage A — 基础 depth-aware loco（第 1 次正式训练）

| Item | Choice |
|------|--------|
| 初始化 | fresh policy（不从旧 20DoF / HIW checkpoint 完整 resume） |
| 地形 | `GRAVEL` 类：flat 为主 + 低幅 rough；**不开完整楼梯状态机** |
| Command | stand + forward/backward/lateral/turn/curved；范围先用框架候选 `vx∈[-0.6,1.0]`，baseline 后冻结 |
| Jog | **默认并入**：审核通过的 jog motion + 高速 forward / ramp episodes；`t4_run` 仍 hold out 至独立审核 |
| AMP | stand/walk/backward/lateral/turn（+ 通过审核的 jog）；类别权重显式配置 |
| Depth | 最终冻结 preprocessing；真实 depth，无零占位；含 hold/delay/noise/dropout 轻量覆盖 |
| 预算上限 | **50,000** iterations（对齐 `walk_cfg.max_iterations`）；env 数由容量 probe 决定，不预设 4096 |
| 晋级 Gate | 全部基础（含 jog，若已纳入）buckets 达冻结阈值；无 command collapse / hard violation；同导出 policy 过 MuJoCo；回放无持续滑步 / 走跑跳变 |

**失败分流（仍只算诊断，不自动加阶段）**

- 站不稳 → 控制/动作合同
- 能站不走 → reward / command
- 跟踪好但滑步 → gait / AMP
- IsaacLab 好 MuJoCo 坏 → obs/depth/dynamics parity

### Stage B — 强 rough + 上下楼梯（第 2 次正式训练）

| Item | Choice |
|------|--------|
| 起点 | Stage A 通过 checkpoint；Actor/obs/action/depth schema **不变** |
| 地形 | 基于 `ROUGH_TERRAINS_CFG` 思路：`curriculum=True`，混入强 rough、boxes、wave、斜坡；**楼梯作为专用 traversal 信号**，不靠 rough reward 冒充 |
| 任务 | 同 episode 覆盖上楼与下楼；低台阶 → 多阶上 → 多阶下 → 同 episode 上下；成功后结束或进下一段，不在终点刷奖 |
| 回归 | 持续采样 Stage A command/terrain buckets |
| Depth | 加强 delay/dropout/noise curriculum；正式 evaluator 用冻结扰动档 |
| 预算上限 | **追加 60,000–80,000** iterations（覆盖原 Plan T3+T4 的 40k+50k 量级，但不强制跑满） |
| 晋级 Gate | rough buckets + 上楼/下楼分别通过；提前抬脚而非撞击补偿；depth 消融显示感知依赖；全部旧 buckets 回归；MuJoCo rough/stairs 通过 |

若连续两个完整 Gate checkpoint 无关键行为改善 → `diagnose`，不加第三段正式预算硬顶。

### Route — 验收层（默认非正式训练）

```text
route/gate manager -> local velocity command -> Stage B Actor
```

- 合同：corridor + ordered gates + 越界失败 + strict success（与 Spec / `PROJECT_CONTEXT` 一致）
- 默认：用 Stage B 最终 checkpoint 跑 IsaacLab + MuJoCo route evaluator
- **仅当** loco 桶都过但 route 系统性失败（例如转弯接楼梯后 command 跟踪崩）时，才允许 **短 fine-tune**（建议 ≤10k–15k iter，新 lineage 标注），仍不算第三套能力课程

## 6. Budget comparison

| Scheme | Formal trains | Budget ceiling (iter) |
|--------|---------------|------------------------|
| Current Plan T1–T5 | 5 | 50+30+40+50+30 = **200k** |
| This scheme A+B | **2** | 50 + 60–80 = **110–130k**（+ 可选 route ≤15k） |

预算仍是上限，不是成功标准。

## 7. Mapping Plan stages → this scheme

```text
Plan T1 walk          ─┐
Plan T2 jog           ─┴→  Stage A (1 formal train)

Plan T3 rough         ─┐
Plan T4 stairs        ─┴→  Stage B (1 formal train)

Plan T5 route         ──→  Evaluator / optional short fine-tune
```

工作项可相应收成：M4=Stage A，M5=Stage B，M6=route 验收（原 M5–M8 合并）。

## 8. When to split back (>2 formal trains)

只在 **可复现失败** 后拆，不作为开局默认：

1. Stage A 中 jog 导致 walk 退化或 AMP 模式崩溃 → 拆出 jog 续训
2. Stage B 中 rough 过、stairs 不过（或相反）→ 临时把 stairs 从 rough 课程剥离
3. 出现可复现的多风格 AMP 冲突 → 再讨论 conditioning / 第二 discriminator（Spec 已限制）
4. 用户明确改走 HeightScan teacher→student（当前 Spec 拒绝）

## 9. Practical runbook (commands shape)

训练前 Gate 通过后：

```bash
# Stage A
tmux new-session -d -s t4-stage-a \
  'cd <ABS_CHECKOUT> && bash scripts/train_t4_loco_stage.sh stage_a 2>&1 | tee /tmp/t4-stage-a.log'

# Stage A Gate
python legged_lab/scripts/eval_t4_loco.py --stage stage_a --checkpoint <A> --simulators isaaclab mujoco

# Stage B
tmux new-session -d -s t4-stage-b \
  'cd <ABS_CHECKOUT> && bash scripts/train_t4_loco_stage.sh stage_b --resume <A> 2>&1 | tee /tmp/t4-stage-b.log'

# Stage B + route acceptance
python legged_lab/scripts/eval_t4_loco.py --stage stage_b --checkpoint <B> --depth-ablation all
python legged_lab/scripts/eval_t4_loco.py --stage route --checkpoint <B> --simulators isaaclab mujoco
```

（脚本名按实现时落地；此处表达阶段边界，不要求现已存在。）

## 10. Recommendation to plan owners

1. **采纳本两阶段方案作为训练主路径**；保留 Spec 的能力顺序作为 evaluator 清单。
2. 修订 Plan：用 Stage A/B 替换 T1–T5 正式训练叙述；T2/T5 降为「并入 / 验收」。
3. 不修改 Actor 合同、AMP 66D、depth-only、固定 evaluator 等硬边界。
4. 若需改 Spec 措辞：把「五个能力阶段」明确为 **curriculum milestones**，并声明 **formal training runs ≤ 2**。

## 11. Citations (compact)

1. `PROJECT_CONTEXT.md` §7 — 两阶段任务设计原文  
2. `docs/specs/2026-08-12--t4-unified-depth-locomotion.md` — 累积课程、拒绝联训、拒绝 HeightScan 默认蒸馏、route 边界  
3. `docs/archive/plans/2026-08-12--t4-unified-depth-locomotion-plan.md` — T1–T5 正式预算   
4. `legged_lab/envs/tienkung/walk_cfg.py` / `run_cfg.py` — 50k × 2 历史训练粒度  
5. `legged_lab/terrains/terrain_generator_cfg.py` — GRAVEL vs ROUGH(+stairs) curriculum  
6. `legged_lab/scripts/train.py` — resume 加载  
7. PRIOR arXiv:2603.18979 — 单策略 + 多地形 adaptive curriculum（Isaac Lab）  
8. Extreme Parkour arXiv:2309.14341 — 阶段少、难度在环境课程内调度；第二阶段多为感知蒸馏  
9. ANYmal Parkour arXiv:2306.14874 — 导航发局部命令、loco 可冻结  

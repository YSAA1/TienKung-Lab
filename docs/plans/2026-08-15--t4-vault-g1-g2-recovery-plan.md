# Executable Plan - T4 翻箱 G1 专家补齐与 G2 蒸馏重做

> Status: active
> Date: 2026-08-15
> Updated: 2026-08-18 — 本文件仍是翻箱执行面。G2 学生与 G3 技能合并（和走跑 / 梅花桩并成一条策略）要等 G2 过箱之后另开切片，不在本文件里提前写第二套配方。
> Spec: `docs/specs/2026-08-13--t4-vault-loco-merge.md`（已批准；本计划不改 Spec 三段，只修正执行顺序和蒸馏配方）
> Branch: `t4-train`
> Planning surface: docs plan（本文件）；旧合并 plan 已归档到 `docs/archive/plans/2026-08-13--t4-vault-loco-merge-plan.md`
> 关系: 恢复 slice。不取代合并 Spec；暂停该 plan 里「直接进 V7」的裁定。入口：`docs/README.md`。

## Objective

先做出一个**能从第 0 帧翻过 1m 箱**的 G1 跟踪专家，再按「教师开车 → 学生跟车 → DAgger(+PPO)」重做 G2。不再续训已经平台化的 `t4_vault_skill/2026-08-14_05-30-51_g2_from_g1_m28500`。

## Active Slice

R3 正式 5000 仍在 `t4_vault_g2_r3`。R4 学生开车 DAgger + 回报 PPO-clip 已接线，`t4_vault_g2_r4` 等待 `model_4999.pt` 后自动开训。

## 已裁定（用户让本计划自行决定）

1. **不续 G2，不开 G3。** G3 仍被两道硬门挡住：可用的 G2 skill 策略，以及 Stage E teacher evaluator。
2. **G1 先于 G2。** 现专家 48/100：16 次第 0 步 `wrist_body_pos`（评测 reset 伪失败），36 次走到箱前 `time_out`（真实起势缺口）。蒸馏 48% 的跟踪教师没有意义。
3. **训练推进门槛 ≠ 能力声明门槛。** 开 G2 只需 G1 100-trial 成功率 ≥80%，且失败不再以「箱前站死」为主；Spec 的 ≥95% 仍是对外宣称 G1 能力时的 gate。用户此前已放松严格评估，本计划遵守该裁定，但不接受「只看 TB」。
4. **G1 训练先便宜后重来。** 先从 `model_28500.pt` 改 reset 分布续训（奖励/观测不变）。2–5k iter 后箱前超时若仍 ≥25%，作废该续训，开新 lineage。
5. **G2 必须先过「教师开车」信息瓶颈实验。** 学生 MLP 跟不上教师自己滚出来的轨迹，就不要上 DAgger，更不要怪 iteration。
6. **算力。** zhuoqun 做 G1/G2；nubot Stage E 与本机 loco 对照不动。长任务一律 tmux。

## Non-goals

- 不续 `model_9999.pt` 同一套 MSE+学生自己滚。
- 不写 V7 G3 合并环境，不把 G1.pt 当 1155D expert。
- 不改 Stage E plant、1155D actor 合同、`schemas.py`。
- 不把 motion 相位塞进学生 gait 槽（那是参考泄漏，G3 接不上）。
- 不做箱宽/深随机泛化，直到 G2 在固定 1m 箱上能过箱。
- 不把 MuJoCo sim2sim / 本机 Isaac Docker 当作本 slice 的完成条件（并行收尾，不挡 G1）。
- 不以 reward、loss、ckpt 存在替代过箱行为。

## Success Criteria

1. **R1**：`t4_vault_mimic_eval` 在零策略/固定 ckpt 下不再出现「策略未动作就 `wrist_body_pos`」；`model_28500` 复评 JSON 里 step-0 wrist 失败 = 0。箱前超时次数作为 R2 基线写进 JSON。
2. **R2**：某一固定 G1 ckpt 在同一 evaluator 上 ≥80/100 成功（ordered gates + 持续过箱 + 稳定落地 + 无 hard violation）；连续回放能从第 0 帧翻完。这是开 G2 的唯一训练门槛。
3. **R3**：教师开车的 G2 BC 上，behavior loss 明显低于旧平台 0.27（目标量级 ≤0.05），episode 长度接近 G1（≥400/500），body error 进入 0.1 m 量级。过不了则判定 1155D+MLP 不够表达该技能，停下来改观测/结构，不上 DAgger。
4. **R4**：学生自己滚的 DAgger(+PPO) 在固定 1m 箱上能完整过箱（回放 + evaluator JSON）。此时才允许谈泛化。
5. 每个训练门槛：evaluator JSON + 连续回放 + lineage（ckpt 路径、teacher ckpt、git、采样策略）。不宣称 Spec 的 90%/95%，除非真的跑到那个数。

## Verification Path

```text
R1 评测 reset 对齐（本机合同测试 + zhuoqun 100-trial 复评 m28500）
  -> R2 G1 起点偏向续训 probe（zhuoqun tmux，2–5k iter）-> 同 evaluator
       失败则新 lineage，成功则续到 ≥80/100
  -> R3 改 Distillation：教师动作进环境；G2 去 push / 冻结假步态钟
       合同测试 -> zhuoqun 短 probe -> 正式 BC
  -> R4 学生 on-policy DAgger + PPO（仍固定箱；可保留轻度 RSI）
  -> （本计划外）丢 RSI + 箱后奖励 + 箱尺寸随机 = 原 V6 后半
  -> （硬门）Stage E evaluator + 可用 G2 之后才允许 V7
```

### Verification Path Status

`runnable`

- 本机：`pytest` 合同（资产 / skill / evaluator）不依赖 Isaac。
- zhuoqun：`scripts/zhuoqun_run.sh` + `t4-isaac-jammy:v2` 已跑过 G1 训练和 100-trial eval。
- 入口已存在：`train_t4_vault_mimic.py`、`eval_t4_vault.py`、`train_t4_vault_skill.py`。
- 本机 Isaac Docker / Sim zip 未齐，不挡本路径。

## Required Capabilities

- zhuoqun 至少 1 张 4090（建议 CUDA 0，2048–4096 env），tmux。
- 现成 ckpt：`logs/t4_vault_mimic/2026-08-13_09-30-13/model_28500.pt`。
- 冻结参考：`overbox_1m_t4_mjcf_fps50.npz`（SHA256 `67dde158…130e0a0`）。
- nubot / 本机 Stage E 进程保持不动。

## Fallback Evidence

无。过箱能力只能用 evaluator JSON + 回放证明。TB 只用于决定停不停训，不能当 gate。

若 zhuoqun 暂不可用：R1 的纯 Python 合同仍可在本机合入；Isaac 复评和训练排队，不得用本机 conda IsaacLab 2.3 结果替代 zhuoqun 2.1 lineage。

## Final Integration Claim

`final_integration_claim`: 存在一个固定 G1 checkpoint，在 `t4_vault_mimic_eval` 上从第 0 帧成功率 ≥80/100，并有回放；以及一个按教师开车 BC 验证过信息通道后、再经 DAgger(+PPO) 训练的 G2 1155D 策略，在固定 1m 箱上能完整过箱（JSON + 回放 + lineage）。不声称 Spec G2 90%@500 泛化，也不声称 G3。

## 诊断结论（本计划前提，已核实）

| 层 | 事实 | 含义 |
| --- | --- | --- |
| G1 训练 | body error ~0.05 m，长度 ~480/500 | 在 RSI + 150D 参考下能跟踪 |
| G1 评测（旧） | 48/100，16+36 看起来像手腕/箱前死 | 假象：16 env 一批且批次不独立，成绩按 0/16/0/16 交替 |
| G1 评测（R1） | `t4_vault_g1_m28500_resetfix.json` **100/100**，step-0 wrist = 0 | 每批 reset + 丢弃的零动作 step + 再 reset 后，从第 0 帧稳定过箱 |
| G2 实现 | `Distillation` 只做 MSE；`act()` 是学生采样 | 计划写 DAgger(+PPO)，落地缺 PPO，且从一开始学生开车 |
| G2 观测 | 写死 0.8 m/s + 0.85 s 步行钟 + scan | 教师看 54 维参考 `q+dq`；学生观测下教师动作不是函数 |
| G2 环境 | 整份继承 G1（RSI、跟踪终止、1–3 s 推人） | 教师靠参考能撑，学生会被推离轨迹 |
| G2 曲线 | 后 5k 几乎不动，关节误差 ~2 rad | 局部收敛，续同一套没有希望 |

## 工作项

- [x] R0：判定旧 G2 不可续、根因分层（已完成）
  - acceptance_criteria: 旧 run `2026-08-14_05-30-51_g2_from_g1_m28500` 标为废弃配方；G1 评测三分类与蒸馏标签不一致已写明。
  - verification_commands: `python3 -c "import json; d=json.load(open('artifacts/eval/t4_vault_g1_m28500.json')); print(d['summary'])"`
  - success_definition: 后续不再把「再训 1 万 iter」当成选项。

- [x] R1：G1 评测 reset 对齐 + m28500 复评
  - scope: Play/eval 的 motion 扰动收成 0。evaluator 每批独立 `reset`；冷启动要先丢弃一次零动作 step 再 reset，否则第一批 16 条全是 step-0 `wrist_body_pos`。对照 JSON 不覆盖。
  - acceptance_criteria: `artifacts/eval/t4_vault_g1_m28500_resetfix.json` step-0 wrist = 0，100 条可加总；旧 48/100 保留。
  - verification_commands: `python -m pytest tests/test_t4_vault_evaluator_contract.py tests/test_t4_asset_migration.py -q`（29 passed）；zhuoqun 100-trial → 100/100。
  - success_definition: m28500 从第 0 帧真实能翻完；旧 48/100 是评测耦合。

- [x] R2：G1 起点偏向训练，直到 ≥80/100（取消）
  - scope: 训练 `MotionCommand` 改为 `sampling_strategy='start_window'`：`exact_start_probability=0.35`，`start_window_steps=(0, 100)`（约前 2 s 接近/起势），`start_window_probability=0.35`，其余 0.30 均匀覆盖后段。奖励、终止、观测、plant 不变。第一枪从 `model_28500.pt` resume，预算 5000 iter，2k/5k 各评一次。若 5k 后箱前超时仍 ≥25/100，停续训，用同一采样从零开新 lineage（预算仍 2–3 万 iter）。过 80/100 后停，抽连续回放给人看。不把 AMP clip 入库当作本项完成条件（仍按原 V4，过箱稳定后再做）。
  - acceptance_criteria: 固定 ckpt、100-trial、成功率 ≥80%；回放从 frame 0 翻完；lineage 写清是 resume 还是 from-scratch。
  - verification_commands: zhuoqun tmux + `train_t4_vault_mimic.py --resume`（或 from scratch）；同 R1 evaluator 命令换新 ckpt / 新 output JSON。
  - success_definition: 有一个值得蒸馏的从起点翻箱专家。
  - cancelled_reason: R1 独立评测已是 100/100，门槛已满足，不再改采样续训。

- [ ] R3：G2 教师开车 BC（信息瓶颈实验）（当前）
  - scope: `rsl_rl` Distillation 增加采集模式：环境执行教师动作，学生只对同一转移做 MSE（教师开车）。`train_t4_vault_skill.py` 默认走该模式。G2 环境：关掉 `push_robot`；gait 槽置零并冻结（保留 96D 布局，不泄漏 motion 相位）。仍用 RSI，保证标签在轨迹上。教师必须是 R2 过门槛的 G1 ckpt，禁止再用「只会箱前站」的 m28500，除非 R1 复评已经 ≥80。先 200–500 iter probe，再正式几千 iter。
  - acceptance_criteria: 合同测试锁住「step 用教师动作、loss 用学生 vs 教师」；probe 无 NaN；正式 run 达到本计划 Success Criteria 第 3 条。失败则停，不自动开 R4。
  - verification_commands: `python -m pytest tests/test_t4_vault_skill_contracts.py -q`（扩展教师开车合同）；zhuoqun `train_t4_vault_skill.py --teacher_checkpoint <R2 ckpt>`。
  - success_definition: 证明 1155D scan 观测在教师轨迹上足够克隆翻箱动作。

- [ ] R4：学生开车 DAgger + PPO
  - scope: R3 通过后切学生 on-policy。行为损失保留；加上 PPO（或至少用现有跟踪奖励做策略梯度），避免复合误差把标签变成一对多。可选：仅在教师跟踪误差低于阈值时计入 BC。固定 1m 箱，RSI 可减弱但本项不要求 loco 站立 reset。
  - acceptance_criteria: 固定 1m 箱 evaluator JSON + 回放显示完整过箱；behavior loss 不回到 0.27 平台；不得只报 reward。
  - verification_commands: 同 R1 evaluator，task 用 `t4_vault_skill` 的 eval 变体（若还没有，沿用 mimic_eval 加载学生 actor 的最小接线，并在 lineage 写明）；pytest 回归蒸馏合同。
  - success_definition: G2 在固定箱上具备可合并的技能行为，这才是原 V6 蒸馏段的完成态。

## Commit Units

1. `fix(t4): G1 评测从第 0 帧独立 reset`——R1 代码 + 合同（待 review）。
2. ~~`train(t4): G1 起点偏向翻箱专家`~~——R2 已取消。
3. `feat(t4): G2 教师开车行为克隆`——R3。
4. `feat(t4): G2 学生 DAgger 与 PPO`——R4。

每个 unit：实现完成 + review 无 Critical + 对应 verification_commands 通过后再提交。中文 commit message。

## Known Risks / Blockers

- R1 之后 m28500 真实成功率可能仍远低于 80%。这不阻塞 R1，但会把 R2 推向 from-scratch，多几天 GPU。
- 只改采样、从 m28500 续训，可能改不掉「箱前站」局部解。5k 停训规则就是为这个设的。
- R3 失败 = scan+本体历史无法替代 54 维参考时钟。备选（需另开 slice，不在本计划自动做）：递归学生、用箱子距离作家内进度特征但仍遵守「无 reference 关节」合同。禁止把 `command=q_ref` 塞进学生。
- 1m ≈ 0.71H，接近论文 student 有效上限；G1 长期卡在 80% 以下再考虑降高增强，属范围变更。
- zhuoqun checkout 可能仍停在旧 commit；开训前必须同步本计划对应代码，且不要碰 nubot / 本机正在跑的 Stage E。
- 本机 Isaac Sim zip 未齐，不影响 zhuoqun 训练；不要用 conda IsaacLab 2.3 play 当 G1/G2 行为证据。

## Recovery Protocol

```bash
git status --short --branch
sed -n '1,40p' .harness/state.md
rg -n '^[-] \[[ x]\]' docs/plans/2026-08-15--t4-vault-g1-g2-recovery-plan.md
tmux ls
ssh zhuoqun 'tmux ls; ls -lt logs/t4_vault_mimic logs/t4_vault_skill | head'
```

废弃 run：`logs/t4_vault_skill/2026-08-14_05-30-51_g2_from_g1_m28500`（配方作废，ckpt 可留档，不可 resume）。

## 批准后落地

1. [x] 本文件已作为 planning surface。
2. [x] 合并 plan Active slice / Next skill 已改指向本 recovery；V7 blocked。
3. [x] `.harness/work_index.md`、`.harness/state.md`、`.harness/decisions.md` 已同步，当前项 = R1。
4. [x] Play/eval 零扰动合同已合入本机测试。
5. [ ] zhuoqun 对 `model_28500.pt` 写 `artifacts/eval/t4_vault_g1_m28500_resetfix.json`。

## Next Skill

`implement` R3（教师开车 BC）。R1 commit unit 待 review 后再提交。

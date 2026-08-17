# Executable Plan - T4 走跑与 1m 翻箱合并（LightLP V-B/V-C 复现）

> **Status: archived（非权威）** — living work surface: `.harness/work_index.md` → `docs/plans/2026-08-15--t4-vault-g1-g2-recovery-plan.md`。

> Status: superseded
> Date: 2026-08-13
> Spec: `docs/specs/2026-08-13--t4-vault-loco-merge.md`（user-approved）
> Branch: `t4-train`
> Planning surface: docs plan
> 关系: 与 `docs/plans/2026-08-12--t4-unified-depth-locomotion-plan.md`（Stage E，nubot 训练中）并行；
> G3 被 Stage E teacher evaluator 硬 gate。

## Objective

在 TienKung-Lab（LeggedLab 栈、Stage E plant）内复现 LightLP V-B/V-C 三段：

```text
G1 mimic teacher（从零，特权 tracking）
    参考 overbox_1m_t4_mjcf_fps50.npz（346 帧 @50Hz，固定 1m 箱）
    -> 新 plant 上稳定翻越 1m 固定箱 + 成功 rollout（兼作 skill AMP 数据）

G2 heightscan 技能策略
    DAgger(+PPO) 蒸馏到 Stage E teacher actor 合同（1155D，MLP 512/256/128）
    -> 丢 expert，箱宽/深/位置随机化泛化，loco 一致站姿 reset

G3 合并 + transition（单一策略）
    多专家 DAgger（loco 组 = 冻结 Stage E teacher / skill 组 = G2 策略）
    + transition 组 RL 微调（稀疏过箱奖励 + 密集接近奖励 + 相位切换双 prior AMP）
    -> 会走、会翻 1m 箱、凭 scan 与命令自主切换，无 reference / skill label / 状态机
```

三个 gate 各自产出 evaluator JSON + 连续回放视频 + lineage manifest。

## Active Slice

被 `docs/plans/2026-08-15--t4-vault-g1-g2-recovery-plan.md` 接管。旧 G2
`2026-08-14_05-30-51_g2_from_g1_m28500` 配方作废，不可 resume。V7 G3 继续
blocked。当前项是 R1（评测 reset 对齐 + m28500 复评），不是 V7。

## Non-goals

- depth 蒸馏与部署观测（下一 slice，沿用已批 Spec 的 teacher→depth 合同）。
- 第二技能、技能链、高度泛化验收（验收只看 1m；0.8m/梯形只报告）。
- 真机部署与安全验收。
- 修改冻结部署 Actor 合同或 `legged_lab/assets/t4/schemas.py` 观测 schema。
- PHP 栈（`whole_body_tracking`）继续开发；zip 只作配方与数据来源。
- 不以 reward 曲线、loss、checkpoint 存在替代行为验收。

## Success Criteria

1. **G1**：新 plant 上 1m 固定箱 strict evaluator（持续翻越 + 稳定落地 + 无 hard
   violation）成功率 ≥95%，≥100 trials。
2. **G2**：1m 箱、宽 x∈[0.7,1.0]m / 深 y∈[0.5,1.5]m / 位置随机，成功率 ≥90%，
   500 trials；观测中无参考、无障碍特权（仅 1155D 合同）。
3. **G3** 同一 checkpoint 四项全过：
   - Stage E evaluator 全 bucket 相对 loco teacher 回退 ≤5 个百分点；
   - 1m 箱 corridor 成功率 ≥90%；
   - transition course（走 ≥3m → 翻箱 → 走 ≥3m 至出口，corridor + ordered gates）
     成功率 ≥90%，100 trials；
   - 后退命令负例 0/100 上箱。
4. 每个 gate：evaluator JSON（lineage、seed、固定 checkpoint、bucket、成功判定、
   失败原因、禁止接触与 hard-limit 计数）+ 连续回放视频 + lineage manifest。

## Verification Path

```text
V0 纯 Python 数据合同（本机 pytest）
  -> V1 mimic 任务移植合同测试 + IsaacLab 2-env smoke（nubot tmux）
  -> V2 箱体场景 + strict evaluator RED case（未训练 checkpoint 产完整 JSON）
  -> V3 zhuoqun runtime preflight（失败则部署后重跑）
  -> V4 G1 数值健康 probe -> 正式 lineage -> G1 evaluator gate + skill AMP clips 入库
  -> V5 rsl_rl 多组扩展单测（多组 DAgger / per-group / one-hot / 双判别器）
  -> V6 G2 蒸馏与泛化 -> G2 evaluator gate
  -> [硬 gate] Stage E teacher evaluator 通过
  -> V7 G3 合并 + transition -> G3 四项验收
```

所有训练、GPU probe、批量 playback、evaluator 一律 tmux。每个训练阶段先短 probe
（NaN/Inf、OOM、吞吐、关键行为信号）再进正式 lineage。

### Verification Path Status

`runnable`

V0 本机即可跑；IsaacLab smoke/训练/评估在 nubot runtime 已验证可跑（Stage E 先例）；
zhuoqun 是计划内显式 V3 gate，不构成无法规划的外部阻塞。G3 的 Stage E evaluator
硬前置是计划内 stop gate。

## Required Capabilities

- nubot 4 卡（Stage E 专属）+ zhuoqun 4 卡（技能侧，V3 preflight 后；2026-08-13
  用户裁定与跨栏 skill 对半分：vault 2 卡 + 跨栏 2 卡，见
  `docs/plans/2026-08-13--t4-hurdle-skill-plan.md`）+ 本机 1 卡（渲染/调试）；tmux。
- 仓库内置 `rsl_rl` 扩展：多组 DAgger runner、per-group reward/termination、critic
  组别 one-hot、AMP 双判别器相位切换（V5 代码工作）。
- LeggedLab 扩展：1m 箱 corridor 地形、transition 三区域 reset、strict evaluator
  （crossing sustained + stable landing 语义沿用 zip evaluator 定义）。
- 人工复核：回放视频行为裁定需用户目视确认。

## Fallback Evidence

无可替代最终行为 gate 的 fallback。

- zhuoqun 不可用时，V4/V6 可临时排队 nubot 空闲窗口或本机小规模 probe，但正式
  lineage 与 gate 证据必须在目标 zhuoqun runtime 产出（2026-08-13 用户裁定：与
  跨栏 skill 对半分后，2 卡正式 lineage 可接受，原「4 卡级」要求放宽）。
- 本机无 IsaacLab 时，V1/V2 的合同测试先行，IsaacLab smoke 在 nubot 补齐；未补齐
  前对应工作项不得标记完成。

## Final Integration Claim

`final_integration_claim`: 单一 1155D heightscan 策略（Stage E teacher actor 合同，
无 reference / skill label / 运行时状态机）在同一固定 checkpoint 上同时通过 G3 四项
验收（loco 回退 ≤5pp、1m corridor ≥90%、transition course ≥90% @100 trials、后退
负例 0/100），并有 evaluator JSON、连续回放视频与 lineage manifest 三证齐全。

## 关键合同事实（实现输入）

- 参考动作：`overbox_1m_t4_mjcf_fps50.npz`，SHA256
  `67dde1586388656a192fdba5f57ec4760df6bec6342a474b8313e6880130e0a0`（zip 内
  preflight 真值一致），keys = `fps/joint_names/body_names/joint_pos/joint_vel/
  body_pos_w/body_quat_w/body_lin_vel_w/body_ang_vel_w`，346 帧 @50Hz，27 关节
  （MJCF BFS 序，需按名重排到 `T4_JOINT_NAMES`），32 body。
- PHP tracking 合同（zip `t4_contract.py`）：anchor=`Trunk`、feet=`left/right_foot_link`、
  wrist=`AL7/AR7`、tracking bodies 共 14 个——全部存在于 LeggedLab URDF 30 link 中；
  npz 独有 4 个 spherehand body 不参与 tracking；URDF 独有 `Waist_pitch/Waist_roll`。
- 箱体场景（PHP 原件）：size `(1.0, 1.0, 1.0)`，世界位姿 pos `(0.38, 0.2, 0.5)`、
  quat `(1,0,0,0)`；motion 的 `body_pos_w` 与该箱位姿同一世界系。
- plant 唯一真值：`legged_lab/assets/t4/t4.py` 现值（踝 pitch 80/4、roll 20/1，
  self-collisions on，action scale 0.25，屈膝站姿）；plant 再变则技能侧重训。
- plant caveat：npz 的 body 世界量由 PHP MJCF spherehand plant 前向仿真产生，在
  Stage E URDF plant 上仅作 RSI 初始化与奖励目标参考；关节角/速度可直接作 mimic 参考。

## 工作项

- [x] V0：参考动作入库与纯 Python 数据合同（2026-08-13 完成）
  - scope: npz 原件入库 `legged_lab/envs/t4/datasets/motion_tracking/` + `_manifest.json`
    （SHA、帧数、fps、场景箱位姿、tracking body 集、URDF 缺失 body、plant caveat）；
    新增 `legged_lab/assets/t4/tracking_motion.py` 离线加载器（key/shape/fps 校验、
    关节名双射校验、按 `T4_JOINT_NAMES` 重排、nonfinite fail-fast）；合同测试进
    `tests/test_t4_asset_migration.py`；datasets README 同步。
  - progress: npz SHA 与上游 preflight 真值一致；`tests/test_t4_asset_migration.py`
    12 项 + `tests/test_t4_observation_contracts.py` 20 项全绿；black/flake8/isort 干净；
    独立对抗审查 ready-yes（无 Critical；两条 Important——fps 非有限值校验与 manifest
    tracking body 名单钉死——已当场修复并复测）。
  - acceptance_criteria: npz SHA 与 manifest 一致；加载器输出关节序 == `T4_JOINT_NAMES`
    且与独立重排数值一致；manifest tracking bodies ⊆ npz body_names 且 ⊆ URDF links；
    篡改数据（缺 key/改关节名）fail-fast；既有合同测试不回归。
  - verification_commands: `python -m pytest tests/test_t4_asset_migration.py -q`;
    `python -m pytest tests/test_t4_observation_contracts.py -q`
  - success_definition: G1 任务可通过唯一加载器消费重排后的参考动作，数据 lineage 可机器复核。

- [x] V1：G1 mimic 任务移植（2026-08-13 完成）
  - scope: 在 LeggedLab 新增 tracking 任务（建议名 `t4_vault_mimic`）：固定 1m 箱
    场景、RSI（参考状态初始化）、全局 anchor 与 wrist/feet 跟踪奖励、
    `anchor/foot/wrist` 偏差终止语义（移植 PHP 配方，plant 用 Stage E 现值）；150D
    tracking 观测合同只存在于任务内部。
  - progress: `legged_lab/envs/t4/vault_mimic/`（mdp 五模块 + env cfg + agents +
    rsl_rl_compat + gym 注册）+ `train_t4_vault_mimic.py` 入口；URDF 并入 PHP
    spherehand 手部碰撞（`half_sphere.obj`）；合同测试并入
    `tests/test_t4_asset_migration.py`（15 项全绿）而非单独文件；smoke 在 zhuoqun
    docker（目标 runtime，强于原 nubot 方案）通过：policy 150D / critic 276D /
    action 27D / RSI 生效 / 50 步零动作全有限 / 双手碰撞 prim 存活 USD 转换
    （`/tmp/t4_vault_smoke_result.json`）。场景地面因 Nucleus 云资产在国内不可达
    改为程序化 800m 长方体（顶面 z=0），headless 容器下 visual_material 一律 None。
  - acceptance_criteria: 任务注册可被 train 入口解析；观测/动作维度合同测试通过
    （27D action、scale 0.25）；目标 runtime 2-env smoke 无 NaN/Inf/OOM 且 RSI 生效
    （episode 从参考帧姿态起步）。
  - verification_commands: `python -m pytest tests/test_t4_asset_migration.py -q`;
    `ssh zhuoqun 'CUDA_VISIBLE_DEVICES=0 ./scripts/zhuoqun_run.sh legged_lab/scripts/smoke_t4_vault_task.py'`
  - success_definition: G1 训练环境在目标 runtime 端到端可跑，合同由测试锁定。

- [ ] V2：1m 箱 corridor 场景与 strict evaluator（纯 Python 合同已绿，IsaacLab RED 待 GPU）
  - scope: corridor + ordered gates 的箱体评估场景；strict evaluator（crossing
    sustained + stable landing + 禁止接触/hard-limit 计数 + 固定 checkpoint 输入 +
    lineage 字段）；RED case：未训练/零策略 checkpoint 产出完整 JSON 且判失败。
  - progress: `legged_lab/envs/t4/vault_eval.py` 冻结 1m corridor 布局（箱体填满
    车道）与 zip 过箱/落地判定；`t4_vault_mimic_eval` 在 play 场景加两侧墙；
    `tests/test_t4_vault_evaluator_contract.py` 12 项全绿（含参考动作本身是
    strict success、零策略 batch 成功率 0、侧绕/跳 gate/已过箱起步/hard
    violation、JSON lineage 字段）。IsaacLab `--policy zero` RED 等 G1 腾卡后
    在 zhuoqun 跑；不要把 V2 代码同步进正在训的 checkout。
  - acceptance_criteria: evaluator JSON 含 lineage/seed/checkpoint/bucket/成功判定/
    失败原因/违规计数全字段；零策略成功率为 0 且失败原因分类正确；侧绕/跳 gate
    判失败。
  - verification_commands: `python -m pytest tests/test_t4_vault_evaluator_contract.py -q`;
    `ssh zhuoqun 'CUDA_VISIBLE_DEVICES=0 ./scripts/zhuoqun_run.sh legged_lab/scripts/eval_t4_vault.py --task t4_vault_mimic_eval --policy zero --episodes 16 --output artifacts/eval/t4_vault_red.json'`
  - success_definition: G1/G2 gate 的证据机器可产出、失败可分类，评估语义先于训练冻结。

- [x] V3：zhuoqun runtime preflight（2026-08-13 完成，docker 路线）
  - progress: 用户提供 docker 环境；runtime = 容器 `t4-isaac-jammy:v2`（在用户
    原镜像上补 `libxt6`——缺它导致 kit GPU foundation 全灭——及 vulkan-tools/
    zenity）+ host bind-mount isaac-sim 5.1 standalone / IsaacLab / 本仓库，入口
    `scripts/zhuoqun_run.sh`（默认 v2 镜像、PYTHONUNBUFFERED、GPU/cache 挂载）。
    preflight 证据：容器内 torch CUDA 4 卡可见、vulkaninfo 枚举 4×RTX 4090、
    IsaacLab 可导入、`t4_vault_mimic` 2-env spawn + 50 步全有限（即 V1 smoke）。
    坑已记录：Nucleus 云资产不可达（场景禁用外部 USD/材质）、instanceable USD
    需 TraverseInstanceProxies 遍历、`SimulationApp.close()` 容器内会吞 traceback
    且挂死（入口已加 watchdog 强制退出）。
  - acceptance_criteria: preflight 在 zhuoqun 全 PASS（含 CUDA 4 卡可见、IsaacLab
    可导入、t4_vault_mimic spawn）；结果记录进 `.harness/state.md`。
  - verification_commands: `ssh zhuoqun 'CUDA_VISIBLE_DEVICES=0 ./scripts/zhuoqun_run.sh legged_lab/scripts/smoke_t4_vault_task.py'`
  - success_definition: 技能侧训练资源就绪，G1 正式 lineage 有明确落点。

- [ ] V4：G1 正式训练与 gate + skill AMP clips 入库（训练已启动 2026-08-13）
  - progress: zhuoqun tmux `t4_vault_g1`，CUDA 0，4096 env / 30000 iter / seed 42，
    日志 `/tmp/t4_vault_g1_train.log` + 仓库 `logs/t4_vault_mimic/2026-08-13_09-30-13`；
    2026-08-14 12:5x 抽检 iter 27955/30000，mean reward ~47、episode length ~480/500、
    body_pos error ~0.05 m、`time_out` 主导终止，`model_27500.pt` 已落盘。G1 数值
    健康；strict gate 等 V2 IsaacLab RED 通过后对固定 checkpoint 跑。
  - scope: 短 probe（数值健康 + 关键行为信号）→ 正式 lineage（预算 ~20000 iter，
    多卡，tmux）→ G1 strict evaluator（≥95%，≥100 trials）→ 成功 rollout 提取 66D
    skill AMP clips 入库（`legged_lab/envs/t4/datasets/` 下新目录 + manifest）→
    连续回放视频人工复核。
  - acceptance_criteria: G1 成功标准达成且三证齐全；skill AMP clip 66D 特征形状
    与 manifest 合同测试通过；nonfinite/hard-limit/禁止接触计数为零。
  - verification_commands: `tmux new-session -d -s t4-g1-train '<zhuoqun-or-nubot> bash scripts/train_t4_vault_mimic.sh 2>&1 | tee logs/t4-g1-train.log'`;
    `python -m pytest tests/test_t4_asset_migration.py -q`（AMP clip 合同并入）;
    G1 evaluator 命令由 V2 产物固定。
  - success_definition: 新 plant 上有可蒸馏的 1m 翻箱专家与 skill AMP 数据。

- [x] V5：rsl_rl 多组扩展（2026-08-14 库 + 单测完成）
  - scope: 多专家 DAgger runner（按组查询冻结 expert）、per-group reward/termination、
    critic 组别 one-hot、AMP 双判别器相位切换（箱前缘固定偏移触发点）；纯 torch
    单测先行。
  - progress: `rsl_rl/rsl_rl/utils/multi_expert.py`（组 ID、critic-only one-hot、
    loco/skill 查询、transition 不监督、per-group scale）+
    `rsl_rl/rsl_rl/utils/dual_amp.py`（箱前缘 trigger、硬切换、判别器 loss mask）。
    `tests/test_multi_expert_dagger.py` + `tests/test_dual_amp_switch.py` 绿；
    `tests/test_t4_observation_contracts.py` 未回归。未改 AMPPPO 默认路径。
    IsaacLab runner 接线留到 V6/V7 任务。
  - acceptance_criteria: 单测覆盖组路由正确性（loco/skill/transition 各组 expert
    查询与 loss mask）、one-hot 只进 critic、判别器切换点数值正确；现有 AMPPPO
    训练路径不回归（Stage E 合同测试仍绿）。
  - verification_commands: `python -m pytest tests/test_multi_expert_dagger.py tests/test_dual_amp_switch.py -q`;
    `python -m pytest tests/test_t4_observation_contracts.py -q`
  - success_definition: G2/G3 所需训练机制在库内可用且被测试锁定。

- [ ] V6：G2 heightscan 技能蒸馏与泛化（任务已写，严格 gate 推迟；蒸馏待腾卡）
  - scope: DAgger(+PPO) 把 G1 expert 蒸馏到 1155D 合同（去障碍尺寸/距离特权）；
    丢 expert 后按箱宽/深/位置随机 + 箱后目标 task reward 泛化；reset 用 loco 一致
    站姿（不用 RSI）；G2 evaluator（≥90%，500 trials）+ 回放。
  - progress: `t4_vault_skill` + `train_t4_vault_skill.py`（`--teacher_checkpoint` 加载
    G1 actor 为冻结 teacher）。学生 1155D = Stage E teacher actor；教师组 150D；
    `tests/test_t4_vault_skill_contracts.py` 4 项绿。泛化阶段与 90% gate 未做。
  - acceptance_criteria: G2 成功标准达成且三证齐全；观测合同 == Stage E 1155D
    schema（合同测试）；无参考/无障碍特权输入。
  - verification_commands: `python -m pytest tests/test_t4_vault_skill_contracts.py -q`;
    训练/评估命令由 V4/V2 模式固定（tmux）。
  - success_definition: 技能能力已迁移到与 loco 同合同的 heightscan 策略上。

- [ ] V7：G3 合并 + transition 统一策略（blocked：等 recovery R4 + Stage E evaluator）
  - scope: 三组环境（loco 组冻结 Stage E teacher DAgger / skill 组 G2 策略 DAgger /
    transition 组三区域 reset + 稀疏过箱奖励 + 密集接近奖励 + 相位切换双 prior AMP）；
    G3 四项验收 + 报告项（0.8m 箱、梯形）。
  - acceptance_criteria: Success Criteria 第 3 条四项全过，同一固定 checkpoint；
    三证齐全；报告项有数值。
  - verification_commands: 训练/评估命令由 V5/V2 产物固定（tmux）；
    `python -m pytest tests/ -q`（全量合同回归）。
  - success_definition: `final_integration_claim` 成立，产物可交接给 depth 蒸馏 slice。
  - 硬前置: Stage E teacher evaluator 通过（`docs/plans/2026-08-12--t4-unified-depth-locomotion-plan.md` M4）。

## Commit Units

每个 commit unit 在对应工作项实现完成、review 无 Critical、verify PASS 后提交：

1. `feat(t4): 落地翻箱参考动作数据合同与合并计划`——V0 + Spec 批准 + 本计划 +
   harness 同步（本 session）。
2. `feat(t4): 移植 1m 翻箱 mimic 任务`——V1。
3. `feat(t4): 增加翻箱 corridor 场景与 strict evaluator`——V2。
4. `chore(t4): zhuoqun runtime preflight 与部署记录`——V3。
5. `train(t4): G1 mimic teacher lineage 与 skill AMP clips`——V4（大 checkpoint 是否
   入库按仓库 artifact 规范，不默认提交）。
6. `feat(rsl_rl): 多组 DAgger 与双判别器 AMP 扩展`——V5。
7. `train(t4): G2 heightscan 技能蒸馏与泛化`——V6。
8. `train(t4): G3 合并 transition 统一策略`——V7。

## Known Risks / Blockers

- 1m ≈ 0.71H 接近论文 student 有效上限 0.77H；G1 卡 95% 以下时的降高增强路径属
  范围变更，需回 brainstorm 重裁。
- `stage_e_prov5` 失败并再改 plant/MDP ⇒ 技能侧已训部分作废重训（用户已接受）。
- npz body 世界量来自 MJCF plant，前向仿真到 URDF plant 会有接触/惯量差；RSI 与
  奖励目标可能需要小幅容差调整（V1 实现内消化，不改参考数据）。
- per-group reward/termination 与双判别器是最大代码面；缓解：V5 单测先行、每步
  smoke probe。
- transition 稀疏奖励探索难；缓解：论文三区域 reset 照抄。
- zhuoqun 无 IsaacLab（Route W 已证）；V3 预期走部署分支，工期不确定。
- 爬箱中段 scan 遮挡属 depth slice 风险，本计划不处理。

## Recovery Protocol

```bash
git status --short --branch
sed -n '1,60p' .harness/state.md
rg -n '^[-] \[[ x]\]' docs/archive/plans/2026-08-13--t4-vault-loco-merge-plan.md
tmux ls
```

训练阶段恢复时还需核对：当前 commit、绝对 checkout、GPU ownership、tmux session、
日志、最近 checkpoint、数据 SHA、evaluator version；不得从旧对话推断训练状态。

## Next Skill

`implement` on `docs/plans/2026-08-15--t4-vault-g1-g2-recovery-plan.md` R1。

Reason: 旧 G2 已平台化且标签不一致；先对齐 G1 评测 reset，再补从第 0 帧过箱
能力。不要开 V7。

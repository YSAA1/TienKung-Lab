# S12 学生主蒸馏 + 可选 FT 最终执行计划

> Status: superseded
> Scope: 只针对 T4 S12 GRU 深度学生；不改 teacher MDP，不设计自动 supervisor。
> Revised: 2026-08-31。主蒸馏必须含小权 PPO；15k 是预算不是典礼。主蒸馏之后的用户可见 FT 是 residual task-success FT，不是 plant `final_ft`，也不是 Phase C `recon=0.25`。
> 已被 `docs/archive/plans/2026-09-01--t4-s12-repr-first-distill-plan.md` 取代。对照产物仍是 `model_13500.pt`。

## Objective

以当前已经验证的 S12 teacher/student 合同为原型，收敛成两条可人工控制的命令：

1. 一条命令完成一次主蒸馏：Ross mix → critic 校准 → **同一 run 内 DAgger + 小权 PPO**；
2. 主蒸馏评估并选定完整 ckpt 后，按需要由第二条命令做一次 **residual task-success FT**（短 PPO、BC 留 0.5、过采样 hard 踏石/圆桩）。plant `final_ft` 仍可跳过，不在本 slice 落地。

不再把 DAgger、Joint PPO、Deploy FT、Targeted FT 拆成多个用户可见阶段；不自动判断是否 FT，不自动终止训练。

## Active slice

把现有 `train_t4_sparse_depth_student.py` 和 `train_t4_sparse_depth_student_ft.py` 收敛为 S12 的最终两阶段配方，并补齐 teacher-consistent terrain evaluator。

阶段 1 代码已落地。residual FT 与 10k 续训已停。产物锁 `model_13500.pt`。用户主攻老师复现：同一 ckpt 关评测深度噪声后 hard 踏石 18→16、圆桩 23→25，判定不是噪声盖住技能。plant FT 代码在但不开。下一刀若动，动蒸馏合同，仍先不重训直到写出单一旋钮。stairs/rough evaluator 仍随后。

## Non-goals

- 不改 S12 teacher MDP、地形比例、奖励或终止。
- 不写自动 supervisor、自动停训、自动选 ckpt、自动触发 FT。
- 不把 PPO 从主蒸馏挪到 FT。Phase A 纯 DAgger hard 踏石/圆桩只有 `4/32`、`5/32`；Phase B 加上小权 PPO 才到 `18/32`、`27/32`。梅花桩 hard 必须在主蒸馏里用 PPO 学。
- 不启用 Phase C `deploy_ft`（`recon_coef=0.25`）。residual FT 允许 `pg=0.5` / `BC=0.5`，但 `recon_coef=0.5`，且过采样 hard 稀疏。
- 不把 `model_799`、mix0、D3/D3b/D3c 当 parent。
- 不把地形换成只含踏石/圆桩；FT 不加 targeted 制造误差（桩顶倾斜、XY/高度/直径抖动）。
- 不把「先干净深度、蒸完再补噪声」写成论文对齐。主蒸馏从 iter 0 就开部署噪声。
- 不把 1024-env RTX 再探一遍；OOM 只降 env，禁止退回 warp。多卡按 256/卡，不把 CLI 的 `--task_num_envs` 写成总 env。
- 不在同一条 main run 里用 adaptive-KL 换 LR。全程 `schedule=fixed`。
- 不动 zhuoqun 翻箱。

## Success criteria

1. 一条主蒸馏命令即可在 teacher 全地形分布上，从零蒸部署向 GRU 学生；同一 run 内先纯 DAgger，再小权 PPO；`teacher_mix>0` 与有效 PPO 不得重叠。
2. 主蒸馏预算 15k、每 500 存盘；最终候选按 evaluator 选 ckpt，不默认用最后一个。
3. 第二条 FT 命令可选、可回滚；只接受 main 完整 checkpoint；不做第二次 FT。
4. 学生能力按同口径 Isaac evaluator 声明，见「Evaluation and evidence」。Reward / TB / loss / ckpt 存在都不算过门。
5. 相机、观测维、噪声模型和 env 数与已验证的 S12 RTX 学生合同一致。

## Terrain contract

`T4LocoSparseDepthStudentEnvCfg` 已经继承 `T4LocoSparseTeacherEnvCfg`，generator 已是 `T4_STAGE_E_SPARSE_TERRAINS_CFG`。本计划不是「给学生新造全地形」，而是 **停止用学生课表迁就梅花桩**。

必须覆盖的子地形（teacher 现行比例，踏石+圆桩合计 0.40）：

- flat、random_rough、boxes、wave；
- hurdles；
- slope_up、slope_down；
- stairs_up_30、stairs_up_34；
- stairs_down_30、stairs_down_34；
- stepping_stones、raised_pillars。

学生改回 teacher 课表，不再使用现行学生覆盖：

| 项 | 老师 | 现行学生（要改掉） | 本计划 |
| --- | --- | --- | --- |
| terrain generator | `T4_STAGE_E_SPARSE_TERRAINS_CFG` | 已继承 | 保持 |
| `max_init_terrain_level` | 2 | 继承 2 | 保持老师 |
| `random_level_reset_fraction` | 0.10 | **0.50** | 改回 0.10 |
| `random_level_reset_min_level` | `None` | **6** | 改回 `None` |
| `random_level_reset_max_level` | `None` | 未改 | 保持 `None` |
| command / termination / reward | S12 teacher | 已继承 | 保持老师 |

取舍：改回老师 10% 随机 level、从 level 2 起训，楼梯/rough/跨栏曝光会增加，**hard 梅花桩曝光会少于现行 `0.50 / min_level=6`**。这是有意的。稀疏 hard 地板仍按 Phase B 验收；若新课表导致 hard 稀疏掉到门下，用 evaluator 暴露，不靠再写一套学生专用地形去救。

`tests/test_t4_sparse_depth_student_gru_contract.py` 必须钉死老师课表 `0.10 / min_level=None`，不能再断言旧的 `0.50 / min_level=6`。

学生只替换观测为部署向 depth/proprio/GRU，并加入已有部署噪声。

## Camera and env count

与已跑通的 S12 RTX 学生相同，不改眼睛：

- tiled RTX D455，策略分辨率 48×64，不捕获 72×128 再缩；
- 头高 Trunk site，下俯 35°；
- `update_period=0.02`，真实 frame refresh；
- 3 帧 depth history、10 帧 proprio history、3168-D student；
- 每 env 一张 tiled 视图，所以相机数 = `num_envs`。

环境数：

- `--task_num_envs` 永远是 **每卡 256**；
- 老师 1024 是无相机；学生 RTX 1024 已 OOM，禁止再探；
- 正式开训用 **4 卡 × 256 env**（用户授权）。若 PhysX CUDA 在多卡上再失败，降到单卡 256，不降分辨率、不退 warp。

## Main distillation

主蒸馏 = 克隆 + 小权 PPO，不是纯 DAgger。PPO 在这条命令里，不在 FT 里。

- 从零初始化 student；teacher 固定为 Gate 0 的 `model_21500.pt`。
- **预算 15,000 iterations**，单一 main run，每 500 iter 存盘。15k 是上限，不是必须跑满。
- 深度 range/scale/dropout/delay、proprio noise、相机外参 ±1 cm / ±0.025 rad 从 iteration 0 开启。
- 不加入 targeted FT 专属几何。

### 默认 loss schedule

实现必须落到这些 cfg 旋钮，不能只写墙上时钟：

| 旋钮 | 值 |
| --- | --- |
| `teacher_mix` → `teacher_mix_end` | `1.0 → 0.0` |
| `teacher_mix_decay_iters` | `2000` |
| `pg_coef` | `0.2` |
| `pg_delay_iters` | `2000` |
| `critic_warmup_iters` | `200` |
| `pg_coef_ramp_iters` | `800` |
| `behavior_coef` / `_end` / `_decay_iters` | `1.0` / `1.0` / `0` |
| `recon_coef` | `1.0` |
| `schedule` | `fixed` |
| `learning_rate` | `1e-4`（全程，不中途改） |
| `max_iterations` | `15000` |
| `save_interval` | `500` |
| `entropy_coef` | `0.0` |

对应墙上时钟：

0–2,000：mix 1→0，有效 `pg=0`，actor 正常 BC+recon 更新，不冻结。\
2,000–2,200：mix=0，冻 actor，只校准 critic。\
2,200 之后：student-only rollout，`pg` 0→0.2（800 iter ramp），BC/recon 不降。

`SafeRecurrentDistillation` 必须拒绝有效 PPO 与 `teacher_mix>0` 重叠，且 PPO 只使用实际 student 执行的 rows。`pg_delay_iters` 必须 ≥ mix 归零点，否则 critic freeze 会落到 mix 段上。

全程 LR `1e-4`。原计划在 PPO 段写成 `3e-5` 做不到：`schedule=fixed` 只有一个 `learning_rate`，`3e-5` 是 Phase B **另开 continuation** 时的 LR。本配方是一条 run，沿用 D4 的全程 `1e-4`。禁止 adaptive-KL 钉死 LR。

### 预算，不是典礼

默认日程可以按上表启动，但：

- 不自动停训、不自动开 FT、不自动升 `pg_coef`。
- 人工看点：mix 归零后（~2.2k，此时 TB 含过老师搀扶，不当学生能力）、6k、10k、15k。
- 选 ckpt 看 student-only evaluator，不默认 `model_14999`。
- 人看到这些再杀：NaN/OOM；`Loss/behavior` 平台后明显回升（mix0 那种 0.10→0.21）；hard 稀疏在 BC=1 时贴地并继续掉。杀了就评已有 ckpt，不要为了「跑满 15k」接着训。

## FT

主蒸馏之后的用户可见 FT 是 **`residual_ft`**：短任务残差 RL，用来修 scan→depth 落足残差。噪声已经在主蒸馏 iter 0。这 **不是** plant `final_ft`（延迟+执行器），**不是** Phase C `deploy_ft`（`recon=0.25`），也 **不是** D3（`pg=1`、`BC=0`）。

对照现有 mode：

| | `joint`（Phase B） | `deploy_ft`（Phase C） | `targeted_ft` | **`residual_ft`** | 以后的 `final_ft` |
| --- | --- | --- | --- | --- | --- |
| 用户可见 | 旧阶段 | 已否决 | 旧针对性 FT | **当前 FT 入口** | 未落地；残差 FT 过门后再考虑 |
| 预算 | 6k | 1k | 800 | **1k** | 1k |
| `pg_coef` | 0.2 | 0.5 | 0.1 | **0.5** | 0.1 |
| BC / recon | 1.0 / 1.0 | 0.5 / 0.25 | 1.0 / 1.0 | **0.5 / 0.5** | 1.0 / 1.0 |
| 参考动作锚 | 无 | 无 | 0.25 | **0** | 0.25 |
| LR | 3e-5 | 3e-5 | 1e-5 | **3e-5** | 1e-5 |
| 课表 | 老师 0.10 / None | 老师 0.10 / None | 老师 + 制造误差 | **0.50 / min_level=6**，踏石/圆桩比例 0.30/0.30 | 老师 0.10 / None |
| 域 | 与主蒸馏相同 | 与主蒸馏相同 | 延迟+执行器+深度边界+制造误差 | 与主蒸馏相同的深度噪声；**无制造误差、无动作延迟** | 延迟+执行器 |

规则：

- 用户在主蒸馏评估后手动启动；可以不启动。不 resume 15k 主 run。
- parent：主蒸馏选定的完整 checkpoint；拒绝 deploy-only。
- 必须给 teacher checkpoint、teacher eval manifest、capability-gate JSON。
- optimizer / iteration / 算法计数 / action std 重置（`std=0.08`）。
- teacher 重新按 SHA256 校验并冻结。
- 前 200 iter 冻 actor 只校准 critic；`teacher_mix=0`；`pg_coef` 0→0.5（400 iter ramp）。
- 不再启动第二次 FT。
- FT 任一 bucket 相对 main 明显回退则丢掉 FT，保留 main。

落地：`train_t4_sparse_depth_student_ft.py --mode residual_ft`。旧 `joint` / `deploy_ft` / `targeted_ft` 留在代码里。plant `final_ft` 本 slice 不实现。

## Commands

默认 4 卡 × 256 env。两条都在 tmux 里启动。

### 主蒸馏

沿用 `train_t4_sparse_depth_student.py`，新增/固定 final profile（`run_name=s12_final_main`）：

```bash
bash scripts/nubot_run.sh -m torch.distributed.run --nnodes=1 --nproc_per_node=4 \
  legged_lab/scripts/train_t4_sparse_depth_student.py \
  --teacher_checkpoint <model_21500.pt> \
  --teacher_eval_manifest <teacher_gate.json> \
  --task_num_envs 256 --distributed --headless --seed 42 \
  --run_name s12_final_main
```

`--task_num_envs 256` 是每卡环境数。多卡 PhysX 失败时去掉 `torch.distributed.run` / `--distributed`，改 `CUDA_VISIBLE_DEVICES=2` 单卡 256。

### 可选一次 residual FT

```bash
CUDA_VISIBLE_DEVICES=2 bash scripts/nubot_run.sh \
  legged_lab/scripts/train_t4_sparse_depth_student_ft.py \
  --mode residual_ft \
  --student_checkpoint <main_best_full.pt> \
  --teacher_checkpoint <model_21500.pt> \
  --teacher_eval_manifest <teacher_gate.json> \
  --capability_gate_json <main_student_eval.json> \
  --task_num_envs 256 --headless --seed 42
```

中途是否继续、是否执行 FT，由 evaluator 结果人工决定。15k 主蒸馏不要 resume。parent 未用同一 evaluator 选定前不要开训。

## Evaluation and evidence

主蒸馏和 FT 使用同一 evaluator 版本、同一 seed、同一 terrain 名称和同一 command。

固定评估至少覆盖：

- flat；
- random_rough、boxes、wave；
- hurdles；
- stairs up / stairs down（30 与 34 可合并报告，但上下楼要分开）；
- stepping stones easy/hard；
- raised pillars easy/hard。

现有 `eval_t4_hurdle.py` CLI 只有 `hurdles/flat/stepping_stones/raised_pillars`。阶段 3 做最小 terrain-type 扩展，不改变成功定义。每个 bucket 输出 strict success、reach/progress、fall/reset 原因。

开 15k 正式训练前：evaluator 至少要能评 stairs + hurdles + 两类稀疏，否则中途人工看点没有全地形口径。rough/boxes/wave 可同批加上。扩展完成后，先用同一口径评 Phase B `model_5999.pt` 作对照基线，不把它当新 lineage 的 parent。

学生能力门：

- 稀疏不得低于 Phase B：stones strict `32/32`、`18/32`，pillars `29/32`、`27/32`；hard reach_2m `28/32`、`30/32`。
- hurdles：strict ≥16/32，且 reach_2m ≥24/32。
- stairs up 与 stairs down：各自 strict ≥8/32，且 reach_2m ≥16/32。这是 usable 下限，不是老师满分。
- flat / random_rough / boxes / wave：各自 reach_2m ≥24/32，且不得出现 0/32 或接近 0 的坍塌。
- FT 相对 main：任一 bucket strict 或 reach_2m 掉超过 4/32 视为明显回退，弃 FT。
- 本地 MuJoCo sim2sim 至少跑 flat、loco、hurdles、stairs、stepping_stones、raised_pillars。只作 plant/观测交叉诊断，不替代 Isaac gate。

每个候选必须有：evaluator JSON、student lineage、continuous replay、MuJoCo summary、deploy-only export。

## Verification path

本机合同（阶段 1，不需要 Isaac）：

```powershell
D:\anaconda\envs\pytorch\python.exe -m pytest `
  tests/test_t4_sparse_depth_student_gru_contract.py `
  tests/test_safe_recurrent_distillation.py `
  tests/test_t4_observation_contracts.py `
  tests/test_t4_sparse_command_contract.py `
  tests/test_t4_stepping_stone_contracts.py -q
```

阶段 2 追加同一套 GRU/distillation 合同，覆盖 `--mode residual_ft`。

阶段 3：

```powershell
D:\anaconda\envs\pytorch\python.exe -m pytest `
  tests/test_t4_sparse_evaluator_contract.py `
  tests/test_sim2sim_t4_depth_student.py `
  tests/test_t4_stepping_stone_contracts.py -q
```

nubot 正式训练、evaluator、replay 必须 tmux，入口 `scripts/nubot_run.sh`。开训前 RTX checksum probe 仍用已有 `probe_t4_student_depth_render.py`，禁止 1024-env 并发。

## Work Items

- [x] 阶段 1：统一 teacher-consistent 学生环境与 final main 配方
  - acceptance_criteria: 学生 generator/proportion/command/termination 保持 teacher；课表改回 `random_level_reset_fraction=0.10`、`min_level=None`、`max_level=None`；主蒸馏 cfg 为 mix_decay=2000、pg_delay=2000、critic_warmup=200、pg 0→0.2/800、BC=1、recon=1、fixed LR `1e-4`、max 15000、save 500；旧 `0.50/min_level=6` 断言改为新课表。
  - verification_commands: `D:\anaconda\envs\pytorch\python.exe -m pytest tests/test_t4_sparse_depth_student_gru_contract.py tests/test_safe_recurrent_distillation.py tests/test_t4_observation_contracts.py tests/test_t4_sparse_command_contract.py tests/test_t4_stepping_stone_contracts.py -q`
  - success_definition: 一条主蒸馏命令可在 teacher 全地形上从零蒸部署向 student，同一 run 含小权 PPO，且 mix 与 PPO 不重叠。
- [x] 阶段 2：固定可选 `residual_ft` 与 continuation 合同（当前）
  - acceptance_criteria: `--mode residual_ft` 只接受 main full checkpoint；预算 1k；`teacher_mix=0`；critic warmup 200；`pg` 0→0.5 / 400；BC=0.5（不是 0）；recon≥0.5；fixed LR `3e-5`；hard 踏石/圆桩曝光高于老师 `0.10 / None`；无制造误差、无动作延迟；拒绝 plant `pg=0.1`+BC=1+动作锚 和 D3 `pg=1`+BC=0。
  - verification_commands: `D:\anaconda\envs\pytorch\python.exe -m pytest tests/test_t4_sparse_depth_student_gru_contract.py tests/test_safe_recurrent_distillation.py -q`
  - success_definition: 第二条命令能独立启动一次可跳过、可回滚的 residual task-success FT。
- [ ] 阶段 3：扩展 teacher-consistent evaluator 与 5999 对照
  - acceptance_criteria: evaluator 能评 flat/rough/boxes/wave/hurdles/stairs/sparse；输出固定 schema、replay 和 lineage；用同一口径给 Phase B `model_5999.pt` 打一份对照基线（不是 parent）。
  - verification_commands: `D:\anaconda\envs\pytorch\python.exe -m pytest tests/test_t4_sparse_evaluator_contract.py tests/test_sim2sim_t4_depth_student.py tests/test_t4_stepping_stone_contracts.py -q`
  - success_definition: 主蒸馏和 FT 能按同一口径比较 main、FT 与 Phase B；正式 15k 开训前至少能评 stairs/hurdles/sparse。
- [x] 阶段 0：现有基线与失败教训
  - acceptance_criteria: Phase B `model_5999.pt`、model_799 评估、GRU reset/camera freshness/旧 FT 崩塌证据已留存；Phase A vs B 证明梅花桩 hard 需要主蒸馏 PPO。
  - verification_commands: `artifacts/eval/s12_recurrent_statefix_full_gate/`、`artifacts/eval/s12_targeted_ft_m799/`、`artifacts/eval/s12_targeted_ft_m799_sim2sim/`
  - success_definition: 新计划不把 model_799 或旧失败 lineage 当默认 parent，也不把 PPO 从主蒸馏拿掉。

## Commit units

1. `统一S12学生主蒸馏与teacher地形课表`：阶段 1 完成，focused tests PASS，review 无 Critical。
2. `固定S12可选FT与全地形评估流程`：阶段 2–3 完成，验证和 review PASS。正式 15k 训练不进这两个 commit。

训练 checkpoint、视频、TensorBoard event 和大 evaluator 产物不进 Git；关键代码/计划里程碑使用中文 commit message。

## Known risks / blockers

- 现有 evaluator CLI 不能评 stairs、rough、boxes、wave。全地形能力声明在阶段 3 前为 `blocked`。
- 主蒸馏改回老师课表可能降低 hard 稀疏曝光；residual FT 才允许 `0.50 / min_level=6` 和踏石/圆桩 0.30/0.30，主蒸馏和老师仍是 `0.10 / None`。
- 默认 2.2k 开 PPO 比当年 Phase B（先 6–8k 纯 DAgger 再另开 joint）更早。这是一条 run 的代价。人看 6k/10k，不要无监督跑满。
- 单卡 256 在 13 类地形上比老师 1024 稀；踏石+圆桩仍占 0.40，不另造学生世界来加采样。
- MuJoCo 与 Isaac plant 不完全一致；sim2sim 只作交叉诊断。
- 本机正式 sim2sim 必须用 `D:\anaconda\envs\pytorch\python.exe`。
- 远端长任务必须 tmux；不得覆盖当前 dirty worktree 的无关改动。

## Verification path status

`blocked`：阶段 1 的代码合同可用本机 pytest 验；完整 teacher terrain 能力声明要等阶段 3 给 evaluator 补上 stairs/rough/boxes/wave。阶段 3 完成前不得声称全地形已验收。

## Required capabilities

- nubot IsaacLab 5.1 + `scripts/nubot_run.sh` + tmux；
- 本机 `D:\anaconda\envs\pytorch`（torch + mujoco）；
- student-only evaluator、GRU reset 检查、RTX depth freshness probe；
- review 后的 focused pytest。

## Fallback evidence

完整 evaluator 扩展完成前，只接受现有 sparse/flat evaluator JSON、lineage 和 replay 作为局部诊断；不接受全地形能力声明。阶段 1 代码验收走 pytest，不走 15k 训练。

## final_integration_claim

完成后交付一条 teacher-consistent 主蒸馏命令（同一 run：mix → critic 校准 → DAgger+小权 PPO）和一条可选 `residual_ft` 命令（短任务残差 RL）。主蒸馏负责 teacher 全地形能力和部署噪声；residual FT 在人工选定完整 parent 后修 scan→depth 落足残差。plant `final_ft` 不在本 slice。最终候选由人按同口径 evaluator、replay 和 sim2sim 选，不默认最后一个 ckpt，也不默认必须 FT。

## Next skill

`harness-workflow:implement`

# AGENTS.md

本仓库是 TienKung-Lab 的 T4 27DoF locomotion 迁移工作区。优先读取
`PROJECT_CONTEXT.md`、`.harness/state.md`、`.harness/work_index.md`，再进入代码修改。

## Commands

| Command | Description |
| --- | --- |
| `pip install -e .` | 安装根包 `LeggedLab`，需要已安装 Isaac Lab 的 Python 环境 |
| `cd rsl_rl; pip install -e .` | 安装本仓库内置 `rsl_rl` |
| `python -m pytest tests/test_t4_asset_migration.py` | 纯 Python T4 资产与 motion 合同测试 |
| `python -m pytest tests/test_t4_observation_contracts.py tests/test_t4_walk_contracts.py` | T4 观测合同与 `t4_walk` Route W 合同 |
| `python -m pytest tests/test_t4_terrain_curriculum.py` | Stage E 地形课程代数（无 Isaac） |
| `bash scripts/zhuoqun_run.sh ...` | zhuoqun Isaac 入口；必须先导出 ISAACSIM_ROOT / ISAACLAB_ROOT / ISAACLAB_LIB_ENV，当前机器尚未发现 IsaacLab |
| `python legged_lab/scripts/t4_csv_motion_conversion.py --input legged_lab/envs/t4/datasets/motion_source --output-dir legged_lab/envs/t4/datasets/motion_visualization --fps 30` | 重新生成 T4 motion visualization 文件 |
| `pre-commit run --all-files` | 运行格式化和静态检查 |

## Architecture

- `legged_lab/` - IsaacLab locomotion 环境、资产、脚本和 MDP 代码。
- `legged_lab/assets/t4/` - T4 27DoF 资产；`constants.py::T4_JOINT_NAMES` 是唯一关节顺序真值。
- `legged_lab/envs/t4/` - T4 迁移任务包；`t4_loco_teacher` 是 Stage E 特权专家，`t4_walk` 是 TienKung-native proprio-only 对照。
- `legged_lab/envs/t4/datasets/motion_source/` - 原始 T4 CSV，schema 为 `root_xyz(3) + root_quat_xyzw(4) + q27`。
- `legged_lab/envs/t4/datasets/motion_visualization/` - 转换后的 playback 中间数据，不是最终 AMP expert。
- `rsl_rl/` - 仓库内置 RSL-RL 训练库。
- `docs/specs/`、`docs/plans/` - 已批准 Spec 和可执行计划。
- `.harness/` - 当前 active slice、验证路径和恢复入口。

## Current T4 Boundary

- M0 机器审计与 nubot playback 已通过，人工视觉复核仍是正式 AMP expert 的前置 gate。
- 人工复核通过前只允许生成 provisional AMP expert 做 smoke，不启动 teacher 长训练，不启动 student 蒸馏。
- 最终目标是单个 T4 depth student `pi_loco`；`pi_teacher` 只用于训练期 privileged expert。
- 最终导出 Actor 不得包含 HeightScan、高程图、接触真值、terrain truth、route progress 或 teacher-only module。

## Testing And Verification

- 代码改动后先跑最窄测试；T4 资产/motion 合同优先跑 `python -m pytest tests/test_t4_asset_migration.py`。
- T4 观测合同改动跑 `python -m pytest tests/test_t4_observation_contracts.py`。
- 所有训练、GPU probe、批量 playback、evaluator、teacher rollout 采集等长运行命令必须用 tmux。
- 不能用 reward、episode length、checkpoint 存在或 loss 下降替代行为验收；能力声明必须有 evaluator JSON、lineage manifest 和连续回放证据。

## 服务器信息

nubot@100.100.188.39 密码 一个空格 四个GPU
zhuoqun@100.95.109.48 密码123456 四个GPU

Isaac 运行时（nubot）：`bash scripts/nubot_run.sh <script.py> [args...]`。
Isaac 运行时（zhuoqun）：`bash scripts/zhuoqun_run.sh <script.py> [args...]`，但必须显式提供 Isaac 路径。
2026-08-13 探测结果：zhuoqun 没有 Isaac Sim `python.sh` / IsaacLab source；四卡正被
`t4_stair_traversal` MjLab 任务占用。Route W 代码在 `t4-walk` worktree，开训被这道运行时缺口挡住。

## Style

- Python 格式遵循 Black，line length 120；静态检查由 `.pre-commit-config.yaml` 和 `.flake8` 定义。
- 修改要小而准，优先沿用现有 IsaacLab / LeggedLab / RSL-RL 模式。
- 不要改动无关资产、数据文件或用户未跟踪文件。

## Git Workflow

- 项目维护使用 git；关键里程碑提交中文 commit message。
- 提交前检查 `git status --short`，避免把无关用户改动混入提交。

# AGENTS.md

本仓库是 TienKung-Lab 的 T4 27DoF、Unitree G1 29DoF、Z2 29DoF locomotion 工作区，通用执行偏好沿用全局 AGENTS.md。

## 工作入口

- 项目任务先读 `docs/README.md`、`.harness/work_index.md`，再读当前轨道相关规格和计划；若执行工作树在别处，以索引指定的路径为准。
- `docs/specs/` 保存合同，`docs/plans/` 保存现行执行计划，`.harness/` 保存索引和短状态。归档、研究笔记及旧状态段落不作为执行计划。
- 允许多个独立轨道并行；每个轨道只有一个权威执行入口。复用现有恢复文件，不另建重复的根目录计划或状态文件。

## 代码与数据合同

- `legged_lab/` 是 IsaacLab 环境、资产、脚本与 MDP；`rsl_rl/` 是仓库内置训练库。
- `legged_lab/locomotion/` 是共享 locomotion 教师、LightLP、AMP、深度学生运行时、观测、镜像和课程算法。机器人通过显式 `LocomotionRobotSpec` 接入，不继承另一机器人的任务实现。
- `legged_lab/motion_tracking/` 是共享命名参考动作加载和全身跟踪 MDP；`legged_lab/utils/rsl_rl_compat.py` 是共享 IsaacLab/RSL-RL 观测适配。
- G1 关节序：`legged_lab/assets/unitree_g1/constants.py::G1_29DOF_JOINT_NAMES`；Z2 关节序：`legged_lab/assets/z2/constants.py::Z2_29DOF_JOINT_NAMES`。Z2 策略序与 URDF 出现序不同。
- T4 关节顺序唯一真值：`legged_lab/assets/t4/constants.py::T4_JOINT_NAMES`。
- 任务注册入口：`legged_lab/envs/__init__.py`。已注册 `t4_loco_teacher`、`t4_loco_teacher_sparse`、`t4_loco_sparse_depth_student`、`t4_vault_mimic`、`t4_vault_skill`、`g1_loco_teacher`、`z2_loco_teacher`。Stage E 深度学生走独立训练脚本。
- T4 原始 motion：`legged_lab/envs/t4/datasets/motion_source/`，schema 为 `root_xyz + root_quat_xyzw + q27`；`motion_visualization/` 是 playback 中间数据，不是最终 AMP expert。
- G1 原始 motion：`legged_lab/envs/g1/datasets/motion_source/`，schema 为 `root_xyz + quat_xyzw + q29`；`legged_lab/scripts/generate_g1_amp_expert.py` 生成 70D expert。
- Z2 motion 来源、顺序和哈希以 `legged_lab/envs/z2/datasets/` 中 manifest 为准；不能把同为 70D 的 G1 数据作为 Z2 expert。
- 能力声明必须有 evaluator JSON、lineage manifest 和连续回放证据；reward、episode length、checkpoint 存在或 loss 下降不能替代行为验收。

## 环境与运行

- 安装须在目标 Isaac Lab Python 环境中执行：根目录 `python -m pip install -e .`；内置训练库 `python -m pip install -e ./rsl_rl`。
- nubot：`nubot@100.100.188.39`，入口 `bash scripts/nubot_run.sh <script.py> [args...]`；包装器配置 isaaclab 库路径并使用 Isaac Sim standalone Python。
- zhuoqun：`zhuoqun@100.95.109.48`，入口 `bash scripts/zhuoqun_run.sh <script.py> [args...]`。凭据不写入本文件，连接时使用已配置认证或用户提供的凭据。
- 本机正式 play 使用 `scripts/local_run.sh`；缺少运行时时按 `docs/runbooks/local-isaac-docker.md` 执行 `scripts/setup_local_isaac_docker.sh`。保持 Isaac Sim 5.1 + IsaacLab 2.1.0 + `t4-isaac-jammy:v2` 合同，不用 conda `env_isaaclab` 替代。
- 远程 Linux 的训练、长时间 GPU probe、批量 playback、evaluator 和 rollout 采集使用 tmux。Windows 本机长任务使用可独立存活的后台进程并记录日志与 PID；交互 viewer 可前台运行。

## 验证

按影响范围选择最窄有效检查；跨合同变更运行相关组合。纯文档改动检查差异、路径和命令一致性，无需启动训练或全量测试。

| 变更范围 | 检查入口 |
| --- | --- |
| T4 资产 / motion | `python -m pytest tests/test_t4_asset_migration.py` |
| G1 资产 / LAFAN AMP | `python -m pytest tests/test_g1_asset_contract.py` |
| Z2 资产 / AMP | `python scripts/fetch_z2_upstream_fixtures.py`（一次性，gitignored 上游克隆）后 `python -m pytest tests/test_z2_asset_contract.py` |
| 共享 locomotion / 机器人接入 | `python -m pytest tests/test_robot_neutral_locomotion.py tests/test_g1_motion_experiment.py` |
| 共享深度学生运行时 | `python -m pytest tests/test_robot_neutral_depth_env.py tests/test_t4_sparse_depth_student_gru_contract.py`（需要 torch） |
| 共享跟踪 / RSL 适配 | `python -m pytest tests/test_robot_neutral_tracking.py tests/test_t4_vault_rsl_rl_compat.py` |
| T4 观测 | `python -m pytest tests/test_t4_observation_contracts.py` |
| 全量本地回归 | `python -m pytest tests`（conftest 自动引导仓库路径；zl 部署/上游 fixture 缺失时显式 skip） |
| 稀疏奖励、监控、命令、地形或分布式日志 | 从 `docs/README.md` 的稀疏合同测试组合中选择受影响项；跨合同修改运行相关组合 |
| 格式与静态检查 | `pre-commit run --files <本次修改的文件>`；`pre-commit run --all-files` 用于需要全量检查时 |

## 维护

- Python 遵循 Black，line length 120；检查配置见 `.pre-commit-config.yaml`、`.flake8`。
- 完成后清理本任务创建且已无用途的临时脚本、文档与输出；仍用于复现、验收或恢复的证据保留在对应工件目录。不清理无关资产、数据或用户未跟踪文件。
- 通过相关验证的关键里程碑使用简洁中文 commit；提交前检查 `git status --short` 和暂存差异，仅纳入任务相关改动。

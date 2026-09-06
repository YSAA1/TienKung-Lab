# AGENTS.md

本仓库在 TienKung-Lab 上维护 T4 27DoF 与 Unitree G1 29DoF 运动训练。先读 `docs/README.md`，再读
`.harness/work_index.md`。不要把 `docs/archive/`、`docs/research/` 或
`.harness/state.md` 的旧段落当执行计划。

## Commands

| Command | Description |
| --- | --- |
| `pip install -e .` | 安装根包 `LeggedLab`，需要已安装 Isaac Lab 的 Python 环境 |
| `cd rsl_rl; pip install -e .` | 安装本仓库内置 `rsl_rl` |
| `python -m pytest tests/test_g1_asset_contract.py` | 纯 Python G1 29DoF 资产、越障任务与 LAFAN1 AMP 合同 |
| `python -m pytest tests/test_t4_asset_migration.py` | 纯 Python T4 资产与 motion 合同测试 |
| `python legged_lab/scripts/t4_csv_motion_conversion.py --input legged_lab/envs/t4/datasets/motion_source --output-dir legged_lab/envs/t4/datasets/motion_visualization --fps 30` | 重新生成 T4 motion visualization 文件 |
| `python -m legged_lab.scripts.render_t4_motions --output-dir artifacts/motion_review` | 用 MuJoCo 把 motion 渲染成三视角回放视频与足底接触指标，供人工复核；加 `--metrics-only` 只刷新指标，加 `--interactive` 开交互 viewer |
| `pre-commit run --all-files` | 运行格式化和静态检查 |

## Architecture

- `legged_lab/` - IsaacLab locomotion 环境、资产、脚本和 MDP 代码。
- `legged_lab/locomotion/` - 共享 AMP/LightLP 教师、深度学生运行时、观测、镜像与课程算法。机器人通过显式 `LocomotionRobotSpec` 接入，不继承另一机器人的任务实现；接入方法见 `docs/runbooks/robot-locomotion-adapter.md`。
- `legged_lab/config.py` - 共享场景/命令/事件配置字段，导入不触发机器人任务注册。
- `legged_lab/assets/t4/` - T4 27DoF 资产；`constants.py::T4_JOINT_NAMES` 是唯一关节顺序真值。
- `legged_lab/envs/t4/` - T4 任务配方、相机标定及兼容入口。已注册 `t4_loco_teacher`、`t4_loco_teacher_sparse`、`t4_loco_sparse_depth_student`、`t4_vault_mimic`、`t4_vault_skill`。`envs/g1/teacher_cfg.py` 独立组合共享 `LocomotionEnv`、LightLP 与 LAFAN1 AMP，任务名 `g1_loco_teacher`。Stage E 深度学生仍走独立 train 脚本，不改任务名。
- `legged_lab/envs/t4/datasets/motion_source/` - 原始 T4 CSV，schema 为 `root_xyz(3) + root_quat_xyzw(4) + q27`。
- `legged_lab/envs/g1/datasets/motion_source/` - LAFAN1 G1 走跑 CSV（`lvhaidong/LAFAN1_Retargeting_Dataset` 镜像），`root_xyz + quat_xyzw + q29`。专家由 `generate_g1_amp_expert.py` 写成 70D txt。
- `legged_lab/envs/t4/datasets/motion_visualization/` - 转换后的 playback 中间数据，不是最终 AMP expert。
- `rsl_rl/` - 仓库内置 RSL-RL 训练库。
- `docs/README.md` - T4 文档入口。`docs/specs/` 是合同，`docs/plans/` **只放现行执行计划**。
- `.harness/` - 工作面索引与短状态；详细配方在 `docs/plans/`。


## Testing And Verification

- 代码改动后先跑最窄测试；T4 资产/motion 合同优先跑 `python -m pytest tests/test_t4_asset_migration.py`。
- G1 资产 / LAFAN AMP 合同跑 `python -m pytest tests/test_g1_asset_contract.py`。
- 共享算法边界与不同关节数接入跑 `python -m pytest tests/test_robot_neutral_locomotion.py tests/test_robot_neutral_depth_env.py`（需要 torch）。
- T4 观测合同改动跑 `python -m pytest tests/test_t4_observation_contracts.py`。
- 稀疏奖励 / 终止 / 地形列映射 / S12 命令与边框合同改动跑 `python -m pytest tests/test_t4_sparse_reward_contracts.py tests/test_t4_sparse_monitor_contract.py tests/test_t4_sparse_command_contract.py tests/test_t4_terrain_column_map.py tests/test_t4_stepping_stone_contracts.py tests/test_distributed_log_reduce.py`。
- 梅花桩 GRU 学生合同改动跑 `python -m pytest tests/test_t4_sparse_depth_student_gru_contract.py`（需要 torch）。
- 所有训练、GPU probe、批量 playback、evaluator、teacher rollout 采集等长运行命令必须用 tmux。
- 不能用 reward、episode length、checkpoint 存在或 loss 下降替代行为验收；能力声明必须有 evaluator JSON、lineage manifest 和连续回放证据。

## 服务器信息

nubot@100.100.188.39 密码 一个空格 四个GPU
zhuoqun@100.95.109.48 密码123456 四个GPU

Isaac 运行时（nubot）：`bash scripts/nubot_run.sh <script.py> [args...]`，它会激活 conda env
`isaaclab` 的 lib 路径并用 isaac-sim standalone 的 `python.sh` 启动。

本机 play 拉下来的 checkpoint：先 `bash scripts/setup_local_isaac_docker.sh`（国内镜像 / aria2c 拉 Isaac Sim 5.1 + IsaacLab 2.1.0，编 `t4-isaac-jammy:v2`），再用
`scripts/local_run.sh legged_lab/scripts/play.py --task t4_vault_mimic --num_envs=1 --load_run <run> --checkpoint model_*.pt`。
不要用 conda `env_isaaclab`（IsaacLab 2.3.2 / torch 2.7）当正式 play。说明见 `docs/runbooks/local-isaac-docker.md`。

## Style

- Python 格式遵循 Black，line length 120；静态检查由 `.pre-commit-config.yaml` 和 `.flake8` 定义。
- 修改要小而准，优先沿用现有 IsaacLab / LeggedLab / RSL-RL 模式。
- 不要改动无关资产、数据文件或用户未跟踪文件。

## Git Workflow

- 项目维护使用 git；关键里程碑提交中文 commit message。
- 提交前检查 `git status --short`，避免把无关用户改动混入提交。

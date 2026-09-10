# Plan - G1 VITAL v3：curated AMP + ramp 式 plant DR（30k 冷启动）

Spec: `docs/specs/2026-09-11--g1-vital-v3-curated-amp-ramped-dr.md`

## Active slice

v5→v6 AMP 数据决策落地、`vital_v3` 配方实现与合同测试、nubot Isaac 验证、GPU0/2 开训、对抗审查。

## 阶段

### 阶段 1：数据验收与 v6 生成（已完成 2026-09-11）

- v5 T-pose 证实：前 60 帧 |肩外展|>0.6 rad 占比 92–100%（walk1–4、run1–2 全段）；MCP 视觉分析确认为 T-pose 标定帧。
- v6 由 `curate_g1_amp.py` 生成：8 段（walk2×1、walk4×5、run1×2），最早起始帧 2254，support p95 ≤ 0.36，sole 穿透 ≤ 6mm；walk1 全部 9 段因打滑被拒。
- 视觉验收：v6 walk/run 代表帧 MCP 通过（自然摆臂、无穿地/漂浮/自碰撞）。
- acceptance: `artifacts/g1_amp_acceptance_20260911/{arm_quantitative.json,v6_all.mp4,snapshots/}`

### 阶段 2：vital_v3 实现与本地合同（已完成 2026-09-11）

- `legged_lab/mdp/ramp.py` 纯函数；`events.py` 三个 ramp 事件；`motion_experiment.py` v3 profile；`train.py` choices。
- verification_commands: `PYTHONPATH=. python -m pytest tests/test_g1_motion_experiment.py tests/test_g1_dr_ramp.py tests/test_g1_asset_contract.py`；`pre-commit run --files`（black 过；flake8/pyupgrade 在本机 Python 3.13 下工具自身崩溃，非代码问题，以 black+line-length 为准）。

### 阶段 3：nubot Isaac 验证 v6（已完成 2026-09-11）

- 部署 v3 工作树：`rsync -a --exclude logs --exclude artifacts TienKung-Lab-g1-vital-v2-20260910/ → TienKung-Lab-g1-vital-v3-20260911/`，overlay 改动文件 tar。
- `validate_g1_curated_amp.py` 全过：8 段最大特征误差 4.9e-7（门槛 2e-4），产物 `artifacts/portability/g1_vital_v3_20260911/amp_v6_validation.json`（远端工作树内）。

### 阶段 4：GPU0/2 开训 30k（已完成 2026-09-11）

- 前两次启动失败及修复：① `set_time_lag` int64→int32；② `body_names=".*"` resolve 后为列表的守卫误拒（`covers_all_bodies`）。第三次启动成功。
- tmux `g1-vital-v3`（GPU0/2 各 ~8.8-9.1GB，v2 的 GPU1/3 不变）+ `g1-vital-v3-tb`（端口 8051）。
- 核验通过：`params/env.yaml` 三 ramp 事件 + delay (0,1) + velocity_range {}；`params/agent.yaml` 8 段 v6；model_0 存在；interval 事件已在 iter~90-125 首次成功触发（iter 223 零 traceback）。
- 对抗审查（subagent 静态审查）：无 blocking 发现；非阻塞建议（docstring 精确化、死区测试钉死、covers_all_bodies 抽取、脚本工作树断言、注释修正）已全部落实并同步远端。

### 阶段 5：训练期监控（后续会话）

- ~3000 iter 检查 tracking_mean 与地形等级 vs v1 轨迹；中止线见 Spec。
- 30k 后：Isaac evaluator（stones/pillars d0 与 d=1.0 vx=0.5）+ 本机 MuJoCo 探针同命令对照。

## Non-goals

不做 MJCF 脚部几何对齐、不做真机 DR（编码器偏置）、不动 v2/Z2 工作树。

## Fallback evidence

若 GPU0/2 被占或 Isaac 验证失败：保留本地合同与部署产物，记录 progress，不占用 GPU1/3。

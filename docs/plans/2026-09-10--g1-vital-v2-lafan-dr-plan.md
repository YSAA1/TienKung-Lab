# G1 VITAL v2：LAFAN AMP + plant DR 开训

Status: active

Spec: `docs/specs/2026-09-10--g1-vital-v2-lafan-dr.md`（用户确认推荐方案并要求直接执行）

## Objective

把 G1 教师从 T4 rob2rob AMP 换成 LAFAN `unitree_v5`，加上训练期一小包 plant DR，在 nubot GPU1/3 冷启动 30000 更新。

## Active slice

实现 `vital_v2` 配方与 LAFAN 训练钉扎，跑合同测试，在独立远端目录启动 30k；不宣称行走能力。

## Non-goals

- 不重做 T4 IK；不改 T4/Z2 默认 DR；不占 GPU0/2；不改稀疏 0.6 / 地形几何；不开 G1 学生。

## Success criteria

- 单元测试证明 `vital_v2` DR 数值与 `vital_v1` 不变部分。
- 训练 manifest 6 段 LAFAN 哈希匹配文件。
- 远端新目录冷启动，GPU1/3，env.yaml 指向 `unitree_v5` 且 `action_delay.enable` 为 true；model_0 落地。

## Verification path

`python -m pytest tests/test_g1_motion_experiment.py tests/test_g1_asset_contract.py`；`pre-commit run --files` 改动文件；nubot `nvidia-smi` + tmux + 保存的 yaml。

## Verification path status

runnable（CPU 合同）；Isaac 开训依赖 nubot，本机无 Isaac。

## Required capabilities

nubot SSH、Isaac standalone、GPU1/3 空闲、本机 pytest。

## Fallback evidence

若远端暂时不能 overlay 整树：至少完成本地合同与启动脚本；开训失败记入 progress，不占用 Z2 GPU。

## Final integration claim / `final_integration_claim`

`vital_v2` 成为当前 G1 教师执行入口；VITAL `model_29999` 仅作对照。行为验收不属于本 claim。

## 工作项

- [x] 阶段 0：规格与数值（已确认）
  - acceptance_criteria: 用户选 B 并接受推荐 DR 表
  - verification_commands: 对话记录；Spec 文头 user-approved
  - success_definition: 开训配方不再有未决数值
- [x] 阶段 1：配方、测试与启动脚本（当前）
  - acceptance_criteria: `vital_v2` 改变 delay/摩擦/reset 速度/增益；`vital_v1` 不改 delay；LAFAN clips 哈希钉死；`teacher_cfg.py` 默认 delay 仍 False
  - verification_commands: `python -m pytest tests/test_g1_motion_experiment.py tests/test_g1_asset_contract.py` → 29 passed
  - success_definition: 无 Isaac 即可证明配方，启动命令指向 unitree_v5
- [x] 阶段 2：nubot 独立目录冷启动 30k
  - acceptance_criteria: 新工作树；tmux 使用 GPU1/3；未写 GPU0/2；model_0；env.yaml AMP 非 full17
  - verification_commands: 远端 `nvidia-smi`、tmux `g1-vital-v2`、`params/env.yaml`、`params/agent.yaml`
  - success_definition: 训练在跑且合同与 Spec 一致
- [ ] 阶段 3：30k 后行为对照（下一步，本切片不阻塞开训）
  - acceptance_criteria: Isaac vx=0.7 easy 踏石/圆桩 JSON+回放；本机 MuJoCo 同速；平地手臂相对 VITAL 不更夸张
  - verification_commands: 固定 evaluator + `play_t4_sparse_teacher_mujoco.py --robot g1`
  - success_definition: 有 fresh 行为证据，不是 TB

## Commit units

1. 配方/测试/启动脚本/文档：测试绿之后可提交（用户未要求则先不 commit）。
2. 开训工件：只入 `artifacts/` 与 `.harness/`，不含 checkpoint 大文件。

## Known risks / blockers

- 冷启动 delay+增益历史上伤过 G1（v6）；用户接受该风险。
- Z2 占用 GPU0/2；若 GPU1/3 被占则等待，不抢 Z2。

## Next skill

implement

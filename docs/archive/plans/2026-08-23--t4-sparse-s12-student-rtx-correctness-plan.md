# Executable Plan - S12 学生 headless RTX 出图与评估入口

> Status: done
> Successor: `docs/archive/plans/2026-08-23--t4-sparse-s12-student-lightlp-distill-cost-plan.md`（蒸馏成本切片）
> Date: 2026-08-23
> Parent: `docs/plans/2026-08-22--t4-sparse-s12-rim-yaw-student-plan.md`
> Spec: `docs/specs/2026-08-12--t4-unified-depth-locomotion.md`
> Branch: `t4-train`
> Planning surface: docs plan

## Objective

把 S12 GRU 深度学生从「反复读陈旧 Camera buffer」改成「headless 下按 0.06 s 真出 RTX 深度」，并让 train / play / eval 走同一条渲染合同。修完后开新 logdir 蒸馏；不 resume nansync / holdcap / 空转 16.7 Hz checkpoint。

## Active slice

阶段 1–3 已落地：headless RTX 调度 + checksum probe 绿 + nubot `s12_gru_ppo_rtx167` 已开训。能力验收仍属 S12 计划阶段 4 后半。

## Non-goals

- 不把 GPU 利用率或 s/iter 写成成功标准。
- 不改普通 Camera、不预先砍到 512/卡；真渲 OOM 再降 env/分辨率。
- 不重写蒸馏算法，不改 S12 老师 MDP。
- 不 resume `model_3500/4000`，不动 zhuoqun。
- 本切片不完成梅花桩能力验收。

## Success criteria

1. Headless + 深度相机时按 `update_period/physics_dt`（12 物理步）调用 `sim.render()`；老师无相机不 render。
2. Probe：raw depth checksum 只在 RTX tick 变化；非 tick 与上一帧相同；非全 invalid。
3. Reset 行在缺少传送后 RTX tick 时保持零历史。
4. `t4_loco_sparse_depth_student` 的 play 与 `eval_t4_hurdle.py` 打开相机。
5. 新 lineage：老师 `model_21500.pt` + warm-start `model_3000.pt`，新 run name。
6. 能力仍只认后续 evaluator；本修复完成 ≠ 会过桩。

## Verification path

```text
python -m pytest tests/test_t4_sparse_depth_student_gru_contract.py tests/test_safe_recurrent_distillation.py -q
nubot: probe_t4_student_depth_render.py --num_envs 16 --steps 24
新 lineage: train_t4_sparse_depth_student.py --task_num_envs 1024 --distributed --headless --run_name s12_gru_ppo_rtx167
```

### Verification path status

`local runnable / GPU probe passed / lineage restarted after probe-OOM; 13-13-19 running past iter 5`

## Required capabilities

本机 pytest；nubot 4×GPU + `scripts/nubot_run.sh` + tmux。

## Fallback evidence

Probe checksum 仍常数：先查相机/`disable_visual_assets`，不准靠加大 ingest 假装在看。真渲 OOM：报 blocker，不先换 Camera 后端。

## Final integration claim

`final_integration_claim`: 交付 headless T4 学生按 16.7 Hz 生成新 RTX 深度，并堵住 play/eval 关相机；然后用新 logdir 重蒸。不声明过桩能力或墙钟改善。

## 工作项

- [x] 阶段 1：T4 headless RTX 调度 + reset 陈旧 tile + play/eval 开相机
  - acceptance_criteria: 物理循环内、`scene.update` 前按 interval `sim.render()`；reset 行缺 post-reset tick 时不写入；play/eval 对学生 enable_cameras
  - verification_commands: `python -m pytest tests/test_t4_sparse_depth_student_gru_contract.py tests/test_safe_recurrent_distillation.py -q`
  - success_definition: 代码合同保证 headless 学生会调度 RTX
- [x] 阶段 2：停 nansync，nubot checksum probe
  - acceptance_criteria: 16–32 env 只在 16.7 Hz tick 出新画面
  - verification_commands: `bash scripts/nubot_run.sh legged_lab/scripts/probe_t4_student_depth_render.py --headless --num_envs 16 --steps 24 --output artifacts/diagnostics/s12_student_rtx_probe.json`
  - success_definition: JSON `ok: true`
- [x] 阶段 3：新正确性 lineage 开训
  - acceptance_criteria: 新 tmux/logdir；`--task_num_envs 1024 --run_name s12_gru_ppo_rtx167`
  - verification_commands: 开训 log 出现 Learning iteration
  - success_definition: 学生第一次在活深度上蒸馏
- [x] 阶段 0：问题已定位
  - acceptance_criteria: 50 Hz vs 16.7 Hz 墙钟差约 2%；根因是 headless 不 render
  - verification_commands: holdcap/nansync 对照
  - success_definition: 不再把降相机频率当加速手段

## Commit units

1. `student-headless-rtx-schedule`：阶段 1 代码 + 测试 + 本计划 + `.harness`。训练产物不进 git。

## Known risks / blockers

真 16.7 Hz tiled 4096 可能比 5.3 s/iter 更慢。`model_3000` CNN 见过的是盲深度。不要上传 Windows `scripts/nubot_run.sh`。

## Next skill

阶段 1 本地绿灯后 probe；probe 绿开阶段 3。Ready 只交给 review。

# Current State

- Planning surface: `docs/plans/2026-08-12--t4-unified-depth-locomotion-plan.md`
- Approved Spec: `docs/specs/2026-08-12--t4-unified-depth-locomotion.md`
- Active item: M1/M2 T4 观测合同与 Stage E teacher 任务落地（M0 已全部通过，含人工视觉复核）
- Verification path: `python -m pytest tests/test_t4_observation_contracts.py` 在任意机器可跑；IsaacLab/tmux gates verified on target nubot Linux GPU runtime via `scripts/nubot_run.sh`.
- Next skill: `verify`
- Long-running rule: 所有训练、GPU probe、批量 playback 和 evaluator 必须在 tmux 中运行。
- Stop gate: M0 人工视觉复核已于 2026-08-12 通过，正式 AMP expert 与 Stage E 正式 lineage 解锁；后续 gate 变为 teacher evaluator 通过前不启动 student 蒸馏。
- Frozen contracts: `legged_lab/assets/t4/schemas.py` 冻结 AMP 66D、teacher 前向不对称 scan（1.4x1.2m @0.1，offset x=0.9，前向 0.2-1.6m，15x13=195 维）、depth 预处理与 proprio 96 维/10 帧；teacher actor obs 1155 维。
- Pending MDP change (2026-08-13, 未提交): 对照成功参照 VITAL_Lab T4_27 做了四项变更，任何一项都要求开新 lineage、从零训练，不得从 prov3/prov4 checkpoint 续：
  1. `legged_lab/assets/t4/t4.py`: 踝关节增益 10/0.5 → pitch 80/4、roll 20/1（旧值近似被动踝，无真值来源；VITAL 同机器人训成功的值），并打开 self-collisions。
  2. `teacher_cfg.py` 奖励再平衡: track_lin 1.0→2.0、lin_vel_z -1.0→-0.15、删 hip_roll/yaw_action(-1.0)、Shank 接触从 undesired_contacts(-1.0) 拆出为 -0.3、新增 foot_touchdown_impact(-0.08, `mdp.foot_touchdown_impact_penalty`)。
  3. 周期步态奖励保留（bf06c0a 的 tracking 门控不变）。
  4. 对称性: AMPPPO `symmetry_cfg`（augmentation + mirror loss 5.0），镜像计划 `legged_lab/envs/t4/symmetry.py`（arm 01-07 符号 +--+-+-，来自 MJCF 轴向；scan 沿 y 翻转），合同测试在 `tests/test_t4_observation_contracts.py`。
- Stale lineage: `stage_e_prov4`（从 `stage_e_prov3` 的 `model_2000.pt` 续训，nubot tmux `t4-stage-e`，commit `bf06c0a`，4x RTX 4090，每卡 1024 env，日志 `logs/t4-stage-e-teacher.log`）——诊断结论：课程钉 0 的根因是基础行走能力（降到 level 0 平地后 episode_max_radial_dist 均值仍 ≤1.38m < promote 4m），而非课程代数；见上面的 MDP 变更。
- Terminated lineage: `stage_e_prov3`（commit `38a3e12`）于 iter ~2248 停止续写：TB 显示 mean_reward 平台 55–59、terrain_levels 从 ~350 钉 0、episode_max_radial_dist 全程 ≤1.38m、lin tracking 在 0.3–0.7 波动。`stage_e_prov2`/`prov1` 见下。旧日志 `logs/t4-stage-e-teacher.stage_e_prov3.log`。
- Gait fix（commit `bf06c0a`）: 周期步态奖励改为 `moving * exp(-||v_cmd-v_act||^2 / 0.5^2)`（与 track_lin_vel_xy_exp 同核），站立仍为 0 并冻结时钟；日志 `Curriculum/gait_tracking_scale`。高速命令蹲着时步态分应接近 0。
- Curriculum watch: 续训后看 `gait_tracking_scale` 是否明显低于原先 ~0.42 的命令速度缩放、径向是否离开 1.2m、lin tracking 是否随 gait 被关掉而上升。
- Lineage caveat: `stage_e_prov1`/`stage_e_prov2`/`stage_e_prov3`/`stage_e_prov4` 启动时使用 provisional AMP expert（`artifacts/amp_expert_provisional/_manifest.json` 中 `human_playback_review=pending`）。人工复核已通过，且 66D feature 与 root 绝对高度无关，因此该 expert 数值上等同于正式 expert；提升为 formal 需要先比对 provisional manifest 的 clip 列表与 accept 清单一致，再写入 `legged_lab/envs/t4/datasets/motion_amp_expert` 并补 lineage 记录，未完成前仍按 provisional 引用。
- Stage E surface: 任务 `t4_loco_teacher`（`legged_lab/envs/t4/t4_env.py` + `teacher_cfg.py`），地形 `T4_STAGE_E_TERRAINS_CFG` 含上行/下行楼梯，AMP 系数按 terrain difficulty 线性衰减，gait 模式旋钮 `fixed_clock|command_conditioned|difficulty_relaxed`。
- Latest M0 evidence: `artifacts/eval/t4_motion_audit.json` machine-audited 18 motions; 17 accepted, `t4_run` rejected for hard joint limit violations plus holdout rule.
- Latest simulator evidence: nubot IsaacLab spawn smoke loaded T4 with 27 joints and 30 bodies; joint name set matches `T4_JOINT_NAMES`, runtime order differs and playback reorders by name.
- Latest playback evidence: `artifacts/eval/t4_motion_playback_smoke.json` simulated `t4_stand` 5 frames with 0 rejects; `artifacts/eval/t4_motion_playback.json` simulated all 18 motions with 0 rejects.
- Human playback verdict (2026-08-12, 通过): 证据为 `artifacts/motion_review/`（18 条 MP4，每帧含前 3/4、侧视、足部特写三视角，另有 worst-frame 静帧与 `_manifest.json`），渲染入口 `python -m legged_lab.scripts.render_t4_motions`（MuJoCo `t4_std.xml`，30fps）。裁定 accept 17 条，`t4_run` 继续 held out。
- Accepted known defect: 除 `t4_stand`（sole +1.4mm）外全部 clip 未对齐地面——stance sole p05 在 `-57.6mm`(B4) 到 `-11.1mm`(t4_left_rotate) 之间，`t4_jog_backward` 整段悬空 `+43.1mm`；按 p05 归一后 contact frame fraction 仅 0.07-0.15，说明这些 clip 的接触时序不可用。66D AMP frame（`q27+dq27+hands_root6+feet_root6`）全部来自关节 FK 与 root 相对量，不含绝对 root 高度，故该缺陷不改变 expert 数值；任何消费绝对 root 高度或 motion 接触相位的逻辑都不得引用这些 clip。
- Evidence caveat: `artifacts/eval/t4_motion_audit.json` 与 `t4_motion_playback.json` 中的 `human_playback_status` 由脚本生成、恒为 `pending`，人工裁定以本文件与 plan M0 工作项为准。
- Current blocker: 无 M0 阻塞。
- New M0 simulator entrypoint: `legged_lab/scripts/playback_t4_motions.py` writes raw T4 frames into the IsaacLab T4 articulation and emits `artifacts/eval/t4_motion_playback.json` in the target nubot runtime.
- Nubot target: `nubot@100.100.188.39:/home/nubot/phn_ws/t4_train/TienKung-Lab`; M0 headless spawn/playback evidence was collected on commit `dfb69d1` (`fix(t4): 按名称重排动作关节`).
- Nubot runtime: use `/home/nubot/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh` with IsaacLab source paths in `PYTHONPATH`; torch verified as `2.5.1+cu124`, CUDA visible on 4 GPUs.
- Nubot caveat: GitHub fetch can fail with `GnuTLS recv error (-110)`; latest commits were synced through local git bundles over SSH when needed.

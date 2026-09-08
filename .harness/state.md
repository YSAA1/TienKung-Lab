# Current State

- 2026-09-08 核心修复计划：用户仅要求制定方案，尚未实施或改动训练。唯一入口仍为 `docs/plans/2026-09-07--robot-neutral-locomotion.md` 顶部；范围为足速度跨reset、timeout残留、terminal bootstrap、std下限、standing heading、AMP时间采样，加必需传感器缺失检查和依赖溯源。Actor不输入真实根速度是正常设计，不纳修复或默认实验。修复前真实Isaac/CPU/导入证据已保存到 `artifacts/portability/g1_core_fixes_20260908/evidence/`；现有A/B冻结保留。旧短probe通过记录不能覆盖本次已实测发现的缺陷。

- 2026-09-08：用户授权四卡全速 A/B，代码提交 `2f99efb`。共同冻结目录 `/home/nubot/phn_ws/t4_train/TienKung-Lab-g1-progress-ab-20260908`；A GPU0/2、tmux `g1-progress-A`，原路径晋级；B GPU1/3、tmux `g1-progress-B`，相对tile中心最大径向距离>4m晋级。速度缩放均1.0、全17段、权重2、recovery终止、每卡2048env、24steps、seed42+rank、冷启动30000，TB8041。120项CPU检查、独立review、32env真实Isaac命令/部分reset/晋级probe及双卡2更新smoke通过。旧full17/recovery四rank已退出，保留model_18500/model_12500及全部历史日志。启动配置/进程/标量证据见 `artifacts/portability/g1_progress_ab_20260908/`；本状态不声明学习改善。

- 2026-09-07 23:30：G1恢复合同对照已启动。GPU0/2保留full17权重2基线；原GPU1/3权重4已停、6个checkpoint保留（model_2500）。新tmux `g1-recovery-30k`，冻结目录 `/home/nubot/phn_ws/t4_train/TienKung-Lab-g1-recovery-20260907`，主run `logs/g1_recovery_30k/2026-09-07_23-26-29_g1_recovery_30k`，TB8040，rank PID920838/920839。连续塌低.20s且无immunity才触发collapsed；线速度2、全17段、每卡2048env、冷启动30000。56项检查、真实Isaac和独立review通过。固定旧checkpoint探针仅证明合同正确，仍站立、未见稳定恢复，不宣称学习改善。证据 `artifacts/portability/g1_recovery_20260907/`，执行以当前plan顶部为准。

- 2026-09-07 21:13：新增全17段线速度权重4对照，GPU1/3，tmux `g1-full17-lin4`，冷启动30000轮。旧v6已停止，保留18个checkpoint；GPU0/2权重2基线继续。合并TB8039，配置唯一行为差异已核实，新组20轮/model_0/双rank数值健康。证据 `artifacts/portability/g1_full17_lin4_20260907/`。

- 2026-09-07：用户取消对照，正式启动完整17段T4→G1 rob2rob、双卡30000轮。GPU0/2、每卡2048env、全局4096env、冷启动；17段3303帧/110.1秒及原逐文件权重完整保留，Isaac逐帧验证通过。
- Living index: `docs/README.md`
- 当前 G1 唯一执行计划：`docs/plans/2026-09-07--robot-neutral-locomotion.md`。远端 `TienKung-Lab-g1-unitree-v6-20260907`，tmux `g1-full17-30k`，TB8038；日志根 `logs/g1_full17_30k/`，启动工件 `artifacts/portability/g1_full17_30k_20260907/`。
- GPU0旧数据control、GPU2两段组及仅初始化的全17段4000轮组均已停止，工件保留；旧v6现已停止，TB8035保留历史，GPU1/3改跑上方权重4对照。数据回放 `artifacts/portability/t4_rob2rob_full17_v1/all_clips.mp4`；数据验收及更新不代表策略能力成功。
- 旧 v2 按用户要求在 iteration 9662 停止：tmux `g1-full-levels-v2` 与原 5 个训练进程全部退出，GPU 1/3 释放，20 个 checkpoint 保留；TB 8031 仍可查旧曲线。停止证据 `artifacts/portability/v3/stopped_v2.json`。共享教师、深度学生、动作跟踪与全库接入整理已完成；各候选行为能力仍由固定评估和连续回放判定。
- S12 学生 `s12_repr_first` `model_13999` Isaac hard 过门，MuJoCo 残留交给老师 plant 重训。老师本机 `artifacts/checkpoints/nubot/s12_teacher/model_21500.pt`。
- Approved specs: `docs/specs/2026-08-12--t4-unified-depth-locomotion.md`（走跑/学生合同）、`docs/specs/2026-08-13--t4-vault-loco-merge.md`（G1→G2→G3 目标）、`docs/specs/2026-08-15--t4-stepping-stones-and-hurdle-stable.md`（梅花桩目标；执行已改单阶段）、`docs/specs/2026-09-01--t4-s12-repr-first-distill.md`（表示先行蒸馏）

## 梅花桩（nubot）

- 任务 `t4_loco_teacher_sparse`。S12 老师 worktree `/home/nubot/phn_ws/t4_train/TienKung-Lab-s12-from-s11b-5k`，run `t_sparse_lightlp_s12_from_s11b_5k`，从 S11b `model_19000` 加载后冻在 `model_21500.pt`。
- 配方：0.75 m 收尾边框 + 踏石/圆桩 40% 轻转；`vx` 非洞 `[-0.6, 2.0]`，洞上 `[0.6, 2.0]`。终止阈值不改。`random_level_reset_max_level=None` 是 S11 起的老师课表，学生继承，不是 v3 单变量差。
- 学生 `fixed-v3-nanguard` 已在 iteration 3305 左右策略坍塌；失败取证链保留，不得续 `model_3500/4000`。TB 8017 进程仍活。
- 学生 lineage：worktree `TienKung-Lab-s12-gru-ppo`。**保留 D1** `s12_lightlp_dagger_only/model_10000.pt` 作对照。D3/`model_10500`、D3b、D3c 都不续。mix0 仍不 FT。
- **学生最终状态（2026-08-29）：** Phase B `s12_rtx_gated_joint/model_5999.pt` 已完成。修复 evaluator 的 GRU live hidden 污染与 episode reset 后，easy 踏石/圆桩 strict `32/32`、`29/32`，hard `18/32`、`27/32`，hard reach_2m `28/32`、`30/32`。两类 hard 连续回放、lineage 与 deploy-only 包齐全；3168-D 本机 dummy inference 输出 27-D finite action，GRU reset 通过。Phase C 跳过。
- **蒸馏成本切片墙钟已达标；质量未过。** 旧对照 `s12_lightlp_raycast` 已停在 `model_14000.pt`。近窗 collection p50 2.185 s / total p50 2.320 s。**不是** v3 坍塌。成本计划已归档。
- S11b 已停（`model_19000.pt`）。S10/S9/S6–S8 只读对照。S7/S8 是污染 plant，不续训。
- Stage E `t4_loco_teacher` 1155D 未改。能力声明要等 evaluator + 回放；TB success / length→1000 不算。

## 翻箱（zhuoqun）

- 执行面仍是 G1/G2 recovery。G3 / 与走跑合并被 G2 过箱挡住。
- 不要动 nubot 梅花桩 checkout 去盖 zhuoqun 翻箱。

## 已关闭（不要当待办）

- Stage E 走跑 + 跨栏 bucket + 深度学生 `model_24999` / 部署候选 `model_25746`。
- 走跑完成 ≠ 100m `rule`、真机障碍、梅花桩、翻箱 G3。

## 验证

- 本机：`python -m pytest tests/test_t4_sparse_reward_contracts.py tests/test_t4_sparse_monitor_contract.py tests/test_t4_sparse_command_contract.py tests/test_t4_terrain_column_map.py tests/test_t4_stepping_stone_contracts.py tests/test_distributed_log_reduce.py tests/test_t4_sparse_evaluator_contract.py -q`
- 梅花桩 GRU 学生：`python -m pytest tests/test_t4_sparse_depth_student_gru_contract.py`（需要 torch）
- nubot：`scripts/nubot_run.sh`；zhuoqun：`scripts/zhuoqun_run.sh`
- 长任务一律 tmux。

## deferred_cleanup

- `docs/research/*` 与 `docs/archive/plans/` 历史正文不重写。
- `artifacts/eval/*`、视频、checkpoints：对照证据，未逐项核对 lineage，不删。
- `artifacts/diagnostics/*`：一次性 probe/viewer/JSON；正式评估走 `eval_locomotion.py`，旧 `eval_t4_hurdle.py` 为兼容入口。
- `scripts/setup_local_isaac_docker.sh`：会话前已有本地改动，不混入提交。
- `legged_lab/envs/t4/depth_student_cfg.py` 多代 AlgCfg（Dagger/Joint/DeployFt/TargetedFt/ResidualFt/PlantFt）+ `train_t4_sparse_depth_student_ft.py` 多 `--mode`：失败 lineage 对照与合同测试仍引用。删会改行为。
  reason: 不是未引用死代码，是叠代配方。
  reevaluate_when: 表示先行 Spec 批准并落地新 AlgCfg 之后，再单独 implement 切片收口旧 mode。
- pytest cache / 未跟踪 ckpt：不动。
- 本次独立 reviewer 的临时目录 `C:/Users/shash/AppData/Local/Temp/codex-tracking-review-20260907-a`：自动批准审查拒绝其删除，理由 `blocked by policy`。暂留；不影响工作树与训练，无替代工具删除尝试。

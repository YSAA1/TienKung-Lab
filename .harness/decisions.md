# Decisions

- 2026-08-18：s4 不作梅花桩证据。P0 是 `terrain_types` 列号被当成 sub_terrain 下标。先修映射与 LightLP 10% 随机 level / Eq.4–5 / opposite / 路径晋级，再从零开 s5。不在 s4 ckpt 上热补；不改 accel 40、不打开 algebraic。
- 2026-08-18：文档收口。现行只有两条执行轨道（梅花桩、翻箱 G1/G2）。入口改为 `docs/README.md`。撤回梅花桩软/硬二阶段，单阶段真洞。G2 学生与 G3 合并等过箱后再开。
- 2026-08-17：梅花桩完整补全改从零。不续 `v2_resume`；不做旧学生短 FT；踏石必须收到 illegal 能咬偏脚；本任务 Actor 1155D 可打破（scan history=5 + 接触进 Actor，足底 scan 仍只进 Critic）。默认 Stage E 1155D 不动。**软→硬二阶段已于 2026-08-18 撤回。** 执行面 `docs/plans/2026-08-17--t4-sparse-lightlp-complete-plan.md`。
- 2026-08-17：S1d 稳定组/对照组判定失败并回退。证据：v2 @3874 踏石 `reach_2m=0.80`、圆桩≈0；A/B @11000 全局 reward 52–62、length 860–920，两任务目视未成功。一次叠步态/stumble/AMP/脚下 scan/双 Critic/掉坑免 −200，还杀掉正在涨的 v2。**续训恢复 v2 的执行面已被同日「从零补全」取代。**
- 2026-08-16：开稳定组/对照组，不再只赌几何。BeamDojo「拆开」= 两个价值网络分别估走跑奖励和落足奖励，不是再加一项。稳定组只做奖励侧拆开（稀疏 tile 步态/stumble=0、AMP×0.1）+ Critic 脚下 scan，保留掉坑 −200。对照组再加掉坑不吃 −200、AMP=0、双 Critic。1155D 不动。**已被 2026-08-17 否决。**
- 2026-08-16：圆桩/梅花桩失败不是「前向 scan 看不见支撑」。旧 25k 平地 4/4，T-paper 接触标志反而摔得更多。v2 @2.1k 已把踏石 easy `reach_2m` 拉到 ~0.79，但圆桩 `reach_2m≈0`、progress≈1.0 m、`illegal_footstep≈0`。根因是圆盘落点 + 出生台到第一桩 35–60 cm 空隙；缺的是可迈出去的第一跨和「踩在抬高支撑上」的正奖励，不是再给 Actor 加特权。T-compat v2 继续跑踏石；T-paper 停掉，改用同 1155D 开 v3 圆桩探针（更密圆桩间距 + `legal_foothold`）。
- 2026-08-15：梅花桩 = LightLP (5) 桩阵 + (6) 踏石，不是绕桩/窄板。双 teacher 从零并行：T-compat 1155D、T-paper Actor +接触 1157D。旧学生保留，本机 FT 0.30。illegal-footstep / slack 只进 reward；足底 scan 不上 Actor/实机。nubot 2+2，本机 1，zhuoqun 不动。
- 2026-08-15：不续训 G2 `2026-08-14_05-30-51_g2_from_g1_m28500`。根因是跟踪教师标签在学生 1155D 观测下不是函数，且实现只有学生开车的 MSE，没有教师开车 BC / PPO。
- 2026-08-15：不开 G3。硬门仍是可用 G2 + Stage E teacher evaluator。
- 2026-08-15：先补 G1 从第 0 帧过箱。开 G2 的训练门槛是同一 evaluator ≥80/100；Spec 95% 仍是对外宣称门槛。
- 2026-08-15：G1 先从 m28500 改 `start_window` 续训，5k 后箱前超时仍 ≥25% 则 from-scratch。
- 2026-08-15：G2 重做必须先过教师开车 BC（R3），再 DAgger+PPO（R4）。
- 2026-08-15：Play/eval motion RSI 扰动必须为 0；训练扰动保持。旧 48/100 JSON 不得覆盖。
- 2026-08-15：旧 48/100 不是「36 次箱前站死」。评测按 16 env 一批且批次之间不 reset，成绩按 0、16、0、16 交替；16 次 step-0 手腕是冷启动 FK 未刷新。每批必须双 `reset`，中间丢弃一次零动作 step。
- 2026-08-15：`model_28500.pt` 独立 100-trial = 100/100。R2 起点偏向续训取消。G2 教师必须用这个 ckpt。
- 2026-08-15：G2 默认 `collect_mode=teacher`，关 `push_robot`，gait 6 维置零。教师开车 50 iter 后 behavior 0.043、length 436，1155D 足够克隆在轨翻箱动作。

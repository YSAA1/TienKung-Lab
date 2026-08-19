# 交接：T4 梅花桩 s5（2026-08-19）

> **已过期。** 现行执行：`docs/plans/2026-08-19--t4-sparse-easy-geometry-s6-plan.md`。下文是 s5 负结果快照，不要当开训指令。

本文是 **session 交接快照**，不是执行计划。开训 / 改配方仍只认：

- 入口：`docs/README.md`
- 梅花桩计划：`docs/plans/2026-08-19--t4-sparse-easy-geometry-s6-plan.md`
- 规格：`docs/specs/2026-08-15--t4-stepping-stones-and-hurdle-stable.md`
- 论文原文：`docs/research/2608.02653v1/auto/2608.02653v1.md`（只对照，不当计划）

不要用 reward、episode length、TB 曲线或 ckpt 存在宣称梅花桩能力。

---

## 一句话

`t_sparse_lightlp_s5` 在 nubot 四卡从零跑着（列号和 LightLP §IV 公式已落地），连续地形会走，**梅花桩卡在第一脚**：策略按平地走，脚不抬上台，掉进石缝，被 `accel>40` 杀掉。下一步应开新 lineage 改 **easy 几何**，不要再等 40k，也不要先改 accel / OOB / 蒸学生。

---

## 机器与进程

| 项 | 值 |
| --- | --- |
| 训练机 | `nubot@100.100.188.39`（密码：一个空格），仓库 `/home/nubot/phn_ws/t4_train/TienKung-Lab` |
| 启动 | `unset LD_LIBRARY_PATH` 后 `bash scripts/nubot_run.sh ...`（tmux 脏 `LD_LIBRARY_PATH` 会炸 `libcusparse`） |
| 任务 / run | `t4_loco_teacher_sparse` / `2026-08-19_00-20-19_t_sparse_lightlp_s5` |
| tmux | `t4-sparse-lightlp-s5`（训练，交接时仍在）；TB `t4-tb-sparse-lightlp-s5` → `http://100.100.188.39:8008` |
| 卡 | 四卡，约每卡 7 GB / 24 GB |
| 最新 ckpt（交接时） | `logs/t4_loco_teacher_sparse/2026-08-19_00-20-19_t_sparse_lightlp_s5/model_16000.pt` |
| AMP | `T4_AMP_EXPERT_DIR=artifacts/amp_expert_provisional` |
| 本机 Windows | **没有** Isaac Docker，不能正式 play |
| zhuoqun | 翻箱，**不要动** |
| Stage E 1155D | **冻结**，本切片不蒸学生 |

s4 已停，只当错位基线：`2026-08-18_16-01-39_t_sparse_lightlp_s4`，TB `:8007`，`model_12500.pt`。不要读 s4 的 `Terrain/stepping_stones` 当踏石（那是跨栏）。

---

## 代码合同（s5 已在跑的）

任务：`T4LocoSparseTeacherEnvCfg` + `T4LocoSparseTeacherAgentCfg`。

- Actor **1937D**：本体 10×96 + scan×5 + 脚接触 2。足底 scan 只进 Critic。
- Critic **2016D**：另加足底 30 + 免疫 1 bit。
- 列映射：`legged_lab/envs/t4/terrain_columns.py`（Isaac 2.1.0 列号，不是 `sub_terrains` 下标）。20 列：col5 跨栏，6–9 踏石，10–13 圆桩。
- LightLP：slack +1.5、Eq.4 双脚求和、Eq.5 `τ=0.06` `ā=30`、opposite 点积、路径长度晋级、10% 随机 level、10% 免疫每 200 step（与随机 level **不是**同一开关）。
- 终止：horizon 20 s、OOB Chebyshev 4.25 m（半格 4 m + 0.25）、关节 50 rad/s、躯干+臂 >1 N、accel 40（1 s 热身）、倾角 63° 且 p=0.01。掉洞只打日志，不杀局。
- 稀疏列：AMP=0、周期步态=0、stumble=0。连续列仍有 Stage E 正则和 AMP。
- `Reset/*` 分原因日志已进代码（`c848833` 一带），**当前 s5 进程启动早于这次改动，TB 上没有这些 tag**。要看见必须重启；不要为日志丢掉 16k。

本机相对 origin 超前若干提交（列映射 + MDP + s5 harness + 文档清理）。nubot checkout **很脏**（大量未提交/未跟踪），训练是从这个工作树起的，不要在 nubot 上乱 `git reset`。

`legged_lab/scripts/play.py` 本机已加：`--cam_eye` / `--cam_look`、reset 行日志、imageio 失败则 GIF、系统 ffmpeg 压 mp4。nubot 上的 `play.py` 已被 scp 成同一版。

---

## 论文对照（已扫完，未再改配方）

s5 对齐的是 LightLP **§IV 感知走跑老师**，不是整篇（§V 技能扩增 / §VI GRU 深度学生明确 OOS）。

已对齐且 TB 有指纹：Table I 核心项、Eq.4/5、opposite、路径晋级、10% 随机 level、10% 免疫、列 occupancy≈5%/列、稀疏合计 40%。

有意或未齐：20 列不是 32 列；无独木桥/轨道、有跨栏/wave；OOB 是过半格 0.25 m 不是提前 2 m；本体 10 帧 + 步态时钟；连续格 AMP；Table II DR 几乎没做。

---

## 指标（约 iter 14k，之后仍无梅花桩起势）

PPO 健康（entropy≈19.7，noise≈0.51）。平地 / 粗糙 / 楼梯 / 跨栏 `reach_2m` 约 0.60–0.67。

踏石 / 圆桩：`reach_1m≈0.60`，`reach_2m≈0.03`，进度 **≈1.1 m**，从 2k 到 14k **无上升**。easy / mid / hard 几乎一样。踏石 episodes 已约 19 万。`hard_episodes` 上万，10% 随机 level 在工作。

全局 `fall` 是混合桶（accel + 躯干 + 倾倒），**看不出死因**。这是此前误判「踩实被 40 误杀」的原因。

不要等 40k。全局 length≈470 是 OOB 4.25 m + 稀疏早死，不是要拧的旋钮。

---

## 回放结论（必须按这个理解卡点）

文件在本机（也在 nubot 同路径）：

- `artifacts/eval/s5_replay/s5_m14500_stones_d00.mp4`（easy 踏石，`model_14500`，20 s）
- `artifacts/eval/s5_replay/s5_m14500_pillars_d00.mp4`（easy 圆桩）
- 同目录 `.txt` 是每次 reset；`frames/` 里有拆帧

命令（nubot，`play.py --terrain --difficulty 0.0 --terrain_types stepping_stones|raised_pillars --record`）。出生垫视觉几乎看不见（薄平面 + 材质丢失），中间大方块像坑，其实是 1.6 m 垫子。

| 地形 | reset | peak | flags |
| --- | ---: | --- | --- |
| easy 踏石 | 3/3 | 1.19–1.38 m | **accel**，无 pit |
| easy 圆桩 | 6/6 | 1.11–1.57 m | **accel**，无 pit |

几何：easy 踏石顶 26 cm、间距 50 cm、**石间净空 24 cm**、高 16 cm；第一圈中心在出生点外 1.0 m。圆桩 easy 直径 50 cm、缝 5 cm、高 14 cm。

看帧（尤其 `frames/s5_m14500_stones_d00_180.png`）：

1. **第一脚没抬上台**，仍是贴地走跑步态，脚按地面高度伸进第一圈。
2. **URDF 只有左右脚底碰撞**（各 3 块约 1.5 cm 厚的盒子）。大腿 / 小腿 / 膝 / 躯干的 `<collision>` 全注释掉。脚没踩上时小腿穿过石头，整个人从缝里掉下去。
3. 掉下去磕石棱或砸坑底，加速度 > 40，记成 accel。自由落体只有 9.8，单靠坠落触发不了 40。pit 没记上是因为 accel 先掐局。
4. 拿掉腿碰撞是 **Stage E 上楼**用的：台阶踢面会蹭小腿，有碰撞就会弹开/绊倒，课表学不会爬。不是为梅花桩设计的。给小腿加回碰撞能挡住穿模，但会改整条走跑，必须新 lineage。

---

## 建议的下一步（尚未开干，需人点头）

新 lineage（s6），**不要热补 s5 ckpt**。

1. 先停或降 s5（四卡对照不值得再烧到 40k；ckpt 留着当负结果）。
2. **easy 踏石**：顶 ≥ 0.36 m，石间净空 ≤ 0.10 m，上台高降到 ~8–10 cm；hard 仍可收窄。
3. **easy 圆桩**：不要再缩缝；高度 14 cm → ~6–8 cm。
4. 稀疏列的 10% 随机 level 在 easy `reach_2m` 起来之前先锁 0–3 行。
5. 不要先改 `LIGHTLP_ACCEL_LIMIT=40`、免疫 10%、OOB 4.25、horizon 20 s、algebraic。先让第一脚能踩上。
6. s6 监控补上分地形 slack / illegal / Eq.5，以及 `Reset/*`（需用带该日志的进程启动）。

健康门：easy 踏石 `reach_2m` 离开 3% 平台、进度离开 1.1 m。仍不是能力验收。

---

## 明确不要做

- 续训 s4 / 在 s5 上热补几何或 MDP
- 蒸学生、改 1155D、动 zhuoqun
- 用 s4 错名 TB 或 s5 全局 reward 宣称会走梅花桩
- 本机 Windows 当正式 Isaac play（没有 `t4-isaac-jammy:v2`）
- 为看 `Reset/*` 重启已经跑到 16k 的 s5（除非决定停训）

---

## 怎么接

```text
读 docs/README.md
读 docs/plans/2026-08-18--t4-sparse-terrain-index-fix-plan.md
看 artifacts/eval/s5_replay/ 两个 mp4 + stones 帧 180
nubot: tmux a -t t4-sparse-lightlp-s5
TB: http://100.100.188.39:8008
pytest: tests/test_t4_terrain_column_map.py tests/test_t4_sparse_reward_contracts.py tests/test_t4_terrain_curriculum.py
```

回放若要重录（nubot，占 GPU0 约 2 分钟启动 + 20 s）：

```bash
unset LD_LIBRARY_PATH
export T4_AMP_EXPERT_DIR=artifacts/amp_expert_provisional
export CUDA_VISIBLE_DEVICES=0
bash scripts/nubot_run.sh legged_lab/scripts/play.py \
  --task t4_loco_teacher_sparse --num_envs 1 --headless --enable_cameras \
  --terrain --difficulty 0.0 --terrain_types stepping_stones \
  --load_run 2026-08-19_00-20-19_t_sparse_lightlp_s5 \
  --checkpoint model_14500.pt --duration 20 \
  --record artifacts/eval/s5_replay/out.mp4
# Isaac 自带 imageio 写不出 mp4，会落成 .gif；再用系统 /usr/bin/ffmpeg 压小
```

`--cam_eye -3.4,...` 必须写成 `--cam_eye=-3.4,...`（否则 argparse 把负数当新 flag）。

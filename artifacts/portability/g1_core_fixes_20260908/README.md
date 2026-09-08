# 核心缺陷修复与验收证据

状态：F1–F6/R1/R2已完成修复及实际验收，独立审核通过。新训练已启动：B径向判据、GPU0/2各2048env、30000更新，TensorBoard [8042](http://100.100.188.39:8042/#scalars)。启动核验见[launch_verification.json](launch_verification.json)，旧A/B各10个checkpoint与历史保留见[old_stopped.json](old_stopped.json)。开训合同不等同运动能力。

修复后入口：[manifest.json](manifest.json)，包含实际生产源码SHA、精确依赖、资源、验收JSON及独立审核状态。[code_review.md](code_review.md)保存两个审核轴结果。

- `acceptance/`：三组32env/500步、两组8env reset和双GPU两次PPO/AMP更新的原始JSON/日志。7份JSON均通过；6个进程退出码均0。
- `g1-core-final-integration-20260908.tar.gz`：最终原始结果与完整日志；`g1-core-runtime-records-20260908.tar.gz`：实际运行验收脚本、唯一IsaacLab内容补丁与依赖记录。
- `g1-core-single-results-20260908.tar.gz`：三组单卡原始结果及保留的第一轮无效overlap注入失败，未将失败隐藏为通过。
- `cpu_red.log`、`timeout_red.json`：修复前反例；`cpu_regression.log`：173项受影响检查通过。三条基线flake8告警在`flake8_baseline.log`复现。

## 修复前证据（以下文件位于evidence/）

以下仅是修复前审计快照，不能将其中探针`ok`解释为修复后的通过。

唯一计划入口：[机器人无关运动训练计划](../../../docs/plans/2026-09-07--robot-neutral-locomotion.md)。文件完整性见 [evidence/manifest.json](evidence/manifest.json)。这些证据在进入仓库前已实际生成，原件位于本机临时审计目录；此副本避免后续执行依赖临时目录是否仍存在。

证据目录禁用Git换行转换；两个`.py.txt`按原始快照保存，Git不生成文本差异，以保持源字节和SHA不变。它们不是本轮实现文件。

| 文件 | 内容与限制 |
| --- | --- |
| `learning_probe.json`、`learning_probe.py.txt` | CPU生产函数/更新反例：timeout bootstrap、std floor、AMP时间轴；包含不应判为明确bug的mirror风险和休眠normalizer问题。保存时没有重跑。 |
| `mdp_probe.json`、`mdp_probe.py.txt` | CPU隐藏heading目标和timeout对象生命周期反例；瞬时速度积分只是局部奖励机制，不属于本计划调参项。 |
| `runtime_import_audit.json` | 四rank入口、同环境子进程导入、6份字节码、IsaacLab精确commit和1534文件blob核对；唯一内容改动为assets.py。不是活跃解释器内存快照，不能复原历史T4的未记录依赖commit。 |
| `g1_mixed.json`、`t4_mixed.json`、`g1_flat.json` | 实际Isaac各32env、500步加pulse/reset；三者都复现horizon后stale timeout。JSON的ok只表示探针执行完，不是所有合同通过。 |
| `reset_cache.json` | 实际8env G1高速足速度reset反例；新回合实际速度归零，prev仍旧，首步EMA113.93/64.22而正确基线为0/0。最终USD自碰撞属性为True。 |
| `server_results.tar.gz` | 上述三组完整JSON、日志、退出码和原始server_probe.py/run.sh。 |
| `reset_cache_results.tar.gz` | 高速reset探针完整JSON、日志、退出码和原始reset_cache_probe.py/run_cache.sh。执行脚本需要与上个包的server_probe.py放在同一诊断目录。 |

脚本中的旧绝对路径是修复前溯源，不是新的开训指令。实施时先检查脚本，只把依赖路径切换为独立候选并记录差异；不得把诊断注入施加到正在训练的实例。正式验收应将缺陷对应的不一致变为明确失败条件，不只检查脚本退出码。

Actor不接收真实根速度是正常的非对称部署观测合同。速度注入反例仅说明信息接线，不纳入本计划修复或对照。CPU统计、运行时合同和行为能力需要分别报告。

# 核心缺陷修复计划的修复前证据

状态：仅保存既有审计结果，用于2026-09-08用户要求的修复计划。没有实施修复、重新运行测试或修改训练。

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

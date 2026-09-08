# G1 塌低恢复合同对照

执行与验收以 `docs/plans/2026-09-07--robot-neutral-locomotion.md` 顶部为准。

- 远端独立目录：`/home/nubot/phn_ws/t4_train/TienKung-Lab-g1-recovery-20260907`。
- 训练会话：`g1-recovery-30k`；GPU1/3，双rank各2048env、24steps、seed42、权重2、全17段、冷启动30000轮。
- 主run：`logs/g1_recovery_30k/2026-09-07_23-26-29_g1_recovery_30k`。旁边23-26-28目录仅为另一rank创建的params目录；沿用冻结入口的按rank时钟建目录行为，不是第二个训练实验。
- 对照：原GPU0/2 full17权重2继续运行；权重4停止记录及保留checkpoint SHA见 `weight4_stop.json`。
- 新监控：`http://100.100.188.39:8040/#scalars`，tmux `g1-recovery-tb`。8039保留历史权重4视图。

## 证据

- `audit_inputs/`：修复前完整合同差异及model4000审计原始JSON。
- `source_manifest.json`：339份冻结源码/资产SHA；相对基线只改4个源码文件。`config_diff.json`核对实际保存配置；rank/device/路径元数据须与行为差异区分。
- `probe.py.txt`、`flat_probe.py.txt`：实际探针源码快照；远端同目录为可运行`.py`。`train.py.txt`为与原基线SHA完全相同的训练入口；远端`train.py`。
- `legacy.json.gz`、`recovery.json.gz`：32env×1000step原始轨迹，无损压缩；`compressed_traces.json`记录压缩前后SHA。`probe_summary.json`的SHA指解压后原始JSON。远端保留未压缩JSON和完整log/张量。
- `flat.json`：32/32站满6秒，actual vx约-.00165m/s；强制horizon检查终止前AMP帧与计时清零通过。
- `review.json`：独立review-agent两次检查，无代码阻断发现；第二次重算原始轨迹、SHA和启动脚本。
- `launch_verification.json`、`runtime_health.json`：实际rank、保存配置、专家SHA、model_0和TensorBoard健康；仅证明正确开训。

新合同连续低姿态10步且非immunity才因collapsed结束。原始低姿态/被豁免持续低姿态/实际塌低终止记录为`Collapse/low_fraction`、`Collapse/immune_skip_fraction`、`Collapse/terminated_fraction`，分母为环境步，不是重置事件。

## 结论边界

model4000混合探针legacy/new分别27/2次collapsed，新组28次accel（原因可重叠），两组定义内恢复均0；不能把终止原因迁移当成改善。两个探针的episode重置会让后续随机流分岔，不是严格逐状态配对的因果试验。真正的学习对照来自新冷启动；后续按相同采样预算比较速度、净位移、恢复、evaluator与连续回放。

56项相关测试通过。适用pre-commit除环境兼容问题外通过：旧flake8/pyupgrade不支持默认Python3.13，使用已有3.12复核；pyupgrade通过，flake8的R506/SIM901在HEAD基线文件同样复现，不宣称全量lint全绿。

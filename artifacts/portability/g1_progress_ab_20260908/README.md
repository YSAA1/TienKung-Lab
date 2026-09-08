# 全速 G1 A/B 证据

权威计划：`docs/plans/2026-09-07--robot-neutral-locomotion.md`。

- A GPU0/2：全速、原累计路径晋级。B GPU1/3：全速、最大径向距离晋级。
- 两组冷启动，全17段、权重2、recovery终止、2048env/rank、seed42+rank、30000更新。
- `probe.py`：实际Isaac命令采样、部分reset和晋级接线验证，非行为能力测试。
- `patch_hashes.json`：部署补丁LF字节哈希；`source_manifest.json`：342份源码/资产/入口原始字节哈希、17段专家、实现commit。
- `smoke_health.json`：双卡2更新smoke；`old_stop_confirmed.json`：旧两组64个checkpoint清单及退出确认。
- `latest_health.json` / `launch_verification.json`：实际保存配置、四rank/GPU映射、数值与TB8041双run核验。`status.py`是远端系统Python只读刷新入口。
- 本轮使用相同运行源码，实际配置只差晋级距离和实验名称。A先完成初始化，按同更新数比较。
- 原地周期轨迹不允许B晋级，但A保留原行为供因果对照；代码测试在 `tests/test_g1_progress_ab.py`。
- 完成开训不代表策略已经学会走路。500/1000/2000/4000检查点需要同预算固定评估与连续回放。

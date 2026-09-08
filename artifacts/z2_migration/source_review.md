# Z2 上游对抗审查（实施前）

审查对象：`work/upstream-z2`，上游提交 `c78eb1f8e31b7f7872733110c10276b7b2159414`。
方式：独立只读 agent 源码/XML/JSON 解析；不是 Isaac runtime 验收，也不是实现完成审核。

| 项目 | 结论 | 证据与后续 |
| --- | --- | --- |
| 真实模型 | 静态 pass | `legged_lab/assets/z2_description/assembly_urdf_29/assembly.urdf`：29个非固定关节，总质量36.690941kg，URDF引用mesh均存在。其他模型是20或23DoF。 |
| AMP宽度 | 直接复用 fail | 上游 `envs/z2/z2_switch_29dof_cfg.py:835` 为q29+dq29+双腕6=64D。目录另混70D subject与g1_compat，必须白名单/内容核验。 |
| 顺序 | unknown | source/URDF、SDK左右交错和AMP分肢是不同顺序；AMP腰yaw/roll/pitch与source yaw/pitch/roll不同。目标必须显式名称映射、真实articulation核验和关节脉冲。 |
| plant选择 | unknown | 上游配置约527行选WALK_POSE_DAMPED_PD_CFG：踝150Nm/12rad/s、Kp50/Kd4、髋膝Kd6；基础CFG为踝75Nm/10rad/s、Kp60/Kd2、髋膝Kd4。任务根高度.75与资产.8也不同，需沿注册/继承链确定迁移配方。 |
| 原专家速度 | 直接复用 fail | `motion_amp_expert/run.txt`39帧dt.03375；`motion_visualization/run.txt`171帧dt1/30，不能同名假定关联。原run的dq与forward difference MAE4.2456rad/s，max14.395；原walk/walk_l亦不一致；repaired walk误差为0。 |
| 镜像/站姿 | unknown | 髋/肩关节origin含±.2618rad；两脚碰撞引用R_ankle_roll_link.STL；腕yaw无碰撞。须完整FK和mesh确定镜像/足底/site，不可沿用G1值。 |
| runtime与行为 | unknown | 尚无Z2实际Isaac、训练或行为验收。静态通过不构成开训通过。 |

已通过 `work/z2-migration/review-feedback-01.md` 交接原Grok实施会话；队列请求 `14a691a5-6faa-44b7-8c5a-2dc5f4316e0e`，必须确认处理后再验收。

## 补充：完整注册链与差分定义

`envs/__init__.py:117` 的 `z2_switch_29dof_amp_rl_symmetry` 绑定 `Z2EnvGeneric` 与 `Z2Switch29DofAmpEnvCfg(:606)`，继承 MixedAmp(:585)→AmpBase(:520)→29D Env(:444)→AB(:105)→Base(:393)。AMP base最终选择 WALK_POSE_DAMPED_PD_CFG，根高度覆盖为physical.base_link_height=.75。其余walk/walk_stop_phase/walk_stop_phase_torso/mixed四个注册29D AMP任务走同一AMP base。

继承配置：action_scale=.25（base_cfg:418），clip_actions=100(:440)，sim.dt=.005与decimation=4(:596)，物理200Hz/策略50Hz。generic_env:1159先action buffer/clip再target=action*.25+default_joint_pos。默认base_state_mode=pelvis；waist_roll_link奖励/随机化参考体不是free root。以上须真实resolved config复核。

速度误差严格为 `abs(diff(Frames[:,:29],axis=0)/dt - Frames[:-1,29:58])`；t=0..N-2，全部29关节，不重排、无循环边界、不比较最后帧，与上游repair_amp_motion_velocities.py:37一致。最后帧修复另复制末次差分。

上游visualization的70D是 `root_xyz3 + Euler3 + q29 + root_linear_velocity3 + root_angular_velocity3 + dq29`，不是目标AMP70D。不能混用这两个schema。

# Z2 实施首轮审查：未通过开训验收

范围：Grok 首轮实现与独立只读 agent 审查，2026-09-09。后续修复必须重新绑定文件 SHA；本记录不宣称修复后仍有同一缺陷。

## Important findings

1. **导入参数漂移**：初版 `Z2UrdfFileCfg` 未设置 `make_instanceable=False`。上游 YAML/转换函数均为False；父代理在 nubot 实际 `asset_converter_base_cfg.py:41` 核实默认True。已交原Grok会话修复并要求真实resolved config核验。
2. **负向探针假通过（已注入复现）**：`_fail`在`try`内部抛出的AssertionError被`except Exception`捕获，随后记为`rejected=True`。注入始终接受的AMPLoader，两个负样本仍被报告拒绝。原探针SHA `7841f5613cfa0fd5adb672405f5872139c1763e430aaf048eb8064d84d4cc181`；机器证据 `probe_negative_guard_counterexample.json`。还须拒绝把缺文件错误当成schema拒绝。
3. **训练清单不兼容**：原生成器输出`motions`统计，正式`train.py`需要`clips[stem].sha256`。不能去掉正式清单校验规避；须绑定CSV/原始源、实际专家、资产/代码与转换设置。
4. **未证实的物理改变**：初版训练URDF把左脚collision从上游R mesh改为L mesh，并以断言文件名的测试视作修复。独立几何对照没有证明原件有缺陷，已要求恢复上游collision。
5. **探针状态/证据不足**：课程采样写入的-1 sentinel未恢复即继续物理测试；脉冲检查取全部env最大位移，可能由重力/耦合假通过；self_collision硬编码True；实际PD/限幅/质量/接触未核验。须恢复有效地形与origins，并使映射和支撑检查可证伪。

## 独立几何证据

- 初始复制的URDF34、MJCF35、USD7文件与上游对应目录逐字节一致。
- 按URDF完整origin RPY和Rodrigues关节轴旋转，seed20260909、100组Uniform[-.25,.25]姿态，全link的镜像位置/旋转误差为0。此结论是几何，不是动态支撑。
- 原R与备选L脚mesh各7358三角、凸包157顶点、AABB相同：min[-.073,-.03876651,-.042]，max[.14749999,.03886795,.01693164]。
- R/L凸包体积分别.0005473983439987439/.0005474322834651056m³；双方全部凸包面法向上的支持函数最大差.000048937785740839346m，不是声称精确Hausdorff距离。
- 原R与改L放到同一默认姿态左踝后，最低z都为.06504234474368331m；与右脚镜像的支持函数差都为.00019311003578710573m。没有发现换mesh改善足底高度或对称性。
- 默认姿态踝间距.21171590337647067m，当前常量.2114是近似值；rootZ.75是有约6.5cm出生下落的姿态，不可视为静态接地证据。

## MJCF 的有限修复范围

原件 `R_sphere_hand` 等双手零四元数导致MuJoCo3.11加载失败。改为单位四元数有原URDF固定关节rpy=0作依据；仅两处改动，原件另存且与上游逐字节相同。修复后的加载为nq36/nv35/nu29。这只证明加载兼容，不证明MuJoCo与Isaac动态一致，也不授权改URDF碰撞。

审查时原URDF SHA `58772dbedbe89a00f721a503463dbcb4f95586368400a1eafdb95822e196c2b0`；改左collision后的SHA `282bea823c77831ee508adbb95f2fbd4ccd66b69633898b7adabb8ff1afa5a92`。
原MJCF SHA `e03724b5bd473f1e4a86f9d8f97705659ec9d18c3a2f943f28e5d7ec922dcd35`；两处quat修复后SHA `71b2846a4f1c2496abefe85556e6fa141baa1f1c81b494bb4905f7f765c8baed`。

首轮本地pytest日志为52passed，但不覆盖上述关键缺口。Z2真实Isaac articulation、专家FK、训练、1000/2000/3000iter行为验收均未完成。

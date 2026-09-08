# Z2 迁移代码与物理合同审核

结论：当前代码和短时物理接线验收通过；完整专家生成、正式训练和1000/2000/3000iter行为验收仍待完成。

实施：同一CLI Delegate会话的Grok4.6/xhigh。规划、实际Isaac验证与最终审核由父代理完成；独立只读对抗agent已复核来源、几何、参数和负向测试。格式/许可头与局部lint清理由验证流程完成，不改机器人或训练算法。

## 已确认

- 真正29DoF模型来自上游提交`c78eb1f8e31b7f7872733110c10276b7b2159414`。运行时直接加载原始USD及全部5个USD层，字节与上游一致。原URDF与全部mesh保留。
- 不能把本机重新导入的URDF当作同一运行模型：当前导入结果保留neck_link，共31body；上游原USD为30body。两者均29DoF。最终选择原USD，未改变原物理参数来迎合导入结果。
- 实际原USD30body/29joint逐项对照独立源URDF惯性计算和源WALK_POSE_DAMPED_PD工厂：质量、COM、完整惯量矩阵、Kp/Kd、力矩/速度限幅、armature、硬关节限位和friction均通过。实际总质量36.69094116985798kg，最大COM差5.45e-9m、惯量差8.97e-9kg·m²；13个collider启用，自碰撞启用，solver8/4。见`physics_original_usd/comparison.json`、`physx.json`及`source_physics_oracle.json`。
- USD实例代理必须使用`Usd.TraverseInstanceProxies()`读取碰撞体；普通遍历会错误得到0个collider。PhysX的惯量9元素为column-major，已经表达在bodyprim坐标系、绕COM，不应再次按principalAxes旋转。
- 最终USD环境探针：29个正/负脉冲和实际target映射通过，actor1997/critic2088/AMP70；120步actor/reward有限，24个终止前AMP快照与reset前状态一致。接触采样累计0reset，没有声称长期零动作平衡。见`source_usd_probe/probe.json`。
- 早期21帧真实Isaac与独立URDF FK对照的端点误差<4e-7m。它是局部FK证据，不能代替全部专家；见`preflight_fk/independent_fk_comparison.json`。
- 三个原始motion的所有396帧均未超出源硬关节限位；完整专家预计393帧（每clip去掉末帧以计算速度）。见`source_motion_limits.json`。

## 对抗问题已关闭

缺raw文件/伪raw SHA不能通过；正脉冲反向响应、配对初态漂移、错误target通道会拒绝；accepting-loader和missing-fixture不再被算成成功拒绝；sentinel测试后恢复level/origins；原始USD与专家层SHA被绑定；完成后的真实专家不再被“必须永远pending”的测试排斥。最终独立复核未发现该限定范围内剩余Important。

保留的明确适配：上游MJCF双手零四元数会导致MuJoCo加载错误，适配副本仅修正为单位四元数并保留原件；这不是动态等价声明。上游AMP64D/不同排列转换为本项目70D需真实Z2 FK；未复用G1专家。

## 实际检查

- 68项Z2/G1/共享边界相关pytest通过（本地pytorch解释器，显式PYTHONPATH指向本工作树及内置rsl_rl）。
- 28个本次编写/修改文件的pre-commit检查通过。不可变上游资产通过字节哈希/XML/FK核对，未送入改写型formatter。
- 305个Git索引Python源文件及本任务新增源文件的干净快照边界扫描：0违规。原目录扫描包含临时pre-commit工具的刻意非法Python测试样本，5个解析错误均来自该工具缓存；原报告保留在`boundary-final.json`，项目源码报告为`boundary-final-project.json`。

不据此宣称Z2已学会走路或稀疏地形能力。下一步生成并校验全部专家，绑定冻结源码/模型/配置后开训；每1000iter核对OOB、reach2m、速度、终止原因，并补固定evaluator与连续回放。

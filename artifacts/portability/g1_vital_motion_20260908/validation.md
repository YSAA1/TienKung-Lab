# VITAL起步配方实验验收

范围：用户授权GPU1/3独立冷启动实验；主训练GPU0/2保留。用户随后要求保留LightLP奖励，最终仅action_rate权重-.1→-.01，其他项及权重不变；另关闭加速度硬终止、采用VITAL确定性姿态限幅、关闭普通地形步态tracking gate。含多个变量，不能做单变量归因。

验证：62项相关CPU测试通过；真实双GPU各32env、两更新、17专家、非对称timeout与checkpoint重载通过。两rank policy/discriminator/AMP normalizer差值全0。真实cfg断言upright=1、slack=1.5、yaw=2、body_orientation=0，实验开关生效。原始JSON为distributed_rank0.json与distributed_rank1.json；完整远端日志为smoke.log。7个候选文件SHA逐项通过。

review方式self：核对默认配置不改变原LightLP分支；新终止理由同步terminated/bootstrap mask，不把物理失败当截断；G1冷启动及显式专家manifest校验；站立和稀疏地形步态清零保持；不读取或修改主训练进程。

静态检查：Black、isort及其余适用pre-commit hooks通过。缓存flake8/pyupgrade在默认Python3.14出现工具兼容错误，使用已有Python3.12运行同缓存插件；pyupgrade通过，flake8仅有env.py旧R506（原baseline日志已记录），未新增该告警。未将全库测试或未做的能力评估称为通过。

主训练1212更新附近upright贡献分析：最近50次更新Episode_Reward/upright=.12363，正项和=.39777，占31.08%；线速度=.09142占22.98%，角速度=.10094占25.37%，slack=.06406占16.10%。日志为归一化episode奖励；不是逐步原始值，也不含AMP贡献。相对净任务奖励的比例更高是负项抵消所致。论文Eq2/表I与实现和+1权重一致，不能仅凭占比判定过重。

只声明计算接线、配置及启动准备通过，真实行走能力待相同预算固定评估和连续回放。
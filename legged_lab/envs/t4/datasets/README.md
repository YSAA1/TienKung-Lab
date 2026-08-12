# T4 动作数据合同

`motion_source/` 保存用户提供的原始 T4 27DoF CSV。每行固定为：

```text
root_xyz(3) + root_quat_xyzw(4) + q(27)
```

原始数据按 `30 Hz` 处理。27 个关节顺序以
`legged_lab/assets/t4/t4.py::T4_JOINT_NAMES` 为唯一真值。

`motion_visualization/` 是由以下命令生成的中间文件：

```bash
python legged_lab/scripts/t4_csv_motion_conversion.py \
  --input legged_lab/envs/t4/datasets/motion_source \
  --output-dir legged_lab/envs/t4/datasets/motion_visualization \
  --fps 30
```

这些文件用于后续 T4 动作播放与生成 AMP expert。它们不能直接作为正式
AMP 数据：手脚末端位置必须在 IsaacLab 中使用迁移后的 T4 模型重新计算，
并与运行时 T4 AMP observation 使用同一个字段生成函数。

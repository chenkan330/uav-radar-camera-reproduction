# 1.6 条件 rig 敏感性

本分析使用已经冻结的 `local_model.json` 中**全部50个训练bootstrap刚体变换**，不修改或重训模型，不按照测试段误差选draw或调参。模型摘要必须与主结果的准备清单一致；准备 CSV/清单的摘要、固定相机模型和主轨迹也须核对一致。

```powershell
.\.venv\Scripts\python.exe tools/evaluate_mmaud_sensitivity.py
```

默认读取 `data/public/mmaud/prepared_v16`、`data/public/mmaud/calibration/local_model.json` 和 `output/v16_comparison` 的主结果；输出同目录中的 `sensitivity.json`、`sensitivity.png`、`sensitivity_report.md`。不上传原始图像、bag、参考位置或准备 CSV。

## 保持与变化

局部相机到真值坐标变换 `R_camera_to_truth/t_camera_in_truth`、局部K、归一化图像观测、相机噪声、雷达R、离散过程Q、门限、初始化和固定测试段均保持不变。每个训练rig draw仅改变 `p_truth=R_draw@p_radar+t_draw`，对应的相机合成为：

```text
R_camera_to_radar = R_draw.T @ R_camera_to_truth
t_camera_in_radar = R_draw.T @ (t_camera_in_truth - t_draw)
```

两组继续共享相同100 Hz输出和共同事件预测分区。每个draw的仅雷达状态、协方差必须与固定主结果逐元素完全一致，否则分析失败。所有draw的两组核心跟踪先完成，再解析评价真值。

准备真值当前位于主rig的雷达坐标；先恢复原参考坐标，再按每个draw逆变换，给两组使用相同参考：

```text
p_truth_original = R_primary @ p_prepared_radar + t_primary
p_radar_draw = R_draw.T @ (p_truth_original - t_draw)
```

参考时间不变，全部draw与两组的真值匹配数、完整片段/预设启动1秒后评价范围不变。仅雷达轨迹虽然不变，其误差可随共同参考坐标变换而变化；不能把这种变化理解为重新调过雷达参数。

## 可以与不能解释的内容

报告保存每个draw的全部指标、覆盖率、相机实际更新次数、rig及相机合成变换，并给出完整片段和启动1秒后的敏感性分位范围及融合误差较小的draw比例。

**这是条件rig敏感性，不是置信区间。**相机到参考坐标拟合保持固定，未联合重采样相机模型、图像人工标注、时偏及它们与rig的相关性；模型原有噪声不重新估计。50个draw是训练重采样，不是50次独立测试飞行。不能据此声称总体成功概率、完整标定不确定度或全部MMAUD泛化精度。

这仍然是训练GT监督的局部实验投影、人工辅助视觉和不稳定标定下的探索性 held-out 条件。它不替代官方鱼眼标定、完整检测器、设备验证或论文表 I 的逐数复现。

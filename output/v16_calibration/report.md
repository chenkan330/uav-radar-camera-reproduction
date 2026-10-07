# MMAUD V1 局部实验标定（训练真值监督）

本目录保存可复现本次短片段对照的派生参数与训练图像像素标注。**这不是官方相机内外参，也不是物理支架测量或完整鱼眼标定。** 相机使用两个独立训练段的 Leica XYZ 和 image-only 机身区域作有监督局部虚拟针孔拟合；雷达到真值的变换也仅由训练段估计。保留评价段真值没有参与标定、初始化、雷达关联、图像标注、门控或参数选择。

完整来源、读取历史、失败尝试、几何、拟合和限制见 [标定审计](../../docs/v1_6_calibration_audit.md)。两个训练段为 Unix 秒 `[1692846908.0,1692846923.1]` 与 `[1692847012.0,1692847026.1]`；评价段为 `[1692846959.5,1692846969.0]`。后一个训练段晚于评价段，这是离线保留实验。

- [local_model.json](local_model.json)：主参数、约定、噪声、50个相机 bootstrap 和50个雷达到真值的1秒块 bootstrap 变换。
- [training_image_annotations.csv](training_image_annotations.csv)：31个训练像素位置、人工区域与PNG来源哈希；不包含原始图像或评价真值。

主模型SHA-256：`be9281c8dcd3fc7bae85c1b4b884ad039e2733027921f8e32326f3d713c88a8f`。这是从D盘原始资料目录按字节复制的锁定参数，没有依据测试结果改写。工具为 `tools/calibrate_mmaud_local.py`。

生成时工具SHA-256为 `2d63529fdb3a1537b22478684dbe8b106a2823243292689b45ef26b84256e0d6`，已记录在模型provenance。生成后只更新工具开头的说明，明确训练GT用于刚体、局部相机投影与噪声拟合；数值实现没有修改，也没有覆盖冻结模型。重新生成的provenance工具哈希会相应变化。

训练相机重投影RMS6.674 px，固定中间深度probes预测重采样p95为4.607 px。雷达到真值的旋转bootstrap p95为54.469°，物理外参稳定性较弱；必须结合两方法共同变换的标定敏感性结果解读，不能凭固定主变换的小误差宣称硬件精度或通用提升。

本仓库实现代码遵循仓库MIT许可。**MMAUD数据及从其图像/轨迹产生的本目录标注与派生参数遵循原数据的CC BY-NC-SA4.0要求**，不因代码MIT许可变为可商用数据；保留作者署名、非商业及相同方式共享条件。[MMAUD官方许可与引用](https://github.com/ntu-aris/MMAUD/blob/0213ad16c9a6489c6a99f474ac759cebf5cb0f6f/README.md#licence)。

数据引用：Yuan, Shenghai et al., “MMAUD: A Comprehensive Multi-Modal Anti-UAV Dataset for Modern Miniature Drone Threats,” ICRA2024, pp.2745–2751, DOI10.1109/ICRA57147.2024.10610957。

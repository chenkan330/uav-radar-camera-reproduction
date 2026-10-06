# 第 5 步：真实数据输入与评价格式

本格式把录制文件转换成离线复现程序能检查的输入。读取器不猜测单位、不自动标定坐标系、不运行目标检测。真实数据请放在本仓库 `data/` 或其他本地目录；`data/` 已被 Git 忽略，不能把私有飞行日志或论文原始数据混入 GitHub 示例。

每套数据目录需要 `config.json` 和 `radar.csv`；相机、空雷达帧时间表及真值文件可选。CSV 使用 UTF-8，第一行为下面给出的精确列名。读取器允许其他说明列，但不会使用它们。所有数字必须有限，不能含 NaN/Infinity；ID 必须非空。

## 时间、坐标与单位声明

- 所有 `timestamp_s` 是**测量时刻**，`arrival_time_s` 是进入处理程序的**到达时刻**，单位为秒。可以使用同一起点的非负相对秒，也可以使用同一 Unix 时钟；两者不能混用。所有传感器及真值必须已经同步到同一时钟。要求 `0 <= timestamp_s <= arrival_time_s`。
- 所有 XYZ 使用米，速度使用米/秒；所有位置和真值必须已转换到同一全局坐标系。请在 `config.json` 中记录坐标轴方向、原点、标定来源及同步方式。读取器只能检查数值，不能证明坐标已正确对齐。
- `vx_mps` 必须是全局 x 方向速度分量。常见雷达输出的径向速度/多普勒速度**不能直接改列名当作 vx**；要先根据设备格式及坐标关系处理，或修改观测模型。处理说明须记录在数据来源文档。
- `intensity` 是非负**线性强度**。dB、dBm、dB SNR 等对数数值须先明确基准并换算；不能直接用于线性强度权重。
- bbox 单位为像素，顺序为 `xmin,ymin,xmax,ymax`，必须有正面积。配置中的相机内参必须对应检测图像使用的分辨率；缩放、裁切和畸变校正需在输入前完成并记录。

## config.json

必须是一个 JSON 对象。`read_dataset` 原样返回，具体滤波参数和相机参数由复现入口校验。它应包含数据来源、真实/合成标识、单位、全局坐标与时间同步声明，以及相机内外参、噪声假设和关联参数。不要用示例配置替代真实设备的校准数据。

入口要求：`data_kind`为`real`或`synthetic`；`coordinate_frame="global"`，`radar_velocity="cartesian_vx"`，`intensity_scale="linear_nonnegative"`，`timestamps="seconds_shared_clock"`。`camera`内含`intrinsics={fu,fv,u0,v0}`、3×3列向量旋转`rotation`和3维`translation`；约定`global=rotation@camera-translation`。没有相机观测时可以完全省略camera。`fusion`字段见`examples/config.synthetic.json`，`evaluation.max_interpolation_gap_s`为允许插值的最大真值间隔。

相机模式`paper_xyz`严格沿用论文的`state[0]`作光学深度，**即使填写自定义旋转/平移也不自动改成相机前向深度**；倾斜或平移实际相机应选择`bearing`，或明确修改模型，避免把全局X误作光学深度。当前没有自动镜头去畸变及IMU时变姿态处理。

## radar.csv

```csv
timestamp_s,arrival_time_s,frame_id,x_m,y_m,z_m,vx_mps,intensity
0.000,0.020,r000,3.10,0.12,0.30,0.02,8.0
0.000,0.020,r000,3.13,0.10,0.29,0.01,5.0
0.100,0.130,r001,3.11,0.11,0.30,0.03,9.0
```

同一 `frame_id` 的各行组成同一整帧点云。该帧所有行的测量与到达时间必须完全一致；同一 ID 不可复用于其他帧。每个雷达采集时刻最多一帧，即使使用不同 ID，也不允许重复的 `timestamp_s`，包括空帧时间表。返回记录含 `timestamp`、`arrival_time`、`event_id="radar:<frame_id>"` 及 `(N,5)` 的 NumPy `cloud`，列序为 XYZ/vx/intensity。保留点的行序，按帧测量时刻排序，不按到达时刻排序；复现入口负责异步处理。

没有任何点时保留文件表头，不填一行零坐标代替空帧。

## radar_frames.csv（可选）

```csv
timestamp_s,arrival_time_s,frame_id
0.000,0.020,r000
0.100,0.130,r001
0.200,0.230,r002_empty
```

每帧一行，ID 不可重复。可登记全部帧或仅登记空帧。与 `radar.csv` 重复登记的帧须时间一致。不存在点的帧返回 `(0,5)` 点云，保留真实缺测语义，不产生虚构观测。

## camera.csv（可选）

```csv
timestamp_s,arrival_time_s,detection_id,xmin_px,ymin_px,xmax_px,ymax_px,confidence
0.050,0.120,c000,310,225,330,245,0.94
```

每行一个目标检测框，`detection_id` 在整个文件中唯一；返回 `event_id="camera:<detection_id>"`、`bbox` 为 `(4,)` 数组，以及浮点置信度 `[0,1]`。每个相机采集时刻最多一个已选择的目标框，即使不同 ID 也不允许重复的 `timestamp_s`。输入须已确定对应所追踪的单个 UAV；本复现没有多目标跟踪身份分配。读取器保留低置信度框，由相机处理层按照论文 0.7 阈值筛选。缺少文件时返回空相机列表，可以运行仅雷达实验。雷达与相机可以共享同一时间戳；重复限制分别作用于各传感器，防止同一观测被重复更新。

这些框可来自已有检测器，但不是作者 MobileNet V2 权重或其训练数据的替代品。没有检测训练集/标注/模型时，不能宣称复现作者约 70% mAP。

## ground_truth.csv（可选，只用于评价）

```csv
timestamp_s,x_m,y_m,z_m
0.000,3.12,0.11,0.30
0.100,3.13,0.12,0.30
```

真值须与估计位置同一时钟、同一全局坐标、同一目标参考点。文件读取后按时间排序；重复时间拒绝。返回 `(M,4)` 数组，列序为时间/XYZ；缺少文件返回 `None`，不能计算位置精度。真实 VIO 原点、AprilTag 标定、目标质心与雷达反射点偏差都需要记录，不能用滤波输出充当真值。**真值不作为滤波器输入。**

## 评价原则

`evaluate_positions(times, positions, truth, max_interpolation_gap_s=0.2)` 接受严格递增的估计时间和对应 `(N,3)` 位置。真值时间也须严格递增。误差方向为 `estimate - ground_truth`。

仅在真值覆盖的时间内评价。恰好对应真值采样时间时直接比较；其他时刻只在前后真值间隔不超过 `max_interpolation_gap_s` 时按实际时间线性插值。不会向前/向后外推，也不会跨长缺测段插值。报告匹配数、未匹配数、匹配覆盖率和测量/真值/匹配时间跨度，避免部分重合被误当作完整实验。

论文第 7 页表 I 的主指标是 `signed_mean_xyz_m`（各轴有符号平均误差）与 `mean_euclidean_m`（三维欧氏距离平均值），不是各轴 MAE 或三维 RMSE。程序另给 `median_euclidean_m`、`std_euclidean_m`（总体标准差，ddof=0）、`p95_euclidean_m`、`rmse3d_m` 和 `axis_rmse_m`。缺少真值或没有有效重合时，指标为 `None`，状态分别为 `no_ground_truth` / `no_overlap`；不会打印虚构的零误差。

合成验证结果仅证明实现行为；要对比论文表 I，仍需作者两个真实飞行数据集、相机标定、时钟同步和实际噪声参数。

# 1.6 真实短片段 prepared clip 契约

此入口不下载原始素材、不猜标定、不投影测试真值生成视觉观测。实际数据、目标图像、明确来源的投影模型和独立真值坐标链核验后，才运行真实仅雷达/雷达＋相机视线对照。不得把构造的示例文件标为实测；缺材料时保持准备状态，不输出虚构精度。

官方镜头标定未取得时，可另行声明 `projection_model="local_pinhole_experiment"`：仅用独立训练段的 Leica 参考位置和实际图像监督，建立近轴的局部实验针孔映射。此时必须显眼注明“训练GT监督的局部实验投影＋人工辅助视觉；标定不稳定，探索性 held-out 对照”，不能冒称官方外参、完整鱼眼去畸变或可靠的设备标定。

## 文件及运行

目录包含 `clip.json`、`radar.csv`、`camera.csv`、`truth.csv`。原始图像/bag 保留本地，CSV 图像引用使用可公开的相对标签。数据请放在被 Git 忽略的 `data/`；报告不写本机绝对路径。

```powershell
.\.venv\Scripts\python.exe run_mmaud_comparison.py --clip data/mmaud/prepared_clip
# 暂时不绘图，仍保存完整数据及评价
.\.venv\Scripts\python.exe run_mmaud_comparison.py --clip data/mmaud/prepared_clip --no-plots
```

Windows 启动脚本可带同样参数：`run_mmaud_comparison.cmd --clip data/mmaud/prepared_clip`。默认结果保存在 `output/v16_comparison`。未提供路径或数据不满足核验条件时退出，不自动运行合成替代数据。

## clip.json：先固定测试段与参数

必须是 JSON 对象，包含以下字段；所有标定/拟合均需记录来源，禁止以待评价测试段真值拟合。

| 字段 | 要求 |
|---|---|
| `data_kind` / `dataset` | 精确为 `real` / `MMAUD` |
| `source.description` | 实际素材、处理与参考点说明 |
| `source.url` | 实际数据发布/来源 URL |
| `source.recording_id` | 实际录制 ID，不含私有绝对路径 |
| `source.time_origin_ns` | 原始记录时钟原点，非负整数纳秒；CSV 使用减去它后的相对秒 |
| `source.raw_sha256` | 非空对象：原始 bag/图像归档的相对标签 → 64 位 SHA256 |
| `selected_interval_s` | 预先固定 `[开始相对秒,结束相对秒]`，非负、结束大于开始 |
| `test_interval_s` | 可选；若填写须与 `selected_interval_s` 完全一致 |
| `input_sha256` | 可选：准备文件 `radar.csv` / `camera.csv` / `truth.csv` 的预期 SHA256；提供时严格核验 |

主入口没有测试时间裁剪或按结果改参数选项；参数及范围只能由固定清单给定。原始摘要在本入口中仅记录，实际核验应在素材准备阶段完成；本入口另计算三份 CSV 和清单的实际摘要。

### config.calibration：相机到雷达

包含 `rotation`（3×3正交旋转）、`translation`（3维米）、`noise_norm`（一个或两个正的归一化视线标准差），约定：

```text
p_radar = rotation @ p_camera + translation
p_camera = rotation.T @ (p_radar - translation)
```

并包含独立来源字段：

- `source_kind` 为 `official` 或 `held_out`。
- `source_description` 必须说明实际投影/外参模型的出处、方向、单位与核验证据。局部实验模型须说明训练GT监督、近轴适用范围及不稳定性；正式模型须说明镜头去畸变来源。
- `uses_test_truth` 必须精确为 JSON `false`。
- `fit_intervals_s`：至少一个独立拟合区间组成的数组 `[[开始1,结束1],[开始2,结束2],...]`，逐段与固定测试段完全不相交；边界共用也拒绝。可使用测试段之前及之后的分离校准段，但必须标明这是离线、非因果标定，不能称作只用过去信息的在线校准。
- 为兼容旧清单，也接受单个 `fit_interval_s=[开始,结束]`；它与 `fit_intervals_s` 不得同时设置为非空值。官方固定标定且没有在这次日志上拟合时可省略两者或填写 `fit_interval_s=null`，此时 `source_kind` 必须是 `official`。`held_out` 必须明确至少一个区间。

检查的是实际采用的每个分离校准段，而不是把首段开始到末段结束的整个外包范围视为训练数据。例如测试段在两个独立训练段之间时，只要它不与任一训练段相交就合规。必须保证拟合实际只用了声明的这些帧，不能把其间测试帧一并加入。

`projection_model` 可为 `undistorted_pinhole`（默认，必须确有镜头校准/去畸变证据）或 `local_pinhole_experiment`（必须为 `held_out`，使用独立训练段的局部实验投影）。相机使用相应明确声明的归一化视线，不使用论文的状态深度伪观测。局部 K 映射不是完整镜头矫正。标定不确定度在本初版按已知常量处理，应记录该限制。

### config.evaluation：独立真值坐标链

`max_interpolation_gap_s` 明确最大允许插值间隔。`truth_frame_alignment` 必须独立提供上述来源字段 `source_kind/source_description/uses_test_truth` 及 `fit_intervals_s`（兼容单个 `fit_interval_s`），证明 `truth.csv` 已转到雷达坐标。相机外参不是雷达到 Leica/VIO 真值的变换，不能借一个标定冒充另一个。

如果官方真值已经是雷达坐标，`source_description` 应引用明确坐标证据；如果需要拟合变换，仅能使用独立校准区间，并记录变换/时间偏置的来源。缺此声明直接拒绝评分。清单声明不会自动证明物理对齐，仍须人工核对实际证据。

### config.filter：显式滤波与初始化

必须填写 `radar_R`（3×3正定米²协方差）、`acceleration_std`（非负 m/s²，1.5 离散过程模型）、`velocity_std`、`prediction_hz=100`、`radar_gate_nis`、`camera_gate_nis`、`initial_point_index`（非负整数）及 `init_selection_description`（真实雷达选点依据，不能用真值）。

可选 `initial_velocity_from_two` 与 `initial_second_point_index`，意义见 [1.6 算法说明](v1_6_algorithm.md)。默认速度零；首点只初始化一次。默认首点不是已证明的目标，必须明确初始候选选择过程。程序不采用原论文无强度条件下不可执行的最强点初始化，也不把 radial Doppler 伪造为 world vx。

## CSV

UTF-8，必须有以下精确列名。所有数值有限，禁止 NaN/Infinity。CSV 时间统一为原始记录时钟的**相对秒**，不得直接输入十亿量级 Unix 秒；雷达、相机、真值必须使用同一起点。

`radar.csv`：

```csv
frame_id,timestamp_s,x_m,y_m,z_m
```

同一 frame ID 的每行是一枚候选 XYZ，整帧时间相同，单位米、雷达坐标。不同 ID 不得占同一采集时刻。点的行序保留，初始化索引对应帧内行序。此格式不允许填虚构速度、强度，初版也没有空帧表；空点云应在准备审计中单独计数。

`camera.csv`：

```csv
timestamp_s,normalized_x,normalized_y,source_image,annotation_method
```

每时刻一条已经选择的 UAV 视线。`normalized_x/y` 为正式去畸变模型或明确声明的近轴实验 K 映射下的 `(X/Z,Y/Z)`，不是像素，也不是单位向量的前两维。局部 K 映射不得称作真实完整鱼眼去畸变。`source_image` 为实际图像相对标签。`annotation_method` 仅允许 `manual_oracle` 或 `detector`；不接受测试真值投影。人工点击/框中心得到的观测须标为 oracle 条件；使用 detector 时，在来源说明中记录模型及实际推理流程。禁止用测试真值选择检测框或判断点云关联。

`truth.csv`：

```csv
timestamp_s,x_m,y_m,z_m
```

位置必须已经通过独立坐标/时间链转成雷达坐标中的米，采集时间唯一。目标参考点差异须记录，例如机体质心、反射点、Leica 棱镜。可以保存真实边界前后约 0.3 秒的样本；读取器保留固定测试段内所有真值及每端最近一个真实包围样本，仅用于边界插值，不外推、不扩大测试时间或图表范围。

**当前入口在两组跟踪核心完成后才解析评价真值 CSV**，只做事后评价及绘图；测试GT不进入拟合、初始化、雷达门控/关联、相机标注或调参。此前素材语义审计、输入准备/坐标变换、训练段校准可能已经读取参考数据，不能把该顺序说成整个流程第一次打开GT。文件摘要读取字节是溯源操作，不等于把数值标签提供给跟踪器；训练GT监督与测试GT使用必须分别记录。

## 公平评价与输出

两组相同初始化、100 Hz 输出时间及共同事件预测分区。主指标为固定段完整可评价输出，包含初始化；同时按预先约定的初始化后 1 秒另列补充指标。两组的真值匹配数一致；不外推、不跨过长真值缺测段插值，无有效匹配时指标保留空值。

输出包括中文 `trajectory.png`、`time_error.png`、`mean_error_comparison.png`、`report.md`、`metrics.json`、`tracks.csv`、含完整协方差的 `tracks.npz`、`updates.json`、`input_manifest.json` 和 `camera_provenance.json`。所有主要/补充误差、覆盖率、协方差检查、更新门控、输入摘要和参数都保留。结果没有假称 paper_xyz、完整五关联、作者检测器或表 I 逐数复现。

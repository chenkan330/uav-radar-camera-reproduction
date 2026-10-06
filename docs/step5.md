# 第5步：数据接口、统一对比与复现记录

本步做了什么：新增明确单位/坐标/时钟的CSV输入、五种关联统一实验、仅雷达基线、按真实时间插值的评价、结果图与中文报告。每个实验记录参数、数据类型、初始化、迟到事件、真值覆盖率和协方差检查，便于重新运行及逐步追踪。

## 运行

```powershell
# 默认3个种子，5种关联×2相机模式，加1个仅雷达基线，共33组
.\.venv\Scripts\python.exe run_reproduction.py --synthetic --output output/local_reproduction
# 接入自己的已标定数据；没有ground_truth.csv时仍可跟踪，不输出精度
.\.venv\Scripts\python.exe run_reproduction.py --data data/my_flight --output output/local_my_flight
# 完整检查
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Windows可双击`run_reproduction.cmd`运行合成端到端实验；结果存到未跟踪的`output/local_reproduction`，保留GitHub参考结果。实际数据格式见[数据说明](data_format.md)；仅雷达数据不要求相机标定，没有相机观测就不会使用相机占位对象。

`examples/config.synthetic.json`可核对示例参数；`examples/config.real.template.json`必须填写实际相机标定和数据声明后使用，不能把示例内外参当作设备标定。真实雷达已经转到全局坐标、速度确实为笛卡尔vx才可输入；径向Doppler需要另建观测模型，不能猜测换算。

## 输出与指标

- `summary.json`：每组全部指标、配置、各轴有符号误差、协方差/更新统计和跨种子均值/标准差。
- `report.md`：中文对照表和实验边界。
- `representative_tracks.csv`：第一组数据的各方法回放修正轨迹；多种子每组结果均可由入口重建。
- `comparison.png`：代表性paper_xyz轨迹与各方法平均欧氏误差；误差条为跨fixture的总体标准差，不是置信区间。

论文Table I报告有符号各轴平均误差和3D欧氏距离均值，程序单独报告它们及RMSE，避免混称。仅在真值覆盖区间评价，不外推、不跨超过0.2秒的真值缺口插值；超界预测计入未匹配数量，不裁掉早期误差。没有真值或未初始化时不制造零误差。

默认seed42/43/44只是三个受控合成fixture，不是论文两个真实飞行数据集。paper_xyz平均/单簇Kmeans跨种子的平均欧氏误差为0.03004m，bearing为0.03165m；仅雷达为0.11304m。这些数据不能用于验证论文真实Table I精度。Kmeans与Average的轨迹已逐元素核对一致。

## 完成范围与未完成的论文实验

算法及合成验证：手写Kalman、雷达初始化/门控及五种关联、相机检测框几何、100Hz异步回放、CSV入口和统一评价已完成。

真实论文完整复现尚缺：作者两个飞行日志、时间同步及完整标定、11,660张训练图及标注、MobileNet V2实际检测头/训练配置/权重、Coral编译模型与ROS设备运行。当前仓库不声称重训250万步、达到约70% mAP、复现Table I或10W设备运行。这些材料到位后可用同一入口运行，逐步保存实测结果及与论文的差异。

本步所有示例配置的Q/R/P0/gate/beta0未由论文给出，属于明确假设。paper_xyz沿用相关状态深度，bearing为改进路径；修正历史不等于当时发布的实时轨迹。

2026-10-06验证：完整62项数学、数据及集成检查通过；33组结果全部初始化成功，最小协方差特征值4.1813e-5，对称性误差0；双击启动入口实际运行成功，结果图已查看。

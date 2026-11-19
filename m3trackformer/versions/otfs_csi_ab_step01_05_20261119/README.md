# OTFS 预测 CSI 与原 V6 的闭环选束对比

这份快照包含 B 组实验第 1–5 步的实现、总体指标和两张结果图。A 组由原 m³TrackFormer V6 输出波束分数，B 组在正常跟踪时通过过去五帧 OTFS 观测预测下一帧下行 CSI，按归一化预估增益选束。两组使用相同场景、码本和每帧 12 束预算，共用 V6 的匿名关联和失跟重捕流程。

## 已完成的步骤

| 步骤 | 程序或配置 | 验收 |
|---|---|---|
| 1 定义预测时刻 | [时序协议](my_training/isac/b_group_step01_protocol.json) | 50 ms 帧周期下，本帧扫描结束至下一帧首束的间隔为 33.2 ms。 |
| 2 对齐观测与标签 | [样本采集](my_training/isac/step02_b_group_csi_samples.py) | 构造 8 个五帧 OTFS 输入窗口及各自下一帧 CSI。 |
| 3 验证 CSI 选束 | [波束增益验证](my_training/isac/step03_b_group_beam_validation.py) | 单终端几何核对 8/8 通过，三终端 2 个控制算例的前 12 束与独立解析计算一致。 |
| 4 训练预测器 | [CSI 预测](my_training/isac/step04_b_group_csi_predictor.py) | 按原始飞行分区生成训练 756、验证 288、测试 288 个窗口，训练并验证小型回归模型。 |
| 5 闭环配对评估 | [原 V6 与预测 CSI](my_training/isac/step05_b_group_closed_loop_comparison.py) | 在固定阵列单径仿真的 17 组三目标上核对跟踪结果，并按实际发射时隙评价下行增益。 |

## 第 5 步结果

| 指标 | A 原 V6 | B 预测 CSI |
|---|---:|---:|
| 三目标全程跟踪成功 | 14/17 | 15/17 |
| 平均同时跟踪时长 | 42.17 s | 45.07 s |
| 共同扫描目标帧平均增益损失 | 0.043 dB | 0.021 dB |
| 物理漏检后成功重捕 | 0/2 | 0/1 |
| 共同正确关联目标帧位置 RMSE | 8.58 m | 8.58 m |

第 10 帧启用 CSI 策略后，共有 42,762 个共同扫描目标帧。平均时长差来自一组场景，B 覆盖了该场景的完整评价时段。结论限于当前固定阵列单径仿真。第 4 步在单终端回波上训练，第 5 步以三终端场景的共同接收 IQ 特征运行。预训练语言模型支路已提供本地加载接口，当前指标来自小型回归模型。

- [总体指标](results/aggregate_metrics.json)
- [CSI 训练与测试图](results/csi_prediction_metrics.png)
- [闭环对比图](results/stage05_ab_comparison.png)

## 运行所需资产

安装 Python 依赖后，从本目录执行对应入口。阶段 2 和阶段 4 依赖本地准备的原 V6 权重、飞行分区及当前实验配置。配置由程序从 my_training/outputs 下的 current_journal_experiment.json 与 current_identity_closed_loop.json 读取。阶段 3 读取阶段 2 生成的标签及相应参考数据。阶段 5 还需要阶段 4 生成的小模型权重和原 V6 的完整 OTFS 基线归档，入口提供基线及预测器目录参数。

公开快照提供源码、配置协议及总体实验产物。原始飞行记录、IQ、模型权重和机器配置须由有授权的本地实验环境提供。


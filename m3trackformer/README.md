# m³TrackFormer 多目标实验

当前归档版本：**v1_original**。所有旧版本在 `versions/` 中保留完整源代码 ZIP、指标及图片。

本项目使用雷达距离/角度测量和 Transformer 波束跟踪。3目标共享12束，51条轨迹组成17场景，每场景49.8秒；任一目标连续漏检5次则整组停止。检测帧位置RMSE的样本覆盖不同，请结合成功率和时长判断。

原始GNSS数据、训练权重和本机绝对路径配置未上传。下载后需准备已有正式版N-Net/R-Net权重及4个分区NPZ，使用 `--rnet-checkpoint`、`--nnet-checkpoint`、`--data` 指定本地权重及数据。项目环境位于仓库工作目录的 `.venv` 时会自动选择。

原始：`python my_training/step10_single_target_tracking.py --no-show`

扰动未优化：同一命令加 `--measurement-regime stress`；鲁棒版再加 `--quality-mode robust`（V3起）。

V4运动概率版：`python my_training/step10_single_target_tracking.py --maneuver-uncertainty --no-show`；需自行准备未公开的V4数据/权重。V4与V1～V3使用不同测试轨迹，只能在各自匹配场景内比较。

最新版本自动由闭环仿真生成 `figures/single_trajectory_tracking.png`，图中分别给出示例和总体跟踪时间/重捕率。V4蓝线是因果滤波当前位置，示例ID400017固定用于以后版本比较；逐帧绘图样本CSV只保留在本地，不公开。

当前是轨迹驱动的代理测量仿真，理想身份关联和波束命中反馈仍保留，未声称复现论文全部物理回波或数值。V2/V3固定同一噪声种子及模型；鲁棒版只使用已命中的测量和接收机名义协方差，不使用异常真值标签。

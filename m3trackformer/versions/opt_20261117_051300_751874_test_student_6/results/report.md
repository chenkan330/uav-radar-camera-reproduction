# 总体指标报告

版本：`opt_20261117_051300_751874_test_student_6`

本报告由本地冻结仿真结果的总体指标自动生成，数值未重新计算或调整。
公开包包含源码、总体指标、报告和仿真PNG；逐场景、逐目标、逐轨迹及逐帧数据保留在本地。

GNSS用于参考轨迹；距离/SNR相关测量误差及异常值为仿真代理。不同时间片段被组合为多目标场景，并非同步真实多目标雷达记录。
重复噪声试验共享场景几何；既往使用的测试记录不能称为全新盲测。定位RMSE只在可评价的检测/更新帧上统计，须结合覆盖率解读。
V5身份审计未反馈跟踪历史；V6关联结果反馈原Huber和N/R历史。更新恢复率与物理漏检重捕率分别报告。阶段门槛属于本项目约定。

| 总体指标 | 数值 |
| --- | --- |
| `aggregate.scenes_with_noise_repeats` | 17 |
| `aggregate.unique_geometrical_scenes` | 17 |
| `aggregate.success_count` | 13 |
| `aggregate.success_rate` | 0.7647058823529411 |
| `aggregate.mean_all_targets_duration_s` | 40.29117647058824 |
| `aggregate.recoveries` | 34 |
| `aggregate.recovery_rate` | 0.85 |
| `aggregate.position_rmse_m_on_detections` | 10.487697057249852 |
| `aggregate.position_evaluation_count` | 41034 |
| `aggregate.position_evaluation_coverage` | 0.8078195133475077 |
| `aggregate.filter_ms_per_detected_frame` | 0.006143958671157269 |

图像位于本版本的 `results/` 或 `figures/` 子目录，均保留仿真程序原PNG。

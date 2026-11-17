# 总体指标报告

版本：`v5_20261117_062903_997687_beam_identity_b4`

本报告由本地冻结仿真结果的总体指标自动生成，数值未重新计算或调整。
公开包包含源码、总体指标、报告和仿真PNG；逐场景、逐目标、逐轨迹及逐帧数据保留在本地。

GNSS用于参考轨迹；距离/SNR相关测量误差及异常值为仿真代理。不同时间片段被组合为多目标场景，并非同步真实多目标雷达记录。
重复噪声试验共享场景几何；既往使用的测试记录不能称为全新盲测。定位RMSE只在可评价的检测/更新帧上统计，须结合覆盖率解读。
V5身份审计未反馈跟踪历史；V6关联结果反馈原Huber和N/R历史。更新恢复率与物理漏检重捕率分别报告。阶段门槛属于本项目约定。

| 总体指标 | 数值 |
| --- | --- |
| `identity_metrics.beam_argmax.total` | 15767 |
| `identity_metrics.beam_argmax.assigned` | 15767 |
| `identity_metrics.beam_argmax.correct` | 14715 |
| `identity_metrics.beam_argmax.identity_switches` | 179 |
| `identity_metrics.beam_argmax.collision_total` | 1976 |
| `identity_metrics.beam_argmax.collision_correct` | 988 |
| `identity_metrics.beam_argmax.duplicate_track_frames` | 1004 |
| `identity_metrics.beam_argmax.near_tie_returns` | 1249 |
| `identity_metrics.beam_argmax.accuracy_all_returns` | 0.9332783662079026 |
| `identity_metrics.beam_argmax.precision_assigned` | 0.9332783662079026 |
| `identity_metrics.beam_argmax.assignment_coverage` | 1.0 |
| `identity_metrics.beam_argmax.accuracy_same_beam` | 0.5 |
| `identity_metrics.beam_argmax.beam_only_upper_bound` | 0.9373374770089428 |
| `identity_metrics.beam_geometry.total` | 15767 |
| `identity_metrics.beam_geometry.assigned` | 15381 |
| `identity_metrics.beam_geometry.correct` | 15381 |
| `identity_metrics.beam_geometry.identity_switches` | 0 |
| `identity_metrics.beam_geometry.collision_total` | 1976 |
| `identity_metrics.beam_geometry.collision_correct` | 1933 |
| `identity_metrics.beam_geometry.duplicate_track_frames` | 0 |
| `identity_metrics.beam_geometry.near_tie_returns` | 1249 |
| `identity_metrics.beam_geometry.accuracy_all_returns` | 0.9755184879812266 |
| `identity_metrics.beam_geometry.precision_assigned` | 1.0 |
| `identity_metrics.beam_geometry.assignment_coverage` | 0.9755184879812266 |
| `identity_metrics.beam_geometry.accuracy_same_beam` | 0.9782388663967612 |
| `identity_metrics.beam_geometry.beam_only_upper_bound` | 0.9373374770089428 |
| `tracking_reference.scenes_with_noise_repeats` | 51 |
| `tracking_reference.unique_geometrical_scenes` | 17 |
| `tracking_reference.success_count` | 0 |
| `tracking_reference.success_rate` | 0.0 |
| `tracking_reference.mean_all_targets_duration_s` | 5.349019607843136 |
| `tracking_reference.recoveries` | 218 |
| `tracking_reference.recovery_rate` | 0.7898550724637681 |
| `tracking_reference.position_rmse_m_on_detections` | 9.651093349923547 |
| `tracking_reference.position_evaluation_count` | 15767 |
| `tracking_reference.position_evaluation_coverage` | 0.1034661521904612 |
| `tracking_reference.filter_ms_per_detected_frame` | 0.020466708941281487 |

图像位于本版本的 `results/` 或 `figures/` 子目录，均保留仿真程序原PNG。

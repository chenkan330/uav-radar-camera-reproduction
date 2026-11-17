# 总体指标报告

版本：`v4_real_snr_outliers_20261117_r1_robust`

本报告由本地冻结仿真结果的总体指标自动生成，数值未重新计算或调整。
公开包包含源码、总体指标、报告和仿真PNG；逐场景、逐目标、逐轨迹及逐帧数据保留在本地。

GNSS用于参考轨迹；距离/SNR相关测量误差及异常值为仿真代理。不同时间片段被组合为多目标场景，并非同步真实多目标雷达记录。
重复噪声试验共享场景几何；既往使用的测试记录不能称为全新盲测。定位RMSE只在可评价的检测/更新帧上统计，须结合覆盖率解读。
V5身份审计未反馈跟踪历史；V6关联结果反馈原Huber和N/R历史。更新恢复率与物理漏检重捕率分别报告。阶段门槛属于本项目约定。

| 总体指标 | 数值 |
| --- | --- |
| `results.joint.scenes` | 17 |
| `results.joint.targets_per_scene` | 3 |
| `results.joint.success_rate` | 0.7647058823529411 |
| `results.joint.mean_all_targets_duration_s` | 40.29117647058824 |
| `results.joint.recoveries` | 46 |
| `results.joint.recovery_rate` | 0.8846153846153846 |
| `results.joint.mean_recovery_delay_s` | 0.05 |
| `results.joint.position_rmse_m_on_detections` | 10.611313207602018 |
| `results.joint.beam_slots_per_scene_block` | 12 |
| `results.joint.mean_prediction_batch_ms` | 1.8767585342682862 |
| `results.joint.amortized_prediction_ms_per_scene` | 0.13635726604835707 |
| `results.joint.latency_scope` | beam_prediction_and_selection_only_excludes_position_regression_and_IO |
| `results.joint.measurement_update_count` | 41022 |
| `results.joint.measurement_raw_rmse_m_on_detections` | 35.86792458519433 |
| `results.joint.measurement_processed_rmse_m_on_detections` | 6.924387692634733 |
| `results.joint.measurement_mean_weight` | 0.9789954807537771 |
| `results.joint.measurement_downweighted_fraction` | 0.054848617814831066 |
| `results.joint.measurement_normal_downweighted_fraction` | 0.03312051077414206 |
| `results.joint.measurement_outlier_downweighted_fraction` | 0.9956803455723542 |
| `results.joint.measurement_observed_outlier_count` | 926 |
| `results.joint.filter_timing.total_ms` | 252.10129992774455 |
| `results.joint.filter_timing.initialization_ms` | 0.9681999945314601 |
| `results.joint.filter_timing.detected_target_frames` | 41022 |
| `results.joint.filter_timing.mean_ms_per_acquired_target_frame` | 0.006145514600159538 |
| `results.joint.filter_timing.scope` | CPU_measurement_processor_only_excludes_neural_inference_and_logging |
| `results.joint.measurement_metric_scope` | acquired_frames_only_with_method_dependent_coverage; not neural_position_RMSE |
| `results.decoupled.scenes` | 17 |
| `results.decoupled.targets_per_scene` | 3 |
| `results.decoupled.success_rate` | 0.5882352941176471 |
| `results.decoupled.mean_all_targets_duration_s` | 34.26176470588236 |
| `results.decoupled.recoveries` | 39 |
| `results.decoupled.recovery_rate` | 0.8478260869565217 |
| `results.decoupled.mean_recovery_delay_s` | 0.05 |
| `results.decoupled.position_rmse_m_on_detections` | 9.96648124646609 |
| `results.decoupled.beam_slots_per_scene_block` | 12 |
| `results.decoupled.mean_prediction_batch_ms` | 4.542923996027584 |
| `results.decoupled.amortized_prediction_ms_per_scene` | 0.38811323611459736 |
| `results.decoupled.latency_scope` | beam_prediction_and_selection_only_excludes_position_regression_and_IO |
| `results.decoupled.measurement_update_count` | 34873 |
| `results.decoupled.measurement_raw_rmse_m_on_detections` | 34.80855889546994 |
| `results.decoupled.measurement_processed_rmse_m_on_detections` | 6.821194660639712 |
| `results.decoupled.measurement_mean_weight` | 0.9787883164803941 |
| `results.decoupled.measurement_downweighted_fraction` | 0.055315000143377396 |
| `results.decoupled.measurement_normal_downweighted_fraction` | 0.033364634074769646 |
| `results.decoupled.measurement_outlier_downweighted_fraction` | 0.9962264150943396 |
| `results.decoupled.measurement_observed_outlier_count` | 795 |
| `results.decoupled.filter_timing.total_ms` | 250.70209989644354 |
| `results.decoupled.filter_timing.initialization_ms` | 0.9669999999459833 |
| `results.decoupled.filter_timing.detected_target_frames` | 34873 |
| `results.decoupled.filter_timing.mean_ms_per_acquired_target_frame` | 0.007189002950604867 |
| `results.decoupled.filter_timing.scope` | CPU_measurement_processor_only_excludes_neural_inference_and_logging |
| `results.decoupled.measurement_metric_scope` | acquired_frames_only_with_method_dependent_coverage; not neural_position_RMSE |

图像位于本版本的 `results/` 或 `figures/` 子目录，均保留仿真程序原PNG。

# 总体指标报告

版本：`v6_20261117_071813_412022_validation_v5_gate`

本报告由本地冻结仿真结果的总体指标自动生成，数值未重新计算或调整。
公开包包含源码、总体指标、报告和仿真PNG；逐场景、逐目标、逐轨迹及逐帧数据保留在本地。

GNSS用于参考轨迹；距离/SNR相关测量误差及异常值为仿真代理。不同时间片段被组合为多目标场景，并非同步真实多目标雷达记录。
重复噪声试验共享场景几何；既往使用的测试记录不能称为全新盲测。定位RMSE只在可评价的检测/更新帧上统计，须结合覆盖率解读。
V5身份审计未反馈跟踪历史；V6关联结果反馈原Huber和N/R历史。更新恢复率与物理漏检重捕率分别报告。阶段门槛属于本项目约定。

| 总体指标 | 数值 |
| --- | --- |
| `controller_tracking.scenes_with_noise_repeats` | 60 |
| `controller_tracking.unique_geometrical_scenes` | 30 |
| `controller_tracking.success_count` | 40 |
| `controller_tracking.success_rate` | 0.6666666666666666 |
| `controller_tracking.mean_all_targets_duration_s` | 39.55083333333335 |
| `controller_tracking.recoveries` | 3253 |
| `controller_tracking.recovery_rate` | 0.9932824427480916 |
| `controller_tracking.position_rmse_m_on_detections` | 10.150689780127937 |
| `controller_tracking.position_evaluation_count` | 138687 |
| `controller_tracking.position_evaluation_coverage` | 0.7735776439089692 |
| `controller_tracking.filter_ms_per_detected_frame` | 0.0033184624350609446 |
| `ideal_huber_reference.scenes_with_noise_repeats` | 60 |
| `ideal_huber_reference.unique_geometrical_scenes` | 30 |
| `ideal_huber_reference.success_count` | 43 |
| `ideal_huber_reference.success_rate` | 0.7166666666666667 |
| `ideal_huber_reference.mean_all_targets_duration_s` | 42.30750000000001 |
| `ideal_huber_reference.recoveries` | 152 |
| `ideal_huber_reference.recovery_rate` | 0.8941176470588236 |
| `ideal_huber_reference.position_rmse_m_on_detections` | 10.127451557018826 |
| `ideal_huber_reference.position_evaluation_count` | 152048 |
| `ideal_huber_reference.position_evaluation_coverage` | 0.848103525211959 |
| `ideal_huber_reference.filter_ms_per_detected_frame` | 0.003736789698816956 |
| `identity_tracking.scenes_with_noise_repeats` | 60 |
| `identity_tracking.unique_geometrical_scenes` | 30 |
| `identity_tracking.success_count` | 40 |
| `identity_tracking.success_rate` | 0.6666666666666666 |
| `identity_tracking.mean_all_targets_duration_s` | 39.55083333333335 |
| `identity_tracking.recoveries` | 3282 |
| `identity_tracking.recovery_rate` | 0.9933414043583535 |
| `identity_tracking.position_rmse_m_on_detections` | 10.150689780127937 |
| `identity_tracking.position_evaluation_count` | 138687 |
| `identity_tracking.position_evaluation_coverage` | 0.7735776439089692 |
| `identity_tracking.filter_ms_per_detected_frame` | 0.0033184624350609446 |
| `acceptance.passed` | False |
| `acceptance.checks.success_retention` | True |
| `acceptance.checks.duration_retention` | False |
| `acceptance.checks.assigned_precision` | True |
| `acceptance.criteria.max_success_loss` | 0.05 |
| `acceptance.criteria.min_duration_retention` | 0.95 |
| `acceptance.criteria.min_assigned_precision` | 0.995 |
| `acceptance.label` | project_stage_criteria_not_an_industry_or_paper_standard |
| `association.total` | 142166 |
| `association.assigned` | 138687 |
| `association.correct` | 138656 |
| `association.wrong` | 31 |
| `association.rejected` | 3479 |
| `association.collision_total` | 9802 |
| `association.collision_correct` | 9522 |
| `association.identity_switches` | 58 |
| `association.identity_switches_consecutive` | 58 |
| `association.unassigned_with_returns` | 3696 |
| `association.confirmed_empty_scan` | 0 |
| `association.accuracy_all_returns` | 0.9753105524527664 |
| `association.precision_assigned` | 0.9997764750841823 |
| `association.assignment_coverage` | 0.9755286074026138 |
| `association.accuracy_same_beam` | 0.9714344011426239 |
| `association.correct_update_coverage` | 0.7734047300312361 |
| `association.zero_wrong_update_scene_success_rate` | 0.5833333333333334 |

图像位于本版本的 `results/` 或 `figures/` 子目录，均保留仿真程序原PNG。

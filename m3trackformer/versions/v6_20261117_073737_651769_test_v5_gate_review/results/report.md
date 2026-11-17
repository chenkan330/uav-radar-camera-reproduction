# 总体指标报告

版本：`v6_20261117_073737_651769_test_v5_gate_review`

本报告由本地冻结仿真结果的总体指标自动生成，数值未重新计算或调整。
公开包包含源码、总体指标、报告和仿真PNG；逐场景、逐目标、逐轨迹及逐帧数据保留在本地。

GNSS用于参考轨迹；距离/SNR相关测量误差及异常值为仿真代理。不同时间片段被组合为多目标场景，并非同步真实多目标雷达记录。
重复噪声试验共享场景几何；既往使用的测试记录不能称为全新盲测。定位RMSE只在可评价的检测/更新帧上统计，须结合覆盖率解读。
V5身份审计未反馈跟踪历史；V6关联结果反馈原Huber和N/R历史。更新恢复率与物理漏检重捕率分别报告。阶段门槛属于本项目约定。

| 总体指标 | 数值 |
| --- | --- |
| `physical_reception.recoveries` | 100 |
| `physical_reception.missed_target_frames` | 196 |
| `physical_reception.evaluated_target_frames` | 111426 |
| `physical_reception.recovery_rate` | 0.8403361344537815 |
| `controller_tracking.scenes_with_noise_repeats` | 51 |
| `controller_tracking.unique_geometrical_scenes` | 17 |
| `controller_tracking.success_count` | 34 |
| `controller_tracking.success_rate` | 0.6666666666666666 |
| `controller_tracking.mean_all_targets_duration_s` | 36.41372549019608 |
| `controller_tracking.recoveries` | 2465 |
| `controller_tracking.recovery_rate` | 0.9899598393574297 |
| `controller_tracking.position_rmse_m_on_detections` | 10.146996544222603 |
| `controller_tracking.position_evaluation_count` | 108579 |
| `controller_tracking.position_evaluation_coverage` | 0.7125167336010709 |
| `controller_tracking.filter_ms_per_detected_frame` | 0.00563857283237853 |
| `ideal_huber_reference.scenes_with_noise_repeats` | 51 |
| `ideal_huber_reference.unique_geometrical_scenes` | 17 |
| `ideal_huber_reference.success_count` | 35 |
| `ideal_huber_reference.success_rate` | 0.6862745098039216 |
| `ideal_huber_reference.mean_all_targets_duration_s` | 38.15 |
| `ideal_huber_reference.recoveries` | 111 |
| `ideal_huber_reference.recovery_rate` | 0.8409090909090909 |
| `ideal_huber_reference.position_rmse_m_on_detections` | 10.168743000797624 |
| `ideal_huber_reference.position_evaluation_count` | 116526 |
| `ideal_huber_reference.position_evaluation_coverage` | 0.7646665091739507 |
| `ideal_huber_reference.filter_ms_per_detected_frame` | 0.006161799082926862 |
| `identity_tracking.scenes_with_noise_repeats` | 51 |
| `identity_tracking.unique_geometrical_scenes` | 17 |
| `identity_tracking.success_count` | 34 |
| `identity_tracking.success_rate` | 0.6666666666666666 |
| `identity_tracking.mean_all_targets_duration_s` | 36.41372549019608 |
| `identity_tracking.recoveries` | 2486 |
| `identity_tracking.recovery_rate` | 0.9900438072481084 |
| `identity_tracking.position_rmse_m_on_detections` | 10.146996544222603 |
| `identity_tracking.position_evaluation_count` | 108579 |
| `identity_tracking.position_evaluation_coverage` | 0.7125167336010709 |
| `identity_tracking.filter_ms_per_detected_frame` | 0.00563857283237853 |
| `acceptance.passed` | True |
| `acceptance.checks.success_retention` | True |
| `acceptance.checks.duration_retention` | True |
| `acceptance.checks.assigned_precision` | True |
| `acceptance.criteria.max_success_loss` | 0.05 |
| `acceptance.criteria.min_duration_retention` | 0.95 |
| `acceptance.criteria.min_assigned_precision` | 0.995 |
| `acceptance.label` | project_stage_criteria_not_an_industry_or_paper_standard |
| `association.total` | 111230 |
| `association.assigned` | 108579 |
| `association.correct` | 108553 |
| `association.wrong` | 26 |
| `association.rejected` | 2651 |
| `association.collision_total` | 10247 |
| `association.collision_correct` | 9972 |
| `association.identity_switches` | 44 |
| `association.identity_switches_consecutive` | 42 |
| `association.unassigned_with_returns` | 2847 |
| `association.confirmed_empty_scan` | 0 |
| `association.accuracy_all_returns` | 0.9759327519554077 |
| `association.precision_assigned` | 0.9997605430147635 |
| `association.assignment_coverage` | 0.976166501843028 |
| `association.accuracy_same_beam` | 0.973162876939592 |
| `association.correct_update_coverage` | 0.7123461164921122 |
| `association.zero_wrong_update_scene_success_rate` | 0.5686274509803921 |

图像位于本版本的 `results/` 或 `figures/` 子目录，均保留仿真程序原PNG。

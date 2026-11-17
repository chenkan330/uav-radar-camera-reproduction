# 总体指标报告

版本：`v6_20261117_071813_412022_validation_gate36`

本报告由本地冻结仿真结果的总体指标自动生成，数值未重新计算或调整。
公开包包含源码、总体指标、报告和仿真PNG；逐场景、逐目标、逐轨迹及逐帧数据保留在本地。

GNSS用于参考轨迹；距离/SNR相关测量误差及异常值为仿真代理。不同时间片段被组合为多目标场景，并非同步真实多目标雷达记录。
重复噪声试验共享场景几何；既往使用的测试记录不能称为全新盲测。定位RMSE只在可评价的检测/更新帧上统计，须结合覆盖率解读。
V5身份审计未反馈跟踪历史；V6关联结果反馈原Huber和N/R历史。更新恢复率与物理漏检重捕率分别报告。阶段门槛属于本项目约定。

| 总体指标 | 数值 |
| --- | --- |
| `controller_tracking.scenes_with_noise_repeats` | 60 |
| `controller_tracking.unique_geometrical_scenes` | 30 |
| `controller_tracking.success_count` | 42 |
| `controller_tracking.success_rate` | 0.7 |
| `controller_tracking.mean_all_targets_duration_s` | 40.63750000000002 |
| `controller_tracking.recoveries` | 2923 |
| `controller_tracking.recovery_rate` | 0.9932042133876996 |
| `controller_tracking.position_rmse_m_on_detections` | 10.15821506705456 |
| `controller_tracking.position_evaluation_count` | 142971 |
| `controller_tracking.position_evaluation_coverage` | 0.7974732262382864 |
| `controller_tracking.filter_ms_per_detected_frame` | 0.0031989949051203304 |
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
| `identity_tracking.mean_all_targets_duration_s` | 40.08583333333335 |
| `identity_tracking.recoveries` | 3027 |
| `identity_tracking.recovery_rate` | 0.992133726647001 |
| `identity_tracking.position_rmse_m_on_detections` | 10.11289544724806 |
| `identity_tracking.position_evaluation_count` | 141028 |
| `identity_tracking.position_evaluation_coverage` | 0.7866354306113342 |
| `identity_tracking.filter_ms_per_detected_frame` | 0.0031989949051203304 |
| `acceptance.passed` | False |
| `acceptance.checks.success_retention` | True |
| `acceptance.checks.duration_retention` | False |
| `acceptance.checks.assigned_precision` | False |
| `acceptance.criteria.max_success_loss` | 0.05 |
| `acceptance.criteria.min_duration_retention` | 0.95 |
| `acceptance.criteria.min_assigned_precision` | 0.995 |
| `acceptance.label` | project_stage_criteria_not_an_industry_or_paper_standard |
| `association.total` | 146076 |
| `association.assigned` | 142971 |
| `association.correct` | 142180 |
| `association.wrong` | 791 |
| `association.rejected` | 3105 |
| `association.collision_total` | 12010 |
| `association.collision_correct` | 11033 |
| `association.identity_switches` | 752 |
| `association.identity_switches_consecutive` | 742 |
| `association.unassigned_with_returns` | 3324 |
| `association.confirmed_empty_scan` | 0 |
| `association.accuracy_all_returns` | 0.9733289520523563 |
| `association.precision_assigned` | 0.9944674094746487 |
| `association.assignment_coverage` | 0.978743941509899 |
| `association.accuracy_same_beam` | 0.9186511240632806 |
| `association.correct_update_coverage` | 0.7930611334225792 |
| `association.zero_wrong_update_scene_success_rate` | 0.5833333333333334 |

图像位于本版本的 `results/` 或 `figures/` 子目录，均保留仿真程序原PNG。

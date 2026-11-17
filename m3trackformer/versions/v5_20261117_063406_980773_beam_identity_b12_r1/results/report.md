# 总体指标报告

版本：`v5_20261117_063406_980773_beam_identity_b12_r1`

本报告由本地冻结仿真结果的总体指标自动生成，数值未重新计算或调整。
公开包包含源码、总体指标、报告和仿真PNG；逐场景、逐目标、逐轨迹及逐帧数据保留在本地。

GNSS用于参考轨迹；距离/SNR相关测量误差及异常值为仿真代理。不同时间片段被组合为多目标场景，并非同步真实多目标雷达记录。
重复噪声试验共享场景几何；既往使用的测试记录不能称为全新盲测。定位RMSE只在可评价的检测/更新帧上统计，须结合覆盖率解读。
V5身份审计未反馈跟踪历史；V6关联结果反馈原Huber和N/R历史。更新恢复率与物理漏检重捕率分别报告。阶段门槛属于本项目约定。

| 总体指标 | 数值 |
| --- | --- |
| `identity_metrics.beam_argmax.total` | 116526 |
| `identity_metrics.beam_argmax.assigned` | 116526 |
| `identity_metrics.beam_argmax.correct` | 111098 |
| `identity_metrics.beam_argmax.identity_switches` | 798 |
| `identity_metrics.beam_argmax.collision_total` | 9891 |
| `identity_metrics.beam_argmax.collision_correct` | 4889 |
| `identity_metrics.beam_argmax.duplicate_track_frames` | 5161 |
| `identity_metrics.beam_argmax.near_tie_returns` | 6597 |
| `identity_metrics.beam_argmax.accuracy_all_returns` | 0.9534181212776548 |
| `identity_metrics.beam_argmax.precision_assigned` | 0.9534181212776548 |
| `identity_metrics.beam_argmax.assignment_coverage` | 1.0 |
| `identity_metrics.beam_argmax.accuracy_same_beam` | 0.4942877363259529 |
| `identity_metrics.beam_argmax.beam_only_upper_bound` | 0.9570739577433363 |
| `identity_metrics.beam_geometry.total` | 116526 |
| `identity_metrics.beam_geometry.assigned` | 113710 |
| `identity_metrics.beam_geometry.correct` | 113686 |
| `identity_metrics.beam_geometry.identity_switches` | 40 |
| `identity_metrics.beam_geometry.collision_total` | 9891 |
| `identity_metrics.beam_geometry.collision_correct` | 9628 |
| `identity_metrics.beam_geometry.duplicate_track_frames` | 0 |
| `identity_metrics.beam_geometry.near_tie_returns` | 6597 |
| `identity_metrics.beam_geometry.accuracy_all_returns` | 0.9756277568954568 |
| `identity_metrics.beam_geometry.precision_assigned` | 0.9997889367689737 |
| `identity_metrics.beam_geometry.assignment_coverage` | 0.9758337195132417 |
| `identity_metrics.beam_geometry.accuracy_same_beam` | 0.9734101708624001 |
| `identity_metrics.beam_geometry.beam_only_upper_bound` | 0.9570739577433363 |
| `tracking_reference.scenes_with_noise_repeats` | 51 |
| `tracking_reference.unique_geometrical_scenes` | 17 |
| `tracking_reference.success_count` | 35 |
| `tracking_reference.success_rate` | 0.6862745098039216 |
| `tracking_reference.mean_all_targets_duration_s` | 38.15 |
| `tracking_reference.recoveries` | 111 |
| `tracking_reference.recovery_rate` | 0.8409090909090909 |
| `tracking_reference.position_rmse_m_on_detections` | 10.168743000797624 |
| `tracking_reference.position_evaluation_count` | 116526 |
| `tracking_reference.position_evaluation_coverage` | 0.7646665091739507 |
| `tracking_reference.filter_ms_per_detected_frame` | 0.004893555945719899 |

图像位于本版本的 `results/` 或 `figures/` 子目录，均保留仿真程序原PNG。

# 第5步：统一复现报告

数据类型：synthetic。本报告不代表作者真实飞行数据复现。

本步完成CSV导入、五方法比较、雷达单传感器基线、时间对齐评价和结果保存。

| 数据 | 相机模式 | 方法 | 平均欧氏误差(m) | 3D RMSE(m) | 真值匹配率 |
|---|---|---|---:|---:|---:|
| synthetic_seed42 | none | radar_only | 0.117609 | 0.125893 | 99.6% |
| synthetic_seed42 | paper_xyz | strongest | 0.046186 | 0.050208 | 99.6% |
| synthetic_seed42 | paper_xyz | closest | 0.062797 | 0.068164 | 99.6% |
| synthetic_seed42 | paper_xyz | average | 0.035076 | 0.038936 | 99.6% |
| synthetic_seed42 | paper_xyz | kmeans | 0.035076 | 0.038936 | 99.6% |
| synthetic_seed42 | paper_xyz | weighted | 0.033230 | 0.036941 | 99.6% |
| synthetic_seed42 | bearing | strongest | 0.053948 | 0.059367 | 99.6% |
| synthetic_seed42 | bearing | closest | 0.047993 | 0.054010 | 99.6% |
| synthetic_seed42 | bearing | average | 0.028820 | 0.031144 | 99.6% |
| synthetic_seed42 | bearing | kmeans | 0.028820 | 0.031144 | 99.6% |
| synthetic_seed42 | bearing | weighted | 0.028330 | 0.030788 | 99.6% |
| synthetic_seed43 | none | radar_only | 0.129677 | 0.142342 | 99.6% |
| synthetic_seed43 | paper_xyz | strongest | 0.044023 | 0.050034 | 99.6% |
| synthetic_seed43 | paper_xyz | closest | 0.049851 | 0.056022 | 99.6% |
| synthetic_seed43 | paper_xyz | average | 0.030437 | 0.032699 | 99.6% |
| synthetic_seed43 | paper_xyz | kmeans | 0.030437 | 0.032699 | 99.6% |
| synthetic_seed43 | paper_xyz | weighted | 0.028249 | 0.031193 | 99.6% |
| synthetic_seed43 | bearing | strongest | 0.054044 | 0.060778 | 99.6% |
| synthetic_seed43 | bearing | closest | 0.033172 | 0.037220 | 99.6% |
| synthetic_seed43 | bearing | average | 0.035600 | 0.039433 | 99.6% |
| synthetic_seed43 | bearing | kmeans | 0.035600 | 0.039433 | 99.6% |
| synthetic_seed43 | bearing | weighted | 0.033271 | 0.037172 | 99.6% |
| synthetic_seed44 | none | radar_only | 0.091831 | 0.100870 | 99.6% |
| synthetic_seed44 | paper_xyz | strongest | 0.061443 | 0.066782 | 99.6% |
| synthetic_seed44 | paper_xyz | closest | 0.077726 | 0.086568 | 99.6% |
| synthetic_seed44 | paper_xyz | average | 0.024612 | 0.027314 | 99.6% |
| synthetic_seed44 | paper_xyz | kmeans | 0.024612 | 0.027314 | 99.6% |
| synthetic_seed44 | paper_xyz | weighted | 0.030665 | 0.033995 | 99.6% |
| synthetic_seed44 | bearing | strongest | 0.074944 | 0.079745 | 99.6% |
| synthetic_seed44 | bearing | closest | 0.100157 | 0.107170 | 99.6% |
| synthetic_seed44 | bearing | average | 0.030516 | 0.034413 | 99.6% |
| synthetic_seed44 | bearing | kmeans | 0.030516 | 0.034413 | 99.6% |
| synthetic_seed44 | bearing | weighted | 0.036773 | 0.040925 | 99.6% |

各轴有符号平均误差、匹配时间范围、协方差和迟到事件统计见summary.json。
真实飞行精度、MobileNet V2训练/mAP和Coral/ROS硬件运行仍需原始数据、模型及设备。
paper_xyz使用相关状态深度；bearing为另行标明的改进。代表性轨迹是回放修正的历史。

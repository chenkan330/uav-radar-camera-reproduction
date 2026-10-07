# 版本1.6运行与恢复

当前机器双击仓库根目录 `run_v16.cmd`，会重跑已核验的固定真实短片段对照及50组标定敏感性分析。结果在 `output/v16_comparison`，全部图表中文。数据、Python环境及下载缓存均在D盘。

## 固定输入

- 数据：官方MMAUD V1 Mavic3，测试Unix秒 `[1692846959.5,1692846969.0]`；时间原点为Unix纳秒 `1692846887000000000`。
- 两个独立训练段 `[1692846908.0,1692846923.1]`、`[1692847012.0,1692847026.1]` 用于局部投影、雷达到Leica坐标链和观测噪声估计。后段训练意味着离线非因果标定。
- 测试段48张实际图像，首张早于雷达初始化，因此实际重放47个相机事件；143个有目标候选的雷达源帧，去除46个相同候选集后97个快照。每帧内部也去除精确重复点，不按点数缩小协方差。
- 图像机身中心来自人工初始选点和图像局部连通域跟踪，逐帧审核；属于人工辅助视觉条件，不是论文检测器。没有把测试真值投影成检测点。
- 测试GT用于最终坐标转换、评价和绘图，不进入跟踪、关联、初始化、图像标注或调参。整个流程的素材审计/准备曾读取GT，不能声称整个流程到最后才打开GT。

## 本机命令

```powershell
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe run_mmaud_comparison.py --clip data/public/mmaud/prepared_v16
.\.venv\Scripts\python.exe tools/evaluate_mmaud_sensitivity.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

## 从GitHub恢复数据与准备文件

原始图像、NPY和bag不随仓库发布。新电脑先在D盘创建Python环境，用 `requirements-v16.lock.txt` 安装固定依赖。取得[作者公开数据](https://ntu-aris.github.io/MMAUD/)时仍需遵守CC BY-NC-SA 4.0非商业学术许可。

```powershell
# 首先取得真实GT bag和ZIP目录；下载过程受预算限制，不下载整个10.75GB包。
.\.venv\Scripts\python.exe tools/fetch_mmaud.py
# 根据固定选段生成79张图像及全部GT/雷达NPY成员清单。
.\.venv\Scripts\python.exe tools/select_mmaud_clip.py
.\.venv\Scripts\python.exe tools/fetch_mmaud.py --members-file data/public/mmaud/v16_required_members.json
# 已有本地归档跨度缓存时可以离线核验与提取。
.\.venv\Scripts\python.exe tools/fetch_mmaud.py --offline --import-spans
```

公开下载地址或访问状态可能改变；工具对登录/权限页、错误格式、CRC或范围不符直接报错，不自动绕过。每个成功取得的原始成员立即保存，摘要见 `output/v16_data_audit/manifest.json`。本机还有9张早期探索图，固定实验不用它们；新机器只需固定79张图像。

为精确重跑已发布版本，恢复本版本的派生参数与图像标注，然后再次按原文件摘要核验：

```powershell
New-Item -ItemType Directory -Path data/public/mmaud/calibration -Force | Out-Null
Copy-Item output/v16_calibration/local_model.json data/public/mmaud/calibration/local_model.json
Copy-Item output/v16_calibration/training_image_annotations.csv data/public/mmaud/calibration/annotations_training.csv
Copy-Item output/v16_calibration/evaluation_image_annotations.csv data/public/mmaud/annotations_evaluation.csv
Copy-Item output/v16_calibration/evaluation_annotation_audit.json data/public/mmaud/evaluation_annotation_audit.json
.\.venv\Scripts\python.exe tools/prepare_mmaud_clip.py
.\run_v16.cmd
```

这会恢复已审核标注的声明；准备工具仍逐张核对原始图像SHA，并拒绝不匹配的标注/QA片段。标注CSV仅含派生像素、来源标签和摘要，不含原图或测试三维GT。重跑应得到同一数值结果；NPZ压缩文件的容器元数据未要求逐字节一致。

若研究重新标定，可独立运行 `tools/calibrate_mmaud_local.py --inspect-training`、`--annotate-training`、`--fit-training --camera-reference training_truth --bootstrap-count 50`。必须检查训练图像标注及各自区间，不得用测试GT改变投影、噪声或选择最好的重采样结果。重新生成模型后须作为新实验记录，不能覆盖本版本的已冻结来源而沿用旧结果。

## 解释范围

本次是训练GT监督的局部虚拟针孔投影、人工辅助视觉和XYZ雷达适配的探索性对照。完整官方鱼眼标定未取得；雷达—Leica变换重采样不稳定，原始增强点云含相关历史回波，传感器同步及目标参考点偏差尚未确证。不能把误差数值当作设备精度、原论文表I逐数复现或无人机对无人机实测证明。敏感性分析只扰动训练拟合的雷达—Leica变换，其他不确定性仍需独立验证。

数据及其派生标注/结果引用Yuan等人的MMAUD，ICRA 2024，DOI：10.1109/ICRA57147.2024.10610957；派生数据材料沿用CC BY-NC-SA 4.0。项目算法代码按仓库原软件许可；不能以代码许可替代原数据许可。

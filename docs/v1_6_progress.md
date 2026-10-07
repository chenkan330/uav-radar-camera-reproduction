# 版本1.6：MMAUD真实短片段对照

开始日期：2026-10-07；分支`v1.6-mmaud`，基于已冻结`v1.5`（`06dc1bf`）。当前阶段记录不代表版本已完成。

用户要求：先选一段有清晰目标、雷达回波和连续真值的真实短片段，比较“仅雷达”与“雷达＋相机”；每一步保存到GitHub并说明工作，图表标注中文。

## 第0步：隔离版本与准备环境

已建立独立D盘工作目录和虚拟环境，保留1.5和2.0工作目录。固定离线数据处理依赖于`requirements-v16.lock.txt`。新环境中原有62项算法检查全部通过。

安装：`uv pip sync requirements-v16.lock.txt --python .venv/Scripts/python.exe`。设置uv缓存和临时目录到D盘。检查：`.venv/Scripts/python.exe -m unittest discover -s tests -q`。

GitHub提交：`7101d9b`，已推送。

## 第1a步：取得并审计独立真实真值

官方Mavic3独立真值bag已保存在D盘，158,950字节，SHA-256 `c8e2adfb3528ce93dd2c37fe2ccf9a3853108a26bfee04845b78695f87cb0cf3`。实际解析`absolute`与`relative`各904个样本，时间覆盖184.208秒；存在一次0.701923秒缺口。按0.30秒覆盖阈值分为两个连续区间，后续不跨缺口评价。

`/tf`只有随目标运动的Leica绝对/相对位置，不能当作雷达—相机或雷达—世界标定。位置原始bag和导出CSV留本地忽略目录，GitHub只保存来源、哈希、格式、覆盖审计和中文时序图。

运行：`.venv/Scripts/python.exe tools/audit_mmaud_truth.py`；[审计报告](../output/v16_truth_audit/report.md)。本步骤未运行算法精度评价，也未宣称三种模态已配准。

## 完成条件

1. 保存可核验的数据来源、文件哈希、真实格式、时钟和标定审计。原始数据留本地，GitHub保存获取工具、清单与分析结果。
2. 从实际图像检查目标，从实际雷达回波检查观测，按真值覆盖连续性选固定区间。不得按最终误差挑片段。
3. 明确算法适配：MMAUD径向速度不等于原法世界系vx；若采用XYZ位置观测，标为删减适配。相机必须来自图像，不能来自真值投影。
4. 同一初始雷达观测、预测时间网格、雷达输入及评价时刻运行两组，只改变相机更新。记录门控、初始化和实际有效评价范围。
5. 输出中文图表、指标、限制与可重复入口，经检查后分阶段提交推送，完成后才发布v1.6标签。

## 数据适用边界

[MMAUD官网](https://ntu-aris.github.io/MMAUD/)提供地面多传感器观测无人机数据。它适合原论文框架的外部真实数据适配验证；不能据此宣称重现原论文作者实测数值，也不能验证2.0的移动自机空对空补偿。

官方论文：[MMAUD: A Comprehensive Multi-Modal Anti-UAV Dataset for Modern Miniature UAVs](https://arxiv.org/html/2402.03706v1)。原始数据按作者CC BY-NC-SA 4.0学术用途许可使用。

# m3复现的实验约定

- 保留第1～10步入口；新增实现放core，验证放checks，维护工具放tools。
- 每个新版本必须保留旧源代码与结果。使用core.versions.save_version，以新版本名写入experiments；不得覆盖已有manifest/source.zip/results。
- V1为原始多目标；V2为距离/SNR相关噪声与异常值未优化；V3为相同扰动上的因果鲁棒处理。比较必须保持场景成员、预算、种子、模型权重一致。
- 异常真值标签与参考位置只用于模拟测量和离线评价，不能输入质量估计、滤波或选束。
- 采用任一目标连续漏检5次即整组终止的口径；位置RMSE只针对检测帧，报告样本覆盖。
- 保留源代码ZIP、清单校验值、配置、逐场景与逐目标结果、指标图。原始飞行数据、权重、缓存与本机路径配置不发布到公共GitHub。
- GitHub发布必须有具体目标授权，并保留每个实验版本的独立提交/分支；自动审批拦截时继续完成本地工作，说明受阻内容。
- 运行项目.venv，测试至少包含checks/test_multi_target.py、test_measurement_quality.py、test_metric_charts.py、test_versions.py。

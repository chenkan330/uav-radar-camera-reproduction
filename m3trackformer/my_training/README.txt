第1～10步与原文多目标运行说明

第10步入口：step10_single_target_tracking.py。
现在直接点击VS Code右上角运行按钮，默认执行原文多目标核心流程：
  3个目标，共享12束，复用现有正式版N-Net/R-Net权重，比较联合选束与独立均分。
程序会自动使用项目.venv，结束后保存并显示跟踪状态图和整组持续跟踪比例曲线。

项目根目录PowerShell命令：
  & '.\.venv\Scripts\python.exe' '.\my_training\step10_single_target_tracking.py'

指定目标数量、总波束预算或后台运行：
  & '.\.venv\Scripts\python.exe' '.\my_training\step10_single_target_tracking.py' --targets 3 --total-beams 12 --no-show
  --targets 2 / --targets 4：目标数量变化，网络参数仍然共享，不重新训练。
  --total-beams：整组目标共同的波束预算，按原文要求大于目标数。
  --scenes：仅运行指定个数的场景，无放回组合，不能超过测试轨迹数/目标数。
  --split validation：切换验证集；默认test。
  --rnet-checkpoint：指定正式版权重；默认读取现有实验记录。
  --single-target：使用原来的教学版单目标流程及旧权重。

结果位于：outputs/step10_multi_target/run_*，终端会打印完整路径。
  summary.json：场景成员、数据与权重路径、汇总指标和实验限制。
  report.txt：中文简报。
  joint_scenes.csv / decoupled_scenes.csv：每个场景的结果。
  joint_targets.csv / decoupled_targets.csv：每个目标的恢复、定位和截尾记录。
  tracking_events.csv：逐目标逐时刻N/R模式、命中和恢复。
  beam_scans.csv：每个场景实际扫描的波束及扫描名额。
  scene_membership.npz：固定的场景组成，两种方法严格共用。
  tracking_status.png / tracking_survival.png：两张结果图。

多目标采用同场景任一目标连续5次漏检即整组终止的严格口径。
其他目标在此处记为同伴失败截尾，不能算成自身失败或完整独立跟踪时长。
独立基线每目标floor(总预算/目标数)束；仅使用该目标分配的扫描反馈，余数不用。
联合方案先对每个目标Softmax再求和，统一取Top-Bs，允许同一束命中多个目标。

现有飞行轨迹在测试分区内无放回组合；属于轨迹驱动的多目标仿真。
高斯测量、理想身份关联/同束可分辨性及已有单目标模型限制仍然存在。
已完成原文多目标选束/恢复核心流程，不声称复现全部物理回波或论文全部数值。

第1～10步原始教学程序保留；正式版模型和多目标实现位于core/内部目录。
详细的论文/源码差异和训练入口见JOURNAL_ALIGNMENT.txt。

2026-11-15：距离/SNR与异常值实验（固定权重，不重新训练）
  原始：第10步默认模式。
  扰动未优化：第10步后加 --measurement-regime stress --no-show
  鲁棒优化：第10步后加 --measurement-regime stress --quality-mode robust --no-show
  --noise-seed 20261116：默认固定扰动；未优化/鲁棒版必须使用相同值。
  --nnet-checkpoint：从其他电脑运行时可覆盖R-Net权重内的旧N-Net路径。

扰动模型为显式实验假设：参考300米/20dB，单站r^-4接收功率关系叠加2dB相关衰落；
距离与角度标准差随线性SNR平方根的倒数变化，并设噪声下限。
加入2%孤立异常及概率0.1%的2～4帧异常片段；前5帧初始化不加入异常。
异常额外偏移20～80米、两个角度各3～10度。SNR/名义协方差是理想化接收机质量元数据。
原有真实轨迹/波束真值及理想检出反馈保持一致；未增加低SNR随机漏检机制。
鲁棒版为常速度转换测量卡尔曼处理，使用一阶协方差传播和Huber软降权。
只处理实际命中的测量；异常标签仅离线评价，不参与更新或波束决策。
质量降权不产生失败波束证据；漏检位置不填成虚构观测。处理额外耗时单独记录。

旧版本完整保存在experiments/v1_original、v2_noise_raw、v3_robust。
每个版本都有source.zip、manifest.json、results与图表；数据和权重以SHA256引用。
新增版本必须使用新名称保存，禁止覆盖旧版本。检查命令：
  & '.\.venv\Scripts\python.exe' '.\my_training\tools\verify_versions.py'
全部比较图及CSV保存在experiments/comparison；不同版本RMSE评价帧集合可能不同，
应同时查看成功率、时长和evaluation_coverage_comparison.png。
第10步今后每次成功运行都会自动保存代码/结果快照，默认版本名为v_run_时间戳。
也可以用 --archive-version 自定义英文版本名；已有名称会被拒绝，防止覆盖。

V4：加速、急转弯、突然机动与混合运动的不确定性
  & '.\.venv\Scripts\python.exe' '.\my_training\step10_single_target_tracking.py' --maneuver-uncertainty --no-show
默认复用已保存V4运动网络，重新创建结果/代码快照；--retrain 才重新训练12轮。
新网络输入20帧已获取历史，输出未来0.05、0.25、0.5、1秒的3分量位置分布。
前5帧初始化与旧网络相同，其他历史左侧填充为无效，不提供额外初始化测量。
验证集选择模型，独立校准集调整标准差尺度，测试集不用于选择参数。
新测试轨迹与V3不同；V4内部比较旧N/R网络、新运动均值+固定扩散、新概率分布。
后两者使用相同新权重与均值，区分使用概率信息的作用；旧N/R只作为迁移基线。
90%覆盖率为逐坐标边际覆盖率的平均值，不能当作三维/整条轨迹联合覆盖率。
当前能量化部分未来不确定性；急转弯仍偏自信，不能提前预知随机机动命令。
本版归档：experiments/v4_maneuver_uncertainty；图表与报告在其中results/。

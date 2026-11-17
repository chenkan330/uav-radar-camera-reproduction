第1～10步与原文多目标运行说明

2026-11-16 用户明确的当前主线约束：
  必须保留原Transformer N-Net与R-Net，网络结构、输入输出接口及正常/重捕分工不变。
  正常跟踪使用N-Net；失跟后通过原R-Net调整重捕波束。
  可以在测量预处理、身份关联、轨迹管理等外围环节优化算法。
  第10步默认入口仍运行原N/R主线，没有切换到独立运动概率网络。
  V4虽然也使用Transformer，但替换了原N/R预测分支，只作为历史探索保存，后续不沿此分支继续。
  下文V4命令仅供复现历史版本。后续主线轨迹图以N-Net位置输出为蓝线；滤波位置另行标注。
  已核对core/models.py与V1源码归档字节完全一致；不存在原模型定义被覆盖的情况。

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

旧V4（历史探索）：独立运动概率网络，已停止作为主线
  此节描述历史归档，不是当前--maneuver-uncertainty入口。
默认复用已保存V4运动网络，重新创建结果/代码快照；--retrain 才重新训练12轮。
新网络输入20帧已获取历史，输出未来0.05、0.25、0.5、1秒的3分量位置分布。
前5帧初始化与旧网络相同，其他历史左侧填充为无效，不提供额外初始化测量。
验证集选择模型，独立校准集调整标准差尺度，测试集不用于选择参数。
新测试轨迹与V3不同；V4内部比较旧N/R网络、新运动均值+固定扩散、新概率分布。
后两者使用相同新权重与均值，区分使用概率信息的作用；旧N/R只作为迁移基线。
90%覆盖率为逐坐标边际覆盖率的平均值，不能当作三维/整条轨迹联合覆盖率。
当前能量化部分未来不确定性；急转弯仍偏自信，不能提前预知随机机动命令。
本版最终归档：experiments/v4_maneuver_uncertainty_r1；图表与报告在其中results/。

2026-11-16：每版必须保留的单轨迹仿真效果图
  第10步和V4入口现在自动生成 figures/single_trajectory_tracking.png。
  图中包含真实轨迹、实际位置估计、漏检位置、重捕点、单条跟踪时间/重捕率和全部场景汇总。
  蓝线在漏检/终止后留空，不平滑、不插值，不使用真实位置代替估计位置。
  历史V4图的蓝线是滤波位置；当前主线和重做V4图的蓝线统一为原N-Net实际位置输出。
  single_trajectory_tracking_samples.csv：图上使用的逐帧原始数值，仅供本地复核。
  single_trajectory_tracking.json：指标定义、示例编号、实际估计器和样本/图片/绘图代码校验值。
  V4默认固定示例400017（旧版中包含重捕事件，用于以后版本对照，不代表平均效果）。
  --plot-trajectory-id 可以显式指定其他实际存在的轨迹；其他默认多目标模式使用最小轨迹编号。
  V4运行结束自动显示单轨迹图和全部场景指标图；原有其他图继续保存。
  本次仅增加结果记录与绘图，不训练新网络，不调整波束选择或滤波参数。
  下一步：以原Transformer N-Net/R-Net为基础，建立没有真实目标编号的检测集合，真实身份只供评价，随后再加入身份关联。

重做V4：原Transformer N-Net/R-Net完全保留
  & '.\.venv\Scripts\python.exe' '.\my_training\step10_single_target_tracking.py' --maneuver-uncertainty --no-show
  当前入口改为core/maneuver_nnet_rnet.py。N/R结构、权重、输入输出及失跟路由均保持原样。
  复用旧V4的同一批51条机动测试轨迹，比较直接测量与鲁棒输入预处理，均共享12束。
  不使用独立运动概率网络；不重新训练，不对测试结果调参，不用全观测filtered_measurement数组。
  主图蓝线来自原N-Net位置头，初始化测量不冒充网络输出，漏检及终止后不补造位置。
  single_trajectory_tracking.png：固定ID400017的实际闭环轨迹、跟踪时间与重捕率。
  nr_tracking_metrics.png：两种输入处理的全场景指标；RMSE为原N-Net检测帧误差。
  beam_probability_reliability.png：原N/R的下一步波束概率评价；不代表三维位置/轨迹不确定性。
  network_integrity.json：原模型/权重校验值和实际N-Net/R-Net调用次数。
  默认结果根目录outputs/step10_v4_original_nr；每次成功运行自动创建新版本快照。
  旧V4各归档目录原样保留，不能把其概率网络结果当作本次原N/R结果。

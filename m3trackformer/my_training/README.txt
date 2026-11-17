第1～10步与原文多目标运行说明

V6：匿名身份关联真正接回闭环（Huber与原N-Net/R-Net不改）
  第10步加 --identity-closed-loop --no-show：验证集比较并锁定门限，随后测试，保存每个候选及全部图。
  加 --apply-selected：复用锁定配置，重跑三组固定噪声，保存新版本并显示两张图；单轨迹图只展示首组。
  当前阶段通过后，无参数点击第10步默认运行这个已锁定闭环；--current-real仍可显式运行理想身份Huber参照。
  主指标为正确身份的有效跟踪：任一初始身份连续5帧无正确关联即事后计时结束；控制器不读取这个真值判据。
  真实匹配结果进入Huber和原N/R历史，不再由正确身份维护历史；未关联不补造观测。
  R-Net只编码确认空回波的扫描波束；有回波但归属不确定的波束保持-1。
  图中紫菱形为错身份导致的实际N-Net位置输出；不能隐藏为正确轨迹。
  阶段验收：成功率损失≤5个百分点、时长保留≥95%、已接受关联正确率≥99.5%，是项目阶段门槛，不是通用标准。
  输出outputs/step10_identity_closed_loop；每个版本源码、指标、图和原始逐帧CSV独立保存，旧版本不覆盖。
  本版不增加杂波/随机检测漏失/目标出生消失；首先检查已有3个初始化目标的身份保持。

当前使用方案（2026-11-17用户指定）：Huber基准，不采用Student-t作为主线。
  Huber预处理保持不变；若已有通过阶段验收的身份闭环，无参数运行它；否则运行原真实轨迹Huber参照。
  --current-real：显式运行当前用户配置；--best-real保留为兼容入口，也优先使用当前用户配置。
  当前配置在outputs/current_tracking_configuration.json；用户选择记录单独锁定，不改写过去验证集结果。
  Huber阈值3、加速度标准差12、外部温度1；原Transformer N-Net/R-Net结构、权重及分工不变。
  --optimize-real仍可复现比较，不会覆盖上述用户配置。

波束贡献身份区分第一版：
  & '.\.venv\Scripts\python.exe' '.\my_training\step10_single_target_tracking.py' --beam-identity
  默认测试3个目标共享4束和12束，分别比较最高贡献直接归属、评分+几何门控+全局一对一匹配。
  --total-beams 4 / --total-beams 12：只运行一个预算。--no-show：后台保存，不弹窗。
  本版是条件身份判别审计，规则只接收匿名检出点；原Huber基准的过去历史仍保持理想关联。
  身份判断暂不反馈Huber或N/R历史，不能把本版跟踪率称为未知身份闭环效果。
  指标：全部检出点正确归属率、已分配精度、分配覆盖率、同束混叠正确率和归属切换。
  拒配计入全部检出点分母，真实编号只用于事后评价；同束多目标仍假设回波可分辨。
  输出在outputs/step10_beam_identity，版本快照在experiments/v5_*_beam_identity_b4或b12。
  identity_metrics.png / identity_confusion.png / identity_timeline.png：身份指标、混淆矩阵、逐帧归属。
  identity_spatial.png：实际匿名测量点按判别身份着色；红×为错配、灰点为拒配，不冒充N-Net轨迹。
  beam_contributions.png：展开各目标对选中波束的原网络评分贡献。
  single_trajectory_tracking.png：原Huber参照的实际N-Net轨迹，含时间和重捕率。
  identity_report.md解释范围；下一步才接回关联结果并检验错误累积。
  --replot-run 既有run目录：从原始CSV重新出图并保存新展示版本，旧版与仿真数值均保留。

真实轨迹多方案优化比较（原N-Net/R-Net不改、不重新训练）：
  第10步加 --optimize-real --no-show：验证集选参并锁定方案，再对照测试。
  第10步加 --best-real --no-show：优先读取用户当前配置；未指定时读取current_optimized_tracking.json。
  候选包括普通卡尔曼、Huber、Student-t式降权、自适应过程噪声、鲁棒双模型IMM、外部概率温度调整。
  先按验证集完整跟踪成功率、平均跟踪时间、检测覆盖率排序；测试集不参与选参或改选。
  验证集2个噪声种子，测试集3个噪声种子；同一分区各方法严格共享轨迹、扰动、预算和权重。
  每个候选独立归档，包含真实N-Net轨迹图；所有旧版本均保留。
  同样的17个测试场景重复添加噪声不等于增加独立场景数；测试记录已用过，属于复测。

2026-11-17 数据选择更新：后续主线采用真实飞行GNSS轨迹。
  原始压缩包位于datasets/idf_ds/，Speedybee DataSet.zip和Holybro Pixhawk.zip均保留。
  自行查看可先打开Speedybee DataSet.zip内的processed/GPS/Lap001_gps.csv。
  经纬度已是小数角度；时间time (us)需减首帧时间后除以1000000，得到相对秒数。
  该样本有2323行、约441.99秒，典型采样间隔约0.2秒；并非每个间隔完全相同。
  实测GNSS位置是参考轨迹；雷达测量仍另行仿真，不将其称为真实雷达测量数据。
  原V4机动入口仅用于生成轨迹历史实验；当前真实轨迹主线使用下述--real-trajectories入口。

真实轨迹＋距离/SNR误差＋异常值，一次运行完成两组对照：
  & '.\.venv\Scripts\python.exe' '.\my_training\step10_single_target_tracking.py' --real-trajectories
  无界面保存图片：同一命令加 --no-show。
  复用已按飞行记录组划分的50秒真实GPS切片，核对原始CSV重建结果，不改变原N/R结构、权重或切换规则。
  两组分别为：扰动测量直接输入；完全相同的扰动测量加鲁棒输入预处理。
  两组采用相同场景、相同12束预算和噪声种子，轨迹图蓝线均为N-Net实际位置输出。
  原始GPS约5Hz；20Hz输入来自插值。GNSS位置为参考估计，雷达角距测量是仿真。
  real_trajectory_sources.csv：每个真实切片的原始Lap文件、记录组、起止时间。
  real_trajectory_audit.json：原始ZIP校验、GPS重建及插值校验结果。
  批次结果位于outputs/step10_real_snr_outliers，每次生成独立raw/robust版本，旧版本不覆盖。
  结束自动显示鲁棒组单轨迹图与指标对照图，其余图和CSV同时保存。
  默认展示固定场景种子的首场景首目标，当前为轨迹729；不是按成功率或误差排名挑选。
  --plot-trajectory-id 可以指定其他实际存在的编号。编号193立即失跟的失败图也已另版保留。

2026-11-16 用户明确的当前主线约束：
  必须保留原Transformer N-Net与R-Net，网络结构、输入输出接口及正常/重捕分工不变。
  正常跟踪使用N-Net；失跟后通过原R-Net调整重捕波束。
  可以在测量预处理、身份关联、轨迹管理等外围环节优化算法。
  第10步默认入口仍运行原N/R主线，没有切换到独立运动概率网络。
  V4虽然也使用Transformer，但替换了原N/R预测分支，只作为历史探索保存，后续不沿此分支继续。
  下文V4命令仅供复现历史版本。后续主线轨迹图以N-Net位置输出为蓝线；滤波位置另行标注。
  已核对core/models.py与V1源码归档字节完全一致；不存在原模型定义被覆盖的情况。

第10步入口：step10_single_target_tracking.py。
历史未加测量预处理的原文多目标入口（显式多目标参数仍可调用）：
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

2026-11-17：真实轨迹上比较外围优化，原N-Net/R-Net完全冻结
  & '.\.venv\Scripts\python.exe' '.\my_training\step10_single_target_tracking.py' --best-real
  该次自动比较选择Student-t式创新降权（df=6）；用户随后指定继续使用Huber，当前入口优先用户配置。
  --best-real只重跑首个噪声种子；后台运行加 --no-show，不弹窗。
  --optimize-real重新完成15组验证配置与8组测试配置；验证2种子、测试3种子，不根据测试调参。
  选择顺序：完整跟踪成功率、平均同时跟踪时长、检测覆盖率。
  方法：普通卡尔曼、Huber调参、Student-t式降权、自适应过程噪声、鲁棒双模型IMM、外部温度。
  Student-t式/IMM均为声明的工程近似；滤波只预处理已获取测量，不替换原N/R网络或位置输出。
  总预算12束，逐目标Softmax后求和取Top-B；N/R正常/失跟分工保持一致。
  最新比较：outputs/step10_optimization/latest_complete_run.txt；已选方案重跑：latest_selected_run.txt。
  完整报告results_report.md、test_comparison.csv、optimization_comparison.png位于比较结果目录。
  current_optimized_tracking.json记录配置、选择锁定文件校验值及原N/R代码/权重校验值。
  17组测试轨迹组合×3组噪声不等于51个独立场景；此前测试记录已使用，不称为盲测。
  单轨迹图只绘固定首场景首目标和首组噪声，图中指标与三组噪声汇总口径不同。
  三组汇总：Student-t式成功率70.59%、平均39.10秒、重捕率82.91%；Huber为68.63%、38.15秒、84.09%。
  仅多1次完整成功，收益有限且重捕率略降；不声称统计显著或全部指标最优。
  所有试验均保存独立source.zip/manifest/results/轨迹图，旧版本不覆盖。

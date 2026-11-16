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

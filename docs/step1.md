# 第1步：部署并验证现有 Kalman 基线

本步做了什么：保持已有同步合成 Radar/Camera XYZ 实验，建立 D 盘 Python 环境、可双击启动入口和固定依赖记录；后续阶段以此数学基线继续。

运行：`python demo_step1.py --output output/local_step1`，Windows 可双击 `run_step1.cmd`。

验证：`python -m unittest discover -s tests -p test_kalman3d.py -v`，8 项全部通过。默认 seed42、20秒、201帧融合3D RMSE=0.11028075748317094m，与仓库已有参考在1e-12内一致，两张图已检查。

限制：本步两传感器都是相互独立的人工 XYZ 观测。没有真实雷达点云、相机像素、异步或作者实测数据，不能视为论文完整复现。后续步骤见各自文档。

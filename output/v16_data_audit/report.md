# MMAUD 真实数据获取与审计

日期：2026-10-07。来源：[作者官方页面](https://ntu-aris.github.io/MMAUD/)。仅使用作者发布的匿名文件页所提供的公开下载入口，不登录或绕过权限。原始材料位于 D 盘本项目 `data/public/mmaud`，不上传 GitHub。

Mavic3 归档共 10,750,360,115 字节；本次审计进程网络读取 0 字节（离线核验时为 0）。已核验目录中 11797 个成员，保存并校验 3470 个材料文件；另保留 186,464,349 字节归档局部跨度供再次提取。

| ZIP 文件夹 | 归档数量 | 已保存数量 | 首/末文件名时间（秒） |
|---|---:|---:|---|
| Mavic3/ground_truth | 833 | 833 | 1692846887.830421–1692847057.686649 |
| Mavic3/image | 5091 | 88 | 1692846887.841523–1692847057.669525 |
| Mavic3/lidar_360 | 1622 | 0 | 1692846887.899958–1692847057.600147 |
| Mavic3/livox_avia | 1698 | 0 | 1692846887.914906–1692847057.613956 |
| Mavic3/radar_enhance_pcl | 2548 | 2548 | 1692846887.84602–1692847057.645759 |

以下展示各类已保存文件的一个例子；全部逐文件 SHA-256 与 ZIP CRC32 见同目录 `manifest.json`。

| 已保存文件示例 | 字节 | SHA-256 |
|---|---:|---|
| data/public/mmaud/2023-08-24-11-14-40_mavic3.bag | 158950 | `c8e2adfb3528ce93dd2c37fe2ccf9a3853108a26bfee04845b78695f87cb0cf3` |
| data/public/mmaud/Mavic3/ground_truth/1692846887.830421.npy | 152 | `a21107f1eb7606b5469e5d6285f3fd76f03ff4d47348fcebfdf1b26fb9ff26d7` |
| data/public/mmaud/Mavic3/image/1692846889.005295.png | 2207491 | `1156a6a74bbefd612b1437502d5473e9d15dd78b178e30458bcb64b987616b3d` |
| data/public/mmaud/Mavic3/radar_enhance_pcl/1692846887.846020.npy | 8120 | `6e69148a619803524c3e3a71fb624913fd688369be03fe6d285822985579f9c7` |

材料获取与格式核验不等于算法精度验证。本获取审计未将真值作为雷达或相机观测；未执行压缩包内代码。后续若使用独立训练真值做固定标定，须明确披露，评估段真值只在两组滤波完成后用于评价。相机和雷达字段、坐标、时间及标定必须根据真实文件核验。

实际 enhanced radar 导出数组：2548 个 NPY，数据类型 ['<f8']，每帧 224–631 行、[3] 列。文件没有速度、强度、字段名、rig 外参；列的物理轴定义未在此审计中证明。目录中 1083 对相邻雷达文件 CRC 相同，不能把所有文件当作独立测量。

本 ZIP 仅包含 `radar_enhance_pcl`，未包含原始 `radar_pcl`、`radar_trk`、内外参或图像标签。文件名是导出的时间戳，不能据此宣称硬件同步。已下载相机 PNG 为双鱼眼拼接图；本材料审计不证明可直接转成原论文普通双目模型。论文所述 MATLAB 标定不能替代缺少的实际参数。

可复用入口：`tools/fetch_mmaud.py --offline --import-spans` 会只读已有 D 盘局部归档，验证局部成员名、压缩大小、原始大小与 ZIP CRC32，立即保存有效成员；`--members-file` 可传入 JSON 文件名清单。`--offline` 禁止网络请求。正常公开获取使用原作者分享页，Windows 匿名会话 cookie 只在进程内保留，不写入结果；下载按精确 HTTP 206/Content-Range 检查，单范围最多 8 MB，总预算可设。

数据许可：[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)，限非商业学术用途；上游软件MIT许可与数据许可分开。数据引用：Yuan et al., MMAUD: A Comprehensive Multi-Modal Anti-UAV Dataset for Modern Miniature Drone Threats, ICRA 2024, pp.2745–2751, [DOI](https://doi.org/10.1109/ICRA57147.2024.10610957)。来源与许可说明：[作者官网](https://ntu-aris.github.io/MMAUD/)。

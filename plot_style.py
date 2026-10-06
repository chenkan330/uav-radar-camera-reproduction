"""Shared Chinese figure labels and installed CJK font selection."""

METHOD_LABELS = {
    "strongest": "最强回波",
    "closest": "最近观测",
    "average": "算术平均",
    "kmeans": "单簇聚类",
    "weighted": "概率加权",
    "radar_only": "仅雷达",
}
MODE_LABELS = {
    "paper_xyz": "论文三维观测",
    "bearing": "像素方位观测",
    "none": "仅雷达",
}
DATA_KIND_LABELS = {"synthetic": "合成数据", "real": "实测数据"}


def configure_chinese_font():
    """Use an installed Chinese font; report missing fonts instead of tofu.

    No font is downloaded or bundled. Windows normally provides Microsoft
    YaHei; Linux/macOS can use Noto/Source Han/PingFang, respectively.
    Call after general rcParams styling and before creating a figure.
    """
    import matplotlib
    from matplotlib import font_manager

    candidates = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                  "Source Han Sans SC", "WenQuanYi Zen Hei", "PingFang SC",
                  "Heiti SC", "Arial Unicode MS")
    for family in candidates:
        try:
            font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        matplotlib.rcParams.update({"font.family": "sans-serif",
                                    "font.sans-serif": [family, "DejaVu Sans"],
                                    "axes.unicode_minus": False})
        return family
    raise RuntimeError("未找到支持中文的字体，请安装微软雅黑、思源黑体或 Noto Sans CJK SC 后重新绘图。")


def dataset_label(name):
    prefix = "synthetic_seed"
    return "合成数据·种子" + name[len(prefix):] if name.startswith(prefix) else name

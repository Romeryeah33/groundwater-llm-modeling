# -*- coding: utf-8 -*-
"""
配色与字体：web 界面与 matplotlib 图件的共同来源。

为什么放在 `app/utils/`（中性层）
--------------------------------
界面配色（`app/frontend/theme.py`）和图件配色（`app/analysis/visualizer.py`）
必须用**同一套值**，否则界面是靛蓝、图件是 tab10 彩虹，一眼就不像一个系统。
但两者之间不能互相 import：

    analysis/ ──✗──> frontend/      分析层依赖界面层，方向是反的
    frontend/ ──?──> analysis/      界面引入绘图库也不合理

所以下沉到中性层。这跟 `app/utils/grid_coords.py` 是同一个理由 —— 那个文件是
为了"建模用的原点"与"画图用的原点"不再各算一遍（见 docs/调试日志-202606.md）。

配色依据
--------
分类色取 NPG（Nature Publishing Group）标准 10 色：低饱和、印刷与屏幕都稳，
是顶刊插图里最常见的通用色板。强调色用其中的深靛蓝 #3C5488。

连续场用感知均匀色阶（viridis / Blues）：色盲安全、灰度打印仍能分档。
**不用 jet**：非感知均匀、色盲不可读、打印变泥。

字体
----
matplotlib 3.6+ 支持**逐字形回退**：把字体栈写成列表，拉丁字母/数字取第一个
命中的字体（Arial），中文取后面命中的 CJK 字体。这比"整张图统一用一个 CJK 字体"
好 —— 后者的西文和数字是黑体的拉丁字形，和期刊排版放在一起很突兀。
"""

from __future__ import annotations

import logging

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# 一、调色板（唯一来源；theme.py 从这里取，visualizer.py 也从这里取）
# --------------------------------------------------------------------------- #

# 强调色：NPG 深靛蓝
ACCENT = "#3C5488"
ACCENT_STRONG = "#2C3F6B"
ACCENT_SOFT = "#8491B4"
ACCENT_WASH = "#EDF0F7"

# NPG 标准 10 色分类色板（图件里区分井位/情景/过程时按序取用）
NPG10 = [
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
    "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85",
]

# NPG 红：阈值线、超限标记等需要"警示性高对比"的地方
NPG_RED = "#DC0000"

# 中性色：偏冷石墨灰阶
INK = "#141A24"
INK_2 = "#3B4553"
INK_3 = "#66707F"
INK_4 = "#98A2B0"
LINE = "#E4E8EE"
LINE_STRONG = "#CDD4DE"
SURFACE = "#FFFFFF"
SURFACE_2 = "#F7F8FA"
SURFACE_3 = "#EFF2F6"

# 状态色（-fg 文字档 / -bg 底色档 / -bd 描边档）
OK_FG, OK_BG, OK_BD = "#0F7B5F", "#EAF6F2", "#BFE3D8"
WARN_FG, WARN_BG, WARN_BD = "#8A5A00", "#FBF4E6", "#EBD9B0"
ERR_FG, ERR_BG, ERR_BD = "#A32222", "#FBEDED", "#EFC7C7"

# 连续场色阶
CMAP_HEAD = "Blues"     # 水头：浅底图上"高水头=深色"符合直觉
CMAP_PLUME = "viridis"  # 浓度：感知均匀、色盲安全、灰度可辨
CMAP_CONC = "viridis"   # 浓度（通用）
CMAP_DIFF = "RdBu_r"    # 差值：双向发散

# 字体栈
FONT_UI = '"Inter", "Segoe UI", "Microsoft YaHei", system-ui, sans-serif'
FONT_MONO = '"JetBrains Mono", Consolas, "Cascadia Mono", monospace'

# matplotlib 的拉丁字体（按可用性回退）与 CJK 候选。
# 刻意不放 Helvetica：本机没装，留在栈里只会刷一屏 "findfont: Font family
# 'Helvetica' not found." 的噪音日志（Arial 是它的等价替代）。
LATIN_FONT_STACK = ["Arial", "DejaVu Sans"]
CJK_FONT_CANDIDATES = [
    "Microsoft YaHei", "SimHei", "SimSun", "Source Han Sans SC",
    "Noto Sans CJK SC", "Noto Sans SC", "WenQuanYi Micro Hei",
]


def categorical(n: int) -> list[str]:
    """
    取 n 个分类色（NPG 10 色循环，超出 10 个时按序重复）。

    为什么不用 `plt.cm.tab10`：tab10 是 matplotlib 的默认观感，饱和度高、
    和顶刊色板并置时明显"不是一套"；而且它是**连续映射取点**（linspace），
    取 3 个时拿到的是色板首中尾，配色不协调。这里直接按序取，前 3 个色
    本身就是协调的三色组合。
    """
    if n <= 0:
        return []
    return [NPG10[i % len(NPG10)] for i in range(n)]


# --------------------------------------------------------------------------- #
# 二、字体探测
# --------------------------------------------------------------------------- #

def detect_cjk_font() -> str | None:
    """
    返回第一个**真实可用**的 CJK 字体名，找不到返回 None。

    用 `fallback_to_default=False` 是为了让"找不到"抛异常而不是静默返回
    DejaVu Sans —— 否则每个候选都会"命中"，中文实际渲染成方块却毫无提示。
    """
    for name in CJK_FONT_CANDIDATES:
        try:
            fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
            return name
        except Exception:
            continue
    return None


def apply_mpl_style() -> str | None:
    """
    设置全局 rcParams：字体栈 + 期刊版式。返回选中的 CJK 字体（可能为 None）。

    版式取向是"可印刷的科研插图"而非"演示文稿"：
      - 字号小（8.5~11pt）：期刊单栏图的实际尺寸常在 85mm 宽，字号大了就挤；
      - 线细（0.8~1.4pt）：避免版面上过重；
      - 网格淡而细、刻度朝外、图例无边框：让数据而不是容器占据注意力；
      - 四边框线用浅灰（--axes.edgecolor），不做"去上右边框"的处理 ——
        matplotlib **没有**控制边框可见性的 rcParam，逐个坐标轴去边要改
        9 个绘图函数，收益不抵风险；浅灰整框在版面上同样干净。
    """
    cjk = detect_cjk_font()
    stack = list(LATIN_FONT_STACK)
    if cjk:
        # 拉丁在前、CJK 在后：逐字形回退让西文取 Arial、中文取雅黑
        stack = [LATIN_FONT_STACK[0], cjk] + LATIN_FONT_STACK[1:]
        logger.info(f"matplotlib CJK 字体已配置: {cjk}（拉丁用 {LATIN_FONT_STACK[0]}）")
    else:
        logger.warning("未找到 CJK 字体，图表中文可能显示为方块。请安装中文字体。")

    plt.rcParams.update({
        # ---- 字体 ---- #
        # 必须把**具体字体名的列表**直接给 font.family。实测：若写成
        # font.family="sans-serif" + font.sans-serif=[...]，matplotlib 只会取
        # 该列表里第一个可解析到的字体（Arial），**不做逐字形回退** —— 中文会
        # 报 "Glyph 27700 missing from font(s) Arial" 并渲染成方块。直接给列表
        # 才会真正启用 3.6+ 的逐字形回退（西文 Arial、中文雅黑）。
        "font.family": stack,
        "font.sans-serif": stack,
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,   # 负号用 ASCII，避免雅黑缺字形变方块

        # ---- 分辨率 ---- #
        "figure.dpi": 150,
        "savefig.dpi": 150,

        # ---- 字号（期刊插图尺度）---- #
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,

        # ---- 线条与边框 ---- #
        "axes.linewidth": 0.8,
        "axes.edgecolor": LINE_STRONG,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        # 用 normal 而不是 medium：Arial 没有 500 字重，取 medium 会按 400 渲染
        # （视觉上完全一样），但每张图刷 3 行 "findfont: Failed to find font
        # weight medium"。层级改由字号区分（标题 11pt / 标签 10pt）。
        "axes.titleweight": "normal",
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "patch.linewidth": 0.8,

        # ---- 刻度：朝外、细、短 ---- #
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.color": INK_3,
        "ytick.color": INK_3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,

        # ---- 网格：淡、细、不抢视线（默认仍是关，各图自行 ax.grid 开启）---- #
        "grid.color": LINE,
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,

        # ---- 图例：无边框、无底色 ---- #
        "legend.frameon": False,

        # ---- 底色：纯白，便于直接放进文档 ---- #
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,

        # ---- 等值线：负值虚线（在读图上是通用约定）---- #
        "contour.negative_linestyle": "dashed",
    })
    return cjk


def mpl_version_note() -> str:
    """版本自检用的一句话（字体回退需要 matplotlib ≥ 3.6）。"""
    return f"matplotlib {matplotlib.__version__}，字体回退{'可用' if _supports_fallback() else '不可用（需 ≥3.6）'}"


def _supports_fallback() -> bool:
    try:
        parts = tuple(int(x) for x in matplotlib.__version__.split(".")[:2])
        return parts >= (3, 6)
    except Exception:
        return False

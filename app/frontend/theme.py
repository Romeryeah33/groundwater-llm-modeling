# -*- coding: utf-8 -*-
"""
界面主题：顶刊（NPG）配色 + 浅色默认。

色值从哪来
----------
**本文件不定义色值**，全部从 `app/utils/palette.py` 取。因为同一套颜色 web 界面、
Gradio 主题变量、matplotlib 图件三处都要用，而 `analysis/` 不该反过来 import
`frontend/`（分层方向），所以调色板放在中性层 `app/utils/palette.py`，
本文件只负责"怎么把颜色用到 Gradio 上"：

    palette.py ──► theme.py          build_theme()  Gradio 主题变量
                                 ├─ build_head()   :root{--gw-*} 令牌
                                 └─ 排版片段         panel_header / kpi_row / …
               ──► visualizer.py      CMAP_* 与图件 rcParams（直接 import）

`assets/gw_theme.css` **不写任何色值**：只用 var(--gw-*) 与 Gradio 自己那套变量，
所以换配色只改 palette.py 一处。
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from app.utils.palette import (
    ACCENT,
    ACCENT_SOFT,
    ACCENT_STRONG,
    ACCENT_WASH,
    ERR_BD,
    ERR_BG,
    ERR_FG,
    FONT_MONO,
    FONT_UI,
    INK,
    INK_2,
    INK_3,
    INK_4,
    LINE,
    LINE_STRONG,
    OK_BD,
    OK_BG,
    OK_FG,
    SURFACE,
    SURFACE_2,
    SURFACE_3,
    WARN_BD,
    WARN_BG,
    WARN_FG,
)

# CSS 变量名 → 色值。新增颜色只加在 palette.py，然后到 gw_theme.css 里用 var() 引用。
TOKENS: dict[str, str] = {
    "--gw-accent": ACCENT,
    "--gw-accent-strong": ACCENT_STRONG,
    "--gw-accent-soft": ACCENT_SOFT,
    "--gw-accent-wash": ACCENT_WASH,
    "--gw-ink": INK,
    "--gw-ink-2": INK_2,
    "--gw-ink-3": INK_3,
    "--gw-ink-4": INK_4,
    "--gw-line": LINE,
    "--gw-line-strong": LINE_STRONG,
    "--gw-surface": SURFACE,
    "--gw-surface-2": SURFACE_2,
    "--gw-surface-3": SURFACE_3,
    "--gw-ok-fg": OK_FG, "--gw-ok-bg": OK_BG, "--gw-ok-bd": OK_BD,
    "--gw-warn-fg": WARN_FG, "--gw-warn-bg": WARN_BG, "--gw-warn-bd": WARN_BD,
    "--gw-err-fg": ERR_FG, "--gw-err-bg": ERR_BG, "--gw-err-bd": ERR_BD,
    "--gw-font-ui": FONT_UI,
    "--gw-font-mono": FONT_MONO,
    # 阴影与"压在强调色上的文字"也走令牌，这样 gw_theme.css 里可以做到**零色值**
    "--gw-on-accent": "#FFFFFF",
    "--gw-shadow-1": "0 1px 2px rgba(20, 26, 36, 0.05)",
    "--gw-shadow-2": "0 6px 18px rgba(20, 26, 36, 0.07)",
    "--gw-ring": "0 0 0 3px rgba(60, 84, 136, 0.12)",
}

ASSETS = Path(__file__).parent / "assets"
CSS_PATH = ASSETS / "gw_theme.css"


# --------------------------------------------------------------------------- #
# 二、Gradio 主题
# --------------------------------------------------------------------------- #

def build_theme() -> gr.themes.Base:
    """
    构建 Gradio 主题。

    为什么用 `gr.themes.Base` 而不是 `Soft`/`Default`：后两者的圆角、阴影、间距是
    "入门模板"的观感（大圆角 + 明显投影 + 高饱和强调色），正是要摆脱的东西。
    Base 中性，全部靠 `.set()` 收窄。

    实测（Gradio 6.18）Gradio 内部组件类名（.tab-nav/.message/.bubble 等）**不在**
    shipped CSS 里 —— 组件样式在 JS chunk 内联。所以主力杠杆是这里的主题变量，
    次要杠杆是 elem_classes（见 gw_theme.css），而不是猜它的类名。
    """
    # 圆角/间距/字号三档必须走构造器：`set()` 里**没有** radius_*/spacing_*/text_*
    # 这些名字（实测 267 个可设置名里一个都没有），只能给整条 Scale 换掉。
    theme = gr.themes.Base(
        primary_hue=_navy_ramp(),
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "Segoe UI", "Microsoft YaHei", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "Consolas", "monospace"],
        # Size 的 name 是**末位关键字**参数，不是第一个 —— 位置传会整条错位一格
        # （--radius-xxs 会变成字符串 "gw" 这种非法值），必须写 name=
        radius_size=gr.themes.Size("2px", "3px", "5px", "7px", "9px", "11px", "13px", name="gw"),
        spacing_size=gr.themes.Size("2px", "4px", "7px", "10px", "14px", "20px", "28px", name="gw"),
        text_size=gr.themes.Size("10px", "11px", "12px", "13px", "15px", "18px", "22px", name="gw"),
    )

    return theme.set(
        # ---- 版式与底色 ------------------------------------------------- #
        body_background_fill=SURFACE_2,
        body_background_fill_dark="#12161D",
        body_text_color=INK,
        body_text_color_subdued=INK_3,
        block_background_fill=SURFACE,
        block_border_color=LINE,
        block_border_width="1px",
        block_label_background_fill=SURFACE,
        block_label_text_color=INK_3,
        block_label_text_weight="500",
        block_title_text_color=INK_2,
        block_title_text_weight="500",
        block_shadow="none",
        panel_background_fill=SURFACE,
        panel_border_width="1px",
        border_color_primary=LINE,
        border_color_accent=ACCENT_SOFT,

        # ---- 阴影：几乎不用，靠描边分层 --------------------------------- #
        shadow_drop="0 1px 2px rgba(20, 26, 36, 0.04)",
        shadow_drop_lg="0 4px 12px rgba(20, 26, 36, 0.06)",
        shadow_inset="none",

        # ---- 输入 ------------------------------------------------------- #
        input_background_fill=SURFACE,
        input_background_fill_focus=SURFACE,
        input_border_color=LINE_STRONG,
        input_border_color_focus=ACCENT,
        input_border_width="1px",
        input_placeholder_color=INK_4,
        input_shadow="none",
        input_shadow_focus="0 0 0 2px " + ACCENT_WASH,
        input_radius="5px",

        # ---- 按钮：主按钮实心靛蓝、次按钮白底描边 ------------------------ #
        button_primary_background_fill=ACCENT,
        button_primary_background_fill_hover=ACCENT_STRONG,
        button_primary_text_color="#FFFFFF",
        button_primary_border_color=ACCENT,
        button_secondary_background_fill=SURFACE,
        button_secondary_background_fill_hover=SURFACE_3,
        button_secondary_text_color=INK_2,
        button_secondary_border_color=LINE_STRONG,
        button_large_radius="5px",
        button_small_radius="4px",
        button_large_padding="7px 14px",
        button_small_padding="4px 9px",
        button_transition="background-color 120ms ease, border-color 120ms ease",

        # ---- 表格：表头浅灰、行分隔细线、无斑马纹 ------------------------ #
        table_border_color=LINE,
        table_even_background_fill=SURFACE,
        table_odd_background_fill=SURFACE_2,
        table_row_focus=SURFACE_3,
        table_radius="5px",

        # ---- 复选框 / 滑块 ---------------------------------------------- #
        checkbox_background_color=SURFACE,
        checkbox_background_color_selected=ACCENT,
        checkbox_border_color=LINE_STRONG,
        checkbox_border_color_selected=ACCENT,
        checkbox_label_background_fill=SURFACE,
        checkbox_label_text_color=INK_2,
        slider_color=ACCENT,
        slider_color_dark=ACCENT,

        # ---- 代码 / JSON ------------------------------------------------ #
        # 注意：Gradio 只暴露 --code-background-fill，**没有** code_text_color
        # （实测 267 个可设置名里没有），代码文字色在 gw_theme.css 里给 .prose code
        code_background_fill=SURFACE_3,
        code_background_fill_dark="#161B23",
        link_text_color=ACCENT,
        link_text_color_hover=ACCENT_STRONG,
        link_text_color_visited=ACCENT_SOFT,
    )


def _navy_ramp() -> gr.themes.Color:
    """
    以 NPG 深靛蓝 #3C5488 为 500 档，向两端插值出 11 档。

    为什么要造而不是用预置的 `indigo`：预置色板的 500 档是 #6366f1 那类高饱和紫蓝，
    "科技感"过头；NPG 的靛蓝更沉、饱和度低，是和顶刊插图放一起不打架的那个蓝。
    """
    return gr.themes.Color(
        c50="#F2F4F9", c100="#E4E8F1", c200="#C7CFE2",
        c300="#A3AFCB", c400="#6E7EA6", c500="#3C5488",
        c600="#354B7A", c700="#2C3F6B", c800="#223252",
        c900="#1A2740", c950="#121B2D", name="npg-navy",
    )


# --------------------------------------------------------------------------- #
# 三、注入 <head> 的内容
# --------------------------------------------------------------------------- #

def build_head() -> str:
    """
    返回注入页面的 `<head>` 片段：

    1. `:root{--gw-*}` —— 我们自己的 CSS 变量，色值来自本文件 TOKENS；
    2. 把初始明暗模式钉在浅色。

    第 2 点是必须的：**Gradio 6.18 没有服务端明暗开关**（`Blocks.__init__` 与
    `launch()` 都没有 theme_mode，`theme_mode` 只在 config 里当只读字段）。
    前端 `Index-*.js` 的取值顺序是：

        localStorage 里的偏好  ||  URL 的 ?__theme=  ||  "system"（跟随操作系统）

    所以系统是深色时，浅色设计会被整体盖掉。`head` 脚本在 Svelte 引导**之前**执行，
    此时补上 `?__theme=light` 即可让解析器读到 "light"；用 replaceState 改地址栏
    不产生跳转。放在 localStorage **之后**是有意的：用户若曾手工切过深色，
    尊重他的显式选择，不再强行拉回浅色。
    """
    tokens = "".join(f"{k}:{v};" for k, v in TOKENS.items())
    return (
        f"<style>:root{{{tokens}}}</style>\n"
        "<script>\n"
        "(function () {\n"
        "  try {\n"
        "    var u = new URL(window.location.toString());\n"
        "    if (u.searchParams.get('__theme') === null) {\n"
        "      u.searchParams.set('__theme', 'light');\n"
        "      history.replaceState(null, '', u.toString());\n"
        "    }\n"
        "  } catch (e) { /* 老浏览器：退回跟随系统，不影响功能 */ }\n"
        "})();\n"
        "</script>"
    )


# --------------------------------------------------------------------------- #
# 四、排版片段（gradio_app 用，把标记与类名收在本文件里）
# --------------------------------------------------------------------------- #

_ICONS: dict[str, str] = {
    # 24x24 线性图标，stroke 用 currentColor —— 颜色跟随文字，不单独配色
    "chat": "M4 5h16v11H8l-4 4z",
    "sliders": "M4 7h10M18 7h2M4 12h4M12 12h8M4 17h13M20 17h0",
    "doc": "M7 3h7l4 4v14H7zM14 3v4h4",
    "chart": "M4 20V4M4 20h16M8 16v-5M12 16V8M16 16v-8",
    "drop": "M12 3c3.5 4 6 7 6 10a6 6 0 0 1-12 0c0-3 2.5-6 6-10z",
    "beaker": "M8 3h8M9 3v7L5 18a2 2 0 0 0 2 3h10a2 2 0 0 0 2-3l-4-8V3",
    "image": "M3 5h18v14H3zM3 15l5-5 4 4 3-3 6 6",
    "file": "M6 3h8l4 4v14H6zM14 3v4h4",
    "play": "M7 4l12 8-12 8z",
    "check": "M4 12l5 5L20 6",
    "warn": "M12 3l9 17H3zM12 9v5M12 17h0",
    "info": "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM12 11v6M12 8h0",
    "download": "M12 3v12M7 11l5 5 5-5M4 20h16",
    "search": "M11 3a8 8 0 1 0 0 16 8 8 0 0 0 0-16zM21 21l-5-5",
    "wave": "M3 12c2.5 0 2.5-6 5-6s2.5 12 5 12 2.5-6 5-6h3",
    "grid": "M3 3h18v18H3zM9 3v18M15 3v18M3 9h18M3 15h18",
}


def icon(name: str, cls: str = "gw-ic") -> str:
    """线性 SVG 图标。取不到名字时返回空串（**不抛异常** —— 图标缺失不该弄崩界面）。"""
    path = _ICONS.get(name)
    if not path:
        return ""
    return (
        f'<svg class="{cls}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        f'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" '
        f'aria-hidden="true"><path d="{path}"/></svg>'
    )


def panel_header(zh: str, en: str, ic: str = "", right: str = "") -> str:
    """
    面板标题：中文主标签 + 等宽大写英文副标签。

    双语不是装饰：用户群体里"污染羽/plume""水头/head"这类术语中英混用是常态，
    副标签用等宽小字排，既提供英文对照又形成排版层次（无需靠加粗或颜色抢注意力）。
    `right` 放右侧的计数/状态小字。
    """
    right_html = f'<span class="gw-ph-right">{right}</span>' if right else ""
    return (
        f'<div class="gw-ph">{icon(ic) if ic else ""}'
        f'<span class="gw-ph-zh">{zh}</span>'
        f'<span class="gw-ph-en">{en}</span>'
        f'<span class="gw-ph-rule"></span>{right_html}</div>'
    )


def kpi_row(items: list[tuple]) -> str:
    """
    指标块一行。items 每项 (标签, 数值, 单位, 补充说明)。

    数字用等宽字体并做等宽对齐（`tabular-nums`）：同一量在不同时刻刷新时不会
    左右跳字，这是仪表盘的可用性底线。
    """
    cells = []
    for label, value, unit, hint in items:
        cells.append(
            '<div class="gw-kpi">'
            f'<div class="gw-kpi-label">{label}</div>'
            f'<div class="gw-kpi-value">{value}'
            f'<span class="gw-kpi-unit">{unit}</span></div>'
            f'<div class="gw-kpi-hint">{hint}</div>'
            "</div>"
        )
    return f'<div class="gw-kpi-row">{"".join(cells)}</div>'


def telemetry(rows: list[tuple]) -> str:
    """
    遥测式键值列表。rows 每项 (键, 值, 强调等级)。

    等级："" 普通 / "strong" 加重 / "muted" 弱化 / "ok" "warn" "err" 状态色。
    比一行行 <b>键:</b> 值 更好扫读 —— 值右对齐成一列，一眼能比大小。
    """
    body = []
    for key, value, tone in rows:
        t = f" gw-tel-{tone}" if tone else ""
        body.append(
            f'<div class="gw-tel-row"><span class="gw-tel-key">{key}</span>'
            f'<span class="gw-tel-val{t}">{value}</span></div>'
        )
    return f'<div class="gw-tel">{"".join(body)}</div>'


def callout(kind: str, title: str, body: str) -> str:
    """提示块。kind ∈ {ok, warn, err, info}。body 允许含少量内联标签。"""
    ic = {"ok": "check", "warn": "warn", "err": "warn", "info": "info"}.get(kind, "info")
    return (
        f'<div class="gw-callout gw-callout-{kind}">'
        f'<span class="gw-callout-ic">{icon(ic)}</span>'
        f'<div><div class="gw-callout-title">{title}</div>'
        f'<div class="gw-callout-body">{body}</div></div></div>'
    )


def empty_state(title: str, hint: str) -> str:
    """空态占位：说明"这里将会出现什么、需要先做什么"，而不是留白或写"暂无数据"。"""
    return (
        f'<div class="gw-empty"><div class="gw-empty-title">{title}</div>'
        f'<div class="gw-empty-hint">{hint}</div></div>'
    )

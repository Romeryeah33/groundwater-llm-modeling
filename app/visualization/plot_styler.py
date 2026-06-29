"""
自然语言绘图修改引擎

用户用自然语言描述对图件的修改需求 → LLM 解析为结构化样式参数 → 重新渲染

用法:
    styler = PlotStyler()

    # 保存当前图件数据
    styler.save_context(head=head, delr=delr, delc=delc, wells=wells)

    # 自然语言修改
    new_fig, explanation = styler.apply_nl(
        "突出显示等水位线，将等水位线线宽设置为3，间隔设置为10m/根"
    )
"""
import json
import logging
import copy
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from app.llm_agent.llm_client import llm_client

logger = logging.getLogger(__name__)


# LLM prompt: 将用户NL转换为结构化样式参数
STYLE_PARSING_PROMPT = """你是matplotlib绘图样式专家。将用户对地下水模拟图件的修改需求转换为JSON格式的样式参数。

当前图件类型: {plot_type}
当前参数: {current_params}

请将用户的修改需求转换为JSON。只修改用户提到的参数，未提及的参数保留null。
支持的参数:
- contour_levels: 等值线填充层数 (int)
- contour_lines: 等值线数量 (int)
- contour_linewidth: 等值线线宽 (float)
- contour_interval: 等值线间隔(m) (float, 设置此项时contour_levels自动计算)
- cmap: 配色方案名称 (str, 如 "Blues"/"YlOrRd"/"coolwarm"/"viridis")
- title: 图标题 (str)
- title_fontsize: 标题字号 (int)
- label_fontsize: 轴标签字号 (int)
- tick_fontsize: 刻度字号 (int)
- font_family: 字体族 (str, 如 "Times New Roman"/"SimHei"/"Arial", null=不修改)
- font_weight: 字体粗细 (str, "bold"/"normal", null=不修改)
- show_arrows: 是否显示流向箭头 (bool)
- arrow_density: 箭头密度 (int, 1-30)
- well_marker_size: 井标记大小 (float)
- well_marker_color: 井标记颜色 (str, 如 "black"/"red"/"darkgreen")
- well_marker_shape: 井标记形状 (str, "^"=三角形/ "o"=圆形/ "s"=方形)
- highlight_wells: 是否突出显示井 (bool)
- grid_visible: 是否显示网格 (bool)
- dpi: 图片分辨率 (int)

注意：用户说"Times New Roman"时font_family应为"Times New Roman"，说"加粗"时font_weight应为"bold"。
"标注抽水井、显示井"等需求应设置highlight_wells=true。
"黑色三角形"对应well_marker_color="black", well_marker_shape="^"。

用户需求: {user_request}

返回纯JSON（不要markdown代码块）:"""


class PlotStyler:
    """自然语言绘图修改器"""

    # 默认样式
    DEFAULT_STYLE = {
        "contour_levels": 15,
        "contour_lines": 10,
        "contour_linewidth": 0.8,
        "contour_interval": None,  # None = 自动计算
        "cmap": "Blues",
        "title": "Hydraulic Head Distribution",
        "title_fontsize": 12,
        "label_fontsize": 11,
        "tick_fontsize": 10,
        "font_family": None,      # None = 不修改
        "font_weight": None,      # None = 不修改
        "show_arrows": True,
        "arrow_density": 15,
        "well_marker_size": 10.0,
        "well_marker_color": "red",
        "well_marker_shape": "v",
        "highlight_wells": False,
        "grid_visible": False,
        "dpi": 150,
    }

    def __init__(self):
        # 当前图件上下文
        self._head: Optional[np.ndarray] = None
        self._delr: Optional[np.ndarray] = None
        self._delc: Optional[np.ndarray] = None
        self._wells: Optional[list] = None
        self._conc_frames: Optional[list] = None
        self._current_style: dict = copy.deepcopy(self.DEFAULT_STYLE)
        self._plot_type: str = "head"

    # ================================================================
    # 保存/获取上下文
    # ================================================================

    def save_context(
        self,
        head: np.ndarray = None,
        delr: np.ndarray = None,
        delc: np.ndarray = None,
        wells: list = None,
        conc_frames: list = None,
        plot_type: str = "head",
    ):
        """保存当前图件数据和样式"""
        if head is not None:
            self._head = head
        if delr is not None:
            self._delr = delr
        if delc is not None:
            self._delc = delc
        if wells is not None:
            self._wells = wells
        if conc_frames is not None:
            self._conc_frames = conc_frames
        self._plot_type = plot_type

    def has_data(self) -> bool:
        return self._head is not None

    def get_current_style(self) -> dict:
        return copy.deepcopy(self._current_style)

    # ================================================================
    # 核心：NL → 样式参数 → 重新渲染
    # ================================================================

    def parse_nl_to_style(self, user_request: str) -> Tuple[dict, str]:
        """
        将用户自然语言转换为结构化样式参数

        Returns:
            (style_changes, explanation) — 只包含需要修改的字段
        """
        current = json.dumps(self._current_style, ensure_ascii=False)
        prompt = STYLE_PARSING_PROMPT.format(
            plot_type=self._plot_type,
            current_params=current,
            user_request=user_request,
        )

        try:
            response = llm_client.chat(prompt)
            # 清理 LLM 可能包裹的 markdown 代码块
            json_str = response.strip()
            if json_str.startswith("```"):
                json_str = json_str.split("\n", 1)[1]
                if json_str.endswith("```"):
                    json_str = json_str[:-3]
            style_changes = json.loads(json_str)
        except Exception as e:
            # 回退：返回空修改
            logger.warning(f"LLM 样式解析失败: {e}, 原始响应: {response[:200] if 'response' in dir() else 'N/A'}")
            return {}, f"无法解析修改需求，请换一种说法重试。（错误: {e}）"

        if not isinstance(style_changes, dict):
            return {}, "解析结果格式异常，请重试。"

        return style_changes

    def apply_changes(
        self,
        style_changes: dict,
        head: np.ndarray = None,
        delr: np.ndarray = None,
        delc: np.ndarray = None,
        wells: list = None,
    ) -> plt.Figure:
        """
        将样式修改应用到图件并重新渲染

        Args:
            style_changes: LLM 返回的样式修改字典
            head/delr/delc/wells: 可选覆盖数据

        Returns:
            新的 matplotlib Figure
        """
        h = head if head is not None else self._head
        dr = delr if delr is not None else self._delr
        dc = delc if delc is not None else self._delc
        w = wells if wells is not None else self._wells

        if h is None:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.text(0.5, 0.5, "无可用数据", ha="center", va="center", transform=ax.transAxes)
            return fig

        # 合并样式
        style = copy.deepcopy(self._current_style)
        for key, val in style_changes.items():
            if val is not None and key in style:
                style[key] = val

        # 构建图件
        fig, ax = plt.subplots(figsize=(10, 7))

        # 字体设置
        font_kwargs = {}
        if style.get("font_family"):
            font_kwargs["family"] = style["font_family"]
        if style.get("font_weight"):
            font_kwargs["weight"] = style["font_weight"]

        nrow, ncol = h.shape
        y_edges = np.concatenate([[0], np.cumsum(dc)])
        x_edges = np.concatenate([[0], np.cumsum(dr)])
        X, Y = np.meshgrid(
            (x_edges[:-1] + x_edges[1:]) / 2,
            (y_edges[:-1] + y_edges[1:]) / 2,
        )

        # 确定等值线层级
        if style["contour_interval"] is not None:
            interval = float(style["contour_interval"])
            h_min = np.floor(h.min() / interval) * interval
            h_max = np.ceil(h.max() / interval) * interval
            levels = np.arange(h_min, h_max + interval, interval)
        else:
            levels = np.linspace(h.min(), h.max(), style["contour_levels"])

        # 填色
        cmap_name = style["cmap"]
        cf = ax.contourf(X, Y, h, levels=levels, cmap=cmap_name, alpha=0.85)
        cbar = fig.colorbar(cf, ax=ax, shrink=0.9)
        cbar.set_label("Head (m)", fontsize=style["label_fontsize"], **font_kwargs)

        # 等值线
        lw = style["contour_linewidth"]
        line_levels = np.linspace(h.min(), h.max(), style["contour_lines"])
        cs = ax.contour(X, Y, h, levels=line_levels, colors="black", linewidths=lw)
        ax.clabel(cs, inline=True, fontsize=style["tick_fontsize"] - 2, fmt="%.1f")

        # 流向箭头
        if style["show_arrows"]:
            gy, gx = np.gradient(h)
            gy = -gy / dc.reshape(-1, 1)
            gx = -gx / dr.reshape(1, -1)
            step = max(1, min(nrow, ncol) // style["arrow_density"])
            ax.quiver(
                X[::step, ::step], Y[::step, ::step],
                gx[::step, ::step], gy[::step, ::step],
                color="darkblue", alpha=0.6, scale=50, width=0.002,
            )

        # 井 (始终显示井 — 如果用户要求突出则加大标记和标签)
        if w:
            msize = style["well_marker_size"]
            hlight = style["highlight_wells"]
            mcolor = style.get("well_marker_color", "black")
            mshape = style.get("well_marker_shape", "^")
            for well in w:
                r, c = well["row"], well["col"]
                rate = well.get("rate", 0)
                wx = x_edges[c] + np.mean(dr) / 2
                wy = y_edges[r] + np.mean(dc) / 2
                ax.plot(
                    wx, wy, mshape,
                    color=mcolor, markersize=msize if hlight else msize * 0.7,
                    markeredgecolor="black" if hlight else mcolor,
                    markeredgewidth=1.5 if hlight else 0.5,
                )
                # 始终标注流量（有井就应该标注）
                label = well.get("label", f"Q={rate:.0f}")
                ax.annotate(
                    label, (wx, wy),
                    fontsize=style["tick_fontsize"], **font_kwargs,
                    xytext=(5, 5), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8) if hlight else None,
                )

        ax.set_xlabel("X (m)", fontsize=style["label_fontsize"], **font_kwargs)
        ax.set_ylabel("Y (m)", fontsize=style["label_fontsize"], **font_kwargs)
        ax.set_title(style["title"], fontsize=style["title_fontsize"], **font_kwargs)
        ax.tick_params(labelsize=style["tick_fontsize"])
        ax.set_aspect("equal")

        if style["grid_visible"]:
            ax.grid(True, linestyle="--", alpha=0.3)

        # 保存新样式
        self._current_style = style

        return fig

    def apply_nl(
        self,
        user_request: str,
    ) -> Tuple[Optional[plt.Figure], str]:
        """
        一站式：接收自然语言 → 解析 → 应用 → 返回新图

        Returns:
            (new_figure, explanation_message)
        """
        if not self.has_data():
            return None, "⚠️ 没有可修改的图件数据，请先建立模型生成图件。"

        # Step 1: 解析 NL
        style_changes = self.parse_nl_to_style(user_request)
        if not style_changes:
            return None, "⚠️ 无法理解修改需求，请换一种方式描述。"

        # Step 2: 应用修改
        try:
            new_fig = self.apply_changes(style_changes)
        except Exception as e:
            logger.error(f"应用样式修改失败: {e}")
            return None, f"❌ 应用修改时出错: {e}"

        # Step 3: 生成说明
        change_desc = []
        for k, v in style_changes.items():
            if v is not None:
                old = self._current_style.get(k, "?")
                change_desc.append(f"  • {k}: {old} → {v}")
        explanation = "✅ 已应用修改:\n" + "\n".join(change_desc) if change_desc else "✅ 图件已更新"

        return new_fig, explanation

    def reset_style(self):
        """重置为默认样式"""
        self._current_style = copy.deepcopy(self.DEFAULT_STYLE)

    def generate_code(self) -> str:
        """生成当前样式的完整 matplotlib 绘图代码"""
        s = self._current_style
        nrow = self._head.shape[0] if self._head is not None else 40
        ncol = self._head.shape[1] if self._head is not None else 50

        lines = [
            "# -*- coding: utf-8 -*-",
            "# 地下水水头等值线图 — 自动生成代码",
            "# 可通过修改参数后重新运行此代码来调整图件",
            "",
            "import numpy as np",
            "import matplotlib.pyplot as plt",
            "",
            f"# ===== 数据（需替换为实际模拟结果）=====",
            f"# head: np.ndarray, shape=({nrow}, {ncol})  # 水头数据",
            f"# delr: np.ndarray, shape=({ncol},)          # X方向网格间距",
            f"# delc: np.ndarray, shape=({nrow},)          # Y方向网格间距",
            f"# wells: list[dict] = [...]                   # 井数据",
            "",
            f"# ===== 样式参数 =====",
        ]

        for k, v in s.items():
            if v is not None:
                if isinstance(v, str):
                    lines.append(f"{k} = \"{v}\"")
                else:
                    lines.append(f"{k} = {v}")

        lines += [
            "",
            f"# ===== 绘图 =====",
            f"fig, ax = plt.subplots(figsize=(10, 7))",
            "",
            f"# 网格坐标",
            f"y_edges = np.concatenate([[0], np.cumsum(delc)])",
            f"x_edges = np.concatenate([[0], np.cumsum(delr)])",
            f"X, Y = np.meshgrid(",
            f"    (x_edges[:-1] + x_edges[1:]) / 2,",
            f"    (y_edges[:-1] + y_edges[1:]) / 2,",
            f")",
            "",
        ]

        if s["contour_interval"] is not None:
            lines += [
                f"# 等值线（定间隔）",
                f"interval = {s['contour_interval']}",
                f"h_min = np.floor(head.min() / interval) * interval",
                f"h_max = np.ceil(head.max() / interval) * interval",
                f"levels = np.arange(h_min, h_max + interval, interval)",
            ]
        else:
            lines += [
                f"# 等值线（自动分层）",
                f"levels = np.linspace(head.min(), head.max(), {s['contour_levels']})",
            ]

        font_kwargs = []
        if s.get("font_family"):
            font_kwargs.append(f"family=\"{s['font_family']}\"")
        if s.get("font_weight"):
            font_kwargs.append(f"weight=\"{s['font_weight']}\"")
        font_str = ", ".join(font_kwargs)

        lines += [
            "",
            f"# 填色",
            f"cf = ax.contourf(X, Y, head, levels=levels, cmap=\"{s['cmap']}\", alpha=0.85)",
            f"cbar = fig.colorbar(cf, ax=ax, shrink=0.9)",
            f"cbar.set_label(\"Head (m)\", fontsize={s['label_fontsize']}{', ' + font_str if font_str else ''})",
            "",
            f"# 等值线",
            f"line_levels = np.linspace(head.min(), head.max(), {s['contour_lines']})",
            f"cs = ax.contour(X, Y, head, levels=line_levels, colors=\"black\", linewidths={s['contour_linewidth']})",
            f"ax.clabel(cs, inline=True, fontsize={s['tick_fontsize'] - 2}, fmt=\"%.1f\")",
        ]

        if s["show_arrows"]:
            lines += [
                "",
                f"# 流向箭头",
                f"gy, gx = np.gradient(head)",
                f"gy = -gy / delc.reshape(-1, 1)",
                f"gx = -gx / delr.reshape(1, -1)",
                f"step = max(1, min({nrow}, {ncol}) // {s['arrow_density']})",
                f"ax.quiver(X[::step, ::step], Y[::step, ::step],",
                f"           gx[::step, ::step], gy[::step, ::step],",
                f"           color=\"darkblue\", alpha=0.6, scale=50, width=0.002)",
            ]

        if self._wells:
            lines += [
                "",
                f"# 井",
                f"for w in wells:",
                f"    wx = x_edges[w[\"col\"]] + np.mean(delr) / 2",
                f"    wy = y_edges[w[\"row\"]] + np.mean(delc) / 2",
                f"    ax.plot(wx, wy, \"{s.get('well_marker_shape', '^')}\",",
                f"            color=\"{s.get('well_marker_color', 'black')}\",",
                f"            markersize={s['well_marker_size']}, markeredgewidth=0.5)",
                f"    ax.annotate(w.get(\"label\", f\"Q={{w['rate']:.0f}}\"), (wx, wy),",
                f"                fontsize={s['tick_fontsize']}{', ' + font_str if font_str else ''},",
                f"                xytext=(5, 5), textcoords=\"offset points\",",
                f"                bbox=dict(boxstyle=\"round,pad=0.3\", facecolor=\"white\", alpha=0.8))",
            ]

        lines += [
            "",
            f"ax.set_xlabel(\"X (m)\", fontsize={s['label_fontsize']}{', ' + font_str if font_str else ''})",
            f"ax.set_ylabel(\"Y (m)\", fontsize={s['label_fontsize']}{', ' + font_str if font_str else ''})",
            f"ax.set_title(\"{s['title']}\", fontsize={s['title_fontsize']}{', ' + font_str if font_str else ''})",
            f"ax.tick_params(labelsize={s['tick_fontsize']})",
            f"ax.set_aspect(\"equal\")",
        ]
        if s["grid_visible"]:
            lines.append("ax.grid(True, linestyle=\"--\", alpha=0.3)")

        lines += [
            "",
            "plt.tight_layout()",
            "# plt.savefig(\"head_contour.png\", dpi=150, bbox_inches=\"tight\")",
            "# plt.show()",
        ]

        return "\n".join(lines)


# 全局实例
plot_styler = PlotStyler()

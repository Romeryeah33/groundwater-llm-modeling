"""
图件修改引擎 — 自然语言驱动的 matplotlib 图件样式修改

通过自然语言指令修改水头等值线图、污染羽图的样式，
使用 LLM 将 NL → 结构化样式参数 → 重新渲染 matplotlib。
"""

import json
import logging
import copy
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.llm_agent.llm_client import llm_client

logger = logging.getLogger(__name__)

# LLM Prompt: 将用户自然语言转换为结构化样式修改
STYLE_PARSING_PROMPT = """将用户对图件的修改需求转换为结构化样式参数。

## 当前图件样式
{current_style}

## 用户修改需求
{user_request}

## 支持的样式参数一览

| 参数名 | 类型 | 说明 | 可选值/范围 |
|--------|------|------|-------------|
| title | str | 图件标题 | 任意字符串 |
| font_family | str | 字体族 | sans-serif, serif, monospace, SimHei, Microsoft YaHei |
| font_weight | str | 字体粗细 | normal, bold, light |
| fontsize | int | 字体大小 | 8~36 |
| label_fontsize | int | 轴标签字体大小 | 8~36 |
| tick_fontsize | int | 刻度标签字体大小 | 6~24 |
| title_fontsize | int | 标题字体大小 | 10~48 |
| cmap | str | 颜色映射 | viridis, plasma, Blues, RdBu, coolwarm, jet, terrain |
| contour_interval | float | 等值线间隔(m) | 0.1~100 |
| contour_levels | int | 等值线分层数 | 5~100 |
| contour_lines | int | 等值线线条数 | 5~100（与 contour_levels 配合使用） |
| contour_linewidth | float | 等值线线宽 | 0.3~3.0 |
| show_arrows | bool | 是否显示流向箭头 | true / false |
| arrow_density | int | 箭头密度 | 5~50（值越小越密） |
| well_marker_size | int | 井标记大小 | 20~200 |
| well_marker_color | str | 井标记颜色 | red, blue, green, black, darkblue |
| well_marker_shape | str | 井标记形状 | ^, o, s, D, *, v |
| highlight_wells | bool | 是否高亮井位 | true / false |
| grid_visible | bool | 是否显示网格线 | true / false |
| figsize_width | float | 图件宽度(英寸) | 6~20 |
| figsize_height | float | 图件高度(英寸) | 4~16 |
| dpi | int | 分辨率 | 72~300 |

## 规则
- 只返回用户提到的需要修改的参数，不要返回所有参数
- 参数名使用与上方一致的英文名
- 返回 JSON 对象，不要额外解释

返回格式：
```json
{{
  "style_changes": {{
    "标题": "新标题",
    "cmap": "Blues",
    ...
  }},
  "explanation": "已应用：修改标题为'新标题'，颜色映射改为 Blues"
}}
```"""


class PlotStyler:
    """自然语言驱动的图件样式修改器"""

    def __init__(self):
        self.llm = llm_client
        # 默认样式
        self.default_style = {
            "title": "Groundwater Head Distribution",
            "font_family": "sans-serif",
            "font_weight": "normal",
            "fontsize": 11,
            "label_fontsize": 12,
            "tick_fontsize": 10,
            "title_fontsize": 14,
            "cmap": "viridis",
            "contour_interval": None,  # None = 自动计算
            "contour_levels": 15,
            "contour_lines": 8,
            "contour_linewidth": 0.6,
            "show_arrows": True,
            "arrow_density": 25,
            "well_marker_size": 80,
            "well_marker_color": "red",
            "well_marker_shape": "^",
            "highlight_wells": True,
            "grid_visible": False,
            "figsize_width": 10,
            "figsize_height": 7,
            "dpi": 100,
        }
        self.style = copy.deepcopy(self.default_style)
        # 保存的上下文
        self._head = None
        self._delr = None
        self._delc = None
        self._wells = None
        self._conc_frames = None
        self._plot_type = "head"

    # ================================================================
    # 上下文管理
    # ================================================================

    def save_context(self, head=None, delr=None, delc=None,
                     wells=None, conc_frames=None, plot_type="head"):
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
        if plot_type:
            self._plot_type = plot_type
        logger.debug(f"PlotStyler 上下文已保存: head={self._head is not None}, "
                     f"wells={len(self._wells) if self._wells else 0}")

    def has_data(self) -> bool:
        """是否有可用的绘图数据"""
        return self._head is not None

    def get_current_style(self) -> dict:
        """获取当前完整样式"""
        return copy.deepcopy(self.style)

    # ================================================================
    # NL → 样式解析
    # ================================================================

    def parse_nl_to_style(self, user_request: str) -> Tuple[Optional[dict], str]:
        """
        将用户自然语言转换为结构化样式参数

        Returns:
            (style_changes, explanation) — 只包含需要修改的字段
        """
        current = json.dumps(self.style, ensure_ascii=False, indent=2)
        prompt = STYLE_PARSING_PROMPT.format(
            current_style=current,
            user_request=user_request,
        )
        try:
            response = self.llm.chat(prompt)
            # 提取 JSON 块
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            parsed = json.loads(json_str.strip())

            style_changes = parsed.get("style_changes", {})
            explanation = parsed.get("explanation", "图件已更新")
            return style_changes, explanation

        except json.JSONDecodeError as e:
            logger.warning(f"LLM 样式解析失败: {e}, 原始响应: {response[:200]}")
            return None, f"无法解析修改需求，请换一种说法重试。（错误: {e}）"
        except Exception as e:
            logger.error(f"LLM 样式解析异常: {e}")
            return None, f"解析结果格式异常，请重试。"

    # ================================================================
    # 应用样式 → 重新渲染
    # ================================================================

    def apply_changes(self, style_changes: dict,
                      head=None, delr=None, delc=None, wells=None):
        """
        将样式修改应用到图件并重新渲染

        Args:
            style_changes: LLM 返回的样式修改字典
            head/delr/delc/wells: 可选覆盖数据

        Returns:
            新的 matplotlib Figure
        """
        if not self.has_data() and head is None:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "无可用数据", ha="center", va="center",
                    transform=ax.transAxes)
            return fig

        h = head if head is not None else self._head
        dr = delr if delr is not None else self._delr
        dc = delc if delc is not None else self._delc
        w = wells if wells is not None else self._wells

        # 合并样式修改
        self._merge_style(style_changes)

        # 重建图形
        fig, ax = plt.subplots(
            figsize=(self.style["figsize_width"], self.style["figsize_height"]),
            dpi=self.style["dpi"],
        )

        # 网格坐标
        y_edges = np.concatenate([[0], np.cumsum(dc)])
        x_edges = np.concatenate([[0], np.cumsum(dr)])
        X, Y = np.meshgrid(
            (x_edges[:-1] + x_edges[1:]) / 2,
            (y_edges[:-1] + y_edges[1:]) / 2,
        )

        # 等值线分层
        interval = self.style.get("contour_interval")
        if interval:
            h_min = np.floor(np.nanmin(h) / interval) * interval
            h_max = np.ceil(np.nanmax(h) / interval) * interval
            levels = np.arange(h_min, h_max + interval, interval)
        else:
            levels = np.linspace(np.nanmin(h), np.nanmax(h),
                                self.style["contour_levels"])

        # 填色
        cf = ax.contourf(X, Y, h, levels=levels,
                         cmap=self.style["cmap"], alpha=0.85)
        cbar = fig.colorbar(cf, ax=ax, shrink=0.9)
        cbar.set_label("Head (m)", fontsize=self.style["label_fontsize"])

        # 等值线
        line_levels = np.linspace(np.nanmin(h), np.nanmax(h),
                                  self.style["contour_lines"])
        cs = ax.contour(X, Y, h, levels=line_levels,
                        colors="black", linewidths=self.style["contour_linewidth"])
        ax.clabel(cs, inline=True, fontsize=self.style["tick_fontsize"], fmt="%.1f")

        # 流向箭头
        if self.style["show_arrows"]:
            gy, gx = np.gradient(h)
            gy = -gy / dc.reshape(-1, 1)
            gx = -gx / dr.reshape(1, -1)
            step = max(1, min(h.shape) // self.style["arrow_density"])
            ax.quiver(X[::step, ::step], Y[::step, ::step],
                       gx[::step, ::step], gy[::step, ::step],
                       color="darkblue", alpha=0.6, scale=50, width=0.002)

        # 井
        if w and self.style["highlight_wells"]:
            for well in w:
                if isinstance(well, dict):
                    wx = x_edges[well.get("col", 0)] + np.mean(dr) / 2
                    wy = y_edges[well.get("row", 0)] + np.mean(dc) / 2
                    ax.plot(wx, wy,
                            self.style["well_marker_shape"],
                            color=self.style["well_marker_color"],
                            markersize=self.style["well_marker_size"],
                            markeredgewidth=0.5)
                    label = well.get("label",
                                     f"Q={well.get('rate', 0):.0f}")
                    ax.annotate(label, (wx, wy),
                                fontsize=self.style["fontsize"],
                                xytext=(5, 5), textcoords="offset points",
                                bbox=dict(boxstyle="round,pad=0.3",
                                          facecolor="white", alpha=0.8))

        # 标签和标题
        ax.set_xlabel("X (m)", fontsize=self.style["label_fontsize"])
        ax.set_ylabel("Y (m)", fontsize=self.style["label_fontsize"])
        ax.set_title(self.style["title"], fontsize=self.style["title_fontsize"])
        ax.tick_params(labelsize=self.style["tick_fontsize"])
        ax.set_aspect("equal")

        if self.style["grid_visible"]:
            ax.grid(True, linestyle="--", alpha=0.3)

        plt.tight_layout()
        return fig

    def _merge_style(self, changes: dict):
        """将修改合并到当前样式"""
        # 参数名映射（中文 → 英文）
        key_map = {
            "标题": "title", "字体": "font_family", "字体大小": "fontsize",
            "颜色映射": "cmap", "颜色": "cmap", "colormap": "cmap",
            "等值线间隔": "contour_interval", "等值线数量": "contour_levels",
            "等值线线宽": "contour_linewidth", "等值线条数": "contour_lines",
            "箭头": "show_arrows", "流向箭头": "show_arrows",
            "箭头密度": "arrow_density", "井大小": "well_marker_size",
            "井颜色": "well_marker_color", "井形状": "well_marker_shape",
            "高亮井": "highlight_wells", "网格": "grid_visible",
            "宽度": "figsize_width", "高度": "figsize_height",
            "分辨率": "dpi", "标题大小": "title_fontsize",
            "轴标签大小": "label_fontsize", "刻度大小": "tick_fontsize",
        }
        normalized = {}
        for k, v in changes.items():
            mapped = key_map.get(k, k)
            normalized[mapped] = v

        for k, v in normalized.items():
            if k in self.style:
                # 类型转换
                if isinstance(self.style[k], bool) and isinstance(v, str):
                    v = v.lower() in ("true", "yes", "on", "1", "显示", "是")
                elif isinstance(self.style[k], (int, float)) and isinstance(v, str):
                    try:
                        v = float(v) if "." in v else int(v)
                    except ValueError:
                        continue
                elif isinstance(self.style[k], (int, float)) and isinstance(v, bool):
                    continue  # 不转换 bool → 数值
                self.style[k] = v

    # ================================================================
    # NL → 完整管道
    # ================================================================

    def apply_nl(self, user_request: str) -> Tuple[object, str]:
        """
        一站式：接收自然语言 → 解析 → 应用 → 返回新图

        Returns:
            (new_figure, explanation_message)
        """
        style_changes, explanation = self.parse_nl_to_style(user_request)
        if style_changes is None:
            return None, explanation

        try:
            new_fig = self.apply_changes(style_changes)
            # 构建详细的修改说明
            change_desc = []
            for k, v in style_changes.items():
                old = self.style.get(k, "?")
                change_desc.append(f"  • {k}: {old} → {v}")
            msg = f"✅ 已应用修改:\n" + "\n".join(change_desc)
            return new_fig, msg
        except Exception as e:
            logger.error(f"应用样式修改失败: {e}")
            return None, f"❌ 应用修改时出错: {e}"

    def reset_style(self):
        """重置为默认样式"""
        self.style = copy.deepcopy(self.default_style)
        logger.debug("样式已重置为默认值")

    # ================================================================
    # 代码生成
    # ================================================================

    def generate_code(self) -> str:
        """生成当前样式的完整 matplotlib 绘图代码"""
        lines = [
            "# -*- coding: utf-8 -*-",
            "# 地下水水头等值线图 — 自动生成代码",
            "# 可通过修改参数后重新运行此代码来调整图件",
            "",
            "import numpy as np",
            "import matplotlib.pyplot as plt",
            "",
            "# ===== 数据（需替换为实际模拟结果）=====",
            f"# head: np.ndarray, shape=({self._head.shape if self._head is not None else '?'})  # 水头数据",
            f"# delr: np.ndarray, shape=(?,)          # X方向网格间距",
            f"# delc: np.ndarray, shape=(?,)          # Y方向网格间距",
            "# wells: list[dict] = [...]                   # 井数据",
            "",
            "# ===== 样式参数 =====",
        ]
        for k, v in self.style.items():
            if isinstance(v, str):
                lines.append(f'{k} = "{v}"')
            else:
                lines.append(f"{k} = {v}")

        lines.extend([
            "",
            "# ===== 绘图 =====",
            "fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))",
            "",
            "# 网格坐标",
            "y_edges = np.concatenate([[0], np.cumsum(delc)])",
            "x_edges = np.concatenate([[0], np.cumsum(delr)])",
            "X, Y = np.meshgrid(",
            "    (x_edges[:-1] + x_edges[1:]) / 2,",
            "    (y_edges[:-1] + y_edges[1:]) / 2,",
            ")",
            "",
        ])
        if self.style.get("contour_interval"):
            lines.extend([
                "# 等值线（定间隔）",
                "interval = contour_interval",
                "h_min = np.floor(head.min() / interval) * interval",
                "h_max = np.ceil(head.max() / interval) * interval",
                "levels = np.arange(h_min, h_max + interval, interval)",
            ])
        else:
            lines.append("# 等值线（自动分层）")
            lines.append("levels = np.linspace(head.min(), head.max(), contour_levels)")

        lines.extend([
            "",
            "# 填色",
            'cf = ax.contourf(X, Y, head, levels=levels, cmap=cmap, alpha=0.85)',
            "cbar = fig.colorbar(cf, ax=ax, shrink=0.9)",
            'cbar.set_label("Head (m)", fontsize=label_fontsize)',
            "",
            "# 等值线",
            "line_levels = np.linspace(head.min(), head.max(), contour_lines)",
            'cs = ax.contour(X, Y, head, levels=line_levels, colors="black", linewidths=contour_linewidth)',
            'ax.clabel(cs, inline=True, fontsize=tick_fontsize, fmt="%.1f")',
            "",
        ])

        if self.style.get("show_arrows"):
            lines.extend([
                "# 流向箭头",
                "gy, gx = np.gradient(head)",
                "gy = -gy / delc.reshape(-1, 1)",
                "gx = -gx / delr.reshape(1, -1)",
                "step = max(1, min(head.shape) // arrow_density)",
                "ax.quiver(X[::step, ::step], Y[::step, ::step],",
                "           gx[::step, ::step], gy[::step, ::step],",
                '           color="darkblue", alpha=0.6, scale=50, width=0.002)',
                "",
            ])

        lines.extend([
            "# 井",
            "for w in wells:",
            '    wx = x_edges[w["col"]] + np.mean(delr) / 2',
            '    wy = y_edges[w["row"]] + np.mean(delc) / 2',
            "    ax.plot(wx, wy, well_marker_shape,",
            "            color=well_marker_color,",
            "            markersize=well_marker_size, markeredgewidth=0.5)",
            '    ax.annotate(w.get("label", f"Q={w[\"rate\"]:.0f}"), (wx, wy),',
            "                fontsize=fontsize,",
            '                xytext=(5, 5), textcoords="offset points",',
            '                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))',
            "",
            'ax.set_xlabel("X (m)", fontsize=label_fontsize)',
            'ax.set_ylabel("Y (m)", fontsize=label_fontsize)',
            "ax.set_title(title, fontsize=title_fontsize)",
            "ax.tick_params(labelsize=tick_fontsize)",
            'ax.set_aspect("equal")',
        ])

        if self.style.get("grid_visible"):
            lines.append('ax.grid(True, linestyle="--", alpha=0.3)')

        return "\n".join(lines)


# 全局实例
plot_styler = PlotStyler()

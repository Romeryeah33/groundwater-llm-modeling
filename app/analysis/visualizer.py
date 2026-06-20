"""
可视化引擎

自动生成标准图件：
- 水位等值线图（含流向箭头）
- 污染羽时空演化序列图
- 剖面浓度分布图
- 质量平衡桑基图
- 主控过程贡献率图
- 穿透曲线
"""

import logging
from typing import Optional, List, Dict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互后端
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm

logger = logging.getLogger(__name__)

# ── CJK 中文字体配置 ──
# Windows: 优先 SimHei（黑体）→ Microsoft YaHei → SimSun → 回退 DejaVu Sans
_CJK_FONT_CANDIDATES = ["SimHei", "Microsoft YaHei", "SimSun", "WenQuanYi Micro Hei",
                         "Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans SC"]
_available_cjk = None
for _font_name in _CJK_FONT_CANDIDATES:
    try:
        fm.findfont(fm.FontProperties(family=_font_name), fallback_to_default=False)
        _available_cjk = _font_name
        break
    except Exception:
        continue

if _available_cjk:
    plt.rcParams["font.family"] = _available_cjk
    logger.info(f"matplotlib CJK 字体已配置: {_available_cjk}")
else:
    logger.warning("未找到 CJK 字体，图表中文可能显示为方块。请安装中文字体。")

# 全局样式设置
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.unicode_minus": False,  # 避免负号显示为方块
})


class VisualizationEngine:
    """可视化引擎"""

    # 推荐配色方案
    CMAP_HEAD = "Blues"
    CMAP_CONC = "YlOrRd"
    CMAP_PLUME = "Reds"
    CMAP_DIFF = "RdBu_r"

    def __init__(self, output_dir: Optional[Path] = None):
        from app.config import settings
        self.output_dir = output_dir or settings.output_dir / "figures"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # 水头可视化
    # ================================================================

    def plot_head_contour(
        self,
        head: np.ndarray,
        delr: np.ndarray,
        delc: np.ndarray,
        title: str = "Hydraulic Head Distribution",
        contour_levels: int = 15,
        wells: Optional[List[dict]] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        水位等值线图（含流向箭头）

        Args:
            head: 二维水头数组 (nrow, ncol)
            delr/delc: 网格尺寸
            wells: [{"row": int, "col": int, "rate": float, "label": str}, ...]
        """
        nrow, ncol = head.shape
        fig, ax = plt.subplots(figsize=(10, 7))

        # 网格坐标
        y_edges = np.concatenate([[0], np.cumsum(delc)])
        x_edges = np.concatenate([[0], np.cumsum(delr)])
        X, Y = np.meshgrid(
            (x_edges[:-1] + x_edges[1:]) / 2,
            (y_edges[:-1] + y_edges[1:]) / 2,
        )

        # 水头填色
        levels = np.linspace(head.min(), head.max(), contour_levels)
        cf = ax.contourf(X, Y, head, levels=levels, cmap=self.CMAP_HEAD, alpha=0.85)
        cbar = fig.colorbar(cf, ax=ax, label="Head (m)", shrink=0.9)

        # 等值线
        cs = ax.contour(X, Y, head, levels=10, colors="black", linewidths=0.5)
        ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f")

        # 流向箭头（基于水力梯度）
        gy, gx = np.gradient(head)
        gy = -gy / delc.reshape(-1, 1)
        gx = -gx / delr.reshape(1, -1)

        # 降采样箭头避免过密
        step = max(1, min(nrow, ncol) // 15)
        ax.quiver(
            X[::step, ::step], Y[::step, ::step],
            gx[::step, ::step], gy[::step, ::step],
            color="darkblue", alpha=0.6, scale=50, width=0.002,
        )

        # 标记抽/注水井
        if wells:
            for w in wells:
                r, c = w["row"], w["col"]
                rate = w["rate"]
                color = "red" if rate < 0 else "blue"
                marker = "v" if rate < 0 else "^"
                ax.plot(x_edges[c] + np.mean(delr)/2, y_edges[r] + np.mean(delc)/2,
                        marker, color=color, markersize=10, markeredgecolor="black")
                label = w.get("label", f"Q={rate:.0f}")
                ax.annotate(label, (x_edges[c], y_edges[r]), fontsize=8,
                            xytext=(5, 5), textcoords="offset points")

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(title)
        ax.set_aspect("equal")

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        return fig

    # ================================================================
    # 污染羽演化
    # ================================================================

    def plot_plume_evolution(
        self,
        conc_frames: List[np.ndarray],
        times: List[float],
        delr: np.ndarray,
        delc: np.ndarray,
        threshold: float = 1.0,
        n_cols: int = 3,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        污染羽时空演化序列图

        Args:
            conc_frames: 各时刻浓度场列表
            times: 时刻列表
            threshold: 污染羽边界阈值
        """
        n_frames = len(conc_frames)
        n_rows = int(np.ceil(n_frames / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
        axes = np.atleast_1d(axes).flatten()

        y_edges = np.concatenate([[0], np.cumsum(delc)])
        x_edges = np.concatenate([[0], np.cumsum(delr)])
        X, Y = np.meshgrid(
            (x_edges[:-1] + x_edges[1:]) / 2,
            (y_edges[:-1] + y_edges[1:]) / 2,
        )

        vmax = max(np.max(c) for c in conc_frames)

        for i, (conc, ax) in enumerate(zip(conc_frames, axes)):
            masked = np.ma.masked_where(conc < threshold * 0.1, conc)
            cf = ax.contourf(X, Y, masked, levels=20,
                             cmap=self.CMAP_PLUME, vmin=0, vmax=vmax)
            # 污染羽边界
            ax.contour(X, Y, conc, levels=[threshold],
                       colors="red", linewidths=1.5, linestyles="--")
            ax.set_title(f"t = {times[i]:.0f} d")
            ax.set_aspect("equal")

        # 隐藏多余子图
        for j in range(n_frames, len(axes)):
            axes[j].set_visible(False)

        fig.colorbar(cf, ax=axes[:n_frames], label="Concentration (mg/L)",
                     shrink=0.8, pad=0.02)
        fig.suptitle("Plume Evolution", fontsize=14, y=0.98)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        return fig

    # ================================================================
    # 穿透曲线
    # ================================================================

    def plot_breakthrough_curve(
        self,
        times: List[float],
        concentrations: Dict[str, List[float]],
        title: str = "Breakthrough Curves",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """穿透曲线（监测点浓度时间序列）"""
        fig, ax = plt.subplots(figsize=(8, 5))

        colors = plt.cm.tab10(np.linspace(0, 1, len(concentrations)))
        for (label, conc), color in zip(concentrations.items(), colors):
            ax.plot(times, conc, "-o", color=color, label=label,
                    markersize=4, linewidth=1.5)

        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Concentration (mg/L)")
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        return fig

    # ================================================================
    # 过程贡献率图
    # ================================================================

    def plot_process_contribution(
        self,
        process_analysis: dict,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        主控过程贡献率图

        Args:
            process_analysis: {
                "advection": 45.2, "dispersion": 20.1,
                "decay": 25.3, "sorption": 9.4
            }
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        processes = list(process_analysis.keys())
        contributions = list(process_analysis.values())

        colors = plt.cm.Set2(np.linspace(0, 1, len(processes)))

        # 饼图
        wedges, texts, autotexts = ax1.pie(
            contributions, labels=processes, autopct="%1.1f%%",
            colors=colors, startangle=90, pctdistance=0.8,
        )
        ax1.set_title("Process Contribution (Pie)")

        # 柱状图
        bars = ax2.barh(processes, contributions, color=colors)
        ax2.set_xlabel("Contribution (%)")
        ax2.set_title("Process Contribution (Bar)")
        ax2.invert_yaxis()
        for bar, val in zip(bars, contributions):
            ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                     f"{val:.1f}%", va="center", fontsize=9)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        return fig

    # ================================================================
    # 质量平衡桑基图数据
    # ================================================================

    def plot_water_balance(
        self,
        budget: dict,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        水量平衡组件分解图

        Args:
            budget: ResultProcessor.extract_water_budget() 的输出
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        inflow = budget.get("inflow", {})
        outflow = budget.get("outflow", {})

        all_labels = list(inflow.keys()) + ["---"] + list(outflow.keys())
        all_values = list(inflow.values()) + [0] + list(outflow.values())

        y_pos = range(len(all_labels))

        colors_in = [plt.cm.Blues(0.6 + 0.4 * i / max(len(inflow), 1))
                     for i in range(len(inflow))]
        colors_out = [plt.cm.Oranges(0.6 + 0.4 * i / max(len(outflow), 1))
                      for i in range(len(outflow))]
        all_colors = colors_in + ["white"] + colors_out

        bars = ax.barh(y_pos, all_values, color=all_colors, edgecolor="black")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(all_labels)
        ax.set_xlabel("Flow Rate (m³/d)")
        ax.set_title(f"Water Balance (Error: {budget.get('balance_error', 0)*100:.4f}%)")

        # 在柱子上标注值
        for bar, val in zip(bars, all_values):
            if val > 0:
                ax.text(bar.get_width() + max(all_values) * 0.02,
                        bar.get_y() + bar.get_height()/2,
                        f"{val:.1f}", va="center", fontsize=9)

        ax.axvline(0, color="black", linewidth=0.5)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        return fig

    # ================================================================
    # 综合仪表盘
    # ================================================================

    def plot_dashboard(
        self,
        head: np.ndarray,
        plume_frames: List[np.ndarray],
        times: List[float],
        plume_metrics: dict,
        delr: np.ndarray,
        delc: np.ndarray,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """综合结果仪表盘（四合一图）"""
        fig = plt.figure(figsize=(16, 12))

        # 1. 水头分布 (左上)
        ax1 = fig.add_subplot(2, 2, 1)
        y_edges = np.concatenate([[0], np.cumsum(delc)])
        x_edges = np.concatenate([[0], np.cumsum(delr)])
        X, Y = np.meshgrid((x_edges[:-1]+x_edges[1:])/2, (y_edges[:-1]+y_edges[1:])/2)
        cf1 = ax1.contourf(X, Y, head, levels=15, cmap=self.CMAP_HEAD)
        ax1.contour(X, Y, head, levels=8, colors="black", linewidths=0.5)
        plt.colorbar(cf1, ax=ax1, label="Head (m)")
        ax1.set_title("Head Distribution")
        ax1.set_aspect("equal")

        # 2. 最终污染羽 (右上)
        ax2 = fig.add_subplot(2, 2, 2)
        final_plume = np.ma.masked_where(plume_frames[-1] < 0.01, plume_frames[-1])
        cf2 = ax2.contourf(X, Y, final_plume, levels=20, cmap=self.CMAP_PLUME)
        plt.colorbar(cf2, ax=ax2, label="Conc (mg/L)")
        ax2.set_title(f"Final Plume (t={times[-1]:.0f} d)")
        ax2.set_aspect("equal")

        # 3. 羽面积/最大浓度 vs 时间 (左下)
        ax3 = fig.add_subplot(2, 2, 3)
        ax3_twin = ax3.twinx()
        l1, = ax3.plot(times, plume_metrics["plume_area_m2"], "b-o", markersize=4)
        l2, = ax3_twin.plot(times, plume_metrics["max_conc"], "r-s", markersize=4)
        ax3.set_xlabel("Time (d)")
        ax3.set_ylabel("Plume Area (m²)", color="blue")
        ax3_twin.set_ylabel("Max Concentration (mg/L)", color="red")
        ax3.legend([l1, l2], ["Area", "Max Conc"], loc="best")
        ax3.set_title("Plume Metrics vs Time")
        ax3.grid(True, alpha=0.3)

        # 4. 质心轨迹 (右下)
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(plume_metrics["centroid_x"], plume_metrics["centroid_y"], "k-o", markersize=5)
        ax4.scatter(plume_metrics["centroid_x"][0], plume_metrics["centroid_y"][0],
                    s=100, c="green", marker="o", label="Start", zorder=5)
        ax4.scatter(plume_metrics["centroid_x"][-1], plume_metrics["centroid_y"][-1],
                    s=100, c="red", marker="s", label="End", zorder=5)
        ax4.set_xlabel("X (m)")
        ax4.set_ylabel("Y (m)")
        ax4.set_title("Plume Centroid Trajectory")
        ax4.legend()
        ax4.set_aspect("equal")
        ax4.grid(True, alpha=0.3)

        fig.suptitle("Modeling Results Dashboard", fontsize=16, y=0.99)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        return fig


# 全局实例
visualizer = VisualizationEngine()

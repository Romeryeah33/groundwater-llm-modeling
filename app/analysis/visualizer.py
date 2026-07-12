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

    # ================================================================
    # 底图叠加可视化
    # ================================================================

    def plot_head_on_basemap(
        self,
        head: np.ndarray,
        delr: np.ndarray,
        delc: np.ndarray,
        basemap_path: Optional[str] = None,
        shp_bounds: Optional[list] = None,
        shp_path: Optional[str] = None,
        clip_shp_path: Optional[str] = None,
        title: str = "Hydraulic Head Distribution",
        contour_levels: int = 15,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        在底图上叠加水头等值线，支持 SHP 边界裁剪。

        Args:
            head: 二维水头数组 (nrow, ncol)
            delr/delc: 网格尺寸
            basemap_path: 底图图片路径 (PNG/JPG/TIF)，可选
            shp_bounds: [xmin, ymin, xmax, ymax] 用于地理参考底图
            shp_path: SHP 文件路径（线要素），用于叠加边界线
            clip_shp_path: SHP 文件路径（面要素），用于裁剪等值线
            title: 图标题
            contour_levels: 等值线层数
            save_path: 保存路径
        """
        import matplotlib.image as mpimg
        import os
        nrow, ncol = head.shape

        # 模型网格坐标（m）— 使用从 SHP 提取的范围
        if shp_bounds and len(shp_bounds) >= 4:
            x_min, y_min, x_max, y_max = shp_bounds[0], shp_bounds[1], shp_bounds[2], shp_bounds[3]
        else:
            # 回退：使用 delr/delc 累积坐标
            x_min, y_min = 0.0, 0.0
            x_max = sum(delr) if hasattr(delr, '__iter__') else delr * ncol
            y_max = sum(delc) if hasattr(delc, '__iter__') else delc * nrow

        y_edges = np.linspace(y_min, y_max, nrow + 1)
        x_edges = np.linspace(x_min, x_max, ncol + 1)
        X, Y = np.meshgrid(
            (x_edges[:-1] + x_edges[1:]) / 2,
            (y_edges[:-1] + y_edges[1:]) / 2,
        )
        model_extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

        fig, ax = plt.subplots(figsize=(12, 9))

        # ── 1. 加载底图 ──
        basemap_loaded = False
        if basemap_path and os.path.exists(basemap_path):
            try:
                ext = os.path.splitext(basemap_path)[1].lower()
                img = None
                img_extent = None

                if ext in ('.tif', '.tiff'):
                    # GeoTIFF: 使用 rasterio 读取并自动获取地理参考
                    try:
                        import rasterio
                        from rasterio.plot import show as rio_show
                        with rasterio.open(basemap_path) as src:
                            # 读取 RGB 波段（假设前3个波段为RGB）
                            n_bands = src.count
                            if n_bands >= 3:
                                img_data = src.read([1, 2, 3])
                                img = np.transpose(img_data, (1, 2, 0))
                            else:
                                img_data = src.read(1)
                                img = img_data
                            # 地理范围直接从 GeoTIFF 获取
                            img_extent = [src.bounds.left, src.bounds.right,
                                         src.bounds.bottom, src.bounds.top]
                            logger.info(f"GeoTIFF 底图加载成功: {src.width}x{src.height}, "
                                       f"范围={img_extent}")
                    except ImportError:
                        logger.warning("rasterio 未安装，尝试 PIL 读取 GeoTIFF")
                    except Exception as e:
                        logger.warning(f"rasterio 读取失败: {e}")

                if img is None:
                    # 回退：使用 matplotlib/PIL 读取（可能不支持地理参考）
                    img = mpimg.imread(basemap_path)
                    # 对于非 GeoTIFF 图片，使用 SHP 范围或模型范围
                    if img_extent is None:
                        if shp_bounds and len(shp_bounds) >= 4:
                            img_extent = [shp_bounds[0], shp_bounds[2],
                                         shp_bounds[1], shp_bounds[3]]
                        else:
                            img_extent = model_extent

                # 显示底图
                if img is not None:
                    ax.imshow(img, extent=img_extent or model_extent,
                             aspect='auto', alpha=0.65, zorder=0, cmap='gray' if img.ndim == 2 else None)
                    basemap_loaded = True
                    logger.info(f"底图已加载: {basemap_path}")
            except Exception as e:
                logger.warning(f"底图加载失败: {e}")

        # ── 2. 叠加 SHP 边界线（线要素，如 Boundary_arcs） ──
        if shp_path and os.path.exists(shp_path):
            try:
                import geopandas as gpd
                gdf = gpd.read_file(shp_path)
                gdf.boundary.plot(ax=ax, edgecolor='red', linewidth=2.0,
                                  alpha=0.9, zorder=2, label='Boundary Line')
                logger.info(f"SHP 边界线已叠加: {shp_path} ({len(gdf)} 要素)")
            except Exception as e:
                logger.warning(f"SHP 边界线叠加失败: {e}")

        # ── 2b. 加载裁剪面要素（面要素，如 Rivers_polys / Aquifer Layer 1_polys） ──
        boundary_polygon = None
        if clip_shp_path and os.path.exists(clip_shp_path):
            try:
                import geopandas as gpd
                gdf = gpd.read_file(clip_shp_path)
                if len(gdf) > 0:
                    boundary_polygon = gdf.geometry.iloc[0]
                # 绘制面边界
                gdf.boundary.plot(ax=ax, edgecolor='black', linewidth=2.5,
                                  alpha=0.9, zorder=2, label='Model Domain')
                logger.info(f"SHP 裁剪面已加载: {clip_shp_path} ({len(gdf)} 要素, geom={boundary_polygon.geom_type if boundary_polygon else 'N/A'})")
            except Exception as e:
                logger.warning(f"SHP 裁剪面加载失败: {e}")

        # ── 3. 水头填色（半透明，裁剪到边界） ──
        # 如果有 SHP 边界多边形，用它来 mask 网格数据
        masked_head = head
        if boundary_polygon is not None and hasattr(boundary_polygon, 'contains'):
            try:
                from matplotlib.path import Path as MplPath
                # 将 Shapely 多边形转为 matplotlib Path
                if boundary_polygon.geom_type == 'Polygon':
                    xy = np.array(boundary_polygon.exterior.coords)
                    boundary_path = MplPath(xy)
                    # 对每个网格单元中心检查是否在边界内
                    mask = np.ones_like(head, dtype=bool)
                    for i in range(nrow):
                        for j in range(ncol):
                            mask[i, j] = boundary_path.contains_point((X[i, j], Y[i, j]))
                    masked_head = np.ma.masked_where(~mask, head)
                    logger.info(f"边界裁剪完成: {mask.sum()}/{mask.size} 个网格单元在边界内")
            except Exception as e:
                logger.warning(f"边界裁剪失败: {e}")

        levels = np.linspace(masked_head.min() if hasattr(masked_head, 'min') else head.min(),
                             masked_head.max() if hasattr(masked_head, 'max') else head.max(),
                             contour_levels)
        cf = ax.contourf(X, Y, masked_head, levels=levels, cmap=self.CMAP_HEAD,
                         alpha=0.5 if basemap_loaded else 0.85, zorder=1)
        cbar = fig.colorbar(cf, ax=ax, label="Head (m)", shrink=0.85)

        # ── 4. 等值线 ──
        cs = ax.contour(X, Y, masked_head, levels=10, colors="black",
                        linewidths=0.5, zorder=3)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")

        # ── 5. 流向箭头 ──
        gy, gx = np.gradient(head)
        gy = -gy / delc.reshape(-1, 1)
        gx = -gx / delr.reshape(1, -1)
        step = max(1, min(nrow, ncol) // 15)
        ax.quiver(
            X[::step, ::step], Y[::step, ::step],
            gx[::step, ::step], gy[::step, ::step],
            color="darkblue", alpha=0.5, scale=50, width=0.002, zorder=4,
        )

        # ── 6. 标签和标题 ──
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(title)
        ax.set_aspect("equal")
        if basemap_loaded:
            ax.legend(loc='upper right', fontsize=8)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        return fig


# 全局实例
visualizer = VisualizationEngine()

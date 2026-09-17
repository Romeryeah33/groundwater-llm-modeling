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

from app.utils.grid_coords import as_sizes, grid_cell_centers, warn_extent_mismatch

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
    # 网格坐标与底图机制（水头图 / 污染羽图共用）
    # ================================================================

    @staticmethod
    def _as_sizes(del_):
        """把 delr/delc 归一化为一维定长 ndarray"""
        return as_sizes(del_)

    def _grid_coords(
        self,
        nrow: int,
        ncol: int,
        delr: np.ndarray,
        delc: np.ndarray,
        shp_bounds: Optional[list] = None,
    ):
        """
        统一的网格坐标基准。

        历史问题：水头图用 ``linspace(shp_bounds)`` 把网格**拉伸**去铺满 SHP
        外框，污染羽图用 ``cumsum`` 从 0 起算。两张图叠同一张底图时必然错位，
        且非均匀网格下拉伸会静默扭曲格子宽高比。这里统一为
        "真实格子尺寸 + SHP 原点偏移"：均匀网格且 SHP 外框等于模型域时，
        与原 linspace 结果逐元素相同。

        **实现已下沉到 `app.utils.grid_coords.grid_cell_centers`**：源区选格
        （`app.modeling.source_zone.cells_in_geometry`）必须与这里的绘图坐标
        逐元素一致，否则会出现"模型补给的格子"与"图上画在填埋场轮廓内的格子"
        不是同一批这种最难查的错位。而 `app.modeling` 不能反向依赖绘图层，
        故两边共用中性层的同一份坐标数学；`tests/test_source_zone.py` 有
        `np.array_equal` 全等断言兜底。

        Args:
            shp_bounds: SHP 外框 ``[xmin, ymin, xmax, ymax]``；取前两位作原点

        Returns:
            (X, Y, model_extent)：单元中心坐标网格与 [xmin, xmax, ymin, ymax]
        """
        origin = None
        if shp_bounds and len(shp_bounds) >= 4:
            origin = [float(shp_bounds[0]), float(shp_bounds[1])]

        X, Y, model_extent = grid_cell_centers(
            nrow, ncol, delr, delc, origin_xy=origin
        )
        # 模型域与 SHP 外框不吻合时明确告警：底图会偏，不静默
        # 传本模块的 logger，使告警仍然归属 app.analysis.visualizer
        warn_extent_mismatch(model_extent, shp_bounds, logger)
        return X, Y, model_extent

    def _draw_basemap(
        self,
        ax,
        X: np.ndarray,
        Y: np.ndarray,
        model_extent: list,
        basemap_path: Optional[str] = None,
        shp_bounds: Optional[list] = None,
        shp_path: Optional[str] = None,
        clip_shp_path: Optional[str] = None,
        alpha: float = 0.65,
    ):
        """
        把底图影像 / SHP 边界线 / SHP 裁剪面画到 ax 上（全部为可选）。

        Returns:
            (basemap_loaded, boundary_polygon)
            basemap_loaded 表示真的画上了栅格影像（用于决定填色透明度）
            boundary_polygon 是裁剪面几何对象，供调用方做网格掩膜
        """
        import os

        basemap_loaded = False
        boundary_polygon = None

        # ── 1. 底图影像 ──
        if basemap_path and os.path.exists(basemap_path):
            try:
                import matplotlib.image as mpimg
                ext = os.path.splitext(basemap_path)[1].lower()
                img = None
                img_extent = None

                if ext in ('.tif', '.tiff'):
                    # GeoTIFF: 用 rasterio 取真实地理参考
                    try:
                        import rasterio
                        with rasterio.open(basemap_path) as src:
                            if src.count >= 3:
                                img = np.transpose(src.read([1, 2, 3]), (1, 2, 0))
                            else:
                                img = src.read(1)
                            img_extent = [src.bounds.left, src.bounds.right,
                                          src.bounds.bottom, src.bounds.top]
                            logger.info(f"GeoTIFF 底图加载成功: {src.width}x{src.height}, "
                                        f"范围={img_extent}")
                    except ImportError:
                        logger.warning("rasterio 未安装，尝试 PIL 读取 GeoTIFF")
                    except Exception as e:
                        logger.warning(f"rasterio 读取失败: {e}")

                if img is None:
                    img = mpimg.imread(basemap_path)
                    if img_extent is None:
                        if shp_bounds and len(shp_bounds) >= 4:
                            img_extent = [shp_bounds[0], shp_bounds[2],
                                          shp_bounds[1], shp_bounds[3]]
                        else:
                            img_extent = model_extent

                if img is not None:
                    ax.imshow(img, extent=img_extent or model_extent,
                              aspect='auto', alpha=alpha, zorder=0,
                              cmap='gray' if getattr(img, "ndim", 2) == 2 else None)
                    basemap_loaded = True
                    logger.info(f"底图已加载: {basemap_path}")
            except Exception as e:
                logger.warning(f"底图加载失败: {e}")

        # ── 2. SHP 边界线（线要素） ──
        if shp_path and os.path.exists(shp_path):
            try:
                import geopandas as gpd
                gdf = gpd.read_file(shp_path)
                self._plot_boundary(gdf, ax, edgecolor='red', linewidth=2.0,
                                    label='Boundary Line')
                logger.info(f"SHP 边界线已叠加: {shp_path} ({len(gdf)} 要素)")
            except Exception as e:
                logger.warning(f"SHP 边界线叠加失败: {e}")

        # ── 3. SHP 裁剪面（面要素） ──
        if clip_shp_path and os.path.exists(clip_shp_path):
            try:
                import geopandas as gpd
                gdf = gpd.read_file(clip_shp_path)
                if len(gdf) > 0:
                    boundary_polygon = gdf.geometry.iloc[0]
                self._plot_boundary(gdf, ax, edgecolor='black', linewidth=2.5,
                                    label='Model Domain')
                logger.info(f"SHP 裁剪面已加载: {clip_shp_path} "
                            f"({len(gdf)} 要素, "
                            f"geom={boundary_polygon.geom_type if boundary_polygon else 'N/A'})")
            except Exception as e:
                logger.warning(f"SHP 裁剪面加载失败: {e}")

        return basemap_loaded, boundary_polygon

    @staticmethod
    def _plot_boundary(gdf, ax, edgecolor, linewidth, label):
        """
        画 SHP 的几何轮廓。

        线要素必须直接 ``gdf.plot()``：线自己的"边界"是它的两个端点（闭合环则
        为空），所以 ``gdf.boundary`` 作用在线图层上会得到一堆空几何，
        matplotlib 只会抛一句 "composed of empty geometries" 然后什么都不画 ——
        红色边界线会静默消失。面要素才该用 ``.boundary`` 画轮廓。
        """
        has_poly = any(str(t) in ("Polygon", "MultiPolygon") for t in gdf.geom_type)
        if has_poly:
            gdf.boundary.plot(ax=ax, edgecolor=edgecolor, linewidth=linewidth,
                              alpha=0.9, zorder=2, label=label)
        else:
            gdf.plot(ax=ax, color=edgecolor, linewidth=linewidth,
                     alpha=0.9, zorder=2, label=label)

    @staticmethod
    def _mask_to_polygon(data: np.ndarray, X: np.ndarray, Y: np.ndarray,
                         polygon) -> np.ndarray:
        """
        把网格数据掩膜到多边形之外。

        原先用 Python 双重循环 + ``contains_point`` 逐格判断，且只认 Polygon，
        MultiPolygon 直接漏判。shapely 2.x 的 ``contains_xy`` 可一次向量化完成
        并对任意几何类型生效；不可用时回退到 matplotlib Path（仅 Polygon）。
        """
        try:
            from shapely import contains_xy
            mask = contains_xy(polygon, X.ravel(), Y.ravel()).reshape(X.shape)
        except Exception:
            if getattr(polygon, "geom_type", None) != "Polygon":
                logger.warning(f"几何类型 {getattr(polygon, 'geom_type', '?')} "
                               f"无法回退掩膜（缺 shapely 2.x），跳过裁剪")
                return data
            from matplotlib.path import Path as MplPath
            path = MplPath(np.array(polygon.exterior.coords))
            pts = np.column_stack([X.ravel(), Y.ravel()])
            mask = path.contains_points(pts).reshape(X.shape)

        logger.info(f"边界裁剪完成: {mask.sum()}/{mask.size} 个网格单元在边界内")
        return np.ma.masked_where(~mask, data)

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

    def _draw_plume_panel(
        self,
        ax,
        conc: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        vmin: float,
        vmax: float,
        levels: int,
        threshold: float,
        alpha: float,
    ):
        """
        在单个 axes 上画一帧污染羽。总览图与单时刻图共用，避免两处各写一遍。

        掩膜从 ``threshold`` 起（旧代码用 ``threshold*0.1``，填色比红色虚线阈值线
        向外多扩了 10 倍，看着像"污染羽远大于阈值范围"）。
        """
        floor = min(threshold, vmin) if vmin is not None else threshold
        masked = np.ma.masked_where(conc < floor, conc)
        if vmax <= vmin:
            vmax = vmin + 1e-12
        # 只有数据真的超出上限时才开 extend，否则 contourf 会白画一个箭头
        over = masked.count() > 0 and float(np.ma.max(masked)) > vmax
        cf = ax.contourf(X, Y, masked, levels=np.linspace(vmin, vmax, int(levels)),
                         cmap=self.CMAP_PLUME, vmin=vmin, vmax=vmax, alpha=alpha,
                         zorder=1, extend="max" if over else "neither")
        # 阈值线：正好是填色的可见边缘
        ax.contour(X, Y, conc, levels=[threshold], colors="red",
                   linewidths=1.5, linestyles="--", zorder=2)
        return cf

    def _draw_flow_arrows(self, ax, X, Y, velocity, nrow, ncol,
                          color="dimgray", alpha=0.35, zorder=3):
        """在污染羽图上叠加浅灰流场箭头，让读者能参照水流模型看迁移方向"""
        if velocity is None:
            return
        try:
            qx, qy = velocity[0], velocity[1]
            step = max(1, min(nrow, ncol) // 20)
            mag = np.sqrt(qx ** 2 + qy ** 2)
            if not np.any(np.isfinite(mag)) or float(np.nanmax(mag)) <= 0:
                return
            ax.quiver(X[::step, ::step], Y[::step, ::step],
                      qx[::step, ::step], qy[::step, ::step],
                      color=color, alpha=alpha, zorder=zorder,
                      scale_units="xy", angles="xy",
                      scale=1.0 / max(1e-12, float(np.nanmax(mag)) * 0.05))
        except Exception as e:
            logger.warning(f"流场箭头叠加失败: {e}")

    def plot_plume_evolution(
        self,
        conc_frames: List[np.ndarray],
        times: List[float],
        delr: np.ndarray,
        delc: np.ndarray,
        threshold: float = 1.0,
        n_cols: int = 3,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        levels: int = 20,
        alpha: float = 0.85,
        basemap_path: Optional[str] = None,
        shp_path: Optional[str] = None,
        clip_shp_path: Optional[str] = None,
        shp_bounds: Optional[list] = None,
        velocity: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        污染羽时空演化序列图（总览）

        Args:
            conc_frames: 各时刻浓度场列表
            times: 时刻列表
            threshold: 污染羽边界阈值（红色虚线；也是填色的显示下界）
            n_cols: 列数
            vmin/vmax: 色标范围；None 表示 vmin=0 / vmax=各帧最大浓度
            levels: 填色分层数
            alpha: 填色透明度（叠底图时调低更"淡显"）
            basemap_path/shp_path/clip_shp_path/shp_bounds: 底图与边界，与
                plot_head_on_basemap 用同一套参数即可保证两图对齐
            velocity: 可选比流量场 (3, nrow, ncol)，用于叠加流向箭头
        """
        n_frames = len(conc_frames)
        n_rows = int(np.ceil(n_frames / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(4.5 * n_cols, 3.8 * n_rows),
                                 squeeze=False)
        # constrained 布局会为 suptitle + 底部 colorbar 预留空间；
        # tight_layout 不预留，这正是原先标题/图例压住子图的原因之一。
        # 注意：matplotlib 3.11 的 subplots() 没有 layout 参数，只能后设。
        fig.set_layout_engine("constrained")
        axes = axes.ravel()

        nrow, ncol = conc_frames[0].shape
        X, Y, model_extent = self._grid_coords(nrow, ncol, delr, delc, shp_bounds)

        if vmax is None:
            vmax = float(max(np.max(c) for c in conc_frames))
        if vmin is None:
            vmin = 0.0

        # 底图逐帧画：只在第一帧画会让六格看起来不像同一张图（用户明确要求
        # 污染羽参考水流模型、叠在底图之上）。裁剪面几何取首帧那份给所有帧共用,
        # 否则会出现"某一帧裁剪、另一帧不裁剪"的不一致。
        clip_polygon = None
        cf = None
        for i, (conc, ax) in enumerate(zip(conc_frames, axes)):
            _loaded, poly = self._draw_basemap(
                ax, X, Y, model_extent,
                basemap_path=basemap_path, shp_bounds=shp_bounds,
                shp_path=shp_path, clip_shp_path=clip_shp_path, alpha=0.6,
            )
            if clip_polygon is None and poly is not None:
                clip_polygon = poly

            c = conc
            if clip_polygon is not None and hasattr(clip_polygon, 'contains'):
                try:
                    c = self._mask_to_polygon(conc, X, Y, clip_polygon)
                except Exception as e:
                    logger.warning(f"污染羽边界裁剪失败: {e}")
            cf = self._draw_plume_panel(ax, c, X, Y, vmin, vmax,
                                        levels, threshold, alpha)
            self._draw_flow_arrows(ax, X, Y, velocity, nrow, ncol)
            ax.set_title(f"t = {times[i]:.0f} d")
            ax.set_aspect("equal")
            ax.set_xlabel("X (m)", fontsize=8)
            ax.set_ylabel("Y (m)", fontsize=8)
            ax.tick_params(labelsize=7)

        # 隐藏多余子图
        for j in range(n_frames, len(axes)):
            axes[j].set_visible(False)

        if cf is not None:
            # colorbar 挂到"实际使用到的轴列表"上，并放到整图底部。
            # 旧代码用 pad=0.02 挂在 axes[:n_frames] 上，会压缩每一列的宽度。
            fig.colorbar(cf, ax=list(axes[:n_frames]), location="bottom",
                         shrink=0.55, pad=0.03, aspect=35,
                         label="Concentration (mg/L)")

        fig.suptitle("Plume Evolution", fontsize=14)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        return fig

    def plot_plume_at_time(
        self,
        conc: np.ndarray,
        t: float,
        delr: np.ndarray,
        delc: np.ndarray,
        threshold: float = 1.0,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        levels: int = 20,
        alpha: float = 0.85,
        basemap_path: Optional[str] = None,
        shp_path: Optional[str] = None,
        clip_shp_path: Optional[str] = None,
        shp_bounds: Optional[list] = None,
        velocity: Optional[np.ndarray] = None,
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        污染羽单时刻图（交互窗口用）

        单张大图 + 独立 colorbar，用 constrained 布局保证两者不重叠。
        ``subtitle`` 用于说明该帧是否由插值得到。
        """
        nrow, ncol = conc.shape
        X, Y, model_extent = self._grid_coords(nrow, ncol, delr, delc, shp_bounds)

        if vmax is None:
            vmax = float(np.max(conc))
        if vmin is None:
            vmin = 0.0

        fig, ax = plt.subplots(figsize=(9, 6.5))
        fig.set_layout_engine("constrained")

        basemap_loaded, clip_polygon = self._draw_basemap(
            ax, X, Y, model_extent,
            basemap_path=basemap_path, shp_bounds=shp_bounds,
            shp_path=shp_path, clip_shp_path=clip_shp_path, alpha=0.6,
        )

        c = conc
        if clip_polygon is not None and hasattr(clip_polygon, 'contains'):
            try:
                c = self._mask_to_polygon(conc, X, Y, clip_polygon)
            except Exception as e:
                logger.warning(f"污染羽边界裁剪失败: {e}")

        cf = self._draw_plume_panel(ax, c, X, Y, vmin, vmax,
                                   levels, threshold, alpha)
        self._draw_flow_arrows(ax, X, Y, velocity, nrow, ncol)

        # colorbar 挂在单个 ax 上并交给 constrained 布局排布 → 不重叠
        fig.colorbar(cf, ax=ax, location="right", shrink=0.85, pad=0.02,
                     label="Concentration (mg/L)")

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")
        ax.set_title(title or f"Plume at t = {t:.1f} d", fontsize=13)

        # 图例：阈值线（底图另在图例里标一笔，避免出现空图例）
        # 放到坐标轴**下方**而不是图内 —— 图内任何位置都会在某个时刻被污染羽
        # 盖住（用户反馈的"图例和图片重叠"就是这一类），挪出去才能保证永不遮挡。
        handles = [plt.Line2D([], [], color="red", linestyle="--", linewidth=1.5,
                              label=f"Threshold = {threshold:g} mg/L")]
        if basemap_loaded:
            handles.append(plt.Rectangle((0, 0), 1, 1, color="gray", alpha=0.4,
                                         label="Basemap"))
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13),
                  ncol=len(handles), fontsize=8, frameon=False)

        if subtitle:
            fig.suptitle(subtitle, fontsize=9, color="dimgray")

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
    # 主控过程分析
    # ================================================================

    _PROCESS_ZH = {
        "advection": "对流", "dispersion": "弥散", "decay": "降解",
        "sorption": "吸附",
        "advection_dominated": "对流主导", "dispersion_dominated": "弥散主导",
        "mixed_transport": "对流-弥散混合",
        "reaction_limited": "反应受限", "transport_limited": "运移受限",
        "mixed_kinetics": "反应-运移混合",
        "insufficient_data": "数据不足",
    }

    def _zh_process(self, key: str) -> str:
        """把 ProcessIdentifier 的英文枚举译成中文（报告与图件都是给人看的）"""
        if not isinstance(key, str):
            return str(key)
        # "advection+mixed+sorption_retarded" 这类复合标签逐段翻译
        return "+".join(self._PROCESS_ZH.get(part, part) for part in key.split("+"))

    def plot_process_analysis(
        self,
        analysis: dict,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        主控过程分析图（饼图 + 条形图 + 参数注记）

        直接吃 ``ProcessIdentifier.identify_dominant_process()`` 的**嵌套**返回值
        ``{"contributions": {...}, "peclet_number": ..., "explanation": ...}``，
        比 ``plot_process_contribution`` 期望的扁平 dict 信息量大得多，因此单独
        实现而不复用后者（后者保持原样不动）。
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
        fig.set_layout_engine("constrained")
        # 底部留一条固定带放 Pe/Da/R 注记。constrained 布局不认识任意的
        # fig.text，不显式让位的话注记会压在饼图左下角 —— 用 rect 把两个轴
        # 限制在上方 78% 内即可（rect 是 (x0, y0, w, h)，0 为画布左下角）。
        fig.get_layout_engine().set(rect=(0.0, 0.18, 1.0, 0.78))

        # 数据不足：给明确文案，而不是画一张空白图
        if analysis.get("dominant_process") == "insufficient_data" or \
                not analysis.get("contributions"):
            for ax in (ax1, ax2):
                ax.axis("off")
            fig.suptitle("主控过程分析：数据不足", fontsize=14)
            fig.text(0.5, 0.5,
                     "可用输出时刻少于 2 个，无法分解各过程对质量变化的贡献。\n"
                     "请增加输出频率后重新运行。",
                     ha="center", va="center", fontsize=11, color="dimgray")
            if save_path:
                fig.savefig(save_path, bbox_inches="tight", dpi=150)
            return fig

        contrib = analysis.get("contributions", {})
        labels = [self._zh_process(k) for k in contrib.keys()]
        values = [float(v) for v in contrib.values()]

        # 全零兜底：ax.pie 在总和为 0 时会画出空白
        if sum(abs(v) for v in values) <= 1e-12:
            for ax in (ax1, ax2):
                ax.axis("off")
            fig.suptitle("主控过程分析：各过程通量可忽略", fontsize=14)
            fig.text(0.5, 0.5,
                     "浓度场全为零或通量低于数值精度，\n各过程贡献无法分解。",
                     ha="center", va="center", fontsize=11, color="dimgray")
            if save_path:
                fig.savefig(save_path, bbox_inches="tight", dpi=150)
            return fig

        colors = plt.cm.Set2(np.linspace(0, 1, max(len(labels), 3)))

        ax1.pie(values, labels=labels, autopct="%1.1f%%",
                colors=colors[:len(labels)], startangle=90, pctdistance=0.8)
        ax1.set_title("各过程贡献率（饼图）")

        bars = ax2.barh(labels, values, color=colors[:len(labels)])
        ax2.set_xlabel("贡献率 (%)")
        ax2.set_title("各过程贡献率（条形图）")
        ax2.invert_yaxis()
        for bar, val in zip(bars, values):
            ax2.text(bar.get_width() + max(values) * 0.02,
                     bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f}%", va="center", fontsize=9)

        dominant = self._zh_process(analysis.get("dominant_process", "N/A"))
        fig.suptitle(f"主控过程分析 — 主控过程: {dominant}", fontsize=14)

        lines = [
            f"Peclet 数 Pe = {analysis.get('peclet_number', 'N/A')}"
            f"（{self._zh_process(analysis.get('transport_regime', ''))}）",
            f"Damkohler 数 Da = {analysis.get('damkohler_number', 'N/A')}"
            f"（{self._zh_process(analysis.get('reaction_regime', ''))}）",
            f"迟滞因子 R = {analysis.get('retardation_factor', 'N/A')}",
            analysis.get("explanation", ""),
            "注：通量统计基于第 0 层浓度场与第 0 层比流量场（非全层积分）。",
        ]
        fig.text(0.02, 0.02, "\n".join(t for t in lines if t),
                 ha="left", va="bottom", fontsize=9, color="dimgray",
                 wrap=True)

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
        nrow, ncol = head.shape

        # 统一坐标基准 + 底图机制（与污染羽图共用，保证两张图对齐）
        X, Y, model_extent = self._grid_coords(nrow, ncol, delr, delc, shp_bounds)

        fig, ax = plt.subplots(figsize=(12, 9))

        basemap_loaded, boundary_polygon = self._draw_basemap(
            ax, X, Y, model_extent,
            basemap_path=basemap_path, shp_bounds=shp_bounds,
            shp_path=shp_path, clip_shp_path=clip_shp_path, alpha=0.65,
        )

        # ── 3. 水头填色（半透明，裁剪到边界） ──
        masked_head = head
        if boundary_polygon is not None and hasattr(boundary_polygon, 'contains'):
            try:
                masked_head = self._mask_to_polygon(head, X, Y, boundary_polygon)
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
        gy = -gy / self._as_sizes(delc).reshape(-1, 1)
        gx = -gx / self._as_sizes(delr).reshape(1, -1)
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

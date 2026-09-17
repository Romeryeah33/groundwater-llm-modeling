"""
网格坐标原语 —— 建模侧与可视化侧共用的唯一一份坐标数学。

为什么单独成模块：模型本身**没有地理参考**（`ModflowGwfdis` 不传 xoff/yoff），
全仓库唯一的地理参考是可视化层里"以 SHP 外框左下角为原点 + cumsum 格子尺寸"
这套约定。源区选格（`app/modeling/source_zone.py`）必须与绘图
（`app/analysis/visualizer.py`）**逐元素一致**，否则会出现"模型补给的格子和图上
画在填埋场轮廓内的格子不是同一批"这种最难查的错位。

但 `app/modeling` 不能反向依赖 `app/analysis`（visualizer 是绘图层），所以把这段
坐标数学提到中性层 `app/utils/`。`visualizer._grid_coords` 与 `source_zone` 都调用
`grid_cell_centers`，两边不可能漂移；`tests/test_source_zone.py` 里还有一条
`np.array_equal` 的锁死断言兜底。
"""

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def as_sizes(del_) -> np.ndarray:
    """把 delr/delc 归一化为一维定长 ndarray（标量、列表、ndarray 都吃）"""
    return np.atleast_1d(np.asarray(del_, dtype=float)).ravel()


def grid_cell_centers(
    nrow: int,
    ncol: int,
    delr,
    delc,
    origin_xy: Optional[list] = None,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    统一的网格坐标基准。

    约定（与历史实现逐元素相同，务必保持）：
      - ``x_edges = concat([[0], cumsum(delr)]) + origin_x``，``y_edges`` 同理
      - 单元中心 = ``(edges[:-1] + edges[1:]) / 2``
      - ``X.shape == Y.shape == (nrow, ncol)``
      - **row 0 在最南**（y 向北增长）、**col 0 在最西**
      - ``len(delr) != ncol`` 时整体退化为 ``mean(delr)``（非定长输入的保护）

    Args:
        nrow / ncol: 行数 / 列数
        delr: 列方向格子尺寸（沿 x），长度应为 ncol
        delc: 行方向格子尺寸（沿 y），长度应为 nrow
        origin_xy: 网格原点 ``[x0, y0]``，即对外宣称的模型域左下角。缺省 ``[0, 0]``
            ——此时坐标是纯局部坐标系，与真实地理无关。

    Returns:
        ``(X, Y, model_extent)``，``model_extent = [xmin, xmax, ymin, ymax]``
    """
    delr = as_sizes(delr)
    delc = as_sizes(delc)
    if len(delr) != ncol:
        logger.warning(
            f"delr 长度({len(delr)})与 ncol({ncol})不符，退化为均值 {float(np.mean(delr)):.4g}"
        )
        delr = np.full(ncol, float(np.mean(delr)))
    if len(delc) != nrow:
        logger.warning(
            f"delc 长度({len(delc)})与 nrow({nrow})不符，退化为均值 {float(np.mean(delc)):.4g}"
        )
        delc = np.full(nrow, float(np.mean(delc)))

    x_edges = np.concatenate([[0.0], np.cumsum(delr)])
    y_edges = np.concatenate([[0.0], np.cumsum(delc)])

    if origin_xy is not None and len(origin_xy) >= 2:
        x_edges = x_edges + float(origin_xy[0])
        y_edges = y_edges + float(origin_xy[1])

    X, Y = np.meshgrid(
        (x_edges[:-1] + x_edges[1:]) / 2,
        (y_edges[:-1] + y_edges[1:]) / 2,
    )
    model_extent = [float(x_edges[0]), float(x_edges[-1]),
                    float(y_edges[0]), float(y_edges[-1])]
    return X, Y, model_extent


def warn_extent_mismatch(model_extent: list, shp_bounds: Optional[list], log=None) -> None:
    """
    模型域与 SHP 外框不吻合时明确告警：底图会偏，不静默。

    East Texas 案例实测：模型域 4360×2060 m，SHP 外框 3960×1871 m，两个方向都差
    10.1%。这是**既有**问题（根因是 ``domain.x_range/y_range`` 与已有 nrow/ncol 的
    网格互不相关），不在本轮修；但绝不能悄悄吞掉。

    Args:
        log: 用哪个 logger 发这条告警。缺省用本模块的。调用方（如
            ``visualizer._grid_coords``）应传入**自己的** logger，这样"哪个模块
            在抱怨"对日志过滤/排查仍然可辨 —— 委托坐标数学不该顺带改变告警的
            归属 logger。
    """
    if not shp_bounds or len(shp_bounds) < 4:
        return
    log = log or logger
    span_x = float(shp_bounds[2]) - float(shp_bounds[0])
    span_y = float(shp_bounds[3]) - float(shp_bounds[1])
    for _name, _model, _shp in (("X", model_extent[1] - model_extent[0], span_x),
                                ("Y", model_extent[3] - model_extent[2], span_y)):
        if _shp > 0 and abs(_model - _shp) > 0.01 * _shp:
            log.warning(
                f"{_name} 方向模型域({_model:.1f} m)与 SHP 外框({_shp:.1f} m)"
                f"相差 {abs(_model-_shp)/_shp*100:.1f}%，底图与网格可能错位"
            )

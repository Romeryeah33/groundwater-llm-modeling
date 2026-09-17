"""
污染源区定位：把"填埋场""垃圾场"这类**命名区域**解析成具体的网格单元。

## 为什么需要这个模块

`model_mapper.map_boundary_conditions` 只能消费 `geometry.grid_indices`（显式的
逐格索引）。而 LLM 无法做多边形与网格求交，它给出的是语义位置：

```json
{"type": "rch", "name": "landfill_boundary", "position": "landfill",
 "parameters": {"rate": 1e-4, "concentration": 20000}}
```

旧代码在这里退化成"从名称推断方位"（只认 west/east/north/south），`landfill` 一个
都不匹配，于是落到"有浓度但没给注入位置 → 展开为 layer0 全网格"的分支 ——
East Texas 案例里把 132 格的填埋场源悄悄放大成 22454 格的全域面源（170 倍），
污染羽看起来像水流模型结果。本模块 + mapper 里的硬拒绝就是为消灭这类静默放大。

## 四级解析优先级

1. `geometry.grid_indices` —— 显式索引，原样透传（脚本与既有测试走这条）
2. `geometry.coordinates` —— IR 直接给几何（面 / 线 / 点）
3. `geometry.bbox` —— IR 直接给外包框（`[xmin, ymin, xmax, ymax]`）
4. 上传 SHP 的**名称匹配**（`landfill_boundary` ↔ `landfill_arcs.shp`）

四级全失败 → 抛 `SourceZoneUnresolved`。**绝不回落成全网格。**

## 坐标约定

模型本身没有地理参考，唯一的一份坐标数学在 `app/utils/grid_coords.py`，与绘图
层共用，保证"模型补给的格子"就是"图上画在填埋场轮廓内的格子"。
"""

import logging
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from app.utils.grid_coords import as_sizes, grid_cell_centers

logger = logging.getLogger(__name__)


# ====================================================================
# 方位型边界名推断（唯一实现，model_mapper 委托到这里）
# ====================================================================

# 只使用有区分力的关键词，不含单字母匹配 —— 否则 "well" 含 "w" 会被判成 west，
# "northeast" 含 "east" 会被判成 east。
WEST_KEYWORDS = ("west", "西", "左", "left", "xmin")
EAST_KEYWORDS = ("east", "东", "右", "right", "xmax")
NORTH_KEYWORDS = ("north", "北", "上", "top", "ymax")
SOUTH_KEYWORDS = ("south", "南", "下", "bottom", "ymin")

# 这些包"整条边界的均匀作用"几乎不可能是本意：
# 整条北边界的补给带、整条西边界的井排，都更像名称误匹配。
# 一个叫 top_of_landfill 的补给边界，今天会因 north_keywords 含 "top"
# 静默变成整条北边界 —— 这里把它挡掉，让源区解析器去处理。
NO_EDGE_INFERENCE_TYPES = ("rch", "evt", "wel")

_EDGE_KEYWORDS = (
    (WEST_KEYWORDS, "west"),
    (EAST_KEYWORDS, "east"),
    (NORTH_KEYWORDS, "north"),
    (SOUTH_KEYWORDS, "south"),
)


def _word_match(text: str, keywords: tuple) -> bool:
    """关键词是否作为独立单词出现（避免 northeast 匹配到 east）"""
    for kw in keywords:
        idx = text.find(kw)
        if idx < 0:
            continue
        # 中文关键词用简单子串匹配，因为 "西" 应该匹配 "西侧边界"
        if any(ord(c) > 127 for c in kw):
            return True
        # ASCII 关键词检查单词边界
        before_ok = idx == 0 or not text[idx - 1].isalpha()
        after_ok = (idx + len(kw) >= len(text)) or not text[idx + len(kw)].isalpha()
        if before_ok and after_ok:
            return True
    return False


def _unwrap(value, default=None):
    """兼容裸值与 {value, unit, source} 两种 IR 写法"""
    if value is None:
        return default
    if isinstance(value, dict):
        inner = value.get("value")
        return default if inner is None else inner
    return value


def grid_dims(grid: dict) -> tuple:
    """
    从 IR 的 grid 段取出 (nrow, ncol, delr, delc, nlay)。

    delr/delc 缺省 20.0 —— 与 model_mapper 和 model_builder._resolve_dis 的默认值
    保持一致（East Texas 案例的 grid 里就没有这两个键，全靠这个默认值）。
    """
    nrow = int(_unwrap(grid.get("nrow"), 50) or 50)
    ncol = int(_unwrap(grid.get("ncol"), 50) or 50)
    nlay = int(_unwrap(grid.get("nlay"), 1) or 1)

    delr_raw = _unwrap(grid.get("delr"), 20.0)
    delc_raw = _unwrap(grid.get("delc"), 20.0)
    delr = as_sizes(delr_raw) if delr_raw is not None else np.array([20.0])
    delc = as_sizes(delc_raw) if delc_raw is not None else np.array([20.0])
    if len(delr) != ncol:
        delr = np.full(ncol, 20.0)
    if len(delc) != nrow:
        delc = np.full(nrow, 20.0)
    return nrow, ncol, delr, delc, nlay


def infer_edge_indices(bc_name: str, grid: dict, layer=None, bc_type: str = None) -> list:
    """
    从边界名称推断网格边界的 IJ 索引（west/east/north/south 及中英文同义词）。

    行号约定：**row 0 在南、row nrow-1 在北**（与 `grid_coords.grid_cell_centers`
    的 y 递增方向、以及绘图时的"上北下南"一致）。列号：col 0 在西、col ncol-1 在东。

    Args:
        layer: `None` 表示**铺满所有层**（多层模型若只在 layer0 设边界，其余层
            水头无处排泄）。对 RCH 这类面源必须显式传 `layer=0`，否则会错误地铺满各层。
        bc_type: 边界类型。`rch`/`evt`/`wel` **一律返回空** —— 整条边界的补给带或
            井排几乎不可能是本意，而一个叫 `top_of_landfill` 的补给边界今天会因
            方位关键词含 `"top"` 静默变成整条北边界（`left_bank` 变整条西边界同理）。
            这类名称应当交给源区解析器处理，而不是被方位推断"捡走"。
    """
    if bc_type in NO_EDGE_INFERENCE_TYPES:
        return []

    name_lower = (bc_name or "").lower()
    nrow = int(_unwrap(grid.get("nrow"), 50) or 50)
    ncol = int(_unwrap(grid.get("ncol"), 50) or 50)

    if layer is None:
        nlay = int(_unwrap(grid.get("nlay"), 1) or 1)
        layers = list(range(max(1, nlay)))
    else:
        layers = [int(layer)]

    indices = []

    def _edge(cells):
        for lay in layers:
            for r, c in cells:
                indices.append({"layer": lay, "row": r, "col": c, "period": 0})

    # 行号方向：**row 0 在南、row nrow-1 在北**。
    #
    # 这个约定由 `app/utils/grid_coords.grid_cell_centers` 定死，也是真实坐标的
    # 方向：y_edges 从 domain.origin_xy[1] 起累加，所以 row 索引越大 y 越大，
    # 而 y 就是北向坐标（East Texas 案例里 y≈3427xxx，UTM 北向）。
    # 绘图链路 `contourf(X, Y, head)` 不做任何翻转，于是 row 0 画在图**下方**、
    # row nrow-1 画在上方 —— 与"上北下南"一致。
    #
    # 这里曾经把 north 映射到 row 0（= 图的下方 = 真实南方），于是名为
    # `north_boundary` 的边界被写到最南一排：实测 `.ghb` 里 `south_boundary`
    # 的 GHB 212 落在 1-based 第 103 行（= 0-based nrow-1 = 最北），水头图随之
    # 上下镜像。用户按"上北下南"读图时南北是反的 —— 这就是当时的现象。
    #
    # 判据是与 SHP 对齐：landfill_arcs 在 y 上占 55.6%（3428453..3428678 落在
    # 3427412..3429283 内），栅格化后是 rows 52..62（中心 55%）—— 只有"row 越大
    # 越靠北"这一套能同时对上 SHP 坐标、可视化底图和 `.rch` 的实测行号。
    if _word_match(name_lower, WEST_KEYWORDS):
        _edge([(r, 0) for r in range(nrow)])
    elif _word_match(name_lower, EAST_KEYWORDS):
        _edge([(r, ncol - 1) for r in range(nrow)])
    elif _word_match(name_lower, NORTH_KEYWORDS):
        _edge([(nrow - 1, c) for c in range(ncol)])
    elif _word_match(name_lower, SOUTH_KEYWORDS):
        _edge([(0, c) for c in range(ncol)])

    return indices


# ====================================================================
# 异常
# ====================================================================

class SourceZoneUnresolved(Exception):
    """
    污染源区无法定位。

    `str(exc)` 就是**可直接展示给用户的中文多行文本**，不是 repr —— 这样即使
    异常一路穿到 gradio_app 的兜底 `except Exception`（渲染成
    `建模异常: {e}`），用户看到的仍然是可操作的说明。
    """


# ====================================================================
# 几何
# ====================================================================

def bbox_to_geometry(bbox):
    """``[xmin, ymin, xmax, ymax]`` → shapely box"""
    from shapely.geometry import box

    if isinstance(bbox, dict):
        bbox = _unwrap(bbox)
    vals = [float(v) for v in bbox]
    if len(vals) != 4:
        raise SourceZoneUnresolved(
            f"geometry.bbox 需要 4 个数 [xmin, ymin, xmax, ymax]，收到 {bbox!r}"
        )
    xmin, ymin, xmax, ymax = vals
    if xmax <= xmin or ymax <= ymin:
        raise SourceZoneUnresolved(
            f"geometry.bbox 退化（{bbox!r}）：要求 xmax > xmin 且 ymax > ymin"
        )
    return box(xmin, ymin, xmax, ymax)


def polygonize_layer(geometries) -> list:
    """
    把一条图层（可能是一堆首尾相接的 LineString）转成面。

    `landfill_arcs.shp` 是 4 条 LineString 拼出的闭合环，`LineString` 自己的
    "边界"是两个端点（闭合环则为空），所以既不能用 `.boundary` 画轮廓，也不能
    直接对线做 `contains_xy`（对 MultiLineString 恒为全 False）。必须先
    `polygonize(unary_union(...))`。

    已经是面（或面 + 线混合）的图层原样透传。
    """
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.ops import polygonize, unary_union

    geoms = [g for g in geometries if g is not None and not g.is_empty]
    if not geoms:
        return []

    polys = [g for g in geoms if isinstance(g, (Polygon, MultiPolygon))]
    if polys:
        return polys

    merged = unary_union(geoms)
    faces = list(polygonize(merged))
    if not faces:
        raise SourceZoneUnresolved(
            "图层里的线要素无法闭合成面（首尾不相接）。请确认导出的是闭合的"
            "填埋场轮廓，而不是开放的折线。"
        )
    return faces


def _merge(geometries):
    """把若干几何合并成一个（多个面则合成 MultiPolygon）"""
    from shapely.ops import unary_union

    if len(geometries) == 1:
        return geometries[0]
    return unary_union(geometries)


def _looks_like_degrees(geom, model_extent) -> bool:
    """
    bbox 是不是经纬度。模型域以米计（East Texas 是 61 万～62 万），若几何的坐标
    绝对值都落在 ±180/±90 以内，几乎肯定是把经纬度当米传了 —— 这会选中 0 格
    或 1 格，报错必须点破，否则用户只会看到"没选中任何格子"。
    """
    x0, y0, x1, y1 = geom.bounds
    if max(abs(x0), abs(x1), abs(y0), abs(y1)) > 180.0:
        return False
    span = max(model_extent[1] - model_extent[0], model_extent[3] - model_extent[2])
    return span > 1000.0


def cells_in_geometry(
    geom,
    nrow: int,
    ncol: int,
    delr,
    delc,
    origin_xy: Optional[list] = None,
    *,
    layer: int = 0,
    period: int = 0,
    min_overlap: float = 0.0,
) -> list:
    """
    选出**格心落在几何内**的网格单元，返回 `grid_indices` 形状的记录列表。

    用格心判据（`shapely.contains_xy`）而不是面积重叠，是为了与
    `visualizer._mask_to_polygon` 的判据完全一致 —— 图上把浓度裁剪到填埋场轮廓内
    用的就是它，两边必须是同一批格子。

    Args:
        min_overlap: >0 时改用"格内被几何覆盖的面积占比 ≥ min_overlap"判据。
            逃生阀，用于源区小于一个格子（格心判据会合法地选中 0 格）的场景。

    Returns:
        `[{"layer": 0, "row": 52, "col": 137, "period": 0}, ...]`，行优先升序。
    """
    from shapely import contains_xy

    X, Y, model_extent = grid_cell_centers(nrow, ncol, delr, delc, origin_xy)

    if _looks_like_degrees(geom, model_extent):
        raise SourceZoneUnresolved(
            f"源区坐标 {tuple(round(v, 4) for v in geom.bounds)} 疑似经纬度，"
            f"而模型域以米计（{model_extent[0]:.0f}..{model_extent[1]:.0f} m）。"
            f"请使用与 SHP / 模型一致的投影坐标（East Texas 为 UTM 14N）。"
        )

    # 完全落在模型域外 → 明确区分于"名称没匹配上"
    gx0, gy0, gx1, gy1 = geom.bounds
    outside = (gx1 < model_extent[0] or gx0 > model_extent[1]
               or gy1 < model_extent[2] or gy0 > model_extent[3])
    if outside:
        raise SourceZoneUnresolved(
            f"源区范围 [{gx0:.0f}, {gy0:.0f}, {gx1:.0f}, {gy1:.0f}] 完全落在模型域外"
            f"（模型域 x {model_extent[0]:.0f}..{model_extent[1]:.0f}, "
            f"y {model_extent[2]:.0f}..{model_extent[3]:.0f}）。\n"
            f"常见原因：网格原点不是 (0, 0)，或源区用了另一套坐标系。"
        )

    # 部分越界要告警但不拒绝（East Texas 的模型域比 SHP 外框大 10.1%，
    # 是既有偏差，不该让用户没法建模）
    if (gx0 < model_extent[0] or gx1 > model_extent[1]
            or gy0 < model_extent[2] or gy1 > model_extent[3]):
        over_x = max(model_extent[0] - gx0, gx1 - model_extent[1])
        over_y = max(model_extent[2] - gy0, gy1 - model_extent[3])
        logger.warning(
            f"源区部分超出模型域：x 方向 {max(over_x, 0):.0f} m、"
            f"y 方向 {max(over_y, 0):.0f} m 在域外，域外部分不参与补给"
        )

    if min_overlap and min_overlap > 0:
        cell_area = float(delr[0]) * float(delc[0])
        from shapely.geometry import box as _box

        rows, cols = [], []
        for r in range(nrow):
            for c in range(ncol):
                cell = _box(X[r, c] - delr[c] / 2, Y[r, c] - delc[r] / 2,
                            X[r, c] + delr[c] / 2, Y[r, c] + delc[r] / 2)
                inter = cell.intersection(geom).area
                if cell_area > 0 and inter / cell_area >= min_overlap:
                    rows.append(r)
                    cols.append(c)
        rr = np.asarray(rows, dtype=int)
        cc = np.asarray(cols, dtype=int)
    else:
        mask = contains_xy(geom, X.ravel(), Y.ravel()).reshape(X.shape)
        rr, cc = np.nonzero(mask)

    return [{"layer": int(layer), "row": int(r), "col": int(c), "period": int(period)}
            for r, c in zip(rr, cc)]


# ====================================================================
# 名称匹配
# ====================================================================

_SUFFIXES = ("_boundary", "_arcs", "_polys", "_pts", "_lines", "_lines",
             "_line", "_zone", "_area", "_polygons", "_polygon", "_shp")

_STOPWORDS = {
    "boundary", "boundaries", "layer", "poly", "polys", "polygon", "polygons",
    "arcs", "arc", "pts", "point", "points", "line", "lines", "zone", "area",
    "shp", "condition", "conditions", "cond", "shape", "shapes",
}

_TOKEN_SPLIT = re.compile(r"[^0-9a-zA-Z一-鿿]+")


def _tokens(text: str) -> set:
    """
    把图层名 / BC 名切成有区分力的 token。

    朴素 `"landfill" in name` 会误命中，必须分词并剔噪：
      - `Aquifer Layer 1_polys` → {aquifer}（`layer`/`polys` 是停用词，`1` 太短）
      - `Boundary_arcs`        → {}（全是停用词，不参与匹配）
      - `landfill_arcs`        → {landfill}
    """
    if not text:
        return set()
    low = str(text).lower()
    for suf in _SUFFIXES:
        if low.endswith(suf):
            low = low[: -len(suf)]
            break
    out = set()
    for tok in _TOKEN_SPLIT.split(low):
        if not tok or tok in _STOPWORDS:
            continue
        has_cjk = any(ord(ch) > 127 for ch in tok)
        # 中文两字即可（"填埋场"），ASCII 太短无区分力（"1"、"rch"、"wel"）
        if (has_cjk and len(tok) >= 2) or (not has_cjk and len(tok) >= 4):
            out.add(tok)
    return out


def _geom_rank(geometry_type: str) -> int:
    """面 > 线 > 点（面最可能是一个区域）"""
    gt = (geometry_type or "").lower()
    if "polygon" in gt:
        return 2
    if "line" in gt:
        return 1
    if "point" in gt:
        return 0
    return 0


def _resolve_path(stem: str, uploaded_paths: Iterable[str]) -> Optional[str]:
    """按 stem 从上传路径列表里找回 .shp（既有 _detect_basemap_inputs 的写法）"""
    for up in uploaded_paths or ():
        p = Path(str(up))
        if p.stem == stem and p.suffix.lower() == ".shp":
            return str(up)
    return None


def _load_geometries(shp_path: str):
    """读 SHP 的几何序列；geopandas 缺失或文件已不在磁盘上时给出可读消息"""
    try:
        import geopandas as gpd
    except ImportError as e:  # pragma: no cover
        raise SourceZoneUnresolved(
            f"需要 geopandas 才能读取 SHP 定位源区，但导入失败：{e}"
        )

    if not Path(shp_path).exists():
        raise SourceZoneUnresolved(
            f"上传的 SHP 已不在磁盘上：{shp_path}\n"
            f"上传文件存放在系统临时目录里，重启或清理后会失效。请重新上传基础材料压缩包。"
        )
    gdf = gpd.read_file(shp_path)
    return list(gdf.geometry.values)


def _candidate_stems(file_index, uploaded_paths) -> dict:
    """
    候选图层名 → 几何类型提示。

    **两个来源都要看**：`file_index`（即 `file_uploads.shapefiles`，带
    geometry_type）与**原始上传路径** `uploaded_paths`。后者不是冗余 ——
    `file_index` 未必被调用方一路传下来（会话里真实存在的是
    `_uploaded_file_paths`），只信 file_index 会让磁盘上明明躺着的
    `landfill_arcs.shp` 匹配不上，报出"没有名称匹配的图层"这种**假失败**，
    而用户明明上传了正确的文件。
    """
    cands: dict = {}
    for stem, info in (file_index or {}).items():
        cands[str(stem)] = (info or {}).get("geometry_type")
    for up in uploaded_paths or ():
        p = Path(str(up))
        if p.suffix.lower() == ".shp" and p.stem not in cands:
            cands[p.stem] = None
    return cands


def _match_by_name(bc, file_index, uploaded_paths, grid, origin_xy, layer, period):
    """
    按 BC 的 position/name 匹配上传 SHP 图层。

    排在最高名次的候选**并列时必须报错**并列出候选 —— 静默挑一个是本次要
    消灭的 bug 类型。
    """
    want = _tokens(bc.get("position")) | _tokens(bc.get("name"))
    if not want:
        return None

    candidates = []
    for stem, gt_hint in _candidate_stems(file_index, uploaded_paths).items():
        have = _tokens(stem)
        if not have:
            continue
        if want & have:
            score = 3          # token 精确相等
        elif any(w in h or h in w for w in want for h in have if len(w) >= 4):
            score = 2          # 子串包含
        else:
            continue
        candidates.append((score, _geom_rank(gt_hint), stem, gt_hint))

    if not candidates:
        return None

    def _try(entry):
        _s, _hint, stem, _gt = entry
        shp_path = _resolve_path(stem, uploaded_paths)
        if not shp_path:
            return None, stem, "（该图层的 .shp 已不在磁盘上）", 0
        try:
            geoms = _load_geometries(shp_path)
            # 名次必须用 **polygonize 之前**的原始类型定：`polygonize_layer`
            # 会把闭合线也变成面，若在它之后取 `geom.geom_type`，线图层与面图层
            # 的几何类型全都成了 Polygon，"面优于线"这条规则就永远不生效。
            raw_rank = max((_geom_rank(g.geom_type) for g in geoms if g is not None),
                           default=0)
            faces = polygonize_layer(geoms)
            geom = _merge(faces)
            idx = cells_in_geometry(geom, *_dims(grid, origin_xy),
                                    layer=layer, period=period)
        except SourceZoneUnresolved as e:
            return None, stem, str(e), 0
        if not idx:
            return None, stem, "（选中 0 个网格单元）", 0
        return (idx, geom, shp_path), stem, "", raw_rank

    tried, hits = [], []
    for entry in candidates:
        got, stem, why, rank = _try(entry)
        if got:
            hits.append((entry[0], rank, stem, got))
        else:
            tried.append(f"  · {stem} {why}")

    if hits:
        hits.sort(key=lambda t: (-t[0], -t[1], t[2]))
        top_key = (hits[0][0], hits[0][1])
        top = [h for h in hits if (h[0], h[1]) == top_key]
        if len(top) > 1:
            raise SourceZoneUnresolved(
                f"边界 '{bc.get('name')}' 的源区名称同时匹配到多个图层，无法确定用哪个：\n"
                + "\n".join(f"  · {s}" for _sc, _rk, s, _g in top)
                + "\n请把该边界改成一个能被唯一识别的名称，或在边界下直接补 geometry.bbox。"
            )
        _score, _rank, stem, (idx, geom, shp_path) = top[0]
        return {
            "grid_indices": idx,
            "resolved_from": f"shp_name_match:{stem}",
            "geometry": geom,
            "area_m2": float(geom.area),
            "source_shp": shp_path,
        }

    # 一个都没成：把尝试过的候选和原因带出去，供调用方拼消息
    return {"_tried": tried, "_candidate_stems": [c[2] for c in candidates]}


def _dims(grid, origin_xy):
    nrow, ncol, delr, delc, _nlay = grid_dims(grid)
    return nrow, ncol, delr, delc, origin_xy


# ====================================================================
# 编排
# ====================================================================

def unresolved_message(bc, grid, tried=(), candidate_stems=()) -> str:
    """把"为什么定位不了"和"你可以怎么办"写成一段可直接展示的中文"""
    name = bc.get("name", "?")
    pos = bc.get("position", "?")
    geo = bc.get("geometry") or {}
    lines = [
        f"补给边界 '{name}'（type={bc.get('type')}, position='{pos}'）需要注入位置，"
        f"但无法确定：",
        "  · IR 未提供 geometry.grid_indices",
    ]
    if not geo.get("bbox"):
        lines.append("  · IR 未提供 geometry.bbox / geometry.coordinates")
    if candidate_stems:
        lines.append(f"  · 名称匹配到但不可用的候选：{', '.join(candidate_stems)}")
    else:
        lines.append(f"  · 已上传文件中没有名称匹配 '{pos}' 的图层")
    lines.extend(tried or [])
    lines.append("请任选其一：")
    lines.append(f"  ① 重新上传含该区域的 SHP / ZIP（图层名需含 '{pos}'）；")
    lines.append("  ② 在该边界下补 geometry.bbox = [xmin, ymin, xmax, ymax]（与网格同一坐标系）；")
    lines.append("  ③ 直接提供 geometry.grid_indices 逐格索引。")
    lines.append("")
    lines.append("注：程序拒绝把带浓度的点源静默放大成全域面源 —— 那正是污染羽看起来"
                 "像水流模型结果的原因。")
    return "\n".join(lines)


def resolve_source_zone(
    bc: dict,
    grid: dict,
    origin_xy: Optional[list] = None,
    *,
    uploaded_paths: Iterable[str] = (),
    file_index: Optional[dict] = None,
) -> dict:
    """
    按四级优先级解析一个边界条件的源区。

    Raises:
        SourceZoneUnresolved: 四级全失败，或名称匹配并列，或几何退化。
    """
    geo = bc.get("geometry") or {}
    layer = int(_unwrap(geo.get("layer"), 0) or 0)
    period = int(_unwrap(geo.get("period"), 0) or 0)
    dims = _dims(grid, origin_xy)

    def _wrap(indices, geom, source, kind):
        return {
            "grid_indices": indices,
            "resolved_from": kind,
            "geometry": geom,
            "area_m2": float(geom.area) if geom is not None else None,
            "source_shp": source,
        }

    # 1) 显式 grid_indices —— 原样透传，连 shapely 都不碰
    explicit = geo.get("grid_indices") or []
    if explicit:
        return _wrap(list(explicit), None, None, "ir_grid_indices")

    # 2) IR 给的几何
    coords = geo.get("coordinates")
    if coords:
        geom = _geometry_from_ir(coords, geo.get("shape"))
        idx = cells_in_geometry(geom, *dims, layer=layer, period=period)
        if idx:
            return _wrap(idx, geom, None, "ir_coordinates")

    # 3) IR 给的 bbox（也接受 bc["bbox"] / parameters.bbox 两种懒写法）
    raw_bbox = geo.get("bbox") or bc.get("bbox") or (bc.get("parameters") or {}).get("bbox")
    if raw_bbox:
        geom = bbox_to_geometry(raw_bbox)
        idx = cells_in_geometry(geom, *dims, layer=layer, period=period)
        if idx:
            return _wrap(idx, geom, None, "ir_bbox")

    # 4) 上传 SHP 名称匹配
    matched = _match_by_name(bc, file_index, uploaded_paths, grid,
                             origin_xy, layer, period)
    if matched and matched.get("grid_indices"):
        return matched

    tried = (matched or {}).get("_tried", [])
    stems = (matched or {}).get("_candidate_stems", [])
    raise SourceZoneUnresolved(unresolved_message(bc, grid, tried, stems))


def _geometry_from_ir(coords, shape):
    """把 IR 里的坐标列表转成 shapely 几何"""
    from shapely.geometry import LineString, Point, Polygon

    shape = (shape or "polygon").lower()
    pts = coords
    if pts and isinstance(pts[0], (int, float)):
        pts = [pts]
    if shape.startswith("point"):
        return Point(pts[0][0], pts[0][1])
    if shape.startswith("line"):
        return LineString(pts)
    return Polygon(pts)


def materialize_source_zones(
    json_ir: dict,
    *,
    uploaded_paths: Iterable[str] = (),
    file_index: Optional[dict] = None,
) -> tuple:
    """
    就地为一个 IR 里所有**带浓度但还没有注入位置**的边界解析源区，
    把结果写进 `geometry.grid_indices`，使下游 mapper 完全不必知道文件的存在。

    为什么物化进 IR 而不是给 build_and_run 加参数：
      - 20 个 `build_and_run` 调用点、5 个 `build_simulation`、4 个
        `map_boundary_conditions` 的签名全部不用动；
      - `app/utils/debug_monitor.py` 里 monkeypatch 了 `build_and_run`，wrapper 是
        `_traced_build(self, json_ir, sim_name="auto_model")`，**新增位置参数会被它
        吞掉**；
      - 解析结果落在会话 JSON 里，上传文件所在临时目录失效后重跑仍然可用
        （下次走优先级 1 直接命中）。

    Returns:
        ``(json_ir, resolved, issues)``
        - `resolved`: `{bc_name: 解析结果 dict}`，供 UI 展示"132 格 / 5.28 m³/d"
        - `issues`: 失败项的 `(bc_name, 消息)` 列表，**空表示没有需要提示的**
    """
    grid = json_ir.get("grid", {}) or {}
    origin_xy = _unwrap_domain_origin(json_ir)
    resolved = {}
    issues = []

    for bc in json_ir.get("boundary_conditions", []) or []:
        if not isinstance(bc, dict):
            continue
        params = bc.get("parameters") or {}
        conc = _unwrap(params.get("concentration"))
        if conc is None:
            continue
        geo = bc.setdefault("geometry", {})
        if geo.get("grid_indices"):
            continue  # 已有显式索引，交给 mapper 透传

        # 方位型名称（north_river 之类）交给 mapper 的方位推断，不算失败 ——
        # 否则会对一条本该被正确推断的方位边界误报"源区定位失败"
        if infer_edge_indices(bc.get("name", ""), grid, bc_type=bc.get("type")):
            continue

        try:
            out = resolve_source_zone(bc, grid, origin_xy,
                                      uploaded_paths=uploaded_paths,
                                      file_index=file_index)
        except SourceZoneUnresolved as e:
            issues.append((bc.get("name", "?"), str(e)))
            continue

        geo["grid_indices"] = out["grid_indices"]
        geo["resolved_from"] = out["resolved_from"]
        if out.get("source_shp"):
            geo["source_shp"] = out["source_shp"]
        geo["grid_signature"] = grid_signature(grid)
        resolved[bc.get("name", "?")] = out
        logger.info(
            f"源区 '{bc.get('name')}' 已定位：{len(out['grid_indices'])} 格"
            f"（{out['resolved_from']}）"
        )

    return json_ir, resolved, issues


def _unwrap_domain_origin(json_ir: dict) -> Optional[list]:
    """读取 `domain.origin_xy`（网格原点），缺省 None → 局部坐标系原点 (0,0)"""
    origin = (json_ir.get("domain") or {}).get("origin_xy")
    if origin is None:
        return None
    if isinstance(origin, dict):
        origin = origin.get("value")
    if not origin or len(origin) < 2:
        return None
    return [float(origin[0]), float(origin[1])]


def grid_signature(grid: dict) -> str:
    """
    网格指纹，用于识别陈旧的 `grid_indices`。

    East Texas 的 grid 里没有 delr/delc，132 格完全依赖 20.0 这个默认值；一旦用户
    后来给出了真实的 delr/delc，旧格子就失效了，必须重新解析而不是沿用。
    """
    nrow, ncol, delr, delc, nlay = grid_dims(grid)
    return f"{nrow}x{ncol}x{nlay}@{float(delr[0]):g}x{float(delc[0]):g}"


def zone_signature_stale(json_ir: dict) -> list:
    """返回 `grid_signature` 与当前网格不符的边界名列表"""
    cur = grid_signature(json_ir.get("grid", {}) or {})
    stale = []
    for bc in json_ir.get("boundary_conditions", []) or []:
        if not isinstance(bc, dict):
            continue
        geo = bc.get("geometry") or {}
        sig = geo.get("grid_signature")
        if sig and sig != cur:
            stale.append(bc.get("name", "?"))
    return stale

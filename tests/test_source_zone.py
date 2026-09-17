"""
源区定位（Source Zone）与时间离散（total_time）回归测试

覆盖本轮修复的两个缺陷：

  缺陷 A（源区）—— East Texas 填埋场只有 132 格（占全域 0.59%），但带浓度的
  RCH 边界找不到注入位置时会被**静默**展开成 layer0 全网格（22454 格），源被
  放大 170 倍，污染羽于是变成一张覆盖全域的毯子、看起来像水流模型结果。
  修复：四级解析（IR grid_indices → IR coordinates → IR bbox → SHP 名称匹配），
  全失败**硬拒绝**，绝不回落全网格。

  缺陷 B（时间）—— 提示词要"模拟时长15000天，均匀分为10个应力期"，但
  `transport.stress_periods.perlen=15000` 按 MF6 语义是**单期**长度，
  总时长变成 150000 d（差 10 倍）。修复：新增一等公民字段 `total_time`，
  `perlen = total_time / nper`。

**本文件必须自包含**：真实案例的 `landfill_arcs.shp` 躺在系统临时目录
（`%TEMP%\gm_zip_*`）里，随时会被清理，所以这里用 `TemporaryDirectory()`
现造一份几何完全相同的 SHP（241×225 m 的 4 条 LineString 闭合环）。

不依赖 pytest，沿用仓库的脚本式风格（`check(name, condition, detail)`）。
运行：`python tests/test_source_zone.py`
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import tempfile
import numpy as np

from app.modeling.source_zone import (
    SourceZoneUnresolved,
    bbox_to_geometry,
    cells_in_geometry,
    grid_dims,
    infer_edge_indices,
    materialize_source_zones,
    polygonize_layer,
    resolve_source_zone,
)
from app.modeling.model_mapper import FloPyModelMapper
from app.physics_engine.validator import ClosedLoopValidator
from app.utils.grid_coords import grid_cell_centers
from app.utils.ir_normalize import merge_time_sources, normalize_stress_periods

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name}  FAILED  {detail}")


def section(title):
    print(f"\n--- {title} ---")


# ============================================================
# 常量：East Texas 的真实几何与网格
# ============================================================
# 填埋场外框（与真实 landfill_arcs.shp 逐位相同）
X0, Y0, X1, Y1 = 615026.0, 3428453.0, 615267.0, 3428678.0
AREA = (X1 - X0) * (Y1 - Y0)          # 241 × 225 = 54225.0 m²

# 模型网格：103×218，格尺寸 20 m（IR 里没有 delr/delc，全靠默认值 20.0）
NROW, NCOL, DELR, DELC = 103, 218, 20.0, 20.0

# 网格原点 = SHP 外框左下角（_detect_basemap_inputs / domain.origin_xy 的约定）
ORIGIN = [612284.0, 3427412.0]

# 期望：132 格 = 11 行 × 12 列
EXP_ROWS = list(range(52, 63))        # 52..62
EXP_COLS = list(range(137, 149))      # 137..148
EXP_NCELLS = len(EXP_ROWS) * len(EXP_COLS)

RECT_RING = [
    [(X0, Y0), (X1, Y0)],
    [(X1, Y0), (X1, Y1)],
    [(X1, Y1), (X0, Y1)],
    [(X0, Y1), (X0, Y0)],
]


def make_shp(dirpath: Path, stem: str = "landfill_arcs", ring=None, as_polygon=False):
    """
    在 dirpath 下现造一个 SHP，返回 .shp 路径。

    as_polygon=False → 4 条 LineString（与真实 landfill_arcs.shp 一致，需 polygonize）
    as_polygon=True  → 1 个 Polygon（用于测"面 > 线"的排名规则）
    """
    import geopandas as gpd
    from shapely.geometry import LineString, Polygon

    ring = ring if ring is not None else RECT_RING
    if as_polygon:
        geoms = [Polygon([(X0, Y0), (X1, Y0), (X1, Y1), (X0, Y1)])]
    else:
        geoms = [LineString(seg) for seg in ring]
    gdf = gpd.GeoDataFrame({"id": list(range(len(geoms)))}, geometry=geoms)
    path = dirpath / f"{stem}.shp"
    gdf.to_file(path)
    return str(path)


def make_rch_ir(**params):
    """构造一个只有一条带浓度 RCH 边界的 IR（形态与 East Texas 一致）。"""
    p = {"rate": {"value": 0.0001, "unit": "m/d", "source": "user_input"},
         "concentration": {"value": 20000.0, "unit": "ppm", "source": "user_input"}}
    p.update(params)
    return {
        "ir_version": "1.0",
        "project": {"name": "tz", "model_type": "solute_transport"},
        "domain": {"x_range": [0, 4360], "y_range": [0, 2060],
                   "units": "meters", "origin_xy": list(ORIGIN)},
        "grid": {"nrow": NROW, "ncol": NCOL, "nlay": 2,
                 "top": 230.0, "bottom": 175.0},
        "aquifer": {"type": "confined",
                    "parameters": {"hk": {"value": 5.5, "unit": "m/d"},
                                   "porosity": {"value": 0.3}}},
        "boundary_conditions": [
            {"type": "rch", "name": "landfill_boundary", "position": "landfill",
             "parameters": p},
        ],
        "stress_periods": 10,
        "transport": {"enabled": True,
                      "dispersion": {"al": {"value": 20.0}, "trpt": 0.2, "trpv": 0.1},
                      "stress_periods": {"nper": 10, "perlen": 15000, "nstp": 1,
                                         "steady_state": False, "period_data": []}},
    }


mapper = FloPyModelMapper()
validator = ClosedLoopValidator()

print("=" * 62)
print("源区定位 + 时间离散 回归测试")
print("=" * 62)

# ============================================================
# 1. 几何：polygonize / bbox
# ============================================================
section("1. 几何基元")

from shapely.geometry import box, LineString, Polygon  # noqa: E402

faces = polygonize_layer([LineString(s) for s in RECT_RING])
check("1.1 4 条闭合线 → 1 个面", len(faces) == 1, f"got {len(faces)}")
check("1.2 面积 = 54225.0 m²", len(faces) == 1 and abs(faces[0].area - AREA) < 1e-6,
      f"got {faces[0].area if faces else None}")

poly = Polygon([(X0, Y0), (X1, Y0), (X1, Y1), (X0, Y1)])
check("1.3 已是面则原样透传", polygonize_layer([poly])[0].area == AREA)

try:
    polygonize_layer([LineString([(0, 0), (10, 0)]),
                      LineString([(20, 20), (30, 20)])])
    check("1.4 不闭合的线 → raise", False, "没有抛异常")
except SourceZoneUnresolved as e:
    check("1.4 不闭合的线 → raise", "无法闭合成面" in str(e), str(e)[:80])

bb = bbox_to_geometry([X0, Y0, X1, Y1])
check("1.5 bbox_to_geometry 面积一致", abs(bb.area - AREA) < 1e-6)

try:
    bbox_to_geometry([X1, Y0, X0, Y1])
    check("1.6 退化 bbox → raise", False, "没有抛异常")
except SourceZoneUnresolved:
    check("1.6 退化 bbox → raise", True)

# ============================================================
# 2. 验收锁：132 格 / rows 52..62 / cols 137..148
# ============================================================
section("2. 验收锁（本轮缺陷 A 的核心断言）")

idx = cells_in_geometry(bb, NROW, NCOL, DELR, DELC, ORIGIN, layer=0, period=0)
rows = sorted({i["row"] for i in idx})
cols = sorted({i["col"] for i in idx})

check(f"2.1 恰好 {EXP_NCELLS} 格", len(idx) == EXP_NCELLS, f"got {len(idx)}")
check("2.2 rows == 52..62", rows == EXP_ROWS, f"got {rows}")
check("2.3 cols == 137..148", cols == EXP_COLS, f"got {cols}")
check("2.4 记录形状正确",
      all(set(i) == {"layer", "row", "col", "period"} for i in idx) and idx[0] ==
      {"layer": 0, "row": 52, "col": 137, "period": 0},
      f"first={idx[0] if idx else None}")
check("2.5 行优先升序", idx == sorted(idx, key=lambda d: (d["row"], d["col"])))
check("2.6 全部在 layer 0（RCH 必须是 layer 0）",
      all(i["layer"] == 0 for i in idx))
check("2.7 占全域比例 ≈ 0.59%",
      abs(len(idx) / (NROW * NCOL) * 100 - 0.59) < 0.01,
      f"got {len(idx)/(NROW*NCOL)*100:.3f}%")
check("2.8 绝非全网格（旧 bug 是 22454 格）",
      len(idx) != NROW * NCOL and len(idx) < 0.01 * NROW * NCOL,
      f"got {len(idx)}")

# 完全落在域外 → 明确消息（区别于"名称没匹配上"）
try:
    cells_in_geometry(box(1e6, 1e6, 1e6 + 10, 1e6 + 10), NROW, NCOL, DELR, DELC, ORIGIN)
    check("2.9 域外几何 → raise", False, "没有抛异常")
except SourceZoneUnresolved as e:
    check("2.9 域外几何 → raise", "完全落在模型域外" in str(e), str(e)[:80])

# 经纬度当米传 → 点破
try:
    cells_in_geometry(box(-95.2, 31.5, -95.1, 31.6), NROW, NCOL, DELR, DELC, ORIGIN)
    check("2.10 经纬度 → raise", False, "没有抛异常")
except SourceZoneUnresolved as e:
    check("2.10 经纬度 → raise", "疑似经纬度" in str(e), str(e)[:80])

# ============================================================
# 3. 坐标一致性锁：建模侧与绘图侧必须逐元素相同
# ============================================================
section("3. 坐标一致性锁（建模 vs 绘图）")

from app.analysis.visualizer import VisualizationEngine  # noqa: E402

viz = VisualizationEngine()
shp_bounds = [ORIGIN[0], ORIGIN[1], 616244.0, 3429283.0]
Xa, Ya, extent_a = grid_cell_centers(NROW, NCOL, DELR, DELC, origin_xy=ORIGIN)
Xb, Yb, extent_b = viz._grid_coords(NROW, NCOL, DELR, DELC, shp_bounds=shp_bounds)

check("3.1 X 逐元素全等", np.array_equal(Xa, Xb))
check("3.2 Y 逐元素全等", np.array_equal(Ya, Yb))
check("3.3 model_extent 相同", np.allclose(extent_a, extent_b))
check("3.4 形状 == (nrow, ncol)", Xa.shape == (NROW, NCOL), f"got {Xa.shape}")

# delr/delc 长度不符时整体退化为均值（两条路都要退化，否则会错位）
Xc, Yc, _ = grid_cell_centers(4, 4, DELR, DELC, origin_xy=ORIGIN)
Xd, Yd, _ = viz._grid_coords(4, 4, DELR, DELC, shp_bounds=shp_bounds)
check("3.5 退化路径也全等", np.array_equal(Xc, Xd) and np.array_equal(Yc, Yd))

# ============================================================
# 4. 四级解析优先级
# ============================================================
section("4. 四级解析优先级")

tmp = tempfile.TemporaryDirectory()
tmpdir = Path(tmp.name)
shp_path = make_shp(tmpdir, "landfill_arcs")   # 线要素，需 polygonize
shp_paths = [shp_path]
file_index = {"landfill_arcs": {"geometry_type": "LineString", "feature_count": 4}}

grid = make_rch_ir()["grid"]

# 4.1 显式 grid_indices 最高优先（原样透传，连 shapely 都不碰）
bc_explicit = {"type": "rch", "name": "landfill_boundary", "position": "landfill",
               "geometry": {"grid_indices": [{"layer": 0, "row": 1, "col": 2}]},
               "parameters": {"concentration": 20000.0, "rate": 1e-4}}
out = resolve_source_zone(bc_explicit, grid, ORIGIN)
check("4.1 IR grid_indices 优先", out["resolved_from"] == "ir_grid_indices"
      and out["grid_indices"] == [{"layer": 0, "row": 1, "col": 2}], str(out["resolved_from"]))

# 4.2 IR coordinates
bc_coords = {"type": "rch", "name": "landfill_boundary", "position": "landfill",
             "geometry": {"coordinates": [(X0, Y0), (X1, Y0), (X1, Y1), (X0, Y1)]},
             "parameters": {"concentration": 20000.0, "rate": 1e-4}}
out = resolve_source_zone(bc_coords, grid, ORIGIN)
check("4.2 IR coordinates", out["resolved_from"] == "ir_coordinates"
      and len(out["grid_indices"]) == EXP_NCELLS, str(out.get("resolved_from")))

# 4.3 IR bbox（也接受 bc["bbox"] / parameters.bbox 的懒写法）
bc_bbox = {"type": "rch", "name": "landfill_boundary", "position": "landfill",
           "geometry": {"bbox": [X0, Y0, X1, Y1]},
           "parameters": {"concentration": 20000.0, "rate": 1e-4}}
out = resolve_source_zone(bc_bbox, grid, ORIGIN)
check("4.3 IR bbox", out["resolved_from"] == "ir_bbox"
      and len(out["grid_indices"]) == EXP_NCELLS, str(out.get("resolved_from")))

bc_bbox2 = {"type": "rch", "name": "landfill_boundary", "position": "landfill",
            "bbox": [X0, Y0, X1, Y1],
            "parameters": {"concentration": 20000.0, "rate": 1e-4}}
check("4.3b bc 顶层 bbox 懒写法也认",
      resolve_source_zone(bc_bbox2, grid, ORIGIN)["resolved_from"] == "ir_bbox")

# 4.4 SHP 名称匹配（landfill_boundary ↔ landfill_arcs.shp）
bc_name = {"type": "rch", "name": "landfill_boundary", "position": "landfill",
           "parameters": {"concentration": 20000.0, "rate": 1e-4}}
out = resolve_source_zone(bc_name, grid, ORIGIN, uploaded_paths=shp_paths,
                          file_index=file_index)
check("4.4 SHP 名称匹配", out["resolved_from"] == "shp_name_match:landfill_arcs",
      str(out.get("resolved_from")))
check("4.4b 名称匹配得到同样 132 格", len(out["grid_indices"]) == EXP_NCELLS,
      f"got {len(out['grid_indices'])}")
check("4.4c 面积用图层自身投影单位（不是 EPSG:3857 的膨胀值）",
      abs(out["area_m2"] - AREA) < 1e-6, f"got {out['area_m2']}")

# 4.4d 只给 uploaded_paths、不给 file_index 也必须能匹配
#      （file_index 未必被调用方一路传下来，只信它会造成"假失败"）
out = resolve_source_zone(dict(bc_name), grid, ORIGIN, uploaded_paths=shp_paths)
check("4.4d 仅凭 uploaded_paths 也能匹配",
      out["resolved_from"] == "shp_name_match:landfill_arcs",
      str(out.get("resolved_from")))

# ============================================================
# 5. 名称匹配的负例与歧义
# ============================================================
section("5. 名称匹配负例 / 歧义 / 三重失败")

# 5.1 名称对不上 → raise，消息里要有 BC 名、position、候选清单、三条出路
bc_nomatch = {"type": "rch", "name": "recharge_zone_a", "position": "somewhere",
              "parameters": {"concentration": 20000.0, "rate": 1e-4}}
try:
    resolve_source_zone(bc_nomatch, grid, ORIGIN, uploaded_paths=shp_paths,
                        file_index=file_index)
    check("5.1 名称不匹配 → raise", False, "没有抛异常")
except SourceZoneUnresolved as e:
    msg = str(e)
    check("5.1 名称不匹配 → raise", True)
    check("5.1b 消息含 BC 名", "recharge_zone_a" in msg, msg[:120])
    check("5.1c 消息含 position", "somewhere" in msg, msg[:120])
    check("5.1d 消息含三条出路",
          "geometry.grid_indices" in msg and "geometry.bbox" in msg, msg[:200])
    check("5.1e 消息点明拒绝静默放大", "全域面源" in msg, msg[-120:])

# 5.2 最高名次并列 → 必须 raise（静默挑一个是本次要消灭的 bug 类型）
tmp2 = tempfile.TemporaryDirectory()
shp_polys = make_shp(Path(tmp2.name), "landfill_polys", as_polygon=True)
# 同名的两个图层：landfill_arcs（线）+ landfill_polys（面）
paths_two = [shp_path, shp_polys]
# 面（rank 2）应胜过线（rank 1），因此这里**不该**报歧义，而是选中 polys
out = resolve_source_zone(dict(bc_name), grid, ORIGIN, uploaded_paths=paths_two)
check("5.2 面 > 线：同名时选面不报歧义",
      out["resolved_from"] == "shp_name_match:landfill_polys",
      str(out.get("resolved_from")))

# 真正并列：两个同类型同名的候选（landfill_x / landfill_y 都是线）
shp_x = make_shp(Path(tmp2.name), "landfill_x")
shp_y = make_shp(Path(tmp2.name), "landfill_y")
try:
    resolve_source_zone(dict(bc_name), grid, ORIGIN,
                        uploaded_paths=[shp_x, shp_y])
    check("5.3 同级并列 → raise", False, "没有抛异常")
except SourceZoneUnresolved as e:
    check("5.3 同级并列 → raise", "多个图层" in str(e), str(e)[:120])

# 5.4 三重失败（IR 无索引/无几何/名称不匹配）绝不回落全网格
ir_fail = make_rch_ir()
ir_fail["boundary_conditions"][0]["name"] = "recharge_zone_a"
ir_fail["boundary_conditions"][0]["position"] = "somewhere"
_ir, resolved_f, issues_f = materialize_source_zones(
    ir_fail, uploaded_paths=[shp_path], file_index=file_index)
check("5.4 三重失败 → issues 非空", len(issues_f) == 1, str(issues_f))
check("5.4b 三重失败 → resolved 为空", resolved_f == {}, str(resolved_f))
check("5.4c 三重失败 → IR 里没有 grid_indices",
      not (ir_fail["boundary_conditions"][0].get("geometry") or {}).get("grid_indices"))

# 5.4d mapper 在"有浓度但没位置"时必须硬拒绝，绝不展开全网格
try:
    mapper.map_boundary_conditions(ir_fail)
    check("5.4d mapper 硬拒绝", False, "没有抛异常")
except SourceZoneUnresolved:
    check("5.4d mapper 硬拒绝", True)

# 5.5 方位型名称交给 mapper 的方位推断，不算失败（不误报）
ir_dir = make_rch_ir()
ir_dir["boundary_conditions"][0] = {
    "type": "rch", "name": "natural_recharge",
    "parameters": {"rate": 1e-5},
}
_ir2, resolved_d, issues_d = materialize_source_zones(ir_dir, uploaded_paths=[shp_path])
check("5.5 无浓度的补给边界不算源区失败", issues_d == [], str(issues_d))

# rch/evt/wel 一律不参与方位推断
for t in ("rch", "evt", "wel"):
    check(f"5.6 {t} 不做方位推断",
          infer_edge_indices("north_boundary", grid, bc_type=t) == [])
check("5.7 其它类型仍做方位推断",
      len(infer_edge_indices("north_boundary", grid, bc_type="chd")) > 0)
check("5.8 top_of_landfill 不会被误判成北边界",
      infer_edge_indices("top_of_landfill", grid, bc_type="rch") == [])

# ---- 5.9 方位名 → 行/列号，必须与绘图坐标同向（本 bug 曾无测试而存活）----
# 不变量：叫 "north" 的边界必须落在 **y 最大**的那一排。
# 此前 north → row 0，而 grid_cell_centers 让 row 0 的 y 最小（图的下方），
# 于是名为 north 的边界被画在图底部、水头图上下镜像，且没有任何断言拦得住。
_nrow_t = int(grid["nrow"])
_ncol_t = int(grid["ncol"])
_dr = [20.0] * _ncol_t
_dc = [20.0] * _nrow_t
_Xt, _Yt, _ = grid_cell_centers(_nrow_t, _ncol_t, _dr, _dc, origin_xy=[0.0, 0.0])

_north = infer_edge_indices("north_boundary", grid, bc_type="chd")
_south = infer_edge_indices("south_boundary", grid, bc_type="chd")
_west = infer_edge_indices("west_boundary", grid, bc_type="chd")
_east = infer_edge_indices("east_boundary", grid, bc_type="chd")

check("5.9a north 全部落在最北一排(row=nrow-1)",
      {i["row"] for i in _north} == {_nrow_t - 1}, str(sorted({i["row"] for i in _north})))
check("5.9b south 全部落在最南一排(row=0)",
      {i["row"] for i in _south} == {0}, str(sorted({i["row"] for i in _south})))
check("5.9c west 全部落在最西一列(col=0)",
      {i["col"] for i in _west} == {0}, str(sorted({i["col"] for i in _west})))
check("5.9d east 全部落在最东一列(col=ncol-1)",
      {i["col"] for i in _east} == {_ncol_t - 1}, str(sorted({i["col"] for i in _east})))
# 两条约定合起来才成立：方位名指定的那一排，其 y 必须真的比对面那排大
check("5.9e north 一排的 y > south 一排的 y（名与图同向）",
      float(_Yt[_nrow_t - 1, 0]) > float(_Yt[0, 0]),
      f"y(row{_nrow_t-1})={float(_Yt[_nrow_t-1, 0]):.1f} y(row0)={float(_Yt[0, 0]):.1f}")
check("5.9f west 一列的 x < east 一列的 x",
      float(_Xt[0, 0]) < float(_Xt[0, _ncol_t - 1]),
      f"x(col0)={float(_Xt[0, 0]):.1f} x(col{_ncol_t-1})={float(_Xt[0, _ncol_t-1]):.1f}")

# ============================================================
# 6. 物化 + recharge_scope
# ============================================================
section("6. 物化进 IR 与 recharge_scope")

ir = make_rch_ir()
ir_m, resolved_m, issues_m = materialize_source_zones(
    ir, uploaded_paths=[shp_path], file_index=file_index)
check("6.1 无 issue", issues_m == [], str(issues_m))
check("6.2 resolved 含 landfill_boundary", "landfill_boundary" in resolved_m)
geo = ir_m["boundary_conditions"][0]["geometry"]
check("6.3 grid_indices 已写回 IR", len(geo.get("grid_indices", [])) == EXP_NCELLS,
      f"got {len(geo.get('grid_indices', []))}")
check("6.4 resolved_from 已写回", geo.get("resolved_from") == "shp_name_match:landfill_arcs",
      str(geo.get("resolved_from")))
check("6.5 source_shp 已写回", bool(geo.get("source_shp")))
check("6.6 grid_signature 已写回", geo.get("grid_signature") == f"{NROW}x{NCOL}x2@20x20",
      str(geo.get("grid_signature")))

# 6.7 scope="zone"（默认）：只有 132 条记录，全部 conc=20000
pkgs = mapper.map_boundary_conditions(ir_m)
recs_zone = pkgs["rch"][0]
check("6.7 默认 scope=zone → 132 条", len(recs_zone) == EXP_NCELLS,
      f"got {len(recs_zone)}")
check("6.7b 全部 conc=20000",
      all(abs(r[4] - 20000.0) < 1e-9 for r in recs_zone),
      f"got {sorted({r[4] for r in recs_zone})[:5]}")
check("6.7c 全部 rate=1e-4",
      all(abs(r[3] - 1e-4) < 1e-12 for r in recs_zone))
check("6.7d 全部 layer 0",
      all(r[0] == 0 for r in recs_zone))

# 6.8 scope="domain"：全域 103×218 条，其中恰好 132 条 conc=20000、其余 0
ir_dom = make_rch_ir(recharge_scope="domain")
ir_dom, _res, _iss = materialize_source_zones(
    ir_dom, uploaded_paths=[shp_path], file_index=file_index)
pkgs_dom = mapper.map_boundary_conditions(ir_dom)
recs_dom = pkgs_dom["rch"][0]
n_all = NROW * NCOL
check("6.8 scope=domain → 全域条数", len(recs_dom) == n_all, f"got {len(recs_dom)}")
check("6.8b 恰好 132 条 conc=20000",
      sum(1 for r in recs_dom if abs(r[4] - 20000.0) < 1e-9) == EXP_NCELLS)
check("6.8c 其余 22322 条 conc=0",
      sum(1 for r in recs_dom if abs(r[4]) < 1e-12) == n_all - EXP_NCELLS)
ir_bg = make_rch_ir(recharge_scope="domain", background_concentration=5.0)
ir_bg, _r2, _i2 = materialize_source_zones(
    ir_bg, uploaded_paths=[shp_path], file_index=file_index)
recs_bg = mapper.map_boundary_conditions(ir_bg)["rch"][0]
check("6.8e background_concentration 生效",
      sum(1 for r in recs_bg if abs(r[4] - 5.0) < 1e-9) == n_all - EXP_NCELLS,
      f"got {sum(1 for r in recs_bg if abs(r[4] - 5.0) < 1e-9)}")

# 6.9 全场标量补给（无浓度）走标量分支，且不会覆盖已有的逐格记录
ir_scalar = make_rch_ir()
ir_scalar["boundary_conditions"] = [
    {"type": "rch", "name": "natural_recharge", "parameters": {"rate": 1e-5}},
]
check("6.9 无浓度 RCH 走标量分支",
      mapper.map_boundary_conditions(ir_scalar)["rch"][0] == 1e-5,
      str(mapper.map_boundary_conditions(ir_scalar)["rch"][0]))

# 6.10 逐格记录在前、标量在后 → 保留逐格，不静默覆盖
ir_mix = make_rch_ir()
ir_mix, _r3, _i3 = materialize_source_zones(
    ir_mix, uploaded_paths=[shp_path], file_index=file_index)
ir_mix["boundary_conditions"].append(
    {"type": "rch", "name": "natural_recharge", "parameters": {"rate": 1e-5}})
recs_mix = mapper.map_boundary_conditions(ir_mix)["rch"][0]
check("6.10 标量不覆盖已有逐格记录", isinstance(recs_mix, list)
      and len(recs_mix) == EXP_NCELLS, f"got {type(recs_mix).__name__} {len(recs_mix) if isinstance(recs_mix, list) else ''}")

# ============================================================
# 7. total_time（缺陷 B）
# ============================================================
section("7. total_time 与 perlen 语义")

# 7.1 裸整数 stress_periods 归一化
check("7.1 裸整数 10 → {nper: 10}", normalize_stress_periods(10) == {"nper": 10})
check("7.1b bool 不算 nper", normalize_stress_periods(True) == {})
check("7.1c None → {}", normalize_stress_periods(None) == {})
check("7.1d 字符串 → {}", normalize_stress_periods("10") == {})
check("7.1e 列表取首元素",
      normalize_stress_periods([{"nper": 3}]) == {"nper": 3})
check("7.1f dict 原样", normalize_stress_periods({"nper": 2}) == {"nper": 2})

# 7.2 total_time=15000 / nper=10 → 每期 1500，总 15000
ir_t = make_rch_ir()
ir_t["stress_periods"] = {"nper": 10, "steady_state": False,
                          "total_time": 15000, "nstp": 1}
t = mapper._map_time_params(ir_t)
check("7.2 total_time 按 nper 等分", abs(t["perlen"] - 1500.0) < 1e-9,
      f"got perlen={t['perlen']}")
check("7.2b 总时长 = 15000 d",
      abs(sum(p["perlen"] for p in t["period_data"]) - 15000.0) < 1e-6,
      f"got {sum(p['perlen'] for p in t['period_data'])}")
check("7.2c nper=10", t["nper"] == 10 and len(t["period_data"]) == 10)

# 7.3 total_time 覆盖陈旧的平铺 perlen（East Texas 的真实形态）
ir_stale = make_rch_ir()
ir_stale["stress_periods"] = {"nper": 10, "steady_state": False,
                              "total_time": 15000, "perlen": 15000, "nstp": 1}
t2 = mapper._map_time_params(ir_stale)
check("7.3 total_time 覆盖平铺 perlen", abs(t2["perlen"] - 1500.0) < 1e-9,
      f"got perlen={t2['perlen']}")
check("7.3b 总时长仍是 15000 而非 150000",
      abs(sum(p["perlen"] for p in t2["period_data"]) - 15000.0) < 1e-6,
      f"got {sum(p['perlen'] for p in t2['period_data'])}")

# 7.4 period_data 优先级最高
ir_pd = make_rch_ir()
ir_pd["stress_periods"] = {"nper": 2, "steady_state": False, "total_time": 9999,
                           "period_data": [{"perlen": 100, "nstp": 10},
                                           {"perlen": 900, "nstp": 30}]}
t3 = mapper._map_time_params(ir_pd)
check("7.4 period_data 最高优先",
      [p["perlen"] for p in t3["period_data"]] == [100.0, 900.0],
      str([p["perlen"] for p in t3["period_data"]]))

# 7.5 兼容性承诺：没有 total_time 时，平铺 perlen=15000 × 10 仍然是 150000
ir_old = make_rch_ir()
ir_old["stress_periods"] = {"nper": 10, "steady_state": False,
                            "perlen": 15000, "nstp": 1}
t4 = mapper._map_time_params(ir_old)
check("7.5 无 total_time 时 perlen 仍是单期长度（总计 150000）",
      abs(sum(p["perlen"] for p in t4["period_data"]) - 150000.0) < 1e-6,
      f"got {sum(p['perlen'] for p in t4['period_data'])}")

# 7.6 transport.stress_periods 回退（East Texas 的真实形态：顶层裸整数）
ir_fb = make_rch_ir()          # 顶层 stress_periods = 10，时间信息在 transport
t5 = mapper._map_time_params(ir_fb)
check("7.6 顶层裸整数 → 回退读 transport", t5["nper"] == 10, f"got {t5['nper']}")
check("7.6b 回退后 perlen=15000 而非退化成 1 天",
      abs(sum(p["perlen"] for p in t5["period_data"]) - 150000.0) < 1e-6,
      f"got {sum(p['perlen'] for p in t5['period_data'])}")

# 7.7 merge_time_sources 合并规则
m = merge_time_sources({"stress_periods": 10,
                        "transport": {"stress_periods": {"nper": 10, "perlen": 15000}}})
check("7.7 合并取到 perlen", m.get("perlen") == 15000, str(m))
m2 = merge_time_sources({"stress_periods": {"nper": 3, "perlen": 5},
                         "transport": {"stress_periods": {"nper": 3, "perlen": 15000}}})
check("7.7b 顶层已给时间键则不合并", m2.get("perlen") == 5, str(m2))

# ============================================================
# 8. 校验层
# ============================================================
section("8. 校验层：不再崩溃 / 时间离散绊线")

# 8.1 裸整数 stress_periods 不再抛 AttributeError
try:
    iss = validator._check_pre_run({"stress_periods": 10,
                                    "boundary_conditions": []})
    check("8.1 _check_pre_run 吃裸整数不再崩", True)
except AttributeError as e:
    check("8.1 _check_pre_run 吃裸整数不再崩", False, str(e))

# 8.2 total_time 与 period_data 冲突 → ERROR
ir_conflict = make_rch_ir()
ir_conflict["stress_periods"] = {
    "nper": 2, "steady_state": False, "total_time": 15000,
    "period_data": [{"perlen": 100, "nstp": 1}, {"perlen": 100, "nstp": 1}],
}
iss2 = validator._check_time_discretization(ir_conflict)
errs = [i for i in iss2 if i.severity.name == "ERROR"]
check("8.2 total_time 与 period_data 冲突 → ERROR", len(errs) == 1,
      f"got {[(i.severity.name, i.description[:60]) for i in iss2]}")

# 8.3 total_time 与平铺 perlen 不符 → 只 WARNING（不阻断）
#     此时 elapsed = 15000×10 = 150000 d > 1e5，长时段提醒也会一并触发，
#     所以是两条 WARNING 而不是一条；两条都是对的，关键是**没有 ERROR**。
iss3 = validator._check_time_discretization(ir_stale)
check("8.3 total_time 与平铺 perlen 不符 → 全为 WARNING 且无 ERROR",
      len(iss3) >= 1 and all(i.severity.name == "WARNING" for i in iss3)
      and any("平铺 perlen" in i.description for i in iss3),
      f"got {[(i.severity.name, i.description[:60]) for i in iss3]}")

# 8.4 East Texas 真实形态（顶层裸整数 + transport 里 perlen=15000）
#     → 必须被绊线抓到"总时长 150000 d"
iss4 = validator._check_time_discretization(ir_fb)
check("8.4 East Texas 形态被抓到（150000 d）",
      any("150000" in i.description for i in iss4),
      f"got {[(i.severity.name, i.description[:70]) for i in iss4]}")

# 8.5 自洽的 total_time 不报任何 issue（保住既有基线）
ir_ok = make_rch_ir()
ir_ok["stress_periods"] = {"nper": 10, "steady_state": False,
                           "total_time": 15000, "nstp": 10}
check("8.5 自洽 total_time 无 issue",
      validator._check_time_discretization(ir_ok) == [],
      str([(i.severity.name, i.description[:50])
           for i in validator._check_time_discretization(ir_ok)]))

# 8.6 超出物理范围 → ERROR（PHYSICAL_RANGES 真的生效，不只是加了个字典条目）
from app.physics_engine.constraints import ParameterRangeConstraint  # noqa: E402
prc = ParameterRangeConstraint()
check("8.6 perlen 上界 1e5（15000 仍合法）",
      prc.check_param("perlen", 15000.0).is_valid)
check("8.6b perlen 超界被拒", not prc.check_param("perlen", 2e5).is_valid)
check("8.6c nper 超界被拒", not prc.check_param("nper", 5000).is_valid)
check("8.6d total_time 超界被拒", not prc.check_param("total_time", 1e8).is_valid)

# 8.7 check_json_ir_params 真的会遍历 stress_periods
res_range = prc.check_json_ir_params({
    "aquifer": {}, "transport": {},
    "stress_periods": {"nper": 10, "perlen": 2e5},   # perlen 超界
})
viol = [r for r in res_range if not r.is_valid]
check("8.7 check_json_ir_params 遍历到 stress_periods", len(viol) >= 1,
      f"got {[r.param_name for r in res_range]}")
check("8.7b 并在描述里标出来源",
      any("stress_periods" in (r.description or "") for r in viol),
      str([r.description[:60] for r in viol]))

# 8.8 合成的代表期不得升级为 ERROR（否则用虚构的 Δt 阻断用户）
iss5 = validator._check_stability({
    "grid": {"nrow": 10, "ncol": 10, "delr": 20.0},
    "domain": {"x_range": [0, 1000], "y_range": [0, 1000]},
    "aquifer": {"parameters": {"hk": {"value": 5.0}, "porosity": {"value": 0.3}}},
    "boundary_conditions": [{"type": "chd", "name": "north", "parameters": {"head": 100}},
                            {"type": "chd", "name": "south", "parameters": {"head": 90}}],
    "stress_periods": {"nper": 1, "perlen": 100000},
})
check("8.8 合成代表期的结论不高于 WARNING",
      all(i.severity.name != "ERROR" for i in iss5),
      str([(i.severity.name, i.description[:50]) for i in iss5]))

tmp.cleanup()
tmp2.cleanup()

# ============================================================
# 汇总
# ============================================================
print("\n" + "=" * 62)
total = passed + failed
print(f"源区定位 + 时间离散 测试结果: {passed}/{total} 通过")
if failed == 0:
    print("✅ 全部通过")
else:
    print(f"⚠️ {failed} 项失败")
print("=" * 62)
sys.exit(1 if failed else 0)

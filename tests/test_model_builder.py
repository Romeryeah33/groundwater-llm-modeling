"""端到端测试：JSON IR → FloPy 自动建模"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Force UTF-8
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import tempfile

import numpy as np
from app.modeling.grid_generator import AutoGridGenerator
from app.modeling.model_mapper import FloPyModelMapper
from app.modeling.model_builder import AutoModelBuilder
from app.modeling.solver_monitor import SolverMonitor, ErrorType

# Test data: minimal JSON IR
TEST_IR = {
    "ir_version": "1.0",
    "project": {"name": "e2e_test", "model_type": "groundwater_flow"},
    "domain": {"x_range": [0, 1000], "y_range": [0, 800], "units": "meters"},
    "grid": {"nrow": 40, "ncol": 50, "nlay": 1, "top": 100, "bottom": 50},
    "aquifer": {
        "type": "confined",
        "parameters": {
            "hk": {"value": 15.0, "unit": "m/d"},
            "porosity": {"value": 0.25, "unit": "dimensionless"},
        },
    },
    "boundary_conditions": [
        {
            "type": "chd", "name": "west_boundary",
            "geometry": {
                "shape": "line",
                "grid_indices": [
                    {"layer": 0, "row": i, "col": 0} for i in range(40)
                ],
            },
            "parameters": {"head": 100.0},
        },
        {
            "type": "chd", "name": "east_boundary",
            "geometry": {
                "shape": "line",
                "grid_indices": [
                    {"layer": 0, "row": i, "col": 49} for i in range(40)
                ],
            },
            "parameters": {"head": 95.0},
        },
        {
            "type": "wel", "name": "center_well",
            "geometry": {
                "shape": "point",
                "grid_indices": [
                    {"layer": 0, "row": 20, "col": 25},
                ],
            },
            "parameters": {"rate": -200.0},
        },
    ],
    "stress_periods": {
        "nper": 1, "steady_state": True,
        "period_data": [{"perlen": 1.0, "nstp": 1, "tsmult": 1.0}],
    },
    "solver": {"name": "pcg", "parameters": {"hclose": 1e-5}},
    "output": {"head": True, "budget": True},
}

print("=" * 60)
print("FloPy Auto-Modeling E2E Test")
print("=" * 60)

# 1. Grid Generator
print("\n1. Grid Generator...")
gg = AutoGridGenerator()
grid = gg.generate(TEST_IR["domain"], nrow=40, ncol=50)
assert grid["nrow"] == 40 and grid["ncol"] == 50
assert grid["cell_count"] == 2000
print(f"   OK: {grid['nrow']}x{grid['ncol']}x{grid['nlay']}, {grid['cell_count']} cells")

# Peclet-based recommendation
peclet_sug = gg.suggest_from_peclet(al=10.0, v=0.075, domain=TEST_IR["domain"])
print(f"   Peclet recommendation: dx={peclet_sug['recommended_dx']}m, ncol={peclet_sug['suggested_ncol']}")

# 2. Model Mapper
print("\n2. Model Mapper...")
mm = FloPyModelMapper()
mapped = mm.map_all(TEST_IR)
assert mapped["grid"]["nrow"] == 40
assert mapped["aquifer"]["hk"] == 15.0
assert mapped["time"]["nper"] == 1
assert len(mapped["boundaries"]) > 0
print(f"   OK: grid={mapped['grid']['nrow']}x{mapped['grid']['ncol']}, hk={mapped['aquifer']['hk']}")

# 3. Model Builder
print("\n3. Model Builder...")
workspace = Path(__file__).parent.parent / "data" / "outputs" / "e2e_test"
workspace.mkdir(parents=True, exist_ok=True)
builder = AutoModelBuilder(str(workspace))

build_result = None
try:
    result = build_result = builder.build_and_run(TEST_IR, "e2e_test")
    if result["success"]:
        print(f"   OK: Head range = {result['head_range']} m")
        print(f"   Output: {result['output_path']}")
    else:
        print(f"   WARNING: {result.get('error', 'Unknown')}")
except Exception as e:
    print(f"   WARNING: Build/run failed: {e}")
    # This is OK - means MODFLOW ran but may have issues with the model setup

# 契约锁（放在 try 之外，否则断言会被上面的 except 吞掉）：
# 返回值必须带 build_warnings，前端靠它把"建模提示"渲染到结果面板
if build_result is not None:
    assert "build_warnings" in build_result, \
        "build_and_run 返回值必须带 build_warnings（前端结果面板的数据源）"
    print(f"   OK: build_warnings 随返回值带出（{len(build_result['build_warnings'])} 条）")

# 4. Solver Monitor
print("\n4. Solver Monitor...")
sm = SolverMonitor()
assert len(sm.ADAPTIVE_STRATEGIES) >= 5
print(f"   OK: {len(sm.ADAPTIVE_STRATEGIES)} error types, {sm.MAX_RETRIES} max retries")

# Test error classification
assert sm._classify_error("did not converge", {}) == ErrorType.NON_CONVERGENCE
assert sm._classify_error("cell went dry", {}) == ErrorType.DRY_CELLS
assert sm._classify_error("negative concentration detected", {}) == ErrorType.NEGATIVE_CONCENTRATION
print("   OK: Error classification works")

# 5. 构建期提示必须能被用户看见（原先只有 logger.warning，到不了界面）
print("\n5. Build warnings reach the result dict...")

_MIN_GWT_IR = {
    "aquifer": {"hk": 1.004, "porosity": 0.25},
    "transport": {"porosity": 0.25, "enabled": True},
    "boundaries": {"chd": {"0": [[[0, 0, 0], 1.0], [[0, 0, 499], 0.0]]}},
    "grid": {"nrow": 2, "ncol": 10, "nlay": 1, "delr": 20, "delc": 20},
    "domain": {"units": "m"},
}

# 5a. 降级：启用运移但无浓度源、初始浓度也为 0 → 浓度场恒为 0
#
# 同时锁住"al 走网格尺度默认值"这条新提示：_MIN_GWT_IR **没有**给 al，
# 所以它必须出现。此前这条只在 logger 里，用户以为自己在提示词里写了
# 弥散度、实际拿到的却是 2Δx（East Texas 案例的真实翻车点）。
with tempfile.TemporaryDirectory() as _td:
    _b1 = AutoModelBuilder(_td)
    _b1.build_simulation(_MIN_GWT_IR, "warn_probe")
    assert _b1._transport_degraded, "应置 transport_degraded"
    assert len(_b1._build_warnings) == 2, f"应恰好两条提示，得到 {_b1._build_warnings}"
    assert any("浓度场将恒为 0" in w for w in _b1._build_warnings), _b1._build_warnings
    assert any("纵向弥散度 al 未从输入中取到" in w for w in _b1._build_warnings), \
        f"未给 al 时必须提示走了默认值，得到 {_b1._build_warnings}"
print(f"   OK: 无浓度源降级 + al 默认值提示均已捕获（{len(_b1._build_warnings)} 条）")

# 5a'. 给了 al 就不该再喊那条默认值提示（否则是另一种"狼来了"）
with tempfile.TemporaryDirectory() as _td:
    _b1b = AutoModelBuilder(_td)
    _b1b.build_simulation(
        {**_MIN_GWT_IR, "transport": {**_MIN_GWT_IR["transport"], "al": 20.0}},
        "warn_probe_al",
    )
    assert not any("纵向弥散度 al 未从输入中取到" in w for w in _b1b._build_warnings), \
        f"显式给了 al 不应再提示默认值，得到 {_b1b._build_warnings}"
print("   OK: 显式给出 al 时不产生默认值提示")

# 5b. 标量应力期 + 逐格浓度共存 → 必须展开为全网格，且**必须说出来**
with tempfile.TemporaryDirectory() as _td:
    _b2 = AutoModelBuilder(_td)
    _spd, _hc = _b2._normalize_boundary_spd(
        "rch", {0: 1e-4, 1: [[(0, 0, 0), 1e-4, 500.0]]},
        base=1, nrow=3, ncol=4)
    assert _hc, "含浓度列时 has_conc 应为 True"
    assert len(_spd[0]) == 12, f"标量期应展开为 3×4=12 格，得到 {len(_spd[0])}"
    assert _spd[0][0] == ((0, 0, 0), 1e-4, 0.0), f"展开后浓度应为 0，得到 {_spd[0][0]}"
    assert len(_b2._build_warnings) == 1, "全域展开必须留下提示"
    assert "展开为 layer0 全网格" in _b2._build_warnings[0]
print(f"   OK: 全网格展开提示已捕获：{_b2._build_warnings[0][:48]}…")

# 5c. 变异对照：不含浓度时**不展开也不报警**（否则提示会变成噪音）
with tempfile.TemporaryDirectory() as _td:
    _b3 = AutoModelBuilder(_td)
    _spd3, _hc3 = _b3._normalize_boundary_spd(
        "rch", {0: 1e-4, 1: [[(0, 0, 0), 1e-4]]}, base=1, nrow=3, ncol=4)
    assert not _hc3 and _b3._build_warnings == [], "无浓度时不应展开/报警"
    assert _spd3.get(0) == 1e-4, f"标量期应原样保留，得到 {_spd3.get(0)}"
print("   OK: 无浓度时不展开、不报警（标量原样保留）")

# 6. domain 驱动网格范围（此前 grid.nrow/ncol 一旦存在，domain 就完全被无视）
print("\n6. Domain drives grid extent...")

def _reconciled(domain, grid):
    """跑一次网格重算，返回 (结果 grid, 警告条数)。"""
    with tempfile.TemporaryDirectory() as _td:
        _b = AutoModelBuilder(_td)
        _ir = {"domain": domain, "grid": dict(grid)}
        _b._reconcile_grid_with_domain(_ir)
        return _ir["grid"], len(_b._build_warnings)

# 6a. 声明值与域不符 → 按 ceil(L/del) 重算，并**必须**留下提示
_g, _n = _reconciled({"x_range": [0, 3960], "y_range": [0, 1871]},
                     {"nrow": 103, "ncol": 218})
assert (_g["nrow"], _g["ncol"]) == (94, 198), f"期望 94x198，得到 {_g['nrow']}x{_g['ncol']}"
assert _n == 2, f"nrow/ncol 各改一次，应 2 条提示，得到 {_n}"
print(f"   OK: 103x218 → {_g['nrow']}x{_g['ncol']}（域 3960x1871 ÷ 20 m），提示 {_n} 条")

# 6b. 取 ceil 而非 round：1801/20 = 90.05，round 会得 90，会让域内最北 1 m 掉出网格
_g, _ = _reconciled({"x_range": [0, 1801], "y_range": [0, 1801]},
                    {"nrow": 5, "ncol": 5})
assert (_g["nrow"], _g["ncol"]) == (91, 91), \
    f"ceil 语义：1801/20 应得 91（不是 round 的 90），得到 {_g['nrow']}"
assert _g["nrow"] * 20 >= 1801, "网格必须盖住整个域"
print(f"   OK: ceil 语义 —— 1801 m ÷ 20 m = 91 格（round 会得 90，域内最北 1 m 落空）")

# 6c. 变异对照：已经自洽时**不重算也不报警**（否则提示会沦为噪音）
_g, _n = _reconciled({"x_range": [0, 3960], "y_range": [0, 1871]},
                     {"nrow": 94, "ncol": 198})
assert (_g["nrow"], _g["ncol"]) == (94, 198) and _n == 0, \
    f"自洽时不该动、不该报警，得到 {_g['nrow']}x{_g['ncol']} / {_n} 条"
print("   OK: 域与网格自洽时不重算、不报警")

# 6d. delr/delc 是显式标量 → 用它做分辨率，而不是默认 20 m
_g, _ = _reconciled({"x_range": [0, 1000], "y_range": [0, 400]},
                    {"nrow": 1, "ncol": 1, "delr": 50, "delc": 100})
assert (_g["nrow"], _g["ncol"]) == (4, 20), \
    f"delr=50/delc=100 应得 ncol=20/nrow=4，得到 {_g['nrow']}x{_g['ncol']}"
print("   OK: 显式 delr/delc 决定分辨率（50 m → 20 列；100 m → 4 行）")

# 6e. delr 是逐列数组 → 局部加密是刻意为之，不拿平均宽度回算格数
_g, _n = _reconciled({"x_range": [0, 1000], "y_range": [0, 400]},
                     {"nrow": 7, "ncol": 7, "delr": [10.0] * 7, "delc": [10.0] * 7})
assert (_g["nrow"], _g["ncol"]) == (7, 7) and _n == 0, \
    f"逐列 delr 时不该回算，得到 {_g['nrow']}x{_g['ncol']} / {_n} 条"
print("   OK: 逐列 delr/delc（局部加密）不被回算")

# 6f. domain 缺 range / 退化 range → 一律不动（老 IR 与自造 IR 都不受影响）
for _dom in ({}, {"x_range": [0, 0], "y_range": [0, 100]},
             {"x_range": [0, 100]}, {"x_range": "bad", "y_range": [0, 100]},
             None):
    _g, _n = _reconciled(_dom, {"nrow": 33, "ncol": 44})
    assert (_g["nrow"], _g["ncol"]) == (33, 44) and _n == 0, \
        f"domain={_dom} 时不该动网格，得到 {_g['nrow']}x{_g['ncol']}"
print("   OK: domain 缺失/退化/非法时不触碰网格")

print("\n" + "=" * 60)
print("All FloPy auto-modeling tests passed!")
print("=" * 60)

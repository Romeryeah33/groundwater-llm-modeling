"""
Span 6 诊断脚本：IR → FloPy 参数映射完整验证

测试 map_grid, map_aquifer_params, map_boundary_conditions,
map_solver_params, map_transport_params, _map_time_params, map_all
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
from app.modeling.model_mapper import FloPyModelMapper

# ============================================================
# 标准测试 IR（含所有典型参数）
# ============================================================
IR_STANDARD = {
    "ir_version": "1.0",
    "project": {"name": "try", "model_type": "groundwater_flow"},
    "domain": {"x_range": [0, 1000], "y_range": [0, 800], "units": "meters"},
    "grid": {
        "nrow": 40, "ncol": 50, "nlay": 1,
        "delr": 20.0, "delc": 20.0,
        "top": 100.0, "bottom": 50.0,
    },
    "aquifer": {
        "type": "unconfined",
        "parameters": {
            "hk": {"value": 15.0, "unit": "m/d"},
            "sy": {"value": 0.2, "unit": "dimensionless"},
            "porosity": {"value": 0.25, "unit": "dimensionless"},
        },
    },
    "boundary_conditions": [
        {
            "type": "chd", "name": "west_boundary",
            "parameters": {"head": 100.0},
        },
        {
            "type": "chd", "name": "east_boundary",
            "parameters": {"head": 95.0},
        },
        {
            "type": "wel", "name": "center_well",
            "parameters": {"rate": -200.0},
        },
    ],
    "stress_periods": {
        "nper": 1, "steady_state": True,
        "period_data": [{"perlen": 1.0, "nstp": 1, "tsmult": 1.0}],
    },
    "solver": {
        "name": "pcg",
        "parameters": {"hclose": 1e-5},
    },
    "output": {"head": True, "budget": True},
}

mm = FloPyModelMapper()
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

print("=" * 60)
print("Span 6 诊断: IR → FloPy 参数映射")
print("=" * 60)

# ============================================================
# S6.1: map_grid — 网格参数映射
# ============================================================
print("\n--- S6.1: map_grid ---")
g = mm.map_grid(IR_STANDARD)

check("nrow=40", g["nrow"] == 40)
check("ncol=50", g["ncol"] == 50)
check("nlay=1", g["nlay"] == 1)
check("top=100.0", g["top"] == 100.0)
check("botm=50.0", g["botm"] == 50.0)
check("delr is ndarray", isinstance(g["delr"], np.ndarray))
check("delr length=50", len(g["delr"]) == 50)
check("delr all=20.0", np.allclose(g["delr"], 20.0))
check("delc length=40", len(g["delc"]) == 40)
check("delc all=20.0", np.allclose(g["delc"], 20.0))

# S6.1b: _unwrap_value
v1 = mm._unwrap_value({"value": 15.0, "unit": "m/d"})
check("_unwrap_value extracts value", v1 == 15.0)
v2 = mm._unwrap_value(None, default=100)
check("_unwrap_value uses default", v2 == 100)
v3 = mm._unwrap_value(42)
check("_unwrap_value passes scalar", v3 == 42)

# S6.1c: delr=scalar expansion
ir_scalar = {"grid": {"ncol": 10, "nrow": 5, "delr": 25, "delc": 30}}
gs = mm.map_grid(ir_scalar)
check("scalar delr expanded to array", len(gs["delr"]) == 10 and np.allclose(gs["delr"], 25))
check("scalar delc expanded to array", len(gs["delc"]) == 5 and np.allclose(gs["delc"], 30))

# S6.1d: delr=list truncation
ir_list = {"grid": {"ncol": 5, "nrow": 3, "delr": [10, 20, 30, 40, 50, 60, 70], "delc": [5, 10, 15, 20]}}
gl = mm.map_grid(ir_list)
check("list delr truncated to ncol", len(gl["delr"]) == 5)
check("list delc truncated to nrow", len(gl["delc"]) == 3)

# S6.1e: no grid → defaults
ir_nogrid = {"domain": {"x_range": [0, 500], "y_range": [0, 300]}}
gn = mm.map_grid(ir_nogrid)
check("no grid → ncol defaults to 50", gn["ncol"] == 50)
check("no grid → nrow defaults to 50", gn["nrow"] == 50)
check("no grid → top defaults to 100", gn["top"] == 100.0)

# ============================================================
# S6.2: map_aquifer_params — 含水层参数映射
# ============================================================
print("\n--- S6.2: map_aquifer_params ---")
aq = mm.map_aquifer_params(IR_STANDARD)

check("aquifer_type=unconfined", aq["aquifer_type"] == "unconfined")
check("hk=15.0", aq["hk"] == 15.0)
check("sy=0.2", aq["sy"] == 0.2)
check("porosity=0.25", aq["porosity"] == 0.25)
check("vka not set (no anisotropy)", "vka" not in aq)
check("ss not set", "ss" not in aq)

# S6.2b: confined with anisotropy
ir_aniso = {
    "aquifer": {
        "type": "confined",
        "parameters": {
            "hk": {"value": 10.0},
            "hani": {"value": 5.0},  # Kh/Kv = 5
        },
    },
}
aq2 = mm.map_aquifer_params(ir_aniso)
check("hani → vka=hk*hani=2.0", abs(aq2.get("vka", 0) - 2.0) < 0.01)

# S6.2c: empty aquifer → defaults
ir_empty = {}
aq3 = mm.map_aquifer_params(ir_empty)
check("empty → confined default", aq3["aquifer_type"] == "confined")
check("empty → no hk", "hk" not in aq3)

# ============================================================
# S6.3: _infer_edge_indices — 边界位置推断
# ============================================================
print("\n--- S6.3: _infer_edge_indices ---")
grid_info = {"nrow": 40, "ncol": 50}

indices_west = mm._infer_edge_indices("west_boundary", grid_info)
check("west → 40 indices (all rows col=0)", len(indices_west) == 40)
check("west[0] col=0", indices_west[0]["col"] == 0)
check("west[-1] row=39", indices_west[-1]["row"] == 39)

indices_east = mm._infer_edge_indices("east_boundary", grid_info)
check("east → 40 indices", len(indices_east) == 40)
check("east[0] col=49", indices_east[0]["col"] == 49)

indices_north = mm._infer_edge_indices("north_boundary", grid_info)
check("north → 50 indices (all cols row=0)", len(indices_north) == 50)
check("north[0] row=0", indices_north[0]["row"] == 0)

indices_south = mm._infer_edge_indices("south_boundary", grid_info)
check("south → 50 indices", len(indices_south) == 50)
check("south[0] row=39", indices_south[0]["row"] == 39)

# 中文关键词
indices_xi = mm._infer_edge_indices("西侧边界", grid_info)
check("西 → west inference", len(indices_xi) == 40 and indices_xi[0]["col"] == 0)

indices_dong = mm._infer_edge_indices("东边", grid_info)
check("东 → east inference", len(indices_dong) == 40 and indices_dong[0]["col"] == 49)

indices_nan = mm._infer_edge_indices("南边界", grid_info)
check("南 → south inference", len(indices_nan) == 50 and indices_nan[0]["row"] == 39)

# 无法推断的边界名
indices_unknown = mm._infer_edge_indices("northeast_corner", grid_info)
check("unknown name → empty list", len(indices_unknown) == 0)

# ============================================================
# S6.4: map_boundary_conditions — 边界条件映射
# ============================================================
print("\n--- S6.4: map_boundary_conditions ---")
bcs = mm.map_boundary_conditions(IR_STANDARD)

check("chd package exists", "chd" in bcs)
check("wel package exists", "wel" in bcs)

chd_data = bcs["chd"]
check("chd has period 0", 0 in chd_data)
check("chd has 80 records (40W+40E)", len(chd_data[0]) == 80)

# 检查西侧 CHD (前40条)
chd_record_west = chd_data[0][0]
check("chd west record IJ=(0,0,0)", chd_record_west[0] == (0, 0, 0))
check("chd west head=100", chd_record_west[1] == 100.0)

# 检查东侧 CHD (后40条)
chd_record_east = chd_data[0][40]
check("chd east record col=49", chd_record_east[0] == (0, 0, 49))
check("chd east head=95", chd_record_east[1] == 95.0)

# 检查井
wel_data = bcs["wel"]
check("wel has period 0", 0 in wel_data)
check("wel has 1 record", len(wel_data[0]) == 1)
check("wel at center (20,25)", wel_data[0][0][0] == (0, 20, 25))
check("wel rate=-200", wel_data[0][0][1] == -200.0)

# S6.4b: 带显式 grid_indices 的边界
ir_explicit = {
    "boundary_conditions": [{
        "type": "chd",
        "name": "point_chd",
        "geometry": {
            "grid_indices": [
                {"layer": 0, "row": 10, "col": 5, "period": 0},
            ],
        },
        "parameters": {"head": 50.0},
    }],
    "stress_periods": {"nper": 1},
    "grid": {"nrow": 40, "ncol": 50, "nlay": 1},
}
bcs2 = mm.map_boundary_conditions(ir_explicit)
check("explicit indices → 1 chd record", len(bcs2["chd"][0]) == 1)
check("explicit chd at (10,5)", bcs2["chd"][0][0][0] == (0, 10, 5))

# S6.4c: GHB 边界
ir_ghb = {
    "boundary_conditions": [{
        "type": "ghb",
        "name": "river_leakage",
        "geometry": {
            "grid_indices": [
                {"layer": 0, "row": 15, "col": 0, "period": 0},
            ],
        },
        "parameters": {"head": 98.0, "conductance": 500.0},
    }],
    "stress_periods": {"nper": 1},
    "grid": {"nrow": 40, "ncol": 50, "nlay": 1},
}
bcs3 = mm.map_boundary_conditions(ir_ghb)
check("ghb package exists", "ghb" in bcs3)
check("ghb record has 3 values", len(bcs3["ghb"][0][0]) == 3)  # (cellid, head, cond)

# ============================================================
# S6.5: map_solver_params — 求解器映射
# ============================================================
print("\n--- S6.5: map_solver_params ---")
sp = mm.map_solver_params(IR_STANDARD)
check("pcg → CG", sp["name"] == "CG")
check("hclose=1e-5", sp["hclose"] == 1e-5)
check("rclose not set (no default)", "rclose" not in sp)

# S6.5b: nwt → BICGSTAB
ir_nwt = {"solver": {"name": "nwt", "parameters": {}}}
sp2 = mm.map_solver_params(ir_nwt)
check("nwt → BICGSTAB", sp2["name"] == "BICGSTAB")

# S6.5c: no solver → defaults
ir_nosolv = {}
sp3 = mm.map_solver_params(ir_nosolv)
check("no solver → CG", sp3["name"] == "CG")
check("no solver → hclose=1e-4", sp3["hclose"] == 1e-4)

# ============================================================
# S6.6: _map_time_params — 时间参数映射
# ============================================================
print("\n--- S6.6: _map_time_params ---")
tp = mm._map_time_params(IR_STANDARD)
check("nper=1", tp["nper"] == 1)
check("steady_state=True", tp["steady_state"] == True)
check("period_data nstp=1", tp["period_data"][0]["nstp"] == 1)

# 瞬态
ir_transient = {
    "stress_periods": {
        "nper": 3,
        "steady_state": False,
        "period_data": [
            {"perlen": 365, "nstp": 10, "tsmult": 1.2},
            {"perlen": 365, "nstp": 10, "tsmult": 1.2},
            {"perlen": 365, "nstp": 10, "tsmult": 1.0},
        ],
    },
}
tp2 = mm._map_time_params(ir_transient)
check("transient nper=3", tp2["nper"] == 3)
check("transient steady_state=False", tp2["steady_state"] == False)

# ============================================================
# S6.7: map_transport_params — 溶质运移参数
# ============================================================
print("\n--- S6.7: map_transport_params ---")
transport = mm.map_transport_params(IR_STANDARD)
check("no transport → disabled", transport["enabled"] == False)

ir_transport = {
    "transport": {
        "enabled": True,
        "species": ["tce"],
        "dispersion": {
            "al": {"value": 20.0},
            "trpt": {"value": 0.2},
        },
        "advection": {"scheme": "upstream"},
        "reaction": {
            "decay": {"type": "first_order", "first_order_rate": {"value": 0.01}},
        },
    },
}
tran = mm.map_transport_params(ir_transport)
check("transport enabled", tran["enabled"] == True)
check("al=20.0", tran["al"] == 20.0)
check("trpt=0.2", tran["trpt"] == 0.2)
check("trpv default=0.01", tran["trpv"] == 0.01)
check("decay_rate=0.01", tran["decay_rate"] == 0.01)

# ============================================================
# S6.8: map_all — 完整映射
# ============================================================
print("\n--- S6.8: map_all ---")
all_mapped = mm.map_all(IR_STANDARD)
check("has grid", "grid" in all_mapped)
check("has aquifer", "aquifer" in all_mapped)
check("has boundaries", "boundaries" in all_mapped)
check("has solver", "solver" in all_mapped)
check("has time", "time" in all_mapped)
check("has transport", "transport" in all_mapped)
check("map_all grid nrow=40", all_mapped["grid"]["nrow"] == 40)
check("map_all aquifer hk=15", all_mapped["aquifer"]["hk"] == 15.0)

# ============================================================
# 结果汇总
# ============================================================
print("\n" + "=" * 60)
total = passed + failed
print(f"Span 6 映射测试结果: {passed}/{total} 通过")
if failed == 0:
    print("✅ 所有测试通过！IR→FloPy 映射层工作正常。")
else:
    print(f"⚠️ {failed} 项测试失败，需排查。")
print("=" * 60)

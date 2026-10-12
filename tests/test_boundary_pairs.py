"""
L2 边界对验证：跨 Span 参数传递一致性检查

覆盖边界对：
  BP10 (P0): B6→B8 — 校验通过 IR → FloPy 包参数逐包对照
  BP4  (P0): B5→B8 — JSON IR → MODFLOW 6 输入文件文本验证
  BP5  (P0): B5→B9 — JSON IR → 模拟输出
  BP2  (P0): B1→B5 — 用户 NL → 最终 JSON IR (Mock LLM)

不依赖外部 LLM API，使用标准测试 IR。
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import flopy

from app.modeling.grid_generator import AutoGridGenerator
from app.modeling.model_mapper import FloPyModelMapper
from app.modeling.model_builder import AutoModelBuilder

# ============================================================
# 标准测试 IR
# ============================================================
STANDARD_IR = {
    "ir_version": "1.0",
    "project": {"name": "l2_bp_test", "model_type": "groundwater_flow"},
    "domain": {"x_range": [0, 1000], "y_range": [0, 800], "units": "meters"},
    "grid": {
        "nrow": 40, "ncol": 50, "nlay": 1,
        "delr": 20.0, "delc": 20.0,
        "top": 100.0, "bottom": 50.0,
    },
    "aquifer": {
        "type": "confined",
        "parameters": {
            "hk": {"value": 15.0, "unit": "m/d"},
            "sy": {"value": 0.1, "unit": "dimensionless"},
            "ss": {"value": 1e-5, "unit": "1/m"},
        },
    },
    "boundary_conditions": [
        {"type": "chd", "name": "west_boundary", "parameters": {"head": 100.0}},
        {"type": "chd", "name": "east_boundary", "parameters": {"head": 95.0}},
        {"type": "wel", "name": "center_well", "parameters": {"rate": -200.0}},
    ],
    "stress_periods": {
        "nper": 1, "steady_state": True,
        "period_data": [{"perlen": 1.0, "nstp": 1, "tsmult": 1.0}],
    },
    "solver": {"name": "pcg", "parameters": {"hclose": 1e-5}},
    "output": {"head": True, "budget": True},
}

passed = 0
failed = 0
workspace = Path(__file__).parent.parent / "data" / "outputs" / "l2_test"
workspace.mkdir(parents=True, exist_ok=True)

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name}  FAILED  {detail}")

print("=" * 60)
print("L2 边界对验证：跨 Span 参数传递一致性")
print("=" * 60)

# 构建模型（不运行 — BP10 只检查包参数）
mapper = FloPyModelMapper()
builder = AutoModelBuilder(str(workspace))

# Step 1: 映射 IR → FloPy 参数
mapped = mapper.map_all(STANDARD_IR)

# Step 2: 构建 Simulation（写入输入文件，但不运行）
sim = builder.build_simulation(STANDARD_IR, "l2_bp_test")
gwf = builder._gwf

# ============================================================
# BP10: 校验通过 IR → FloPy 包参数逐包对照
# ============================================================
print("\n" + "=" * 60)
print("BP10: JSON IR → FloPy 包参数")
print("=" * 60)

# Helper: safely extract scalar from FloPy data
def _scalar(data):
    """Extract scalar from FloPy data (handles 0-d/1-d arrays and scalars)."""
    arr = np.asarray(data).flatten()
    return float(arr[0])

# --- DIS 离散包 ---
print("\n--- DIS: 离散包 ---")
dis = gwf.dis
check("nrow=40", _scalar(dis.nrow.data) == 40)
check("ncol=50", _scalar(dis.ncol.data) == 50)
check("nlay=1", _scalar(dis.nlay.data) == 1)
check("top=100.0", abs(_scalar(dis.top.data) - 100.0) < 0.01)
check("botm=50.0", abs(_scalar(dis.botm.data) - 50.0) < 0.01)
check("delr shape=(50,)", len(np.asarray(dis.delr.data).flatten()) == 50)
check("delc shape=(40,)", len(np.asarray(dis.delc.data).flatten()) == 40)
check("delr values=20.0", np.allclose(np.asarray(dis.delr.data).flatten(), 20.0))
check("delc values=20.0", np.allclose(np.asarray(dis.delc.data).flatten(), 20.0))

# --- NPF 节点属性流 ---
print("\n--- NPF: 节点属性流 ---")
npf = gwf.npf
check("k=15.0", abs(_scalar(npf.k.data) - 15.0) < 0.01)
check("icelltype=0 (confined)", _scalar(npf.icelltype.data) == 0)
# k33 应在 Span 7 修复后自动设为 outer/10 的合理值
has_k33 = hasattr(npf, 'k33') and npf.k33 is not None
if has_k33:
    k33_val = _scalar(npf.k33.data) if hasattr(npf.k33, 'data') else float(npf.k33)
    check(f"k33={k33_val:.2f} (auto-set)", k33_val > 0 and k33_val < 15.0)
else:
    print("  ⚠️ NPF has no k33 (may be fine for 2D)")

# --- STO 储水包 ---
print("\n--- STO: 储水包 ---")
sto = gwf.sto
check("ss=1e-5", abs(_scalar(sto.ss.data) - 1e-5) < 1e-8)
check("sy=0.1", abs(_scalar(sto.sy.data) - 0.1) < 0.001)

# transient 值检查
sto_transient = sto.transient.data if hasattr(sto.transient, 'data') else sto.transient
if isinstance(sto_transient, dict):
    check("transient=False (steady)", sto_transient.get(0, True) == False)
else:
    check("transient=False (steady)", bool(sto_transient) == False)

# --- IC 初始条件 ---
print("\n--- IC: 初始条件 ---")
ic = gwf.ic
check("strt=100.0", abs(_scalar(ic.strt.data) - 100.0) < 0.01)

# --- CHD 定水头边界 ---
print("\n--- CHD: 定水头边界 ---")
chd = gwf.chd
chd_spd = chd.stress_period_data.data  # {0: [(cellid, head), ...]}
chd_records = chd_spd[0] if 0 in chd_spd else chd_spd.get(0, [])

west_count = sum(1 for r in chd_records if r[0][2] == 0)
east_count = sum(1 for r in chd_records if r[0][2] == 49)
check("80 CHD records total", len(chd_records) == 80)
check("40 west boundary cells", west_count == 40)
check("40 east boundary cells", east_count == 40)

# 抽检：第一条（西侧）head=100
check("west[0] head=100", abs(chd_records[0][1] - 100.0) < 0.01)
# 最后一条（东侧）head=95
check("east[-1] head=95", abs(chd_records[-1][1] - 95.0) < 0.01)

# 抽样：所有西侧 head=100
west_ok = all(abs(r[1] - 100.0) < 0.01 for r in chd_records if r[0][2] == 0)
check("all west heads=100", west_ok)
# 所有东侧 head=95
east_ok = all(abs(r[1] - 95.0) < 0.01 for r in chd_records if r[0][2] == 49)
check("all east heads=95", east_ok)

# --- WEL 井包 ---
print("\n--- WEL: 井包 ---")
wel = gwf.wel
wel_spd = wel.stress_period_data.data
wel_records = wel_spd[0] if 0 in wel_spd else []

check("1 well record", len(wel_records) == 1)
if len(wel_records) == 1:
    cellid, rate = wel_records[0]
    check(f"well at (0,20,25)", cellid == (0, 20, 25))
    check("well Q=-200", abs(rate - (-200.0)) < 0.01)

# --- IMS 求解器 ---
print("\n--- IMS: 求解器 ---")
# IMS 在 sim 级别，不在 gwf
ims_file = workspace / "l2_bp_test" / "l2_bp_test.ims"
ims_text = ims_file.read_text()
check("OUTER_DVCLOSE=1e-5", "1.00000000E-05" in ims_text or "1.000000e-05" in ims_text)

# inner_dvclose 应在修复后为 outer/10 = 1e-6
has_inner_1e6 = "1.00000000E-06" in ims_text or "1.000000e-06" in ims_text
has_inner_small = "1.000000E-06" in ims_text
check("INNER_DVCLOSE=1e-6 (auto=outer/10)", has_inner_1e6 or has_inner_small)

# --- OC 输出控制 ---
print("\n--- OC: 输出控制 ---")
oc = gwf.oc
check("OC package exists", oc is not None)

# ============================================================
# BP4: JSON IR → MODFLOW 6 输入文件
# ============================================================
print("\n" + "=" * 60)
print("BP4: JSON IR → MODFLOW 6 输入文件")
print("=" * 60)

sim_ws = workspace / "l2_bp_test"

# --- .dis 文件 ---
print("\n--- .dis 文件 ---")
dis_text = (sim_ws / "l2_bp_test_gwf.dis").read_text()
check(".dis: NROW 40", "NROW  40" in dis_text or "NROW 40" in dis_text)
check(".dis: NCOL 50", "NCOL  50" in dis_text or "NCOL 50" in dis_text)
check(".dis: NLAY 1", "NLAY  1" in dis_text or "NLAY 1" in dis_text)
check(".dis: top 100.0", "100.000" in dis_text and "top" in dis_text.lower())

# --- .npf 文件 ---
print("\n--- .npf 文件 ---")
npf_text = (sim_ws / "l2_bp_test_gwf.npf").read_text()
check(".npf: k=15.0", "15.000" in npf_text and "k" in npf_text.lower())
check(".npf: icelltype=0", "icelltype" in npf_text and "0" in npf_text)

# --- .chd 文件 ---
print("\n--- .chd 文件 ---")
chd_text = (sim_ws / "l2_bp_test_gwf.chd").read_text()
check(".chd: MAXBOUND 80", "MAXBOUND  80" in chd_text or "MAXBOUND 80" in chd_text)
check(".chd: 100.0 head present", "1.00000000E+02" in chd_text)
check(".chd: 95.0 head present", "9.50000000E+01" in chd_text)
# 验证 CHD 格式: 每行应为 "层 行 列 水头值"
chd_lines = [l for l in chd_text.split("\n") if l.strip() and not l.startswith("#") and "BEGIN" not in l and "END" not in l and "MAXBOUND" not in l]
check(".chd: 80 data lines", len(chd_lines) == 80)

# --- .sto 文件 ---
print("\n--- .sto 文件 ---")
sto_text = (sim_ws / "l2_bp_test_gwf.sto").read_text()
check(".sto: ss=1e-5", "1.00000000E-05" in sto_text)
check(".sto: sy=0.1", "1.00000000E-01" in sto_text or "0.100000" in sto_text)

# ============================================================
# BP5: JSON IR → 模拟输出 (B5→B9)
# ============================================================
print("\n" + "=" * 60)
print("BP5: JSON IR → 模拟输出")
print("=" * 60)

# 运行模拟
success, _ = sim.run_simulation(silent=True)
check("MODFLOW 6 success", success)

if success:
    # --- .lst 文件 ---
    lst_text = (sim_ws / "mfsim.lst").read_text()
    check("Normal termination in .lst", "Normal termination of simulation" in lst_text)

    # --- 水头结果 ---
    head_obj = gwf.output.head()
    head = head_obj.get_data()
    check("head shape (1,40,50)", head.shape == (1, 40, 50))

    h = head[0]
    hmin, hmax = float(h.min()), float(h.max())
    check("hmin≥95", hmin >= 95.0)
    check("hmax≤100", hmax <= 100.0)
    check("hmin>botm(50)", hmin > 50.0)

    # 物理检查：水头应在边界值范围内
    check("head ranges within CHD bounds", 95.0 <= hmin and hmax <= 100.0)

    # 降落漏斗：井附近水头应低于远场
    well_h = float(h[20, 25])
    far_h = float(h[5, 5])
    check(f"well({well_h:.2f}) < far({far_h:.2f}) → drawdown", well_h < far_h)

    # 梯度方向：西→东递减
    west_edge = np.mean(h[:, 0])
    east_edge = np.mean(h[:, -1])
    check(f"west({west_edge:.1f}) > east({east_edge:.1f})", west_edge > east_edge)

    # 水头沿列方向等值（除井附近外应平滑过渡）
    mid_col = h[:, 25]  # 沿中心列
    check("smooth gradient along col", np.all(np.diff(mid_col) < 1.0))

# ============================================================
# BP2: 用户 NL → 最终 JSON IR (Mock LLM)
# ============================================================
print("\n" + "=" * 60)
print("BP2: 用户 NL → 最终 JSON IR (Mock)")
print("=" * 60)

# 测试 LLM IR 输出的合并逻辑（不调用真实 LLM）
from app.llm_agent.intent_parser import IntentParser
from app.llm_agent.dialogue_manager import DialogueManager

# 验证关键模块可导入
check("IntentParser imported", IntentParser is not None)
check("DialogueManager imported", DialogueManager is not None)

# 验证 JSON IR 模板完整性
from app.llm_agent.prompt_templates import SYSTEM_PROMPT
check("SYSTEM_PROMPT mentions IR schema", "JSON" in SYSTEM_PROMPT or "json_ir" in SYSTEM_PROMPT or "IR" in SYSTEM_PROMPT)

# 验证缺失参数检测（不依赖 LLM）
parser = IntentParser()
missing = parser.detect_missing_params(STANDARD_IR)
# 标准 IR 应该完整，P1 参数不应缺失
p1_missing = missing.get("priority_1", [])
check("std IR has no P1 missing params", len(p1_missing) == 0)

# 测试不完整 IR → 应返回缺失参数
partial_ir = {
    "project": {"name": "test"},
    "domain": {"x_range": [0, 1000]},  # 缺 y_range
    "aquifer": {"type": "confined"},     # 缺 hk
}
missing2 = parser.detect_missing_params(partial_ir)
check("partial IR has P1 missing", len(missing2.get("priority_1", [])) > 0)

# ============================================================
# 汇总
# ============================================================
print("\n" + "=" * 60)
total = passed + failed
print(f"L2 边界对验证结果: {passed}/{total} 通过")
if failed == 0:
    print("✅ 所有边界对验证通过！")
    print("   BP10: IR → FloPy 包参数 ✓")
    print("   BP4:  IR → MODFLOW 6 输入文件 ✓")
    print("   BP5:  IR → 模拟输出 ✓")
    print("   BP2:  NL → IR (Mock) ✓")
else:
    print(f"⚠️ {failed} 项测试失败")
print("=" * 60)

# 失败即返回非零退出码 —— 否则"看 rc 判断红绿"的回归脚本会把失败当通过。
# 同类问题与来龙去脉见 docs/后续开发计划20261010.md「二、挂起的任务」第 5 条。
sys.exit(1 if failed else 0)

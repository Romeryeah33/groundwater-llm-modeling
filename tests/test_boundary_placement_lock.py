"""
边界定位回归锁 —— 防止"边界静默失效"再次发生

缺陷（2026-10-13 定位，用户选择"只警告、照旧放行"的处置）：

  需要**整条边界**的包（CHD/GHB/RIV/DRN）落到哪一排，是从边界 `name` 里的方位词
  猜出来的（west/east/north/south 及 西/东/南/北/左/右/上/下/top/bottom…）。
  IR 里正经的 `position` 字段**不参与定位**。由此有**三个面**：

    ① 静默丢弃  name 里没有方位词（`边界A` / `inflow` / `river`）时推断出 0 格，
                于是整个包**根本不生成** —— 不是"小一点的边界"，是没有边界。
                而 `build_and_run` 照样返回 `success=True`。用户拿到一个没有定水头
                的模型：要么退化成平凡解（head=初始值，秒收敛），要么根本不收敛。

    ② position 被无视  `{"name": "上游定水头", "position": "west"}` 里白纸黑字的
                `west` 完全不算数，边界落到名字误命中的**北**边界上。

    ③ 流向词误命中  `上游定水头` 含 "上"（∈ NORTH_KEYWORDS）→ 写成**整条北边界**；
                `下游边界` → 南边界。而"上/下游"是**顺水流方向**，与南北毫无关系。

  三个面都不会报错、不会告警、不会体现在 `success` 上。这正是本锁存在的理由：
  **只断言"跑成功"是锁不住这类缺陷的**（参考稳态那次，同一个教训）。

本锁查的是**生成物**（`.chd` 里的 row/col 到底是什么）与**面向用户的告警**
（`unplaced_boundaries` / `apply_pregates["boundary_warnings"]`），两者之一变了就红。

⚠ 本文件**必须能变红**。已知的三种改法都应让它失败：
    A. 把 `infer_edge_indices` 里 `position` 那一路去掉（还原缺陷②）
    B. 把 `strip_flow_direction_words` 调用去掉（还原缺陷③）
    C. 让 `unplaced_boundaries` 永远返回空列表（还原缺陷①的"无人知晓"）
  跑过变异自检再相信它。

不依赖 pytest，沿用仓库脚本式风格（`check(name, condition, detail)`）。
运行：`python tests/test_boundary_placement_lock.py`
"""
import io
import re
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from app.modeling.model_mapper import FloPyModelMapper          # noqa: E402
from app.modeling.run_gates import apply_pregates               # noqa: E402
from app.modeling.source_zone import (                          # noqa: E402
    infer_edge_indices,
    strip_flow_direction_words,
    unplaced_boundaries,
)

passed = 0
failed = 0
skipped = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name}" + (f"  —— {detail}" if detail else ""))


def skip(name, why):
    global skipped
    skipped += 1
    print(f"  ⏭ {name}  —— {why}")


GRID = {"nlay": 1, "nrow": 3, "ncol": 3, "delr": 10.0, "delc": 10.0}


def where(cells):
    """把推断出的单元列表压成一个可读的落点描述。"""
    if not cells:
        return "（无）"
    rows = sorted({c["row"] for c in cells})
    cols = sorted({c["col"] for c in cells})
    if len(cols) == 1 and len(rows) == 3:
        return f"col{cols[0]}"
    if len(rows) == 1 and len(cols) == 3:
        return f"row{rows[0]}"
    return f"rows{rows}cols{cols}"


# ======================================================================
print("\n[一] 方位词照旧工作（不能因为修缺陷把正常的弄坏）")

for name, want in [("west", "col0"), ("east", "col2"),
                   ("north", "row2"), ("south", "row0"),
                   ("west_boundary", "col0"), ("西边界", "col0"),
                   ("北边界", "row2"), ("南侧边界", "row0"), ("东", "col2")]:
    got = where(infer_edge_indices(name, GRID, bc_type="chd"))
    check(f"name={name!r} -> {want}", got.replace("col", "col").replace("row", "row") == want,
          f"got {got}")

# 行号方向：row0 在南、row nrow-1 在北（与绘图"上北下南"同向）
check("north 落在最北一排(row=nrow-1)，不是 row0",
      {i["row"] for i in infer_edge_indices("north", GRID, bc_type="chd")} == {2})
check("south 落在最南一排(row=0)",
      {i["row"] for i in infer_edge_indices("south", GRID, bc_type="chd")} == {0})

# ======================================================================
print("\n[二] position 优先于名字（缺陷②）")

check("name'上游定水头'+position'west' -> 西边界（不是名字误命中的北）",
      where(infer_edge_indices("上游定水头", GRID, bc_type="chd",
                               position="west")) == "col0",
      where(infer_edge_indices("上游定水头", GRID, bc_type="chd", position="west")))
check("name'边界A'+position'west' -> 西边界（名字本无方位，position 救回）",
      where(infer_edge_indices("边界A", GRID, bc_type="chd",
                               position="west")) == "col0")
check("position 支持中文'西'",
      where(infer_edge_indices("边界A", GRID, bc_type="chd",
                               position="西")) == "col0")
check("position 支持 'north'", where(
    infer_edge_indices("边界A", GRID, bc_type="chd", position="north")) == "row2")
check("position 与名字一致时结果不变（East Texas 的形态）",
      where(infer_edge_indices("south_boundary", GRID, bc_type="ghb",
                               position="south")) == "row0")
check("position 认不出（'northeast'）-> 不给落点，而不是瞎猜一个",
      infer_edge_indices("边界A", GRID, bc_type="chd", position="northeast") == [])

# 真实 mapper 也要把 position 传下去 —— 光改 source_zone 而 mapper 不传是无效的
_mapper = FloPyModelMapper()
_spd = _mapper.map_boundary_conditions({
    "grid": GRID, "stress_periods": {"nper": 1},
    "boundary_conditions": [{"type": "chd", "name": "上游定水头",
                             "position": "west", "parameters": {"head": 8.0}}],
}).get("chd", {})
_recs = list(_spd.values())[0] if _spd else []
check("mapper 端到端：position=west 真的写进 col=0",
      bool(_recs) and {r[0][2] for r in _recs} == {0},
      f"got {[r[0] for r in _recs]}")
check("mapper 端到端：**没有**写到北边界 row=2",
      bool(_recs) and {r[0][1] for r in _recs} != {2},
      f"got {[r[0] for r in _recs]}")

# geometry.position 同样要认（intent_parser 两个都写）
_spd2 = _mapper.map_boundary_conditions({
    "grid": GRID, "stress_periods": {"nper": 1},
    "boundary_conditions": [{"type": "chd", "name": "边界A",
                             "geometry": {"type": "edge", "position": "west"},
                             "parameters": {"head": 8.0}}],
}).get("chd", {})
check("geometry.position 也认", bool(_spd2), f"got {_spd2}")

# ======================================================================
print("\n[三] 流向词不再被当成方位词（缺陷③）")

check("strip_flow_direction_words('上游定水头') == '定水头'",
      strip_flow_direction_words("上游定水头") == "定水头",
      strip_flow_direction_words("上游定水头"))
check("strip_flow_direction_words('左岸') == ''",
      strip_flow_direction_words("左岸") == "")

for name in ("上游定水头", "下游边界", "左岸", "右岸", "upstream_bc"):
    got = infer_edge_indices(name, GRID, bc_type="chd")
    check(f"name={name!r} 不再误命中南北东西 -> 无落点", got == [], f"got {where(got)}")

# 剔除流向词后**还有**真方位词的，仍要正常推断（不能一刀切放弃）
check("'上游西边界' 剔除流向词后仍推断出西边界",
      where(infer_edge_indices("上游西边界", GRID, bc_type="chd")) == "col0",
      where(infer_edge_indices("上游西边界", GRID, bc_type="chd")))

# rch/evt/wel 一律不参与方位推断（既有约定，不能被本次改动破坏）
for t in ("rch", "evt", "wel"):
    check(f"{t} 仍不做方位推断",
          infer_edge_indices("north_boundary", GRID, bc_type=t) == [])

# ======================================================================
print("\n[四] unplaced_boundaries —— 面向用户的告警来源（缺陷①）")


def _bc(bc_type, name, position=None, grid_indices=None):
    bc = {"type": bc_type, "name": name, "parameters": {}}
    if position:
        bc["position"] = position
    if grid_indices:
        bc["geometry"] = {"grid_indices": grid_indices}
    return {"grid": GRID, "boundary_conditions": [bc]}


check("chd 无方位词、无 position -> 报警告",
      len(unplaced_boundaries(_bc("chd", "边界A"))) == 1)
check("chd 有 position -> 不报",
      unplaced_boundaries(_bc("chd", "边界A", position="west")) == [])
check("chd 名字含方位词 -> 不报",
      unplaced_boundaries(_bc("chd", "west_boundary")) == [])
check("chd 含流向词 -> 报警告，且理由说明'流向词不是方位'",
      (lambda u: len(u) == 1 and "流向词" in u[0]["reason"])(
          unplaced_boundaries(_bc("chd", "上游定水头"))),
      str(unplaced_boundaries(_bc("chd", "上游定水头"))))
check("chd 有显式 grid_indices -> 不报（最可靠的定位方式）",
      unplaced_boundaries(_bc("chd", "边界A", grid_indices=[
          {"layer": 0, "row": 0, "col": 0, "period": 0}])) == [])
check("position 认不出时理由提示改成合法方位值",
      (lambda u: len(u) == 1 and "west/east" in u[0]["reason"])(
          unplaced_boundaries(_bc("ghb", "边界A", position="northeast"))),
      str(unplaced_boundaries(_bc("ghb", "边界A", position="northeast"))))

for t in ("ghb", "riv", "drn"):
    check(f"{t} 无位置 -> 报警告（同属'必须有整条边界'）",
          len(unplaced_boundaries(_bc(t, "边界A"))) == 1)

for t in ("wel", "rch", "evt"):
    check(f"{t} 无位置 -> **不**报警告（wel 有中心兜底，rch/evt 归源区闸）",
          unplaced_boundaries(_bc(t, "边界A")) == [])

check("多个未定位边界全部列出", len(unplaced_boundaries({
    "grid": GRID,
    "boundary_conditions": [{"type": "chd", "name": "甲", "parameters": {}},
                            {"type": "ghb", "name": "乙", "parameters": {}},
                            {"type": "chd", "name": "west", "parameters": {}}],
})) == 2)

# ======================================================================
print("\n[五] apply_pregates 把告警交给界面/接口（三处调用方共用的那一份）")


def make_ir(bcs, chd_name_is_west=True):
    return {
        "ir_version": "1.0",
        "project": {"name": "bnd_lock", "model_type": "groundwater_flow"},
        "domain": {"x_range": [0.0, 30.0], "y_range": [0.0, 30.0], "units": "meters"},
        "grid": {"nlay": 1, "nrow": 3, "ncol": 3,
                 "delr": 10.0, "delc": 10.0, "top": 10.0, "bottom": 0.0},
        "aquifer": {"type": "confined",
                    "parameters": {"hk": {"value": 1.0, "unit": "m/d"},
                                   "ss": {"value": 1e-5, "unit": "1/m"}}},
        "stress_periods": {"nper": 1, "steady_state": True,
                           "period_data": [{"perlen": 1.0, "nstp": 1, "tsmult": 1.0}]},
        "boundary_conditions": bcs,
        "solver": {"name": "pcg", "parameters": {"hclose": 1e-6, "rclose": 1e-6}},
    }


_g_ok = apply_pregates(make_ir([{"type": "chd", "name": "west",
                                 "parameters": {"head": 8.0}}]))
check("正常边界：boundary_warnings 为空",
      _g_ok.get("boundary_warnings") == [], str(_g_ok.get("boundary_warnings")))
check("apply_pregates 的返回值里一定有这个键（界面无条件读它）",
      "boundary_warnings" in _g_ok)

_g_bad = apply_pregates(make_ir([{"type": "chd", "name": "边界A",
                                  "parameters": {"head": 8.0}}]))
check("未定位边界：告警出现在 boundary_warnings 里",
      len(_g_bad.get("boundary_warnings") or []) == 1,
      str(_g_bad.get("boundary_warnings")))
check("告警文案里有边界名、有'不会写入模型'",
      (lambda w: w and "边界A" in w[0] and "不会写入模型" in w[0])(
          _g_bad.get("boundary_warnings")),
      str(_g_bad.get("boundary_warnings")))
check("未定位边界**不阻断**建模（用户决定：只警告、照旧放行）",
      _g_bad.get("stage") != "source_zone",
      f"stage={_g_bad.get('stage')}")

# ======================================================================
print("\n[六] 端到端：生成物里到底是什么（不是只看 success）")

try:
    from app.modeling.model_builder import AutoModelBuilder

    def run_and_read_chd(bcs):
        ws = Path(tempfile.mkdtemp(prefix="bnd_lock_"))
        try:
            out = AutoModelBuilder(workspace=str(ws)).build_and_run(
                make_ir(bcs), "lock")
            chds = sorted(ws.rglob("*.chd"))
            text = chds[0].read_text(encoding="utf-8", errors="replace") if chds else ""
            rows = re.findall(r"^\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.eE+-]+)\s*$",
                              text, re.M)
            return out, [(int(r), int(c)) for _l, r, c, _v in rows]
        finally:
            shutil.rmtree(ws, ignore_errors=True)

    _out_a, _cells_a = run_and_read_chd(
        [{"type": "chd", "name": "上游定水头", "position": "west",
          "parameters": {"head": 8.0}}])
    check("E2E position=west：.chd 里全部是 col=1（1-based 的西边界）",
          bool(_cells_a) and {c for _r, c in _cells_a} == {1},
          f"cells={_cells_a}")
    check("E2E position=west：**没有**写在第 3 行（北边界，缺陷②的错误落点）",
          bool(_cells_a) and {r for r, _c in _cells_a} != {3},
          f"cells={_cells_a}")

    _out_b, _cells_b = run_and_read_chd(
        [{"type": "chd", "name": "边界A", "parameters": {"head": 8.0}}])
    check("E2E 无方位名：.chd 里 0 条记录（包不生成）", _cells_b == [],
          f"cells={_cells_b}")
    check("E2E 无方位名：**但 success 仍是 True** —— 这正是它骗人的地方",
          bool(_out_b.get("success")),
          "如果这条红了，说明程序已经改成失败返回，本锁的注释需要更新")

    _out_c, _cells_c = run_and_read_chd(
        [{"type": "chd", "name": "上游定水头", "parameters": {"head": 8.0}}])
    check("E2E 含流向词且无 position：不再落到北边界",
          _cells_c == [] or {r for r, _c in _cells_c} != {3},
          f"cells={_cells_c}")

except Exception as e:  # noqa: BLE001
    skip("端到端段", f"建模环境不可用：{type(e).__name__}: {e}")

# ======================================================================
print("\n[七] 界面卡片真的会显示（算了但没人看得见 = 等于没有）")

try:
    from app.frontend.gradio_app import _boundary_warning_card_html

    check("无告警 -> 空串（不占地方）", _boundary_warning_card_html([]) == "")
    check("无告警(None) -> 空串", _boundary_warning_card_html(None) == "")
    _html = _boundary_warning_card_html(["边界 '边界A'（CHD）不会写入模型：……"])
    check("有告警 -> 卡片里含'不会写入模型'", "不会写入模型" in _html)
    check("有告警 -> 卡片里含那条告警原文", "边界A" in _html)
except Exception as e:  # noqa: BLE001
    skip("界面卡片段", f"gradio_app 不可导入：{type(e).__name__}: {e}")

# ======================================================================
print("\n[八] 接线：告警真的从闸走到了界面上（函数存在 ≠ 接上了）")
# 教训（见 gradio_app 里"浓度框逐字符重绘"那次）：只断言"函数存在、数据正确"
# 会留下一个**空过的绿勾** —— 函数写好了但没人调它，界面照样什么都不显示。
# 所以这里直接扫源码，确认两根线都在。
ROOT = Path(__file__).parent.parent

try:
    _gp_src = (ROOT / "app" / "frontend" / "gradio_app.py").read_text(
        encoding="utf-8")
    check("界面：run_model_handler 把 gates['boundary_warnings'] 交给了卡片函数",
          re.search(r"_boundary_warning_card_html\(\s*gates\[.boundary_warnings.\]",
                    _gp_src) is not None)
    check("界面：卡片并进了 warn_html，最终会进结果面板",
          "warn_html += bw_html" in _gp_src)
except Exception as e:  # noqa: BLE001
    skip("界面接线检查", f"{type(e).__name__}: {e}")

try:
    _rt_src = (ROOT / "app" / "api" / "routes.py").read_text(encoding="utf-8")
    check("接口：completed 任务里带上了 boundary_warnings",
          "boundary_warnings=gates[\"boundary_warnings\"]" in _rt_src)
    check("接口：/result 也透传 boundary_warnings",
          '"boundary_warnings": t.get("boundary_warnings")' in _rt_src)
except Exception as e:  # noqa: BLE001
    skip("接口接线检查", f"{type(e).__name__}: {e}")

# ======================================================================
print(f"\n{'=' * 60}")
print(f"边界定位回归锁 测试结果: {passed}/{passed + failed} 通过"
      + (f"，{skipped} 跳过" if skipped else ""))
if failed:
    print(f"❌ 失败 {failed} 项")
print("=" * 60)
sys.exit(1 if failed else 0)

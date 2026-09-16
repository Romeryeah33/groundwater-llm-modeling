# -*- coding: utf-8 -*-
"""把三个验证算例导出成 JSON 文件，便于人工审阅与手动修改

算例的**唯一真源**是 verify_gwt_cases.py / verify_east_texas.py 里的构造函数，
本脚本只是把它们序列化出来，避免手抄的 JSON 与代码里的定义漂移。

用法:
    PYTHONIOENCODING=utf-8 python scripts/export_cases.py
输出:
    data/cases/case_a_column.json      一维土柱（有解析解）
    data/cases/case_b_rch_aux_l2.json  两层 + GHB + 带浓度 RCH
    data/cases/case_c_east_texas.json  East Texas 补齐后的 IR
"""
import copy
import io
import json
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

OUT_DIR = _ROOT / "data" / "cases"


def _num(v, default=None):
    """裸值 / {value,unit} 两种写法都取到数值"""
    if isinstance(v, dict):
        v = v.get("value")
    return default if v is None else v


def ui_compatible(ir: dict) -> dict:
    """把验证算例改写成**能过界面两道前置闸**的形态

    gradio_app.run_model_handler 在建模前会先跑 detect_missing_params 和
    quick_validate，任一 P1/P2 缺失或校验报错就直接返回、不进建模器。
    验证算例是直接喂给建模器的，绕过了这两道闸，所以照搬进界面会被拦。

    这里做的是**纯形态适配**，物理设置一律不动，逐条说明：

    1. 补 `domain`：P1 必需，且 validator 报"缺少顶层必需字段: domain"。
       `_needs_grid_generation` 只在缺 nrow/ncol 时才用 domain 生成网格
       （model_builder.py:405-408），三个算例都显式给了网格，所以补进来的
       domain 只用于展示，不影响 DIS。

    2. 补 `initial_conditions`：P2 必需。用 `{"type": "top_elevation"}` 而
       **不带 head 键** —— model_builder.py:695-699 只在显式给了
       `initial_conditions.head` 时才覆盖 strt，没给就走原逻辑，故物理中性。

    3. `stress_periods` 拆成两层：顶层放裸整数 nper，完整时间信息挪到
       `transport.stress_periods`。原因见 _has_value 的形状约定：
       顶层必须是裸值或带 "value" 键的 dict，**规范的
       {"nper":..,"period_data":[..]} 反而不被识别**（会被判为缺失）。
       而 model_mapper 早有"顶层只有 nper、时间信息在 transport 里就回退合并"
       的分支（就是为 East Texas 那种 IR 写的），两层配合正好既过闸又能解析。

    4. Case A 的 trpt/trpv/diffc 原为 0，被 validator 的范围检查判为错误
       （trpt∈[0.01,0.5]、trpv∈[0.001,0.1]、diffc∈[1e-10,1e-8]）。
       取区间下限。nrow=1 的网格没有横向方向，trpt/trpv 物理上不起作用；
       diffc 取 1e-10 对 D=al·v=0.04008 的相对贡献是 2.5e-9，可忽略。
       ——这三项取的是下限而非 0，属于适配闸门，不影响解析解对照结论。
    """
    ir = copy.deepcopy(ir)

    # 1) domain —— 由网格反推，只用于展示
    grid = ir.setdefault("grid", {})
    nrow, ncol = int(_num(grid.get("nrow"), 1)), int(_num(grid.get("ncol"), 1))
    delr, delc = _num(grid.get("delr"), 1.0), _num(grid.get("delc"), 1.0)
    ir.setdefault("domain", {
        "x_range": [0.0, round(ncol * float(delr), 6)],
        "y_range": [0.0, round(nrow * float(delc), 6)],
    })

    # 2) initial_conditions —— 不带 head，物理中性
    ic = ir.get("initial_conditions")
    if not isinstance(ic, dict) or not ic:
        ir["initial_conditions"] = {"type": "top_elevation"}

    # 3) stress_periods 拆两层
    sp = ir.get("stress_periods")
    if isinstance(sp, dict) and sp.get("period_data"):
        tr = ir.setdefault("transport", {})
        tr["stress_periods"] = copy.deepcopy(sp)
        ir["stress_periods"] = int(sp.get("nper") or len(sp["period_data"]))
    elif isinstance(sp, int):
        pass  # 已经是界面能接受的裸整数

    # 4) 弥散系数的闸门下限（仅当原来是 0 才动）
    disp = (ir.get("transport") or {}).get("dispersion")
    if isinstance(disp, dict):
        for key, floor in (("trpt", 0.01), ("trpv", 0.001), ("diffc", 1e-10)):
            cur = _num(disp.get(key))
            if cur is not None and float(cur) < floor:
                disp[key] = {"value": floor, "unit": disp[key].get("unit", "")
                             if isinstance(disp[key], dict) else "",
                             "source": "ui_gate_floor"}
    return ir


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []

    # Case A / B 直接来自 verify_gwt_cases.py
    from scripts.verify_gwt_cases import case_a_ir, case_b_ir

    # Case C 的来源是 data/sessions/default.json + 内存补齐
    sys.path.insert(0, str(_ROOT / "scripts"))
    from verify_east_texas import build_ir as et_build_ir

    cases = [
        ("case_a_column.json", case_a_ir,
         "Case A — 一维土柱，500 格 × 0.2 m，CHD 两端，有解析解可定量校验"),
        ("case_b_rch_aux_l2.json", case_b_ir,
         "Case B — 10×40×2，GHB 两侧 + 带浓度 RCH 面源，覆盖 AUX/AUXMIXED 机制"),
        ("case_c_east_texas.json", et_build_ir,
         "Case C — East Texas 本体（含 v2 在 default.json 上的 7 项补齐）"),
    ]

    for fname, fn, desc in cases:
        ir = fn()
        variants = [(fname, ir)]
        try:
            ui_ir = ui_compatible(ir)
            variants.append((fname.replace(".json", ".ui.json"), ui_ir))
        except Exception as e:
            print(f"!! {fname} 生成界面兼容版失败: {type(e).__name__}: {e}")

        for vname, vir in variants:
            path = OUT_DIR / vname
            path.write_text(
                json.dumps(vir, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            grid = vir.get("grid", {})
            tr = vir.get("transport", {})
            printed = _time_summary(vir)
            tag = "界面兼容版" if vname.endswith(".ui.json") else "原始版"
            print(f"[写出] {vname:<28} {tag}")
            print(f"       {desc}")
            print(f"       网格 {grid.get('nrow')}×{grid.get('ncol')}×{grid.get('nlay')}"
                  f"  {printed}"
                  f"  边界 {len(vir.get('boundary_conditions', []))} 个"
                  f"  transport.enabled={tr.get('enabled')}")
            written.append(path)

    print(f"\n共写出 {len(written)} 个文件到 {OUT_DIR}")
    print("\n运行方式（走的是界面【运行模型】按钮的同一个函数）:")
    print("  PYTHONIOENCODING=utf-8 python scripts/run_case.py data/cases/case_a_column.ui.json")
    print("  PYTHONIOENCODING=utf-8 python scripts/run_case.py data/cases/case_b_rch_aux_l2.ui.json")
    return 0


def _time_summary(ir: dict) -> str:
    """打印时间设置。界面兼容版把 period_data 挪到了 transport 下，两处都要看"""
    sp = ir.get("stress_periods")
    pd = []
    if isinstance(sp, dict):
        pd = sp.get("period_data") or []
    if not pd:
        pd = ((ir.get("transport") or {}).get("stress_periods") or {}).get("period_data") or []
    total = sum(float(p.get("perlen", 0)) for p in pd) if pd else 0.0
    nper = sp.get("nper") if isinstance(sp, dict) else sp
    return f"应力期 {nper}  时间步 {len(pd)} 项  总时长 {total:g} d"


if __name__ == "__main__":
    sys.exit(main())

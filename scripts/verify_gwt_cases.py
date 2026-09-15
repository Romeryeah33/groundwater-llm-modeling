# -*- coding: utf-8 -*-
"""溶质运移（GWT）链路验证脚本

Case A — 一维 Ogata–Banks 土柱：有解析解，可定量校验浓度场
Case B — RCH + AUX + AUXMIXED + nlay=2：覆盖 East Texas 需要的全套机制

用法:
    PYTHONIOENCODING=utf-8 python scripts/verify_gwt_cases.py [A|B]
"""
import io
import sys
from pathlib import Path

# Windows 控制台下 stdout 默认 GBK，中文/符号会炸
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from app.modeling.model_builder import AutoModelBuilder  # noqa: E402

OUT_ROOT = Path(__file__).resolve().parents[1] / "data" / "outputs" / "gwt_verify"


def case_a_ir() -> dict:
    """一维土柱：x 方向 100 m，500 格，左端 C=1 右端 C=0

    自洽参数：v = 0.04008 m/d，D = al*v = 0.04008 m²/d，
    t = 1250 d 时 v*t = 50.1 m（落在域中部），Courant = 0.20。
    """
    return {
        "ir_version": "1.0",
        "project": {"name": "ob_column", "model_type": "solute_transport"},
        "grid": {"nrow": 1, "ncol": 500, "nlay": 1,
                 "delr": 0.2, "delc": 1.0, "top": 1.0, "bottom": 0.0},
        "aquifer": {"type": "confined", "parameters": {
            "hk": {"value": 1.0, "unit": "m/d"},
            "vka": {"value": 0.1, "unit": "m/d"},
            "ss": {"value": 1e-6, "unit": "1/m"},
            "sy": {"value": 0.2},
            "porosity": {"value": 0.25},
        }},
        "boundary_conditions": [
            {"type": "chd", "name": "inlet_west",
             "parameters": {"head": {"value": 1.0}, "concentration": {"value": 1.0}}},
            {"type": "chd", "name": "outlet_east",
             "parameters": {"head": {"value": 0.0}, "concentration": {"value": 0.0}}},
        ],
        "stress_periods": {"nper": 1, "steady_state": False,
                           "period_data": [{"perlen": 1250.0, "nstp": 2500, "tsmult": 1.0}]},
        "transport": {
            "enabled": True,
            "porosity": {"value": 0.25},
            "advection": {"scheme": "tvd"},
            "dispersion": {"al": {"value": 1.0}, "trpt": {"value": 0.0},
                           "trpv": {"value": 0.0}, "diffc": {"value": 0.0}},
            "initial_concentration": 0.0,
        },
        "solver": {"name": "pcg", "parameters": {"hclose": 1e-8, "rclose": 1e-10}},
    }


def case_b_ir() -> dict:
    """两层网格 + GHB + 带浓度的 RCH，覆盖 East Texas 的全套机制"""
    return {
        "ir_version": "1.0",
        "project": {"name": "rch_aux_l2", "model_type": "solute_transport"},
        "grid": {"nrow": 10, "ncol": 40, "nlay": 2,
                 "delr": 10.0, "delc": 10.0, "top": 30.0, "bottom": 10.0},
        "aquifer": {"type": "confined", "parameters": {
            "hk": {"value": 2.0, "unit": "m/d"},
            "vka": {"value": 0.2, "unit": "m/d"},
            "ss": {"value": 1e-5, "unit": "1/m"},
            "sy": {"value": 0.15},
            "porosity": {"value": 0.3},
        }},
        "boundary_conditions": [
            {"type": "ghb", "name": "west_boundary",
             "parameters": {"stage": {"value": 30.0}, "conductance": {"value": 200.0}}},
            {"type": "ghb", "name": "east_boundary",
             "parameters": {"stage": {"value": 25.0}, "conductance": {"value": 200.0}}},
            {"type": "rch", "name": "landfill",
             "parameters": {"rate": {"value": 1e-4}, "concentration": {"value": 500.0}},
             "geometry": {"grid_indices": [
                 {"layer": 0, "row": r, "col": c, "period": 0}
                 for r in range(10) for c in range(10)
             ]}},
        ],
        "stress_periods": {"nper": 1, "steady_state": False,
                           "period_data": [{"perlen": 365.0, "nstp": 365, "tsmult": 1.0}]},
        "transport": {
            "enabled": True,
            "porosity": {"value": 0.3},
            "advection": {"scheme": "tvd"},
            "dispersion": {"al": {"value": 20.0}, "trpt": {"value": 0.1},
                           "trpv": {"value": 0.01}, "diffc": {"value": 1e-8}},
            "initial_concentration": 0.0,
        },
        "solver": {"name": "pcg", "parameters": {"hclose": 1e-6, "rclose": 1e-9}},
    }


def ogata_banks(x, t, v, D, c0=1.0):
    """半无限土柱、阶跃入口、零初值的解析解（**第一类 / Dirichlet 入口**）

        C = 0.5·erfc((x-vt)/(2√(Dt))) + 0.5·exp(vx/D)·erfc((x+vt)/(2√(Dt)))

    注意：**这不是 MF6 应该对照的参考解**，见 advective_inlet() 的说明。
    """
    from scipy.special import erfc
    a = (x - v * t) / (2.0 * np.sqrt(D * t))
    b = (x + v * t) / (2.0 * np.sqrt(D * t))
    return 0.5 * c0 * erfc(a) + 0.5 * c0 * np.exp(v * x / D) * erfc(b)


def advective_inlet(x, t, v, D, c0=1.0):
    """对流型入口的精确解（单项）

        C = 0.5·erfc((x-vt)/(2√(Dt)))

    等价于"无限域 + 阶跃初始条件"，也是**流速 v 的流体以浓度 c0 流入**
    且入口处不附加弥散通量时的解。

    ── 为什么 MF6 要对照这个而不是两项式 ──
    MF6 的弥散只在**相邻单元之间的内部界面**上计算，外部边界面上没有弥散
    通量。所以 CHD 入口供质纯粹是对流项 Q·C_aux，不包含 -D·∂C/∂x|₀ 那部分。
    第一类(Dirichlet)入口多送进来的质量总恰好是 αL·C0，它随前沿一起平移，
    表现为前沿整体下移 αL、域内总质量比 v·t 多出 αL。

    实测佐证（Case A，αL=1 m，v·t=50.1 m，t=1250 d）：
      · MF6 自身质量预算 CHD IN = 12.5251 = q·C0·T = 0.010020×1250 = 12.525
        —— 恰好是对流量，没有任何额外弥散供质；
      · MF6 域内质量 50.0802，与 v·t=50.1000 差 0.04%（等于 MF6 自己的
        预算离散度）；两项式参考为 51.1000，差 2.0%；
      · 前沿位移恒为 1.00 m，不随推进距离(8→67 m)变化、不随 Δx(0.1/0.2/0.4)
        变化，且 δ/αL = 1.01 —— 是刚性平移而非数值弥散或流速偏差；
      · max|C_MF6 − C_单项| = 0.0022，max|C_MF6 − C_两项| = 0.0414。
    """
    from scipy.special import erfc
    return 0.5 * c0 * erfc((x - v * t) / (2.0 * np.sqrt(D * t)))


def _crossing(x, c, level=0.5):
    """线性插值求 c(x)=level 的位置（要求剖面单调下降）"""
    idx = np.where(np.diff(np.sign(c - level)))[0]
    if not len(idx):
        return float("nan")
    i = idx[0]
    return x[i] + (level - c[i]) / (c[i + 1] - c[i]) * (x[i + 1] - x[i])


def check_case_a(res, ws):
    print("\n--- Case A 定量校验（对照对流型入口的解析解）---")
    gwt = res["gwt"]
    times = gwt.output.concentration().get_times()
    t_end = float(times[-1])
    conc = gwt.output.concentration().get_data(totim=t_end).ravel()

    ncol, delr = 500, 0.2
    x = (np.arange(ncol) + 0.5) * delr
    v = 1.0 * (1.0 - 0.0) / ((ncol - 1) * delr) / 0.25
    D = 1.0 * v
    ref = advective_inlet(x, t_end, v, D)
    wrong = ogata_banks(x, t_end, v, D)

    print(f"  t_end={t_end:g} d   v={v:.5f} m/d   D={D:.5f} m2/d   v*t={v * t_end:.2f} m")
    print(f"  C_num: min={conc.min():.4e} max={conc.max():.4e}")

    ok = True

    # ── 判据 1：与对流型入口解析解的最大偏差 ──
    max_err = float(np.max(np.abs(conc - ref)))
    print(f"  max|C_num - C_对流型入口| = {max_err:.4f}  (阈值 0.01)")
    ok &= max_err < 0.01
    print(f"  [参考] max|C_num - C_两项式| = {float(np.max(np.abs(conc - wrong))):.4f}"
          f"  （预期偏大 ~αL，说明入口边界类型不同，非模型错误）")

    # ── 判据 2：各浓度水平的穿透点（对数值弥散不敏感，最硬的判据）──
    # 每个浓度水平对应解析解自己的位置，不能都拿 v*t 当参考——
    # v*t 只是 C=0.5 的位置，C=0.25/0.75 各自偏移约 ±13%。
    for lv in (0.25, 0.5, 0.75):
        idx = np.where(np.diff(np.sign(conc - lv)))[0]
        if not len(idx):
            print(f"  !! 找不到 C={lv} 穿透点")
            ok = False
            continue
        i = idx[0]
        xc = x[i] + (lv - conc[i]) / (conc[i + 1] - conc[i]) * delr
        xr = _crossing(x, ref, lv)
        err = abs(xc - xr) / xr
        print(f"  C={lv:.2f} 穿透点: 数值={xc:.3f} m  解析={xr:.3f} m  相对误差={err * 100:.3f}%")
        ok &= err < 0.01

    # ── 判据 3：质量守恒（MF6 的入口供质应等于纯对流输入）──
    mass = float((conc * delr).sum())
    mass_ref = v * t_end
    mass_err = abs(mass - mass_ref) / mass_ref
    print(f"  域内总质量: MF6={mass:.4f}  纯对流输入 v*t={mass_ref:.4f}  偏差={mass_err * 100:.3f}%")
    ok &= mass_err < 0.01

    # ── 判据 4：无负浓度、无超调 ──
    neg = float(conc.min()) < -1e-9
    over = float(conc.max()) > 1.0 * (1 + 1e-6)
    print(f"  负浓度: {neg}   超调: {over}")
    ok &= not neg and not over

    print(f"  ==> Case A {'通过' if ok else '未通过'}")
    return ok


def parse_mass_budget(lst_path):
    """解析 GWT 列表文件中最后一段 MASS BUDGET

    Returns:
        {"in": {包名: 累积量}, "out": {...}, "discrepancy": float}  解析失败返回 None
    """
    text = Path(lst_path).read_text(encoding="utf-8", errors="replace")
    idx = text.upper().rfind("MASS BUDGET")
    if idx < 0:
        return None
    block = text[idx : idx + 3000]

    def _grab(section):
        out, capturing = {}, False
        for line in block.splitlines():
            s = line.strip()
            if s.startswith("IN:") and section == "in":
                capturing = True
                continue
            if s.startswith("OUT:") and section == "out":
                capturing = True
                continue
            if s.startswith("TOTAL") or s.startswith("IN - OUT"):
                capturing = False
                continue
            if capturing and "=" in s:
                name, val = s.split("=", 1)
                name = name.strip()
                val = val.strip().split()[0]
                try:
                    out[name] = float(val)
                except ValueError:
                    pass
        return out

    disc = None
    for line in block.splitlines():
        if "PERCENT DISCREPANCY" in line.upper():
            try:
                disc = float(line.split("=")[1].split()[0])
            except (IndexError, ValueError):
                pass
            break
    return {"in": _grab("in"), "out": _grab("out"), "discrepancy": disc}


def check_case_b(res, ws):
    print("\n--- Case B 机制校验（RCH + AUX + AUXMIXED + nlay=2）---")
    ok = True

    nam = (ws / "auto_model_gwt.nam").read_text(encoding="utf-8", errors="replace")
    has_mst = "MST6" in nam
    print(f"  [{'OK' if has_mst else 'FAIL'}] gwt.nam 含 MST6")
    ok &= has_mst

    a = AutoModelBuilder._griddata_block(ws / "auto_model_gwf.dis")
    b = AutoModelBuilder._griddata_block(ws / "auto_model_gwt.dis")
    same = a == b
    print(f"  [{'OK' if same else 'FAIL'}] GWF/GWT griddata 逐行一致")
    if not same:
        print(f"      GWF={a}")
        print(f"      GWT={b}")
    # 顺带确认多层 botm 真的写成了分层形式（这条以前会静默退化成 CONSTANT）
    layered = any("LAYERED" in s.upper() for s in a)
    print(f"  [{'OK' if layered else 'FAIL'}] botm 为 LAYERED 分层形式: "
          f"{[s for s in a if 'CONSTANT' in s.upper() or 'LAYERED' in s.upper()]}")
    ok &= same and layered

    rch = (ws / "auto_model_gwf.rch").read_text(encoding="utf-8", errors="replace")
    has_aux = "AUXILIARY" in rch.upper() and "CONCENTRATION" in rch.upper()
    print(f"  [{'OK' if has_aux else 'FAIL'}] rch 声明 AUXILIARY CONCENTRATION")
    ok &= has_aux

    ssm = (ws / "auto_model_gwt.ssm").read_text(encoding="utf-8", errors="replace")
    has_src = "RCHA" in ssm.upper() and "AUXMIXED" in ssm.upper()
    print(f"  [{'OK' if has_src else 'FAIL'}] ssm SOURCES = RCHA / AUXMIXED / CONCENTRATION")
    ok &= has_src

    # ── 质量守恒：注入量必须等于解析期望值 ──
    # 100 格 × (10 m × 10 m) × 1e-4 m/d × 500 × 365 d = 182500
    budget = parse_mass_budget(ws / "auto_model_gwt.lst")
    n_cells, delr, delc, rate, conc, perlen = 100, 10.0, 10.0, 1e-4, 500.0, 365.0
    expected = n_cells * delr * delc * rate * conc * perlen
    if budget is None:
        print("  [FAIL] 未能解析 MASS BUDGET")
        ok = False
    else:
        got = budget["in"].get("RCH", float("nan"))
        err = abs(got - expected) / expected
        print(f"  [{'OK' if err < 1e-3 else 'FAIL'}] RCH 累积注入 = {got:.4g}"
              f"  期望 = {expected:.4g}  偏差 = {err * 100:.4f}%")
        ok &= err < 1e-3
        disc = budget["discrepancy"]
        print(f"  [{'OK' if disc is not None and disc < 1.0 else 'FAIL'}] "
              f"GWT 质量离散度 = {disc}%  (阈值 1%)")
        ok &= disc is not None and disc < 1.0

    # ── 浓度场 ──
    gwt = res["gwt"]
    times = gwt.output.concentration().get_times()
    c = np.asarray(gwt.output.concentration().get_data(totim=float(times[-1])))
    c0 = c[0].ravel()
    c1 = c[1].ravel() if c.shape[0] > 1 else np.zeros_like(c0)
    print(f"  layer0: max={c0.max():.4f}   layer1: max={c1.max():.4f}")
    # layer0 有浓度 => 源真的流进来了
    l0_ok = c0.max() > 0.01
    # layer1 有浓度 => 垂向连接与网格分层是通的（第二层厚度为 0 时这里恒为 0）
    l1_ok = c1.max() > 0.0
    # 补给水被含水层稀释：源浓度 500，层0 峰值应远低于它
    dil_ok = c0.max() < 0.1 * conc
    print(f"  [{'OK' if l0_ok else 'FAIL'}] layer0 浓度 > 0.01（源已流入）")
    print(f"  [{'OK' if l1_ok else 'FAIL'}] layer1 浓度 > 0（垂向路径连通）")
    print(f"  [{'OK' if dil_ok else 'FAIL'}] layer0 峰值远低于源浓度 500（被稀释）")
    ok &= l0_ok and l1_ok and dil_ok

    neg = float(c.min()) < -1e-9
    overshoot = float(c.max()) > conc * (1 + 1e-6)
    print(f"  [{'OK' if not neg else 'FAIL'}] 无负浓度 (min={c.min():.3e})")
    print(f"  [{'OK' if not overshoot else 'FAIL'}] 无超调 (max={c.max():.4f} <= {conc})")
    ok &= not neg and not overshoot

    print(f"  ==> Case B {'通过' if ok else '未通过'}")
    return ok


def main():
    which = (sys.argv[1] if len(sys.argv) > 1 else "AB").upper()
    results = {}

    if "A" in which:
        print("=" * 70)
        print("Case A — 一维 Ogata–Banks 土柱")
        print("=" * 70)
        ws = OUT_ROOT / "caseA"
        builder = AutoModelBuilder(str(ws))
        res = builder.build_and_run(case_a_ir(), "auto_model")
        if not res["success"]:
            print("  !! 运行失败")
            print(res.get("detail") or res.get("error"))
            results["A"] = False
        else:
            print(f"  运行成功，水头范围 {res['head_range']}")
            results["A"] = check_case_a(res, ws / "auto_model")

    if "B" in which:
        print("\n" + "=" * 70)
        print("Case B — RCH + AUX + AUXMIXED + nlay=2")
        print("=" * 70)
        ws = OUT_ROOT / "caseB"
        builder = AutoModelBuilder(str(ws))
        res = builder.build_and_run(case_b_ir(), "auto_model")
        if not res["success"]:
            print("  !! 运行失败")
            print(res.get("detail") or res.get("error"))
            results["B"] = False
        else:
            print(f"  运行成功，水头范围 {res['head_range']}")
            results["B"] = check_case_b(res, ws / "auto_model")

    print("\n" + "=" * 70)
    for k, v in results.items():
        print(f"  Case {k}: {'通过' if v else '未通过'}")
    print("=" * 70)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

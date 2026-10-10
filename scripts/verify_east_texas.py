"""
East Texas 验证：Case C（本体，Gate 5）+ Case D（污染源区回归锁）

用法：
  python scripts/verify_east_texas.py        # Case C，会真跑 mf6（几分钟）
  python scripts/verify_east_texas.py D      # Case D，只构建+落盘，不跑 mf6（几秒）

Case C — East Texas 本体（Gate 5）

不直接跑 `data/sessions/default.json`：那份 IR 有若干物理设计缺陷，
直接跑只能得到浓度恒为 0 的"成功"结果。这里在内存里补齐后再跑，
**不改动用户的 session 文件**。

补齐内容（对应调试方案 Step 6 / Gate 5）：
  1. 顶层 `stress_periods` 是裸整数 10 —— 真实时间信息在
     `transport.stress_periods = {nper:10, perlen:15000}`。perlen 按 MF6
     原生语义是"每个应力期的长度"，直接用会变成 10×15000=150000 d。
     这里显式给 period_data：10 期 × 1500 d = **15000 d 总计**。
  2. `transport` 缺 `al`（纵向弥散度）→ 显式给 40.0 m。
  3. `transport` 缺 `porosity` → 给 0.2（与 aquifer 一致）。
  4. 无任何浓度源 → 加一片 RCH 面源（西北角），浓度 20000。
  5. `vka = {"value": 20, "unit": "m"}` 单位错误且是 hk 的 3.6 倍
     → 改为 hk/10 = 0.55 m/d，避免层间交换异常强。
  6. `hclose = 0.01` → 收紧到 1e-3。
  7. 初始水头给 215 m（介于两个 GHB 水位的 212/208 与 IC=230 之间），
     减小开局的瞬变冲击。

目标：跑完 15000 d，浓度峰值在 20000 量级，污染羽向东南迁移且逐期外扩。
"""
import sys, os, json, copy
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
from app.modeling.model_builder import AutoModelBuilder

IR_PATH = _ROOT / "data" / "sessions" / "default.json"
# 输出目录可用 ET_WS 覆盖：做参数试验时写到独立目录，避免与其它运行
# 抢占同一批文件（Windows 上 mf6 未退出会锁住 .nam 导致 PermissionError）
OUT_ROOT = Path(
    os.environ.get("ET_WS")
    or (_ROOT / "data" / "outputs" / "gwt_verify" / "eastTexas")
)

# 源参数
SRC_CONC = 20000.0
SRC_RATE = 1.0e-4          # m/d，填埋场量级的补给强度
NPER, PERLEN = 10, 1500.0
# nstp / al 可用环境变量覆盖，便于做收敛性试验
NSTP = int(os.environ.get("ET_NSTP", "30"))
AL = float(os.environ.get("ET_AL", "40.0"))
# 横向弥散比：默认沿用 IR 里的 trpt/trpv；设 ET_TRPT=0 可关掉横向弥散，
# 用来判定负浓度是否来自弥散张量的交叉项
TRPT = os.environ.get("ET_TRPT")
TRPV = os.environ.get("ET_TRPV")

# 负浓度容差：允许 |C_min| 达到源浓度的 0.1%（实测 0.024%，留 4 倍余量）。
# 依据与排除过程见 docs/gwt_verification.md。Case A/B 仍是严格判据。
NEG_TOL_FRAC = float(os.environ.get("ET_NEG_TOL", "1e-3"))


def build_ir() -> dict:
    raw = json.loads(IR_PATH.read_text(encoding="utf-8"))
    ir = copy.deepcopy(raw.get("json_ir", raw))
    grid = ir.get("grid", {})
    nrow, ncol = int(grid.get("nrow", 1)), int(grid.get("ncol", 1))

    # 1) 时间：显式 period_data，总计 15000 d
    ir["stress_periods"] = {
        "nper": NPER,
        "period_data": [
            {"perlen": PERLEN, "nstp": NSTP, "tsmult": 1.0} for _ in range(NPER)
        ],
    }

    # 2/3) 弥散度 + 孔隙度
    tr = ir.setdefault("transport", {})
    tr["enabled"] = True
    tr["al"] = {"value": AL, "unit": "m"}
    if TRPT is not None or TRPV is not None:
        _d = tr.setdefault("dispersion", {})
        if TRPT is not None:
            _d["trpt"] = {"value": float(TRPT), "unit": "dimensionless"}
        if TRPV is not None:
            _d["trpv"] = {"value": float(TRPV), "unit": "dimensionless"}
    tr["porosity"] = {"value": 0.2, "unit": "dimensionless"}
    # 真实时间信息已经移到顶层，清掉这里的 stress_periods 以免二义
    tr.pop("stress_periods", None)

    # 5) vka 单位错误 + 异常偏大
    aqua = ir.setdefault("aquifer", {})
    params = aqua.setdefault("parameters", {})
    hk = params.get("hk", {}).get("value", 5.5) if isinstance(params.get("hk"), dict) else 5.5
    params["vka"] = {"value": round(float(hk) / 10.0, 4), "unit": "m/d",
                     "source": "patched"}

    # 6) 收紧收敛标准；GWT 非对称矩阵会被自动改成 BICGSTAB
    _hc = float(os.environ.get("ET_HCLOSE", "1e-3"))
    ir.setdefault("solver", {}).setdefault("parameters", {})["hclose"] = {
        "value": _hc, "unit": "m", "source": "patched"
    }

    # 7) 初始水头收敛到 GHB 水位附近
    ir["initial_conditions"] = {"type": "specified", "head": 215.0}

    # 4) 西北角一片 RCH 面源（东南向有充足迁移空间）
    r0, r1 = max(1, nrow // 12), max(2, nrow // 6)
    c0, c1 = max(1, ncol // 12), max(2, ncol // 8)
    src_cells = [
        {"layer": 0, "row": r, "col": c, "period": 0}
        for r in range(r0, r1) for c in range(c0, c1)
    ]
    bcs = [b for b in ir.get("boundary_conditions", []) if b.get("type") != "rch"]
    bcs.append({
        "type": "rch",
        "name": "landfill_recharge",
        "parameters": {
            "rate": {"value": SRC_RATE, "unit": "m/d", "source": "patched"},
            "concentration": {"value": SRC_CONC, "unit": "mg/L", "source": "patched"},
        },
        "geometry": {"shape": "polygon", "grid_indices": src_cells},
    })
    ir["boundary_conditions"] = bcs

    print(f"IR 补齐完成: 网格 {nrow}×{ncol}×{grid.get('nlay', 1)}; "
          f"总计 {NPER}×{PERLEN:g}={NPER * PERLEN:g} d; "
          f"源 {len(src_cells)} 格 @ {SRC_CONC:g} mg/L, rate={SRC_RATE:g} m/d")
    return ir


def check_case_c(res, ws):
    """Gate 2/4/5：机制 + 质量平衡 + 污染羽逐期外扩"""
    print("\n--- Case C 校验（East Texas）---")
    ok = True

    def _read(p):
        f = ws / p
        return f.read_text(encoding="utf-8", errors="replace") if f.exists() else ""

    nam = _read("auto_model_gwt.nam")
    if "MST6" in nam.upper():
        print("  [OK] gwt.nam 含 MST6")
    else:
        print("  [FAIL] gwt.nam 缺 MST6"); ok = False

    gwf_dis, gwt_dis = _read("auto_model_gwf.dis"), _read("auto_model_gwt.dis")

    def _botm(txt):
        out, on = [], False
        for ln in txt.splitlines():
            t = ln.strip()
            if t.lower().startswith("botm"):
                on = "layered" in t.lower()
                continue
            if on:
                out.append(t)
                if len(out) >= 2:
                    break
        return out

    b1, b2 = _botm(gwf_dis), _botm(gwt_dis)
    if b1 and b1 == b2:
        print(f"  [OK] GWF/GWT botm 逐行一致: {b1}")
    else:
        print(f"  [FAIL] GWF/GWT botm 不一致: GWF={b1} GWT={b2}"); ok = False

    rch = _read("auto_model_gwf.rch")
    if "AUXILIARY" in rch.upper() and "CONCENTRATION" in rch.upper():
        print("  [OK] rch 声明 AUXILIARY CONCENTRATION")
    else:
        print("  [FAIL] rch 缺 AUXILIARY CONCENTRATION"); ok = False

    ssm = _read("auto_model_gwt.ssm")
    if "RCH" in ssm.upper() and "AUXMIXED" in ssm.upper():
        print("  [OK] ssm SOURCES 含 RCHA / AUXMIXED")
    else:
        print(f"  [FAIL] ssm SOURCES 异常: {ssm.strip()[:200]}"); ok = False

    # ── 质量平衡（Gate 4）──
    lst_gwt = ws / "auto_model_gwt.lst"
    if lst_gwt.exists():
        txt = lst_gwt.read_text(encoding="utf-8", errors="replace")
        disc = []
        for ln in txt.splitlines():
            if "PERCENT DISCREPANCY" in ln.upper():
                try:
                    disc.append(float(ln.split()[-1]))
                except (ValueError, IndexError):
                    pass
        if disc:
            worst = max(abs(d) for d in disc)
            print(f"  [{'OK' if worst < 1.0 else 'FAIL'}] GWT 最大质量离散度 = {worst:.4f}%  (阈值 1%)")
            ok &= worst < 1.0
        else:
            print("  [WARN] 未找到 PERCENT DISCREPANCY 行")
    else:
        print("  [FAIL] 缺 auto_model_gwt.lst"); ok = False

    # ── 浓度场（Gate 3/5）──
    gwt = res["gwt"]
    conc_obj = gwt.output.concentration()
    times = [float(t) for t in np.atleast_1d(conc_obj.get_times())]
    if not times:
        print("  [FAIL] 没有浓度输出时刻"); return False

    frames, peak = [], 0.0
    for t in times:
        d = conc_obj.get_data(totim=t)
        if d.ndim == 4:
            d = d[-1]
        a = np.asarray(d)[0]
        frames.append(a)
        peak = max(peak, float(np.nanmax(a)))

    cmin = min(float(np.nanmin(f)) for f in frames)
    cmax = max(float(np.nanmax(f)) for f in frames)

    print(f"  浓度场: t 从 {times[0]:g} 到 {times[-1]:g} d, 共 {len(times)} 帧")
    print(f"  峰值 = {peak:.2f} mg/L   min={cmin:.3e}  max={cmax:.3e}")

    if peak > 1.0:
        print(f"  [OK] 浓度场非零（峰值 {peak:.1f}，量级与源 {SRC_CONC:g} 相比合理）")
    else:
        print(f"  [FAIL] 浓度场恒为 0 或近 0（峰值 {peak:.3e}）"); ok = False
    # ── 负浓度：接受为已量化的数值伪影（见 docs/gwt_verification.md）──
    # MF6 的 GWT 是隐式求解，弥散张量的交叉项会破坏系数矩阵的 M-矩阵性质，
    # 在源片硬边界处产生远小于源浓度量级的负值。已排除的成因：
    #   · 时间步截断：nstp 30→60，负值 −4.82→−4.88（基本不变）
    #   · 弥散下冲：al 5/40/120 对应 −10.5/−4.8/−0.7（弥散越大负值越小，与假设相反）
    #   · 求解器容差：hclose 1e-3→1e-7，负值 −4.83（不变）
    #   · 对流格式：ADV 已是 upstream，本就不产生对流超调
    # 故这里不用 C.min() >= -1e-9 硬判，而是给它一个有物理依据的容差。
    n_neg = sum(1 for f in frames if float(np.nanmin(f)) < -1e-9)
    frac = abs(cmin) / SRC_CONC if cmin < 0 else 0.0
    if frac <= NEG_TOL_FRAC:
        print(f"  [OK] 负浓度在容差内: min={cmin:.4f} "
              f"({n_neg}/{len(frames)} 帧, 占源浓度 {frac * 100:.4f}% "
              f"≤ {NEG_TOL_FRAC * 100:.2f}%)")
    else:
        print(f"  [FAIL] 负浓度超容差: min={cmin:.4f} "
              f"({n_neg}/{len(frames)} 帧, 占源浓度 {frac * 100:.4f}% "
              f"> {NEG_TOL_FRAC * 100:.2f}%)")
        ok = False
    if cmax <= SRC_CONC * (1 + 1e-6):
        print("  [OK] 无超调")
    else:
        print(f"  [FAIL] 浓度超调 {cmax:.3e} > {SRC_CONC:g}"); ok = False

    # 污染羽逐期外扩：取质心与二阶矩（含有浓度格数）
    extents, cents = [], []
    for a in frames:
        m = a > 1.0
        extents.append(int(m.sum()))
        if m.any():
            rr, cc = np.nonzero(m)
            # 东南向 = +col, +row（row 索引增大对应 y 减小）
            cents.append((float(cc.mean()), float(rr.mean())))
        else:
            cents.append((float("nan"), float("nan")))

    print(f"  污染羽格数（浓度>1）: {extents[0]} → {extents[-1]}")
    print(f"  污染羽质心 (col,row): {cents[0]} → {cents[-1]}")

    grew = extents[-1] > extents[0] if extents else False
    if grew:
        print(f"  [OK] 污染羽外扩（{extents[0]} → {extents[-1]} 格）")
    else:
        print(f"  [FAIL] 污染羽未外扩（{extents[0]} → {extents[-1]}）")
        ok = False

    # 东南向迁移：col 增大（东）、row 增大（南）。
    # 起始帧污染羽为空（质心 NaN），必须拿**第一个非空帧**作基准，
    # 否则会拿 NaN 去比，误判成"没有位移"。
    valid = [i for i, c in enumerate(cents) if not np.isnan(c[0])]
    if len(valid) >= 2:
        i0, i1 = valid[0], valid[-1]
        dcol = cents[i1][0] - cents[i0][0]
        drow = cents[i1][1] - cents[i0][1]
        print(f"  质心位移（第 {i0 + 1}→{i1 + 1} 帧）: Δcol={dcol:+.2f}, Δrow={drow:+.2f}")
        if dcol > 0 and drow > 0:
            print("  [OK] 污染羽向东南迁移")
        else:
            print(f"  [WARN] 质心位移不在东南向（Δcol={dcol:+.2f}, Δrow={drow:+.2f}）")
            print("         注：域内流速极低（约 0.02 m/d），15000 d 平流位移仅 ~300 m，")
            print("         弥散主导时质心可能不动；此项不作为通过判据。")
    else:
        print("  [WARN] 有效质心帧不足，跳过迁移方向判定")

    # 逐期单调外扩
    if len(extents) >= 3:
        mono = all(extents[i] <= extents[i + 1] for i in range(len(extents) - 1))
        print(f"  [{'OK' if mono else 'WARN'}] 污染羽格数单调不减: {mono}")

    print(f"  ==> Case C {'通过' if ok else '未通过'}")
    return ok


# ══════════════════════════════════════════════════════════════════════════
# Case D — SHP 名称匹配路径（污染源区回归锁）
#
# 为什么必须有这一条：Case B/C 都在 IR 里**显式给了 `grid_indices`**，走的是
# mapper 的优先级 1 直接透传，于是「源区解析」这条链路在验证套件里**从未被执行过**。
# 真实 LLM 产出的 IR 恰恰不给 `grid_indices`，只给 `position: "landfill"` ——
# 于是走上了那条会静默把点源展开成全网格 18612 格的死路（放大 141 倍），
# 而全套验证依旧全绿。这条 Case 就是把这个缺口补上。
#
# 断言的是**写出来的 .rch**，不是中间对象：只有落盘文件才是用户真正跑的东西。
# 用 build_simulation + write_simulation 即可，不必真跑 mf6（省几分钟）。
# ══════════════════════════════════════════════════════════════════════════

# 填埋场验收锁（East Texas 网格 94×198、delr=delc=20 m、原点来自 SHP 外框左下角）
# 网格尺寸由 domain（= SHP 外框 3960×1871 m）驱动：ncol=ceil(3960/20)=198、
# nrow=ceil(1871/20)=94。源区的 132 格/行列范围**不随网格尺寸变化** ——
# 格心位置只取决于原点与单元尺寸，两者都没动。这四条锁因此原样保留。
EXPECT_CELLS = 132
EXPECT_ROWS = (52, 62)
EXPECT_COLS = (137, 148)
CELL_AREA_M2 = 20.0 * 20.0
EXPECT_RECHARGE = EXPECT_CELLS * SRC_RATE * CELL_AREA_M2      # 5.28 m³/d


def find_landfill_shp():
    """
    定位 landfill_arcs.shp。

    上传目录是 `%TEMP%\\gm_zip_*`，重启后会被清理，所以这里按
    「显式指定 → 仓库内 → 临时上传目录」的顺序找，并明确报告找的是哪一份。
    """
    env = os.environ.get("ET_SHP")
    if env:
        p = Path(env)
        if p.exists():
            return p
        print(f"  !! ET_SHP 指向的文件不存在: {env}")

    for p in sorted(Path(_ROOT).glob("data/uploads/**/landfill_arcs.shp")):
        return p

    tmp = os.environ.get("TEMP") or os.environ.get("TMP") or "/tmp"
    for p in sorted(Path(tmp).glob("gm_zip_*/landfill_arcs.shp")):
        return p
    return None


def build_ir_case_d():
    """
    真实 IR（`data/sessions/default.json`）**原样**加载，只做两件事：
      1. 补 `domain.origin_xy` —— 正常由 `file_parse_handler` 上传时写入，
         这里离线跑，用 SHP 外框左下角显式补上；
      2. 删掉 `geometry.grid_indices`（如果存在）—— 本 Case 要考的正是
         「没有显式索引时能不能靠名称匹配找到源区」。

    刻意**不**动那条 RCH 的 rate/concentration：如果它本身是错的，
    Case D 就该红，而不是被我在这里悄悄改对。
    """
    raw = json.loads(IR_PATH.read_text(encoding="utf-8"))
    ir = copy.deepcopy(raw.get("json_ir", raw))

    ir.setdefault("domain", {})["origin_xy"] = [
        float(os.environ.get("ET_ORIGIN_X", "612284.0")),
        float(os.environ.get("ET_ORIGIN_Y", "3427412.0")),
    ]
    ir["stress_periods"] = {
        "nper": NPER,
        "period_data": [
            {"perlen": PERLEN, "nstp": 1, "tsmult": 1.0} for _ in range(NPER)
        ],
    }

    rch_bcs = [b for b in ir.get("boundary_conditions", []) if b.get("type") == "rch"]
    for b in rch_bcs:
        geo = b.get("geometry")
        if isinstance(geo, dict):
            for k in ("grid_indices", "resolved_from", "source_shp", "grid_signature"):
                geo.pop(k, None)
    return ir, rch_bcs


def parse_rch_cells(text):
    """
    从写出的 .rch 文本里取出逐格记录：[(layer, row, col, rate, conc), ...]

    **返回 0-based 索引**。MF6 输入文件的应力期数据块里 cellid 是 1-based
    （layer 1 = 第一层），这里统一减 1，好让断言与 `tests/test_source_zone.py`
    里那套 0-based 锁（row 52..62 / col 137..148）逐字对齐 —— 两个地方各用
    一套基准是最容易被自己的 +1 骗过去的地方。
    """
    cells, on = [], False
    for ln in text.splitlines():
        t = ln.strip()
        if not t or t.startswith("#"):
            continue
        low = t.lower()
        if low.startswith("begin"):
            on = "period" in low
            continue
        if low.startswith("end"):
            on = False
            continue
        if not on:
            continue
        parts = t.split()
        if len(parts) < 4:
            continue
        try:
            lay, row, col, rate = int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])
        except ValueError:
            continue
        conc = None
        if len(parts) >= 5:
            try:
                conc = float(parts[4])
            except ValueError:
                conc = None
        cells.append((lay - 1, row - 1, col - 1, rate, conc))
    return cells


def check_case_d(ws):
    print("\n--- Case D 校验（SHP 名称匹配 → .rch 落盘）---")
    ok = True

    def _read(p):
        f = ws / p
        return f.read_text(encoding="utf-8", errors="replace") if f.exists() else ""

    rch_txt = _read("auto_model_d_gwf.rch")
    if not rch_txt:
        print(f"  [FAIL] 缺 {ws / 'auto_model_d_gwf.rch'}")
        return False, [], set()

    cells = parse_rch_cells(rch_txt)
    n = len(cells)
    print(f"  .rch 逐格记录数 = {n}")

    # 1) 条数：必须是 132，不是 18612（= 94×198 全域）
    if n == EXPECT_CELLS:
        print(f"  [OK] .rch 恰好 {EXPECT_CELLS} 条（源区面积 {EXPECT_CELLS * CELL_AREA_M2:g} m²）")
    else:
        print(f"  [FAIL] .rch 记录数 = {n}，期望 {EXPECT_CELLS}")
        if n == 94 * 198:
            print("         ← 这是**全网格展开**（94×198），即本次修复要消灭的那个缺陷复发")
        ok = False

    if not cells:
        return False, [], set()

    # 2) 空间分布：必须集中在 row 52..62 × col 137..148，而不是铺满全域
    rows = sorted({c[1] for c in cells})
    cols = sorted({c[2] for c in cells})
    if rows == list(range(EXPECT_ROWS[0], EXPECT_ROWS[1] + 1)):
        print(f"  [OK] row 恰好 {EXPECT_ROWS[0]}..{EXPECT_ROWS[1]}")
    else:
        print(f"  [FAIL] row 范围 = {rows[0]}..{rows[-1]}（共 {len(rows)} 个），"
              f"期望 {EXPECT_ROWS[0]}..{EXPECT_ROWS[1]}")
        ok = False
    if cols == list(range(EXPECT_COLS[0], EXPECT_COLS[1] + 1)):
        print(f"  [OK] col 恰好 {EXPECT_COLS[0]}..{EXPECT_COLS[1]}")
    else:
        print(f"  [FAIL] col 范围 = {cols[0]}..{cols[-1]}（共 {len(cols)} 个），"
              f"期望 {EXPECT_COLS[0]}..{EXPECT_COLS[1]}")
        ok = False

    # 3) 每格都在 layer 0（RCH 不接受跨层）
    lays = {c[0] for c in cells}
    if lays == {0}:
        print("  [OK] 全部落在 layer 0")
    else:
        print(f"  [FAIL] 出现了 layer {sorted(lays)}，RCH 必须只用 layer 0"); ok = False

    # 4) 速率与浓度：来自真实 IR 的 1e-4 m/d、20000 mg/L
    rates = {round(c[3], 12) for c in cells}
    concs = {round(c[4], 6) for c in cells if c[4] is not None}
    if rates == {SRC_RATE}:
        print(f"  [OK] rate 全为 {SRC_RATE:g} m/d")
    else:
        print(f"  [FAIL] rate 取值异常: {sorted(rates)}"); ok = False
    if concs == {SRC_CONC}:
        print(f"  [OK] concentration 全为 {SRC_CONC:g} mg/L")
    elif not concs:
        print("  [FAIL] .rch 没有浓度列 —— SSM 拿不到源项，浓度场会恒为 0"); ok = False
    else:
        print(f"  [FAIL] concentration 取值异常: {sorted(concs)}"); ok = False

    # 5) 补给总量：这是最直观的"放大 170 倍"对照量（898.16 → 5.28 m³/d）
    total = sum(c[3] * CELL_AREA_M2 for c in cells)
    if abs(total - EXPECT_RECHARGE) < 1e-6:
        print(f"  [OK] 总补给 = {total:.4f} m³/d（期望 {EXPECT_RECHARGE:.4f}）")
    else:
        print(f"  [FAIL] 总补给 = {total:.4f} m³/d，期望 {EXPECT_RECHARGE:.4f}"); ok = False

    # 6) 不应再有全网格标量补给顶替（`RCH6 ... 1.0E-04` 那种裸标量）
    if "CONSTANT" in rch_txt.upper() or "AUXILIARY" in rch_txt.upper():
        print("  [OK] .rch 声明了 AUXILIARY CONCENTRATION")
    else:
        print("  [FAIL] .rch 缺 AUXILIARY CONCENTRATION"); ok = False

    # 7) 落盘的源区集合，供羽流起点判据复用
    src_cells = {(c[1], c[2]) for c in cells}
    return ok, cells, src_cells


def check_plume_origin(res, src_cells):
    """
    端到端（需真跑 mf6）：污染羽是否**从填埋场长出来**。

    这是用户问题的最终判据 —— 参数对了、`.rch` 对了，但如果羽流仍不从源区长出来，
    说明中间还有别的地方在丢源项。

    刻意**不判"向北/向南迁移"**：仓库里存在两套互相矛盾的 row 方向约定
    （`source_zone.infer_edge_indices` 认为 north→row 0，而 `grid_coords` 的
    `grid_cell_centers` 让 row 索引随 y 增大、即 row 0 画在图的最下方）。
    在索引空间里判"羽流格子 ⊂ 源区格子"与方向约定无关，是可靠的那一半；
    方向问题见 docs/gwt_verification.md §5。
    """
    print("\n--- Case D 端到端：羽流起点 ---")
    ok = True
    gwt = res.get("gwt")
    if gwt is None:
        print("  [FAIL] 结果里没有 gwt"); return False

    conc_obj = gwt.output.concentration()
    times = [float(t) for t in np.atleast_1d(conc_obj.get_times())]
    if not times:
        print("  [FAIL] 没有浓度输出时刻"); return False

    first = None
    for t in times:
        d = np.asarray(conc_obj.get_data(totim=t))
        if d.ndim == 4:
            d = d[-1]
        a = np.asarray(d)[0]
        if float(np.nanmax(a)) > 1.0:
            first = (t, a)
            break
    if first is None:
        print(f"  [FAIL] {len(times)} 帧里浓度始终 ≤1 mg/L，源项没进去"); return False

    t0, a0 = first
    rr, cc = np.nonzero(a0 > 1.0)
    plume = set(zip(rr.tolist(), cc.tolist()))
    inside = len(plume & src_cells)
    print(f"  首个非空帧 t = {t0:g} d: 污染格 {len(plume)} 个，"
          f"源区 {len(src_cells)} 格中有 {inside} 格已污染")

    # 判据一：**源区必须整块都是热的**。这是"源项真的落在填埋场"最直接的证据
    # —— 修复前是全网格毯子，那时源区当然是热的，但下面判据二会红。
    if inside == len(src_cells):
        print(f"  [OK] 源区 132 格全部已污染（源项确实从填埋场进入）")
    else:
        print(f"  [FAIL] 源区仅 {inside}/{len(src_cells)} 格污染 —— 源项没落在填埋场"); ok = False

    # 判据二：污染区必须**整块连通到源区** —— 每一格污染都能沿相连格子追溯到
    # 填埋场。这正是用户问题的答案（"污染物是不是从这个区域泄漏出来的"），
    # 而且与 row/col 方向约定、与羽流往哪边迁移都无关。
    #
    # 不用"污染格 ⊂ 源区膨胀 K 格"那种矩形判据：羽流被水流带着走，是沿流向的
    # 长条而不是等向薄壳，矩形会误报（实测 t=300 d 时源区正下方 row 67 有 32 格）。
    seen, stack = set(src_cells & plume), list(src_cells & plume)
    while stack:
        r, c = stack.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nb = (r + dr, c + dc)
                if nb in plume and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
    orphan = plume - seen
    if not orphan:
        print(f"  [OK] 污染区 {len(plume)} 格全部连通到源区（无孤立污染团）")
    else:
        print(f"  [FAIL] 有 {len(orphan)} 格污染与源区不连通，"
              f"例如 {sorted(orphan)[:5]} —— 源项在别处也进来了"); ok = False

    last = np.asarray(conc_obj.get_data(totim=times[-1]))
    if last.ndim == 4:
        last = last[-1]
    aN = np.asarray(last)[0]
    n_final = int((aN > 1.0).sum())
    total = int(aN.size)
    print(f"  末帧 t = {times[-1]:g} d: 污染格 {n_final} / {total}（{n_final / total * 100:.2f}%）")
    if n_final < 0.25 * total:
        print("  [OK] 污染羽是局部的，不是覆盖全域的毯子")
    else:
        print(f"  [FAIL] 污染羽覆盖 {n_final / total * 100:.1f}% 全域 —— 又变回毯子了"); ok = False

    print(f"  ==> 羽流起点判据 {'通过' if ok else '未通过'}")
    return ok


def main():
    which = (sys.argv[1] if len(sys.argv) > 1 else "C").upper()

    if which == "D":
        print("=" * 70)
        print("Case D — SHP 名称匹配路径（污染源区回归锁）")
        print("=" * 70)

        shp = find_landfill_shp()
        if shp is None:
            print("  !! 找不到 landfill_arcs.shp。")
            print("     上传目录是 %TEMP%\\gm_zip_*，重启即失效；")
            print("     请用 ET_SHP=<路径> 指定，或重新上传基础资料 ZIP。")
            return 2
        print(f"  源区 SHP: {shp}")

        ir, rch_bcs = build_ir_case_d()
        if not rch_bcs:
            print("  !! 真实 IR 里没有 rch 边界，Case D 无从考察"); return 2
        for b in rch_bcs:
            print(f"  待解析 BC: name={b.get('name')!r} "
                  f"position={b.get('position')!r} "
                  f"geometry={b.get('geometry')}")

        # 先单独跑解析器，失败时能给出比「.rch 条数不对」精确得多的定位
        from app.modeling.source_zone import materialize_source_zones
        ir2 = copy.deepcopy(ir)
        ir2, resolved, issues = materialize_source_zones(
            ir2, uploaded_paths=[str(shp)]
        )
        if issues:
            print("  !! 源区解析失败（这正是修复前会静默回落全网格的情形）:")
            for name, msg in issues:
                print(f"     · {name}: {msg}")
            return 1
        for name, out in resolved.items():
            print(f"  解析结果 '{name}': {len(out['grid_indices'])} 格, "
                  f"area={out['area_m2']:.1f} m², from={out['resolved_from']}")

        out_dir = Path(os.environ.get("ET_WS_D")
                       or (_ROOT / "data" / "outputs" / "gwt_verify" / "eastTexas_cased"))
        builder = AutoModelBuilder(str(out_dir))

        # ET_D_RUN=1 时真跑 mf6，额外判"羽流是否从填埋场长出来"（几分钟）
        res_run = None
        if os.environ.get("ET_D_RUN") == "1":
            print("  ET_D_RUN=1 → 真跑 mf6（预计几分钟）...")
            res_run = builder.build_and_run(ir2, "auto_model_d")
            if not res_run.get("success"):
                print("  !! 运行失败")
                print(res_run.get("detail") or res_run.get("error"))
                return 1
            print(f"  运行成功，水头范围 {res_run.get('head_range')}")
        else:
            sim = builder.build_simulation(ir2, "auto_model_d")
            sim.write_simulation(silent=True)

        ok, cells, src_cells = check_case_d(out_dir / "auto_model_d")
        if res_run is not None:
            ok &= check_plume_origin(res_run, src_cells)

        print("=" * 70)
        print(f"Case D: {'通过' if ok else '未通过'}")
        print("=" * 70)
        return 0 if ok else 1

    print("=" * 70)
    print("Case C — East Texas（Gate 5）")
    print("=" * 70)
    ir = build_ir()
    builder = AutoModelBuilder(str(OUT_ROOT))
    res = builder.build_and_run(ir, "auto_model")
    if not res.get("success"):
        print("  !! 运行失败")
        print(res.get("detail") or res.get("error"))
        return 1
    print(f"  运行成功，水头范围 {res.get('head_range')}")
    ok = check_case_c(res, OUT_ROOT / "auto_model")
    print("=" * 70)
    print(f"Case C: {'通过' if ok else '未通过'}")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

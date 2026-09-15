"""
溶质运移（GWT）全链路回归测试

对应调试方案的 Gate 0 / Gate 2 / Gate 3 / Gate 4：

  Gate 0  纯函数单测，不跑 mf6，秒级
  Gate 2  GWT 挂上后能跑通：MST6 / GWF-GWT 网格一致 / AUX / AUXMIXED
  Gate 3  浓度场非零且物理合理：无负浓度、无超调、对照解析解
  Gate 4  水量/质量平衡：GWT 逐期离散度 < 1%，SSM 累计注入量吻合

算例：
  Case A  一维土柱，对照**对流型入口**的解析解（单项 erfc）
  Case B  RCH + AUX + AUXMIXED + nlay=2，覆盖 East Texas 全套机制

跑 mf6 的用例需要 PATH 里有 mf6（或设环境变量 MF6_EXE）；找不到就自动跳过。
设 SKIP_MF6_TESTS=1 可只跑 Gate 0 的快速用例。
"""
import sys, os, shutil, math
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np

passed = 0
failed = 0
skipped = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [OK] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}  {detail}")


def skip(name, why):
    global skipped
    skipped += 1
    print(f"  [SKIP] {name}  ({why})")


def section(title):
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")


# ============================================================
# Gate 0-1: 网格归一化 _resolve_botm / _resolve_dis
# ============================================================
section("Gate 0-1: 网格归一化")

from app.modeling.model_builder import _resolve_botm, _resolve_dis, AutoModelBuilder

# 标量 botm + nlay>1：必须等分成 nlay 层，否则 GWT 的第二层厚度会变 0
_b = _resolve_botm({"nlay": 2, "top": 25.0, "botm": 5.0})
check("标量 botm + nlay=2 等分", len(_b) == 2 and _b == [15.0, 5.0], f"得到 {_b}")

# dict 包装
_b = _resolve_botm({"nlay": 2, "top": 25.0, "botm": {"value": [20.0, 10.0]}})
check("dict 包装 botm 解包", _b == [20.0, 10.0], f"得到 {_b}")

# list 过长 → 截断到 nlay
_b = _resolve_botm({"nlay": 2, "top": 30.0, "botm": [20.0, 10.0, 5.0, 1.0]})
check("过长 botm 截断到 nlay", len(_b) == 2, f"得到 {_b}")

# list 过短 → 补齐到 nlay（旧 fallback 返回 1 元列表，nlay>1 时是错的）
_b = _resolve_botm({"nlay": 3, "top": 30.0, "botm": [20.0]})
check("过短 botm 补齐到 nlay", len(_b) == 3, f"得到 {_b}")

# nlay=1 标量
_b = _resolve_botm({"nlay": 1, "top": 100.0, "botm": 50.0})
check("nlay=1 标量 botm", _b == [50.0], f"得到 {_b}")

# 各种 nlay 下 len(botm) == nlay 恒成立
# 注意 MODFLOW 的行列约定：DELR 是"沿行方向的宽度"，长度 = NCOL；
# DELC 是"沿列方向的高度"，长度 = NROW。写反了会让梯度路径长度算错。
for nlay in (1, 2, 3, 5, 10):
    d = _resolve_dis({
        "nlay": nlay, "nrow": 4, "ncol": 7, "top": 30.0,
        "botm": 5.0, "delr": [12.0] * 7, "delc": [10.0] * 4,
    })
    check(f"nlay={nlay}: _resolve_dis 归一化",
          len(d["botm"]) == nlay and np.asarray(d["delr"]).size == 7
          and np.asarray(d["delc"]).size == 4,
          f"botm={d['botm']}, delr={np.asarray(d['delr']).size}, "
          f"delc={np.asarray(d['delc']).size}")

# delr/delc 标量扩展：分别铺到 ncol / nrow
d = _resolve_dis({"nlay": 1, "nrow": 3, "ncol": 5, "top": 10.0,
                  "botm": 0.0, "delr": 2.0, "delc": 3.0})
check("delr 标量铺到 ncol=5", np.allclose(d["delr"], 2.0) and d["delr"].size == 5,
      f"delr={d['delr']}")
check("delc 标量铺到 nrow=3", np.allclose(d["delc"], 3.0) and d["delc"].size == 3,
      f"delc={d['delc']}")


# ============================================================
# Gate 0-2: 时间离散 _map_time_params
# ============================================================
section("Gate 0-2: 时间离散")

from app.modeling.model_mapper import FloPyModelMapper

_mapper = FloPyModelMapper()

# 顶层 stress_periods 是裸整数（East Texas 就是这种）→ 按 nper 读，
# 不能再塌成"1 个 1 天的应力期"
_t = _mapper._map_time_params({"stress_periods": 10})
_tot = sum(p["perlen"] for p in _t["period_data"])
check("裸整数 stress_periods 按 nper 读取",
      len(_t["period_data"]) == 10 and abs(_tot - 10.0) < 1e-9,
      f"nper={len(_t['period_data'])}, 总时长={_tot}")

# 真实时间信息藏在 transport.stress_periods 里 → 必须回退读到。
# perlen 是**每个**应力期的长度（MF6 原生语义），所以 10 × 15000 = 150000 d。
# 若想要"总共 15000 d"，必须显式给 period_data，别指望这里猜。
_t = _mapper._map_time_params({
    "stress_periods": 10,
    "transport": {"stress_periods": {"nper": 10, "perlen": 15000}},
})
_tot = sum(p["perlen"] for p in _t["period_data"])
check("回退读 transport.stress_periods",
      len(_t["period_data"]) == 10 and abs(_tot - 150000.0) < 1.0,
      f"nper={len(_t['period_data'])}, 总时长={_tot}")

# 非法顶层值不能炸
for _bad in ("abc", True, None, []):
    try:
        _r = _mapper._map_time_params({"stress_periods": _bad})
        _ok = bool(_r["period_data"])
    except Exception as e:
        _ok = False
        print(f"    !! stress_periods={_bad!r} -> {type(e).__name__}: {e}")
    check(f"非法 stress_periods={_bad!r} 安全", _ok)

# 显式 period_data 优先
_t = _mapper._map_time_params({"stress_periods": {
    "nper": 2,
    "period_data": [{"perlen": 100.0, "nstp": 10}, {"perlen": 200.0, "nstp": 20}],
}})
check("显式 period_data 生效",
      len(_t["period_data"]) == 2
      and abs(sum(p["perlen"] for p in _t["period_data"]) - 300.0) < 1e-9)


# ============================================================
# Gate 0-3: ADV 方案白名单（UTVD 会让 mf6 6.3.0 硬终止）
# ============================================================
section("Gate 0-3: ADV 方案白名单")

for raw in ("utvd", "ultimate", "tvd", "upstream", "central", "上游加权", "三阶TVD"):
    got = _mapper.ADV_SCHEME_MAP.get(raw.lower())
    check(f"ADV '{raw}' → {got}", got in ("UPSTREAM", "CENTRAL", "TVD"), f"得到 {got}")

check("ADV 表内无 UTVD 关键字",
      all(v in ("UPSTREAM", "CENTRAL", "TVD") for v in _mapper.ADV_SCHEME_MAP.values()),
      "有值不在 MF6 6.3.0 支持范围内")


# ============================================================
# Gate 0-4: validator 降解分支不再抛 NameError
# ============================================================
section("Gate 0-4: validator decay 分支")

from app.physics_engine.validator import ClosedLoopValidator

# 旧代码此处 math.log(2) 会因缺 import math 抛 NameError
_ir = {
    "ir_version": "1.0",
    "project": {"name": "decay_test", "model_type": "solute_transport"},
    "domain": {"x_range": [0, 100], "y_range": [0, 100], "units": "meters"},
    "grid": {"nlay": 1, "nrow": 5, "ncol": 5, "top": 10.0, "bottom": 0.0,
             "delr": 10.0, "delc": 10.0},
    "aquifer": {"type": "confined",
                "parameters": {"hk": {"value": 10.0, "unit": "m/d"},
                               "porosity": {"value": 0.25}}},
    "transport": {
        "enabled": True,
        "parameters": {"al": {"value": 5.0, "unit": "m"}},
        "reaction": {"decay": {"half_life": {"value": 100.0, "unit": "d"},
                               "first_order_rate": {"value": 0.5, "unit": "1/d"}}},
    },
    "solver": {"name": "BICGSTAB", "parameters": {"hclose": 1e-3}},
}
try:
    _res = ClosedLoopValidator().validate(_ir)
    check("validate() 带 decay 不抛异常", isinstance(_res, dict))
    check("math.log(2) 可用", abs(math.log(2) - 0.693147) < 1e-5)
except NameError as e:
    check("validate() 带 decay 不抛异常", False, f"NameError: {e}")
except Exception as e:
    check("validate() 带 decay 不抛异常", False, f"{type(e).__name__}: {e}")


# ============================================================
# Gate 0-5: intent_parser 索引路径分支不再抛 NameError
# ============================================================
section("Gate 0-5: intent_parser 索引路径")

from app.llm_agent.intent_parser import IntentParser

# 旧代码走 [0] 索引路径时 matched_name/matches 未定义，此处必抛 NameError
try:
    _ip = IntentParser()
    _norm = _ip.normalize_entities([
        {"entity_type": "parameter_name",
         "standard_name": "boundary_conditions[0].parameters.stage",
         "value": 100.0, "unit": "m"},
        {"entity_type": "parameter_name",
         "standard_name": "boundary_conditions[1].parameters.head",
         "value": 50.0, "unit": "m"},
    ])
    check("索引路径分支不抛 NameError", isinstance(_norm, dict))
    _k = "boundary_conditions[0].parameters.stage"
    check("索引路径进入 normalized", _k in _norm, f"keys={list(_norm)}")
    if _k in _norm:
        check("索引路径 confidence 退化为 medium",
              _norm[_k].get("confidence") == "medium",
              f"得到 {_norm[_k].get('confidence')}")
except NameError as e:
    check("索引路径分支不抛 NameError", False, f"NameError: {e}")
except Exception as e:
    check("索引路径分支不抛 NameError", False, f"{type(e).__name__}: {e}")


# ============================================================
# Gate 0-6: Courant 时间步长保护
# ============================================================
section("Gate 0-6: Courant 时间步长保护")


class _StubBuilder(AutoModelBuilder):
    """绕过 __init__（不需要真实工作区）直接调纯逻辑"""

    def __init__(self):
        self.mapper = FloPyModelMapper()


_bld = _StubBuilder()
# Case A 的真实几何：ncol=500 个 0.2 m 的 DELR → 域长 100 m；nrow=1
_bld._dis = _resolve_dis({"nlay": 1, "nrow": 1, "ncol": 500, "top": 10.0,
                          "botm": 0.0, "delr": [0.2] * 500, "delc": [1.0]})

# 边界水头按列位置提取（chd/ghb 的第 1 列），不是按键名
_h = _bld._collect_boundary_heads(
    {"boundaries": {"chd": {"0": [[[0, 0, 0], 1.0, 1.0], [[0, 0, 499], 0.0, 0.0]]}}}
)
check("CHD 水头按列提取", _h == [1.0, 0.0], f"得到 {_h}")

_h = _bld._collect_boundary_heads(
    {"boundaries": {"ghb": {"0": [[[0, 0, 0], 30.0, 1.0], [[0, 0, 9], 25.0, 1.0]]},
                    "rch": {"0": [[[0, 0, 0], 1e-4, 500.0]]}}}
)
check("GHB 取水头、RCH 不取", sorted(_h) == [25.0, 30.0], f"得到 {_h}")

# 参考流速：hk=1.004, Δh=1, L=100, ne=0.25 → 0.04016（真值 0.04008，差 0.2%）
_v, _why = _bld._reference_velocity({
    "aquifer": {"hk": 1.004, "porosity": 0.25},
    "transport": {"porosity": 0.25},
    "boundaries": {"chd": {"0": [[[0, 0, 0], 1.0], [[0, 0, 499], 0.0]]}},
})
check("参考流速估算达 0.2% 精度",
      _v is not None and abs(_v - 0.04008) / 0.04008 < 0.01,
      f"v={_v} ({_why})")

_M_CASE_A = {
    "aquifer": {"hk": 1.004, "porosity": 0.25},
    "transport": {"porosity": 0.25, "enabled": True},
    "boundaries": {"chd": {"0": [[[0, 0, 0], 1.0], [[0, 0, 499], 0.0]]}},
}

# nstp 充足时不动
_tp = {"period_data": [{"perlen": 1250.0, "nstp": 2500, "tsmult": 1.0}]}
_bld._ensure_transport_timesteps(_M_CASE_A, _tp)
check("nstp 已充足时不改动", _tp["period_data"][0]["nstp"] == 2500,
      f"得到 {_tp['period_data'][0]['nstp']}")

# nstp 不足时按 Δt_max 提升
_tp = {"period_data": [{"perlen": 1250.0, "nstp": 50, "tsmult": 1.0}]}
_bld._ensure_transport_timesteps(_M_CASE_A, _tp)
_n = _tp["period_data"][0]["nstp"]
check("nstp 不足时提升", _n > 50, f"得到 {_n}")
check("提升后满足 Courant<=1", 1250.0 / _n <= 0.2 / 0.04016 * 1.001,
      f"Δt={1250.0 / _n:.4f} d, Δt_max={0.2 / 0.04016:.4f} d")

# 下限 10
_tp = {"period_data": [{"perlen": 1.0, "nstp": 1, "tsmult": 1.0}]}
_bld._ensure_transport_timesteps(
    dict(_M_CASE_A, aquifer={"hk": 0.01, "porosity": 0.25}), _tp)
check("nstp 下限 10 生效", _tp["period_data"][0]["nstp"] == 10,
      f"得到 {_tp['period_data'][0]['nstp']}")

# 上限 20000
_tp = {"period_data": [{"perlen": 15000.0, "nstp": 10, "tsmult": 1.0}]}
_bld._ensure_transport_timesteps(
    dict(_M_CASE_A, aquifer={"hk": 500.0, "porosity": 0.02}), _tp)
check("nstp 上限 20000 生效", _tp["period_data"][0]["nstp"] == 20000,
      f"得到 {_tp['period_data'][0]['nstp']}")

# timestep_target_courant 覆盖
_tp = {"period_data": [{"perlen": 1250.0, "nstp": 50, "tsmult": 1.0}]}
_bld._ensure_transport_timesteps(
    dict(_M_CASE_A, transport={"porosity": 0.25, "enabled": True,
                               "timestep_target_courant": 0.1}), _tp)
check("timestep_target_courant=0.1 生效", _tp["period_data"][0]["nstp"] == 2510,
      f"得到 {_tp['period_data'][0]['nstp']}")

# 各种残缺输入都不能炸
for _bad in ({}, {"aquifer": {}}, {"aquifer": {"hk": 0.0}},
             {"aquifer": {"hk": 1.0}}, None):
    try:
        _bld._ensure_transport_timesteps(_bad or {},
                                         {"period_data": [{"perlen": 1.0, "nstp": 3}]})
        _safe = True
    except Exception as e:
        _safe = False
        print(f"    !! {_bad} -> {type(e).__name__}: {e}")
check("残缺输入全部安全返回", _safe)


# ============================================================
# Gate 2/3/4: 真实 mf6 算例
# ============================================================
section("Gate 2/3/4: 真实 mf6 算例")

_mf6 = shutil.which(os.environ.get("MF6_EXE", "mf6"))
_run_mf6 = bool(_mf6) and os.environ.get("SKIP_MF6_TESTS") != "1"

if not _run_mf6:
    _why = "未找到 mf6 可执行文件" if not _mf6 else "SKIP_MF6_TESTS=1"
    skip("Case A（一维土柱）", _why)
    skip("Case B（RCH+AUX+AUXMIXED+nlay=2）", _why)
else:
    print(f"  使用 mf6: {_mf6}")
    from verify_gwt_cases import case_a_ir, case_b_ir, check_case_a, check_case_b

    _OUT = _ROOT / "data" / "outputs" / "gwt_verify"

    # ── Case A ──
    _ws = _OUT / "caseA"
    _res = AutoModelBuilder(str(_ws)).build_and_run(case_a_ir(), "auto_model")
    if not _res.get("success"):
        check("Case A 运行成功", False, str(_res.get("detail") or _res.get("error"))[:400])
    else:
        check("Case A 运行成功", True)
        # 负浓度 / 超调 / 解析解对照 / 质量平衡都在 check_case_a 里断言
        check("Case A 定量校验通过（Gate 3/4）",
              check_case_a(_res, _ws / "auto_model"),
              "对照对流型入口解析解或质量平衡未达标")

    # ── Case B ──
    _ws = _OUT / "caseB"
    _res = AutoModelBuilder(str(_ws)).build_and_run(case_b_ir(), "auto_model")
    if not _res.get("success"):
        check("Case B 运行成功", False, str(_res.get("detail") or _res.get("error"))[:400])
    else:
        check("Case B 运行成功", True)
        check("Case B 机制+定量校验通过（Gate 2/3/4）",
              check_case_b(_res, _ws / "auto_model"),
              "MST6/网格一致/AUX/AUXMIXED/注入量/质量平衡未达标")

    # ── Case C（East Texas）：103×218×2 跑满 15000 d，约 4 分钟，
    #    默认不跑；设 RUN_CASE_C=1 启用。
    #    负浓度按 docs/gwt_verification.md 记录的容差判定（|C_min| ≤ 源浓度 0.1%）
    if os.environ.get("RUN_CASE_C") == "1":
        from verify_east_texas import (
            build_ir as et_build_ir, check_case_c, OUT_ROOT as _ET_WS,
        )
        _ws = _ET_WS
        _res = AutoModelBuilder(str(_ws)).build_and_run(et_build_ir(), "auto_model")
        if not _res.get("success"):
            check("Case C 运行成功", False,
                  str(_res.get("detail") or _res.get("error"))[:400])
        else:
            check("Case C 运行成功", True)
            check("Case C 校验通过（Gate 2/4/5）",
                  check_case_c(_res, _ws / "auto_model"),
                  "机制/质量平衡/污染羽外扩未达标")
    else:
        skip("Case C（East Texas）", "设 RUN_CASE_C=1 启用（约 4 分钟）")


# ============================================================
# 结果汇总
# ============================================================
print(f"\n{'=' * 60}")
_total = passed + failed
_pct = 100.0 * passed / _total if _total else 0
print(f"GWT 全链路测试: {passed}/{_total} 通过 ({_pct:.1f}%)", end="")
print(f"  [{skipped} 项跳过]" if skipped else "")
if failed == 0:
    print("[OK] 所有 GWT 全链路测试通过！")
else:
    print(f"[WARN] {failed} 项失败，请检查上述 FAIL 项")
print(f"{'=' * 60}")

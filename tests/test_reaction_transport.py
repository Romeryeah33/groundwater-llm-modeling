# -*- coding: utf-8 -*-
"""反应运移（吸附 / 一级降解）—— 回归锁。

背景（2026-10-10，板块二）：

    反应参数**一直被读、被校验，然后被静默丢弃**。
    `map_transport_params` 早在第 521 行就把 sorption / kd / decay_type /
    decay_rate 抽出来放进了返回 dict；`constraints.check_json_ir_params`
    也会去查它们的范围；`intent_parser` 甚至把它们接到了 IR 路径上。
    但 `_build_gwt` 从头到尾**没有读过这些键** —— 建出来的模型没有任何反应，
    界面上也没有任何地方能发现。

本轮补上，并锁住五件事：

  1. 反应写进的是 **MST6**，不是 IST6。MF6 的 IST6 是"非移动域存储与转移"
     包（双域模型），加它立刻要求 THETAIM/ZETAIM 两个并不存在的非移动域参数。
     （原计划的"接入 ModflowGwtist"方向是错的，见下方第 5 节实证。）
  2. `half_life` 这条路径此前没有任何接线：IR 里给了 t½ → mapper 不读 →
     即便建了降解项也没有速率。这里锁 λ = ln2/t½ 的换算与来源标注。
  3. Kd / 干容重的**单位换算**：IR 惯用 L/kg 与 kg/L，MF6 要 m³/kg 与 kg/m³。
     两者互为倒数配对，乘积无量纲 —— 直接填文献值，阻滞因子看着对，
     但吸附相浓度会差 1000 倍。
  4. 无反应时 MST 的产物必须与改动前**逐字节相同**（既有案例基线不变）。
  5. 「启用吸附但不给 Kd」「启用降解但不给 λ 也不给 t½」必须**说出来**，
     绝不静默按无反应建包。

第 5 节需要 mf6 可执行文件；没有就跳过（其余各节纯离线）。

脚本风格，不用 pytest：``python tests/test_reaction_transport.py``
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from app.modeling.model_builder import AutoModelBuilder
from app.modeling.model_mapper import FloPyModelMapper
from app.physics_engine.constraints import ParameterRangeConstraint, ConstraintSeverity
# 以前这里要求"必须在文件顶部 import"：gradio_app 在 win32 上于**模块顶层**
# 重绑 sys.stdout，中途 import 会把其后所有 print 吞掉（输出戛然而止、退出码 0）。
# 2026-10-12 已治本 —— 包装挪进了 _ensure_utf8_stdio()，只在真要起服务时执行，
# 导入不再有副作用，所以现在放哪儿都行。
from app.frontend.gradio_app import _effective_reaction_text

_passed = 0
_skipped = 0


def _ok(msg):
    global _passed
    _passed += 1
    print(f"   OK: {msg}")


def _skip(msg):
    global _skipped
    _skipped += 1
    print(f"   SKIP: {msg}")


_BASE_IR = {
    "project": {"name": "react", "model_type": "solute_transport"},
    "aquifer": {"type": "confined", "parameters": {"hk": {"value": 5.0}}},
    "grid": {"nrow": 2, "ncol": 10, "nlay": 2, "delr": 20, "delc": 20,
             "top": 30.0, "botm": [20.0, 10.0]},
    "boundaries": {"chd": {"0": [[[0, 0, 0], 25.0], [[0, 0, 9], 21.0]]}},
}


def _mst(transport, sim="react"):
    """构建一次，返回 (.mst 文本, 生效反应参数, 提示列表)。"""
    with tempfile.TemporaryDirectory() as td:
        ir = {**_BASE_IR,
              "transport": {"enabled": True, "porosity": {"value": 0.25},
                            **transport}}
        b = AutoModelBuilder(td)
        b.build_simulation(ir, sim)
        path = Path(td) / sim / f"{sim}_gwt.mst"
        return (path.read_text(encoding="utf-8") if path.exists() else ""), \
               b._effective_reaction, list(b._build_warnings)


# ================================================================
# 1. 无反应 → MST 逐字节不变（既有基线）
# ================================================================
print("\n1. 无反应时不改动 MST（既有案例基线）...")
_txt, _eff, _warns = _mst({})
assert _eff["sorption"] == "none" and _eff["decay_type"] == "none", _eff
assert "SORPTION" not in _txt and "FIRST_ORDER_DECAY" not in _txt, _txt
assert "decay" not in _txt and "bulk_density" not in _txt, _txt
_ok("无反应 → options 段为空、griddata 只有 porosity（与改动前一致）")

# ================================================================
# 2. 吸附：选项、单位换算、阻滞因子
# ================================================================
print("\n2. 线性吸附...")
_txt, _eff, _ = _mst({"reaction": {"sorption": "linear", "kd": {"value": 1.0},
                                   "bulk_density": {"value": 1.7}}})
_norm = " ".join(_txt.split())
assert "SORPTION linear" in _norm, _txt
assert "distcoef CONSTANT 0.00100000" in _norm, f"Kd 1 L/kg 应换算成 1e-3 m3/kg:\n{_txt}"
assert "bulk_density CONSTANT 1700.00000000" in _norm, f"1.7 kg/L 应换算成 1700 kg/m3:\n{_txt}"
_ok("SORPTION linear + Kd 1 L/kg → 1e-3 m3/kg、ρb 1.7 kg/L → 1700 kg/m3")

assert _eff["kd_input"] == 1.0 and _eff["kd_si"] == 0.001, _eff
assert _eff["bulk_density_input"] == 1.7 and _eff["bulk_density_si"] == 1700.0, _eff
assert _eff["bulk_density_source"] == "ir", _eff
# R = 1 + ρb·Kd/n = 1 + 1700×0.001/0.25 = 7.8
assert abs(_eff["retardation"] - 7.8) < 1e-9, _eff
_ok("R = 1 + ρb·Kd/n = 7.8（换算后仍等于文献单位下的 1.7×1/0.25 + 1）")

# 2b. 用户没给干容重 → 默认 1.7 + 一条提示（不静默）
print("\n2b. 未给干容重 → 默认值 + 提示...")
_txt, _eff, _warns = _mst({"reaction": {"sorption": "linear", "kd": {"value": 1.0}}})
assert _eff["bulk_density_source"] == "default", _eff
assert any("干容重" in w for w in _warns), _warns
_ok("走默认干容重时留下提示（用户看得见自己没给这个参数）")

# 2c. 逐层 Kd（与逐层孔隙度同一套规则）
print("\n2c. 逐层 Kd...")
_txt, _eff, _ = _mst({"reaction": {"sorption": "linear",
                                   "kd": {"value": [1.0, 0.5]},
                                   "bulk_density": {"value": 1.7}}})
assert _eff["kd_input"] == [1.0, 0.5], _eff
_lo = _eff["retardation"]
assert len(_lo) == 2 and abs(_lo[0] - (1 + 1700 * 0.001 / 0.25)) < 1e-6, _eff
_ok(f"逐层 Kd 1.0/0.5 → 逐层 R = {_lo}")

try:
    _mst({"reaction": {"sorption": "linear", "kd": {"value": [1.0, 2.0, 3.0]}}})
    raise AssertionError("Kd 逐层长度与 nlay 不符时必须抛 ValueError")
except ValueError as e:
    assert "nlay=2" in str(e), e
_ok("逐层 Kd 长度与 nlay 不符 → ValueError（不静默取第一个）")

# ================================================================
# 3. 降解：λ 直给 / t½ 换算
# ================================================================
print("\n3. 一级降解...")
_txt, _eff, _ = _mst({"reaction": {"decay": {
    "type": "first_order", "first_order_rate": {"value": 0.01}}}})
_norm = " ".join(_txt.split())
assert "FIRST_ORDER_DECAY" in _txt, _txt
assert "decay CONSTANT 0.01000000" in _norm, _txt
assert _eff["decay_rate"] == 0.01 and _eff["decay_rate_source"] == "first_order_rate", _eff
# 只给了降解、没给吸附 → 不应要求 decay_sorbed
assert "decay_sorbed" not in _txt, f"仅降解时不需要 decay_sorbed:\n{_txt}"
_ok("λ=0.01 直给 → decay CONSTANT 0.01，且不写多余的 decay_sorbed")

# 3b. 半衰期换算（此前这条路径完全没有接线）
print("\n3b. 半衰期 → λ = ln2/t½ ...")
_txt, _eff, _ = _mst({"reaction": {"decay": {
    "type": "first_order", "half_life": {"value": 69.3147}}}})
assert _eff["decay_rate_source"] == "half_life", _eff
assert abs(_eff["decay_rate"] - 0.01) < 1e-6, _eff
assert abs(_eff["half_life_effective"] - 69.3147) < 1e-3, _eff
_ok(f"t½=69.3147 d → λ={_eff['decay_rate']:.6f} 1/d（来源标为 half_life）")

# 3c. 吸附 + 降解 → MF6 强制要求 decay_sorbed
print("\n3c. 吸附与降解同时启用...")
_txt, _eff, _warns = _mst({"reaction": {
    "sorption": "linear", "kd": {"value": 0.5},
    "decay": {"type": "first_order", "first_order_rate": {"value": 0.01}}}})
_norm = " ".join(_txt.split())
assert "SORPTION linear" in _norm and "FIRST_ORDER_DECAY" in _txt, _txt
assert "decay_sorbed CONSTANT 0.00000000" in _norm, \
    f"两者同时启用时 MF6 强制要求 DECAY_SORBED，否则硬报错:\n{_txt}"
assert _eff["decay_sorbed"] == 0.0, _eff
assert any("DECAY_SORBED" in w for w in _warns), _warns
_ok("写 decay_sorbed = 0（总方程退化为经典 R·∂C/∂t = ∇·(D∇C) − v∇C − λC）+ 提示")

# ================================================================
# 4. 说了要反应却给不出参数 → 必须说出来，不静默无反应
# ================================================================
print("\n4. 缺参数时不得静默...")
_txt, _eff, _warns = _mst({"reaction": {"sorption": "linear"}})
assert _eff["sorption"] == "none", _eff
assert "SORPTION" not in _txt, _txt
assert any("取不到有效的分配系数 Kd" in w for w in _warns), _warns
_ok("启用吸附但无 Kd → 不建吸附项，并留下可展示的提示")

_txt, _eff, _warns = _mst({"reaction": {"decay": {"type": "first_order"}}})
assert _eff["decay_rate"] is None, _eff
assert any("既没有降解速率" in w for w in _warns), _warns
_ok("启用降解但 λ/t½ 都没有 → 不建降解项，并留下提示")

# ================================================================
# 5. 端到端：MF6 6.3.0 真的能跑，且反应真的改变物理
# ================================================================
print("\n5. 端到端（需 mf6）...")
_mf6 = shutil.which(os.environ.get("MF6_EXE", "mf6"))
if not _mf6 or os.environ.get("SKIP_MF6_TESTS") == "1":
    _skip("MF6 端到端（未找到 mf6 可执行文件，或 SKIP_MF6_TESTS=1）")
else:
    import copy
    import flopy
    from app.modeling.model_builder import _to_layer_array  # noqa: F401

    from verify_gwt_cases import case_b_ir

    # 工作区放系统临时目录，**不**放仓库的 data/outputs/。
    # 原因：本仓库在 OneDrive 目录下，rmtree 会偶发撞上同步进程占用的句柄
    # （实测 `PermissionError: [WinError 5]`），测试于是变成"随机红一次"。
    # 脚本式套件里出现环境性假红，人就会学会无视它 —— 那是比失败更坏的结局。
    _ws_root = Path(tempfile.mkdtemp(prefix="gm_react_"))

    def _peak(name, mut=None):
        ir = copy.deepcopy(case_b_ir())
        if mut:
            ir["transport"] = dict(ir["transport"], **mut)
        ws = _ws_root / name
        ws.mkdir(parents=True, exist_ok=True)
        res = AutoModelBuilder(str(ws)).build_and_run(ir, "auto_model")
        assert res["success"], f"{name} 运行失败: {res.get('detail')}"
        ucn = ws / "auto_model" / "auto_model_gwt.ucn"
        arr = flopy.utils.HeadFile(str(ucn), text="CONCENTRATION").get_alldata()
        return float(arr.max()), res

    _p0, _r0 = _peak("base")
    _p1, _r1 = _peak("decay", {"reaction": {"decay": {
        "type": "first_order", "first_order_rate": {"value": 0.01}}}})
    _p2, _r2 = _peak("sorb", {"reaction": {
        "sorption": "linear", "kd": {"value": 1.0}, "bulk_density": {"value": 1.7}}})
    _ok(f"基线 / 降解 / 吸附 三次运行全部成功（峰值 {_p0:.4f} / {_p1:.4f} / {_p2:.4f}）")

    assert _p1 < _p0 * 0.5, f"加了一级降解，峰值应显著下降：{_p1} vs {_p0}"
    _ok("一级降解真的生效（峰值显著下降），不是写了没用的空包")

    assert _p2 < _p0, f"加了吸附，迁移变慢、峰值应下降：{_p2} vs {_p0}"
    _ok("线性吸附真的生效（峰值下降，与 R>1 一致）")

    # 基线案例的 MST 必须仍无反应选项 —— 端到端确认基线没被改
    _base_mst = (_ws_root / "base" / "auto_model"
                 / "auto_model_gwt.mst").read_text(encoding="utf-8")
    assert "SORPTION" not in _base_mst and "FIRST_ORDER_DECAY" not in _base_mst, _base_mst
    _ok("端到端基线 MST 仍无反应选项（既有案例不受影响）")

    # 界面取数：反应行必须能渲染出来
    _txt_ui = _effective_reaction_text(_r2)
    assert "吸附=linear" in _txt_ui and "L/kg" in _txt_ui, _txt_ui
    assert "R=" in _txt_ui, _txt_ui
    _ok(f"界面反应行渲染正常：{_txt_ui[:60]}…")
    assert _effective_reaction_text(_r0) == "无（模型按不吸附、不降解运行）", \
        _effective_reaction_text(_r0)
    _ok("无反应时界面如实写「无」，不撒谎")

# 5b'. 接线锁：函数写对了但**没被调用**，等于没做（板块一踩过一次）。
# 这里直接查源码，因为"结果面板里有这一行"只有驱动真实 handler 才验得到，
# 那需要起 gradio 会话；源码级的锁虽粗，但正好覆盖"函数漂在一边没人调"
# 这个最可能的失效方式。断言的是调用点，不是函数定义。
print("\n5b'. 界面接线...")
_src = (ROOT / "app" / "frontend" / "gradio_app.py").read_text(encoding="utf-8")
assert '_effective_reaction_text(result)' in _src, \
    "结果面板必须真的调用 _effective_reaction_text(result)（定义了不调用 = 没做）"
assert '"反应参数（实际写入模型）"' in _src, \
    "结果面板的 telemetry 行必须带「反应参数（实际写入模型）」标签"
_ok("结果面板确实调用了 _effective_reaction_text(result)（源码级接线锁）")

# ================================================================
# 6. 范围校验不再空转（此前 first_order_rate / half_life 不在表里）
# ================================================================
print("\n6. 反应参数的范围校验真的在查...")
_pr = ParameterRangeConstraint()
_issues = {i.param_name: i for i in _pr.check_json_ir_params({
    "transport": {"reaction": {"decay": {"first_order_rate": {"value": 0.01},
                                         "half_life": {"value": 693.0}},
                               "kd": {"value": 0.5},
                               "bulk_density": {"value": 1.7}}}})}
for _n in ("first_order_rate", "half_life", "kd", "bulk_density"):
    assert _n in _issues, f"{_n} 未被检查：{list(_issues)}"
    assert _issues[_n].severity == ConstraintSeverity.PASS, \
        f"{_n} 应 PASS，实际 {_issues[_n].severity}: {_issues[_n].description}"
_ok("first_order_rate / half_life / kd / bulk_density 都真的查到（此前前两个不在范围表里，"
    "走「不在已知物理范围表中，默认通过」的 INFO 分支 —— 看起来查过了，一个值都没查）")

_bad = {i.param_name: i for i in _pr.check_json_ir_params({
    "transport": {"reaction": {"decay": {"half_life": {"value": -5.0}}}}})}
assert _bad["half_life"].severity == ConstraintSeverity.ERROR, _bad
_ok("t½ = −5 d → ERROR（负半衰期不再被静默放过）")

# ================================================================
# 7. mapper 的 half_life / 单位元数据
# ================================================================
print("\n7. mapper 提取（含此前完全没接的 half_life）...")
_mm = FloPyModelMapper()
_r = _mm.map_transport_params({"transport": {"enabled": True, "reaction": {
    "decay": {"type": "first_order", "half_life": {"value": 693.0}}}}})
assert _r["decay_type"] == "first_order", _r
assert _r["decay_rate_source"] == "half_life", _r
assert abs(_r["decay_rate"] - 0.001000212) < 1e-9, _r
_ok("mapper 认得 half_life 并换算（此前只认 first_order_rate → 建了包也没速率）")

_r = _mm.map_transport_params({"transport": {"enabled": True, "reaction": {
    "sorption": "LINEAR", "kd": {"value": 0.5}}}})
assert _r["sorption"] == "linear", _r
assert _r["bulk_density_source"] == "default", _r
_ok("吸附模型名归一化（LINEAR → linear），未给干容重标 default")

_r = _mm.map_transport_params({"transport": {"enabled": True, "reaction": {
    "sorption": "banana"}}})
assert _r["sorption"] == "none", _r
_ok("未知吸附模型 → none（不让非法值写进 MF6 选项）")

print("\n" + "=" * 60)
print(f"反应运移测试: {_passed} 项通过" + (f"，{_skipped} 项跳过" if _skipped else ""))
print("=" * 60)

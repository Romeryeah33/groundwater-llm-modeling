# -*- coding: utf-8 -*-
"""逐层孔隙度 / 弥散度支持 —— 回归锁。

背景（East Texas 案例，2026-10-10）：用户提示词逐层给了参数

    第一层：孔隙度0.3，纵向弥散度20m，横向比0.2，垂向比0.1
    第二层：孔隙度0.2，纵向弥散度20m，横向比0.2，垂向比0.1

程序只取**一个**值：``porosity CONSTANT 0.20000000``、``alh CONSTANT 40.0``，
分层信息被静默丢弃 —— 第 1 层真实孔隙度 0.3 却按 0.2 算，运移速度偏快。

本文件锁四件事：
  1. ``_to_layer_array`` 的标量透传 / 逐层展开 / 长度校验；
  2. 真正写进 ``.mst`` / ``.dsp`` 的是逐层数组，**且标量输入仍然是 CONSTANT**
     （不能为了新功能把既有案例的基线从 CONSTANT 改成 ARRAY）；
  3. 参数溯源（``al_source`` / ``porosity_source``）如实标注，供界面核对；
  4. 逐层值**逐个**参与物理范围校验，而不是整列被跳过。

脚本风格，不用 pytest：``python tests/test_per_layer_transport.py``
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from app.modeling.model_builder import AutoModelBuilder, _to_layer_array
from app.modeling.model_mapper import FloPyModelMapper
from app.physics_engine.constraints import ParameterRangeConstraint, ConstraintSeverity
# 这个 import 必须在顶部：gradio_app 在 win32 上会重绑 sys.stdout，
# 中途 import 会把其后所有 print 吞掉（表现为"输出戛然而止但退出码 0"）。
from app.frontend.gradio_app import _extract_process_params

_passed = 0


def _ok(msg):
    global _passed
    _passed += 1
    print(f"   OK: {msg}")


# ================================================================
# 1. _to_layer_array
# ================================================================
print("\n1. _to_layer_array 归一化...")

assert _to_layer_array(0.2, 2, 3, 4, "n") == 0.2
assert isinstance(_to_layer_array(0.2, 2, 3, 4, "n"), float)
_ok("标量原样透传（保持 float，不展成数组）")

_arr = _to_layer_array([0.3, 0.2], 2, 3, 4, "孔隙度")
assert _arr.shape == (2, 3, 4), _arr.shape
assert np.allclose(_arr[0], 0.3) and np.allclose(_arr[1], 0.2)
_ok("长度 = nlay 的列表展开成 (nlay,nrow,ncol)，每层铺满")

_wrapped = _to_layer_array({"value": [0.3, 0.2], "unit": "dimensionless"}, 2, 3, 4, "孔隙度")
assert np.allclose(_wrapped[:, 0, 0], [0.3, 0.2])
_ok("LLM 常见的 {'value': [...]} 包装能解包")

_nested = _to_layer_array([[0.3], [0.2]], 2, 3, 4, "孔隙度")
assert np.allclose(_nested[:, 0, 0], [0.3, 0.2])
_ok("只给一列的嵌套列表也能展平")

assert _to_layer_array([0.25], 2, 3, 4, "n") == 0.25
_ok("长度为 1 的列表退化为标量（不是逐层）")

try:
    _to_layer_array([0.1, 0.2, 0.3], 2, 3, 4, "孔隙度")
    raise AssertionError("长度与 nlay 不符时必须抛 ValueError")
except ValueError as e:
    assert "nlay=2" in str(e), e
_ok("长度与 nlay 不符 → ValueError（消息带 nlay）")

# ================================================================
# 2. 写进 .mst / .dsp 的到底是什么
# ================================================================
print("\n2. 实际写入 MST / DSP 包...")

_BASE_IR = {
    "project": {"name": "per_layer", "model_type": "solute_transport"},
    "aquifer": {"type": "confined", "parameters": {"hk": {"value": 5.5}}},
    "grid": {"nrow": 2, "ncol": 10, "nlay": 2, "delr": 20, "delc": 20,
             "top": 30.0, "botm": [20.0, 10.0]},
    "boundaries": {"chd": {"0": [[[0, 0, 0], 25.0], [[0, 0, 9], 21.0]]}},
    "transport": {
        "enabled": True,
        "dispersion": {"al": {"value": [20.0, 20.0], "unit": "m"},
                       "trpt": {"value": 0.2}, "trpv": {"value": 0.1}},
        "porosity": {"value": [0.3, 0.2], "unit": "dimensionless"},
        "initial_concentration": 0.0,
    },
}

with tempfile.TemporaryDirectory() as _td:
    _b = AutoModelBuilder(_td)
    _b.build_simulation(_BASE_IR, "per_layer")
    _mst = (_b.workspace / "per_layer" / "per_layer_gwt.mst").read_text(encoding="utf-8")
    _dsp = (_b.workspace / "per_layer" / "per_layer_gwt.dsp").read_text(encoding="utf-8")

    assert "porosity  LAYERED" in _mst, f"逐层孔隙度应写 LAYERED，实际:\n{_mst}"
    assert "CONSTANT" not in _mst, f"逐层孔隙度不该再是 CONSTANT:\n{_mst}"
    _ok("逐层孔隙度 → porosity LAYERED（不再是 CONSTANT 0.2）")

    assert "alh  LAYERED" in _dsp, f"逐层弥散度应写 LAYERED，实际:\n{_dsp}"
    # diffc 仍是标量、理应保持 CONSTANT，所以只针对 alh 断言
    assert "alh  CONSTANT" not in _dsp, f"逐层弥散度不该再是 CONSTANT:\n{_dsp}"
    assert "ath1  LAYERED" in _dsp and "atv  LAYERED" in _dsp, _dsp
    _ok("逐层弥散度 → alh/ath1/atv LAYERED（diffc 仍为 CONSTANT，标量不被误展）")

    # 逐层值必须真的按层落位。走**文件文本**而不是 flopy 回读：
    # MFSimulation.load 在本案例的小模型上会因某个空标量包抛 IndexError，
    # 而且它会占住文件句柄导致 Windows 下临时目录删不掉。
    _mst_clean = _mst.replace(" ", "").replace("\t", "")
    assert "0.30000000" in _mst_clean and "0.20000000" in _mst_clean, _mst
    assert _mst_clean.index("0.30000000") < _mst_clean.index("0.20000000"), \
        "第 1 层（0.3）必须先于第 2 层（0.2）写出 —— 顺序反了就是把层写倒了"
    _ok("文件里第 1 层 0.3、第 2 层 0.2，顺序正确（分层没有被压平、没写倒）")

    _gwt = _b._gwt
    _por = np.asarray(_gwt.mst.porosity.array)
    assert _por.shape == (2, 2, 10), _por.shape
    assert np.allclose(_por[0], 0.3) and np.allclose(_por[1], 0.2), _por[:, 0, 0]
    _ok("MST 包内存值：形状 (nlay,nrow,ncol) = (2,2,10)，逐层 0.3 / 0.2")

    _alh = np.asarray(_gwt.dsp.alh.array)
    assert _alh.shape == (2, 2, 10) and np.allclose(_alh[1], 20.0), _alh[:, 0, 0]
    _ok("DSP 包内存值：alh 逐层 = 20 m")
print(f"   → 逐层提示: {[w[:40] for w in _b._build_warnings]}")
assert _b._effective_transport["porosity"] == [0.3, 0.2], _b._effective_transport
assert _b._effective_transport["al"] == [20.0, 20.0], _b._effective_transport
_ok("transport_effective 带回逐层值，供界面核对")

# 2b. 标量输入必须**保持** CONSTANT —— 否则既有案例基线被改掉
print("\n2b. 标量输入仍写 CONSTANT（不改变既有基线）...")
with tempfile.TemporaryDirectory() as _td:
    _b2 = AutoModelBuilder(_td)
    _b2.build_simulation(
        {**_BASE_IR,
         "transport": {"enabled": True,
                       "dispersion": {"al": {"value": 40.0}},
                       "porosity": {"value": 0.2},
                       "initial_concentration": 0.0}},
        "scalar_probe")
    _mst2 = (_b2.workspace / "scalar_probe" / "scalar_probe_gwt.mst").read_text(encoding="utf-8")
    _dsp2 = (_b2.workspace / "scalar_probe" / "scalar_probe_gwt.dsp").read_text(encoding="utf-8")
    _norm = lambda s: " ".join(s.split())
    assert "porosity CONSTANT 0.20000000" in _norm(_mst2), _mst2
    assert "alh CONSTANT 40.0" in _norm(_dsp2), _dsp2
    _ok("标量 → 仍是 CONSTANT（East Texas 等既有基线不受影响）")

# ================================================================
# 3. 参数溯源
# ================================================================
print("\n3. al / porosity 来源标注...")
_mapper = FloPyModelMapper()

_r = _mapper.map_transport_params({"transport": {"enabled": True, "al": 20.0},
                            "grid": {"delr": 20, "delc": 20}})
assert _r["al"] == 20.0 and _r["al_source"] == "ir", _r
_ok("显式给了 al → al_source='ir'")

_r = _mapper.map_transport_params({"transport": {"enabled": True},
                            "grid": {"delr": 20, "delc": 20}})
assert _r["al"] == 40.0 and _r["al_source"] == "grid_default", _r
_ok("未给 al → al_source='grid_default'，值 = 2Δx = 40（Péclet≈0.5 的稳定取法）")

# ================================================================
# 4. 逐层值逐个参与范围校验
# ================================================================
print("\n4. 逐层参数的范围校验...")
_pr = ParameterRangeConstraint()

_issues = _pr.check_json_ir_params({
    "aquifer": {"parameters": {"porosity": {"value": [0.3, 0.2]}}},
    "transport": {"dispersion": {"al": {"value": [20.0, 20.0]}}},
})
_names = [i.param_name for i in _issues]
assert "porosity[层1]" in _names and "porosity[层2]" in _names, _names
_ok("逐层孔隙度拆成 porosity[层1] / porosity[层2] 两条结果")

_por1 = [i for i in _issues if i.param_name == "porosity[层1]"][0]
assert _por1.severity == ConstraintSeverity.PASS, _por1
_ok("0.3 落在物理范围 (0.05, 0.5) 内 → PASS")

_bad = _pr.check_json_ir_params({
    "aquifer": {"parameters": {"porosity": {"value": [0.3, 0.9]}}},
})
_bad2 = [i for i in _bad if i.param_name == "porosity[层2]"][0]
assert _bad2.severity == ConstraintSeverity.ERROR, _bad2
assert _bad2.current_value == 0.9, _bad2
_ok("第 2 层 0.9 超上限 → ERROR（修复前整列被跳过、一个值都不查）")

# ================================================================
# 5. 过程贡献图的参数提取
# ================================================================
print("\n5. _extract_process_params 取数与逐层取均值...")
# (import 已提到顶部)

_p = _extract_process_params({
    "transport": {"dispersion": {"al": {"value": [20.0, 20.0]},
                                 "trpt": {"value": 0.2}}},
    "aquifer": {"parameters": {"porosity": {"value": [0.3, 0.2]}}},
})
assert _p["al"] == 20.0, _p
_ok("al 从 transport.dispersion 取到（此前只读 transport 顶层 → 永远落 10.0 默认值）")

assert abs(_p["porosity"] - 0.25) < 1e-12, _p
_ok("逐层孔隙度取均值 (0.3+0.2)/2 = 0.25")

_p2 = _extract_process_params({"transport": {"al": {"value": 7.0}}})
assert _p2["al"] == 7.0, _p2
_ok("顶层写法仍然兜底认得（两种 IR 写法都收）")

print("\n" + "=" * 60)
print(f"逐层运移参数测试: {_passed} 项全部通过")
print("=" * 60)

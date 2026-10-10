"""验证物理约束引擎模块"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from app.physics_engine.constraints import (
    ConstraintSeverity,
    MassBalanceConstraint,
    GoverningEquationConstraint,
    ParameterRangeConstraint,
    NumericalStabilityConstraint,
    UnitConsistencyConstraint,
)
from app.physics_engine.validator import ClosedLoopValidator, quick_validate
from app.physics_engine.reaction_pathway import ReactionPathwayValidator

# Test 1: Mass balance
print("1. Testing MassBalanceConstraint...")
r = MassBalanceConstraint.check_water_balance_pre(100, 100, 0, 0)
assert r.is_valid
print(f"   ✅ Water balance: error={r.current_value:.6f}")

# Test 2: Darcy velocity
print("2. Testing GoverningEquationConstraint...")
r = GoverningEquationConstraint.check_darcy_velocity(K=15, gradient=0.005, n=0.25)
assert 0 < r.current_value < 100
print(f"   ✅ Darcy velocity: v_pore={r.current_value:.4f} m/d")

# Test 3: Parameter range
print("3. Testing ParameterRangeConstraint...")
checker = ParameterRangeConstraint()
r = checker.check_param("hk", 15.0)
assert r.is_valid
print(f"   ✅ Valid param: {r.description}")
r = checker.check_param("hk", 1e10)
assert not r.is_valid
print(f"   ✅ Invalid param detected: {r.suggestion}")

# Test 4: Numerical stability
print("4. Testing NumericalStabilityConstraint...")
r = NumericalStabilityConstraint.check_courant_number(
    v_darcy=15*0.005, dt=10.0, dx=20.0, n=0.25
)
print(f"   ✅ Courant: Cr={r.current_value:.3f}, valid={r.is_valid}")

nq = NumericalStabilityConstraint.check_grid_quality(50, 40)
print(f"   ✅ Grid quality: {len(nq)} issues")

# Test 5: Validator quick_validate
print("5. Testing ClosedLoopValidator (quick)...")
test_ir = {
    "ir_version": "1.0",
    "project": {"name": "test", "model_type": "groundwater_flow"},
    "domain": {"x_range": [0, 1000], "y_range": [0, 800]},
    "grid": {"nrow": 50, "ncol": 40, "nlay": 1, "top": 100, "bottom": 50},
    "aquifer": {"type": "confined", "parameters": {
        "hk": {"value": 15, "unit": "m/d"},
        "porosity": {"value": 0.25, "unit": "dimensionless"},
    }},
    "boundary_conditions": [
        {"type": "chd", "name": "west", "parameters": {"head": 100}},
        {"type": "chd", "name": "east", "parameters": {"head": 95}},
    ],
}
result = quick_validate(test_ir)
print(f"   Valid: {result['valid']}, Errors: {result['error_count']}, Warnings: {result['warning_count']}")

# Test 6: Reaction pathway
print("6. Testing ReactionPathwayValidator...")
rv = ReactionPathwayValidator()
cat = rv.identify_contaminant_category("PCE")
assert cat == "chlorinated_ethenes"
print(f"   ✅ PCE category: {cat}")

sug = rv.suggest_kinetic_params("TCE", redox_condition="anaerobic")
assert sug is not None
print(f"   ✅ TCE rate sug: {sug['first_order_rate']['value']} {sug['first_order_rate']['unit']}")

# Test invalid params
print("7. Testing intentional violations...")
bad_ir = test_ir.copy()
bad_ir["aquifer"]["parameters"]["hk"]["value"] = 1e10
result = quick_validate(bad_ir)
assert result["error_count"] > 0
print(f"   ✅ hk=1e10 detected: {result['error_count']} error(s)")

# Test 8: 渗透系数量纲校验（hk/vka 必须是 m/d，不能是裸长度）
# 缺陷态：vka = {"value": 20, "unit": "m"} —— 单位漏了时间分母，20 m 与 20 m/d
# 会被混为一谈。此检查此前只在 docstring 里承诺过，函数体里没有实现。
print("8. Testing penetration-coefficient unit check...")
uc = UnitConsistencyConstraint()


def _unit_issues(ir):
    return [r for r in uc.check_unit_consistency(ir)
            if r.param_name.startswith("aquifer.parameters.")]


assert len(_unit_issues(
    {"aquifer": {"parameters": {"vka": {"value": 20, "unit": "m"}}}})) == 1, \
    "vka 单位写成裸长度 m 应报警"
print("   ✅ vka unit 'm' flagged")
assert _unit_issues(
    {"aquifer": {"parameters": {"vka": {"value": 20, "unit": "m/d"}}}}) == [], \
    "vka 单位 m/d 不应报警"
assert _unit_issues(
    {"aquifer": {"parameters": {"hk": {"value": 60, "unit": "ft/d"}}}}) == [], \
    "ft/d 也是合法的流速量纲"
assert _unit_issues({"aquifer": {"parameters": {"hk": {"value": 10}}}}) == [], \
    "没写单位就不判（无法判断，非错误）"
assert _unit_issues({"aquifer": {"parameters": {"hk": 1.004}}}) == [], \
    "裸数字（非 dict）应跳过而非抛异常"
assert len(_unit_issues(
    {"aquifer": {"parameters": {"vka": {"value": 20, "unit": "-"}}}})) == 1, \
    "无量纲记号对渗透系数是错的，应报警"
print("   ✅ m/d & ft/d pass, bare length & dimensionless flagged")

# Test 9: 裸数字参数不能让 _check_physics 崩溃
# 修复前 `aqua_params.get(k, {}).get("value")` 对 float 会抛
# AttributeError: 'float' object has no attribute 'get'。LLM 直接吐数字很常见。
print("9. Testing bare-number parameter robustness...")
_v = ClosedLoopValidator.__new__(ClosedLoopValidator)
for _tag, _ap in [("bare hk/vka", {"hk": 1.004, "vka": 0.15}),
                  ("bare porosity", {"hk": {"value": 10}, "porosity": 0.3}),
                  ("字符串 hk", {"hk": "10.0"})]:
    _ir = {"aquifer": {"parameters": _ap},
           "domain": {"x_range": [0, 1000], "y_range": [0, 1000]}}
    _out = _v._check_physics(_ir)   # 修复前此处抛 AttributeError
    assert isinstance(_out, list)
print("   ✅ bare float / str / dict all accepted without AttributeError")

# Test 10: hani 与 vka 同时给出要被讲出来（原先 model_mapper 里是一句裸 pass）
print("10. Testing hani/vka conflict is surfaced...")


def _phys(ap):
    return _v._check_physics(
        {"aquifer": {"parameters": ap},
         "domain": {"x_range": [0, 1000], "y_range": [0, 1000]}})


def _conflict(ap):
    return [r for r in _phys(ap) if r.param_name == "anisotropy_conflict"]


assert _conflict({"hk": {"value": 15}, "hani": {"value": 3},
                  "vka": {"value": 5}}) != [], "两者同时给出必须报警"
_m = _conflict({"hk": {"value": 15}, "hani": {"value": 3},
                "vka": {"value": 5}})[0].description
assert "hani" in _m and "vka" in _m and "被忽略" in _m, f"消息没说清谁被忽略: {_m}"
print(f"   ✅ 冲突已报: {_m}")
# hk/hani 推得 Kv=5.0，与 vka=5.0 一致 → 不算数值矛盾，但仍提示重复
assert "相差" not in _conflict({"hk": {"value": 15}, "hani": {"value": 3},
                                "vka": {"value": 5}})[0].description
# hk/hani 推得 Kv=5.0，vka=50 → 相差 900%，消息要带上差距
_m2 = _conflict({"hk": {"value": 15}, "hani": {"value": 3},
                 "vka": {"value": 50}})[0].description
assert "相差" in _m2, f"数值矛盾时应给出差距: {_m2}"
print(f"   ✅ 数值矛盾带差距: {_m2}")
# 只给其中一个 → 无冲突
assert _conflict({"hk": {"value": 15}, "vka": {"value": 5}}) == []
assert _conflict({"hk": {"value": 15}, "hani": {"value": 3}}) == []
print("   ✅ 单给其一不报警")

print("\n✅ All physics engine tests passed!")

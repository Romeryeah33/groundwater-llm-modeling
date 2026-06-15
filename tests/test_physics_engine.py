"""验证物理约束引擎模块"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.physics_engine.constraints import (
    ConstraintSeverity,
    MassBalanceConstraint,
    GoverningEquationConstraint,
    ParameterRangeConstraint,
    NumericalStabilityConstraint,
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

print("\n✅ All physics engine tests passed!")

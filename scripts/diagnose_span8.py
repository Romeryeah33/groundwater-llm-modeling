"""
Span 8 诊断脚本：SolverMonitor 错误分类与自适应策略验证

测试 _classify_error, ADAPTIVE_STRATEGIES 覆盖, _apply_strategy,
MAX_RETRIES 上限, 以及 monitor_run 的基本流程
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
from app.modeling.solver_monitor import SolverMonitor, ErrorType

sm = SolverMonitor()
passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name}  FAILED  {detail}")

print("=" * 60)
print("Span 8 诊断: SolverMonitor 自适应策略")
print("=" * 60)

# ============================================================
# S8.1: 错误分类
# ============================================================
print("\n--- S8.1: _classify_error ---")

# 不收敛
check("NON_CONVERGENCE (did not converge)",
      sm._classify_error("did not converge", {}) == ErrorType.NON_CONVERGENCE)
check("NON_CONVERGENCE (failed to converge)",
      sm._classify_error("FAILED TO CONVERGE", {}) == ErrorType.NON_CONVERGENCE)

# 干单元
check("DRY_CELLS (cell went dry)",
      sm._classify_error("cell went dry", {}) == ErrorType.DRY_CELLS)
check("DRY_CELLS (cells went dry plural)",
      sm._classify_error("one or more cells went dry", {}) == ErrorType.DRY_CELLS)

# 负浓度
check("NEGATIVE_CONCENTRATION",
      sm._classify_error("negative concentration detected", {}) == ErrorType.NEGATIVE_CONCENTRATION)
check("NEGATIVE_CONCENTRATION (conc < 0)",
      sm._classify_error("conc < 0 at cell", {}) == ErrorType.NEGATIVE_CONCENTRATION)

# 内存错误
check("MEMORY_ERROR",
      sm._classify_error("memory allocation failed", {}) == ErrorType.MEMORY_ERROR)

# 水量平衡
check("MASS_BALANCE_ERROR",
      sm._classify_error("budget percent error", {}) == ErrorType.MASS_BALANCE_ERROR)

# 未知错误
check("UNKNOWN (empty)",
      sm._classify_error("", {}) == ErrorType.UNKNOWN)
check("UNKNOWN (gibberish)",
      sm._classify_error("some random text", {}) == ErrorType.UNKNOWN)

# 从 list 格式解析（模拟 flopy 输出）
check("list-format error",
      sm._classify_error(["line 1", "did not converge", "line 3"], {}) == ErrorType.NON_CONVERGENCE)

# ============================================================
# S8.2: 自适应策略覆盖
# ============================================================
print("\n--- S8.2: ADAPTIVE_STRATEGIES ---")

expected_types = {ErrorType.NON_CONVERGENCE, ErrorType.DRY_CELLS, ErrorType.MASS_BALANCE_ERROR,
                  ErrorType.NEGATIVE_CONCENTRATION, ErrorType.MEMORY_ERROR, ErrorType.TIMEOUT}
check("6 error types covered", set(sm.ADAPTIVE_STRATEGIES.keys()) == expected_types)

# 每个类型都有至少一条策略
for et in expected_types:
    strategies = sm.ADAPTIVE_STRATEGIES.get(et, [])
    check(f"{et.value} has strategies", len(strategies) > 0)

# NON_CONVERGENCE 策略顺序
nc_strategies = sm.ADAPTIVE_STRATEGIES[ErrorType.NON_CONVERGENCE]
check("NON_CONV: strategy 1 = relaxation", nc_strategies[0]["action"] == "relaxation")
check("NON_CONV: strategy 2 = solver_switch", nc_strategies[1]["action"] == "solver_switch")
check("NON_CONV: strategy 3 = time_step_reduce", nc_strategies[2]["action"] == "time_step_reduce")
check("NON_CONV: strategy 4 = grid_refine (manual)",
      nc_strategies[3]["action"] == "grid_refine" and not nc_strategies[3]["auto"])

# DRY_CELLS 策略
dc_strategies = sm.ADAPTIVE_STRATEGIES[ErrorType.DRY_CELLS]
check("DRY: rewetting auto", dc_strategies[0]["action"] == "rewetting" and dc_strategies[0]["auto"])
check("DRY: bottom_adjust manual", dc_strategies[1]["action"] == "bottom_adjust" and not dc_strategies[1]["auto"])

# ============================================================
# S8.3: MAX_RETRIES
# ============================================================
print("\n--- S8.3: MAX_RETRIES ---")
check("MAX_RETRIES = 3", sm.MAX_RETRIES == 3)

# 验证尝试次数不会超标
# monitor_run 的 for 循环范围是 range(3) = 0,1,2
check("attempt loop 0..2", list(range(sm.MAX_RETRIES)) == [0, 1, 2])

# ============================================================
# S8.4: _apply_strategy 压力测试
# ============================================================
print("\n--- S8.4: _apply_strategy ---")

# 无对应策略的类型
result_unknown = sm._apply_strategy(ErrorType.UNKNOWN, attempt=0, sim=None, json_ir={})
check("UNKNOWN → no strategy", result_unknown is None)

# 超出 attempt 范围
result_oob = sm._apply_strategy(ErrorType.NON_CONVERGENCE, attempt=5, sim=None, json_ir={})
check("attempt 5 → out of range", result_oob is None)

# 合法 attempt 0: relaxation
result_0 = sm._apply_strategy(ErrorType.NON_CONVERGENCE, attempt=0, sim=None, json_ir={})
check("attempt 0 → relaxation applied",
      result_0 is not None and result_0["strategy"] == "relaxation" and result_0["success"])

# 合法 attempt 1: solver_switch（需人工确认）
result_1 = sm._apply_strategy(ErrorType.NON_CONVERGENCE, attempt=1, sim=None, json_ir={})
check("attempt 1 → solver_switch",
      result_1 is not None and result_1["strategy"] == "solver_switch")

# 手动策略（auto=False）应有 note
result_3 = sm._apply_strategy(ErrorType.NON_CONVERGENCE, attempt=3, sim=None, json_ir={})
check("attempt 3 → grid_refine (auto=False)",
      result_3 is not None and result_3["strategy"] == "grid_refine")

# ============================================================
# S8.5: _reduce_time_step
# ============================================================
print("\n--- S8.5: _reduce_time_step ---")

ir_ts = {"stress_periods": {"period_data": [{"perlen": 10.0, "nstp": 10, "tsmult": 1.0}]}}
sm._reduce_time_step(ir_ts, factor=0.5)
check("nstp: 10→20 (factor=0.5 doubles steps)", ir_ts["stress_periods"]["period_data"][0]["nstp"] == 20)

ir_ts2 = {"stress_periods": {"period_data": []}}
sm._reduce_time_step(ir_ts2, factor=0.5)
check("empty period_data → no crash", True)

# ============================================================
# S8.6: _adjust_dispersion
# ============================================================
print("\n--- S8.6: _adjust_dispersion ---")

ir_disp = {"transport": {"dispersion": {
    "al": {"value": 10.0},
    "trpt": {"value": 0.2},
    "trpv": {"value": 0.01},
}}}
sm._adjust_dispersion(ir_disp, factor=1.5)
check("al: 10→15", ir_disp["transport"]["dispersion"]["al"]["value"] == 15.0)
check("trpt: 0.2→0.3", abs(ir_disp["transport"]["dispersion"]["trpt"]["value"] - 0.3) < 1e-9)
check("trpv: 0.01→0.015", ir_disp["transport"]["dispersion"]["trpv"]["value"] == 0.015)

# ============================================================
# S8.7: 适配日志
# ============================================================
print("\n--- S8.7: adaptation_log ---")
sm2 = SolverMonitor()
# 手动应用几个策略
sm2._apply_strategy(ErrorType.NON_CONVERGENCE, attempt=0, sim=None, json_ir={})
sm2._apply_strategy(ErrorType.NON_CONVERGENCE, attempt=1, sim=None, json_ir={})
check("adaptation_log length=2", len(sm2.adaptation_log) == 2)
check("log[0] strategy=relaxation", sm2.adaptation_log[0]["strategy"] == "relaxation")
check("log[1] strategy=solver_switch", sm2.adaptation_log[1]["strategy"] == "solver_switch")
check("last_error is None", sm2.last_error is None)

# ============================================================
# 结果汇总
# ============================================================
print("\n" + "=" * 60)
total = passed + failed
print(f"Span 8 诊断结果: {passed}/{total} 通过")
if failed == 0:
    print("✅ SolverMonitor 错误分类和自适应策略工作正常。")
    print("   注意: _adjust_relaxation / _adjust_convergence 为预留桩方法。")
else:
    print(f"⚠️ {failed} 项测试失败，需排查。")
print("=" * 60)

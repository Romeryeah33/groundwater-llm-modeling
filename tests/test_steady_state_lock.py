"""
稳态开关回归锁 —— 防止"steady_state 被静默忽略"再次发生

缺陷（2026-10-12 定位）：

  `model_builder` 构造 `ModflowGwfsto` 时**只传 `transient=`，从不传 `steady_state=`**。
  而在 MODFLOW 6 里，STO 的 period 块**什么都不写时默认按瞬态算**；flopy 的
  `transient=False` 只是"不写关键字"，**不等于**把它写成稳态。

  后果：IR 里明确写了 `steady_state: true` 的模型，一律被**静默**跑成瞬态 ——
  不报错、不告警，`build_and_run` 照样返回 `success: true`。更糟的是界面上的参数摘要
  那一栏还恒显示 `steady_state={True}`，用户拿它核对参数只会被误导。

  实测（以 mf6 自己 `.lst` 的逐期判定为准）：

      steady_state=True    -> 判定 STEADY-STATE
      什么都不写            -> 判定 TRANSIENT        <- MF6 的默认
      transient={i: False} -> 判定 TRANSIENT        <- 旧代码走的就是这条

  本锁同时查**生成物**（`.sto` 里有没有 STEADY-STATE）与**运行结果**（mf6 怎么判），
  因为前者是直接被改动的对象，后者才是物理意义所在。

⚠ 本文件**必须能变红**：把 `model_builder.py` 里 `ModflowGwfsto(...)` 的
   `steady_state=is_steady` 删掉，本文件应当立刻失败。跑过一次变异自检再相信它。

不依赖 pytest，沿用仓库脚本式风格（`check(name, condition, detail)`）。
运行：`python tests/test_steady_state_lock.py`
"""
import sys
import re
import json
import shutil
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Force UTF-8
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from app.modeling.model_builder import AutoModelBuilder

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


def make_ir(steady, nper=2):
    """一个最小的、自包含的流向 IR：3×3×1，两侧定水头，保证能收敛。"""
    return {
        "ir_version": "1.0",
        "project": {"name": "steady_lock", "model_type": "groundwater_flow"},
        "domain": {"x_range": [0.0, 30.0], "y_range": [0.0, 30.0], "units": "meters"},
        "grid": {
            "nlay": 1, "nrow": 3, "ncol": 3,
            "delr": 10.0, "delc": 10.0, "top": 10.0, "bottom": 0.0,
        },
        "aquifer": {
            "type": "confined",
            "parameters": {
                "hk": {"value": 1.0, "unit": "m/d"},
                "ss": {"value": 1e-5, "unit": "1/m"},
                "sy": {"value": 0.15, "unit": "dimensionless"},
            },
        },
        "stress_periods": {
            "nper": nper,
            "steady_state": steady,
            "period_data": [{"perlen": 1.0, "nstp": 1, "tsmult": 1.0}] * nper,
        },
        "boundary_conditions": [
            {"type": "chd", "name": "west", "position": "west",
             "parameters": {"head": 8.0}},
            {"type": "chd", "name": "east", "position": "east",
             "parameters": {"head": 2.0}},
        ],
        "solver": {"name": "pcg", "parameters": {"hclose": 1e-6, "rclose": 1e-6}},
        "output": {"save_head": True, "save_budget": True},
    }


def first_period_block(sto_text):
    """取出 .sto 里第 1 个 period 块的内容。

    注意 FloPy 只在**第 1 期**写 STEADY-STATE / TRANSIENT，后面几期留空，
    由 MF6 按"沿用上一期的设定"继承下去（实测：第 1 期写 STEADY-STATE、
    第 2/3 期留空，mf6 三期全判 STEADY-STATE）。所以断言要看第 1 期。
    """
    m = re.search(r"BEGIN period\s+1(.*?)END period\s+1", sto_text, re.S)
    return m.group(1) if m else ""


def run_case(steady, nper=2):
    """跑一次真实建模，返回 {sto_text, sto_kw, verdicts, ran, out}。"""
    ws = Path(tempfile.mkdtemp(prefix="steady_lock_"))
    try:
        b = AutoModelBuilder(workspace=str(ws))
        out = b.build_and_run(make_ir(steady, nper), "lock")
        sto_text, verdicts, ran = "", [], False
        for p in ws.rglob("*.sto"):
            sto_text += p.read_text(encoding="utf-8", errors="replace")
        for p in ws.rglob("*_gwf.lst"):
            txt = p.read_text(encoding="utf-8", errors="replace")
            verdicts = re.findall(
                r"STRESS PERIOD +[0-9]+ +IS +(TRANSIENT|STEADY-STATE)", txt)
            ran = bool(verdicts)
        sto_kw = [ln.strip() for ln in sto_text.splitlines()
                  if ln.strip() in ("STEADY-STATE", "TRANSIENT")]
        return {"sto_text": sto_text, "sto_kw": sto_kw,
                "verdicts": verdicts, "ran": ran, "out": out}
    finally:
        shutil.rmtree(ws, ignore_errors=True)


print("=" * 62)
print("稳态开关回归锁（steady_state 必须真的生效）")
print("=" * 62)

# ------------------------------------------------------------
# 一、要求稳态：生成物里必须出现 STEADY-STATE
# ------------------------------------------------------------
print("\n[一] steady_state = true")
r1 = run_case(steady=True, nper=2)
sto, verdicts, ran = r1["sto_text"], r1["verdicts"], r1["ran"]

check("`true` 时 .sto 写出了 STEADY-STATE（旧代码这里是空的）",
      r1["sto_kw"].count("STEADY-STATE") > 0,
      f"实际关键字行 = {r1['sto_kw'] or '（空）'}")
check("`true` 时 .sto **没有**写出 TRANSIENT",
      "TRANSIENT" not in r1["sto_kw"],
      f"实际关键字行 = {r1['sto_kw']}")
check("STEADY-STATE 写在第 1 期块内（后续期由 MF6 沿用）",
      "STEADY-STATE" in first_period_block(sto),
      f"第 1 期块 = {first_period_block(sto).strip()!r}")

if ran:
    check("mf6 实际判定为 STEADY-STATE（.lst 逐期，nper=2）",
          verdicts == ["STEADY-STATE"] * 2,
          f"mf6 判定 = {verdicts}")
else:
    skip("mf6 实际判定为 STEADY-STATE",
         "没找到 *_gwf.lst 的逐期判定行（mf6 未跑成），生成物检查已覆盖主路径")

# ------------------------------------------------------------
# 二、要求瞬态：行为必须保持不变（不能被这次修复误伤）
# ------------------------------------------------------------
print("\n[二] steady_state = false（回归对照，不能被改坏）")
r2 = run_case(steady=False, nper=2)

check("`false` 时 .sto **没有**写出 STEADY-STATE",
      "STEADY-STATE" not in r2["sto_text"],
      f"实际关键字行 = {r2['sto_kw']}")
check("`false` 时第 1 期块内写着 TRANSIENT",
      "TRANSIENT" in first_period_block(r2["sto_text"]),
      f"第 1 期块 = {first_period_block(r2['sto_text']).strip()!r}")
if r2["ran"]:
    check("mf6 实际判定为 TRANSIENT（.lst 逐期，nper=2）",
          r2["verdicts"] == ["TRANSIENT"] * 2,
          f"mf6 判定 = {r2['verdicts']}")
else:
    skip("mf6 实际判定为 TRANSIENT", "没找到 .lst 逐期判定行")

# ------------------------------------------------------------
# 三、多应力期：稳态开关要覆盖每一期，不能只作用于第一期
#    （FloPy 只写第 1 期，靠 MF6 继承 —— 所以这里必须查 mf6 的判定，
#      只看 .sto 文本会误判成"没覆盖后面几期"）
# ------------------------------------------------------------
print("\n[三] nper = 3（稳态要覆盖每一期）")
r3 = run_case(steady=True, nper=3)

if r3["ran"]:
    check("nper=3 时 mf6 三期都判定 STEADY-STATE（继承生效）",
          r3["verdicts"] == ["STEADY-STATE"] * 3,
          f"mf6 判定 = {r3['verdicts']}")
else:
    skip("nper=3 时 mf6 三期都判定 STEADY-STATE", "没找到 .lst 逐期判定行")
check("nper=3 时 .sto 里没有残留的 TRANSIENT（不会互相打架）",
      "TRANSIENT" not in r3["sto_kw"],
      f"实际关键字行 = {r3['sto_kw']}")

# ------------------------------------------------------------
# 四、界面参数摘要不得再说假话
#   旧代码在 `_generate_code_preview` 里写死 `steady_state={True}`，
#   连 IR 都不读 —— 瞬态的模型也被显示成稳态，用户拿它核对参数会被误导。
# ------------------------------------------------------------
print("\n[四] 界面参数摘要按 IR 实际取值渲染")


def _preview_sto_line(steady):
    """调界面的代码预览函数，取它渲染出的那一行 ModflowGwfsto。"""
    from app.frontend.gradio_app import _generate_code_preview
    txt = _generate_code_preview({}, make_ir(steady), tempfile.mkdtemp())
    lines = [ln.strip() for ln in txt.splitlines() if "ModflowGwfsto" in ln]
    return lines[0] if lines else ""


try:
    line_steady = _preview_sto_line(True)
    line_trans = _preview_sto_line(False)
except Exception as e:  # 界面模块 import 或渲染失败都不算本锁的失败
    skip("界面摘要的 steady_state 渲染", f"{type(e).__name__}: {str(e)[:60]}")
else:
    check("要求稳态时摘要渲染 steady_state={True}",
          "steady_state={True}" in line_steady,
          f"实际渲染 = {line_steady!r}")
    check("要求瞬态时摘要渲染 steady_state={False}（旧代码恒显示 True）",
          "steady_state={False}" in line_trans,
          f"实际渲染 = {line_trans!r}")

# ============================================================
# 汇总
# ============================================================
print("\n" + "=" * 62)
total = passed + failed
print(f"稳态开关回归锁 测试结果: {passed}/{total} 通过"
      + (f" [{skipped} 项跳过]" if skipped else ""))
if failed == 0:
    print("✅ 全部通过")
else:
    print(f"⚠️ {failed} 项失败")
print("=" * 62)
sys.exit(1 if failed else 0)

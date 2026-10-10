"""
水量平衡提取回归测试

覆盖本轮修复的缺陷（报告问题 3：结果图标题写着 "Balance Error: 100.000%"）：

  缺陷 —— `ResultProcessor._process_budget_records` 按 `dtype.names` 把一条预算
  记录的**所有字段**相加。MF6 的预算记录里带名字的只有两种：
      ('node', 'node2', 'q', 'qx', 'qy', 'qz')        —— 只有 q 是流量
      ('node', 'node2', 'q', 'CONCENTRATION')         —— 只有 q 是流量
  node/node2 是格点索引（几万量级）、CONCENTRATION 是辅助浓度、qx/qy/qz 是分量。
  全量累加后 CHD 一项就被撑到几万，不平衡率于是显示 100%，而 .lst 里的真实值
  是 -0.05%。修复：改由 .lst 的权威文本解析（连 PERCENT DISCREPANCY 都直接给出，
  且不受 FloPy 版本差异影响 —— 本环境 flopy 3.10 读 .bud 得到的 CHD 流量是
  -1.8e-13，.lst 里是 1.0e-2），.lst 不可用时回退到读 .bud，且回退路径只认 `q`。

本文件必须自包含：合成一份格式与 MF6 真实输出逐字符一致的 .lst，另对磁盘上
真实存在的 .lst 做一次全局对账（没有就跳过，不算失败）。

不依赖 pytest，沿用仓库的脚本式风格（`check(name, condition, detail)`）。
运行：`python tests/test_water_budget.py`
"""
import sys
import re
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np

from app.analysis.result_processor import ResultProcessor

HERE = Path(__file__).parent.parent

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
        print(f"  ❌ {name}  {detail}")


# ============================================================
# 合成的 .lst 预算块（与 MF6 真实输出逐字符一致，取自
# data/outputs/cases/_selftest_a_ui/auto_model/auto_model_gwf.lst）
# ============================================================
SYNTHETIC_LST = """
                        MODFLOW 6
                U.S. GEOLOGICAL SURVEY MODULAR HYDROLOGIC MODEL
                  GROUNDWATER FLOW MODEL - VERSION 6.3.0

  VOLUME BUDGET FOR ENTIRE MODEL AT END OF TIME STEP 2500, STRESS PERIOD   1
  ---------------------------------------------------------------------------------------------------

     CUMULATIVE VOLUME      L**3       RATES FOR THIS TIME STEP      L**3/T          PACKAGE NAME
     ------------------                 ------------------------                     ----------------

           IN:                                      IN:
           ---                                      ---
              STO-SS =       4.9800E-05                STO-SS =       7.9311E-19     STORAGE
                 CHD =          12.5250                   CHD =       1.0020E-02     CHDA

            TOTAL IN =          12.5251              TOTAL IN =       1.0020E-02

          OUT:                                     OUT:
          ----                                     ----
              STO-SS =       1.0543E-14                STO-SS =       3.2236E-18     STORAGE
                 CHD =          12.5251                   CHD =       1.0020E-02     CHDA

           TOTAL OUT =          12.5251             TOTAL OUT =       1.0020E-02

            IN - OUT =      -2.0402E-10              IN - OUT =      -1.8067E-13

 PERCENT DISCREPANCY =          -0.00     PERCENT DISCREPANCY =          -0.00

  1 ... 1 ... 1 ... 1 ... 1 ... 1 ... 1 ... 1 ... 1 ... 1 ... 1 ... 1 ... 1 ... 1 ... 1 ... 1

      SUMMARY OF FLOW TERMS
"""


class _StubGwf:
    """只需要 model_ws + name 就能定位 .lst。"""

    def __init__(self, ws, name):
        self.model_ws = str(ws)
        self.name = name


def _write_lst(tmp: Path, name: str, body: str = SYNTHETIC_LST) -> _StubGwf:
    (tmp / f"{name}.lst").write_text(body, encoding="utf-8")
    return _StubGwf(tmp, name)


# ============================================================
# 1. `_flow_of` —— 一条记录里到底哪个字段是流量
# ============================================================
print("\n[1] _flow_of：从预算记录中取流量")

fp = ResultProcessor._flow_of

# 带名字段、且含 q —— 只有 q 是流量；node/node2/CONCENTRATION 是干扰项
rec_named = np.array(
    [(1000, 1001, 5.0, 999.0), (1000, 1002, -2.0, 888.0)],
    dtype=[("node", "<i4"), ("node2", "<i4"), ("q", "<f8"),
           ("CONCENTRATION", "<f8")],
)
v = fp(rec_named)
check("带名字段且含 q：只累加 q", v == 3.0, f"得到 {v}，应为 3.0 (=5.0-2.0)")
# 变异锁：把全部字段一起加会得到 1000+1001+5+999+1000+1002-2+888 = 5893
_inflated = float(sum(rec_named[n].sum() for n in rec_named.dtype.names))
check("变异自检：全字段累加确会得到荒谬值（证明修复非空过）",
      abs(_inflated - 5893.0) < 1e-9 and abs(_inflated - v) > 1000.0,
      f"全字段={_inflated}，正确 q={v}")
check("带名字段且含 q：node 索引不参与",
      fp(rec_named) < 100.0, f"得到 {fp(rec_named)}，若卷入 node 会 >1000")

# 带名字段、无 q（只有分量）—— 不是流量，按 0 计
rec_comp = np.array([(1.0, 2.0, 3.0)], dtype=[("qx", "<f8"), ("qy", "<f8"), ("qz", "<f8")])
check("带名字段但无 q：记 0，不猜分量", fp(rec_comp) == 0.0, f"得到 {fp(rec_comp)}")

# 不带名字段 —— 整个数组就是流量
rec_plain = np.array([1.5, -0.5, 2.0], dtype=float)
check("不带名字段：整数组求和", fp(rec_plain) == 3.0, f"得到 {fp(rec_plain)}")

check("非 ndarray：返回 None 而非抛异常", fp([1, 2, 3]) is None, f"得到 {fp([1,2,3])}")

# ============================================================
# 2. `_budget_from_lst` —— 合成 .lst 的精确解析
# ============================================================
print("\n[2] _budget_from_lst：合成 .lst 解析")

rp = ResultProcessor()
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    gwf = _write_lst(tmp, "auto_model_gwf")
    b = rp._budget_from_lst(gwf)

    check("解析成功（不为 None）", b is not None, "返回 None")
    if b is not None:
        # 左列 = 累积量。CHD 累积 12.5250（进）/ 12.5251（出）
        check("取累积列而非速率列（CHD 入 = 12.5250）",
              abs(b["inflow"].get("CHD", 0) - 12.5250) < 1e-9,
              f"得到 {b['inflow'].get('CHD')}，速率列会是 1.0020E-02")
        check("取累积列（CHD 出 = 12.5251）",
              abs(b["outflow"].get("CHD", 0) - 12.5251) < 1e-9,
              f"得到 {b['outflow'].get('CHD')}")
        check("TOTAL IN 精确", abs(b["total_in"] - 12.5251) < 1e-9,
              f"得到 {b['total_in']}")
        check("TOTAL OUT 精确", abs(b["total_out"] - 12.5251) < 1e-9,
              f"得到 {b['total_out']}")
        # PERCENT DISCREPANCY 是百分数（-0.00），须除以 100
        check("PERCENT DISCREPANCY 换算为比例（不是百分数）",
              abs(b["balance_error"]) < 1e-6,
              f"得到 {b['balance_error']}，应为 ~0 而非 0.05")
        check("来源标记为 lst", b.get("source") == "lst", f"得到 {b.get('source')}")

        # **核心不变量**：各分量之和必须等于总量。旧代码就在这里崩掉。
        s_in = sum(b["inflow"].values())
        s_out = sum(b["outflow"].values())
        check("不变量：inflow 分量之和 == TOTAL IN",
              abs(s_in - b["total_in"]) <= abs(b["total_in"]) * 1e-3 + 1e-9,
              f"分量和 {s_in} vs TOTAL IN {b['total_in']}")
        check("不变量：outflow 分量之和 == TOTAL OUT",
              abs(s_out - b["total_out"]) <= abs(b["total_out"]) * 1e-3 + 1e-9,
              f"分量和 {s_out} vs TOTAL OUT {b['total_out']}")

# ============================================================
# 3. 缺 .lst 时回退与优雅降级
# ============================================================
print("\n[3] 缺 .lst：回退路径与降级")
print("  （上方若有 '水量平衡提取失败' 的日志，是回退路径的预期告警，非测试失败）")

with tempfile.TemporaryDirectory() as td:
    gwf = _StubGwf(td, "nothere_gwf")
    check("无 .lst 文件时 _model_lst_path 返回 None",
          ResultProcessor._model_lst_path(gwf) is None, "返回了路径")
    check("无 .lst 时 _budget_from_lst 返回 None",
          rp._budget_from_lst(gwf) is None, "返回了 dict")

with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    gwf = _write_lst(tmp, "nolump_gwf", body="MODFLOW 6\n没有水量平衡块\n")
    check("有 .lst 但无预算块时返回 None",
          rp._budget_from_lst(gwf) is None, "返回了 dict")

# extract_water_budget 的整体契约：永远返回 dict，绝不抛
with tempfile.TemporaryDirectory() as td:
    gwf = _StubGwf(td, "bogus_gwf")
    out = rp.extract_water_budget(gwf)
    check("extract_water_budget 在无模型输出时仍返回 dict",
          isinstance(out, dict), f"得到 {type(out)}")
    check("回退路径不冒充 lst 来源",
          out.get("source") != "lst", f"得到 {out.get('source')}")

# ============================================================
# 4. 对磁盘上真实 .lst 全局对账（没有则跳过）
# ============================================================
print("\n[4] 真实 .lst 全局对账")

TOTAL_IN_RE = re.compile(r"TOTAL IN\s*=\s*([-\d.]+(?:[Ee][+-]?\d+)?)")
TOTAL_OUT_RE = re.compile(r"TOTAL OUT\s*=\s*([-\d.]+(?:[Ee][+-]?\d+)?)")

lst_files = sorted(HERE.glob("data/outputs/**/*.lst"))
# 只保留含水量平衡块的（GWT 或非水流模型的 .lst 没有）
real = []
for p in lst_files:
    try:
        if "VOLUME BUDGET FOR ENTIRE MODEL" in p.read_text(
                encoding="utf-8", errors="replace"):
            real.append(p)
    except OSError:
        continue

if not real:
    skipped += 1
    print("  ⏭  磁盘上没有含水量平衡块的 .lst，跳过对账")
else:
    bad_in = bad_out = bad_inv = 0
    for p in real:
        gwf = _StubGwf(p.parent, p.stem)
        b = rp._budget_from_lst(gwf)
        if b is None:
            bad_in += 1
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        idx = text.rfind("VOLUME BUDGET FOR ENTIRE MODEL")
        seg = text[idx:]
        # 独立的第二遍读取（不复用生产解析器），只认第一个 TOTAL IN/OUT
        m_in = TOTAL_IN_RE.search(seg)
        m_out = TOTAL_OUT_RE.search(seg)
        if m_in and abs(b["total_in"] - float(m_in.group(1))) > 1e-6:
            bad_in += 1
        if m_out and abs(b["total_out"] - float(m_out.group(1))) > 1e-6:
            bad_out += 1
        # 不变量：分量和 == 总量（旧代码在此崩到 100%）
        s_in = sum(b["inflow"].values())
        if abs(s_in - b["total_in"]) > abs(b["total_in"]) * 1e-3 + 1e-9:
            bad_inv += 1

    n = len(real)
    check(f"全部 {n} 个 .lst 的 TOTAL IN 与独立复读一致",
          bad_in == 0, f"{bad_in}/{n} 不一致")
    check(f"全部 {n} 个 .lst 的 TOTAL OUT 与独立复读一致",
          bad_out == 0, f"{bad_out}/{n} 不一致")
    check(f"全部 {n} 个 .lst 的不变量（分量和 == 总量）成立",
          bad_inv == 0, f"{bad_inv}/{n} 违反 —— 即旧缺陷复活")


# ============================================================
# 汇总
# ============================================================
print("\n" + "=" * 62)
total = passed + failed
print(f"水量平衡提取 测试结果: {passed}/{total} 通过"
      + (f" [{skipped} 项跳过]" if skipped else ""))
if failed == 0:
    print("✅ 全部通过")
else:
    print(f"⚠️ {failed} 项失败")
print("=" * 62)
sys.exit(1 if failed else 0)

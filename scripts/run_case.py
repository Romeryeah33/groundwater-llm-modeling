# -*- coding: utf-8 -*-
"""跑一个 JSON 算例 —— 走的是**和界面"运行模型"按钮完全相同的入口**

gradio_app.run_model_handler 最终执行的就是
    AutoModelBuilder(workspace).build_and_run(json_ir, "auto_model")
本脚本调同一个函数，所以这里能跑通 = 程序里能跑通。

与界面的唯一差别：界面在建模前还有两道闸（detect_missing_params /
quick_validate），本脚本默认也过一遍并打印结果，可用 --no-gate 跳过。

用法:
    PYTHONIOENCODING=utf-8 python scripts/run_case.py data/cases/case_a_column.json
    PYTHONIOENCODING=utf-8 python scripts/run_case.py <json> --out data/outputs/gwt_verify/myrun
    PYTHONIOENCODING=utf-8 python scripts/run_case.py <json> --no-gate --plot

退出码: 0 成功 / 1 失败
"""
import argparse
import io
import json
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from app.modeling.model_builder import AutoModelBuilder  # noqa: E402


def _load_ir(path: Path) -> dict:
    """读 IR。兼容两种形态：裸 IR，或 session 文件里的 {"json_ir": {...}}"""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw.get("json_ir", raw) if isinstance(raw, dict) else raw


def _describe(ir: dict) -> None:
    grid = ir.get("grid", {}) or {}

    # stress_periods 有三种形态：裸整数 nper / {nper,period_data} / 顶层只有 nper
    # 而时间信息在 transport.stress_periods（East Texas 与界面兼容版都是这种）
    def _num(v):
        return v.get("value") if isinstance(v, dict) else v

    sp = ir.get("stress_periods")
    pd = []
    if isinstance(sp, dict):
        pd = sp.get("period_data") or []
    if not pd:
        pd = ((ir.get("transport") or {}).get("stress_periods") or {}).get("period_data") or []
    nper = sp.get("nper") if isinstance(sp, dict) else sp
    total = sum(float(p.get("perlen", 0)) for p in pd) if pd else float("nan")
    tr = ir.get("transport", {}) or {}

    print(f"  项目      : {(ir.get('project') or {}).get('name')}")
    print(f"  网格      : {grid.get('nrow')} × {grid.get('ncol')} × {grid.get('nlay')}")
    print(f"  时间      : nper={nper}  period_data={len(pd)} 项  "
          f"总时长={total:g} d")
    print(f"  边界      : {[b.get('type') for b in ir.get('boundary_conditions', [])]}")
    print(f"  transport : enabled={tr.get('enabled')}  "
          f"al={_num(tr.get('al'))}  porosity={_num(tr.get('porosity'))}  "
          f"scheme={(tr.get('advection') or {}).get('scheme')}")


def _ui_gate(ir: dict) -> bool:
    """复现界面建模前的两道闸，返回是否放行"""
    from app.llm_agent.intent_parser import intent_parser
    from app.physics_engine.validator import quick_validate

    ok = True
    try:
        missing = intent_parser.detect_missing_params(ir)
        p1, p2 = missing.get("priority_1", []), missing.get("priority_2", [])
        print(f"  [缺失参数] P1={p1 or '无'}  P2={p2 or '无'}")
        if p1 or p2:
            ok = False
    except Exception as e:
        print(f"  [缺失参数] 检测抛异常: {type(e).__name__}: {e}")
        ok = False

    try:
        val = quick_validate(ir)
        print(f"  [物理校验] valid={val.get('valid')}  "
              f"error={val.get('error_count')}  warning={val.get('warning_count')}")
        for i in val.get("issues", []):
            if i.get("severity") == "error":
                print(f"      ERROR {i.get('param_name')}: {i.get('description')}")
        if val.get("error_count", 0) > 0:
            ok = False
    except Exception as e:
        print(f"  [物理校验] 抛异常: {type(e).__name__}: {e}")
        ok = False
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("case", help="算例 JSON 路径")
    ap.add_argument("--out", default=None, help="输出目录（默认 data/outputs/cases/<名字>）")
    ap.add_argument("--no-gate", action="store_true", help="跳过界面那两道前置闸")
    ap.add_argument("--plot", action="store_true", help="额外画一张末时刻浓度剖面/平面图")
    args = ap.parse_args()

    case_path = Path(args.case)
    if not case_path.exists():
        print(f"!! 找不到算例文件: {case_path}")
        return 1

    ir = _load_ir(case_path)
    name = case_path.stem
    ws = Path(args.out) if args.out else (_ROOT / "data" / "outputs" / "cases" / name)
    ws.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"算例: {case_path}")
    print(f"工作目录: {ws}")
    print("=" * 70)
    _describe(ir)

    if not args.no_gate:
        print("\n--- 界面前置闸（与 gradio_app.run_model_handler 一致）---")
        if not _ui_gate(ir):
            print("\n!! 该算例会被界面的前置闸拦下（上述 P1/P2 缺失或校验错误）。")
            print("   界面里走不通 ≠ 建模器有问题 —— 这两道闸是对话层的参数完备性检查。")
            print("   要用建模器本身验证请加 --no-gate。")
            return 1

    print("\n--- 建模 + 运行（AutoModelBuilder.build_and_run）---")
    builder = AutoModelBuilder(str(ws))
    res = builder.build_and_run(ir, "auto_model")

    if not res.get("success"):
        print(f"!! 运行失败: {res.get('error')}")
        detail = res.get("detail")
        if detail:
            print(detail)
        return 1

    print(f"  运行成功   水头范围: {res.get('head_range')}")
    gwf, gwt = res.get("gwf"), res.get("gwt")
    if gwf is None:
        print("  (未启用溶质运移，只有水流模型)")
        return 0

    co = gwt.output.concentration()
    if co is None:
        print("!! 没有浓度输出 —— GWT 没有真正产出 .ucn")
        return 1
    times = [float(t) for t in np.atleast_1d(co.get_times())]
    print(f"  输出时刻数 : {len(times)}  (t = {times[0]:g} … {times[-1]:g} d)")

    c_end = np.asarray(co.get_data(totim=times[-1]))
    c0 = c_end[0].ravel() if c_end.ndim == 3 else c_end.ravel()
    print(f"  末时刻浓度 : min={c0.min():.4e}  max={c0.max():.4e}"
          f"  非零格数={int((np.abs(c0) > 1e-12).sum())}/{c0.size}")

    if c0.max() <= 0:
        print("  !! 浓度场恒为 0 —— 没有任何浓度源流入，链路不算跑通")
        return 1

    print(f"  .ucn       : {list(ws.rglob('*.ucn'))}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        nrow, ncol = ir.get("grid", {}).get("nrow", 1), ir.get("grid", {}).get("ncol", 1)
        if nrow == 1:
            delr = ir["grid"].get("delr", 1.0)
            delr = delr.get("value", 1.0) if isinstance(delr, dict) else delr
            ax.plot((np.arange(ncol) + 0.5) * delr, c0, "o-", ms=3)
            ax.set_xlabel("x (m)")
        else:
            ax.imshow(np.asarray(c_end[0] if c_end.ndim == 3 else c_end).reshape(nrow, ncol),
                      origin="upper")
            ax.set_xlabel("col")
            ax.set_ylabel("row")
        ax.set_title(f"{name}  t={times[-1]:g} d")
        out_png = ws / f"{name}_final.png"
        fig.tight_layout()
        fig.savefig(out_png, dpi=120)
        print(f"  图         : {out_png}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

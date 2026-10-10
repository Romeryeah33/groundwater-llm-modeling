# -*- coding: utf-8 -*-
"""
为中期报告的界面截图采集 Case D 的真实结果。

只读：加载已跑完的 Case D（data/outputs/gwt_verify/eastTexas_cased/auto_model_d），
用项目自己的 ResultProcessor / VisualizationEngine 提取场与图件，把指标写到
data/outputs/report_figs/metrics.json，图件写到同目录。

**不写 data/sessions/default.json**（用户真实会话），不重跑 mf6。
"""
import sys, os, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import flopy

from app.analysis.result_processor import ResultProcessor
from app.analysis.visualizer import VisualizationEngine
from app.analysis.process_identifier import process_identifier

WS = ROOT / "data" / "outputs" / "gwt_verify" / "eastTexas_cased" / "auto_model_d"
OUT = ROOT / "data" / "outputs" / "report_figs"
OUT.mkdir(parents=True, exist_ok=True)

# 不含 'sto'：见 main() 里的说明
LOAD_ONLY = ["tdis", "ims", "dis", "npf", "ic", "oc", "chd", "ghb", "rch",
             "adv", "dsp", "mst", "ssm", "gwfgwt"]

BASEDIR = ROOT.parent / "溶质运移模型基础资料"
LANDFILL = BASEDIR / "landfill_arcs.shp"
AQUIFER = BASEDIR / "Aquifer Layer 1_arcs.shp"
# 裁剪必须用**面**图层：_draw_basemap 取 gdf.geometry.iloc[0] 当"boundary_polygon"，
# 不检查几何类型。若给线图层，contains_xy 只命中落在线上的极少数格心，掩膜会把
# 整个浓度场裁掉（填色全白，且只有一行 INFO 没有警告）。
# 程序自己也是这么选的：gradio_app 的 _detect_basemap_inputs 只把 polygon 图层
# 放进 polygon_candidates 当 clip_shp_path。
AQUIFER_POLYS = BASEDIR / "Aquifer Layer 1_polys.shp"


def main():
    print(f"加载 {WS.name} ...", flush=True)
    # 必须给 load_only：直接全量加载会在 .sto 包上崩（flopy 3.10 的
    # MFDataScalar.set_data 对空 scalar 数组取 data[0] → IndexError）。
    # 本项目的水流模型稳态、STO 里没有有效数据，读不出来也不影响取水头/浓度。
    sim = flopy.mf6.MFSimulation.load(sim_ws=str(WS), verbosity_level=0,
                                      load_only=LOAD_ONLY)
    gwf = sim.get_model("auto_model_d_gwf")
    gwt = sim.get_model("auto_model_d_gwt")
    print("  gwf =", gwf.name, " gwt =", gwt.name, flush=True)

    rp = ResultProcessor()
    viz = VisualizationEngine()

    # ---- 网格 ---- #
    dis = gwf.dis
    nlay, nrow, ncol = dis.nlay.data, dis.nrow.data, dis.ncol.data
    delr = np.asarray(dis.delr.array, dtype=float)
    delc = np.asarray(dis.delc.array, dtype=float)
    print(f"  网格 {nlay} 层 × {nrow} 行 × {ncol} 列, delr={delr[0]}, delc={delc[0]}", flush=True)

    # ---- 水头 ---- #
    head = rp.extract_head_field(gwf, layer=0)
    h_min, h_max = float(np.nanmin(head)), float(np.nanmax(head))
    print(f"  水头 {h_min:.3f} ~ {h_max:.3f} m", flush=True)

    # ---- 水均衡 ---- #
    try:
        budget = rp.extract_water_budget(gwf)
    except Exception as e:
        print("  水均衡提取失败:", e, flush=True)
        budget = {}
    print("  budget keys:", list(budget.keys())[:12], flush=True)
    print("  budget:", json.dumps({k: v for k, v in list(budget.items())[:14]},
                                  ensure_ascii=False, default=str)[:600], flush=True)

    # ---- 浓度场 ---- #
    conc_obj = gwt.output.concentration()
    times = conc_obj.get_times()
    times = [float(t) for t in np.atleast_1d(times)] if times is not None else []
    print(f"  共 {len(times)} 个输出时刻: {[round(t,1) for t in times]}", flush=True)

    n_frames = min(6, len(times))
    # 跳过第 0 帧：t=150 d 时源区浓度还没超过填色阈值，那一格会是整片空白，
    # 在报告图里看着像"图坏了"，实际只是羽流尚未成形。
    start = 1 if len(times) > n_frames else 0
    idx = np.unique(np.linspace(start, len(times) - 1, n_frames).astype(int))
    frames = [rp.extract_concentration_field(gwt, layer=0, totim=times[i]) for i in idx]
    frame_times = [times[i] for i in idx]
    c_max = float(max(np.max(f) for f in frames))
    print(f"  取 {len(frames)} 帧, 时刻 {[round(t,1) for t in frame_times]}", flush=True)
    print(f"  浓度峰值 {c_max:.1f}", flush=True)

    # 填色阈值取 10 mg/L（源浓度 20000 的 0.05%）。
    # 注意：程序里 run_model_handler 硬编码 threshold=0.01 —— 那是**绝对值**，
    # 对本案例相当于源浓度的 5e-7，会把 16% 的模型域都填上极淡的颜色，
    # 图看起来是一片均匀点纹而不是一支羽流。这里用有物理含义的阈值。
    thr = 10.0
    last = frames[-1]
    n_cells_total = last.size
    n_active = int(np.sum(last > thr))
    print(f"  末帧 >{thr} mg/L 的格数 {n_active} / {n_cells_total} "
          f"({100.0*n_active/n_cells_total:.2f}%)", flush=True)

    velocity = rp.extract_velocity_field(gwf, layer=0)
    print("  流速场:", "OK" if velocity is not None else "不可得", flush=True)

    # shp_bounds 必须给：程序里它来自上传边界 SHP 的 bounds，_grid_coords 用它
    # 把"0 起点的网格坐标"平移到 SHP 的投影坐标。不给的话网格画在 0 起点、
    # 而 SHP 在 UTM（6.1e5, 3.4e6），matplotlib 自动缩放会把轴拉到 UTM 量级，
    # 浓度场缩成一个点 —— 图看起来是空的。
    shp_bounds = None
    try:
        import geopandas as gpd
        shp_bounds = [float(v) for v in gpd.read_file(str(AQUIFER)).total_bounds]
        print(f"  shp_bounds = {[round(v,1) for v in shp_bounds]}", flush=True)
    except Exception as e:
        print("  shp_bounds 读取失败:", e, flush=True)

    # GATHER_NO_CLIP=1 → 不做边界裁剪，用于核对"羽流是否被裁断"
    # GATHER_TAG=<s>  → 输出文件名改用 _{s}_<key>.png，便于并存对比
    no_clip = os.environ.get("GATHER_NO_CLIP") == "1"
    tag = os.environ.get("GATHER_TAG", "gather")
    shp_kw = dict(shp_path=str(LANDFILL) if LANDFILL.exists() else None,
                  clip_shp_path=(None if no_clip
                                 else (str(AQUIFER_POLYS) if AQUIFER_POLYS.exists() else None)),
                  shp_bounds=shp_bounds)

    # ---- 图件 ---- #
    figs = {}
    # plot_head_contour 不接受 shp 参数（只有水头/网格/井/标题），
    # 所以水头图在局部坐标（0 起点）下绘制，自带流向箭头。
    try:
        figs["head"] = viz.plot_head_contour(head, delr, delc)
        print("  head 图 OK", flush=True)
    except Exception as e:
        print("  head 图失败:", e, flush=True)

    # vmin 必须一并给：_draw_plume_panel 里 floor = min(threshold, vmin)，
    # vmin 默认 0.0 时 min(10, 0) = 0，阈值被架空、填色从 0 起铺满全域。
    # （该式与自己的 docstring 矛盾，是缺陷；这里用 vmin=thr 绕过，不改源码。）
    try:
        figs["plume_evo"] = viz.plot_plume_evolution(
            frames, frame_times, delr, delc, threshold=thr, vmin=thr,
            n_cols=min(3, len(frames)), velocity=velocity, **shp_kw)
        print("  plume_evolution 图 OK", flush=True)
    except Exception as e:
        print("  plume_evolution 图失败:", e, flush=True)

    try:
        figs["plume_single"] = viz.plot_plume_at_time(
            frames[-1], frame_times[-1], delr, delc, threshold=thr, vmin=thr,
            velocity=velocity, **shp_kw)
        print("  plume_single 图 OK", flush=True)
    except Exception as e:
        print("  plume_single 图失败:", e, flush=True)

    proc_ok = False
    try:
        if len(frames) >= 2 and velocity is not None:
            import json as _json
            sess = _json.loads((ROOT / "data" / "sessions" / "default.json")
                               .read_text(encoding="utf-8"))
            ir = sess.get("json_ir", {}) or {}
            tp = ir.get("transport", {}) or {}
            params = {}
            for k in ("al", "decay_rate", "kd", "porosity"):
                v = tp.get(k)
                if isinstance(v, dict):
                    v = v.get("value")
                if isinstance(v, (int, float)):
                    params[k] = float(v)
            print("  过程辨识参数:", params, flush=True)
            proc = process_identifier.identify_dominant_process(
                conc_history=frames, velocity_field=[velocity] * len(frames),
                parameters=params, delr=delr, delc=delc)
            figs["process"] = viz.plot_process_analysis(proc)
            proc_ok = True
            print("  process 图 OK", flush=True)
            print("  process 结果:", _json.dumps(proc, ensure_ascii=False,
                                                default=str)[:800], flush=True)
    except Exception as e:
        import traceback
        print("  process 图失败:", e, flush=True)
        traceback.print_exc()

    for key, fig in figs.items():
        p = OUT / f"_{tag}_{key}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  已存 {p.name} ({p.stat().st_size} bytes)", flush=True)

    metrics = dict(head_min=h_min, head_max=h_max, c_max=c_max,
                   n_times=len(times), times=times,
                   frame_times=frame_times, n_active_last=n_active,
                   n_cells_total=int(n_cells_total), proc_ok=proc_ok,
                   budget={k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                           for k, v in budget.items()})
    (OUT / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n指标已写 {OUT/'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()

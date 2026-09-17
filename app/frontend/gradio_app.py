"""
Gradio Web 前端 — Gradio 6.x 兼容版

集成流程：
  自然语言/文件输入 → 意图解析 → 参数提取 → 缺失检测 →
  物理约束校验 → FloPy自动建模 → 结果可视化 → 报告下载
"""

import sys
import io
import os
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
import logging
import json
import traceback
import numpy as np
import matplotlib.pyplot as plt

from app.config import settings
from app.llm_agent.dialogue_manager import DialogueManager
from app.llm_agent.intent_parser import intent_parser
from app.knowledge.param_guide import render_missing_params_guidance
from app.llm_agent.file_handler import file_handler
from app.physics_engine.validator import quick_validate
from app.modeling.model_builder import AutoModelBuilder
from app.modeling.source_zone import (
    SourceZoneUnresolved,
    grid_dims,
    materialize_source_zones,
)
from app.analysis.visualizer import VisualizationEngine, visualizer
from app.analysis.result_processor import ResultProcessor
from app.utils.ir_normalize import merge_time_sources
from app.analysis.process_identifier import process_identifier
from app.analysis.report_generator import ReportGenerator
from app.visualization.plot_styler import plot_styler

import pandas as pd
import copy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- 调试监控器 ----
from app.utils.debug_monitor import monitor
logger.info("调试监控器已启用 — 日志: %s", monitor.get_log_file_path())

# ============================================================
sessions = {}

def get_session(sid: str) -> DialogueManager:
    if sid not in sessions:
        dm = DialogueManager()
        # ── 自动恢复上次会话（重启后无需重新输入）──
        restored = dm.load_session(sid)
        sessions[sid] = dm
        if restored:
            logger.info(f"会话 {sid}: 已从磁盘恢复，历史 {len(dm.state.get('history', []))} 轮")
            # 标记为刚恢复，供 chat_handler 显示欢迎信息
            dm._just_restored = True
    return sessions[sid]


# ============================================================
# 结果缓存 —— 支撑污染羽"手动选时刻 / 选色标范围"的交互重绘
# ============================================================
# 为什么用模块级 dict 而不是 gr.State：缓存里必须留住 gwt（MF6 模型对象）
# 才能按需读取 .ucn，而它不可序列化。本仓库已有两个同类先例：
# sessions（sid → DialogueManager）与 plot_styler 单例。
_result_cache: dict = {}   # sid -> {gwt, gwf, times, delr, delc, layer, c_range,
                          #         basemap, workspace, velocity, _frame_cache}


def _detect_basemap_inputs(sid: str) -> dict:
    """
    从会话上传文件中推断底图与边界文件。

    原先这段逻辑内联在 run_model_handler 里、只服务于水头图。抽出来供污染羽图
    复用，两张图因此必定拿到同一套参数（各扫一次可能选中不同文件）。

    Returns:
        {"basemap_path": str|None, "shp_path": str|None,
         "clip_shp_path": str|None, "shp_bounds": list|None,
         "origin_xy": list|None}
    """
    found = {"basemap_path": None, "shp_path": None,
             "clip_shp_path": None, "shp_bounds": None,
             "origin_xy": None}
    try:
        dm = get_session(sid)
        file_data = dm.state.get("file_uploads", {})
        if not file_data:
            return found

        uploaded_paths = dm.state.get("_uploaded_file_paths", [])
        basemap_path = None
        shp_bounds = None
        shp_path = None

        # 1. 查找底图图片（优先 .tif GeoTIFF，其次 .png/.jpg）
        for up in uploaded_paths:
            fname = Path(up).name.lower()
            ext = Path(up).suffix.lower()
            # 底图关键词匹配
            is_basemap_name = any(kw in fname for kw in
                ['底图', 'basemap', 'base', 'map', '影像', 'image', 'satellite', '卫片'])
            if ext in ('.tif', '.tiff'):
                basemap_path = up
                if is_basemap_name:
                    break  # 优先使用明确命名的底图
            elif ext in ('.png', '.jpg', '.jpeg', '.bmp'):
                if is_basemap_name:
                    basemap_path = up
                    break
                elif basemap_path is None:
                    basemap_path = up  # 回退到任意图片

        # 2. 查找 SHP 边界 — 按名称匹配 "boundary"/"边界"，优先级：面 > 线
        shp_data = file_data.get("shapefiles", {})
        boundary_sph_name = dm.state.get("_boundary_shp_name", "")

        # 先收集所有候选边界文件
        boundary_candidates = []
        for name, info in shp_data.items():
            name_lower = name.lower()
            is_boundary = ('boundary' in name_lower or '边界' in name or
                           'border' in name_lower or name == boundary_sph_name)
            if is_boundary and info.get("bounds") and len(info["bounds"]) >= 4:
                b = info["bounds"]
                if all(isinstance(x, (int, float)) and not (x != x) for x in b[:4]):
                    is_polygon = 'polygon' in info.get('geometry_type', '').lower()
                    boundary_candidates.append((name, info, is_polygon))

        # 也加入 _polys 面要素作为裁剪用
        for name, info in shp_data.items():
            if 'poly' in name.lower() and info.get('geometry_type', '').lower() == 'polygon':
                if info.get("bounds"):
                    b = info["bounds"]
                    if all(isinstance(x, (int, float)) and not (x != x) for x in b[:4]):
                        if not any(c[0] == name for c in boundary_candidates):
                            boundary_candidates.append((name, info, True))

        # 选择边界：优先面要素，其次线要素
        polygon_candidates = [c for c in boundary_candidates if c[2]]
        line_candidates = [c for c in boundary_candidates if not c[2]]

        # 用于裁剪的面要素
        clip_shp_path = None
        if polygon_candidates:
            name, info, _ = polygon_candidates[0]
            b = info["bounds"]
            shp_bounds = b if shp_bounds is None else shp_bounds
            for up in uploaded_paths:
                if Path(up).stem == name and Path(up).suffix.lower() == '.shp':
                    clip_shp_path = up
                    break
            logger.info(f"边界裁剪面: {name} ({info['geometry_type']})")

        # 用于显示的线要素（Boundary_arcs）
        if line_candidates:
            name, info, _ = line_candidates[0]
            b = info["bounds"]
            shp_bounds = b if shp_bounds is None else shp_bounds
            if shp_path is None:
                for up in uploaded_paths:
                    if Path(up).stem == name and Path(up).suffix.lower() == '.shp':
                        shp_path = up
                        break
            logger.info(f"边界显示线: {name} ({info['geometry_type']})")

        # 如果没有面要素用于裁剪，尝试找任何 _polys 文件
        if clip_shp_path is None:
            for name, info in shp_data.items():
                if 'poly' in name.lower() and 'polygon' in info.get('geometry_type', '').lower():
                    for up in uploaded_paths:
                        if Path(up).stem == name and Path(up).suffix.lower() == '.shp':
                            clip_shp_path = up
                            break
                    if clip_shp_path:
                        logger.info(f"使用面要素裁剪: {name}")
                        break

        # ── 网格原点 ──
        # 优先读 IR 里已经物化的 `domain.origin_xy`（file_parse_handler 上传时写入），
        # 缺则回退到 SHP 外框左下角。两条路都走通后，把 `shp_bounds` 的前两位
        # **覆盖成同一个原点**，让"建模用的原点"与"可视化用的原点"是同一个值
        # 而不是两个恰好相等的数 —— 否则源区选格与绘图会悄悄错开。
        origin_xy = None
        ir = (get_session(sid).state or {}).get("json_ir", {}) or {}
        _ir_origin = (ir.get("domain") or {}).get("origin_xy")
        if isinstance(_ir_origin, dict):
            _ir_origin = _ir_origin.get("value")
        if _ir_origin and len(_ir_origin) >= 2:
            origin_xy = [float(_ir_origin[0]), float(_ir_origin[1])]
        elif shp_bounds and len(shp_bounds) >= 4:
            origin_xy = [float(shp_bounds[0]), float(shp_bounds[1])]

        if origin_xy and shp_bounds and len(shp_bounds) >= 4:
            if (abs(float(shp_bounds[0]) - origin_xy[0]) > 1e-6 or
                    abs(float(shp_bounds[1]) - origin_xy[1]) > 1e-6):
                logger.warning(
                    f"IR 的 origin_xy={origin_xy} 与 SHP 外框左下角"
                    f"[{shp_bounds[0]:.1f}, {shp_bounds[1]:.1f}] 不一致，"
                    f"统一采用 IR 中的 origin_xy"
                )
            shp_bounds = [origin_xy[0], origin_xy[1], shp_bounds[2], shp_bounds[3]]

        logger.info(f"底图叠加: 显示线={shp_path}, 裁剪面={clip_shp_path}, "
                    f"底图={basemap_path}, origin_xy={origin_xy}")
        found.update(basemap_path=basemap_path, shp_path=shp_path,
                     clip_shp_path=clip_shp_path, shp_bounds=shp_bounds,
                     origin_xy=origin_xy)
    except Exception as e:
        logger.warning(f"底图输入探测失败: {e}")
    return found


def _sample_conc_at(cache: dict, t: float):
    """
    取指定时刻的浓度场，必要时在**已有输出时刻之间**做线性插值。

    约束：flopy 只接受精确的输出时刻作为 ``totim``，传任意时刻会失败。所以想看
    "360 天"这类非输出时刻，必须自己取相邻两个输出时刻做插值，且端点值一律
    原样来自 ``get_times()``。

    不做外推：t 早于首帧或晚于末帧时一律钳到端点并明确告知。

    Returns:
        (conc2d, t_lo, t_hi, interpolated, note)
    """
    gwt = cache.get("gwt")
    times = cache.get("times")
    if gwt is None or times is None or len(times) == 0:
        return None, None, None, False, "尚无结果，请先运行建模"

    rp = ResultProcessor()
    layer = cache.get("layer", 0)
    fcache = cache.setdefault("_frame_cache", {})

    def _frame(ti):
        key = float(ti)
        if key not in fcache:
            fcache[key] = rp.extract_concentration_field(gwt, layer=layer, totim=key)
        return fcache[key]

    t = float(t)
    t_min, t_max = float(times[0]), float(times[-1])

    if t <= t_min:
        note = (f"t = {t:g} d 早于首个输出时刻 {t_min:g} d，显示首帧"
                if t < t_min else f"t = {t:g} d（输出时刻）")
        return _frame(t_min), t_min, t_min, False, note
    if t >= t_max:
        note = (f"t = {t:g} d 超出模拟时长，显示末时刻 {t_max:g} d"
                if t > t_max else f"t = {t:g} d（输出时刻）")
        return _frame(t_max), t_max, t_max, False, note

    i = int(np.searchsorted(times, t))
    if float(times[i]) == t:
        return _frame(t), t, t, False, f"t = {t:g} d（输出时刻）"

    t0, t1 = float(times[i - 1]), float(times[i])
    C0, C1 = _frame(t0), _frame(t1)
    w = (t - t0) / (t1 - t0)
    note = f"t = {t:g} d（由 {t0:g} d 与 {t1:g} d 线性插值，权重 {w:.2f}）"
    return C0 + (C1 - C0) * w, t0, t1, True, note


def _extract_process_params(json_ir: dict) -> dict:
    """
    从 IR 的 transport 段取出 ProcessIdentifier 需要的参数。

    缺项一律不填：ProcessIdentifier 自带默认值（al=10.0 / decay_rate=0.0 /
    kd=0.0 / porosity=0.25），取不到参数不该让整个流程报错。
    """
    transport = json_ir.get("transport", {}) or {}
    params = {}
    for key in ("al", "decay_rate", "kd", "porosity"):
        val = transport.get(key)
        if isinstance(val, dict):
            val = val.get("value")
        if isinstance(val, (int, float)):
            params[key] = float(val)
    return params


def _pick_vmin_vmax(auto_range: bool, vmin_in, vmax_in, span: tuple):
    """
    解析用于单时刻图的色标范围。

    勾选"自动"时忽略输入框；未勾选时以输入框为准，单个框留空则该项回退自动。
    返回的是**实际生效**的数值，供调用方回写进输入框，让用户看得见当前范围。
    """
    auto_lo, auto_hi = span
    if auto_range:
        return auto_lo, max(auto_hi, auto_lo + 1e-12)

    lo = float(vmin_in) if isinstance(vmin_in, (int, float)) else auto_lo
    hi = float(vmax_in) if isinstance(vmax_in, (int, float)) else auto_hi
    if hi <= lo:
        hi = lo + 1e-12
    return lo, hi


def _render_plume(t, auto_range, vmin_in, vmax_in, sid: str):
    """
    污染羽单时刻图的渲染内核（供滑块/数字框/色标控件共用）。

    Returns:
        (fig, status_md, vmin_out, vmax_out)
    """
    cache = _result_cache.get(sid)
    if not cache or cache.get("times") is None or len(cache["times"]) == 0:
        return None, "⚠️ 还没有可重绘的结果，请先点击「开始建模」。", vmin_in, vmax_in

    if t is None:
        t = float(cache["times"][-1])

    conc, t_lo, t_hi, interp, note = _sample_conc_at(cache, t)
    if conc is None:
        return None, f"⚠️ {note}", vmin_in, vmax_in

    span = cache.get("c_range", (0.0, float(np.max(conc))))
    vmin, vmax = _pick_vmin_vmax(auto_range, vmin_in, vmax_in, span)

    bm = cache.get("basemap", {}) or {}
    try:
        fig = visualizer.plot_plume_at_time(
            conc, float(t), cache["delr"], cache["delc"],
            threshold=0.01, vmin=vmin, vmax=vmax,
            basemap_path=bm.get("basemap_path"),
            shp_path=bm.get("shp_path"),
            clip_shp_path=bm.get("clip_shp_path"),
            shp_bounds=bm.get("shp_bounds"),
            velocity=cache.get("velocity"),
            title=f"污染羽 t = {float(t):g} d",
            subtitle=note,
        )
    except Exception as e:
        logger.warning(f"单时刻污染羽绘制失败: {e}")
        logger.warning(traceback.format_exc())
        return None, f"❌ 绘图失败: {e}", vmin_in, vmax_in

    flags = []
    if interp:
        flags.append("（插值帧）")
    if float(t) < float(cache["times"][0]) or float(t) > float(cache["times"][-1]):
        flags.append("（已钳到输出范围端点）")
    status = (f"✅ {note}　色标 {vmin:.4g} ~ {vmax:.4g} mg/L　"
              f"本帧最大浓度 {float(np.max(conc)):.4g} mg/L {' '.join(flags)}")
    return fig, status, vmin, vmax


def _on_plume_slider(slider_t, auto_range, vmin_in, vmax_in, sid):
    fig, status, vmin_o, vmax_o = _render_plume(slider_t, auto_range,
                                                vmin_in, vmax_in, sid)
    # 回写数字框，两个时间控件始终同步
    return fig, status, gr.update(value=slider_t), vmin_o, vmax_o


def _on_plume_number(num_t, auto_range, vmin_in, vmax_in, sid):
    fig, status, vmin_o, vmax_o = _render_plume(num_t, auto_range,
                                                vmin_in, vmax_in, sid)
    # 回写滑块，两个时间控件始终同步
    return fig, status, gr.update(value=num_t), vmin_o, vmax_o


def _on_plume_apply(slider_t, num_t, auto_range, vmin_in, vmax_in, sid):
    """
    「确定重绘」：把色标上下限（连同时刻）一次性应用后重绘。

    为什么需要这个按钮：`gr.Number.change` 是**逐字符**触发的 —— 想输入 `300`，
    键入 `3` 的那一刻就开始跑一次计算，用户必须等这次结果回来才能接着键入 `0`。
    改成显式确认后，键入过程完全不触发，按一次算一次。
    （数字框里按回车走同一个处理器：回车是明确的"输入完毕"动作，不会误触发。）

    时刻取值：数字框有有效值就用它，否则用滑块位置 —— 滑块 `release` 会把位置
    回写进数字框，两者本就同步。
    """
    t = num_t if isinstance(num_t, (int, float)) else slider_t
    fig, status, vmin_o, vmax_o = _render_plume(t, auto_range,
                                                vmin_in, vmax_in, sid)
    # 同步回写两个时间控件，与滑块/数字框那两条路径保持同一行为
    return (fig, status,
            gr.update(value=t), gr.update(value=t), vmin_o, vmax_o)


def _sync_plume_controls(sid: str):
    """
    运行结束后同步交互控件：滑块范围来自本次运行的输出时刻，
    并渲染末时刻作为默认视图。由 run_btn 的 .then() 调用。
    """
    cache = _result_cache.get(sid)
    if not cache or cache.get("times") is None or len(cache["times"]) == 0:
        return (gr.update(), gr.update(), gr.update(), gr.update(),
                None, "⚠️ 本次运行没有可交互的溶质运移结果。")

    times = cache["times"]
    t_min, t_max = float(times[0]), float(times[-1])
    # 滑块步长取最小输出间隔：拖动时自然落在真实输出时刻上；
    # 想看非输出时刻（如 45000 之外的值）用数字框输入，那里走插值。
    diffs = np.diff(times)
    step = float(diffs.min()) if len(diffs) else 1.0
    step = max(step, 1.0)

    fig, status, vmin_o, vmax_o = _render_plume(t_max, True, None, None, sid)
    return (gr.update(minimum=t_min, maximum=t_max, value=t_max, step=step),
            gr.update(value=t_max),
            gr.update(value=vmin_o), gr.update(value=vmax_o),
            fig, status)


# ============================================================
# 参数列表定义（在"建模对话"和"参数审查"之间显示的可编辑表格）
# ============================================================

PARAM_TABLE_DEF = [
    # (dot_path, display_name, unit, priority)
    ("project.name",           "项目名称",          "",      "P1"),
    ("project.model_type",     "模拟类型",          "",      "P1"),
    ("domain.x_range",         "东西范围",          "m",     "P1"),
    ("domain.y_range",         "南北范围",          "m",     "P1"),
    ("grid.nrow",              "网格行数",          "",      "P1"),
    ("grid.ncol",              "网格列数",          "",      "P1"),
    ("aquifer.type",           "含水层类型",        "",      "P1"),
    ("aquifer.parameters.hk",  "渗透系数",          "m/d",   "P1"),
    ("boundary_conditions",    "边界条件",          "",      "P1"),
    ("grid.top",               "顶板高程",          "m",     "P2"),
    ("grid.bottom",            "底板高程",          "m",     "P2"),
    ("stress_periods",         "应力期设置",        "",      "P2"),
    ("aquifer.parameters.sy",  "给水度",            "",      "P2"),
    ("aquifer.parameters.ss",  "储水系数",          "1/m",   "P2"),
    ("initial_conditions",     "初始条件",          "",      "P2"),
    ("aquifer.parameters.porosity", "孔隙度",       "",      "P3"),
    ("aquifer.parameters.vka", "垂向渗透系数",      "m/d",   "P3"),
    ("transport.dispersion.al","纵向弥散度",        "m",     "P3"),
    ("grid.delr",              "网格行宽",          "m",     "P3"),
    ("grid.delc",              "网格列宽",          "m",     "P3"),
    ("grid.nlay",              "网格层数",          "",      "P3"),
]

def _dot_get(ir: dict, dot_path: str):
    """按点路径从 IR 读取值，支持 bare value 和 dict 包装"""
    parts = dot_path.split(".")
    current = ir
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    if isinstance(current, dict):
        return current.get("value", current)
    return current

def _format_value_for_table(val) -> str:
    """格式化为表格显示字符串"""
    if val is None:
        return ""
    if isinstance(val, dict):
        inner = val.get("value", val)
        if isinstance(inner, (int, float)):
            return str(inner)
        return json.dumps(inner, ensure_ascii=False) if isinstance(inner, (list, dict)) else str(inner)
    if isinstance(val, (list, dict)):
        return json.dumps(val, ensure_ascii=False)
    if isinstance(val, bool):
        return str(val)
    return str(val)

def _parse_table_value(raw: str, dot_path: str):
    """解析用户编辑的字符串回 Python 类型"""
    raw = raw.strip()
    if not raw:
        return None, None
    # 列表/字典类型
    if dot_path in ("domain.x_range", "domain.y_range"):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list) and len(parsed) == 2:
                return parsed, None
            return None, f"需要 [min, max] 格式，如 [0, 1000]"
        except Exception:
            return None, f"JSON 格式错误，需要 [min, max]"
    if dot_path == "boundary_conditions":
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed, None
            return None, "边界条件需要 JSON 数组格式"
        except Exception:
            return None, "JSON 格式错误"
    if dot_path == "stress_periods":
        try:
            parsed = json.loads(raw)
            return parsed, None
        except Exception:
            return None, "JSON 格式错误"
    if dot_path in ("project.model_type", "aquifer.type"):
        return raw, None  # 字符串值
    # 数值类型
    try:
        if "." in raw:
            return float(raw), None
        return int(raw), None
    except ValueError:
        return raw, None  # 可能是字符串

def _get_param_status(dot_path: str, value) -> str:
    """判断参数状态"""
    if dot_path == "boundary_conditions":
        if isinstance(value, list) and len(value) > 0:
            return "✅ 已设置"
        return "❌ 缺失"
    if dot_path == "stress_periods":
        if value and (isinstance(value, (list, dict)) and len(value) > 0):
            return "✅ 已设置"
        return "❌ 缺失"
    if value is not None and value != "" and value != []:
        return "✅ 已设置"
    return "❌ 缺失"

def build_param_table(json_ir: dict) -> pd.DataFrame:
    """从 JSON IR 构建参数列表 DataFrame"""
    if not isinstance(json_ir, dict):
        json_ir = {}
    rows = []
    for dot_path, display_name, unit, priority in PARAM_TABLE_DEF:
        val = _dot_get(json_ir, dot_path)
        formatted = _format_value_for_table(val)
        status = _get_param_status(dot_path, formatted)
        rows.append({
            "参数名": display_name,
            "当前值": formatted,
            "单位": unit,
            "优先级": priority,
            "状态": status,
        })
    return pd.DataFrame(rows)

def merge_table_to_ir(table_data: list, current_ir: dict) -> dict:
    """将表格编辑数据合并回 JSON IR"""
    import pandas as pd
    ir = copy.deepcopy(current_ir) if current_ir else {"ir_version": "1.0"}

    if isinstance(table_data, dict):
        df = pd.DataFrame(table_data)
    elif isinstance(table_data, list):
        if len(table_data) == 0:
            return ir
        if isinstance(table_data[0], dict):
            df = pd.DataFrame(table_data)
        else:
            # list of lists (Gradio 6 format)
            cols = ["参数名", "当前值", "单位", "优先级", "状态"]
            df = pd.DataFrame(table_data, columns=cols[:len(table_data[0])])
    else:
        return ir

    for _, row in df.iterrows():
        display_name = row.get("参数名", "")
        raw_value = str(row.get("当前值", "")).strip()
        if not raw_value:
            continue  # 空值跳过

        # 找到对应的 dot_path
        dot_path = None
        for dp, dn, _, _ in PARAM_TABLE_DEF:
            if dn == display_name:
                dot_path = dp
                break
        if not dot_path:
            continue

        parsed, err = _parse_table_value(raw_value, dot_path)
        if err or parsed is None:
            continue

        # 写入 IR
        parts = dot_path.split(".")
        current = ir
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]

        final_key = parts[-1]
        # 参数字段包装
        if "parameters" in dot_path or "dispersion" in dot_path:
            current[final_key] = {"value": parsed, "unit": row.get("单位", ""), "source": "table_edit"}
        elif dot_path in ("domain.x_range", "domain.y_range"):
            current[final_key] = parsed
        elif dot_path in ("grid.delr", "grid.delc", "grid.nrow", "grid.ncol", "grid.nlay", "grid.top", "grid.bottom"):
            current[final_key] = parsed
        elif dot_path in ("project.name", "project.model_type", "aquifer.type"):
            current[final_key] = parsed
        elif isinstance(parsed, (list, dict)):
            current[final_key] = parsed
        else:
            current[final_key] = parsed

    return ir

def param_transfer_handler(table_data, json_ir, sid):
    """参数传递按钮处理：将表格编辑写入 IR"""
    if not isinstance(json_ir, dict):
        json_ir = {}
    updated_ir = merge_table_to_ir(table_data, json_ir)

    # 更新会话状态
    dm = get_session(sid)
    dm.state["json_ir"] = updated_ir
    # 重新检测缺失参数
    missing = intent_parser.detect_missing_params(updated_ir)
    dm.state["missing_params"] = missing
    dm.save_session(sid)

    # 重建表格
    new_table = build_param_table(updated_ir)

    # 构建缺失参数高亮
    highlighted = []
    for p in missing.get("priority_1", []):
        highlighted.append((f"P1 {p}", "blocking"))
    for p in missing.get("priority_2", []):
        highlighted.append((f"P2 {p}", "important"))
    for p in missing.get("priority_3", []):
        highlighted.append((f"P3 {p}", "optional"))

    set_count = sum(1 for _, _, _, _, s in new_table.itertuples(index=False) if "✅" in str(s))
    total = len(PARAM_TABLE_DEF)
    logger.info(f"参数传递: {set_count}/{total} 参数已设置")

    return new_table, updated_ir, highlighted

# ============================================================
# 对话处理
# ============================================================

def chat_handler(message: str, history: list, sid: str):
    """处理用户自然语言输入 — Gradio 6 格式"""
    monitor.step(f"收到用户消息: {message[:60]}...")
    # history is list of {"role": ..., "content": ...} in Gradio 6
    if not message or not message.strip():
        return history, {}, [], "", sid

    dm = get_session(sid)

    # ── 会话恢复提示：首次交互时告知用户已恢复的状态 ──
    was_restored = getattr(dm, '_just_restored', False)
    dm._just_restored = False

    msg_lower = message.strip().lower()
    if msg_lower in ("开始建模", "运行", "run", "start"):
        if was_restored:
            history.append({"role": "assistant",
                "content": f"📌 **会话已恢复** — 项目: {dm.state.get('json_ir', {}).get('project', {}).get('name', '?')}, "
                           f"参数完整，无需重新输入。正在启动建模..."})
        return _run_model_cmd(history, dm, sid)
    elif msg_lower in ("使用建议值", "自动补全", "auto fill"):
        return _apply_suggestions_cmd(history, dm, sid)
    elif msg_lower in ("清空", "重置", "reset"):
        dm.reset(sid, delete_file=True)
        # 结果缓存必须一起清：缓存里留着上一轮的 gwt（MF6 模型对象）与时刻表，
        # 不清的话"对话已重置"之后污染羽 Tab 仍会画出上一轮的结果，前后矛盾。
        _result_cache.pop(sid, None)
        history.append({"role": "assistant", "content": "对话已重置。请描述您的建模需求。"})
        return history, {}, [], "intake", sid

    if was_restored:
        proj_name = dm.state.get('json_ir', {}).get('project', {}).get('name', '?')
        n = len(dm.state.get('history', []))
        history.append({"role": "assistant",
            "content": f"📌 **会话已恢复**（{n} 轮历史，项目: {proj_name}）。您可以继续之前的对话，或直接输入 **\"开始建模\"**。"})

    try:
        result = dm.process(message, sid)
    except Exception as e:
        monitor.capture_error(e, "对话处理")
        reply = f"处理出错: {e}\n请重试或简化描述。"
        history.append({"role": "user", "content": message})
        # 防御：确保 json_ir 是 dict
        json_ir = dm.state.get("json_ir", {})
        if not isinstance(json_ir, dict):
            json_ir = {}
        return history, json_ir, [], dm.state.get("phase", "intake"), sid

    reply = result["reply"]
    missing = result["missing_params"]

    highlighted = []
    for p in missing.get("priority_1", []):
        highlighted.append((f"P1 {p}", "blocking"))
    for p in missing.get("priority_2", []):
        highlighted.append((f"P2 {p}", "important"))
    for p in missing.get("priority_3", []):
        highlighted.append((f"P3 {p}", "optional"))

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    return history, result["json_ir"], highlighted, result["phase"], sid


def _apply_suggestions_cmd(history, dm, sid):
    try:
        reply = dm.apply_all_suggestions(sid)
    except Exception as e:
        logger.error(f"apply_all_suggestions 失败: {e}")
        logger.error(traceback.format_exc())
        reply = f"❌ 自动补全失败: {e}\n请重试或手动输入缺失参数。"
        history.append({"role": "user", "content": "使用建议值"})
        history.append({"role": "assistant", "content": reply})
        return history, dm.state.get("json_ir", {}), [], dm.state.get("phase", "intake"), sid

    missing = dm.state["missing_params"]
    highlighted = []
    for p in missing.get("priority_1", []):
        highlighted.append((f"P1 {p}", "blocking"))
    for p in missing.get("priority_2", []):
        highlighted.append((f"P2 {p}", "important"))
    for p in missing.get("priority_3", []):
        highlighted.append((f"P3 {p}", "optional"))
    history.append({"role": "user", "content": "使用建议值"})
    history.append({"role": "assistant", "content": reply})
    return history, dm.state["json_ir"], highlighted, dm.state["phase"], sid


def _run_model_cmd(history, dm, sid):
    json_ir = dm.state["json_ir"]
    project = json_ir.get("project", {})
    if not project.get("name") and not json_ir.get("aquifer", {}).get("type"):
        reply = "⚠️ 参数信息不足。请先描述建模需求（如：模拟一个承压含水层...）。"
        history.append({"role": "user", "content": "开始建模"})
        history.append({"role": "assistant", "content": reply})
        return history, json_ir, [], dm.state["phase"], sid

    # 缺失参数检测（优先于物理校验，实时从 JSON IR 获取）
    missing = intent_parser.detect_missing_params(json_ir)
    p1_missing = missing.get("priority_1", [])
    p2_missing = missing.get("priority_2", [])

    if p1_missing or p2_missing:
        reply = render_missing_params_guidance(p1_missing, p2_missing, missing.get("priority_3", []))
        history.append({"role": "user", "content": "开始建模"})
        history.append({"role": "assistant", "content": reply})
        return history, json_ir, [], dm.state["phase"], sid

    val_result = quick_validate(json_ir)
    if val_result["error_count"] > 0:
        issues = "\n".join(
            f"  {i['param_name']}: {i['description']}"
            for i in val_result["issues"] if i["severity"] == "error"
        )
        reply = f"❌ 参数校验发现 {val_result['error_count']} 个错误:\n{issues}"
        history.append({"role": "user", "content": "开始建模"})
        history.append({"role": "assistant", "content": reply})
        return history, json_ir, [], dm.state["phase"], sid

    # 警告虽不阻断，但要在**这条回复里**说出来 —— 只写 logger/monitor 等于没说：
    # "模拟总时长为 150000 d（410.7 年）"这类提示，用户看不到就永远不知道
    # 该去改 total_time，只会觉得"程序怎么还是按 150000 天跑"。
    warn_lines = [
        f"  ⚠️ {i.get('param_name', '?')}: {i.get('description', '')}"
        for i in (val_result.get("issues") or [])
        if i.get("severity") == "warning"
    ]
    reply = "✅ 参数校验通过！正在启动自动建模...\n请查看下方 **📊 建模结果** 面板。"
    if warn_lines:
        reply += f"\n\n⚠️ 另有 {len(warn_lines)} 条警告（不阻断建模）:\n" + "\n".join(warn_lines)
    history.append({"role": "user", "content": "开始建模"})
    history.append({"role": "assistant", "content": reply})
    dm.state["phase"] = "execution"
    return history, json_ir, [], dm.state["phase"], sid

# ============================================================
# 文件上传
# ============================================================

def file_parse_handler(files, history, sid):
    """处理文件上传 — 解析并持久化到会话状态"""
    if not files:
        return history, {}, ""
    file_list = files if isinstance(files, list) else [files]
    result = file_handler.parse_files(file_list)
    reply = result.get("summary", "文件已接收")

    # ── 关键修复：将文件数据持久化到 DM 状态，不再丢弃 ──
    dm = get_session(sid)
    dm.state["file_uploads"] = {
        "shapefiles": result.get("shapefiles", {}),
        "excel_data": result.get("excel_data", {}),
        "csv_data": result.get("csv_data", {}),
        "raster_data": result.get("raster_data", {}),
    }
    # 保存实际文件路径（用于可视化和建模）
    all_paths = [f.name for f in file_list if hasattr(f, 'name')]
    # 加入 ZIP 中解压出的文件路径
    all_paths.extend(result.get("extracted_file_paths", []))
    dm.state["_uploaded_file_paths"] = all_paths
    dm.state["_parsed_files_dir"] = str(file_handler.output_dir)
    dm.save_session(sid)

    # 将 SHP 范围信息注入 IR（自动设定模型 domain）
    # 优先级：1) 名称含"boundary"/"边界" 2) 名称含"Boundary" 3) 多边形面要素
    ir = dm.state.get("json_ir", {})
    boundary_shp = None
    for name, info in result.get("shapefiles", {}).items():
        name_lower = name.lower()
        if 'boundary' in name_lower or '边界' in name or 'border' in name_lower:
            if info.get("bounds") and len(info["bounds"]) >= 4:
                # 检查 bounds 是否有效（不是 NaN）
                b = info["bounds"]
                if all(isinstance(x, (int, float)) and not (x != x) for x in b[:4]):
                    boundary_shp = (name, info)
                    break

    # 回退：查找多边形面要素
    if not boundary_shp:
        for name, info in result.get("shapefiles", {}).items():
            if info.get("suggested_use") == "model_boundary_or_zones":
                if info.get("bounds") and len(info["bounds"]) >= 4:
                    b = info["bounds"]
                    if all(isinstance(x, (int, float)) and not (x != x) for x in b[:4]):
                        boundary_shp = (name, info)
                        break

    if boundary_shp:
        name, info = boundary_shp
        bounds = info["bounds"]
        # 用 SHP 坐标覆盖模型域
        if "domain" not in ir or not isinstance(ir.get("domain"), dict):
            ir["domain"] = {}
        ir["domain"]["x_range"] = [bounds[0], bounds[2]]
        ir["domain"]["y_range"] = [bounds[1], bounds[3]]
        # 网格原点 = SHP 外框左下角。放在同一个 bounds、同一个分支里算，
        # 原点与域范围因此不可能分叉。
        #
        # 为什么写进 IR 而不是让建模侧再探测一次：`_detect_basemap_inputs`
        # 在 run_model_handler 里被调用，**晚于** build_and_run；而源区选格
        # （app/modeling/source_zone.py）必须与可视化用同一个原点，否则
        # "模型补给的格子"和"图上画在填埋场轮廓内的格子"会错开。写进 IR 后
        # 建模侧只读 IR，不必给 build_and_run 加参数（那会被 debug_monitor
        # 里 monkeypatch 的 wrapper 吞掉）。
        ir["domain"]["origin_xy"] = [bounds[0], bounds[1]]
        # 标记边界 SHP 路径，供可视化裁剪使用
        dm.state["_boundary_shp_name"] = name
        reply += f"\n🌐 已从 **{name}** 提取模型范围: X=[{bounds[0]:.0f}, {bounds[2]:.0f}], Y=[{bounds[1]:.0f}, {bounds[3]:.0f}]"
    else:
        logger.warning("未找到有效的边界 SHP 文件，将使用手动输入的模型范围")

    # 将 CSV 水头数据注入 IR
    for name, info in result.get("csv_data", {}).items():
        if info.get("n_rows", 0) > 0 and "preview" in info:
            preview = info["preview"]
            reply += f"\n📊 已读取 **{name}**: {info['n_rows']} 行 × {info['n_cols']} 列"
            # 存入 IR 供 modeling 使用
            csv_key = f"_csv_{Path(name).stem}"
            ir[csv_key] = {
                "columns": info.get("columns", []),
                "data": preview,
                "n_rows": info["n_rows"],
            }

    dm.state["json_ir"] = ir
    dm.save_session(sid)

    history.append({"role": "user", "content": "上传文件"})
    history.append({"role": "assistant", "content": reply})
    return history, ir, reply

# ============================================================
# 建模执行
# ============================================================

def _planned_duration_text(json_ir: dict) -> str:
    """
    IR 里**配置**的总时长（不是 .ucn 里跑出来的那个），用于和实际结果对账。

    为什么需要它：修复前用户看到的时间轴末刻是 150000 d，而提示词要的是 15000 d，
    界面上**没有任何地方**能把"我配的时间"和"跑出来的时间"摆在一起看，
    于是只能看到"数不对"却看不到"哪一段配错了"。
    """
    stress = merge_time_sources(json_ir or {})
    try:
        nper = int(stress.get("nper") or 1)
    except (TypeError, ValueError):
        nper = 1

    period_data = stress.get("period_data") or []
    if period_data:
        try:
            total = sum(
                float(p.get("perlen", 1.0) or 1.0)
                for p in period_data if isinstance(p, dict)
            )
            return f"{total:g} d（period_data {len(period_data)} 期）"
        except (TypeError, ValueError):
            return f"{len(period_data)} 期（period_data 的 perlen 无法解析）"

    total_time = stress.get("total_time")
    if total_time:
        try:
            return f"{float(total_time):g} d（total_time，{nper} 期等分）"
        except (TypeError, ValueError):
            pass

    perlen = stress.get("perlen")
    if perlen:
        try:
            perlen = float(perlen)
            return f"{perlen * nper:g} d（{nper} 期 × 每期 {perlen:g} d）"
        except (TypeError, ValueError):
            pass

    return "未给出（按默认 1 期 1 天）"


def _warning_card_html(val_result: dict) -> str:
    """
    把校验**警告**渲染成界面上可见的琥珀色卡片。

    此前这里只渲染 `error_count > 0` 的错误，警告只落进 `monitor.info` 的计数，
    从不进界面 —— 于是"模拟总时长为 150000 d（410.7 年），请确认是否符合预期。
    （若 15000 天其实是**总**模拟时长，请改写成 total_time）"这条**算得出、
    也生成了**的提示永远没人看得见。用户只看到结果不对，看不到原因。
    这与 `logger.info` 孤点提示是同一类失效：说了，但没人听得见。
    """
    warns = [i for i in (val_result.get("issues") or [])
             if i.get("severity") == "warning"]
    if not warns:
        return ""

    # 分信噪比展示：East Texas 的 IR 一次能出 12 条警告，其中 8 条是
    # "参数'PCG'为非数值字符串，跳过范围检查（可能是布尔/枚举值）" —— 每个枚举型
    # 参数都来一条。全长铺开会把唯一要紧的那条（总时长 150000 d）淹掉，
    # 那和看不见没区别。所以低优先的折叠成一行计数，重点的逐条展开。
    low_priority, important = [], []
    for i in warns:
        (low_priority if "跳过范围检查" in (i.get("description") or "")
         else important).append(i)

    rows = "<br>".join(
        f"· <b>{i.get('param_name', '?')}</b>: {i.get('description', '')}"
        for i in important
    )
    if low_priority:
        more = f"<br><span style='color:#8d6e00'>另有 {len(low_priority)} 条枚举/字符串类" \
               f"参数未做范围检查的低优先提示（不影响建模）</span>"
    else:
        more = ""
    return (
        "<div style='margin-top:6px;padding:6px;background:#fff8e1;"
        "border-left:3px solid #ffb300'>"
        f"<b style='color:#8d6e00'>参数校验警告 {len(warns)} 条</b>"
        f"{'（' + str(len(important)) + ' 条需留意）' if low_priority else ''}"
        f"<br>{rows}{more}</div>"
    )


def run_model_handler(json_ir: dict, sid: str = "default"):
    """运行自动建模"""
    monitor.step("===== 开始建模 =====")
    monitor.info(f"项目: {json_ir.get('project', {}).get('name', '?')}")
    # 防御：创建空白占位图
    try:
        empty_fig = plt.figure(figsize=(4, 3))
        plt.text(0.5, 0.5, "等待输入...", ha="center", va="center")
        plt.tight_layout()
    except Exception:
        empty_fig = None

    # 防御：确保 json_ir 是 dict
    if not isinstance(json_ir, dict):
        logger.error(f"run_model_handler 收到非dict的 json_ir: {type(json_ir).__name__}")
        return empty_fig, f"<p style='color:red'>参数数据异常，请重新输入建模需求</p>", empty_fig, empty_fig, None, "# 参数数据异常", monitor.get_recent_logs(40)

    if not json_ir or not json_ir.get("project", {}).get("name"):
        return empty_fig, "<p>请先在左侧对话区输入建模需求，完成参数补全后再开始建模</p>", empty_fig, empty_fig, None, "# 等待建模...", monitor.get_recent_logs(40)

    # ================================================================
    # 建模前缺失参数检测（阻断建模前提醒用户）
    # ================================================================
    monitor.step("Step 1/3: 缺失参数检测")
    missing = intent_parser.detect_missing_params(json_ir)
    p1_missing = missing.get("priority_1", [])
    p2_missing = missing.get("priority_2", [])
    p3_missing = missing.get("priority_3", [])
    monitor.info(f"P1缺失={len(p1_missing)}, P2缺失={len(p2_missing)}, P3缺失={len(p3_missing)}")

    if p1_missing or p2_missing:
        monitor.warn(f"存在关键缺失参数: P1={p1_missing}, P2={p2_missing}")
        warning_html = render_missing_params_guidance(p1_missing, p2_missing, p3_missing, format="html")
        return empty_fig, warning_html, empty_fig, empty_fig, None, "# 等待建模...", monitor.get_recent_logs(40)

    monitor.step("Step 2/3: 物理参数校验")
    try:
        val_result = quick_validate(json_ir)
        monitor.info(f"校验结果: valid={val_result.get('valid')}, errors={val_result.get('error_count')}, warnings={val_result.get('warning_count')}")
    except Exception as e:
        monitor.capture_error(e, "物理参数校验")
        return empty_fig, f"<p style='color:red'>参数校验失败: {e}</p>", empty_fig, empty_fig, None, "# 等待建模...", monitor.get_recent_logs(40)

    if val_result["error_count"] > 0:
        issues = "\n".join(
            f"  {i['param_name']}: {i['description']}"
            for i in val_result["issues"] if i.get("severity") == "error"
        )
        monitor.warn(f"校验发现 {val_result['error_count']} 个错误")
        return empty_fig, f"<p style='color:red'>参数校验发现 {val_result['error_count']} 个错误:</p><pre>{issues}</pre>", empty_fig, empty_fig, None, "# 等待建模...", monitor.get_recent_logs(40)

    # 警告不阻断建模，但**必须可见**（此前只渲染错误，警告数只进 monitor 计数）
    warn_html = _warning_card_html(val_result)
    if warn_html:
        monitor.warn(
            f"校验警告 {val_result.get('warning_count', 0)} 条（已显示在结果面板）"
        )

    # ================================================================
    # 源区定位闸（置于建模前，避免把点源静默放大成全域面源）
    # ================================================================
    # East Texas 案例的污染羽之所以看起来像水流模型结果，根因就在这里：
    # 填埋场只占 132 格（0.59%），但带浓度的 RCH 边界找不到注入位置时，
    # 旧代码会**静默**展开成 layer0 全网格（22454 格），源被放大 170 倍。
    # 现在解析不出位置就**硬拒绝**，把原因与三条出路直接摆给用户。
    source_zone_rows = []
    try:
        dm = get_session(sid)
        sz_ir, sz_resolved, sz_issues = materialize_source_zones(
            json_ir,
            uploaded_paths=dm.state.get("_uploaded_file_paths", []),
            file_index=(dm.state.get("file_uploads") or {}).get("shapefiles", {}),
        )
    except SourceZoneUnresolved as e:
        sz_issues = [("?", str(e))]
        sz_resolved = {}
    except Exception as e:
        monitor.capture_error(e, "源区定位")
        sz_issues = [("?", f"源区定位时发生意外错误：{e}")]
        sz_resolved = {}

    if sz_issues:
        monitor.warn(f"源区定位失败: {[n for n, _ in sz_issues]}")
        detail = "\n\n".join(f"【{n}】\n{msg}" for n, msg in sz_issues)
        return (
            empty_fig,
            "<p style='color:red'><b>源区定位失败，已阻止建模</b>"
            "（避免把点源静默放大成全域面源，那正是污染羽看起来像水流模型结果的原因）:</p>"
            f"<pre style='white-space:pre-wrap'>{detail}</pre>",
            empty_fig, empty_fig, None, "# 等待建模...", monitor.get_recent_logs(40),
        )

    if sz_resolved:
        # 结果已物化进 IR（geometry.grid_indices），写回会话，重跑不必依赖上传文件
        dm.state["json_ir"] = json_ir
        dm.save_session(sid)
        _grid = json_ir.get("grid", {}) or {}
        _total = max(int(_grid.get("nrow", 1) or 1) * int(_grid.get("ncol", 1) or 1), 1)
        _nrow, _ncol, _delr, _delc, _ = grid_dims(_grid)
        # 补给量 = 格数 × 单格面积 × 速率。单格面积取 delr/delc 的均值乘积，
        # 不写死 20×20 —— 换个网格就是另一个数。
        _cell_area = float(np.mean(_delr)) * float(np.mean(_delc))
        for _n, _r in sz_resolved.items():
            _cells = len(_r.get("grid_indices") or [])
            _rate = None
            for _bc in json_ir.get("boundary_conditions", []) or []:
                if isinstance(_bc, dict) and _bc.get("name") == _n:
                    _raw = (_bc.get("parameters") or {}).get("rate")
                    _rate = _raw.get("value") if isinstance(_raw, dict) else _raw
                    break
            try:
                _flux_txt = f"，补给 {_cells * float(_rate) * _cell_area:.2f} m³/d"
            except (TypeError, ValueError):
                _flux_txt = ""
            source_zone_rows.append(
                f"源区 '{_n}': {_cells} 格 (占全域 {_cells/_total*100:.2f}%)，"
                f"面积 {_r.get('area_m2', 0):.0f} m²{_flux_txt}"
                f"，来源 {_r.get('resolved_from', '?')}"
            )
            monitor.info(source_zone_rows[-1])

    monitor.step("Step 3/3: FloPy 建模与 MODFLOW 6 模拟")

    # ── 诊断日志：建模前 dump IR 关键信息 ──
    logger.info(f"[RUN_MODEL] project={json_ir.get('project', {})}")
    logger.info(f"[RUN_MODEL] transport keys={list(json_ir.get('transport', {}).keys())}")
    logger.info(f"[RUN_MODEL] transport.enabled={json_ir.get('transport', {}).get('enabled')}")
    logger.info(f"[RUN_MODEL] transport={json_ir.get('transport', {})}")

    workspace = settings.workspace_dir / f"run_{sid}" if sid else settings.workspace_dir / "run_default"
    workspace.mkdir(parents=True, exist_ok=True)

    try:
        builder = AutoModelBuilder(str(workspace))
        result = builder.build_and_run(json_ir, "auto_model")

        if not result.get("success"):
            error_msg = result.get('error', 'Unknown')
            monitor.error(f"模型运行失败: {error_msg}")
            logger.error(f"模型运行失败: {error_msg}")
            return empty_fig, f"<p style='color:red'>模型运行失败: {error_msg}</p><p>请检查参数是否合理，或查看后台日志获取详情。</p>", empty_fig, empty_fig, None, "# 等待建模...", monitor.get_recent_logs(40)
        gwf = result.get("gwf")
        gwt = result.get("gwt")
        rp = ResultProcessor()
        viz_eng = VisualizationEngine()

        head_fig = empty_fig
        map_html = "<p>模型运行成功</p>" + warn_html
        plume_fig = empty_fig
        plume_single = empty_fig
        process_fig = empty_fig
        proc_result = None
        summary = {}

        if gwf:
            try:
                head = rp.extract_head_field(gwf, layer=0)
                summary = rp.generate_summary(gwf, gwt, json_ir)

                grid = json_ir.get("grid", {})
                delr_val = grid.get("delr", 20)
                delc_val = grid.get("delc", 20)
                # 安全处理 delr/delc（可能为 dict 或 标量）
                if isinstance(delr_val, dict):
                    delr_val = delr_val.get("value", 20)
                if isinstance(delc_val, dict):
                    delc_val = delc_val.get("value", 20)
                if not isinstance(delr_val, (int, float)):
                    delr_val = 20
                if not isinstance(delc_val, (int, float)):
                    delc_val = 20

                delr = np.full(grid.get("ncol", 50), delr_val)
                delc = np.full(grid.get("nrow", 40), delc_val)

                head_fig = viz_eng.plot_head_contour(head, delr, delc,
                    f"Head Distribution (Balance Error: {summary.get('water_balance_error', 0)*100:.3f}%)")

                # ── 底图叠加：检查是否上传了基底图/边界文件 ──
                # 探测逻辑已抽到 _detect_basemap_inputs()，污染羽图复用同一结果
                basemap_inputs = _detect_basemap_inputs(sid)
                basemap_path = basemap_inputs["basemap_path"]
                shp_path = basemap_inputs["shp_path"]
                clip_shp_path = basemap_inputs["clip_shp_path"]
                shp_bounds = basemap_inputs["shp_bounds"]

                if basemap_path or shp_bounds or shp_path or clip_shp_path:
                    try:
                        head_fig = viz_eng.plot_head_on_basemap(
                            head, delr, delc,
                            basemap_path=basemap_path,
                            shp_bounds=shp_bounds,
                            shp_path=shp_path,
                            clip_shp_path=clip_shp_path,
                            title=f"Head Distribution (Balance Error: {summary.get('water_balance_error', 0)*100:.3f}%)",
                        )
                        logger.info("底图叠加可视化完成")
                    except Exception as e:
                        logger.warning(f"底图叠加失败，使用默认可视化: {e}")
                        logger.warning(traceback.format_exc())

                # ── 保存图件上下文，供后续 NL 修改使用 ──
                plot_styler.save_context(head=head, delr=delr, delc=delc,
                                         wells=json_ir.get("wells"), plot_type="head")

                map_html = (
                    f"<h3>Model Results</h3>"
                    f"<b>Head Range:</b> {summary.get('head_min', 0):.2f} ~ {summary.get('head_max', 0):.2f} m<br>"
                    f"<b>Water Balance Error:</b> {summary.get('water_balance_error', 0)*100:.4f}%<br>"
                    f"<b>Total Inflow:</b> {summary.get('total_flow_in', 0):.2f} m3/d<br>"
                    f"<b>Total Outflow:</b> {summary.get('total_flow_out', 0):.2f} m3/d<br>"
                    # 配置时长与实际时长并排显示：两者不一致时用户当场就能看出来，
                    # 不用等到读时间轴末刻才发现"怎么是 150000 天"
                    f"<b>计划模拟时长（来自 IR）:</b> {_planned_duration_text(json_ir)}"
                ) + warn_html
                # 源区那一行是用户能亲眼看到的"修复生效"证据：取代了旧代码
                # 那句到不了任何 UI 组件的 logger.info
                if source_zone_rows:
                    map_html += (
                        "<div style='margin-top:6px;padding:6px;background:#eef7ee;"
                        "border-left:3px solid #4caf50'>"
                        "<b style='color:#2e7d32'>源区定位</b><br>"
                        + "<br>".join(source_zone_rows)
                        + "</div>"
                    )

                if gwt:
                    try:
                        # 真实模拟总时长直接读 .ucn 的最后一个时刻。
                        # 旧代码读 json_ir["stress_periods"]["perlen"]，而顶层
                        # stress_periods 常被 LLM 写成裸整数（East Texas 就是 10），
                        # 于是时间轴永远显示 1 d，看起来像没跑动。
                        conc_obj = gwt.output.concentration()
                        times = conc_obj.get_times()
                        times = [float(t) for t in np.atleast_1d(times)] if times is not None else []

                        if len(times) >= 2:
                            # 取 4~6 个时刻画演化序列，而不是只画最后一张快照——
                            # 单张快照看不出污染羽在移动
                            n_frames = min(6, len(times))
                            idx = np.unique(
                                np.linspace(0, len(times) - 1, n_frames).astype(int)
                            )
                            frames = [
                                rp.extract_concentration_field(gwt, layer=0, totim=times[i])
                                for i in idx
                            ]
                            frame_times = [times[i] for i in idx]
                        else:
                            frames = [rp.extract_concentration_field(gwt, layer=0)]
                            frame_times = times[-1:] or [1.0]

                        # 比流量场：污染羽图的流向箭头与主控过程辨识共用一份
                        velocity = rp.extract_velocity_field(gwf, layer=0)

                        plume_fig = viz_eng.plot_plume_evolution(
                            frames, frame_times, delr, delc,
                            threshold=0.01, n_cols=min(3, len(frames)),
                            basemap_path=basemap_path, shp_path=shp_path,
                            clip_shp_path=clip_shp_path, shp_bounds=shp_bounds,
                            velocity=velocity,
                        )

                        # 单时刻交互图的初始帧（末时刻）
                        if frames:
                            plume_single = viz_eng.plot_plume_at_time(
                                frames[-1], frame_times[-1], delr, delc,
                                threshold=0.01,
                                basemap_path=basemap_path, shp_path=shp_path,
                                clip_shp_path=clip_shp_path, shp_bounds=shp_bounds,
                                velocity=velocity,
                            )

                        # ── 主控过程辨识（"过程分析"Tab 的数据源） ──
                        if len(frames) >= 2 and velocity is not None:
                            proc_result = process_identifier.identify_dominant_process(
                                conc_history=frames,
                                # 稳态流场：速度不随时刻变化，按帧数复制即可
                                # （ProcessIdentifier 要求 len >= n_times - 1）
                                velocity_field=[velocity] * len(frames),
                                parameters=_extract_process_params(json_ir),
                                delr=delr, delc=delc,
                            )
                            process_fig = viz_eng.plot_process_analysis(proc_result)
                        else:
                            reason = ("可用输出时刻少于 2 个" if len(frames) < 2
                                      else "流速场不可得（水头场或渗透系数缺失）")
                            logger.warning(f"主控过程分析跳过: {reason}")
                            map_html += (f"<br><span style='color:orange'>"
                                         f"主控过程分析未生成：{reason}</span>")

                        # ── 缓存供交互式重绘 ──
                        c_max = float(max(np.max(c) for c in frames)) if frames else 0.0
                        _result_cache[sid] = {
                            "gwt": gwt, "gwf": gwf,
                            "times": np.asarray(times, dtype=float),
                            "delr": delr, "delc": delc, "layer": 0,
                            "c_range": (0.0, c_max),
                            "velocity": velocity,
                            "basemap": basemap_inputs,
                            "workspace": workspace,
                            "_frame_cache": {float(t): f
                                             for t, f in zip(frame_times, frames)},
                        }

                        # 让"图件修改"Tab 知道存在污染羽数据（_conc_frames 此前无人写）
                        plot_styler.save_context(head=head, delr=delr, delc=delc,
                                                 wells=json_ir.get("wells"),
                                                 conc_frames=frames, plot_type="head")
                    except Exception as e:
                        logger.warning(f"污染羽演化图绘制失败: {e}")
                        logger.warning(traceback.format_exc())

            except Exception as e:
                logger.error(f"结果提取失败: {e}")
                logger.error(traceback.format_exc())
                map_html = f"<p style='color:orange'>结果提取失败: {e}</p>"

        # 图件落盘：必须在 plt.close("all") 之前，且报告要引用这些路径。
        # 文件名用 ASCII、显示名用中文 —— _add_figures 把 key 当段落标题、value 当路径。
        viz_files = {}
        try:
            fig_dir = workspace / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)
            for label, stem, fig in (("水头分布", "head", head_fig),
                                     ("污染羽演化总览", "plume_evolution", plume_fig),
                                     ("污染羽单时刻", "plume_single", plume_single),
                                     ("主控过程分析", "process_analysis", process_fig)):
                if fig is None or fig is empty_fig:
                    continue
                p = fig_dir / f"{stem}.png"
                fig.savefig(p, dpi=150, bbox_inches="tight")
                viz_files[label] = str(p)
            logger.info(f"报告图件已落盘: {list(viz_files.keys())}")
        except Exception as e:
            logger.warning(f"图件落盘失败: {e}")

        report_path = None
        try:
            rg = ReportGenerator()
            report_path = rg.generate(json_ir=json_ir, validation_log=val_result.get("issues", []),
                                      sim_results=summary, model_id=sid or "default",
                                      process_analysis=proc_result,
                                      visualizations=viz_files or None)
        except Exception as e:
            logger.warning(f"报告生成失败: {e}")

        plt.close("all")

        # 生成代码预览
        code_str = _generate_code_preview(result, json_ir, workspace)

        monitor.ok("建模完成")
        return head_fig, map_html, plume_fig, process_fig, report_path, code_str, monitor.get_recent_logs(40)

    except Exception as e:
        monitor.capture_error(e, "建模执行异常")
        return empty_fig, f"<p style='color:red'>建模异常: {e}</p>", empty_fig, empty_fig, None, "# 等待建模...", monitor.get_recent_logs(40)
def _generate_code_preview(result: dict, json_ir: dict, workspace) -> str:
    """从建模结果和 JSON IR 生成 FloPy 代码预览"""
    lines = ["# FloPy MODFLOW 6 自动生成代码",
             f"# 项目: {json_ir.get('project', {}).get('name', '未命名')}",
             f"# 工作目录: {workspace}",
             ""]

    sim = result.get("sim")
    gwf = result.get("gwf")

    if sim:
        lines.append("# ===== 模拟对象 =====")
        lines.append(f"sim = flopy.mf6.MFSimulation(")
        lines.append(f"    sim_name='{json_ir.get('project', {}).get('name', 'model')}',")
        lines.append(f"    sim_ws='{workspace}',")
        lines.append(f"    continue_=False,")
        lines.append(f")")
        lines.append("")

    # 网格信息
    grid = json_ir.get("grid", {})
    domain = json_ir.get("domain", {})
    lines.append("# ===== 网格离散 (DIS) =====")
    nrow = grid.get("nrow", "?")
    ncol = grid.get("ncol", "?")
    nlay = grid.get("nlay", 1)
    delr = grid.get("delr", 20)
    delc = grid.get("delc", 20)
    top = grid.get("top", 100)
    botm = grid.get("bottom", 50)

    if isinstance(delr, dict): delr = delr.get("value", 20)
    if isinstance(delc, dict): delc = delc.get("value", 20)
    if isinstance(top, dict): top = top.get("value", 100)
    if isinstance(botm, dict): botm = botm.get("value", 50)

    lines.append(f"nrow, ncol, nlay = {nrow}, {ncol}, {nlay}")
    lines.append(f"delr = np.full({ncol}, {delr})  # 行宽")
    lines.append(f"delc = np.full({nrow}, {delc})  # 列宽")
    lines.append(f"top = np.full(({nrow}, {ncol}), {top})  # 顶板高程")
    lines.append(f"botm = np.full(({nlay}, {nrow}, {ncol}), {botm})  # 底板高程")
    lines.append("")
    lines.append("dis = flopy.mf6.ModflowGwfdis(")
    lines.append("    gwf, nrow=nrow, ncol=ncol, nlay=nlay,")
    lines.append("    delr=delr, delc=delc, top=top, botm=botm")
    lines.append(")")
    lines.append("")

    # 含水层参数
    aqua = json_ir.get("aquifer", {})
    params = aqua.get("parameters", {}) if isinstance(aqua, dict) else {}
    lines.append("# ===== 节点属性流 (NPF) =====")
    hk_val = _extract_val(params.get("hk", {}))
    lines.append(f"hk = {hk_val}  # 水平渗透系数 m/d")
    lines.append(f"npf = flopy.mf6.ModflowGwfnpf(gwf, k=hk)")
    lines.append("")

    # 储水
    atype = aqua.get("type", "confined") if isinstance(aqua, dict) else "confined"
    lines.append("# ===== 储水 (STO) =====")
    if atype == "confined":
        ss_val = _extract_val(params.get("ss", {}))
        lines.append(f"ss = {ss_val}  # 储水系数 1/m")
        lines.append(f"sto = flopy.mf6.ModflowGwfsto(gwf, ss=ss, steady_state={{True}})")
    else:
        sy_val = _extract_val(params.get("sy", {}))
        lines.append(f"sy = {sy_val}  # 给水度")
        lines.append(f"sto = flopy.mf6.ModflowGwfsto(gwf, sy=sy, steady_state={{True}})")
    lines.append("")

    # 边界条件
    bcs = json_ir.get("boundary_conditions", [])
    if isinstance(bcs, list) and bcs:
        lines.append("# ===== 边界条件 =====")
        lines.append(f"# 共 {len(bcs)} 个边界条件:")
        for bc in bcs:
            if isinstance(bc, dict):
                bc_type = bc.get("type", "?")
                bc_name = bc.get("name", "?")
                lines.append(f"#   - {bc_type}: {bc_name}")
        lines.append("")

    # 求解器
    lines.append("# ===== 求解器 (IMS) =====")
    lines.append("ims = flopy.mf6.ModflowIms(sim, complexity='simple')")
    lines.append("")
    lines.append("# ===== 运行模拟 =====")
    lines.append("sim.write_simulation()")
    lines.append("sim.run_simulation()")
    lines.append("")
    lines.append("# ===== 读取结果 =====")
    lines.append("head = gwf.output.head().get_data()")
    lines.append("budget = gwf.output.budget().get_data()")

    return "\n".join(lines)


def _extract_val(param_info) -> str:
    """从参数dict中提取数值"""
    if isinstance(param_info, dict):
        v = param_info.get("value", "?")
        return str(v) if v is not None else "?"
    return str(param_info) if param_info is not None else "?"


# ================================================================
# 图件修改处理
# ================================================================

def plot_modify_handler(nl_request: str, json_ir: dict):
    """处理用户的图件修改指令（自然语言）"""
    if not nl_request or not nl_request.strip():
        return None, "请输入修改需求，例如：'把颜色改成蓝色系'、'等值线间隔设为2m'"

    if not plot_styler.has_data():
        return None, "⚠️ 还没有图件数据。请先运行建模生成水头图，然后再修改。"

    try:
        new_fig, msg = plot_styler.apply_nl(nl_request.strip())
        return new_fig, msg
    except Exception as e:
        logger.error(f"图件修改失败: {e}")
        logger.error(traceback.format_exc())
        return None, f"❌ 图件修改失败: {e}"

# ================================================================
# UI
# ================================================================

def create_ui():
    with gr.Blocks(title="Groundwater LLM Modeling") as demo:
        sid_state = gr.State("default")

        gr.Markdown("""
        # 🌊 Physical-LLM Groundwater Intelligent Modeling System
        ### 物理约束嵌入LLM的地下水智能建模原型系统 (MVP)
        通过自然语言描述即可完成地下水数值建模 — 无需编写代码、操作 GUI。
        """)

        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown("### 💬 建模对话")
                chatbot = gr.Chatbot(label="对话记录", height=450)
                msg = gr.Textbox(label="输入建模需求...",
                    placeholder="例如：模拟承压含水层1000×800m，渗透系数15m/d，西侧100m东侧95m定水头...",
                    lines=2)
                with gr.Row():
                    send_btn = gr.Button("发送", variant="primary", scale=3)
                    file_upload = gr.File(label="📎 上传文件", file_count="multiple",
                        file_types=[".zip", ".shp", ".shx", ".dbf", ".prj", ".geojson", ".kml",
                                    ".xlsx", ".xls", ".csv", ".tif", ".tiff", ".asc",
                                    ".png", ".jpg", ".jpeg", ".bmp"])
                with gr.Row():
                    clear_btn = gr.Button("清空", variant="secondary", size="sm")
                    auto_fill_btn = gr.Button("建议补全", variant="secondary", size="sm")
                    run_btn = gr.Button("开始建模", variant="primary", size="sm")

                # ── 上传文件说明 ──
                with gr.Accordion("📖 上传文件格式说明", open=False):
                    gr.Markdown("""
### 📦 推荐方式：打包上传 ZIP 压缩包

将所有建模需要的文件（SHP、CSV、Excel、GeoTIFF 等）统一打包为一个 `.zip` 文件上传，
系统会自动解压并识别其中的数据文件。

---

### 支持的文件格式一览

| 格式 | 扩展名 | 建模用途 | 说明 |
|------|--------|----------|------|
| **ZIP 压缩包** | `.zip` | 📦 一键上传所有文件 | 推荐！将所有文件打包上传，系统自动识别 |
| **Shapefile 面** | `.shp` + `.shx` + `.dbf` + `.prj` | 🗺️ 模型边界、参数分区 | GMS/QGIS/ArcGIS 导出时请确保 4 个配套文件齐全 |
| **Shapefile 线** | `.shp` + `.shx` + `.dbf` + `.prj` | 🏞️ 河流弧段、GHB 边界 | 线要素会自动识别为河流/边界 |
| **Shapefile 点** | `.shp` + `.shx` + `.dbf` + `.prj` | 🏭 井位、观测点 | 点要素自动识别为井位或观测点 |
| **GeoJSON** | `.geojson` | 🗺️ 边界、要素 | 单一文件包含所有几何信息和属性 |
| **KML/KMZ** | `.kml` | 🌍 Google Earth 导出 | 支持从 Google Earth 导入边界 |
| **Excel** | `.xlsx`, `.xls` | 📊 井信息、调查点、地层 | 自动识别坐标列（x/y, 经度/纬度, E/N） |
| **CSV** | `.csv` | 📋 时间序列、点位 | 纯文本表格，适合大量数据 |
| **GeoTIFF** | `.tif`, `.tiff` | 🏔️ DEM 高程、参数分布、底图影像 | 含地理参考的栅格数据，会按真实地理坐标叠加 |
| **图片底图** | `.png`, `.jpg`, `.jpeg`, `.bmp` | 🗺️ 底图影像 | 无地理参考，按模型范围或 SHP 外框铺放 |
| **ASCII Grid** | `.asc` | 🏔️ 栅格数据 | ESRI ASCII 栅格格式 |

---

### ⚠️ 使用限制

- **文件总大小**：建议 ≤ 100 MB（过大文件会导致上传超时）
- **Shapefile 配套**：每个 `.shp` 必须附带同名的 `.shx`、`.dbf`、`.prj` 才能正确解析
- **编码**：CSV/Excel 推荐 UTF-8 编码；中文列名可自动识别
- **坐标系**：所有空间数据最好统一坐标系（推荐 UTM 投影或 WGS84），混合坐标系可能导致边界错位
- **井/点数据**：Excel 中请确保有坐标列（名称可为：x/y, 经度/纬度, longitude/latitude, easting/northing, 东/北）
- **压缩包**：ZIP 内的文件会被提取到临时目录，请确保 ZIP 内不包含大量无关文件
                    """)

            with gr.Column(scale=3):
                gr.Markdown("### 📊 参数列表")
                param_table = gr.Dataframe(
                    label="可编辑参数表（修改\"当前值\"列后点击下方按钮同步）",
                    value=build_param_table({}),
                    headers=["参数名", "当前值", "单位", "优先级", "状态"],
                    datatype=["str", "str", "str", "str", "str"],
                    column_count=5,
                    row_count=21,
                    interactive=True,
                )
                param_transfer_btn = gr.Button("📥 参数传递", variant="primary", size="sm")

                gr.Markdown("### 📋 参数审查")
                param_panel = gr.JSON(label="JSON IR", value={}, height=200)
                with gr.Row():
                    phase_text = gr.Textbox(label="会话阶段", value="intake", interactive=False)
                    missing_params = gr.HighlightedText(label="缺失参数", value=[],
                        color_map={"blocking": "red", "important": "orange", "optional": "green"})
                gr.Markdown("### 💻 代码预览")
                code_preview = gr.Code(label="FloPy 建模代码", language="python", lines=12,
                    value="# 等待建模...\n# 完成建模后将在此处显示自动生成的 FloPy 代码")

        gr.Markdown("---")
        gr.Markdown("### 📊 建模结果")
        with gr.Tabs():
            with gr.TabItem("📋 结果摘要"):
                result_text = gr.HTML(value="<p>等待建模...</p>")
            with gr.TabItem("💧 水头分布"):
                head_plot = gr.Plot(label="水头等值线")
            with gr.TabItem("☣️ 污染羽"):
                gr.Markdown("#### 演化总览")
                plume_plot = gr.Plot(label="污染羽演化总览")
                gr.Markdown("#### 单时刻交互图")
                with gr.Row():
                    plume_time_slider = gr.Slider(
                        minimum=0, maximum=1, value=1, step=1,
                        label="输出时刻 (d)",
                        info="拖动选择；落在两个输出时刻之间时自动线性插值")
                    plume_time_num = gr.Number(
                        value=None, precision=0, label="精确时刻 (d)",
                        info="可直接输入，例如 360")
                with gr.Row():
                    plume_auto_range = gr.Checkbox(value=True, label="自动色标范围")
                    plume_vmin = gr.Number(
                        value=None, label="浓度下限 (mg/L)",
                        info="填好后点右侧「确定重绘」或按回车")
                    plume_vmax = gr.Number(
                        value=None, label="浓度上限 (mg/L)",
                        info="填好后点右侧「确定重绘」或按回车")
                    plume_apply_btn = gr.Button("✅ 确定重绘", variant="primary",
                                                scale=0)
                plume_single_plot = gr.Plot(label="污染羽单时刻")
                plume_status = gr.Markdown(
                    "完成建模后可在此选择时刻与浓度范围；"
                    "上下限输入框只在点「确定重绘」或按回车时才重绘。")
            with gr.TabItem("📈 过程分析"):
                process_plot = gr.Plot(label="主控过程贡献")
            with gr.TabItem("🎨 图件修改"):
                gr.Markdown("### 🎨 图件样式修改\n用自然语言描述您想要的图件修改，例如：")
                gr.Markdown("> *\"把颜色改成蓝色系，加粗字体，等值线间隔设为2m\"*\n"
                           "> *\"隐藏流向箭头，标题改为'承压含水层水头分布'\"*\n"
                           "> *\"井标记改成蓝色圆形，加大尺寸\"*")
                plot_nl_input = gr.Textbox(
                    label="修改指令",
                    placeholder="例如：把颜色改成蓝色系，等值线间隔设为2米，标题字体改大",
                    lines=2)
                plot_modify_btn = gr.Button("🎨 应用修改", variant="primary", size="sm")
                plot_modify_msg = gr.Markdown("")
            with gr.TabItem("📄 报告"):
                report_download = gr.File(label="下载报告")

        # === 事件绑定 ===
        def on_send(msg, hist, sid):
            history, ir, miss, phase, sid2 = chat_handler(msg, hist, sid)
            table_df = build_param_table(ir)
            return history, ir, miss, phase, sid2, table_df

        def clear_msg(): return ""

        send_btn.click(on_send, [msg, chatbot, sid_state],
                       [chatbot, param_panel, missing_params, phase_text, sid_state, param_table]
                       ).then(clear_msg, None, [msg])
        msg.submit(on_send, [msg, chatbot, sid_state],
                   [chatbot, param_panel, missing_params, phase_text, sid_state, param_table]
                   ).then(clear_msg, None, [msg])
        file_upload.upload(file_parse_handler, [file_upload, chatbot, sid_state], [chatbot, param_panel, result_text])
        clear_btn.click(lambda sid: ([], {}, [], "intake", sid, build_param_table({})), [sid_state],
                        [chatbot, param_panel, missing_params, phase_text, sid_state, param_table])
        auto_fill_btn.click(lambda hist, sid: (*_apply_suggestions_cmd(hist, get_session(sid), sid),
                            build_param_table(get_session(sid).state.get("json_ir", {}))),
                            [chatbot, sid_state],
                            [chatbot, param_panel, missing_params, phase_text, sid_state, param_table])

        # 参数传递按钮
        param_transfer_btn.click(
            param_transfer_handler,
            [param_table, param_panel, sid_state],
            [param_table, param_panel, missing_params]
        )

        # ---- 调试日志面板 ----
        with gr.Accordion("🔍 调试日志 (点击展开)", open=False):
            debug_log = gr.Textbox(
                label=f"实时管道追踪 — 完整日志: {monitor.get_log_file_path()}",
                value="等待操作...",
                lines=15,
                max_lines=25,
                interactive=False,
            )

        # 按钮绑定（放在 debug_log 定义之后）
        # .then() 里同步污染羽交互控件：这样 run_model_handler 的 7 元组返回值
        # 完全不用动（它有 8 个 return 点，7 个是错误早退分支，逐个加元素易错）。
        run_btn.click(run_model_handler, [param_panel, sid_state],
                      [head_plot, result_text, plume_plot, process_plot, report_download, code_preview, debug_log]
                      ).then(
            _sync_plume_controls, [sid_state],
            [plume_time_slider, plume_time_num, plume_vmin, plume_vmax,
             plume_single_plot, plume_status])

        # 污染羽交互控件：滑块 ⇄ 数字框双向同步
        # 色标上下限**不挂 .change** —— gr.Number.change 逐字符触发：键入 300 时
        # 敲下 "3" 就开始算，用户得等这一次结果回来才能接着键入 "0"。改为显式确认
        # （按钮点击 / 输入框回车）。复选框也走同一个处理器，单击一次即生效、不会
        # 逐字符误触发。全部不绑 .change 也顺带消除了"程序回写输入框再触发事件"的回环。
        plume_time_slider.release(
            _on_plume_slider,
            [plume_time_slider, plume_auto_range, plume_vmin, plume_vmax, sid_state],
            [plume_single_plot, plume_status, plume_time_num, plume_vmin, plume_vmax])
        plume_time_num.submit(
            _on_plume_number,
            [plume_time_num, plume_auto_range, plume_vmin, plume_vmax, sid_state],
            [plume_single_plot, plume_status, plume_time_slider, plume_vmin, plume_vmax])
        for _ev in (plume_apply_btn.click, plume_vmin.submit, plume_vmax.submit,
                    plume_auto_range.change):
            _ev(
                _on_plume_apply,
                [plume_time_slider, plume_time_num, plume_auto_range,
                 plume_vmin, plume_vmax, sid_state],
                [plume_single_plot, plume_status, plume_time_slider, plume_time_num,
                 plume_vmin, plume_vmax])

        # 图件修改按钮
        plot_modify_btn.click(
            plot_modify_handler,
            [plot_nl_input, param_panel],
            [head_plot, plot_modify_msg]
        )

        gr.Markdown("""
        ---
        ### 💡 快速开始
        > 我要开展一个项目名称为try的地下水流模拟工作，该模拟为单层非承压含水层的稳态模拟，模拟范围为东西0-1000m，南北0-800m的矩形范围，地层渗透系数15m/d，边界条件及初始条件为：西侧定水头100m，东侧定水头95m，南北侧为零流量边界。
        """)

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_PORT", "7860")),
        root_path=os.getenv("GRADIO_ROOT_PATH", ""),
        share=False,
    )

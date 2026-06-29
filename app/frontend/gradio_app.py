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
import copy
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from app.config import settings
from app.llm_agent.dialogue_manager import DialogueManager
from app.llm_agent.intent_parser import intent_parser
from app.knowledge.param_guide import render_missing_params_guidance
from app.llm_agent.file_handler import file_handler
from app.physics_engine.validator import quick_validate
from app.modeling.model_builder import AutoModelBuilder
from app.analysis.visualizer import VisualizationEngine, visualizer
from app.analysis.result_processor import ResultProcessor
from app.analysis.report_generator import ReportGenerator
from app.visualization.plot_styler import plot_styler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- 调试监控器 ----
from app.utils.debug_monitor import monitor
logger.info("调试监控器已启用 — 日志: %s", monitor.get_log_file_path())

# ============================================================
sessions = {}

def get_session(sid: str) -> DialogueManager:
    if sid not in sessions:
        sessions[sid] = DialogueManager()
    return sessions[sid]

# ============================================================
# 对话处理
# ============================================================

def chat_handler(message: str, history: list, sid: str):
    """处理用户自然语言输入 — Gradio 6 格式"""
    monitor.step(f"收到用户消息: {message[:60]}...")
    # history is list of {"role": ..., "content": ...} in Gradio 6
    if not message or not message.strip():
        return history, build_param_table({}), {}, [], "", sid

    dm = get_session(sid)

    msg_lower = message.strip().lower()
    if msg_lower in ("开始建模", "运行", "run", "start"):
        return _run_model_cmd(history, dm, sid)
    elif msg_lower in ("使用建议值", "自动补全", "auto fill"):
        return _apply_suggestions_cmd(history, dm, sid)
    elif msg_lower in ("清空", "重置", "reset"):
        dm.reset()
        history.append({"role": "assistant", "content": "对话已重置。请描述您的建模需求。"})
        return history, {}, [], "intake", sid

    try:
        result = dm.process(message)
    except Exception as e:
        monitor.capture_error(e, "对话处理")
        reply = f"处理出错: {e}\n请重试或简化描述。"
        history.append({"role": "user", "content": message})
        # 防御：确保 json_ir 是 dict
        json_ir = dm.state.get("json_ir", {})
        if not isinstance(json_ir, dict):
            json_ir = {}
        return history, build_param_table(json_ir), json_ir, [], dm.state.get("phase", "intake"), sid

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
    return history, build_param_table(result["json_ir"]), result["json_ir"], highlighted, result["phase"], sid


def _apply_suggestions_cmd(history, dm, sid):
    try:
        reply = dm.apply_all_suggestions()
    except Exception as e:
        logger.error(f"apply_all_suggestions 失败: {e}")
        logger.error(traceback.format_exc())
        reply = f"❌ 自动补全失败: {e}\n请重试或手动输入缺失参数。"
        history.append({"role": "user", "content": "使用建议值"})
        history.append({"role": "assistant", "content": reply})
        ir = dm.state.get("json_ir", {})
        return history, build_param_table(ir), ir, [], dm.state.get("phase", "intake"), sid

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
    ir = dm.state["json_ir"]
    return history, build_param_table(ir), ir, highlighted, dm.state["phase"], sid


def _run_model_cmd(history, dm, sid):
    json_ir = dm.state["json_ir"]
    if not isinstance(json_ir, dict):
        json_ir = {}
    param_df = build_param_table(json_ir)
    project = json_ir.get("project", {})
    if not project.get("name") and not json_ir.get("aquifer", {}).get("type"):
        reply = "⚠️ 参数信息不足。请先描述建模需求（如：模拟一个承压含水层...）。"
        history.append({"role": "user", "content": "开始建模"})
        history.append({"role": "assistant", "content": reply})
        return history, param_df, json_ir, [], dm.state["phase"], sid

    # 缺失参数检测（优先于物理校验，实时从 JSON IR 获取）
    missing = intent_parser.detect_missing_params(json_ir)
    p1_missing = missing.get("priority_1", [])
    p2_missing = missing.get("priority_2", [])

    if p1_missing or p2_missing:
        reply = render_missing_params_guidance(p1_missing, p2_missing, missing.get("priority_3", []))
        history.append({"role": "user", "content": "开始建模"})
        history.append({"role": "assistant", "content": reply})
        return history, param_df, json_ir, [], dm.state["phase"], sid

    val_result = quick_validate(json_ir)
    if val_result["error_count"] > 0:
        issues = "\n".join(
            f"  {i['param_name']}: {i['description']}"
            for i in val_result["issues"] if i["severity"] == "error"
        )
        reply = f"❌ 参数校验发现 {val_result['error_count']} 个错误:\n{issues}"
        history.append({"role": "user", "content": "开始建模"})
        history.append({"role": "assistant", "content": reply})
        return history, param_df, json_ir, [], dm.state["phase"], sid

    history.append({"role": "user", "content": "开始建模"})
    history.append({"role": "assistant", "content": "✅ 参数校验通过！正在启动自动建模...\n请查看下方 **📊 建模结果** 面板。"})
    dm.state["phase"] = "execution"
    return history, param_df, json_ir, [], dm.state["phase"], sid

# ============================================================
# 文件上传
# ============================================================

def file_parse_handler(files, history):
    if not files:
        return history, {}, ""
    file_list = files if isinstance(files, list) else [files]
    result = file_handler.parse_files(file_list)
    reply = result.get("summary", "文件已接收")
    history.append({"role": "user", "content": "上传文件"})
    history.append({"role": "assistant", "content": reply})
    return history, {}, reply

# ============================================================
# 建模执行
# ============================================================

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

    monitor.step("Step 3/3: FloPy 建模与 MODFLOW 6 模拟")
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
        map_html = "<p>模型运行成功</p>"
        plume_fig = empty_fig
        process_fig = empty_fig
        summary = {}

        if gwf:
            try:
                head = rp.extract_head_field(gwf, layer=0)
                summary = rp.generate_summary(gwf, gwt, json_ir)

                grid = json_ir.get("grid", {})
                # 安全提取 grid 数值（LLM 可能返回 {"value": X} 或裸值）
                def _gv(key, default):
                    v = grid.get(key, default)
                    if isinstance(v, dict): return v.get("value", default)
                    if v is None: return default
                    return v
                delr_val = _gv("delr", 20)
                delc_val = _gv("delc", 20)
                ncol_val = int(_gv("ncol", 50))
                nrow_val = int(_gv("nrow", 40))
                delr = np.full(ncol_val, delr_val)
                delc = np.full(nrow_val, delc_val)

                head_fig = viz_eng.plot_head_contour(head, delr, delc,
                    f"Head Distribution (Balance Error: {summary.get('water_balance_error', 0)*100:.3f}%)")

                # 提取井信息（防御 dict 包装: {"value": -200, "unit": "m3/d"}）
                wells_data = []
                for bc in json_ir.get("boundary_conditions", []):
                    if bc.get("type") == "wel":
                        geom = bc.get("geometry", {})
                        rate_raw = bc.get("parameters", {}).get("rate", 0)
                        if isinstance(rate_raw, dict):
                            rate_raw = rate_raw.get("value", 0)
                        try:
                            rate_val = float(rate_raw)
                        except (TypeError, ValueError):
                            rate_val = 0.0
                        wells_data.append({
                            "row": int(geom.get("row", nrow_val // 2)),
                            "col": int(geom.get("col", ncol_val // 2)),
                            "rate": rate_val,
                            "label": bc.get("name", "Well"),
                        })

                # 保存图件上下文（用于后续自然语言修改）
                plot_styler.save_context(
                    head=head, delr=delr, delc=delc,
                    wells=wells_data if wells_data else None,
                    plot_type="head",
                )

                map_html = (
                    f"<h3>Model Results</h3>"
                    f"<b>Head Range:</b> {summary.get('head_min', 0):.2f} ~ {summary.get('head_max', 0):.2f} m<br>"
                    f"<b>Water Balance Error:</b> {summary.get('water_balance_error', 0)*100:.4f}%<br>"
                    f"<b>Total Inflow:</b> {summary.get('total_flow_in', 0):.2f} m3/d<br>"
                    f"<b>Total Outflow:</b> {summary.get('total_flow_out', 0):.2f} m3/d"
                )

                if gwt:
                    try:
                        conc = rp.extract_concentration_field(gwt, layer=0)
                        plume_fig = viz_eng.plot_plume_evolution([conc], [1.0], delr, delc, threshold=0.01, n_cols=1)
                    except Exception:
                        pass

            except Exception as e:
                logger.error(f"结果提取失败: {e}")
                logger.error(traceback.format_exc())
                map_html = f"<p style='color:orange'>结果提取失败: {e}</p>"

        report_path = None
        try:
            rg = ReportGenerator()
            report_path = rg.generate(json_ir=json_ir, validation_log=val_result.get("issues", []),
                                      sim_results=summary, model_id=sid or "default")
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


# ============================================================
# 图件自然语言修改
# ============================================================

def plot_edit_handler(nl_request: str):
    """处理自然语言图件修改请求"""
    monitor.step(f"图件修改: {nl_request[:60]}...")

    if not nl_request or not nl_request.strip():
        return None, "请输入修改需求", plot_styler.generate_code()

    if not plot_styler.has_data():
        return None, "请先运行建模生成图件", "# 等待建模..."

    new_fig, explanation = plot_styler.apply_nl(nl_request.strip())

    if new_fig is None:
        return None, explanation, plot_styler.generate_code()

    monitor.ok("图件修改完成")

    # 生成绘图代码
    code = plot_styler.generate_code()

    return new_fig, explanation, code


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
# 参数列表模块 — 常量定义
# ================================================================

# (dot_path, display_name_cn, unit, priority)
PARAM_TABLE_DEF = [
    # ── P1 阻断性参数 ──
    ("project.name",             "项目名称",       "-",       "P1"),
    ("project.model_type",       "模拟类型",       "-",       "P1"),
    ("domain.x_range",           "东西范围",       "m",       "P1"),
    ("domain.y_range",           "南北范围",       "m",       "P1"),
    ("grid.nrow",                "网格行数",       "-",       "P1"),
    ("grid.ncol",                "网格列数",       "-",       "P1"),
    ("aquifer.type",             "含水层类型",     "-",       "P1"),
    ("aquifer.parameters.hk",    "渗透系数",       "m/d",     "P1"),
    ("boundary_conditions",      "边界条件",       "-",       "P1"),
    # ── P2 影响性参数 ──
    ("grid.top",                 "顶板高程",       "m",       "P2"),
    ("grid.bottom",              "底板高程",       "m",       "P2"),
    ("stress_periods",           "应力期设置",     "-",       "P2"),
    ("aquifer.parameters.sy",    "给水度",         "-",       "P2"),
    ("aquifer.parameters.ss",    "储水系数",       "1/m",     "P2"),
    # ── P3 优化性参数 ──
    ("aquifer.parameters.porosity", "孔隙度",      "-",       "P3"),
    ("aquifer.parameters.vka",   "垂向渗透系数",   "m/d",     "P3"),
    ("transport.dispersion.al",  "纵向弥散度",     "m",       "P3"),
    ("grid.delr",                "网格行宽",       "m",       "P3"),
    ("grid.delc",                "网格列宽",       "m",       "P3"),
    ("grid.nlay",                "网格层数",       "-",       "P3"),
]

# ================================================================
# 参数列表模块 — 辅助函数
# ================================================================

def _dot_get(ir: dict, dot_path: str):
    """按点路径从 JSON IR 读取值，缺失时返回 None"""
    parts = dot_path.split(".")
    current = ir
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
            if current is None:
                return None
        else:
            return None
    return current


def _format_value_for_table(val) -> str:
    """格式化为表格显示字符串：空→''，list/dict→紧凑JSON，其他→str"""
    if val is None:
        return ""
    if isinstance(val, str) and not val.strip():
        return ""
    if isinstance(val, (list, dict)):
        if len(val) == 0:
            return ""
        try:
            return json.dumps(val, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            return str(val)
    if isinstance(val, bool):
        return str(val).lower()
    return str(val)


def _get_param_status(dot_path: str, value) -> str:
    """判断参数状态：已设置 / 缺失"""
    if value is None:
        return "缺失"
    if isinstance(value, str) and not value.strip():
        return "缺失"
    if isinstance(value, (list, dict)) and len(value) == 0:
        return "缺失"
    return "已设置"


def _parse_table_value(raw: str, dot_path: str):
    """解析用户编辑的字符串为 Python 类型，返回 (parsed_value, error_msg_or_None)"""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None, None  # 清空参数

    raw = str(raw).strip()

    # 复合类型参数（列表/字典）
    COMPLEX_PARAMS = {"domain.x_range", "domain.y_range", "boundary_conditions", "stress_periods"}
    ARRAY_OR_SCALAR = {"grid.delr", "grid.delc", "grid.top", "grid.bottom"}
    NUMERIC_PARAMS = {
        "grid.nrow", "grid.ncol", "grid.nlay",
        "aquifer.parameters.hk", "aquifer.parameters.vka",
        "aquifer.parameters.sy", "aquifer.parameters.ss",
        "aquifer.parameters.porosity",
        "transport.dispersion.al",
    }

    # 尝试 JSON 解析
    if dot_path in COMPLEX_PARAMS or dot_path in ARRAY_OR_SCALAR:
        if raw.startswith("[") or raw.startswith("{"):
            try:
                return json.loads(raw), None
            except json.JSONDecodeError as e:
                return None, f"JSON 格式错误: {e}"
    if dot_path in COMPLEX_PARAMS:
        # 复合参数必须是有效 JSON
        return raw, None  # 字符串原样保留

    # 数组/标量混合参数
    if dot_path in ARRAY_OR_SCALAR:
        try:
            return int(raw), None
        except ValueError:
            try:
                return float(raw), None
            except ValueError:
                return raw, None  # 字符串值原样保留

    # 数值参数
    if dot_path in NUMERIC_PARAMS:
        try:
            return int(raw), None
        except ValueError:
            try:
                return float(raw), None
            except ValueError:
                return None, f"请输入有效数值，收到: {raw}"

    # 字符串参数（项目名称、模拟类型、含水层类型等）
    return raw, None


def build_param_table(json_ir: dict):
    """从 JSON IR 构建参数表格 DataFrame（5列：参数名|当前值|单位|优先级|状态）"""
    if not isinstance(json_ir, dict):
        json_ir = {}
    rows = []
    for dot_path, display_name, unit, priority in PARAM_TABLE_DEF:
        raw_value = _dot_get(json_ir, dot_path)
        formatted = _format_value_for_table(raw_value)
        status = _get_param_status(dot_path, raw_value)
        rows.append([display_name, formatted, unit, priority, status])
    return pd.DataFrame(rows, columns=["参数名", "当前值", "单位", "优先级", "状态"])


def merge_table_to_ir(table_data, current_ir: dict, dm) -> dict:
    """将编辑后的表格合并回 JSON IR，返回更新后的 IR"""
    if not isinstance(current_ir, dict):
        current_ir = {}
    updated_ir = copy.deepcopy(current_ir)

    # 规范化表格数据为行列表
    if isinstance(table_data, pd.DataFrame):
        rows = table_data.values.tolist()
    elif isinstance(table_data, (list, tuple)):
        rows = list(table_data)
    else:
        return updated_ir

    changes = 0
    for i, (dot_path, display_name, unit, priority) in enumerate(PARAM_TABLE_DEF):
        if i >= len(rows):
            break
        row = rows[i]
        new_raw = ""
        if len(row) > 1 and row[1] is not None:
            new_raw = str(row[1]).strip()
        old_raw = _dot_get(updated_ir, dot_path)
        old_formatted = _format_value_for_table(old_raw)

        if new_raw == old_formatted:
            continue  # 未变更，跳过

        parsed_value, error = _parse_table_value(new_raw, dot_path)
        if error is not None:
            logger.warning(f"参数 {dot_path} 解析失败: {error}")
            continue

        # 复杂类型：直接写入 IR
        if dot_path == "boundary_conditions":
            if isinstance(parsed_value, (list, str)) and parsed_value != "":
                if isinstance(parsed_value, str):
                    try:
                        parsed_value = json.loads(parsed_value)
                    except json.JSONDecodeError:
                        continue
                updated_ir["boundary_conditions"] = parsed_value
            elif parsed_value is None:
                updated_ir.pop("boundary_conditions", None)
            changes += 1
            continue

        if dot_path == "stress_periods":
            if isinstance(parsed_value, dict):
                updated_ir["stress_periods"] = parsed_value
            elif isinstance(parsed_value, str):
                try:
                    updated_ir["stress_periods"] = json.loads(parsed_value)
                except json.JSONDecodeError:
                    continue
            elif parsed_value is None:
                updated_ir.pop("stress_periods", None)
            changes += 1
            continue

        if dot_path.startswith("domain."):
            key = dot_path.split(".", 1)[1]
            if "domain" not in updated_ir:
                updated_ir["domain"] = {}
            if isinstance(parsed_value, (list, tuple)):
                updated_ir["domain"][key] = list(parsed_value)
            elif parsed_value is None:
                updated_ir.get("domain", {}).pop(key, None)
            changes += 1
            continue

        # 数组/标量混合类型
        if dot_path in ("grid.delr", "grid.delc", "grid.top", "grid.bottom"):
            parts = dot_path.split(".")
            target = updated_ir
            for p in parts[:-1]:
                if p not in target:
                    target[p] = {}
                target = target[p]
            if parsed_value is not None:
                target[parts[-1]] = parsed_value
            else:
                target.pop(parts[-1], None)
            changes += 1
            continue

        # 简单类型：使用 dm._set_ir_param 写入
        if parsed_value is not None:
            dm._set_ir_param(dot_path, parsed_value, unit, "user_edit")
        else:
            # 清空参数
            parts = dot_path.split(".")
            target = updated_ir
            for p in parts[:-1]:
                if isinstance(target, dict) and p in target:
                    target = target[p]
                else:
                    target = None
                    break
            if target is not None and isinstance(target, dict):
                target.pop(parts[-1], None)
        changes += 1

    if changes > 0:
        # 从 dm.state 获取更新后的 IR（_set_ir_param 写入 dm.state）
        if dm.state.get("json_ir"):
            updated_ir = copy.deepcopy(dm.state["json_ir"])
        # 最终规范化
        from app.llm_agent.intent_parser import IntentParser
        updated_ir = IntentParser._normalize_ir(updated_ir)
        logger.info(f"参数列表: {changes} 处变更已写入 IR")

    return updated_ir


# ================================================================
# 参数列表模块 — 参数传递处理函数
# ================================================================

def _count_set_params(table_data):
    """统计已设置参数数量，返回状态文本"""
    if isinstance(table_data, pd.DataFrame):
        rows = table_data.values.tolist()
    elif isinstance(table_data, (list, tuple)):
        rows = list(table_data)
    else:
        return "0/0 已设置"
    set_count = sum(1 for r in rows if len(r) > 4 and r[4] == "已设置")
    return f"{set_count}/{len(PARAM_TABLE_DEF)} 已设置"


def _build_highlighted_missing(missing: dict) -> list:
    """将缺失参数字典转换为 HighlightedText 格式 [(label, type), ...]"""
    highlighted = []
    for p in missing.get("priority_1", []):
        highlighted.append((f"P1 {p}", "blocking"))
    for p in missing.get("priority_2", []):
        highlighted.append((f"P2 {p}", "important"))
    for p in missing.get("priority_3", []):
        highlighted.append((f"P3 {p}", "optional"))
    return highlighted


def param_transfer_handler(table_data, json_ir: dict, sid: str):
    """
    参数传递：读取编辑后的表格 → 合并到 IR → 更新会话 → 重建表格
    返回 (new_table_df, updated_json_ir, highlighted_missing)
    """
    if not isinstance(json_ir, dict):
        json_ir = {}

    dm = get_session(sid)
    updated_ir = merge_table_to_ir(table_data, json_ir, dm)

    # 更新会话状态
    dm.state["json_ir"] = updated_ir
    dm.state["missing_params"] = dm.parser.detect_missing_params(updated_ir)
    dm._update_phase()

    # 重建表格 + 缺失参数列表
    new_table = build_param_table(updated_ir)
    highlighted = _build_highlighted_missing(dm.state["missing_params"])
    logger.info(f"参数传递完成: 缺失 P1={len(dm.state['missing_params'].get('priority_1',[]))}, P2={len(dm.state['missing_params'].get('priority_2',[]))}")
    return new_table, updated_ir, highlighted


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
                        file_types=[".shp", ".shx", ".dbf", ".prj", ".geojson", ".kml",
                                    ".xlsx", ".xls", ".csv", ".tif", ".tiff", ".asc"])
                with gr.Row():
                    clear_btn = gr.Button("清空", variant="secondary", size="sm")
                    auto_fill_btn = gr.Button("建议补全", variant="secondary", size="sm")
                    run_btn = gr.Button("开始建模", variant="primary", size="sm")

            with gr.Column(scale=3):
                # ===== NEW: 参数列表模块 =====
                gr.Markdown("### 📊 参数列表")
                param_table = gr.Dataframe(
                    value=build_param_table({}),
                    headers=["参数名", "当前值", "单位", "优先级", "状态"],
                    label="模型参数（双击单元格编辑，支持 JSON 格式输入复合类型）",
                    interactive=True,
                    wrap=True,
                    column_count=5,
                    row_count=len(PARAM_TABLE_DEF),
                )
                with gr.Row():
                    param_transfer_btn = gr.Button(
                        "📥 参数传递", variant="secondary", size="sm", scale=2)
                    param_status = gr.Textbox(
                        label="参数状态", value="0/20 已设置",
                        interactive=False, scale=3)

                # ===== EXISTING: 参数审查 =====
                gr.Markdown("### 📋 参数审查")
                param_panel = gr.JSON(label="JSON IR", value={}, height=200)
                with gr.Row():
                    phase_text = gr.Textbox(label="会话阶段", value="intake", interactive=False)
                    missing_params = gr.HighlightedText(label="缺失参数", value=[],
                        color_map={"blocking": "red", "important": "orange", "optional": "green"})

                # ===== EXISTING: 代码预览 =====
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
                plume_plot = gr.Plot(label="污染羽演化")
            with gr.TabItem("📈 过程分析"):
                process_plot = gr.Plot(label="主控过程贡献")
            with gr.TabItem("📄 报告"):
                report_download = gr.File(label="下载报告")

        # ---- 图件修改 ----
        gr.Markdown("---")
        gr.Markdown("### 🎨 图件修改（自然语言）")
        with gr.Row():
            plot_edit_msg = gr.Textbox(
                label="描述修改需求",
                placeholder="例：突出显示等水位线，将等水位线线宽设置为3，间隔设置为10m/根",
                lines=2, scale=3,
            )
            plot_edit_btn = gr.Button("修改图件", variant="primary", scale=1)
        with gr.Row():
            plot_edit_result = gr.Textbox(label="修改说明", lines=2, scale=1)
            plot_code = gr.Code(
                label="📝 绘图代码（可复制运行）",
                language="python", lines=18, scale=1,
                value="# 建模后将在此处显示绘图代码\n# 修改图件后代码会自动更新",
            )

        # === 事件绑定 ===
        def on_send(msg, hist, sid):
            return chat_handler(msg, hist, sid)
        def clear_msg(): return ""

        # 6 outputs: chatbot, param_table, param_panel, missing_params, phase_text, sid_state
        send_btn.click(on_send, [msg, chatbot, sid_state],
                       [chatbot, param_table, param_panel, missing_params, phase_text, sid_state]
                       ).then(clear_msg, None, [msg]
                       ).then(_count_set_params, [param_table], [param_status])
        msg.submit(on_send, [msg, chatbot, sid_state],
                   [chatbot, param_table, param_panel, missing_params, phase_text, sid_state]
                   ).then(clear_msg, None, [msg]
                   ).then(_count_set_params, [param_table], [param_status])
        file_upload.upload(file_parse_handler, [file_upload, chatbot], [chatbot, param_panel, result_text])

        # 清空：同时重置参数表格
        clear_btn.click(
            lambda sid: ([], build_param_table({}), {}, [], "intake", sid),
            [sid_state],
            [chatbot, param_table, param_panel, missing_params, phase_text, sid_state]
        )

        # 自动补全：增加参数表格输出
        auto_fill_btn.click(
            lambda hist, sid: _apply_suggestions_cmd(hist, get_session(sid), sid),
            [chatbot, sid_state],
            [chatbot, param_table, param_panel, missing_params, phase_text, sid_state]
        )

        # ---- 参数传递按钮 ----
        param_transfer_btn.click(
            param_transfer_handler,
            [param_table, param_panel, sid_state],
            [param_table, param_panel, missing_params]
        ).then(
            _count_set_params, [param_table], [param_status]
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
        run_btn.click(run_model_handler, [param_panel, sid_state],
                      [head_plot, result_text, plume_plot, process_plot, report_download, code_preview, debug_log])
        plot_edit_btn.click(plot_edit_handler, [plot_edit_msg],
                           [head_plot, plot_edit_result, plot_code])

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

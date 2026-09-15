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
from app.analysis.visualizer import VisualizationEngine, visualizer
from app.analysis.result_processor import ResultProcessor
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

    history.append({"role": "user", "content": "开始建模"})
    history.append({"role": "assistant", "content": "✅ 参数校验通过！正在启动自动建模...\n请查看下方 **📊 建模结果** 面板。"})
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
        map_html = "<p>模型运行成功</p>"
        plume_fig = empty_fig
        process_fig = empty_fig
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
                dm = get_session(sid)
                file_data = dm.state.get("file_uploads", {})
                if file_data:
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

                    logger.info(f"底图叠加: 显示线={shp_path}, 裁剪面={clip_shp_path}, 底图={basemap_path}")

                    if basemap_path or shp_bounds:
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
                            import traceback
                            logger.warning(traceback.format_exc())

                # ── 保存图件上下文，供后续 NL 修改使用 ──
                plot_styler.save_context(head=head, delr=delr, delc=delc,
                                         wells=json_ir.get("wells"), plot_type="head")

                map_html = (
                    f"<h3>Model Results</h3>"
                    f"<b>Head Range:</b> {summary.get('head_min', 0):.2f} ~ {summary.get('head_max', 0):.2f} m<br>"
                    f"<b>Water Balance Error:</b> {summary.get('water_balance_error', 0)*100:.4f}%<br>"
                    f"<b>Total Inflow:</b> {summary.get('total_flow_in', 0):.2f} m3/d<br>"
                    f"<b>Total Outflow:</b> {summary.get('total_flow_out', 0):.2f} m3/d"
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

                        plume_fig = viz_eng.plot_plume_evolution(
                            frames, frame_times, delr, delc,
                            threshold=0.01, n_cols=min(3, len(frames)),
                        )
                    except Exception as e:
                        logger.warning(f"污染羽演化图绘制失败: {e}")
                        logger.warning(traceback.format_exc())

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
                                    ".xlsx", ".xls", ".csv", ".tif", ".tiff", ".asc"])
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
| **GeoTIFF** | `.tif`, `.tiff` | 🏔️ DEM 高程、参数分布 | 含地理参考的栅格数据 |
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
                plume_plot = gr.Plot(label="污染羽演化")
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
        run_btn.click(run_model_handler, [param_panel, sid_state],
                      [head_plot, result_text, plume_plot, process_plot, report_download, code_preview, debug_log])

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

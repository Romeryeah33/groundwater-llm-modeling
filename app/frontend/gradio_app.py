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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    # history is list of {"role": ..., "content": ...} in Gradio 6
    if not message or not message.strip():
        return history, {}, [], "", sid

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
        logger.error(f"处理失败: {e}")
        logger.error(traceback.format_exc())
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
        reply = dm.apply_all_suggestions()
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
        return empty_fig, f"<p style='color:red'>参数数据异常，请重新输入建模需求</p>", empty_fig, empty_fig, empty_fig, None

    if not json_ir or not json_ir.get("project", {}).get("name"):
        return empty_fig, "<p>请先在左侧对话区输入建模需求，完成参数补全后再开始建模</p>", empty_fig, empty_fig, empty_fig, None

    # ================================================================
    # 建模前缺失参数检测（阻断建模前提醒用户）
    # ================================================================
    missing = intent_parser.detect_missing_params(json_ir)
    p1_missing = missing.get("priority_1", [])
    p2_missing = missing.get("priority_2", [])
    p3_missing = missing.get("priority_3", [])

    if p1_missing or p2_missing:
        warning_html = render_missing_params_guidance(p1_missing, p2_missing, p3_missing, format="html")
        return empty_fig, warning_html, empty_fig, empty_fig, empty_fig, None

    try:
        val_result = quick_validate(json_ir)
    except Exception as e:
        logger.error(f"参数校验异常: {e}")
        logger.error(traceback.format_exc())
        return empty_fig, f"<p style='color:red'>参数校验失败: {e}</p>", empty_fig, empty_fig, empty_fig, None

    if val_result["error_count"] > 0:
        issues = "\n".join(
            f"  {i['param_name']}: {i['description']}"
            for i in val_result["issues"] if i.get("severity") == "error"
        )
        return empty_fig, f"<p style='color:red'>参数校验发现 {val_result['error_count']} 个错误:</p><pre>{issues}</pre>", empty_fig, empty_fig, empty_fig, None

    workspace = settings.workspace_dir / f"run_{sid}" if sid else settings.workspace_dir / "run_default"
    workspace.mkdir(parents=True, exist_ok=True)

    try:
        builder = AutoModelBuilder(str(workspace))
        result = builder.build_and_run(json_ir, "auto_model")

        if not result.get("success"):
            error_msg = result.get('error', 'Unknown')
            logger.error(f"模型运行失败: {error_msg}")
            return empty_fig, f"<p style='color:red'>模型运行失败: {error_msg}</p><p>请检查参数是否合理，或查看后台日志获取详情。</p>", empty_fig, empty_fig, empty_fig, None

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
        return head_fig, map_html, plume_fig, process_fig, report_path

    except Exception as e:
        logger.error(f"建模异常: {e}")
        logger.error(traceback.format_exc())
        return empty_fig, f"<p style='color:red'>建模异常: {e}</p>", empty_fig, empty_fig, empty_fig, None

# ============================================================
# UI
# ============================================================

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
                gr.Markdown("### 📋 参数审查")
                param_panel = gr.JSON(label="JSON IR", value={}, height=220)
                phase_text = gr.Textbox(label="会话阶段", value="intake", interactive=False)
                missing_params = gr.HighlightedText(label="缺失参数", value=[],
                    color_map={"blocking": "red", "important": "orange", "optional": "green"})

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

        # === 事件绑定 ===
        def on_send(msg, hist, sid):
            return chat_handler(msg, hist, sid)
        def clear_msg(): return ""

        send_btn.click(on_send, [msg, chatbot, sid_state],
                       [chatbot, param_panel, missing_params, phase_text, sid_state]
                       ).then(clear_msg, None, [msg])
        msg.submit(on_send, [msg, chatbot, sid_state],
                   [chatbot, param_panel, missing_params, phase_text, sid_state]
                   ).then(clear_msg, None, [msg])
        file_upload.upload(file_parse_handler, [file_upload, chatbot], [chatbot, param_panel, result_text])
        clear_btn.click(lambda sid: ([], {}, [], "intake", sid), [sid_state],
                        [chatbot, param_panel, missing_params, phase_text, sid_state])
        auto_fill_btn.click(lambda hist, sid: _apply_suggestions_cmd(hist, get_session(sid), sid),
                            [chatbot, sid_state], [chatbot, param_panel, missing_params, phase_text, sid_state])
        run_btn.click(run_model_handler, [param_panel, sid_state],
                      [head_plot, result_text, plume_plot, process_plot, report_download])

        gr.Markdown("""
        ---
        ### 💡 快速开始
        > 我要开展一个单层非承压含水层稳态模拟，范围东西1000×南北800米，渗透系数15m/d，西侧定水头100m，东侧定水头95m，南北侧为零流量边界，项目名称为"try"。
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

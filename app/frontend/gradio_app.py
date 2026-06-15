"""
Gradio Web 前端 — 全模块集成版

集成流程：
  自然语言/文件输入 → 意图解析 → 参数提取 → 缺失检测 →
  物理约束校验 → FloPy自动建模 → 结果可视化 → 报告下载
"""

import sys
import io
import os
from pathlib import Path

# Force UTF-8
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
import logging
import json
import tempfile
import traceback
import numpy as np
import matplotlib.pyplot as plt

from app.config import settings
from app.llm_agent.intent_parser import IntentParser, intent_parser
from app.llm_agent.dialogue_manager import DialogueManager, dialogue_manager
from app.llm_agent.file_handler import FileUploadHandler, file_handler
from app.physics_engine.validator import quick_validate
from app.physics_engine.constraints import ConstraintSeverity
from app.modeling.grid_generator import AutoGridGenerator, grid_generator
from app.modeling.model_mapper import FloPyModelMapper, model_mapper
from app.modeling.model_builder import AutoModelBuilder
from app.analysis.visualizer import VisualizationEngine, visualizer
from app.analysis.result_processor import ResultProcessor, result_processor
from app.analysis.report_generator import ReportGenerator, report_generator

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ============================================================
# 全局会话状态管理
# ============================================================

sessions = {}

def get_session(session_id: str) -> DialogueManager:
    if session_id not in sessions:
        sessions[session_id] = DialogueManager()
    return sessions[session_id]

# ============================================================
# 1. 对话处理（核心）
# ============================================================

def chat_handler(message: str, history: list, session_id: str = "default"):
    """处理用户自然语言输入"""
    if not message or not message.strip():
        return history, {}, [], "intake", session_id

    dm = get_session(session_id)

    # 特殊命令
    msg_lower = message.strip().lower()
    if msg_lower in ("开始建模", "运行", "run", "start"):
        return run_model_from_chat(history, dm, session_id)
    elif msg_lower in ("使用建议值", "自动补全", "auto fill"):
        return apply_suggestions(history, dm, session_id)
    elif msg_lower in ("清空", "重置", "reset"):
        dm.reset()
        history.append(("系统: 已重置", ""))
        return history, {}, [], "intake", session_id

    # 正常对话流程
    try:
        result = dm.process(message)
    except Exception as e:
        reply = f"⚠️ 处理出错: {e}\n请重试或简化描述。"
        history.append((message, reply))
        return history, dm.state.get("json_ir", {}), dm.state.get("missing_params", {}), dm.state.get("phase", "intake"), session_id

    reply = result["reply"]
    json_ir = result["json_ir"]
    missing = {
        "priority_1": result["missing_params"].get("priority_1", []),
        "priority_2": result["missing_params"].get("priority_2", []),
        "priority_3": result["missing_params"].get("priority_3", []),
    }

    # 生成缺失参数高亮文本
    highlighted = []
    for p in missing.get("priority_1", []):
        highlighted.append({"text": f"⚠️ {p}", "label": "阻断"})
    for p in missing.get("priority_2", []):
        highlighted.append({"text": f"⚡ {p}", "label": "影响"})
    for p in missing.get("priority_3", []):
        highlighted.append({"text": f"💡 {p}", "label": "优化"})

    history.append((message, reply))
    return history, json_ir, highlighted, result["phase"], session_id


def apply_suggestions(history, dm, session_id):
    """一键应用建议值"""
    reply = dm.apply_all_suggestions()
    missing = dm.state["missing_params"]
    highlighted = []
    for p in missing.get("priority_1", []):
        highlighted.append({"text": f"⚠️ {p}", "label": "阻断"})
    for p in missing.get("priority_2", []):
        highlighted.append({"text": f"⚡ {p}", "label": "影响"})
    history.append(("🤖 使用建议值", reply))
    return history, dm.state["json_ir"], highlighted, dm.state["phase"], session_id


def run_model_from_chat(history, dm, session_id):
    """从对话触发建模"""
    json_ir = dm.state["json_ir"]
    if not json_ir or not json_ir.get("project", {}).get("name"):
        reply = "⚠️ 参数信息不足，无法开始建模。请先描述您的建模需求。"
        history.append(("开始建模", reply))
        return history, json_ir, [], dm.state["phase"], session_id

    # 快速校验
    val_result = quick_validate(json_ir)
    if val_result["error_count"] > 0:
        issues_text = "\n".join(
            f"  ❌ {i['param_name']}: {i['description']}" for i in val_result["issues"] if i["severity"] == "error"
        )
        reply = f"⚠️ 参数校验发现 {val_result['error_count']} 个错误:\n{issues_text}\n请修正后重试。"
        history.append(("开始建模", reply))
        return history, json_ir, [], dm.state["phase"], session_id

    reply = "✅ 参数校验通过！正在启动自动建模...\n(请查看下方结果面板)"
    history.append(("🚀 开始建模", reply))
    dm.state["phase"] = "execution"
    return history, json_ir, [], dm.state["phase"], session_id

# ============================================================
# 2. 文件上传处理
# ============================================================

def file_parse_handler(files, history):
    """处理上传文件"""
    if not files:
        return history, {}, ""

    file_list = files if isinstance(files, list) else [files]
    result = file_handler.parse_files(file_list)
    reply = result.get("summary", "📁 文件已接收")

    history.append(("📎 上传文件", reply))
    return history, {}, reply

# ============================================================
# 3. 自动建模执行
# ============================================================

def run_model_handler(json_ir: dict, session_id: str = "default"):
    """
    执行完整的自动建模流程：
    JSON IR → 校验 → FloPy构建 → 运行 → 可视化 → 报告
    """
    # 空图占位
    empty_fig = plt.figure(figsize=(4, 3))
    plt.text(0.5, 0.5, "Waiting for input...", ha="center", va="center", transform=plt.gca().transAxes)
    plt.tight_layout()

    if not json_ir or not json_ir.get("project", {}).get("name"):
        return empty_fig, "<p>请先输入建模需求</p>", empty_fig, empty_fig, empty_fig, None

    # Step 1: 快速校验
    val_result = quick_validate(json_ir)
    if val_result["error_count"] > 0:
        return empty_fig, f"<p style='color:red'>参数校验失败: {val_result['error_count']}个错误</p>", empty_fig, empty_fig, empty_fig, None

    # Step 2: 创建临时工作目录
    workspace = settings.workspace_dir / f"run_{session_id}" if session_id else settings.workspace_dir / "run_default"
    workspace.mkdir(parents=True, exist_ok=True)

    try:
        # Step 3: 构建并运行模型
        builder = AutoModelBuilder(str(workspace))
        result = builder.build_and_run(json_ir, "auto_model")

        if not result.get("success"):
            err = result.get("error", "Unknown")
            return empty_fig, f"<p style='color:red'>模型运行失败: {err}</p>", empty_fig, empty_fig, empty_fig, None

        # Step 4: 提取结果并生成可视化
        gwf = result.get("gwf")
        gwt = result.get("gwt")
        rp = ResultProcessor()
        viz_eng = VisualizationEngine()

        # 水头图
        head_fig = empty_fig
        map_html = "<p>模型运行成功</p>"
        plume_fig = empty_fig
        process_fig = empty_fig

        if gwf:
            try:
                head = rp.extract_head_field(gwf, layer=0)
                summary = rp.generate_summary(gwf, gwt, json_ir)

                # 网格参数
                grid = json_ir.get("grid", {})
                delr = np.full(grid.get("ncol", 50), grid.get("delr", 20) if isinstance(grid.get("delr"), (int, float)) else 20)
                delc = np.full(grid.get("nrow", 40), grid.get("delc", 20) if isinstance(grid.get("delc"), (int, float)) else 20)

                head_fig = viz_eng.plot_head_contour(
                    head, delr, delc,
                    f"Head Distribution (Water Balance Error: {summary.get('water_balance_error', 0)*100:.3f}%)",
                )

                # 地图摘要
                hr = summary.get("head_min", 0)
                map_lines = [
                    f"<h3>Model Results</h3>",
                    f"<b>Head Range:</b> {summary.get('head_min', 'N/A'):.2f} ~ {summary.get('head_max', 'N/A'):.2f} m",
                    f"<b>Water Balance Error:</b> {summary.get('water_balance_error', 0)*100:.4f}%",
                    f"<b>Total Inflow:</b> {summary.get('total_flow_in', 0):.2f} m³/d",
                    f"<b>Total Outflow:</b> {summary.get('total_flow_out', 0):.2f} m³/d",
                ]
                if summary.get("conc_max") is not None:
                    map_lines.append(f"<b>Max Concentration:</b> {summary['conc_max']:.4f} mg/L")
                map_html = "<br>".join(map_lines)

                # 如果有溶质运移结果
                if gwt:
                    try:
                        conc = rp.extract_concentration_field(gwt, layer=0)
                        plume_fig = viz_eng.plot_plume_evolution(
                            [conc], [1.0], delr, delc, threshold=0.01, n_cols=1,
                        )
                    except:
                        pass

            except Exception as e:
                map_html = f"<p style='color:orange'>结果提取部分失败: {e}</p>"

        # Step 5: 生成报告
        report_path = None
        try:
            rg = ReportGenerator()
            viz_paths = {}
            if head_fig != empty_fig:
                import tempfile
                tmp = Path(tempfile.gettempdir()) / "head_dash.png"
                head_fig.savefig(str(tmp), bbox_inches="tight", dpi=150)
                viz_paths["Head Distribution"] = str(tmp)

            report_path = rg.generate(
                json_ir=json_ir,
                validation_log=val_result.get("issues", []),
                sim_results=summary if gwf else {},
                visualizations=viz_paths,
                model_id=session_id or "default",
            )
        except Exception as e:
            logger.warning(f"Report generation failed: {e}")

        plt.close("all")
        return head_fig, map_html, plume_fig, process_fig, report_path

    except Exception as e:
        traceback.print_exc()
        return empty_fig, f"<p style='color:red'>建模异常: {e}</p>", empty_fig, empty_fig, None

# ============================================================
# 4. Gradio UI
# ============================================================

def create_ui():
    with gr.Blocks(title="Groundwater LLM Modeling") as demo:
        # Hidden session ID
        session_state = gr.State("default")

        gr.Markdown("""
        # 🌊 Physical-LLM Groundwater Intelligent Modeling System
        ### 物理约束嵌入大语言模型的地下水智能建模原型系统 (MVP)

        通过自然语言描述即可完成地下水数值建模 — 无需编写代码、操作 GUI。
        """)

        with gr.Row():
            # === 左栏：对话 ===
            with gr.Column(scale=4):
                gr.Markdown("### 💬 建模对话")
                chatbot = gr.Chatbot(height=420, label="对话记录")
                with gr.Row():
                    msg = gr.Textbox(
                        label="输入建模需求...",
                        placeholder="例如：模拟承压含水层1000×800m，渗透系数15m/d，西侧100m东侧95m定水头...",
                        lines=2, scale=9,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)

                with gr.Row():
                    file_upload = gr.File(
                        label="📎 上传文件（SHP/GeoJSON/Excel/GeoTIFF/CSV）",
                        file_count="multiple",
                        file_types=[".shp", ".shx", ".dbf", ".prj", ".geojson", ".kml",
                                    ".xlsx", ".xls", ".csv", ".tif", ".tiff", ".asc"],
                    )
                with gr.Row():
                    clear_btn = gr.Button("🗑️ 清空对话", variant="secondary", size="sm")
                    auto_fill_btn = gr.Button("🤖 自动建议补全", variant="secondary", size="sm")
                    run_btn = gr.Button("🚀 开始建模", variant="primary", size="sm")

            # === 右栏：参数面板 ===
            with gr.Column(scale=3):
                gr.Markdown("### 📋 参数审查")
                param_panel = gr.JSON(label="当前参数 (JSON IR)", value={}, height=220)
                status_text = gr.Textbox(label="对话阶段", value="intake", interactive=False)
                missing_params = gr.HighlightedText(
                    label="缺失参数",
                    value=[],
                    show_legend=True,
                    color_map={"阻断": "red", "影响": "orange", "优化": "green"},
                )

        # === 结果展示区 ===
        gr.Markdown("---")
        gr.Markdown("### 📊 建模结果")

        with gr.Tabs():
            with gr.TabItem("📋 结果摘要"):
                result_text = gr.HTML(label="模拟结果摘要", value="<p>等待建模...</p>")
            with gr.TabItem("💧 水头分布"):
                head_plot = gr.Plot(label="水头等值线")
            with gr.TabItem("☣️ 污染羽"):
                plume_plot = gr.Plot(label="污染羽演化")
            with gr.TabItem("📈 过程分析"):
                process_plot = gr.Plot(label="主控过程贡献率")
            with gr.TabItem("📄 报告下载"):
                gr.Markdown("建模完成后可下载标准化Word报告")
                report_download = gr.File(label="下载建模报告 (.docx)")

        # === 事件绑定 ===
        send_btn.click(
            chat_handler,
            [msg, chatbot, session_state],
            [chatbot, param_panel, missing_params, status_text, session_state],
        ).then(lambda: "", None, [msg])

        msg.submit(
            chat_handler,
            [msg, chatbot, session_state],
            [chatbot, param_panel, missing_params, status_text, session_state],
        ).then(lambda: "", None, [msg])

        file_upload.upload(
            file_parse_handler, [file_upload, chatbot], [chatbot, param_panel, result_text]
        )

        clear_btn.click(
            lambda sid: (sessions.pop(sid, None), [], {}, [], "intake"),
            [session_state],
            [chatbot, param_panel, missing_params, status_text],
        ).then(lambda: None, None, [head_plot, plume_plot, process_plot, report_download])

        auto_fill_btn.click(
            lambda h, sid: apply_suggestions(h, get_session(sid), sid),
            [chatbot, session_state],
            [chatbot, param_panel, missing_params, status_text, session_state],
        )

        run_btn.click(
            run_model_handler, [param_panel, session_state],
            [head_plot, result_text, plume_plot, process_plot, report_download],
        )

        # 示例提示
        gr.Markdown("""
        ---
        ### 💡 快速开始示例
        复制以下文本粘贴到对话输入框：
        > 我要模拟一个承压含水层，范围1000×800米，渗透系数15m/d，西侧定水头100m，东侧定水头95m，中心一口抽水井流量200m³/d。

        或上传 SHP 边界文件 + 调查点位 Excel 进行复杂场地建模。
        """)

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name=settings.fastapi_host,
        server_port=settings.gradio_port,
        share=settings.gradio_share,
        show_error=True,
    )

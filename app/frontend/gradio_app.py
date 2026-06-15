"""
Gradio Web 前端界面

启动方式：
    python -m app.frontend.gradio_app
或：
    gradio app/frontend/gradio_app.py
"""

import gradio as gr
import logging

logger = logging.getLogger(__name__)

# ============================================================
# 事件处理函数（占位 — 后续接入完整后端逻辑）
# ============================================================


def chat_handler(message: str, history: list):
    """
    对话处理函数

    TODO: 接入 IntentParser + DialogueManager + LLMClient
    """
    if not message.strip():
        return history, {}, []

    # 临时：回显用户消息
    reply = (
        f"✅ 已收到您的建模需求。\n\n"
        f"📝 当前解析内容：{message[:100]}{'...' if len(message) > 100 else ''}\n\n"
        f"⚠️ 系统正在建设中，以下功能即将上线：\n"
        f"- 自动识别建模参数（渗透系数、边界条件等）\n"
        f"- 缺失参数检测与主动问询\n"
        f"- 物理约束校验\n"
        f"- FloPy 自动建模\n\n"
        f"💡 提示：您也可以上传文件（SHP/Excel/GeoJSON）来提供空间数据。"
    )

    history.append((message, reply))
    return history, {}, []


def file_parse_handler(files):
    """
    文件上传处理函数

    TODO: 接入 FileUploadHandler (SHP/Excel/GeoJSON解析)
    """
    if not files:
        return [], {}, ""

    file_list = files if isinstance(files, list) else [files]
    file_names = [f.name if hasattr(f, 'name') else str(f) for f in file_list]

    reply = f"📁 已收到 {len(file_names)} 个文件：\n"
    for fn in file_names:
        reply += f"  • {fn}\n"
    reply += "\n🔍 文件解析功能正在开发中..."

    return [("📎 文件上传", reply)], {}, ""


def auto_fill_handler(json_ir: dict):
    """
    自动建议补全

    TODO: 接入 DialogueManager.generate_suggestions()
    """
    return json_ir, [("系统", "🤖 自动补全功能正在开发中...")]


def run_model_handler(json_ir: dict):
    """
    开始建模

    TODO: 接入 AutoModelBuilder + SolverMonitor + VisualizationEngine
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # 生成临时占位图
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.linspace(0, 1000, 50)
    y = np.linspace(0, 800, 40)
    X, Y = np.meshgrid(x, y)
    Z = 100 - 0.01 * X - 0.005 * Y + 2 * np.random.randn(40, 50)
    ax.contourf(X, Y, Z, levels=15, cmap='Blues')
    ax.set_title("水头分布 (占位示例)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    return fig, "模型运行功能正在开发中...", fig, fig, None


# ============================================================
# Gradio UI 构建
# ============================================================

def create_ui():
    """创建 Gradio Web 界面"""

    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="地下水智能模拟系统",
        css="""
        .warning { color: #e67e22; }
        .success { color: #27ae60; }
        """
    ) as demo:
        # --- 标题 ---
        gr.Markdown("""
        # 🌊 物理约束嵌入LLM的地下水智能模拟原型系统
        ### Physical-LLM Groundwater Intelligent Modeling System (MVP)

        通过自然语言描述即可完成地下水数值建模 — 无需编写代码、无需操作传统GUI。
        """)

        with gr.Row():
            # --- 左侧：对话区 ---
            with gr.Column(scale=1):
                gr.Markdown("### 💬 建模对话")
                chatbot = gr.Chatbot(height=450, label="对话记录")
                msg = gr.Textbox(
                    label="输入建模需求...",
                    placeholder=(
                        "例如：我要模拟一个1000m×800m的承压含水层，"
                        "渗透系数约15 m/d，西侧定水头边界100m，"
                        "东侧定水头边界95m，中心有一口抽水井..."
                    ),
                    lines=3,
                )
                with gr.Row():
                    file_upload = gr.File(
                        label="📎 上传文件（SHP/GeoJSON/Excel/GeoTIFF/CSV）",
                        file_count="multiple",
                        file_types=[
                            ".shp", ".shx", ".dbf", ".prj",
                            ".geojson", ".kml",
                            ".xlsx", ".xls", ".csv",
                            ".tif", ".tiff", ".asc",
                        ],
                    )
                with gr.Row():
                    clear_btn = gr.Button("🗑️ 清空对话", variant="secondary", size="sm")
                    # 发送按钮 — 与文本输入绑定
                    msg.submit(
                        chat_handler,
                        [msg, chatbot],
                        [chatbot, gr.State(), gr.State()],
                    )

            # --- 右侧：参数面板 ---
            with gr.Column(scale=1):
                gr.Markdown("### 📋 参数审查")
                param_panel = gr.JSON(
                    label="当前参数 (JSON IR)",
                    value={},
                    height=200,
                )
                missing_params = gr.HighlightedText(
                    label="缺失参数检测",
                    value=[],
                    show_legend=True,
                )
                with gr.Row():
                    auto_fill_btn = gr.Button(
                        "🤖 自动建议补全",
                        variant="secondary",
                    )
                    submit_btn = gr.Button(
                        "🚀 开始建模",
                        variant="primary",
                    )

        # --- 底部：结果展示区 ---
        gr.Markdown("---")
        gr.Markdown("### 📊 建模结果")

        with gr.Tabs():
            with gr.TabItem("🗺️ 场地地图"):
                map_viewer = gr.HTML(label="交互地图")
            with gr.TabItem("🔲 模型网格"):
                grid_plot = gr.Plot(label="网格预览")
            with gr.TabItem("💧 水位分布"):
                head_plot = gr.Plot(label="水头等值线")
            with gr.TabItem("☣️ 污染羽"):
                plume_plot = gr.Plot(label="污染羽时空演化")
            with gr.TabItem("📈 过程分析"):
                process_plot = gr.Plot(label="主控过程贡献率")
            with gr.TabItem("📄 报告下载"):
                report_download = gr.File(label="下载建模报告 (.docx)")

        # --- 事件绑定 ---
        msg.submit(
            chat_handler,
            [msg, chatbot],
            [chatbot, param_panel, missing_params],
        ).then(lambda: "", None, [msg])  # 发送后清空输入框

        file_upload.upload(
            file_parse_handler,
            [file_upload],
            [chatbot, param_panel, map_viewer],
        )

        clear_btn.click(
            lambda: ([], {}, [], "", None, None, None, None, None),
            None,
            [chatbot, param_panel, missing_params, msg, grid_plot, head_plot, plume_plot, process_plot, report_download],
        )

        auto_fill_btn.click(
            auto_fill_handler,
            [param_panel],
            [param_panel, chatbot],
        )

        submit_btn.click(
            run_model_handler,
            [param_panel],
            [grid_plot, map_viewer, head_plot, plume_plot, process_plot, report_download],
        )

    return demo


# ============================================================
# 启动入口
# ============================================================

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )

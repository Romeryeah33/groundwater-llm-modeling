"""
标准化 Word 报告自动生成器

从建模过程自动汇编生成完整的 .docx 格式研究报告，涵盖：
1. 封面
2. 建模方案摘要
3. 参数清单
4. 物理约束校验记录
5. 模拟结果
6. 机理分析
7. 结论与建议
8. 附录（可复现代码 + JSON IR）
"""

import logging
import datetime
from pathlib import Path
from typing import Optional, List, Dict

from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
import json
import io

logger = logging.getLogger(__name__)


class ReportGenerator:
    """自动化标准化 Word 报告生成器"""

    def __init__(self, output_dir: Optional[Path] = None):
        from app.config import settings
        self.output_dir = output_dir or settings.output_dir / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        json_ir: dict,
        validation_log: Optional[List[dict]] = None,
        sim_results: Optional[dict] = None,
        visualizations: Optional[Dict[str, str]] = None,
        process_analysis: Optional[dict] = None,
        model_id: str = None,
    ) -> str:
        """
        生成完整 Word 报告

        Args:
            json_ir: 最终 JSON IR
            validation_log: 物理约束校验记录
            sim_results: 模拟结果摘要
            visualizations: 图表文件路径 {name: path}
            process_analysis: 主控过程分析结果
            model_id: 模型唯一ID

        Returns:
            生成的 .docx 文件路径
        """
        if model_id is None:
            model_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        doc = Document()

        # 设置默认字体
        style = doc.styles["Normal"]
        font = style.font
        font.name = "Times New Roman"
        font.size = Pt(11)
        style.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")

        # ==================== 1. 封面 ====================
        self._add_cover(doc, json_ir, model_id)

        # ==================== 2. 建模方案摘要 ====================
        doc.add_heading("1. 建模方案摘要", level=1)
        self._add_model_summary(doc, json_ir)

        # ==================== 3. 参数清单 ====================
        doc.add_heading("2. 参数清单", level=1)
        self._add_parameter_table(doc, json_ir)

        # ==================== 4. 物理约束校验记录 ====================
        doc.add_heading("3. 物理约束校验记录", level=1)
        self._add_validation_log(doc, validation_log or [])

        # ==================== 5. 模拟结果 ====================
        doc.add_heading("4. 模拟结果", level=1)
        self._add_sim_results(doc, sim_results or {})

        # 插入图件
        if visualizations:
            doc.add_heading("4.1 结果图件", level=2)
            self._add_figures(doc, visualizations)

        # ==================== 6. 机理分析 ====================
        if process_analysis:
            doc.add_heading("5. 机理分析", level=1)
            self._add_process_analysis(doc, process_analysis)

        # ==================== 7. 结论与建议 ====================
        doc.add_heading("6. 结论与建议", level=1)
        self._add_conclusions(doc, json_ir, sim_results)

        # ==================== 8. 附录 ====================
        doc.add_heading("7. 附录", level=1)
        doc.add_heading("7.1 JSON IR 参数表", level=2)
        self._add_json_ir_appendix(doc, json_ir)

        # 保存
        filepath = str(self.output_dir / f"modeling_report_{model_id}.docx")
        doc.save(filepath)
        logger.info(f"报告已生成: {filepath}")
        return filepath

    # ================================================================
    # 各章节实现
    # ================================================================

    def _add_cover(self, doc, json_ir: dict, model_id: str):
        """封面"""
        # 空行
        for _ in range(6):
            doc.add_paragraph("")

        title = doc.add_paragraph()
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = title.add_run("地下水数值模拟自动建模报告")
        run.font.size = Pt(22)
        run.bold = True
        run.font.color.rgb = RGBColor(0, 51, 102)

        doc.add_paragraph("")

        info = doc.add_paragraph()
        info.alignment = WD_ALIGN_PARAGRAPH.CENTER
        proj_name = json_ir.get("project", {}).get("name", "未命名项目")
        run = info.add_run(
            f"项目名称: {proj_name}\n"
            f"报告编号: {model_id}\n"
            f"生成日期: {datetime.date.today()}\n"
            f"建模工具: Physical-LLM Groundwater Modeling System v1.0\n"
        )
        run.font.size = Pt(12)

        doc.add_page_break()

    def _add_model_summary(self, doc, json_ir: dict):
        """建模方案摘要"""
        proj = json_ir.get("project", {})
        doc.add_paragraph(f"项目名称: {proj.get('name', 'N/A')}")
        doc.add_paragraph(f"模拟类型: {proj.get('model_type', 'N/A')}")

        domain = json_ir.get("domain", {})
        xr = domain.get("x_range", [0, 0])
        yr = domain.get("y_range", [0, 0])
        doc.add_paragraph(f"模型范围: {xr[0]:.0f}~{xr[1]:.0f}m × {yr[0]:.0f}~{yr[1]:.0f}m")

        grid = json_ir.get("grid", {})
        doc.add_paragraph(
            f"网格: {grid.get('nrow','?')}行 × {grid.get('ncol','?')}列 × {grid.get('nlay','?')}层 "
            f"(共 {grid.get('nrow',0)*grid.get('ncol',0)*grid.get('nlay',0):,} 单元)"
        )

        aqua = json_ir.get("aquifer", {})
        atype_labels = {"confined": "承压", "unconfined": "潜水", "convertible": "可转换"}
        doc.add_paragraph(f"含水层类型: {atype_labels.get(aqua.get('type', ''), 'N/A')}")

        for pname, pinfo in aqua.get("parameters", {}).items():
            if isinstance(pinfo, dict) and pinfo.get("value") is not None:
                doc.add_paragraph(f"  {pname}: {pinfo['value']} {pinfo.get('unit', '')}")

        bcs = json_ir.get("boundary_conditions", [])
        doc.add_paragraph(f"边界条件: {len(bcs)} 个")
        for bc in bcs:
            doc.add_paragraph(f"  - {bc.get('type','?')}: {bc.get('name','?')}")

    def _add_parameter_table(self, doc, json_ir: dict):
        """参数清单表格"""
        table = doc.add_table(rows=1, cols=5)
        table.style = "Light Grid Accent 1"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER

        # 表头
        hdr = table.rows[0].cells
        headers = ["参数", "值", "单位", "来源", "物理范围"]
        for i, h in enumerate(headers):
            hdr[i].text = h
            hdr[i].paragraphs[0].runs[0].bold = True

        # 收集所有参数
        all_params = []
        for section, label in [
            ("aquifer", "含水层"),
            ("transport.dispersion", "溶质-弥散"),
            ("transport.reaction", "溶质-反应"),
            ("solver", "求解器"),
        ]:
            params = {}
            if section == "aquifer":
                params = json_ir.get("aquifer", {}).get("parameters", {})
            elif section == "transport.dispersion":
                params = json_ir.get("transport", {}).get("dispersion", {})
            elif section == "transport.reaction":
                params = json_ir.get("transport", {}).get("reaction", {})
            elif section == "solver":
                params = json_ir.get("solver", {}).get("parameters", {})

            for pname, pinfo in params.items():
                if isinstance(pinfo, dict) and pinfo.get("value") is not None:
                    all_params.append({
                        "name": f"[{label}] {pname}",
                        "value": str(pinfo["value"]),
                        "unit": pinfo.get("unit", ""),
                        "source": pinfo.get("source", "N/A"),
                        "range": f"{pinfo.get('valid_range', ['N/A'])[0]}~{pinfo.get('valid_range', ['N/A',''])[1]}" if pinfo.get("valid_range") else "N/A",
                    })

        for param in all_params:
            row = table.add_row().cells
            row[0].text = param["name"]
            row[1].text = param["value"]
            row[2].text = param["unit"]
            row[3].text = param["source"]
            row[4].text = param["range"]

    def _add_validation_log(self, doc, validation_log: list):
        """物理约束校验记录"""
        if not validation_log:
            doc.add_paragraph("(无校验记录)")
            return

        n_errors = sum(1 for i in validation_log if i.get("severity") == "error")
        n_warnings = sum(1 for i in validation_log if i.get("severity") == "warning")

        doc.add_paragraph(f"校验结果: {n_errors} 个错误, {n_warnings} 个警告")

        for issue in validation_log:
            severity = issue.get("severity", "info")
            symbol = {"error": "❌", "warning": "⚠️", "info": "ℹ️", "pass": "✅"}.get(severity, "")
            doc.add_paragraph(
                f"{symbol} [{severity.upper()}] {issue.get('param_name', '')}: "
                f"{issue.get('description', '')}"
            )
            if issue.get("suggestion"):
                doc.add_paragraph(f"   修正建议: {issue['suggestion']}")

    def _add_sim_results(self, doc, sim_results: dict):
        """模拟结果摘要"""
        if not sim_results:
            doc.add_paragraph("(模拟尚未运行)")
            return

        if sim_results.get("error"):
            doc.add_paragraph(f"模拟失败: {sim_results['error']}")
            return

        doc.add_paragraph(f"水头范围: {sim_results.get('head_min', 'N/A')} ~ {sim_results.get('head_max', 'N/A')} m")

        bal_err = sim_results.get("water_balance_error")
        if bal_err is not None:
            doc.add_paragraph(f"水量平衡误差: {bal_err*100:.4f}%")
            doc.add_paragraph(f"总流入: {sim_results.get('total_flow_in', 'N/A'):.2f} m³/d")
            doc.add_paragraph(f"总流出: {sim_results.get('total_flow_out', 'N/A'):.2f} m³/d")

        if sim_results.get("conc_max") is not None:
            doc.add_paragraph(f"最大浓度: {sim_results['conc_max']:.4f} mg/L")

    def _add_figures(self, doc, visualizations: dict):
        """插入图件"""
        for name, path in visualizations.items():
            if Path(path).exists():
                doc.add_paragraph(f"{name}:")
                doc.add_picture(path, width=Inches(5.5))
                doc.add_paragraph("")

    def _add_process_analysis(self, doc, analysis: dict):
        """机理分析"""
        doc.add_paragraph(f"主控过程: {analysis.get('dominant_process', 'N/A')}")
        doc.add_paragraph(analysis.get("explanation", ""))

        contributions = analysis.get("contributions", {})
        if contributions:
            doc.add_paragraph("各过程贡献率:")
            for process, pct in contributions.items():
                doc.add_paragraph(f"  {process}: {pct}%")

        doc.add_paragraph(f"Peclet数: {analysis.get('peclet_number', 'N/A')}")
        doc.add_paragraph(f"迟滞因子: {analysis.get('retardation_factor', 'N/A')}")

    def _add_conclusions(self, doc, json_ir: dict, sim_results: dict):
        """结论与建议"""
        doc.add_paragraph(
            "本报告由物理约束嵌入大语言模型的地下水智能建模系统自动生成。"
            "建模过程无需人工编写代码或操作传统 GUI。"
        )
        doc.add_paragraph(
            "注意：自动生成的结论仅供参考，实际工程决策需结合场地调查数据"
            "和专业水文地质判断。建议对关键参数进行敏感性分析。"
        )

    def _add_json_ir_appendix(self, doc, json_ir: dict):
        """附录：JSON IR"""
        ir_text = json.dumps(json_ir, ensure_ascii=False, indent=2)
        # 使用等宽字体
        p = doc.add_paragraph()
        run = p.add_run(ir_text)
        run.font.name = "Consolas"
        run.font.size = Pt(8)


# 全局实例
report_generator = ReportGenerator()

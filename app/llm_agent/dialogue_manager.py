"""
多轮对话管理器

管理用户-系统之间的多轮对话，追踪建模上下文状态。
负责：对话状态管理、引导性问题生成、建议值生成。
"""

import json
import logging
from typing import Optional, Dict, List, Tuple

from app.llm_agent.llm_client import LLMClient, llm_client
from app.llm_agent.intent_parser import IntentParser, intent_parser
from app.llm_agent.prompt_templates import (
    QUESTION_GENERATION_PROMPT,
    SUGGESTION_GENERATION_PROMPT,
)
from app.knowledge.semantic_db import SemanticDB, semantic_db

logger = logging.getLogger(__name__)

# 对话状态
DIALOGUE_PHASES = ["intake", "clarification", "confirmation", "execution", "review"]


class DialogueManager:
    """多轮对话管理器"""

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        parser: Optional[IntentParser] = None,
        sem_db: Optional[SemanticDB] = None,
    ):
        self.llm = llm or llm_client
        self.parser = parser or intent_parser
        self.sem_db = sem_db or semantic_db

        # 对话状态
        self.reset()

    def reset(self):
        """重置对话状态"""
        self.state = {
            "phase": "intake",               # intake | clarification | confirmation | execution | review
            "json_ir": {},                    # 当前累积的 JSON IR
            "missing_params": {"priority_1": [], "priority_2": [], "priority_3": []},
            "suggested_values": {},           # {param_name: {value, unit, confidence, reason}}
            "answered_params": set(),         # 用户已确认的参数集
            "param_history": [],              # 参数修改历史
            "history": [],                    # 对话历史 [(user, assistant), ...]
            "file_uploads": [],               # 已上传文件列表
            "model_built": False,             # 是否已构建模型
            "model_results": None,            # 模拟结果
        }

    # ================================================================
    # 主入口：处理用户输入
    # ================================================================

    def process(self, user_message: str) -> dict:
        """
        处理用户输入，返回系统响应

        Returns:
            {
                "reply": str,                # 系统回复文本
                "json_ir": dict,             # 更新后的 JSON IR
                "missing_params": dict,      # 缺失参数
                "phase": str,                # 当前阶段
                "suggestions": dict,         # 可用的建议值
                "ready_to_run": bool,        # 是否可以开始建模
            }
        """
        # 记录用户消息
        self.state["history"].append({"role": "user", "content": user_message})

        # Step 1: 意图解析
        parse_result = self.parser.parse(
            user_message,
            current_ir=self.state["json_ir"],
            session_context={"phase": self.state["phase"]},
        )

        intent = parse_result["intent"]
        self.state["json_ir"] = parse_result["json_ir"]
        self.state["missing_params"] = parse_result["missing_params"]

        # Step 2: 根据意图和阶段生成响应
        if intent == "confirm":
            reply = self._handle_confirmation()
        elif intent == "reject":
            reply = self._handle_rejection()
        elif intent == "query":
            reply = self._handle_query(user_message)
        else:
            # new_model / modify_param / upload_file / unclear
            reply = self._handle_modeling_flow(user_message, parse_result)

        # 更新阶段
        self._update_phase()

        # 记录助手消息
        self.state["history"].append({"role": "assistant", "content": reply})

        return {
            "reply": reply,
            "json_ir": self.state["json_ir"],
            "missing_params": self.state["missing_params"],
            "phase": self.state["phase"],
            "suggestions": self.state["suggested_values"],
            "ready_to_run": self._is_ready_to_run(),
        }

    # ================================================================
    # 阶段更新
    # ================================================================

    def _update_phase(self):
        """根据缺失参数情况更新对话阶段"""
        p1 = len(self.state["missing_params"].get("priority_1", []))
        p2 = len(self.state["missing_params"].get("priority_2", []))

        if p1 == 0 and p2 == 0:
            self.state["phase"] = "confirmation"
        elif p1 > 0:
            self.state["phase"] = "intake" if self.state["phase"] == "intake" else "clarification"
        else:
            self.state["phase"] = "clarification"

    def _is_ready_to_run(self) -> bool:
        """判断是否可以开始建模"""
        p1 = self.state["missing_params"].get("priority_1", [])
        p2 = self.state["missing_params"].get("priority_2", [])
        return len(p1) == 0 and len(p2) == 0

    # ================================================================
    # 意图处理
    # ================================================================

    def _handle_modeling_flow(self, user_message: str, parse_result: dict) -> str:
        """处理建模流程（新建/修改/文件上传）"""
        entities = parse_result.get("normalized_params", {})
        missing = parse_result["missing_params"]

        # 1. 总结已获取的参数
        summary = self._summarize_progress(entities)

        # 2. 生成下一个问题
        questions = self._generate_questions(missing)

        # 3. 检查是否需要提供建议值
        suggestions = self._maybe_generate_suggestions(missing.get("priority_3", []))

        reply_parts = [summary]

        if suggestions:
            reply_parts.append("\n💡 以下参数可使用建议值：")
            for pname, sinfo in suggestions.items():
                reply_parts.append(
                    f"  • {self._param_display_name(pname)}: "
                    f"{sinfo['value']} {sinfo['unit']} ({sinfo['reason']})"
                )
                self.state["suggested_values"][pname] = sinfo
            reply_parts.append("  输入'使用建议值'自动填充所有建议值。")

        if questions:
            reply_parts.append(f"\n{questions}")

        if not questions and not suggestions:
            reply_parts.append("\n✅ 参数已齐全！输入'开始建模'启动自动建模。")

        return "\n".join(reply_parts)

    def _handle_confirmation(self) -> str:
        """处理用户确认"""
        self.state["phase"] = "confirmation"

        if self._is_ready_to_run():
            return (
                "✅ 参数已确认完备。\n\n"
                "📋 **建模方案摘要：**\n" + self._generate_model_summary() +
                "\n\n确认无误后，请输入 **'开始建模'** 启动全自动建模流程。"
            )
        else:
            return "已确认。但仍有部分参数需要明确：\n" + self._generate_questions(
                self.state["missing_params"]
            )

    def _handle_rejection(self) -> str:
        """处理用户拒绝"""
        # 清除最近一次的建议值
        if self.state["suggested_values"]:
            last_key = list(self.state["suggested_values"].keys())[-1]
            del self.state["suggested_values"][last_key]
            return (
                f"已撤销 {self._param_display_name(last_key)} 的建议值。"
                f"请提供您认为合适的值。"
            )
        return "好的，请告诉我您的想法，我们重新确认参数。"

    def _handle_query(self, message: str) -> str:
        """处理用户咨询"""
        return self.llm.chat(
            user_message=message,
            system_message=(
                "你是地下水数值模拟专家。请回答用户的水文地质咨询问题。"
                "回答应专业但易懂，必要时可引用参考文献。"
            ),
        )

    # ================================================================
    # 辅助方法
    # ================================================================

    def _summarize_progress(self, new_params: dict = None) -> str:
        """生成参数获取进度摘要"""
        new_params = new_params or {}
        ir = self.state["json_ir"]
        lines = ["📝 **当前参数状态：**"]

        # 项目信息
        proj = ir.get("project", {})
        name = proj.get("name", "未命名")
        mtype = proj.get("model_type", "未指定")
        lines.append(f"  ✅ 项目名称: {name}")

        mtype_labels = {
            "groundwater_flow": "地下水流",
            "solute_transport": "溶质运移",
            "reactive_transport": "活性传输",
        }
        mtype_display = mtype_labels.get(mtype, mtype)
        status = "✅" if mtype != "未指定" else "⚠️"
        lines.append(f"  {status} 模拟类型: {mtype_display}")

        # 模型范围
        domain = ir.get("domain", {})
        xr = domain.get("x_range")
        yr = domain.get("y_range")
        if xr and yr:
            lines.append(f"  ✅ 模型范围: {xr[0]:.0f}~{xr[1]:.0f}m × {yr[0]:.0f}~{yr[1]:.0f}m")
        else:
            lines.append(f"  ⚠️ 模型范围: 未指定")

        # 网格
        grid = ir.get("grid", {})
        nrow = grid.get("nrow")
        ncol = grid.get("ncol")
        if nrow and ncol:
            lines.append(f"  ✅ 网格: {nrow}行 × {ncol}列 × {grid.get('nlay', 1)}层")
        else:
            lines.append(f"  ⚠️ 网格: 未指定")

        # 含水层
        aqua = ir.get("aquifer", {})
        atype = aqua.get("type")
        atype_labels = {"confined": "承压", "unconfined": "非承压/潜水", "convertible": "可转换"}
        if atype:
            lines.append(f"  ✅ 含水层类型: {atype_labels.get(atype, atype)}")

        params = aqua.get("parameters", {})
        if "hk" in params and params["hk"].get("value") is not None:
            lines.append(f"  ✅ 渗透系数: {params['hk']['value']} {params['hk'].get('unit', 'm/d')}")
        else:
            lines.append(f"  ⚠️ 渗透系数: 未指定")

        # 边界
        bcs = ir.get("boundary_conditions", [])
        if bcs:
            bc_types = {bc.get("type") for bc in bcs}
            lines.append(f"  ✅ 边界条件: {len(bcs)} 个 ({', '.join(bc_types)})")
        else:
            lines.append(f"  ⚠️ 边界条件: 未指定")

        # 溶质
        transport = ir.get("transport", {})
        if transport.get("enabled"):
            lines.append(f"  ✅ 溶质运移: 已启用")

        return "\n".join(lines)

    def _generate_questions(self, missing: dict) -> str:
        """为缺失参数生成引导性问题"""
        p1 = missing.get("priority_1", [])
        p2 = missing.get("priority_2", [])
        p3 = missing.get("priority_3", [])

        if not p1 and not p2 and not p3:
            return ""

        # 使用 LLM 生成自然问题
        missing_summary = f"优先级1: {', '.join(p1) if p1 else '无'}\n优先级2: {', '.join(p2) if p2 else '无'}\n优先级3: {', '.join(p3) if p3 else '无'}"
        context = self._build_context_str()

        try:
            prompt = QUESTION_GENERATION_PROMPT.format(
                missing_params_summary=missing_summary,
                conversation_context=context,
            )
            return self.llm.chat(prompt).strip()
        except Exception as e:
            logger.warning(f"LLM问题生成失败，使用规则生成: {e}")
            return self._rule_based_questions(p1, p2, p3)

    def _rule_based_questions(self, p1: list, p2: list, p3: list) -> str:
        """基于规则生成问题（LLM不可用时的回退）"""
        questions = []

        # 合并前3个最重要的缺失参数
        targets = (p1 + p2 + p3)[:3]
        question_map = {
            "project.model_type": "请问您要模拟的是地下水流、溶质运移还是活性传输？",
            "domain.x_range": "请提供模型的东西向范围（单位：米）",
            "domain.y_range": "请提供模型的南北向范围（单位：米）",
            "grid.nrow": "请指定网格行数（建议 50-200 行）",
            "grid.ncol": "请指定网格列数（建议 50-200 列）",
            "aquifer.type": "请问含水层类型是承压、潜水还是可转换的？",
            "aquifer.parameters.hk": "请提供含水层的渗透系数（单位：m/d）",
            "boundary_conditions": "请描述模型的边界条件（如：西侧定水头100m，东侧定水头95m）",
            "grid.top": "请提供含水层顶板高程（单位：m）",
            "grid.bottom": "请提供含水层底板高程（单位：m）",
            "stress_periods": "请问模拟是稳态还是瞬态？如果是瞬态，需要多少个应力期？",
        }

        for param in targets:
            if param in question_map:
                questions.append(f"• {question_map[param]}")
            else:
                display = self._param_display_name(param)
                questions.append(f"• 请提供 {display} 的值")

        return "请问以下参数：\n" + "\n".join(questions)

    def _maybe_generate_suggestions(self, p3_params: list) -> dict:
        """为三级（优化性）缺失参数生成建议值"""
        if not p3_params:
            return {}

        suggestions = {}
        for pname in p3_params[:3]:  # 最多生成3个建议
            clean_name = pname.replace("aquifer.parameters.", "").replace("transport.", "")
            context = {"aquifer_type": self.state["json_ir"].get("aquifer", {}).get("type", "all")}
            suggestion = self.sem_db.get_suggested_value(clean_name, context["aquifer_type"], context)
            if suggestion and suggestion.get("confidence") != "none":
                suggestions[pname] = suggestion

        return suggestions

    def apply_all_suggestions(self) -> str:
        """一键应用所有建议值，更新 JSON IR"""
        applied = []
        for pname, sinfo in self.state["suggested_values"].items():
            self._set_ir_param(pname, sinfo["value"], sinfo["unit"], "auto_suggestion")
            applied.append(f"  • {self._param_display_name(pname)} = {sinfo['value']} {sinfo['unit']}")

        if applied:
            self.state["suggested_values"] = {}
            self.state["missing_params"] = self.parser.detect_missing_params(self.state["json_ir"])
            self._update_phase()
            return "✅ 已应用以下建议值：\n" + "\n".join(applied)
        return "当前没有可用的建议值。"

    def _set_ir_param(self, dot_path: str, value, unit: str = "", source: str = "user_input"):
        """通过点路径设置 JSON IR 中的参数值（如 'aquifer.parameters.hk'）"""
        ir = self.state["json_ir"]
        parts = dot_path.split(".")
        for part in parts[:-1]:
            if part not in ir:
                ir[part] = {}
            if isinstance(ir[part], list) and part == "boundary_conditions":
                break
            ir = ir[part]

        final_key = parts[-1]
        if isinstance(ir, dict):
            if isinstance(value, dict):
                ir[final_key] = value
            else:
                ir[final_key] = {"value": value, "unit": unit, "source": source}

    def _param_display_name(self, param_path: str) -> str:
        """获取参数的中文显示名"""
        clean = param_path.split(".")[-1]
        term = self.sem_db.get_term_by_name(clean)
        if term:
            return f"{term['display_name_cn']} ({clean})"
        return clean

    def _build_context_str(self) -> str:
        """构建对话上下文字符串"""
        recent = self.state["history"][-6:]  # 最近3轮
        lines = []
        for msg in recent:
            role = "用户" if msg["role"] == "user" else "系统"
            content = str(msg.get("content", ""))[:200]
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _generate_model_summary(self) -> str:
        """生成建模方案文本摘要"""
        ir = self.state["json_ir"]
        lines = []

        proj = ir.get("project", {})
        lines.append(f"**项目名称**: {proj.get('name', '未命名')}")

        domain = ir.get("domain", {})
        xr = domain.get("x_range", [0, 0])
        yr = domain.get("y_range", [0, 0])
        lines.append(f"**模型范围**: {xr[0]:.0f}~{xr[1]:.0f}m × {yr[0]:.0f}~{yr[1]:.0f}m")
        lines.append(f"**面积**: {(xr[1]-xr[0])*(yr[1]-yr[0])/1e6:.2f} km²")

        grid = ir.get("grid", {})
        lines.append(f"**网格**: {grid.get('nrow', '?')}行 × {grid.get('ncol', '?')}列 × {grid.get('nlay', '?')}层")

        aqua = ir.get("aquifer", {})
        atype_map = {"confined": "承压", "unconfined": "潜水", "convertible": "可转换"}
        lines.append(f"**含水层**: {atype_map.get(aqua.get('type', ''), '未知')}")

        for pname, pinfo in aqua.get("parameters", {}).items():
            v = pinfo.get("value")
            u = pinfo.get("unit", "")
            if v is not None:
                lines.append(f"  - {self._param_display_name(pname)}: {v} {u}")

        bcs = ir.get("boundary_conditions", [])
        lines.append(f"**边界条件**: {len(bcs)} 个")
        for bc in bcs:
            lines.append(f"  - {bc.get('type', '?')}: {bc.get('name', '?')}")

        transport = ir.get("transport", {})
        if transport.get("enabled"):
            lines.append(f"**溶质运移**: 已启用")
            for pname, pinfo in transport.get("dispersion", {}).items():
                v = pinfo.get("value")
                if v is not None:
                    lines.append(f"  - {self._param_display_name(pname)}: {v} {pinfo.get('unit', '')}")

        return "\n".join(lines)


# 全局实例
dialogue_manager = DialogueManager()

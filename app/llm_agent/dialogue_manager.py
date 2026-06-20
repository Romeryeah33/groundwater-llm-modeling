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
from app.knowledge.param_guide import PARAM_GUIDE, render_missing_params_guidance

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
            "json_ir": {"ir_version": "1.0"},  # 当前累积的 JSON IR
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

        # Step 1: 意图解析（携带对话历史，让 LLM 理解上下文）
        conversation_history = self._format_history_for_llm()
        parse_result = self.parser.parse(
            user_message,
            current_ir=self.state["json_ir"],
            session_context={"phase": self.state["phase"]},
            conversation_history=conversation_history,
        )

        intent = parse_result["intent"]
        # 防御：确保 json_ir 永远是 dict
        json_ir = parse_result.get("json_ir", {})
        if not isinstance(json_ir, dict):
            logger.warning(f"parse 返回的 json_ir 是 {type(json_ir).__name__}，重置为空 dict")
            json_ir = {}
        self.state["json_ir"] = json_ir
        self.state["missing_params"] = parse_result.get("missing_params", {"priority_1": [], "priority_2": [], "priority_3": []})

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

        # 3. 检查是否需要提供建议值（覆盖 P1/P2/P3 全部优先级）
        suggestions = self._maybe_generate_suggestions(missing)

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
        ir = self.state.get("json_ir", {})
        if not isinstance(ir, dict):
            ir = {}
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
        if isinstance(xr, (list, tuple)) and len(xr) >= 2 and isinstance(yr, (list, tuple)) and len(yr) >= 2:
            try:
                lines.append(f"  ✅ 模型范围: {float(xr[0]):.0f}~{float(xr[1]):.0f}m × {float(yr[0]):.0f}~{float(yr[1]):.0f}m")
            except (TypeError, ValueError):
                lines.append(f"  ⚠️ 模型范围: 数值格式异常")
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
        if isinstance(params, dict):
            hk_val = params.get("hk", {}) if isinstance(params.get("hk"), dict) else {}
            if hk_val.get("value") is not None:
                lines.append(f"  ✅ 渗透系数: {hk_val['value']} {hk_val.get('unit', 'm/d')}")
            else:
                lines.append(f"  ⚠️ 渗透系数: 未指定")
        else:
            lines.append(f"  ⚠️ 渗透系数: 未指定")

        # 边界
        bcs = ir.get("boundary_conditions", [])
        if isinstance(bcs, list) and len(bcs) > 0:
            bc_types = {bc.get("type", "?") for bc in bcs if isinstance(bc, dict)}
            bc_positions = {bc.get("position", bc.get("name", "?")) for bc in bcs if isinstance(bc, dict)}
            lines.append(f"  ✅ 边界条件: {len(bcs)} 个 ({', '.join(bc_types) if bc_types else '未知类型'})")
            # 检查是否4个侧面都已指定
            all_sides = {"west", "east", "north", "south"}
            specified = set()
            for bc in bcs:
                if not isinstance(bc, dict):
                    continue
                pos = (bc.get("position", "") or "").lower()
                name = (bc.get("name", "") or "").lower()
                for side in all_sides:
                    if side in pos or side in name:
                        specified.add(side)
            unspecified = all_sides - specified
            if unspecified:
                cn_map = {"west": "西", "east": "东", "north": "北", "south": "南"}
                cn_sides = [cn_map.get(s, s) for s in sorted(unspecified)]
                lines.append(f"     💡 未指定的{'/'.join(cn_sides)}侧默认为零流量边界（MODFLOW默认）")
        else:
            lines.append(f"  ⚠️ 边界条件: 未指定（未指定侧默认零流量，但至少需要一个驱动边界）")

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

    def _maybe_generate_suggestions(self, missing: dict) -> dict:
        """
        为所有缺失参数（P1/P2/P3）生成建议值

        优先级：
        1. 语义数据库查询（基于含水层类型的文献典型值）
        2. 硬编码默认值（无法从语义库获取的结构性参数）
        3. 跳过无法自动补全的 P1 参数（如 boundary_conditions、domain range）
        """
        if not missing:
            return {}

        p1 = missing.get("priority_1", [])
        p2 = missing.get("priority_2", [])
        p3 = missing.get("priority_3", [])
        all_missing = p1 + p2 + p3

        # 无法自动补全的 P1 参数 — 需要用户明确提供
        UNFILLABLE_P1 = {
            "boundary_conditions", "domain.x_range", "domain.y_range",
            "project.name", "project.model_type", "aquifer.type",
        }

        suggestions = {}
        aquifer_type = self.state["json_ir"].get("aquifer", {}).get("type", "all")

        for pname in all_missing:
            # 跳过必须由用户提供的参数
            if pname in UNFILLABLE_P1:
                continue

            # 提取参数短名（去掉路径前缀）
            clean_name = pname.split(".")[-1] if "." in pname else pname

            # 优先查语义库
            context = {"aquifer_type": aquifer_type}
            suggestion = self.sem_db.get_suggested_value(clean_name, aquifer_type, context)
            if suggestion and suggestion.get("confidence") != "none":
                suggestions[pname] = suggestion
                continue

            # 语义库无数据 → 尝试硬编码默认值
            hard_default = self._get_hardcoded_default(pname, aquifer_type)
            if hard_default:
                suggestions[pname] = hard_default

        return suggestions

    def _get_hardcoded_default(self, param_path: str, aquifer_type: str) -> dict:
        """
        为无法从语义库获取的参数提供硬编码默认值

        这些默认值基于典型水文地质条件：
        - 顶板高程 100m（假设地表附近）
        - 底板高程 50m（50m厚含水层）
        - 稳态单应力期
        - 承压/潜水含水层典型储水参数
        """
        defaults = {
            "grid.nrow": {
                "value": 50, "unit": "",
                "confidence": "low", "reason": "默认50行（请根据模型范围和精度需求调整）",
            },
            "grid.ncol": {
                "value": 50, "unit": "",
                "confidence": "low", "reason": "默认50列（请根据模型范围和精度需求调整）",
            },
            "grid.top": {
                "value": 100.0, "unit": "m",
                "confidence": "low", "reason": "默认顶板高程 100m（可自行修改）",
            },
            "grid.bottom": {
                "value": 50.0, "unit": "m",
                "confidence": "low", "reason": "默认底板高程 50m（50m厚含水层）",
            },
            "stress_periods": {
                "value": {
                    "nper": 1, "perlen": 1.0, "steady_state": True,
                    "period_data": [{"perlen": 1.0, "nstp": 1, "tsmult": 1.0}],
                },
                "unit": "",
                "confidence": "low",
                "reason": "默认稳态单应力期",
            },
            "aquifer.parameters.ss": {
                "value": 1e-5, "unit": "1/m",
                "confidence": "medium",
                "reason": f"{aquifer_type} 含水层典型储水系数",
            },
            "aquifer.parameters.sy": {
                "value": 0.15, "unit": "-",
                "confidence": "medium",
                "reason": f"{aquifer_type} 含水层典型给水度",
            },
            "aquifer.parameters.vka": {
                "value": None, "unit": "m/d",
                "confidence": "low",
                "reason": "默认取 hk × 0.1（Kh/Kv=10）",
            },
            "aquifer.parameters.porosity": {
                "value": 0.25, "unit": "-",
                "confidence": "medium",
                "reason": f"{aquifer_type} 含水层典型孔隙度",
            },
        }
        return defaults.get(param_path)

    def apply_all_suggestions(self) -> str:
        """一键应用所有建议值，更新 JSON IR，并告知无法自动补全的参数"""
        applied = []
        for pname, sinfo in self.state["suggested_values"].items():
            value = sinfo.get("value")
            unit = sinfo.get("unit", "")

            # 特殊处理 vka：若无从语义库获取，则根据 hk 自动计算 Kh/Kv=10
            if pname == "aquifer.parameters.vka" and value is None:
                hk_val = self._get_ir_value("aquifer.parameters.hk")
                if hk_val is not None:
                    value = hk_val * 0.1
                    sinfo["value"] = value
                    sinfo["reason"] = f"Kh/Kv=10，由 hk={hk_val} m/d 推算"

            if value is None:
                continue  # 跳过无法计算的值

            self._set_ir_param(pname, value, unit, "auto_suggestion")
            applied.append(f"  • {self._param_display_name(pname)} = {value} {unit}")

        # 更新缺失参数状态
        self.state["suggested_values"] = {}
        self.state["missing_params"] = self.parser.detect_missing_params(self.state["json_ir"])
        self._update_phase()

        reply_parts = []

        if applied:
            reply_parts.append("✅ 已应用以下建议值：\n" + "\n".join(applied))

        # ── 关键：检查补全后仍缺失的参数，生成详细指南 ──
        remaining_p1 = self.state["missing_params"].get("priority_1", [])
        remaining_p2 = self.state["missing_params"].get("priority_2", [])
        remaining_p3 = self.state["missing_params"].get("priority_3", [])

        if remaining_p1 or remaining_p2:
            guidance = self._generate_unfillable_params_guidance(
                remaining_p1, remaining_p2, remaining_p3
            )
            reply_parts.append(guidance)
        elif remaining_p3:
            reply_parts.append(
                f"\n💡 还有 {len(remaining_p3)} 个可选参数可使用默认值，"
                f"输入 **使用建议值** 再次补全。"
            )

        if not applied and not remaining_p1 and not remaining_p2:
            return "当前没有可用的建议值。"

        return "\n".join(reply_parts)

    def _generate_unfillable_params_guidance(
        self, p1: list, p2: list, p3: list
    ) -> str:
        """为无法自动补全的参数生成详细指南（委托给统一渲染器）"""
        return "\n---\n## ⚠️ 以下参数需要您手动提供（无法自动补全）\n\n" + \
            render_missing_params_guidance(p1, p2, p3, format="markdown")

    def _get_ir_value(self, dot_path: str):
        """从 JSON IR 中通过点路径读取参数值"""
        ir = self.state["json_ir"]
        parts = dot_path.split(".")
        for part in parts:
            if isinstance(ir, dict):
                ir = ir.get(part)
                if ir is None:
                    return None
            else:
                return None

        # 处理 {"value": ..., "unit": ...} 格式
        if isinstance(ir, dict):
            return ir.get("value")
        return ir

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
        """构建对话上下文字符串（用于问题生成等场景）"""
        recent = self.state["history"][-20:]  # 最近10轮
        lines = []
        for msg in recent:
            role = "用户" if msg["role"] == "user" else "系统"
            content = str(msg.get("content", ""))[:200]
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _format_history_for_llm(self) -> str:
        """
        格式化对话历史，用于传递给 LLM 的 prompt

        包含完整的历史消息（最近10轮），不截断到200字符，
        让 LLM 能真正理解对话上下文。
        """
        history = self.state.get("history", [])
        if not history:
            return "（这是对话的第一轮，没有历史记录）"

        # 取最近20轮（40条消息），足够覆盖复杂建模场景的完整上下文
        recent = history[-40:]
        lines = [f"共 {len(history)} 条历史消息，以下是最近 {len(recent)} 条："]
        for i, msg in enumerate(recent):
            role = "👤用户" if msg["role"] == "user" else "🤖系统"
            content = str(msg.get("content", ""))
            # 不截断内容，让 LLM 看到完整上下文
            lines.append(f"[{i+1}] {role}: {content}")
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
        if isinstance(xr, (list, tuple)) and len(xr) >= 2 and isinstance(yr, (list, tuple)) and len(yr) >= 2:
            try:
                lines.append(f"**模型范围**: {float(xr[0]):.0f}~{float(xr[1]):.0f}m × {float(yr[0]):.0f}~{float(yr[1]):.0f}m")
                lines.append(f"**面积**: {(float(xr[1])-float(xr[0]))*(float(yr[1])-float(yr[0]))/1e6:.2f} km²")
            except (TypeError, ValueError):
                lines.append(f"**模型范围**: 数值格式异常")
        else:
            lines.append(f"**模型范围**: 未指定")

        grid = ir.get("grid", {})
        lines.append(f"**网格**: {grid.get('nrow', '?')}行 × {grid.get('ncol', '?')}列 × {grid.get('nlay', '?')}层")

        aqua = ir.get("aquifer", {})
        atype_map = {"confined": "承压", "unconfined": "潜水", "convertible": "可转换"}
        lines.append(f"**含水层**: {atype_map.get(aqua.get('type', ''), '未知')}")

        for pname, pinfo in aqua.get("parameters", {}).items():
            if isinstance(pinfo, dict):
                v = pinfo.get("value")
                u = pinfo.get("unit", "")
            else:
                v = pinfo
                u = ""
            if v is not None:
                lines.append(f"  - {self._param_display_name(pname)}: {v} {u}")

        bcs = ir.get("boundary_conditions", [])
        if isinstance(bcs, list):
            lines.append(f"**边界条件**: {len(bcs)} 个")
            for bc in bcs:
                if isinstance(bc, dict):
                    lines.append(f"  - {bc.get('type', '?')}: {bc.get('name', '?')}")
        else:
            lines.append(f"**边界条件**: 格式异常")

        transport = ir.get("transport", {})
        if transport.get("enabled"):
            lines.append(f"**溶质运移**: 已启用")
            for pname, pinfo in transport.get("dispersion", {}).items():
                if isinstance(pinfo, dict):
                    v = pinfo.get("value")
                    u = pinfo.get("unit", "")
                else:
                    v = pinfo
                    u = ""
                if v is not None:
                    lines.append(f"  - {self._param_display_name(pname)}: {v} {u}")

        return "\n".join(lines)


# 全局实例
dialogue_manager = DialogueManager()

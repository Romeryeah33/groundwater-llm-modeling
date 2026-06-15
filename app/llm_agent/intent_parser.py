"""
意图解析引擎 — 自然语言 → JSON IR

处理管道：
  用户输入 → 意图分类 → NER实体抽取 → 语义库匹配 → 缺失检测 → 响应生成
"""

import json
import logging
from typing import Optional, Dict, List, Tuple

from app.llm_agent.llm_client import LLMClient, llm_client
from app.llm_agent.prompt_templates import (
    INTENT_CLASSIFICATION_PROMPT,
    NER_EXTRACTION_PROMPT,
    IR_GENERATION_PROMPT,
    MISSING_DETECTION_PROMPT,
)
from app.knowledge.semantic_db import SemanticDB, semantic_db

logger = logging.getLogger(__name__)


class IntentParser:
    """意图解析引擎"""

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        sem_db: Optional[SemanticDB] = None,
    ):
        self.llm = llm or llm_client
        self.sem_db = sem_db or semantic_db

    # ================================================================
    # Step 1: 意图分类
    # ================================================================

    def classify_intent(self, user_message: str) -> str:
        """
        识别用户意图类型

        Returns:
            new_model | modify_param | query | analyze | upload_file |
            confirm | reject | unclear
        """
        prompt = INTENT_CLASSIFICATION_PROMPT.format(user_message=user_message)
        try:
            result = self.llm.chat(prompt).strip().lower()
            valid_intents = [
                "new_model", "modify_param", "query", "analyze",
                "upload_file", "confirm", "reject", "unclear",
            ]
            if result in valid_intents:
                return result
            # 模糊匹配
            for intent in valid_intents:
                if intent in result:
                    return intent
            logger.warning(f"未知意图分类: {result}, 回退为 unclear")
            return "unclear"
        except Exception as e:
            logger.error(f"意图分类失败: {e}")
            return "unclear"

    # ================================================================
    # Step 2: 命名实体识别 (NER) + 参数抽取
    # ================================================================

    def extract_entities(
        self, user_message: str, context_info: dict = None
    ) -> List[dict]:
        """
        提取建模相关的命名实体和参数值

        Returns:
            [{"entity_type": "parameter_name", "value": 15.0, "unit": "m/d", "standard_name": "hk"}, ...]
        """
        context_str = ""
        if context_info:
            context_str = f"\n当前已有参数上下文：\n{json.dumps(context_info, ensure_ascii=False, indent=2)}"

        prompt = NER_EXTRACTION_PROMPT.format(
            user_message=user_message, context_info=context_str
        )
        try:
            result = self.llm.chat_json(prompt)
            if isinstance(result, list):
                return result
            elif isinstance(result, dict) and "entities" in result:
                return result["entities"]
            return []
        except Exception as e:
            logger.error(f"实体提取失败: {e}")
            return []

    # ================================================================
    # Step 3: 语义库匹配与标准化
    # ================================================================

    def normalize_entities(self, entities: List[dict]) -> Dict[str, dict]:
        """
        将提取的实体标准化：模糊匹配术语、统一参数名、转换单位

        Returns:
            {"hk": {"value": 15.0, "unit": "m/d", "source": "user_input", "confidence": "high"}, ...}
        """
        normalized = {}

        for entity in entities:
            raw_name = entity.get("entity_type", "")
            raw_value = entity.get("value")
            raw_unit = entity.get("unit", "")

            # 只处理参数类型的实体
            if raw_name not in ("parameter_name", "parameter_value"):
                continue

            name = entity.get("standard_name") or entity.get("value")
            if not name or not isinstance(name, str):
                continue

            # 在语义库中搜索标准参数名
            matches = self.sem_db.search_term(name)
            if matches:
                std_name = matches[0]["standard_name"]
            else:
                std_name = name.lower().strip()

            # 尝试从实体中提取数值
            value = entity.get("value")
            if value is not None and not isinstance(value, (int, float)):
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    value = None

            # 单位标准化
            unit = entity.get("unit") or self._infer_unit(std_name)
            if unit and unit != self._standard_unit(std_name):
                # 尝试单位换算
                std_unit = self._standard_unit(std_name)
                if std_unit and value is not None:
                    converted = self.sem_db.convert_unit(value, unit, std_unit)
                    if converted is not None:
                        value = converted
                        unit = std_unit

            normalized[std_name] = {
                "value": value,
                "unit": unit or "",
                "source": "user_input",
                "confidence": "high" if matches else "medium",
            }

        return normalized

    def _infer_unit(self, param_name: str) -> str:
        """推断参数默认单位"""
        term = self.sem_db.get_term_by_name(param_name)
        if term:
            return term.get("unit_standard", "")
        return ""

    def _standard_unit(self, param_name: str) -> str:
        """获取标准单位"""
        return self._infer_unit(param_name)

    # ================================================================
    # Step 4: 生成/更新 JSON IR
    # ================================================================

    def generate_json_ir(
        self,
        user_message: str,
        current_ir: Optional[dict] = None,
        extracted_entities: Optional[List[dict]] = None,
    ) -> dict:
        """调用 LLM 将提取信息合并到 JSON IR"""
        current_ir = current_ir or {}
        entities = extracted_entities or self.extract_entities(user_message, current_ir)

        prompt = IR_GENERATION_PROMPT.format(
            current_json_ir=json.dumps(current_ir, ensure_ascii=False, indent=2),
            extracted_entities=json.dumps(entities, ensure_ascii=False, indent=2),
            user_message=user_message,
        )
        try:
            return self.llm.chat_json(prompt)
        except Exception as e:
            logger.error(f"JSON IR生成失败: {e}")
            return current_ir

    # ================================================================
    # Step 5: 缺失参数检测
    # ================================================================

    def detect_missing_params(self, json_ir: dict) -> Dict[str, List[str]]:
        """
        检测 JSON IR 中缺失的参数，按优先级分组

        Returns:
            {"priority_1": [...], "priority_2": [...], "priority_3": [...]}
        """
        # 规则检测（快速，不耗LLM）
        rule_based = self._rule_based_missing_check(json_ir)

        if rule_based["priority_1"]:
            # 有关键缺失时，使用规则结果即可
            return rule_based

        # 无关键缺失时，用 LLM 细检
        try:
            prompt = MISSING_DETECTION_PROMPT.format(
                json_ir=json.dumps(json_ir, ensure_ascii=False, indent=2)
            )
            llm_result = self.llm.chat_json(prompt)
            return {
                "priority_1": llm_result.get("priority_1_missing", []),
                "priority_2": llm_result.get("priority_2_missing", []),
                "priority_3": llm_result.get("priority_3_missing", []),
            }
        except Exception:
            return rule_based

    def _rule_based_missing_check(self, json_ir: dict) -> Dict[str, List[str]]:
        """基于规则的缺失参数检测（快速路径）"""
        missing_p1 = []
        missing_p2 = []
        missing_p3 = []

        # 检查阻断性参数 (P1)
        if not json_ir.get("project", {}).get("name"):
            missing_p1.append("project.name")
        if not json_ir.get("project", {}).get("model_type"):
            missing_p1.append("project.model_type")
        if not json_ir.get("domain", {}).get("x_range"):
            missing_p1.append("domain.x_range")
        if not json_ir.get("domain", {}).get("y_range"):
            missing_p1.append("domain.y_range")
        if not json_ir.get("grid", {}).get("nrow"):
            missing_p1.append("grid.nrow")
        if not json_ir.get("grid", {}).get("ncol"):
            missing_p1.append("grid.ncol")
        if not json_ir.get("aquifer", {}).get("type"):
            missing_p1.append("aquifer.type")

        aqua_params = json_ir.get("aquifer", {}).get("parameters", {})
        if not aqua_params.get("hk") or aqua_params.get("hk", {}).get("value") is None:
            missing_p1.append("aquifer.parameters.hk")
        if not json_ir.get("boundary_conditions"):
            missing_p1.append("boundary_conditions")

        # 检查影响性参数 (P2)
        if not json_ir.get("grid", {}).get("top"):
            missing_p2.append("grid.top")
        if not json_ir.get("grid", {}).get("bottom"):
            missing_p2.append("grid.bottom")
        if not json_ir.get("stress_periods"):
            missing_p2.append("stress_periods")

        aug_type = json_ir.get("aquifer", {}).get("type", "")
        if aug_type in ("unconfined", "convertible"):
            if not aqua_params.get("sy") or aqua_params.get("sy", {}).get("value") is None:
                missing_p2.append("aquifer.parameters.sy")
        else:
            if not aqua_params.get("ss"):
                missing_p2.append("aquifer.parameters.ss")

        # 检查优化性参数 (P3)
        if not aqua_params.get("porosity"):
            missing_p3.append("aquifer.parameters.porosity")
        if not aqua_params.get("vka"):
            missing_p3.append("aquifer.parameters.vka")

        transport = json_ir.get("transport", {})
        if transport.get("enabled"):
            disp = transport.get("dispersion", {})
            if not disp.get("al") or disp.get("al", {}).get("value") is None:
                missing_p3.append("transport.dispersion.al")

        return {
            "priority_1": missing_p1,
            "priority_2": missing_p2,
            "priority_3": missing_p3,
        }

    # ================================================================
    # 完整管道：一键执行全部步骤
    # ================================================================

    def parse(
        self, user_message: str, current_ir: Optional[dict] = None, session_context: dict = None
    ) -> dict:
        """
        完整意图解析管道

        Args:
            user_message: 用户输入文本
            current_ir: 当前累积的 JSON IR
            session_context: 会话上下文（已答参数、对话历史等）

        Returns:
            {
                "intent": str,
                "entities": list,
                "json_ir": dict,
                "missing_params": dict,
                "needs_clarification": bool,
            }
        """
        current_ir = current_ir or {}
        session_context = session_context or {}

        # Step 1: 意图分类
        intent = self.classify_intent(user_message)
        logger.info(f"意图分类: {intent}")

        # Step 2: 实体提取
        entities = self.extract_entities(user_message, current_ir)

        # Step 3: 标准化
        normalized = self.normalize_entities(entities)

        # Step 4: 生成 JSON IR
        json_ir = self.generate_json_ir(user_message, current_ir, entities)

        # Step 5: 缺失参数检测
        missing = self.detect_missing_params(json_ir)

        # 判断是否需要澄清
        needs_clarification = (
            len(missing.get("priority_1", [])) > 0 or len(missing.get("priority_2", [])) > 0
        )

        return {
            "intent": intent,
            "entities": entities,
            "normalized_params": normalized,
            "json_ir": json_ir,
            "missing_params": missing,
            "needs_clarification": needs_clarification,
        }


# 全局实例
intent_parser = IntentParser()

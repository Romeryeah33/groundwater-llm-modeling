"""
意图解析引擎 — 自然语言 → JSON IR

处理管道：
  用户输入 → 意图分类 → NER实体抽取 → 语义库匹配 → 缺失检测 → 响应生成
"""

import json
import copy
import re
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

    # ── 标准 IR 顶层键白名单 ──
    VALID_TOP_KEYS = {
        "ir_version", "project", "domain", "grid", "aquifer",
        "boundary_conditions", "stress_periods", "solver",
        "transport", "output", "source_sinks", "initial_conditions",
        "observation_points",
    }

    # ── 已知的标准参数路径（用于检测/指南匹配） ──
    KNOWN_PARAM_PATHS = {
        "project.name", "project.model_type",
        "domain.x_range", "domain.y_range",
        "grid.nrow", "grid.ncol", "grid.nlay", "grid.delr", "grid.delc",
        "grid.top", "grid.bottom",
        "aquifer.type", "aquifer.parameters.hk", "aquifer.parameters.vka",
        "aquifer.parameters.hani", "aquifer.parameters.sy",
        "aquifer.parameters.ss", "aquifer.parameters.porosity",
        "stress_periods",
        "solver.type", "solver.hclose", "solver.rclose",
        "solver.mxiter", "solver.iter1",
        "transport.enabled", "transport.dispersion.al",
        "transport.dispersion.trpt", "transport.dispersion.trpv",
        "transport.reaction.decay.first_order_rate",
        "transport.reaction.sorption.kd",
        "boundary_conditions",
    }

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

    def classify_intent(
        self, user_message: str, conversation_history: str = "", current_ir: dict = None
    ) -> str:
        """
        识别用户意图类型

        Returns:
            new_model | modify_param | query | analyze | upload_file |
            confirm | reject | unclear
        """
        ir_summary = self._brief_ir_summary(current_ir or {})
        prompt = INTENT_CLASSIFICATION_PROMPT.format(
            user_message=user_message,
            conversation_history=conversation_history or "（无历史，这是第一轮对话）",
            current_ir_summary=ir_summary,
        )
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

    def _brief_ir_summary(self, ir: dict) -> str:
        """生成 IR 简要摘要，用于 intent classification 上下文"""
        if not ir:
            return "（空，尚无任何参数）"
        parts = []
        proj = ir.get("project", {})
        if proj.get("name"):
            parts.append(f"项目: {proj['name']}")
        if proj.get("model_type"):
            parts.append(f"类型: {proj['model_type']}")
        domain = ir.get("domain", {})
        if domain.get("x_range"):
            parts.append(f"范围X: {domain['x_range']}")
        if domain.get("y_range"):
            parts.append(f"范围Y: {domain['y_range']}")
        aqua = ir.get("aquifer", {})
        if aqua.get("type"):
            parts.append(f"含水层: {aqua['type']}")
        params = aqua.get("parameters", {})
        if isinstance(params, dict):
            for k, v in params.items():
                if isinstance(v, dict) and v.get("value") is not None:
                    parts.append(f"{k}={v['value']}")
        bcs = ir.get("boundary_conditions", [])
        if isinstance(bcs, list) and bcs:
            parts.append(f"边界: {len(bcs)}个")
        return "; ".join(parts) if parts else "（仅有框架）"

    # ================================================================
    # Step 2: 命名实体识别 (NER) + 参数抽取
    # ================================================================

    def extract_entities(
        self, user_message: str, context_info: dict = None, conversation_history: str = ""
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
            user_message=user_message,
            context_info=context_str,
            conversation_history=conversation_history or "（无历史，这是第一轮对话）",
        )
        try:
            result = self.llm.chat_json(prompt, expect_array=True)
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

        处理所有 NER 实体类型：
        - parameter_name: 参数名+值+单位 → IR 参数路径
        - parameter_value: 补充值信息
        - aquifer_type: 含水层类型 → aquifer.type
        - boundary_type: 边界类型 → 辅助边界条件构建
        - spatial_range: 空间范围 → 辅助 domain 构建
        - contaminant: 污染物 → transport.enabled + 污染物名
        - location: 位置 → 辅助坐标
        - time_range: 时间 → 辅助应力期

        Returns:
            Map of IR dot-path → {value, unit, source, confidence}
            e.g. {"aquifer.parameters.hk": {value: 15.0, ...}, "aquifer.type": {value: "confined", ...}}
        """
        normalized = {}

        for entity in entities:
            etype = entity.get("entity_type", "")
            raw_value = entity.get("value")
            raw_unit = entity.get("unit", "")
            std_name = (entity.get("standard_name") or "").lower().strip()

            if etype == "parameter_name":
                # 标准参数提取: {entity_type: "parameter_name", standard_name: "hk", value: 15.0, unit: "m/d"}
                name = std_name or str(entity.get("value", "")).lower().strip()
                if not name:
                    continue

                # ── 检查标准名是否已经是带数组索引的完整路径 ──
                # 如 "boundary_conditions[0].parameters.stage" → 直接使用
                # 先初始化，避免索引路径分支下 matched_name/matches 未定义
                # （数组路径是 LLM 直接给出的完整 IR 路径，未经语义库匹配，
                #   confidence 因此退化为 medium）
                matches = []
                matched_name = name
                if re.search(r'\[\d+\]', name) and '.' in name:
                    ir_path = name
                    # 验证顶层键有效
                    top_key_base = re.sub(r'\[\d+\]', '', ir_path.split('.')[0])
                    if top_key_base not in self.VALID_TOP_KEYS:
                        continue
                else:
                    # 在语义库中搜索标准参数名
                    matches = self.sem_db.search_term(name)
                    if matches:
                        matched_name = matches[0]["standard_name"]
                    else:
                        matched_name = name

                    # 映射到 IR dot-path
                    ir_path = self._param_to_ir_path(matched_name)
                    if not ir_path:
                        continue

                value = entity.get("value")
                # 跳过字符串类型的参数名（如标准名本身）
                if isinstance(value, str):
                    # 尝试从 additional_info 或 context 获取实际数值
                    extra = entity.get("numeric_value")
                    if extra is not None:
                        value = extra
                    else:
                        # 字符串值可能是分类值（如 aquifer type = "confined"）
                        # 保留字符串值，不尝试转换
                        pass

                if value is not None and not isinstance(value, (int, float)):
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        pass  # 保留字符串值（如分类值）

                # 单位标准化
                unit = raw_unit or self._infer_unit(matched_name)
                if unit and value is not None and isinstance(value, (int, float)):
                    # 只有用户提供非标准单位时才需要换算
                    if raw_unit and unit != raw_unit:
                        converted = self.sem_db.convert_unit(value, raw_unit, unit)
                        if converted is not None:
                            value = converted

                normalized[ir_path] = {
                    "value": value,
                    "unit": unit or "",
                    "source": "user_input",
                    "confidence": "high" if matches else "medium",
                }

            elif etype == "parameter_value":
                # 补充数值: {entity_type: "parameter_value", standard_name: "hk", value: 15.0, unit: "m/d"}
                if not std_name:
                    continue
                ir_path = self._param_to_ir_path(std_name)
                if not ir_path:
                    continue

                value = raw_value
                if value is not None and not isinstance(value, (int, float)):
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        continue

                unit = raw_unit or self._infer_unit(std_name)
                normalized[ir_path] = {
                    "value": value,
                    "unit": unit or "",
                    "source": "user_input",
                    "confidence": "medium",
                }

            elif etype == "aquifer_type":
                # 含水层类型: {entity_type: "aquifer_type", value: "承压", standard_name: "confined"}
                atype = std_name or str(raw_value).lower().strip() if raw_value else ""
                if atype in ("confined", "unconfined", "convertible", "承压", "潜水", "非承压", "可转换"):
                    type_map = {"承压": "confined", "潜水": "unconfined", "非承压": "unconfined", "可转换": "convertible"}
                    atype = type_map.get(atype, atype)
                    normalized["aquifer.type"] = {
                        "value": atype, "unit": "",
                        "source": "user_input", "confidence": "high",
                    }

            elif etype == "boundary_type":
                # 边界类型: {entity_type: "boundary_type", value: "定水头", standard_name: "chd"}
                btype = std_name or str(raw_value).lower().strip() if raw_value else ""
                bc_map = {
                    "chd": "chd", "定水头": "chd", "constant head": "chd",
                    "ghb": "ghb", "通用水头": "ghb", "general head": "ghb",
                    "wel": "wel", "井": "wel", "抽水井": "wel", "注水井": "wel", "well": "wel",
                    "riv": "riv", "河流": "riv", "river": "riv",
                    "drn": "drn", "排水沟": "drn", "drain": "drn",
                    "rch": "rch", "补给": "rch", "面状补给": "rch", "recharge": "rch",
                }
                mapped = bc_map.get(btype)
                if mapped:
                    # 存储为边界条件提示，供 LLM 构建完整边界
                    normalized["_boundary_hint"] = {
                        "value": {"type": mapped, "raw_description": str(raw_value)},
                        "unit": "",
                        "source": "user_input",
                        "confidence": "high",
                    }

            elif etype == "spatial_range":
                # 空间范围: {entity_type: "spatial_range", standard_name: "x_range", value: [0, 1000]}
                if isinstance(raw_value, (list, tuple)) and len(raw_value) == 2:
                    axis = std_name or "x_range"
                    ir_path = f"domain.{axis}" if axis in ("x_range", "y_range") else None
                    if ir_path:
                        normalized[ir_path] = {
                            "value": list(raw_value), "unit": "m",
                            "source": "user_input", "confidence": "high",
                        }

            elif etype == "contaminant":
                # 污染物: {entity_type: "contaminant", value: "苯", standard_name: "benzene"}
                normalized["_contaminant"] = {
                    "value": str(raw_value) if raw_value else "",
                    "unit": "",
                    "source": "user_input", "confidence": "medium",
                }
                # 同时启用溶质运移
                normalized["transport.enabled"] = {
                    "value": True, "unit": "",
                    "source": "inferred", "confidence": "high",
                }

            elif etype == "boundary_condition_full":
                # 完整边界条件: {entity_type: "boundary_condition_full", position: "west", bc_type: "chd", head_value: 100}
                position = entity.get("position", "").lower().strip()
                bc_type = entity.get("bc_type", "").lower().strip()
                # 先收敛成标量再往下走：LLM 会把"水头从212m变到208m再变到214m"
                # 抽成数组，这里统一取第一个，避免下游 float(list) 崩掉整条链路
                head_val = self._scalar(entity.get("head_value"))
                stage_val = self._scalar(entity.get("stage_value"))
                rate_val = self._scalar(entity.get("rate_value"))

                if not position or not bc_type:
                    continue

                # 跳过的零流量边界（MODFLOW 默认行为，无需显式设置）
                if bc_type == "no_flow":
                    normalized[f"_no_flow_boundary_{position}"] = {
                        "value": {"position": position, "type": "no_flow"},
                        "unit": "", "source": "user_input", "confidence": "high",
                    }
                    continue

                # 构建边界条件条目
                bc_entry = {
                    "type": bc_type,
                    "name": f"{position}_boundary",
                    "position": position,
                    "parameters": {},
                    "geometry": {"type": "edge", "position": position},
                }

                if bc_type == "chd" and head_val is not None:
                    bc_entry["parameters"]["head"] = {
                        "value": head_val, "unit": entity.get("unit", "m"),
                        "source": "user_input",
                    }
                elif bc_type == "ghb":
                    if head_val is not None:
                        bc_entry["parameters"]["head"] = {
                            "value": head_val, "unit": entity.get("unit", "m"),
                            "source": "user_input",
                        }
                    bc_entry["parameters"]["conductance"] = {
                        "value": entity.get("conductance", 100.0), "unit": "m²/d",
                        "source": "inferred",
                    }
                elif bc_type == "wel":
                    if rate_val is not None:
                        bc_entry["parameters"]["rate"] = {
                            "value": rate_val, "unit": entity.get("unit", "m³/d"),
                            "source": "user_input",
                        }
                elif bc_type == "riv":
                    if stage_val is not None:
                        bc_entry["parameters"]["stage"] = {
                            "value": stage_val, "unit": entity.get("unit", "m"),
                            "source": "user_input",
                        }
                    bc_entry["parameters"]["conductance"] = {
                        "value": entity.get("conductance", 100.0), "unit": "m²/d",
                        "source": "inferred",
                    }
                    bc_entry["parameters"]["rbot"] = {
                        "value": entity.get("rbot", float(stage_val or 0) - 1.0),
                        "unit": "m", "source": "inferred",
                    }
                elif bc_type == "rch":
                    if rate_val is not None:
                        bc_entry["parameters"]["rate"] = {
                            "value": rate_val, "unit": entity.get("unit", "m/d"),
                            "source": "user_input",
                        }
                    # 补给浓度
                    conc_val = self._scalar(entity.get("concentration"))
                    if conc_val is not None:
                        bc_entry["parameters"]["concentration"] = {
                            "value": conc_val, "unit": entity.get("conc_unit", "ppm"),
                            "source": "user_input",
                        }

                # 使用特殊键 _boundary_entry_N 来避免重复
                idx = len([k for k in normalized if k.startswith("_boundary_entry_")])
                normalized[f"_boundary_entry_{idx}"] = {
                    "value": bc_entry,
                    "unit": "",
                    "source": "user_input",
                    "confidence": "high",
                }

        # ── 自动启用溶质运移 ──
        # 当 model_type 为 solute_transport 或 reactive_transport 时，
        # 或检测到污染物/弥散/对流参数时，自动设置 transport.enabled = true
        model_type_val = normalized.get("project.model_type", {}).get("value")
        need_transport = (
            model_type_val in ("solute_transport", "reactive_transport")
            or any(k.startswith("transport.") or k == "_contaminant" for k in normalized)
        )
        if need_transport and "transport.enabled" not in normalized:
            normalized["transport.enabled"] = {
                "value": True, "unit": "",
                "source": "inferred", "confidence": "high",
            }

        return normalized

    def _param_to_ir_path(self, param_name: str) -> str:
        """将参数短名映射到 JSON IR dot-path"""
        PARAM_PATH_MAP = {
            # 含水层参数
            "hk": "aquifer.parameters.hk",
            "vka": "aquifer.parameters.vka",
            "hani": "aquifer.parameters.hani",
            "sy": "aquifer.parameters.sy",
            "ss": "aquifer.parameters.ss",
            "porosity": "aquifer.parameters.porosity",
            "n": "aquifer.parameters.porosity",
            # 网格
            "nrow": "grid.nrow",
            "ncol": "grid.ncol",
            "nlay": "grid.nlay",
            "delr": "grid.delr",
            "delc": "grid.delc",
            "top": "grid.top",
            "bottom": "grid.bottom",
            # 模型范围
            "x_range": "domain.x_range",
            "y_range": "domain.y_range",
            # 项目
            "project_name": "project.name",
            "model_type": "project.model_type",
            "name": "project.name",
            # 溶质
            "al": "transport.dispersion.al",
            "trpt": "transport.dispersion.trpt",
            "trpv": "transport.dispersion.trpv",
            "decay_rate": "transport.reaction.decay.first_order_rate",
            "kd": "transport.reaction.sorption.kd",
            # 井/源汇项
            "pumping_rate": "source_sinks.well.rate",
            "recharge": "source_sinks.recharge.rate",
            # 应力期
            # ⚠️ 这些必须落到 `stress_periods.<子字段>`，**绝不能**落到
            # `stress_periods` 本身。落到父路径会把 IR 写成裸标量
            # `"stress_periods": 10`，而在 `_merge_ir(patched_ir, json_ir)` 里
            # 标量不是 dict、`_is_empty(10)` 又为假，于是走"保留 base 非空值"
            # 分支 —— **LLM 产出的正确 dict（含 total_time）会被这个标量顶掉**，
            # 用户最终看到总时长 150000 d 而不是 15000 d。
            "stress_periods": "stress_periods.nper",
            "nper": "stress_periods.nper",
            "total_time": "stress_periods.total_time",
            "perlen": "stress_periods.perlen",
            "nstp": "stress_periods.nstp",
            "tsmult": "stress_periods.tsmult",
            # NER 把"每个应力期1个时间步"抽成这个 standard_name，不映射的话会落到
            # 默认的 `aquifer.parameters.time_steps_per_stress_period` —— 一个
            # 语义完全错误的位置（时间步数与含水层参数无关）。
            "time_steps_per_stress_period": "stress_periods.nstp",
        }
        return PARAM_PATH_MAP.get(param_name.lower(), f"aquifer.parameters.{param_name.lower()}")

    def _infer_unit(self, param_name: str) -> str:
        """推断参数默认单位"""
        term = self.sem_db.get_term_by_name(param_name)
        if term:
            return term.get("unit_standard", "")
        return ""

    def _standard_unit(self, param_name: str) -> str:
        """获取标准单位（兼容别名）"""
        return self._infer_unit(param_name)

    # ================================================================
    # Step 3.5: 直接写入 JSON IR（绕过 LLM，确保参数不丢失）
    # ================================================================

    def _patch_json_ir(self, json_ir: dict, normalized: dict) -> dict:
        """
        将标准化后的参数直接写入 JSON IR

        这是关键的可靠性保证：LLM 的 generate_json_ir 可能丢失参数，
        但通过此方法直接写入的参数一定会在 IR 中。

        IR 字段分为两类：
        - 结构字段（直接设值）：project.name, domain.x_range, grid.nrow, aquifer.type 等
        - 参数字段（包装为 {value, unit, source}）：aquifer.parameters.*, transport.dispersion.* 等
        """
        ir = copy.deepcopy(json_ir) if json_ir else {}

        # 确保 ir_version 始终存在
        if "ir_version" not in ir:
            ir["ir_version"] = "1.0"

        # 结构字段列表（直接设值，不包装）
        STRUCTURE_FIELDS = {
            "project.name", "project.model_type",
            "domain.x_range", "domain.y_range",
            "grid.nrow", "grid.ncol", "grid.nlay",
            "aquifer.type",
            "transport.enabled",
        }

        # 收集边界条件条目
        boundary_entries = []
        for key in sorted(normalized.keys()):
            if key.startswith("_boundary_entry_"):
                entry_info = normalized[key]
                entry = entry_info.get("value")
                if isinstance(entry, dict):
                    boundary_entries.append(entry)

        def _is_indexed_path(path: str) -> bool:
            """检查路径是否包含数组索引"""
            return bool(re.search(r'\[\d+\]', path))

        for ir_path, param_info in normalized.items():
            # 跳过特殊提示键（以 _ 开头）
            if ir_path.startswith("_"):
                continue

            value = param_info.get("value")
            unit = param_info.get("unit", "")
            source = param_info.get("source", "user_input")

            if value is None:
                continue

            # ── 带数组索引的路径：使用导航器处理 ──
            if _is_indexed_path(ir_path):
                # 决定值的写入格式
                if isinstance(value, (int, float, str, bool)):
                    write_value = {
                        "value": value, "unit": unit, "source": source,
                    }
                else:
                    write_value = value
                container, key_or_idx = self._navigate_ir_path(ir, ir_path, set_value=write_value)
                if container is None:
                    logger.warning(f"无法解析索引路径: {ir_path}")
                continue

            # ── 常规路径处理（无数组索引）──
            parts = ir_path.split(".")
            # 导航到目标位置，必要时创建中间节点
            current = ir
            for i, part in enumerate(parts[:-1]):
                if part not in current:
                    current[part] = {}
                elif not isinstance(current[part], dict):
                    current[part] = {}
                current = current[part]

            final_key = parts[-1]

            if ir_path in STRUCTURE_FIELDS:
                # 结构字段：直接设值
                current[final_key] = value
            elif "parameters" in ir_path or "dispersion" in ir_path or "reaction" in ir_path:
                # 参数字段：包装为 {value, unit, source}
                if isinstance(value, dict):
                    current[final_key] = value
                else:
                    current[final_key] = {
                        "value": value,
                        "unit": unit,
                        "source": source,
                    }
            else:
                # 其他字段：按值类型判断
                if isinstance(value, (list, dict)):
                    current[final_key] = value
                elif isinstance(value, bool):
                    current[final_key] = value
                else:
                    current[final_key] = value

        # ── 归位：`stress_periods` 必须是 dict，绝不能是裸标量 ──
        # 裸标量（老会话遗留的 `"stress_periods": 10`，或 LLM 把"10个应力期"
        # 直接写成数字）在 `_merge_ir` 里会**顶掉** LLM 产出的正确 dict，
        # 导致 `total_time` 丢失、总时长被按 `perlen × nper` 误算。
        # 这里统一收敛成 `{"nper": N}`，让后续深合并能正常进行。
        _sp = ir.get("stress_periods")
        if _sp is not None and not isinstance(_sp, dict):
            if isinstance(_sp, bool):
                ir.pop("stress_periods", None)
            elif isinstance(_sp, (int, float)):
                ir["stress_periods"] = {"nper": int(_sp)}
                logger.info(f"规范化 stress_periods: 裸标量 {_sp} → {{'nper': {int(_sp)}}}")
            elif isinstance(_sp, (list, tuple)):
                ir["stress_periods"] = {"period_data": list(_sp)}
                logger.info(f"规范化 stress_periods: 列表 → {{'period_data': [{len(_sp)} 期]}}")
            elif isinstance(_sp, str):
                ir.pop("stress_periods", None)
                logger.warning(f"丢弃无法解释的 stress_periods 字符串: {_sp!r}")
        elif isinstance(_sp, dict):
            # dict 但只有裸的 perlen 而没有 total_time / period_data 时提醒一次：
            # `perlen` 是**单期**长度，nper × perlen 才是总时长，最容易差 nper 倍。
            if (_sp.get("perlen") and not _sp.get("total_time")
                    and not _sp.get("period_data")):
                logger.warning(
                    f"stress_periods 只给了 perlen={_sp.get('perlen')}，按 MF6 语义它是"
                    f"**单期**长度，总时长将为 nper×perlen={_sp.get('nper', 1)}×"
                    f"{_sp.get('perlen')}；若该值是**总**模拟时长，请改写为 total_time"
                )

        # 将收集到的边界条目合并到 boundary_conditions 列表
        if boundary_entries:
            existing_bcs = ir.get("boundary_conditions", [])
            if not isinstance(existing_bcs, list):
                existing_bcs = []
            # 按位置去重：相同 position 的只保留一个
            existing_positions = set()
            clean_bcs = []
            for bc in existing_bcs:
                if isinstance(bc, dict):
                    # 用 or 而不是 get 的默认值：position 键存在但为 None/"" 时
                    # get 会直接返回 None，两边键取法不一致就配不上对
                    pos = bc.get("position") or bc.get("name") or ""
                    if pos not in existing_positions:
                        existing_positions.add(pos)
                        clean_bcs.append(bc)
            for entry in boundary_entries:
                pos = entry.get("position") or entry.get("name") or ""

                # 同位置的边界已存在 → **把新参数并进去**，不能把整条新条目丢掉。
                # 丢掉会让"把北部定水头边界的水头改成 215m"这类
                # "修改已有边界参数" 的请求永远落不到 IR 上：LLM 明明抽出了
                # type/name/parameters，走到这里被静默丢弃，用户看到的却是
                # missing_params 里那条参数一直挂着，怎么改都改不掉。
                # 注意 _boundary_entry_* 键在上面的主循环里被 continue 跳过，
                # 所以内层 parameters 只有这一条路径能写进 IR。
                target = None
                for bc in clean_bcs:
                    if isinstance(bc, dict) and (
                        (bc.get("position") or bc.get("name") or "") == pos
                    ):
                        target = bc
                        break

                if target is None:
                    existing_positions.add(pos)
                    clean_bcs.append(entry)
                    continue

                # 参数逐个合并：新值覆盖同名旧值
                new_params = entry.get("parameters")
                if isinstance(new_params, dict):
                    params = target.setdefault("parameters", {})
                    if isinstance(params, dict):
                        for pname, pinfo in new_params.items():
                            if pname in params:
                                logger.info(f"边界 {pos} 的参数 {pname} 被改写为 {pinfo}")
                            params[pname] = pinfo

                # 其余字段只在缺失时补齐。type 等身份字段不在这里覆盖：
                # LLM 对"改参数"这类请求给的 type 未必可靠，覆盖会静默换掉
                # 边界类型（chd ↔ ghb），比丢掉参数更危险。
                for k, v in entry.items():
                    if k == "parameters":
                        continue
                    if k not in target or target[k] in (None, "", {}):
                        target[k] = v
            ir["boundary_conditions"] = clean_bcs

        return ir

    @staticmethod
    def _is_empty(val) -> bool:
        """判断值是否为空（None, 空字符串, 空列表, 空dict）"""
        if val is None:
            return True
        if isinstance(val, str) and not val.strip():
            return True
        if isinstance(val, (list, dict)) and len(val) == 0:
            return True
        return False

    @staticmethod
    def _scalar(v):
        """把 LLM 可能给出的 list/dict 形态收敛成单个数值；取不到就返回 None

        LLM 会把"水头从212m线性变化到208m再变到214m"抽成
        `head_value=[212, 208, 214]`，直接 float() 会 TypeError，
        把整条对话链路崩掉（normalize_entities 抛异常 → parse 失败 →
        IR 退化成空 dict → 用户在界面上看到"什么都没抽到"）。

        dict 形态（{"value":..., "unit":...}）同理要解包。
        数组里取第一个数值，并告警——多值本身是"用户描述了一个变化过程"
        的信号，是否要展开成逐格边界应由上层决定，这里只保证不崩。
        """
        if isinstance(v, dict):
            v = v.get("value")
        if isinstance(v, (list, tuple)):
            nums = [x for x in v
                    if isinstance(x, (int, float)) and not isinstance(x, bool)]
            if not nums:
                return None
            if len(nums) > 1:
                logger.warning(f"参数值为数组 {v}，只取第一个 {nums[0]}")
            v = nums[0]
        if v is None or isinstance(v, bool):
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _has_value(v) -> bool:
        """格式无关的值检查：支持裸值和 dict-wrapped {value, unit, source} 两种格式"""
        if v is None:
            return False
        if isinstance(v, dict):
            inner = v.get("value")
            if inner is None:
                return False
            if isinstance(inner, str) and not inner.strip():
                return False
            if isinstance(inner, (list, dict)) and len(inner) == 0:
                return False
            return True
        if isinstance(v, str) and not v.strip():
            return False
        if isinstance(v, (list, dict)) and len(v) == 0:
            return False
        return True

    # ================================================================
    # 数组索引路径解析器
    # ================================================================

    @staticmethod
    def _parse_path_segment(segment: str) -> tuple:
        """
        解析路径段，分离键名和可选的数组索引。

        Examples:
            "boundary_conditions[0]" → ("boundary_conditions", 0)
            "parameters" → ("parameters", None)
            "hk" → ("hk", None)

        Returns:
            (key_name: str, index: int | None)
        """
        m = re.match(r'^(\w+)\[(\d+)\]$', segment)
        if m:
            return m.group(1), int(m.group(2))
        return segment, None

    @classmethod
    def _navigate_ir_path(cls, ir: dict, dot_path: str, set_value=None):
        """
        在 IR dict 中按点路径导航，支持数组索引。

        Args:
            ir: 目标 IR dict
            dot_path: 如 "boundary_conditions[0].parameters.stage"
            set_value: 如果不为 None，则在路径终点写入该值

        Returns:
            (container, key_or_index) 或写入成功返回 (container, key_or_index)
            如果路径不可达且 set_value 为 None，返回 (None, None)
        """
        segments = dot_path.split(".")
        current = ir

        for i, segment in enumerate(segments[:-1]):
            key, idx = cls._parse_path_segment(segment)

            if idx is not None:
                # 数组访问：先拿到列表
                if key not in current:
                    if set_value is not None:
                        current[key] = []
                    else:
                        return None, None
                arr = current[key]
                if not isinstance(arr, list):
                    if set_value is not None:
                        arr = []
                        current[key] = arr
                    else:
                        return None, None
                # 确保数组有足够长度
                if idx >= len(arr):
                    if set_value is not None:
                        while len(arr) <= idx:
                            arr.append({})
                    else:
                        return None, None
                current = arr[idx]
            else:
                # 普通 dict 访问
                if segment not in current:
                    if set_value is not None:
                        current[segment] = {}
                    else:
                        return None, None
                if not isinstance(current[segment], dict):
                    if set_value is not None:
                        current[segment] = {}
                    else:
                        return None, None
                current = current[segment]

        # 处理最终段
        final_seg = segments[-1]
        final_key, final_idx = cls._parse_path_segment(final_seg)

        if final_idx is not None:
            if final_key not in current:
                if set_value is not None:
                    current[final_key] = []
                else:
                    return None, None
            arr = current[final_key]
            if not isinstance(arr, list):
                if set_value is not None:
                    arr = []
                    current[final_key] = arr
                else:
                    return None, None
            if final_idx >= len(arr):
                if set_value is not None:
                    while len(arr) <= final_idx:
                        arr.append({})
                else:
                    return None, None
            if set_value is not None:
                arr[final_idx] = set_value
            return current, final_key
        else:
            if set_value is not None:
                current[final_key] = set_value
            return current, final_key

    # ================================================================
    # 参数路径净化
    # ================================================================

    def _is_known_param_path(self, path: str) -> bool:
        """检查参数路径是否为已知的标准路径"""
        if path in self.KNOWN_PARAM_PATHS:
            return True
        # 提取顶层键（去除括号索引如 boundary_conditions[0] → boundary_conditions）
        parts = path.split(".")
        top_key = parts[0] if parts else ""
        top_key_base = re.sub(r'\[\d+\]', '', top_key)  # 去掉数组索引
        # 检查顶层键是否在白名单中
        if top_key_base not in self.VALID_TOP_KEYS:
            return False
        # 检查前两段是否形成已知路径的前缀
        for known in self.KNOWN_PARAM_PATHS:
            if path == known or path.startswith(known + "."):
                return True
        # 对于 boundary_conditions 内部路径，允许子字段缺失检查
        if top_key_base == "boundary_conditions":
            return True
        if top_key_base in ("aquifer", "grid", "domain", "project", "transport", "solver", "initial_conditions"):
            return True
        return False

    def _sanitize_missing_paths(self, paths: list) -> list:
        """过滤缺失参数列表，只保留已知标准路径"""
        sanitized = []
        for p in paths:
            if self._is_known_param_path(p):
                sanitized.append(p)
            else:
                logger.debug(f"过滤非标准缺失路径: {p}")
        return sanitized

    def _clean_ir_nonstandard_keys(self, ir: dict) -> dict:
        """移除 LLM 在 IR 中引入的非标准顶层键"""
        if not isinstance(ir, dict):
            return ir
        cleaned = {}
        removed = []
        for key, val in ir.items():
            key_base = re.sub(r'\[\d+\]', '', key)
            if key_base in self.VALID_TOP_KEYS or key.startswith("_"):
                cleaned[key] = val
            else:
                removed.append(key)
                logger.debug(f"移除 IR 中非标准顶层键: {key}")
        if removed:
            logger.info(f"已从 IR 中移除非标准顶层键: {removed}")
        return cleaned

    def _merge_ir(self, base: dict, overlay: dict, _depth: int = 0) -> dict:
        """
        深度合并：base（累积历史 IR）提供已有值，overlay（LLM 输出）补充新增/空字段。

        规则：
        - base 中已有非空值 → 优先保留（前轮参数不丢失）
        - base 中为空值 → overlay 的值可以填入（LLM 可填充空字段）
        - overlay 中的新字段 → 加入结果
        - 嵌套 dict → 递归合并（**不受顶层白名单约束**，见下）

        `_depth`：白名单只在**顶层**（`_depth == 0`）生效。`VALID_TOP_KEYS` 装的是
        `project` / `grid` / `stress_periods` 这类**顶层**键名，把它套到每一层递归上
        会把所有嵌套键都判成"非标准键"丢掉 —— 实测
        `stress_periods = {"nper":10, "steady_state":false, "total_time":15000,
        "nstp":1}` 会被砍成 `{"nper":10}`，`total_time` 静默消失，总时长随后被按
        `perlen × nper` 误算成 150000 d 而不是 15000 d。白名单的注释本来就写着
        "顶层"，所以限制在顶层才是原意。
        """
        if not isinstance(overlay, dict):
            return copy.deepcopy(base) if base else {}
        result = copy.deepcopy(base) if base else {}
        for key, val in overlay.items():
            # 跳过 LLM 编造的非标准**顶层**键
            key_base = re.sub(r'\[\d+\]', '', key)
            if (_depth == 0 and key_base not in self.VALID_TOP_KEYS
                    and not key.startswith("_")):
                logger.debug(f"_merge_ir 跳过非标准顶层键: {key}")
                continue
            if key not in result:
                result[key] = copy.deepcopy(val) if isinstance(val, (dict, list)) else val
            elif isinstance(result[key], dict) and isinstance(val, dict):
                result[key] = self._merge_ir(result[key], val, _depth + 1)
            elif self._is_empty(result[key]) and not self._is_empty(val):
                # base 值为空，overlay 有值 → 接受 overlay
                result[key] = copy.deepcopy(val) if isinstance(val, (dict, list)) else val
            # 否则保留 base 的非空值（不覆盖）
        return result

    # ================================================================
    # Step 4: 生成/更新 JSON IR
    # ================================================================

    def generate_json_ir(
        self,
        user_message: str,
        current_ir: Optional[dict] = None,
        extracted_entities: Optional[List[dict]] = None,
        normalized_params: Optional[Dict[str, dict]] = None,
        conversation_history: str = "",
    ) -> dict:
        """调用 LLM 将提取信息合并到 JSON IR（仅处理复杂实体）"""
        current_ir = current_ir or {}
        entities = extracted_entities or self.extract_entities(user_message, current_ir)
        normalized_params = normalized_params or {}

        # 收集特殊提示（边界条件提示、污染物等）
        hints_parts = []
        for key in ("_boundary_hint", "_contaminant"):
            if key in normalized_params:
                hints_parts.append(f"{key}: {json.dumps(normalized_params[key], ensure_ascii=False)}")

        prompt = IR_GENERATION_PROMPT.format(
            conversation_history=conversation_history or "（无历史，这是第一轮对话）",
            current_json_ir=json.dumps(current_ir, ensure_ascii=False, indent=2),
            extracted_entities=json.dumps(entities, ensure_ascii=False, indent=2),
            user_message=user_message,
        )

        # 如果有边界/污染物提示，追加到 prompt
        if hints_parts:
            prompt += f"\n\n## 额外提示（已通过规则直接写入的参数不必重复处理）\n" + "\n".join(hints_parts)

        prompt += "\n\n注意：当前 JSON IR 中已包含通过规则直接写入的参数值，请保留它们不要覆盖。你只需要处理需要复杂结构推理的实体（如应力期的时间安排等）。"

        try:
            result = self.llm.chat_json(prompt)
            if not isinstance(result, dict):
                logger.warning(f"chat_json 返回了 {type(result).__name__} 而非 dict，回退到 current_ir")
                return current_ir
            # 清理 LLM 生成的 IR 中的非标准键
            result = self._clean_ir_nonstandard_keys(result)
            return result
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
        # 先清理 IR 中的非标准键，避免 LLM 编造路径
        json_ir = self._clean_ir_nonstandard_keys(json_ir)

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
            p1 = self._sanitize_missing_paths(llm_result.get("priority_1_missing", []))
            p2 = self._sanitize_missing_paths(llm_result.get("priority_2_missing", []))
            p3 = self._sanitize_missing_paths(llm_result.get("priority_3_missing", []))
            return {
                "priority_1": p1,
                "priority_2": p2,
                "priority_3": p3,
            }
        except Exception:
            return rule_based

    def _rule_based_missing_check(self, json_ir: dict) -> Dict[str, List[str]]:
        """基于规则的缺失参数检测（快速路径）"""
        missing_p1 = []
        missing_p2 = []
        missing_p3 = []

        # 缓存顶层 dict，避免重复 .get() 调用
        project = json_ir.get("project", {})
        domain = json_ir.get("domain", {})
        grid = json_ir.get("grid", {})
        aquifer = json_ir.get("aquifer", {})
        aqua_params = aquifer.get("parameters", {})
        aqua_type = aquifer.get("type", "")

        # 检查阻断性参数 (P1)
        if not self._has_value(project.get("name")):
            missing_p1.append("project.name")
        if not self._has_value(project.get("model_type")):
            missing_p1.append("project.model_type")
        if not self._has_value(domain.get("x_range")):
            missing_p1.append("domain.x_range")
        if not self._has_value(domain.get("y_range")):
            missing_p1.append("domain.y_range")
        if not self._has_value(grid.get("nrow")):
            missing_p1.append("grid.nrow")
        if not self._has_value(grid.get("ncol")):
            missing_p1.append("grid.ncol")
        if not self._has_value(aqua_type):
            missing_p1.append("aquifer.type")

        if not isinstance(aqua_params, dict) or not self._has_value(aqua_params.get("hk")):
            missing_p1.append("aquifer.parameters.hk")
        bcs = json_ir.get("boundary_conditions")
        if not bcs or not isinstance(bcs, list) or len(bcs) == 0:
            missing_p1.append("boundary_conditions")
        else:
            # 检查每个边界条件是否有必需的参数
            for i, bc in enumerate(bcs):
                if not isinstance(bc, dict):
                    continue
                bc_type = bc.get("type", "")
                bc_params = bc.get("parameters", {})
                if not isinstance(bc_params, dict):
                    bc_params = {}

                if bc_type == "ghb":
                    # GHB 必须指定水头（stage 或 head）
                    if not self._has_value(bc_params.get("stage")) and not self._has_value(bc_params.get("head")):
                        missing_p1.append(f"boundary_conditions[{i}].parameters.stage")
                    if not self._has_value(bc_params.get("conductance")):
                        missing_p2.append(f"boundary_conditions[{i}].parameters.conductance")

                elif bc_type == "chd":
                    if not self._has_value(bc_params.get("head")):
                        missing_p1.append(f"boundary_conditions[{i}].parameters.head")

                elif bc_type == "wel":
                    if not self._has_value(bc_params.get("rate")):
                        missing_p1.append(f"boundary_conditions[{i}].parameters.rate")

                elif bc_type == "riv":
                    if not self._has_value(bc_params.get("stage")):
                        missing_p1.append(f"boundary_conditions[{i}].parameters.stage")
                    if not self._has_value(bc_params.get("conductance")):
                        missing_p2.append(f"boundary_conditions[{i}].parameters.conductance")

                elif bc_type == "drn":
                    if not self._has_value(bc_params.get("elevation")):
                        missing_p1.append(f"boundary_conditions[{i}].parameters.elevation")

        # 检查影响性参数 (P2)
        if not self._has_value(grid.get("top")):
            missing_p2.append("grid.top")
        if not self._has_value(grid.get("bottom")):
            missing_p2.append("grid.bottom")
        if not self._has_value(json_ir.get("stress_periods")):
            missing_p2.append("stress_periods")
        # 初始条件：稳态模拟可选（用水力默认值），瞬态必须
        init = json_ir.get("initial_conditions", {})
        if not isinstance(init, dict) or len(init) == 0:
            missing_p2.append("initial_conditions")

        if isinstance(aqua_params, dict):
            if aqua_type in ("unconfined", "convertible"):
                if not self._has_value(aqua_params.get("sy")):
                    missing_p2.append("aquifer.parameters.sy")
            else:
                if not self._has_value(aqua_params.get("ss")):
                    missing_p2.append("aquifer.parameters.ss")

        # 检查优化性参数 (P3)
        if not self._has_value(grid.get("nlay")):
            missing_p3.append("grid.nlay")
        if isinstance(aqua_params, dict):
            if not self._has_value(aqua_params.get("porosity")):
                missing_p3.append("aquifer.parameters.porosity")
            if not self._has_value(aqua_params.get("vka")):
                missing_p3.append("aquifer.parameters.vka")

        transport = json_ir.get("transport", {})
        if self._has_value(transport.get("enabled")):
            disp = transport.get("dispersion", {})
            if isinstance(disp, dict):
                if not self._has_value(disp.get("al")):
                    missing_p3.append("transport.dispersion.al")
            else:
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
        self, user_message: str, current_ir: Optional[dict] = None,
        session_context: dict = None, conversation_history: str = "",
    ) -> dict:
        """
        完整意图解析管道

        Args:
            user_message: 用户输入文本
            current_ir: 当前累积的 JSON IR
            session_context: 会话上下文（已答参数、对话历史等）
            conversation_history: 格式化的对话历史文本

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

        # Step 1: 意图分类（带上下文）
        intent = self.classify_intent(user_message, conversation_history, current_ir)
        logger.info(f"意图分类: {intent}")

        # Step 2: 实体提取（带对话历史）
        entities = self.extract_entities(user_message, current_ir, conversation_history)

        # Step 3: 标准化 + 直接写入 IR（绕过 LLM，确保参数不丢失）
        normalized = self.normalize_entities(entities)
        logger.debug(f"标准化参数: {list(normalized.keys())}")

        # ── 关键修复：将可靠提取的参数直接写入 IR ──
        patched_ir = self._patch_json_ir(current_ir, normalized)

        # Step 4: LLM 辅助 — 处理复杂实体（应力期等），带对话历史
        json_ir = self.generate_json_ir(
            user_message, patched_ir, entities, normalized, conversation_history
        )

        # ── 防御 1: LLM 返回空/非dict 则回退 ──
        if not json_ir or not isinstance(json_ir, dict):
            logger.warning("LLM 返回的 IR 为空或非dict，回退到直接修补版")
            json_ir = patched_ir

        # ── 防御 2: 将直接提取的参数重新应用到 LLM 输出之上 ──
        json_ir = self._patch_json_ir(json_ir, normalized)

        # ── 防御 3: 深度合并 — 确保前几轮累积的所有参数不丢失 ──
        #   LLM 可能生成不完整的 IR（遗漏前轮已设定的字段），
        #   将 patched_ir（累积所有历史参数）作为底版，LLM 输出仅用于新增字段。
        json_ir = self._merge_ir(patched_ir, json_ir)

        # ── 防御 4: 清理 LLM 引入的非标准顶层键 ──
        json_ir = self._clean_ir_nonstandard_keys(json_ir)

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

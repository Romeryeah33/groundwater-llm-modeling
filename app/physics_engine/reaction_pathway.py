"""
生化反应路径智能辨识与校验

集成团队前期研究成果：
- ES&T 2023: 高维生化指标ML预测模型
- JHM 2025: 多级底物相互作用降解动力学
- 授权专利 ZL202111190476.2: 原位预测建模方法

功能：
1. 基于污染物类型自动推荐降解动力学参数
2. 检查降解途径的热力学/生物化学可行性
3. 多级反应链（PCE→TCE→DCE→VC）参数一致性校验
"""

import math
import logging
from typing import Optional, Dict, List, Tuple

from app.physics_engine.constraints import ConstraintResult, ConstraintSeverity

logger = logging.getLogger(__name__)


# ================================================================
# 已知污染物降解动力学数据（来自文献和团队前期成果）
# ================================================================

# 降解速率参考值（一级反应，单位 1/d）
BIODEGRADATION_RATES = {
    "BTEX": {
        "benzene": (1e-4, 0.1, "1/d"),
        "toluene": (1e-3, 0.5, "1/d"),
        "ethylbenzene": (5e-4, 0.3, "1/d"),
        "xylene": (3e-4, 0.2, "1/d"),
        "electron_acceptors": ["O2", "NO3-", "SO4^2-", "Fe(III)"],
    },
    "chlorinated_ethenes": {
        "PCE": (1e-5, 0.05, "1/d"),
        "TCE": (1e-5, 0.08, "1/d"),
        "cDCE": (1e-5, 0.03, "1/d"),
        "VC": (1e-5, 0.01, "1/d"),
        "ethene": (0.0, 0.0, "1/d"),  # 终端产物
        "electron_acceptors": ["H2"],
    },
    "chlorinated_ethanes": {
        "1,1,1-TCA": (1e-6, 0.02, "1/d"),
        "1,1-DCA": (1e-5, 0.03, "1/d"),
        "electron_acceptors": ["Fe(0)", "H2"],
    },
    "nitrogen": {
        "nitrate": (1e-3, 5.0, "1/d"),
        "nitrite": (1e-3, 5.0, "1/d"),
        "ammonium": (1e-4, 1.0, "1/d"),
        "electron_acceptors": ["O2", "organics"],
    },
    "petroleum_hydrocarbons": {
        "TPH": (1e-5, 0.05, "1/d"),
        "electron_acceptors": ["O2", "NO3-", "SO4^2-"],
    },
    "pesticides": {
        "atrazine": (1e-5, 0.01, "1/d"),
        "lindane": (1e-5, 0.005, "1/d"),
    },
}

# 多级反应链定义（前体 → 产物）
REACTION_CHAINS = {
    "PCE_degradation": {
        "chain": ["PCE", "TCE", "cDCE", "VC", "ethene"],
        "type": "reductive_dechlorination",
        "description": "四氯乙烯→三氯乙烯→顺-1,2-二氯乙烯→氯乙烯→乙烯",
        "requires_anaerobic": True,
        "rate_ratio": {
            "PCE_to_TCE": 1.0,
            "TCE_to_cDCE": 0.8,
            "cDCE_to_VC": 0.3,
            "VC_to_ethene": 0.1,
        },
    },
    "TCE_degradation": {
        "chain": ["TCE", "cDCE", "VC", "ethene"],
        "type": "reductive_dechlorination",
        "requires_anaerobic": True,
    },
}


class ReactionPathwayValidator:
    """生化反应路径智能校验器"""

    def __init__(self):
        pass

    # ================================================================
    # 污染物识别
    # ================================================================

    def identify_contaminant_category(self, contaminant_name: str) -> Optional[str]:
        """
        识别污染物所属类别

        Returns:
            BTEX | chlorinated_ethenes | chlorinated_ethanes |
            nitrogen | petroleum_hydrocarbons | pesticides | unknown
        """
        name_upper = contaminant_name.upper().strip()

        for category, info in BIODEGRADATION_RATES.items():
            if name_upper in info:
                return category

            # 模糊匹配
            for comp_name in info:
                if comp_name != "electron_acceptors" and comp_name in name_upper:
                    return category
                if name_upper in comp_name:
                    return category

        return "unknown"

    # ================================================================
    # 降解途径可行性校验
    # ================================================================

    def check_pathway_feasibility(
        self, contaminant: str, electron_acceptor: Optional[str] = None,
        redox_condition: str = "unknown",
    ) -> ConstraintResult:
        """
        检查降解途径的热力学/生化可行性

        Args:
            contaminant: 污染物名称
            electron_acceptor: 电子受体（如 O2, NO3-, SO4^2-, H2）
            redox_condition: 氧化还原条件 (oxic/anoxic/anaerobic/unknown)
        """
        category = self.identify_contaminant_category(contaminant)

        if category == "unknown":
            return ConstraintResult(
                constraint_type="reaction_pathway",
                param_name=contaminant,
                is_valid=True,
                severity=ConstraintSeverity.INFO,
                description=f"污染物 '{contaminant}' 不在已知降解数据库中，无法自动校验反应路径",
                suggestion="请手动确认降解途径的可行性",
            )

        info = BIODEGRADATION_RATES.get(category, {})
        known_e_acceptors = info.get("electron_acceptors", [])

        # 检查电子受体匹配
        if electron_acceptor and known_e_acceptors:
            if electron_acceptor not in known_e_acceptors:
                return ConstraintResult(
                    constraint_type="reaction_pathway",
                    param_name=f"{contaminant}_{electron_acceptor}",
                    is_valid=False,
                    severity=ConstraintSeverity.WARNING,
                    description=(
                        f"电子受体 '{electron_acceptor}' 不常用于 "
                        f"{contaminant}({category}) 的降解。"
                        f"已知受体: {', '.join(known_e_acceptors)}"
                    ),
                    suggestion=f"请确认 '{electron_acceptor}' 在此降解途径中的可行性，或考虑使用: {known_e_acceptors[0]}",
                )

        # 检查氧化还原条件
        if category == "chlorinated_ethenes":
            chain_info = REACTION_CHAINS.get("PCE_degradation", {})
            if chain_info.get("requires_anaerobic") and redox_condition == "oxic":
                return ConstraintResult(
                    constraint_type="reaction_pathway",
                    param_name=contaminant,
                    is_valid=False,
                    severity=ConstraintSeverity.WARNING,
                    description=(
                        f"氯代乙烯的还原脱氯需要厌氧条件，但当前设置 redox={redox_condition}。"
                        "在有氧条件下还原脱氯速率极低。"
                    ),
                    suggestion="如果场地确实存在还原脱氯，请确认模型设置为厌氧条件或使用极低的降解速率",
                )

        return ConstraintResult(
            constraint_type="reaction_pathway",
            param_name=contaminant,
            is_valid=True,
            severity=ConstraintSeverity.PASS,
            description=f"{contaminant} ({category}) 降解途径可行",
        )

    # ================================================================
    # 降解动力学参数推荐
    # ================================================================

    def suggest_kinetic_params(
        self, contaminant: str, aquifer_type: str = "all",
        redox_condition: str = "unknown",
    ) -> Optional[dict]:
        """
        基于污染物类型和含水层条件推荐降解动力学参数

        Returns:
            {first_order_rate: {value, unit, range}, half_life: {value, unit}, ...}
            或 None（无数据时）
        """
        category = self.identify_contaminant_category(contaminant)
        if category == "unknown":
            return None

        info = BIODEGRADATION_RATES.get(category, {})
        contam_upper = contaminant.upper()

        # 获取该污染物的速率范围
        rate_range = info.get(contam_upper)
        if rate_range is None:
            # 使用类别平均值
            all_rates = [
                v for k, v in info.items()
                if k != "electron_acceptors" and isinstance(v, tuple) and len(v) >= 2
            ]
            if all_rates:
                lo = sum(r[0] for r in all_rates) / len(all_rates)
                hi = sum(r[1] for r in all_rates) / len(all_rates)
                unit = all_rates[0][2] if len(all_rates[0]) > 2 else "1/d"
                rate_range = (lo, hi, unit)
            else:
                return None

        lo, hi = rate_range[0], rate_range[1]
        unit = rate_range[2] if len(rate_range) > 2 else "1/d"

        # 根据氧化还原条件调整推荐值
        if redox_condition == "anaerobic":
            typical = lo + (hi - lo) * 0.2  # 厌氧下取低值
        elif redox_condition == "oxic":
            typical = lo + (hi - lo) * 0.7  # 好氧下取高值
        else:
            typical = lo + (hi - lo) * 0.5  # 取中值

        typical = round(typical, 6)
        half_life = round(math.log(2) / typical, 1) if typical > 1e-10 else float('inf')

        return {
            "contaminant": contaminant,
            "category": category,
            "first_order_rate": {
                "value": typical,
                "unit": unit,
                "valid_range": [lo, hi],
                "confidence": "medium",
            },
            "half_life": {
                "value": half_life,
                "unit": "d",
                "note": f"t½ = ln(2)/λ = ln(2)/{typical} 1/d",
            },
            "reference": self._get_reference(category),
        }

    def _get_reference(self, category: str) -> str:
        """获取参考文献"""
        refs = {
            "BTEX": "Wiedemeier et al. (1999); Suarez & Rifai (1999)",
            "chlorinated_ethenes": "Wiedemeier et al. (1999); EPA (1998)",
            "chlorinated_ethanes": "EPA (1998); Vogel et al. (1987)",
            "nitrogen": "Korom (1992); Robertson et al. (1996)",
            "petroleum_hydrocarbons": "Wiedemeier et al. (1999)",
        }
        return refs.get(category, "")

    # ================================================================
    # 多级反应链一致性校验
    # ================================================================

    def check_reaction_chain_consistency(
        self, species_list: List[str], decay_rates: Dict[str, float],
    ) -> List[ConstraintResult]:
        """
        检查多级反应链中各产物的降解速率是否满足化学计量关系

        对于 PCE→TCE→cDCE→VC→ethene 反应链：
        - 前体的降解速率应 ≥ 产物的降解速率（否则产物会积累）
        - 各步速率比应在已知范围内
        """
        results = []

        for chain_name, chain_info in REACTION_CHAINS.items():
            chain = chain_info["chain"]
            species_in_chain = [s for s in chain if s in species_list]

            if len(species_in_chain) < 2:
                continue  # 不足2个，不构成反应链

            # 检查速率递减性
            for i in range(len(species_in_chain) - 1):
                parent = species_in_chain[i]
                daughter = species_in_chain[i + 1]

                rate_parent = decay_rates.get(parent)
                rate_daughter = decay_rates.get(daughter)

                if rate_parent is not None and rate_daughter is not None:
                    if rate_daughter > rate_parent:
                        results.append(ConstraintResult(
                            constraint_type="reaction_chain",
                            param_name=f"{parent}→{daughter}",
                            is_valid=False,
                            severity=ConstraintSeverity.WARNING,
                            current_value=rate_daughter / max(rate_parent, 1e-10),
                            description=(
                                f"反应链 {parent}→{daughter}: "
                                f"产物降解速率({rate_daughter} 1/d)大于前体({rate_parent} 1/d)，"
                                f"这可能导致中间产物快速消耗（不典型）"
                            ),
                            suggestion=f"通常前体降解速率应≥产物速率，请确认 {parent} 和 {daughter} 的反应速率",
                        ))

                # 检查速率比
                rate_ratios = chain_info.get("rate_ratio", {})
                step_name = f"{parent}_to_{daughter}"
                expected_ratio = rate_ratios.get(step_name)
                if expected_ratio and rate_parent and rate_daughter:
                    actual_ratio = rate_daughter / max(rate_parent, 1e-10)
                    if abs(actual_ratio - expected_ratio) > 0.5:
                        results.append(ConstraintResult(
                            constraint_type="reaction_chain",
                            param_name=f"{step_name}_ratio",
                            is_valid=False,
                            severity=ConstraintSeverity.INFO,
                            current_value=actual_ratio,
                            valid_range=(expected_ratio - 0.3, expected_ratio + 0.3),
                            description=(
                                f"{step_name} 速率比 = {actual_ratio:.3f}, "
                                f"文献典型值 ≈ {expected_ratio}"
                            ),
                            suggestion="可参考文献典型值调整，或保留当前值（如果场地数据支持）",
                        ))

        return results

    # ================================================================
    # 批量检测
    # ================================================================

    def check_json_ir_reactions(self, json_ir: dict) -> List[ConstraintResult]:
        """
        检查 JSON IR 中反应参数的完整性和一致性
        """
        results = []

        transport = json_ir.get("transport", {})
        if not transport.get("enabled"):
            return results

        reaction = transport.get("reaction", {})
        sorption = reaction.get("sorption", "none")
        decay = reaction.get("decay", {})

        # 检查吸附参数
        if sorption != "none":
            kd = reaction.get("kd", {}).get("value")
            if kd is None:
                results.append(ConstraintResult(
                    constraint_type="reaction",
                    param_name="kd",
                    is_valid=False,
                    severity=ConstraintSeverity.WARNING,
                    description=f"已启用吸附模型({sorption})但未提供分配系数 Kd",
                    suggestion=f"请提供 Kd 值或从文献获取：有机碳分配系数 Koc × foc",
                ))

        # 检查降解参数
        decay_type = decay.get("type", "none")
        if decay_type != "none":
            rate = decay.get("first_order_rate", {}).get("value")
            half_life = decay.get("half_life", {}).get("value")

            if rate is None and half_life is None:
                results.append(ConstraintResult(
                    constraint_type="reaction",
                    param_name="decay_rate",
                    is_valid=False,
                    severity=ConstraintSeverity.WARNING,
                    description=f"已启用降解模型({decay_type})但未提供降解速率或半衰期",
                    suggestion="请提供一级降解速率 λ (1/d) 或半衰期 t½ (d)",
                ))

            # 速率-半衰期一致性
            if rate is not None and half_life is not None:
                expected_rate = math.log(2) / max(half_life, 1e-10)
                if abs(rate - expected_rate) / max(expected_rate, 1e-10) > 0.1:
                    results.append(ConstraintResult(
                        constraint_type="reaction",
                        param_name="decay_consistency",
                        is_valid=False,
                        severity=ConstraintSeverity.WARNING,
                        current_value=rate,
                        description=f"λ={rate} 1/d 与 t½={half_life} d 不一致，期望 λ={expected_rate:.6f}",
                        suggestion=f"修正为 λ={expected_rate:.6f} 1/d 或 t½={math.log(2)/max(rate,1e-10):.1f} d",
                    ))

        # 检查多级反应
        species = transport.get("species", [])
        if len(species) > 1 and decay_type == "multi_species":
            # 提取反应速率
            species_rates = {}
            # 从 reaction.chain 或各 species 的独立设置中提取速率
            for sp in species:
                sp_info = reaction.get("species_params", {}).get(sp, {})
                sp_rate = sp_info.get("decay_rate", {}).get("value")
                if sp_rate is not None:
                    species_rates[sp.upper()] = sp_rate

            chain_results = self.check_reaction_chain_consistency(species, species_rates)
            results.extend(chain_results)

        return results


# 全局实例
reaction_validator = ReactionPathwayValidator()

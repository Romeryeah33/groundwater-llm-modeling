"""
物理约束闭环校验引擎

六层校验管道：
  JSON IR → 格式校验 → 参数范围 → 单位一致 → 物理一致 → 数值稳定 → 预运行 → 通过/修正
"""

import json
import logging
import math
import time
from typing import Optional, List, Dict

from app.llm_agent.llm_client import LLMClient, llm_client
from app.physics_engine.constraints import (
    ConstraintResult,
    ConstraintSeverity,
    MassBalanceConstraint,
    GoverningEquationConstraint,
    ParameterRangeConstraint,
    UnitConsistencyConstraint,
    NumericalStabilityConstraint,
)
from app.utils.ir_normalize import merge_time_sources, normalize_stress_periods

logger = logging.getLogger(__name__)


class ClosedLoopValidator:
    """
    物理约束闭环校验引擎

    工作流程：
    1. 接收 LLM 生成的 JSON IR
    2. 逐层进行物理校验
    3. 发现违规时生成物理指向性修正指令
    4. 将修正指令反馈给 LLM 进行重生成
    5. 迭代直到通过全部校验或达到最大轮次
    """

    MAX_ITERATIONS = 5       # 最大纠错轮次
    MAX_PARAM_CORRECTIONS = 3  # 单参数最多修正次数

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or llm_client
        self.param_range = ParameterRangeConstraint()
        self.unit_check = UnitConsistencyConstraint()

    # ================================================================
    # 主入口：完整校验管道
    # ================================================================

    def validate(self, json_ir: dict) -> dict:
        """
        对 JSON IR 进行多层物理校验并自适应纠错

        Args:
            json_ir: LLM 生成的 JSON IR

        Returns:
            {
                "valid": bool,          # 是否通过全部校验
                "ir": dict,             # 最终 JSON IR（可能被修正）
                "issues": list,         # 所有发现的问题
                "iterations": int,      # 消耗的迭代轮次
                "elapsed_s": float,     # 耗时(秒)
                "correction_history": list,  # 修正历史
            }
        """
        start_time = time.time()
        all_issues = []
        correction_history = []
        current_ir = json_ir

        for iteration in range(self.MAX_ITERATIONS):
            iteration_issues = []

            # === Layer 1: JSON Schema 格式校验 ===
            schema_issues = self._check_schema(current_ir)
            if schema_issues:
                iteration_issues.extend(schema_issues)

            # === Layer 2: 参数物理范围校验 ===
            range_issues = self.param_range.check_json_ir_params(current_ir)
            iteration_issues.extend([
                r for r in range_issues
                if r.severity == ConstraintSeverity.ERROR
            ])

            # === Layer 3: 单位一致性校验 ===
            unit_issues = self.unit_check.check_unit_consistency(current_ir)
            iteration_issues.extend(unit_issues)

            # === Layer 4: 物理一致性校验 ===
            physics_issues = self._check_physics(current_ir)
            iteration_issues.extend(physics_issues)

            # === Layer 5: 数值稳定性校验 ===
            stability_issues = self._check_stability(current_ir)
            iteration_issues.extend(stability_issues)

            # === Layer 5b: 时间离散自洽性 ===
            time_issues = self._check_time_discretization(current_ir)
            iteration_issues.extend(time_issues)

            # === Layer 6: 预运行校验 ===
            prun_issues = self._check_pre_run(current_ir)
            iteration_issues.extend(prun_issues)

            # === 判断是否全部通过 ===
            errors = [i for i in iteration_issues if i.severity == ConstraintSeverity.ERROR]
            warnings = [i for i in iteration_issues if i.severity == ConstraintSeverity.WARNING]

            all_issues.extend(iteration_issues)

            if not errors:
                logger.info(
                    f"校验通过 (迭代{iteration+1}轮, "
                    f"{len(warnings)}个警告, 耗时{time.time()-start_time:.2f}s)"
                )
                return {
                    "valid": True,
                    "ir": current_ir,
                    "issues": all_issues,
                    "warnings": warnings,
                    "iterations": iteration + 1,
                    "elapsed_s": round(time.time() - start_time, 3),
                    "correction_history": correction_history,
                }

            # === 生成修正指令 ===
            logger.info(f"校验迭代 {iteration+1}: 发现 {len(errors)} 个错误, {len(warnings)} 个警告")

            correction_record = {
                "iteration": iteration + 1,
                "errors_found": len(errors),
                "warnings_found": len(warnings),
                "errors": [self._issue_to_dict(e) for e in errors],
            }

            # 生成物理指向性修正指令
            correction_prompt = self._build_correction_prompt(current_ir, errors)

            # 调用 LLM 进行修正
            try:
                corrected_ir = self.llm.chat_json(
                    user_message=correction_prompt,
                    system_message=(
                        "你是地下水数值模型物理约束校验专家。"
                        "请根据修正指令修改 JSON IR 中的参数。"
                        "只输出修正后的完整 JSON IR，不要加任何解释。"
                        "确保所有参数在指定的物理范围内。"
                    ),
                )
                correction_record["status"] = "llm_corrected"
                current_ir = corrected_ir
            except Exception as e:
                logger.warning(f"LLM修正失败 (迭代{iteration+1}): {e}")
                # 尝试自动修正（规则方式）
                current_ir = self._auto_correct(current_ir, errors)
                correction_record["status"] = "auto_corrected"

            correction_history.append(correction_record)

        # 达到最大迭代次数
        remaining_errors = len([
            i for i in all_issues if i.severity == ConstraintSeverity.ERROR
        ])
        return {
            "valid": False,
            "ir": current_ir,
            "issues": all_issues,
            "iterations": self.MAX_ITERATIONS,
            "elapsed_s": round(time.time() - start_time, 3),
            "correction_history": correction_history,
            "detail": f"达到最大迭代次数({self.MAX_ITERATIONS})，仍有{remaining_errors}个错误",
        }

    # ================================================================
    # Layer 1: Schema 校验
    # ================================================================

    def _check_schema(self, json_ir: dict) -> List[ConstraintResult]:
        """JSON 格式与结构校验"""
        issues = []

        # 检查顶层必需字段
        required_top = ["ir_version", "project", "domain", "aquifer"]
        optional_autofill = ["grid", "solver", "stress_periods"]
        for field in required_top:
            if field not in json_ir:
                issues.append(ConstraintResult(
                    constraint_type="schema", param_name=field,
                    is_valid=False, severity=ConstraintSeverity.ERROR,
                    description=f"缺少顶层必需字段: {field}",
                    suggestion=f"添加 '{field}' 字段",
                ))
        for field in optional_autofill:
            if field not in json_ir:
                issues.append(ConstraintResult(
                    constraint_type="schema", param_name=field,
                    is_valid=True, severity=ConstraintSeverity.WARNING,
                    description=f"缺少可选字段: {field}（将由 AutoModelBuilder 自动补全）",
                    suggestion=f"可手动添加 '{field}' 字段以获得更精确控制",
                ))

        # 检查 project
        proj = json_ir.get("project", {})
        if "name" not in proj:
            issues.append(ConstraintResult(
                constraint_type="schema", param_name="project.name",
                is_valid=False, severity=ConstraintSeverity.ERROR,
                description="缺少项目名称",
                suggestion="添加 project.name 字段",
            ))

        model_type = proj.get("model_type", "")
        valid_types = ["groundwater_flow", "solute_transport", "reactive_transport"]
        if model_type not in valid_types:
            issues.append(ConstraintResult(
                constraint_type="schema", param_name="project.model_type",
                is_valid=False, severity=ConstraintSeverity.ERROR,
                current_value=model_type,
                description=f"无效的 model_type: '{model_type}'",
                suggestion=f"请设置为 {valid_types} 之一",
            ))

        # 检查 grid 数值类型
        grid = json_ir.get("grid", {})
        for int_param in ("nrow", "ncol", "nlay"):
            v = grid.get(int_param)
            if v is not None and not isinstance(v, int):
                issues.append(ConstraintResult(
                    constraint_type="schema", param_name=f"grid.{int_param}",
                    is_valid=False, severity=ConstraintSeverity.WARNING,
                    current_value=v,
                    description=f"grid.{int_param} 应该是整数，当前为 {type(v).__name__}",
                    suggestion=f"转换为整数",
                ))

        return issues

    # ================================================================
    # Layer 4: 物理一致性
    # ================================================================

    def _check_physics(self, json_ir: dict) -> List[ConstraintResult]:
        """物理一致性综合校验"""
        issues = []

        aqua_params = json_ir.get("aquifer", {}).get("parameters", {})
        domain = json_ir.get("domain", {})
        grid = json_ir.get("grid", {})
        transport = json_ir.get("transport", {})

        # --- 达西流速检查 ---
        # 用 _as_float 统一解包：裸数字 / {"value":…} / 字符串都能吃下，
        # 与下方 vka 走同一条路径（直接 .get("value") 遇裸数字会抛 AttributeError）
        hk = self._as_float(aqua_params.get("hk"))
        if hk is not None:
            xr = domain.get("x_range", [0, 1])
            yr = domain.get("y_range", [0, 1])
            L = max(xr[1] - xr[0], yr[1] - yr[0], 1)

            # 估计水力梯度（基于边界条件或默认值）
            bcs = json_ir.get("boundary_conditions", [])
            gradient = self._estimate_gradient(bcs, L)

            n = self._as_float(aqua_params.get("porosity"))
            if n is None:
                n = 0.25
            issues.append(
                GoverningEquationConstraint.check_darcy_velocity(hk, gradient, n)
            )

            # --- 雷诺数检查 ---
            issues.append(
                GoverningEquationConstraint.check_reynolds_number(hk, gradient)
            )

        # --- 溶质运移参数检查 ---
        if transport.get("enabled"):
            al = transport.get("dispersion", {}).get("al", {}).get("value")
            delr = grid.get("delr")
            # 如果 delr 是数组则取平均
            if isinstance(delr, (list, type(None))) and delr is not None and len(delr) > 0:
                if isinstance(delr, list):
                    delr = sum(delr) / len(delr)
                else:
                    import numpy as np
                    delr = np.mean(delr)

            if al is not None and delr is not None:
                issues.append(
                    GoverningEquationConstraint.check_grid_dispersion_criterion(al, delr)
                )

            # 检查溶质运移参数逻辑一致性
            decay = transport.get("reaction", {}).get("decay", {})
            half_life = decay.get("half_life", {}).get("value")
            first_order = decay.get("first_order_rate", {}).get("value")
            if half_life is not None and first_order is not None:
                # λ = ln(2)/t½
                expected_rate = math.log(2) / half_life
                diff = abs(first_order - expected_rate) / max(expected_rate, 1e-10)
                if diff > 0.1:  # 差异超过10%
                    issues.append(ConstraintResult(
                        constraint_type="physics",
                        param_name="decay_consistency",
                        is_valid=False,
                        severity=ConstraintSeverity.WARNING,
                        description=(
                            f"降解速率不一致: λ={first_order} 1/d, "
                            f"t½={half_life}d, 期望λ=ln(2)/t½={expected_rate:.6f} 1/d"
                        ),
                        suggestion=f"建议统一: λ={expected_rate:.6f} 1/d 或 t½={0.693/first_order:.1f} d",
                    ))

        # --- 含水层厚度检查 ---
        top = grid.get("top")
        bot = grid.get("bottom")
        # 防御性解包
        if isinstance(top, dict): top = top.get("value")
        if isinstance(bot, dict): bot = bot.get("value")
        if top is not None and bot is not None:
            thickness = top - bot
            if thickness <= 0:
                issues.append(ConstraintResult(
                    constraint_type="physics",
                    param_name="aquifer_thickness",
                    is_valid=False,
                    severity=ConstraintSeverity.ERROR,
                    current_value=thickness,
                    description=f"含水层厚度 = {thickness} < 0（顶板低于底板）",
                    suggestion="请确保 top > bottom",
                ))
            elif thickness > 5000:
                issues.append(ConstraintResult(
                    constraint_type="physics",
                    param_name="aquifer_thickness",
                    is_valid=False,
                    severity=ConstraintSeverity.WARNING,
                    current_value=thickness,
                    description=f"含水层厚度 = {thickness}m 极大（>5000m），请确认",
                    suggestion="检查顶底板高程是否正确（是否混淆了米和英尺？）",
                ))

        # --- 各向异性比检查 ---
        # 宽松取数：vka 可能是 {"value":…} 也可能是裸数字，直接 .get 会抛 AttributeError
        vka = self._as_float(aqua_params.get("vka"))
        if hk is not None and vka is not None:
            ratio = hk / max(vka, 1e-10)
            if ratio < 1:
                issues.append(ConstraintResult(
                    constraint_type="physics", param_name="anisotropy_ratio",
                    is_valid=False, severity=ConstraintSeverity.WARNING,
                    current_value=ratio,
                    description=f"Kh/Kv = {ratio:.2f} < 1（垂向渗透系数大于水平），不常见",
                    suggestion="通常 Kh ≥ Kv，请检查渗透系数设置",
                ))

        # --- hani 与 vka 同时给出 ---
        # `map_aquifer_params` 在两个都给定时一律采用 vka，另一支是裸 `pass` ——
        # 原先完全无声。用户改了 hani 却发现结果没变，只会以为参数没生效。
        hani = self._as_float(aqua_params.get("hani"))
        if hani is not None and vka is not None:
            msg = f"同时给出了 hani={hani} 与 vka={vka} m/d"
            if hk is not None and hani > 0:
                kv_from_hani = hk / hani
                rel = abs(kv_from_hani - vka) / max(abs(vka), 1e-10)
                msg += f"；由 hani 推得 Kv = {kv_from_hani:.4g} m/d"
                if rel > 0.1:
                    msg += f"，与 vka 相差 {rel * 100:.0f}%"
            issues.append(ConstraintResult(
                constraint_type="physics",
                param_name="anisotropy_conflict",
                is_valid=False,
                severity=ConstraintSeverity.WARNING,
                current_value=vka,
                description=f"{msg}。实际采用 vka，hani 被忽略",
                suggestion=("二者重复，只会生效一个：想按各向异性比给就删掉 vka，"
                            "想直接给垂向渗透系数就删掉 hani"),
            ))

        return issues

    @staticmethod
    def _as_float(v) -> Optional[float]:
        """宽松取数：``{value: ...}`` 解包 + 不可转换返回 None（不抛）"""
        if isinstance(v, dict):
            v = v.get("value")
        if v is None or isinstance(v, bool):
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _estimate_gradient(self, bcs: list, L: float) -> float:
        """从边界条件估计水力梯度"""
        chd_heads = []
        for bc in bcs:
            if bc.get("type") == "chd":
                params = bc.get("parameters", {})
                if isinstance(params.get("head"), dict):
                    h = params["head"].get("value")
                else:
                    h = params.get("head")
                if h is not None:
                    chd_heads.append(float(h))

        if len(chd_heads) >= 2:
            head_diff = abs(max(chd_heads) - min(chd_heads))
            return head_diff / L

        # 默认梯度 0.001~0.01（常见自然梯度）
        return 0.005

    # ================================================================
    # Layer 5: 数值稳定性
    # ================================================================

    def _check_stability(self, json_ir: dict) -> List[ConstraintResult]:
        """数值求解稳定性校验"""
        issues = []

        grid = json_ir.get("grid", {})
        nrow = grid.get("nrow", 50)
        ncol = grid.get("ncol", 50)

        # 网格质量
        issues.extend(
            NumericalStabilityConstraint.check_grid_quality(nrow, ncol)
        )

        # Courant 数检查（如果有应力期信息）
        stress = normalize_stress_periods(json_ir.get("stress_periods", {}))
        period_data = stress.get("period_data", [])
        # 合成回退：只有平铺 perlen/total_time、没有 period_data 时，
        # 之前这一层**完全瞎了**（`if period_data:` 直接跳过）。
        #
        # 但合成的时期**必须把严重度上限压到 WARNING**：`nstp` 缺省为 1 会算出
        # Δt=perlen 这种虚构的大步长，而 `model_builder._ensure_transport_timesteps`
        # 本来就会自己提升 nstp。用它报 ERROR 等于拿一个不存在的 Δt 去阻断用户。
        synthesized = False
        _n_before = len(issues)
        if not period_data:
            _perlen = self._as_float(stress.get("perlen"))
            _total = self._as_float(stress.get("total_time"))
            _nper = self._as_float(stress.get("nper"))
            if _total and _total > 0 and _nper and _nper > 0:
                _perlen_rep = _total / _nper
            else:
                _perlen_rep = _perlen
            if _perlen_rep and _perlen_rep > 0:
                period_data = [{"perlen": _perlen_rep,
                                "nstp": self._as_float(stress.get("nstp")) or 1}]
                synthesized = True

        if period_data:
            aqua = json_ir.get("aquifer", {}).get("parameters", {})
            hk = aqua.get("hk", {}).get("value", 1.0)
            n = aqua.get("porosity", {}).get("value", 0.25)

            delr = grid.get("delr", 20)
            if isinstance(delr, (list, float)):
                dx = delr if isinstance(delr, float) else (sum(delr)/len(delr) if delr else 20)
            else:
                dx = 20

            for pd in period_data:
                perlen = pd.get("perlen", 1.0)
                nstp = pd.get("nstp", 1)
                dt = perlen / nstp
                issues.append(
                    NumericalStabilityConstraint.check_courant_number(
                        hk * self._estimate_gradient(
                            json_ir.get("boundary_conditions", []),
                            max(grid.get("delr", 1), 1)
                        ), dt, dx, n
                    )
                )

                # 推荐时间步长
                domain_ranges = json_ir.get("domain", {})
                L = max(
                    domain_ranges.get("x_range", [0, 1])[1] - domain_ranges.get("x_range", [0, 0])[0],
                    domain_ranges.get("y_range", [0, 1])[1] - domain_ranges.get("y_range", [0, 0])[0],
                    1
                )
                grad = self._estimate_gradient(
                    json_ir.get("boundary_conditions", []), L
                )
                v_max = hk * max(abs(grad), 1e-10) / max(n, 0.01)
                issues.append(
                    NumericalStabilityConstraint.check_time_step(dx, v_max, n)
                )

        if synthesized:
            # 见上面的说明：这些结论建立在**合成**的时期上，Δt 是估计值。
            # 降级到 WARNING，绝不用它阻断建模。
            for _i in range(_n_before, len(issues)):
                issues[_i].severity = (
                    ConstraintSeverity.WARNING
                    if issues[_i].severity == ConstraintSeverity.ERROR
                    else issues[_i].severity
                )
                issues[_i].description = (
                    f"[基于合成的代表期，Δt 为估计值] {issues[_i].description}"
                )

        return issues

    # ================================================================
    # Layer 5b: 时间离散自洽性
    # ================================================================

    def _check_time_discretization(self, json_ir: dict) -> List[ConstraintResult]:
        """
        总时长 / 应力期数 / 单期长度三者的关系。

        为什么单独立一层：**单参数的上下界管不住这类错**。East Texas 案例里
        ``transport.stress_periods = {nper: 10, perlen: 15000}``，每个值都落在
        物理范围内，但按 MF6 语义 ``perlen`` 是**每个**应力期的长度，
        10 × 15000 = 150000 d，而提示词要的是 15000 d（"均匀分为10个应力期"）。
        用户"想看 360 天"落空的真正原因就在这里，而此前没有任何一层报错 ——
        `_check_stability` 只在 ``period_data`` 非空时跑 Courant，本案例没有
        ``period_data``，整层直接跳过。

        规则：
          - ``total_time`` 与 ``period_data`` 同时给出且总和不符（>1%）→ **ERROR**。
            两个来源直接矛盾，必有一个是错的，不能替用户猜。
          - ``total_time`` 与平铺 ``perlen`` 不符 → **WARNING**。
            ``model_mapper._map_time_params`` 已有确定的优先级（``total_time``
            按 nper 等分并覆盖平铺 perlen），模型能正确建出来，所以不阻断，
            但必须让用户看见。
          - 总时长 > 1e5 d 且启用了运移 → **WARNING**。长时段模拟本身合法
            （百年尺度污染运移很常见），只提醒确认。
        """
        issues: List[ConstraintResult] = []

        stress = merge_time_sources(json_ir)
        if not stress:
            return issues

        total_time = self._as_float(stress.get("total_time"))
        nper = self._as_float(stress.get("nper"))
        perlen = self._as_float(stress.get("perlen"))
        period_data = stress.get("period_data") or []
        if not isinstance(period_data, list):
            period_data = []
        period_data = [pd for pd in period_data if isinstance(pd, dict)]

        if period_data:
            span: Optional[float] = sum(
                self._as_float(pd.get("perlen")) or 0.0 for pd in period_data
            )
            n_per = len(period_data)
        elif perlen is not None and perlen > 0 and nper and nper > 0:
            span = perlen * nper
            n_per = int(nper)
        else:
            span = None
            n_per = int(nper) if nper else 1

        if total_time and total_time > 0 and span and span > 0:
            if abs(span - total_time) > 0.01 * total_time:
                if period_data:
                    issues.append(ConstraintResult(
                        constraint_type="time_discretization",
                        param_name="total_time",
                        is_valid=False,
                        severity=ConstraintSeverity.ERROR,
                        current_value=span,
                        valid_range=(total_time, total_time),
                        unit="d",
                        description=(
                            f"total_time={total_time:g} d 与 period_data 各期 perlen "
                            f"之和 {span:g} d 不符"
                            f"（相差 {abs(span-total_time)/total_time*100:.1f}%）"
                        ),
                        suggestion=(
                            f"二者只能有一个对：要么把 total_time 改成 {span:g} d，"
                            f"要么改 period_data 使 perlen 之和为 {total_time:g} d"
                        ),
                    ))
                else:
                    issues.append(ConstraintResult(
                        constraint_type="time_discretization",
                        param_name="perlen",
                        is_valid=True,
                        severity=ConstraintSeverity.WARNING,
                        current_value=perlen,
                        unit="d",
                        description=(
                            f"total_time={total_time:g} d 与平铺 perlen={perlen:g} d × "
                            f"nper={n_per}（共 {span:g} d）不符。perlen 按 MF6 语义是"
                            f"**单个**应力期的长度，极易被误读成总时长。"
                            f"建模时以 total_time 为准，每期将取 "
                            f"{total_time / max(n_per, 1):g} d"
                        ),
                        suggestion=(
                            f"若 {total_time:g} 天是**总**模拟时长，保留 total_time 即可"
                            "（不要再写平铺 perlen）；若确实是每期长度，请去掉 total_time"
                        ),
                    ))
            # 一致时不提示：total_time == perlen 只在 nper==1 时才自洽，
            # 而 nper>1 时一定会在上面命中 WARNING，无需额外分支

        # 长时段提醒
        if span and span > 1e5:
            hint = ""
            if not total_time and perlen and n_per > 1:
                hint = (
                    f"（若 {perlen:g} 天其实是**总**模拟时长，请改写成 "
                    f"\"total_time\": {perlen:g}，否则会被按每期 {perlen:g} 天解释）"
                )
            issues.append(ConstraintResult(
                constraint_type="time_discretization",
                param_name="total_time",
                is_valid=True,
                severity=ConstraintSeverity.WARNING,
                current_value=span,
                unit="d",
                description=(
                    f"模拟总时长为 {span:g} d（{span/365.25:.1f} 年），请确认是否符合预期。"
                    f"{hint}"
                ),
                suggestion="长时段模拟会显著增加运行时间与输出体积",
            ))

        return issues

    # ================================================================
    # Layer 6: FloPy 预运行
    # ================================================================

    def _check_pre_run(self, json_ir: dict) -> List[ConstraintResult]:
        """
        FloPy 预运行校验

        不使用 LLM，仅检查模型基本完整性：
        - 是否所有必需包都已配置
        - 应力期数据是否与边界条件匹配
        """
        issues = []

        # 检查边界条件是否与应力期数量匹配
        #
        # 必须走 normalize_stress_periods：`stress_periods` 在真实 IR 里可能是
        # 裸整数（East Texas 顶层就是 `10`，表示"分 10 个应力期"），直接
        # `.get("nper")` 会抛 `AttributeError: 'int' object has no attribute 'get'`，
        # 而本函数从 `validate()` 可达（`_check_stability` 有 isinstance 守卫，
        # 这里原先漏了）。
        stress = normalize_stress_periods(json_ir.get("stress_periods"))
        nper = stress.get("nper", 1)
        bcs = json_ir.get("boundary_conditions", [])

        for bc in bcs:
            sp_data = bc.get("stress_periods", [])
            if sp_data:
                periods_covered = {s.get("period", s.get("per", 0)) for s in sp_data}
                if len(periods_covered) != nper and nper > 1:
                    issues.append(ConstraintResult(
                        constraint_type="pre_run",
                        param_name=f"boundary.{bc.get('name', '?')}.stress_periods",
                        is_valid=False,
                        severity=ConstraintSeverity.WARNING,
                        description=f"边界'{bc.get('name')}'的应力期数据({len(periods_covered)}个)与模型总应力期({nper}个)不匹配",
                        suggestion="确保每个应力期都有对应的边界条件数据",
                    ))

        return issues

    # ================================================================
    # 修正指令生成
    # ================================================================

    def _build_correction_prompt(self, json_ir: dict, errors: List[ConstraintResult]) -> str:
        """
        构建物理指向性修正指令

        关键：不给通用的"请修正"，而是具体指出：
        - 哪个参数出错
        - 违反了哪条物理规则
        - 建议的修正方向和取值区间
        """
        prompt_parts = [
            "以下 JSON IR 参数存在物理约束违规，请逐一修正：\n",
            "=" * 50,
            f"\n违规总数: {len(errors)} 个\n\n",
        ]

        for i, err in enumerate(errors):
            prompt_parts.append(f"### 违规 {i+1}: {err.param_name}")
            prompt_parts.append(f"- 约束类型: {err.constraint_type}")
            prompt_parts.append(f"- 当前值: {err.current_value}")
            if err.valid_range:
                prompt_parts.append(f"- 有效范围: [{err.valid_range[0]}, {err.valid_range[1]}]")
            prompt_parts.append(f"- 单位: {err.unit}")
            prompt_parts.append(f"- 说明: {err.description}")
            prompt_parts.append(f"- 修正建议: {err.suggestion}")
            prompt_parts.append("")

        prompt_parts.extend([
            "=" * 50,
            "\n## 当前 JSON IR:\n",
            "```json",
            json.dumps(json_ir, ensure_ascii=False, indent=2),
            "```",
            "\n## 要求",
            "1. 修正以上所有违规参数",
            "2. 保持未违规参数不变",
            "3. 修正后的参数必须在指定物理范围内",
            "4. 输出完整的修正后 JSON IR（不要省略字段）",
            "\n只输出修正后的 JSON，不要加任何解释文字。",
        ])

        return "\n".join(prompt_parts)

    def _auto_correct(self, json_ir: dict, errors: List[ConstraintResult]) -> dict:
        """
        自动修正（LLM不可用时的规则回退）

        只处理可自动修正的错误（范围超限、类型错误等）
        无法自动修正的（逻辑错误、物理不一致）需要人工介入
        """
        import copy
        ir = copy.deepcopy(json_ir)

        for err in errors:
            if err.constraint_type == "param_range" and err.valid_range:
                lo, hi = err.valid_range
                val = err.current_value

                # 自动夹紧到有效范围
                if isinstance(val, (int, float)):
                    if val < lo:
                        corrected = lo
                    elif val > hi:
                        corrected = hi
                    else:
                        continue

                    # 尝试更新 JSON IR
                    self._update_ir_value(ir, err.param_name, corrected)

        return ir

    def _update_ir_value(self, ir: dict, param_name: str, new_value: float):
        """尝试通过参数名更新 JSON IR 中的值"""
        # 遍历 aquifer.parameters
        for pname, pinfo in ir.get("aquifer", {}).get("parameters", {}).items():
            if pname == param_name and isinstance(pinfo, dict):
                pinfo["value"] = new_value
                pinfo["source"] = "auto_corrected"
                return

        # 遍历 transport 参数
        transport = ir.get("transport", {})
        for pname, pinfo in transport.get("dispersion", {}).items():
            if pname == param_name and isinstance(pinfo, dict):
                pinfo["value"] = new_value
                return

        # 遍历 solver 参数
        solver = ir.get("solver", {}).get("parameters", {})
        if param_name in solver:
            solver[param_name] = new_value

    def _issue_to_dict(self, issue: ConstraintResult) -> dict:
        return {
            "constraint_type": issue.constraint_type,
            "param_name": issue.param_name,
            "severity": issue.severity.value,
            "current_value": issue.current_value,
            "valid_range": issue.valid_range,
            "unit": issue.unit,
            "description": issue.description,
            "suggestion": issue.suggestion,
        }


# ================================================================
# 便捷函数：单次快速校验（不调用LLM）
# ================================================================

def quick_validate(json_ir: dict) -> dict:
    """
    快速校验（不调用LLM）
    适用于前端实时反馈

    Returns:
        {valid: bool, error_count: int, warning_count: int, issues: list}
    """
    validator = ClosedLoopValidator()

    all_issues = []
    all_issues.extend(validator._check_schema(json_ir))
    all_issues.extend(validator.param_range.check_json_ir_params(json_ir))
    all_issues.extend(validator.unit_check.check_unit_consistency(json_ir))
    all_issues.extend(validator._check_physics(json_ir))
    all_issues.extend(validator._check_stability(json_ir))
    all_issues.extend(validator._check_time_discretization(json_ir))

    errors = [i for i in all_issues if i.severity == ConstraintSeverity.ERROR]
    warnings = [i for i in all_issues if i.severity == ConstraintSeverity.WARNING]

    return {
        "valid": len(errors) == 0,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "issues": [validator._issue_to_dict(i) for i in all_issues],
    }


# 全局实例
closed_loop_validator = ClosedLoopValidator()

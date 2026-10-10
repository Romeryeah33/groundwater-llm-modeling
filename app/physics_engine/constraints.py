"""
多维物理约束规则库

五大类约束规则的数学化定义与硬编码实现：
1. 质量守恒与水量平衡
2. 控制方程约束（达西定律、对流弥散方程）
3. 参数物理取值空间约束
4. 多维单位一致性校验
5. 数值求解稳定性判定
"""

import math
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

import numpy as np

from app.knowledge.semantic_db import SemanticDB, semantic_db
from app.utils.ir_normalize import normalize_stress_periods

logger = logging.getLogger(__name__)


# ================================================================
# 约束检查结果数据结构
# ================================================================

class ConstraintSeverity(Enum):
    ERROR = "error"       # 阻断性错误，必须修正
    WARNING = "warning"   # 警告，建议修正
    INFO = "info"         # 信息提示
    PASS = "pass"         # 通过


@dataclass
class ConstraintResult:
    """约束检查结果"""
    constraint_type: str           # 约束类型
    param_name: str                # 参数名
    is_valid: bool                 # 是否通过
    severity: ConstraintSeverity   # 严重级别
    current_value: Optional[float] = None
    valid_range: Optional[Tuple[float, float]] = None
    unit: str = ""
    description: str = ""
    suggestion: str = ""


# ================================================================
# 约束类1：质量守恒与水量平衡
# ================================================================

class MassBalanceConstraint:
    """质量守恒定律约束"""

    WATER_BALANCE_TOLERANCE = 0.01   # 水量平衡容忍度 1%
    MASS_BALANCE_TOLERANCE = 0.02    # 质量守恒容忍度 2%

    @staticmethod
    def check_water_balance_pre(
        boundary_in: float, boundary_out: float,
        well_in: float, well_out: float,
        recharge: float = 0.0,
    ) -> ConstraintResult:
        """
        建模前水量平衡预估检查

        检查流入≈流出（稳态时）
        |in - out| / max(in, out) < tolerance
        """
        total_in = abs(boundary_in) + abs(well_in) + abs(recharge)
        total_out = abs(boundary_out) + abs(well_out)

        if total_in < 1e-10 and total_out < 1e-10:
            return ConstraintResult(
                constraint_type="mass_balance",
                param_name="water_balance_pre",
                is_valid=True,
                severity=ConstraintSeverity.PASS,
                description="稳态模型无外部流量",
            )

        error = abs(total_in - total_out)
        denom = max(total_in, total_out, 1e-10)
        relative_error = error / denom

        is_valid = relative_error < MassBalanceConstraint.WATER_BALANCE_TOLERANCE
        return ConstraintResult(
            constraint_type="mass_balance",
            param_name="water_balance_pre",
            is_valid=is_valid,
            severity=ConstraintSeverity.ERROR if not is_valid else ConstraintSeverity.PASS,
            current_value=relative_error,
            valid_range=(0.0, MassBalanceConstraint.WATER_BALANCE_TOLERANCE),
            unit="dimensionless",
            description=f"水量平衡预估误差 = {relative_error*100:.4f}% "
                        f"(流入={total_in:.2f}, 流出={total_out:.2f})",
            suggestion="调整边界流量或抽注水速率使流入≈流出" if not is_valid else "",
        )

    @staticmethod
    def check_water_balance_post(
        flow_in: float, flow_out: float,
        storage_change: float = 0.0,
    ) -> ConstraintResult:
        """
        模拟后水量平衡验证（使用Flopy输出）

        |In - Out - ΔS| / max(|In|, |Out|) < tolerance
        """
        error = abs(flow_in - flow_out - storage_change)
        denom = max(abs(flow_in), abs(flow_out), 1e-10)
        relative_error = error / denom

        is_valid = relative_error < MassBalanceConstraint.WATER_BALANCE_TOLERANCE
        return ConstraintResult(
            constraint_type="mass_balance",
            param_name="water_balance_post",
            is_valid=is_valid,
            severity=ConstraintSeverity.ERROR if not is_valid else ConstraintSeverity.PASS,
            current_value=relative_error,
            valid_range=(0.0, MassBalanceConstraint.WATER_BALANCE_TOLERANCE),
            unit="dimensionless",
            description=f"水量平衡误差 = {relative_error*100:.4f}%",
            suggestion="收紧求解器收敛准则或减小时间步长" if not is_valid else "",
        )

    @staticmethod
    def check_mass_balance(
        total_source: float, total_sink: float,
        mass_change: float, initial_mass: float,
    ) -> ConstraintResult:
        """
        溶质质量守恒检查

        |Source - Sink - ΔM| / M₀ < tolerance
        """
        error = abs(total_source - total_sink - mass_change)
        denom = max(abs(initial_mass), 1e-10)
        relative_error = error / denom

        is_valid = relative_error < MassBalanceConstraint.MASS_BALANCE_TOLERANCE
        return ConstraintResult(
            constraint_type="mass_conservation",
            param_name="mass_balance",
            is_valid=is_valid,
            severity=ConstraintSeverity.ERROR if not is_valid else ConstraintSeverity.PASS,
            current_value=relative_error,
            valid_range=(0.0, MassBalanceConstraint.MASS_BALANCE_TOLERANCE),
            unit="dimensionless",
            description=f"质量守恒误差 = {relative_error*100:.4f}%",
            suggestion="检查源汇项设置，确保溶质质量守恒" if not is_valid else "",
        )


# ================================================================
# 约束类2：控制方程约束
# ================================================================

class GoverningEquationConstraint:
    """地下水动力学控制方程约束"""

    @staticmethod
    def check_darcy_velocity(
        K: float, gradient: float, n: float = 0.25,
    ) -> ConstraintResult:
        """
        达西定律流速检查: v = -K * i / n (实际流速/孔隙流速)

        检查流速是否在物理合理范围（通常 0.001~100 m/d）
        """
        if abs(gradient) < 1e-10:
            v_pore = 0.0
        else:
            v_pore = abs(K * gradient / n)

        is_valid = 0.0005 <= v_pore <= 500  # m/d
        severity = ConstraintSeverity.ERROR if v_pore > 1000 else (
            ConstraintSeverity.WARNING if not is_valid else ConstraintSeverity.PASS
        )

        return ConstraintResult(
            constraint_type="darcy_law",
            param_name="darcy_velocity",
            is_valid=is_valid,
            severity=severity,
            current_value=v_pore,
            valid_range=(0.0005, 500.0),
            unit="m/d",
            description=f"实际(孔隙)流速 = {v_pore:.4f} m/d (K={K} m/d, i={gradient:.4f}, n={n})",
            suggestion=f"流速异常: 请检查渗透系数({K})或水力梯度({gradient})是否合理" if not is_valid else "",
        )

    @staticmethod
    def check_reynolds_number(
        K: float, gradient: float, d50: float = 0.001,
        nu: float = 1.0e-6,
    ) -> ConstraintResult:
        """
        雷诺数检查: Re = v*d₅₀/ν

        Re < 1~10 满足达西流（层流）条件
        Re > 10 可能进入非达西流区域
        """
        v_darcy = abs(K * gradient)
        Re = v_darcy * d50 / nu

        is_valid = Re < 10.0
        severity = ConstraintSeverity.WARNING if not is_valid else ConstraintSeverity.PASS

        return ConstraintResult(
            constraint_type="reynolds_number",
            param_name="reynolds",
            is_valid=is_valid,
            severity=severity,
            current_value=Re,
            valid_range=(0.0, 10.0),
            unit="dimensionless",
            description=f"雷诺数 Re = {Re:.3f} "
                        f"(达西流速={v_darcy:.4f} m/d, d₅₀={d50:.4f} m)",
            suggestion="雷诺数>10，可能进入非达西流区域，考虑使用非线性流模型" if not is_valid else "",
        )

    @staticmethod
    def check_peclet_transport(
        v: float, dx: float, D: float,
    ) -> ConstraintResult:
        """
        运移Peclet数检查: Pe = v·Δx/D

        Pe < 2: 弥散为主
        2 < Pe < 10: 过渡区
        Pe > 10: 对流为主（需注意数值振荡，建议TVD格式或网格加密）
        """
        if D < 1e-20:
            D = 1e-20
        Pe = abs(v * dx / D)

        is_valid = Pe < 50.0  # 网格Peclet不宜过大
        severity = (
            ConstraintSeverity.ERROR if Pe > 100 else
            ConstraintSeverity.WARNING if Pe > 50 else
            ConstraintSeverity.INFO if Pe > 10 else
            ConstraintSeverity.PASS
        )

        suggestion = ""
        if Pe > 50:
            suggestion = f"Pe={Pe:.1f}过大，强烈建议加密网格(当前Δx={dx}m)或启用TVD格式"
        elif Pe > 10:
            suggestion = f"Pe={Pe:.1f}偏大，建议局部网格加密或启用TVD格式"
        elif Pe < 0.1:
            suggestion = f"Pe={Pe:.1f}极小，弥散完全主导，可适当放宽网格精度"

        return ConstraintResult(
            constraint_type="peclet_transport",
            param_name="grid_peclet",
            is_valid=is_valid,
            severity=severity,
            current_value=Pe,
            valid_range=(0.1, 50.0),
            unit="dimensionless",
            description=f"运移Peclet数 Pe = v·Δx/D = {Pe:.2f} (v={v:.4f} m/d, Δx={dx} m)",
            suggestion=suggestion,
        )

    @staticmethod
    def check_grid_dispersion_criterion(
        alpha_l: float, dx: float,
    ) -> ConstraintResult:
        """
        网格弥散准则: Δx ≤ αL (网格尺寸应不大于纵向弥散度)

        这是MODFLOW/MT3DMS运行经验法则：
        - Δx/αL ≤ 1: 理想
        - 1 < Δx/αL ≤ 4: 可接受
        - Δx/αL > 4: 数值弥散严重
        """
        if alpha_l < 1e-10:
            return ConstraintResult(
                constraint_type="grid_dispersion",
                param_name="grid_dispersion",
                is_valid=True,
                severity=ConstraintSeverity.PASS,
                description="弥散度极小（纯对流），跳过网格弥散检查",
            )

        ratio = dx / alpha_l
        is_valid = ratio < 4.0
        severity = (
            ConstraintSeverity.ERROR if ratio > 10 else
            ConstraintSeverity.WARNING if ratio > 4 else
            ConstraintSeverity.PASS
        )

        return ConstraintResult(
            constraint_type="grid_dispersion",
            param_name="grid_dispersion_ratio",
            is_valid=is_valid,
            severity=severity,
            current_value=ratio,
            valid_range=(0.0, 4.0),
            unit="dimensionless",
            description=f"Δx/αL = {ratio:.1f} (Δx={dx}m, αL={alpha_l}m)",
            suggestion=f"网格尺寸{dx}m相对于弥散度{alpha_l}m过大，建议加密至≤{alpha_l}m" if not is_valid else "",
        )


# ================================================================
# 约束类3：参数物理取值空间约束
# ================================================================

class ParameterRangeConstraint:
    """参数物理取值范围约束"""

    # 硬编码的物理取值范围表（回退方案，优先使用语义库）
    PHYSICAL_RANGES: Dict[str, tuple] = {
        "hk": (1e-6, 1e4, "m/d"),
        "vka": (1e-6, 1e4, "m/d"),
        "sy": (0.005, 0.4, "dimensionless"),
        "ss": (1e-7, 1e-2, "1/m"),
        "porosity": (0.05, 0.5, "dimensionless"),
        "al": (0.01, 1000.0, "m"),
        "trpt": (0.001, 0.5, "dimensionless"),
        "trpv": (0.0001, 0.1, "dimensionless"),
        "diffc": (1e-11, 1e-7, "m2/d"),
        "kd": (0.0, 1e6, "L/kg"),
        "decay_rate": (1e-6, 10.0, "1/d"),
        "pumping_rate": (-50000.0, 50000.0, "m3/d"),
        "recharge_rate": (0.0, 5000.0, "mm/yr"),
        "nrow": (1, 10000, "dimensionless"),
        "ncol": (1, 10000, "dimensionless"),
        "nlay": (1, 200, "dimensionless"),
        "hclose": (1e-8, 0.1, "m"),
        "max_iter": (10, 50000, "dimensionless"),
        # ── 时间离散（stress_periods）──
        # perlen 上界刻意取 1e5：East Texas 案例的 15000 仍然合法，不新造错误。
        # 真正的量纲/语义问题（perlen 是"单期长度"而非"总时长"）由
        # `validator._check_time_discretization` 负责，那里能同时看到
        # nper / total_time / period_data 三者的关系，比单参数上下界更准。
        "perlen": (1e-3, 1e5, "d"),
        "nstp": (1, 1e6, "dimensionless"),
        "nper": (1, 1000, "dimensionless"),
        "total_time": (1e-3, 1e7, "d"),
    }

    def __init__(self, sem_db: Optional[SemanticDB] = None):
        self.sem_db = sem_db or semantic_db

    def check_param(
        self, param_name: str, value, aquifer_type: str = "all",
    ) -> ConstraintResult:
        """检查单个参数值是否在物理合理范围内"""

        # 防御性解包：如果 value 是 {value: ..., unit: ...} 格式则取 value
        if isinstance(value, dict):
            value = value.get("value")
        if value is None:
            return ConstraintResult(
                constraint_type="param_range",
                param_name=param_name,
                is_valid=True,
                severity=ConstraintSeverity.INFO,
                current_value=None,
                description=f"参数'{param_name}'值为空，跳过检查",
            )
        # 安全转换为 float：跳过不可转换的非数值（如 "steady" 等字符串/布尔）
        if isinstance(value, bool):
            value = float(value)
        elif isinstance(value, str):
            try:
                value = float(value)
            except (ValueError, TypeError):
                import logging
                logging.getLogger("physics").warning(
                    f"参数'{param_name}'值为非数值字符串'{value}'，跳过范围检查"
                )
                return ConstraintResult(
                    constraint_type="param_range",
                    param_name=param_name,
                    is_valid=True,
                    severity=ConstraintSeverity.WARNING,
                    current_value=value,
                    description=f"参数'{param_name}'值为非数值字符串'{value}'，跳过范围检查（可能是布尔/枚举值）",
                )
        else:
            try:
                value = float(value)
            except (ValueError, TypeError):
                return ConstraintResult(
                    constraint_type="param_range",
                    param_name=param_name,
                    is_valid=True,
                    severity=ConstraintSeverity.INFO,
                    current_value=str(value),
                    description=f"参数'{param_name}'无法转换为数值，跳过范围检查",
                )

        # 优先从语义库查询
        db_range = self.sem_db.get_param_range(param_name, aquifer_type)
        if db_range:
            lo, hi = db_range["min_value"], db_range["max_value"]
            unit = db_range["unit"]
        elif param_name in self.PHYSICAL_RANGES:
            lo, hi, unit = self.PHYSICAL_RANGES[param_name]
        else:
            # 未知参数，默认通过
            return ConstraintResult(
                constraint_type="param_range",
                param_name=param_name,
                is_valid=True,
                severity=ConstraintSeverity.INFO,
                current_value=value,
                description=f"参数'{param_name}'不在已知物理范围表中，默认通过",
            )

        is_valid = lo <= value <= hi
        severity = ConstraintSeverity.ERROR if not is_valid else ConstraintSeverity.PASS

        suggestion = ""
        if not is_valid:
            if value < lo:
                suggestion = f"{param_name}={value} 低于物理下限{lo}，建议增大至≥{lo}"
            else:
                suggestion = f"{param_name}={value} 超出物理上限{hi}，建议减小至≤{hi}"

        # 获取中文显示名
        term = self.sem_db.get_term_by_name(param_name)

        return ConstraintResult(
            constraint_type="param_range",
            param_name=param_name,
            is_valid=is_valid,
            severity=severity,
            current_value=value,
            valid_range=(lo, hi),
            unit=unit,
            description=(
                f"参数 {term['display_name_cn']}({param_name}) = {value} {unit}, "
                f"物理范围 [{lo}, {hi}] {unit}"
            ) if term else f"参数 {param_name} = {value} {unit}, 物理范围 [{lo}, {hi}]",
            suggestion=suggestion,
        )

    def check_json_ir_params(self, json_ir: dict) -> List[ConstraintResult]:
        """
        遍历 JSON IR 中所有已赋值的参数，逐一检查物理范围
        """
        results = []
        aqua_type = json_ir.get("aquifer", {}).get("type", "all")

        def _unwrap(v):
            """安全解包：如果 v 是 {value: ..., unit: ...} 格式则取 value"""
            if isinstance(v, dict):
                return v.get("value")
            return v

        # 遍历 aquifer.parameters
        for pname, pinfo in json_ir.get("aquifer", {}).get("parameters", {}).items():
            v = _unwrap(pinfo)
            if v is not None:
                results.append(self.check_param(pname, v, aqua_type))

        # 遍历 transport.dispersion
        for pname, pinfo in json_ir.get("transport", {}).get("dispersion", {}).items():
            v = _unwrap(pinfo)
            if v is not None:
                results.append(self.check_param(pname, v, aqua_type))

        # 遍历 transport.reaction
        reaction = json_ir.get("transport", {}).get("reaction", {})
        for sub_key in ("decay",):
            sub = reaction.get(sub_key, {})
            for pname in ("first_order_rate", "half_life"):
                v = _unwrap(sub.get(pname))
                if v is not None:
                    results.append(self.check_param(pname, v, aqua_type))
            v = _unwrap(reaction.get("kd"))
            if v is not None:
                results.append(self.check_param("kd", v, aqua_type))

        # 检查网格参数
        grid = json_ir.get("grid", {})
        for pname in ("nrow", "ncol", "nlay"):
            v = _unwrap(grid.get(pname))
            if v is not None:
                results.append(self.check_param(pname, v, aqua_type))

        # 检查求解器参数
        solver = json_ir.get("solver", {}).get("parameters", {})
        for pname in ("hclose", "max_iter"):
            v = _unwrap(solver.get(pname))
            if v is not None:
                results.append(self.check_param(pname, v, aqua_type))

        # 检查时间离散参数
        #
        # 注意：只往 PHYSICAL_RANGES 加字典条目是**空操作** —— 本函数此前从不
        # 遍历 stress_periods，所以必须同时在这里补上遍历块，否则新加的
        # perlen/nstp/nper/total_time 边界永远不会被触发。
        #
        # 顶层与 transport 两处都看：East Texas 的形态正是顶层裸整数 10、
        # 真实时间信息在 transport.stress_periods。归一化用
        # `app.utils.ir_normalize.normalize_stress_periods`（唯一实现，
        # model_mapper._map_time_params 与 validator 三处共用）。
        for _scope, _stress in (
            ("stress_periods", normalize_stress_periods(json_ir.get("stress_periods"))),
            ("transport.stress_periods",
             normalize_stress_periods(
                 (json_ir.get("transport") or {}).get("stress_periods"))),
        ):
            if not isinstance(_stress, dict):
                continue
            for pname in ("perlen", "nstp", "nper", "total_time"):
                v = _unwrap(_stress.get(pname))
                if v is None:
                    continue
                res = self.check_param(pname, v, aqua_type)
                # 同名参数可能在两处出现，标上来源，否则用户看到"perlen 超范围"
                # 却不知道该改哪一个
                if not res.is_valid:
                    res.description = f"[{_scope}] {res.description}"
                results.append(res)

        return results


# ================================================================
# 约束类4：单位一致性校验
# ================================================================

class UnitConsistencyConstraint:
    """多维单位一致性校验"""

    # 标准单位体系
    STANDARD_UNITS = {
        "length": "m",
        "time": "d",
        "mass": "kg",
        "velocity": "m/d",
        "concentration": "mg/L",
        "volume_flux": "m3/d",
    }

    # 已知的正常单位（用于检测用户是否使用了非标准单位）
    KNOWN_UNITS = {
        "length": ["m", "ft", "cm", "km", "in", "mile"],
        "time": ["d", "s", "hr", "yr", "min"],
        "velocity": ["m/d", "m/s", "cm/s", "ft/d", "mm/yr"],
        "concentration": ["mg/L", "μg/L", "ng/L", "g/L", "ppm", "ppb"],
        "volume_flux": ["m3/d", "m3/s", "L/s", "ft3/d", "gpm"],
    }

    def check_unit_consistency(self, json_ir: dict) -> List[ConstraintResult]:
        """
        检查 JSON IR 中所有参数的单位是否一致

        按量纲分组检查：
        - 长度：domain.units, grid.delr, grid.delc, grid.top, grid.bottom
        - 时间：stress_periods.perlen, decay.half_life
        - 流速/渗透系数：aquifer.hk, aquifer.vka
        """
        results = []

        # 检查长度单位是否统一
        length_params = []
        domain_unit = json_ir.get("domain", {}).get("units", "m")
        for pname in ("delr", "delc", "top", "bottom"):
            info = json_ir.get("grid", {}).get(pname)
            if isinstance(info, dict) and info.get("unit"):
                if info["unit"] != domain_unit:
                    results.append(ConstraintResult(
                        constraint_type="unit_consistency",
                        param_name=f"grid.{pname}",
                        is_valid=False,
                        severity=ConstraintSeverity.WARNING,
                        current_value=info.get("value"),
                        unit=f"{info['unit']} (期望: {domain_unit})",
                        description=f"网格参数 grid.{pname} 单位({info['unit']})与模型范围单位({domain_unit})不一致",
                        suggestion=f"请将 grid.{pname} 的单位统一为 {domain_unit}",
                    ))

        # 检查时间单位
        time_unit = "d"  # 标准
        for bc in json_ir.get("boundary_conditions", []):
            params = bc.get("parameters", {})
            if isinstance(params.get("conductance"), dict):
                cu = params["conductance"].get("unit", "")
                if cu and "/d" not in cu and "/day" not in cu:
                    results.append(ConstraintResult(
                        constraint_type="unit_consistency",
                        param_name=f"boundary.{bc.get('name')}.conductance",
                        is_valid=False,
                        severity=ConstraintSeverity.WARNING,
                        current_value=params["conductance"].get("value"),
                        unit=cu,
                        description=f"河床传导系数单位{cu}可能不匹配标准时间单位(d)",
                        suggestion="传导系数应包含 /d 量纲",
                    ))

        # 检查渗透系数的量纲（hk/vka 必须是「长度/时间」，不能是裸长度）
        aquifer_params = json_ir.get("aquifer", {}).get("parameters", {})
        if not isinstance(aquifer_params, dict):
            aquifer_params = {}
        for pname in ("hk", "vka"):
            info = aquifer_params.get(pname)
            if not (isinstance(info, dict) and info.get("unit")):
                continue
            cu = str(info["unit"]).strip()
            if self._is_velocity_unit(cu):
                continue
            results.append(ConstraintResult(
                constraint_type="unit_consistency",
                param_name=f"aquifer.parameters.{pname}",
                is_valid=False,
                severity=ConstraintSeverity.WARNING,
                current_value=info.get("value"),
                unit=f"{cu} (期望: m/d)",
                description=(f"渗透系数 {pname} 的单位写作「{cu}」，"
                             f"但渗透系数是流速量纲（长度/时间），应为 m/d"),
                suggestion=("补上时间分母写成 m/d。写成裸长度单位（如 m）会让 "
                            "20 m 与 20 m/d 混为一谈，数值含义差一个时间维度"),
            ))

        return results

    @staticmethod
    def _is_velocity_unit(unit: str) -> bool:
        """判断单位是否为「长度/时间」量纲（m/d、ft/d、cm/s …）。

        只认带时间分母的写法。裸长度单位（m、ft、cm）与无量纲记号一律判否 ——
        前者正是 `vka = {"value": 20, "unit": "m"}` 这类漏写时间维度的情形。
        """
        u = str(unit).strip().lower().replace(" ", "").rstrip(".")
        if "/" not in u:
            return False
        denom = u.rsplit("/", 1)[-1]
        return denom in UnitConsistencyConstraint.TIME_UNIT_TOKENS

    # 可接受的时间分母（与 STANDARD_UNITS 的 time=d 兼容的各写法）
    TIME_UNIT_TOKENS = {
        "d", "day", "days",
        "s", "sec", "secs", "second", "seconds",
        "hr", "hrs", "h", "hour", "hours",
        "yr", "yrs", "year", "years",
        "min", "mins", "minute", "minutes",
    }

    def get_recommend_units(self, param_name: str) -> str:
        """获取参数推荐的标准单位"""
        for dim, unit in self.STANDARD_UNITS.items():
            candidates = self.KNOWN_UNITS.get(dim, [])
            if param_name in candidates:
                return unit
        return ""


# ================================================================
# 约束类5：数值求解稳定性判定
# ================================================================

class NumericalStabilityConstraint:
    """数值求解稳定性约束"""

    # MODFLOW 6 隐式格式相对宽松的 Courant 限制
    COURANT_IMPLICIT_LIMIT = 5.0
    # 网格 Peclet 数建议上限
    GRID_PECLET_WARNING = 10.0
    GRID_PECLET_ERROR = 100.0

    @staticmethod
    def check_courant_number(
        v_darcy: float, dt: float, dx: float, n: float = 0.25,
    ) -> ConstraintResult:
        """
        库朗数检查: Cr = v_pore · Δt / Δx

        物理意义：一个时间步内流体运移的网格数
        - Cr < 1: 显式稳定条件
        - Cr < 5~10: 隐式格式(MODFLOW)可接受
        - Cr > 10: 时间步过长，可能丢失瞬态特征
        """
        if n < 1e-10:
            n = 0.25
        v_pore = v_darcy / n if n > 0 else v_darcy

        if dx < 1e-10:
            return ConstraintResult(
                constraint_type="courant", param_name="courant_number",
                is_valid=True, severity=ConstraintSeverity.PASS,
                description="网格尺寸为0，跳过Courant检查",
            )

        Cr = abs(v_pore * dt / dx)
        is_valid = Cr < NumericalStabilityConstraint.COURANT_IMPLICIT_LIMIT

        severity = (
            ConstraintSeverity.ERROR if Cr > 20 else
            ConstraintSeverity.WARNING if Cr > NumericalStabilityConstraint.COURANT_IMPLICIT_LIMIT else
            ConstraintSeverity.PASS
        )

        return ConstraintResult(
            constraint_type="courant",
            param_name="courant_number",
            is_valid=is_valid,
            severity=severity,
            current_value=Cr,
            valid_range=(0.0, NumericalStabilityConstraint.COURANT_IMPLICIT_LIMIT),
            unit="dimensionless",
            description=f"Courant数 Cr = v·Δt/Δx = {Cr:.3f} "
                        f"(v_pore={v_pore:.4f}, Δt={dt}, Δx={dx})",
            suggestion=f"Courant数过大({Cr:.1f})，建议减小时间步长至≤{dx/v_pore*3:.1f}d 或加密网格"
            if not is_valid else "",
        )

    @staticmethod
    def check_time_step(
        dx: float, v_max: float, n: float = 0.25,
        safety_factor: float = 0.5,
    ) -> ConstraintResult:
        """
        推荐时间步长: Δt_rec ≤ SF · Δx / v_max

        其中 v_max 为最大达西流速
        safety_factor: 安全系数（0.3~0.7）
        """
        if v_max < 1e-10:
            return ConstraintResult(
                constraint_type="time_step",
                param_name="recommended_dt",
                is_valid=True,
                severity=ConstraintSeverity.PASS,
                description="流速为0（静水），任意时间步长均可",
            )

        dt_recommended = safety_factor * dx / (v_max / max(n, 0.01))
        return ConstraintResult(
            constraint_type="time_step",
            param_name="recommended_dt",
            is_valid=True,
            severity=ConstraintSeverity.INFO,
            current_value=dt_recommended,
            unit="d",
            description=f"推荐时间步长 ≤ {dt_recommended:.2f} d "
                        f"(基于 v_max={v_max:.4f} m/d, Δx={dx} m)",
            suggestion=f"建议设置各应力期时间步长 ≤ {dt_recommended:.2f} d",
        )

    @staticmethod
    def check_grid_quality(
        nrow, ncol, delr: Optional[np.ndarray] = None,
        delc: Optional[np.ndarray] = None,
    ) -> List[ConstraintResult]:
        """网格质量综合评估"""
        # 防御性解包
        if isinstance(nrow, dict): nrow = nrow.get("value")
        if isinstance(ncol, dict): ncol = ncol.get("value")
        nrow = int(nrow) if nrow is not None else 0
        ncol = int(ncol) if ncol is not None else 0
        results = []

        # 纵横比检查
        if delr is not None and delc is not None and len(delr) > 0 and len(delc) > 0:
            avg_delr = np.mean(delr)
            avg_delc = np.mean(delc)
            ratio = max(avg_delr, avg_delc) / max(min(avg_delr, avg_delc), 1e-10)
            if ratio > 5:
                results.append(ConstraintResult(
                    constraint_type="grid_quality",
                    param_name="aspect_ratio",
                    is_valid=False,
                    severity=ConstraintSeverity.WARNING,
                    current_value=ratio,
                    valid_range=(0.2, 5.0),
                    unit="dimensionless",
                    description=f"网格纵横比 = {ratio:.1f} (建议 0.2~5)",
                    suggestion="建议调整行列数使平均单元宽度与高度接近",
                ))

            # 相邻网格尺寸渐变检查
            if len(delr) > 1:
                max_ratio_r = max(
                    delr[i] / max(delr[i+1], 1e-10) for i in range(len(delr)-1)
                )
                max_ratio_r = max(max_ratio_r, 1/max(max_ratio_r, 1e-10))
                if max_ratio_r > 1.5:
                    results.append(ConstraintResult(
                        constraint_type="grid_quality",
                        param_name="size_gradation_row",
                        is_valid=False,
                        severity=ConstraintSeverity.WARNING,
                        current_value=max_ratio_r,
                        valid_range=(1.0, 1.5),
                        unit="dimensionless",
                        description=f"相邻行宽最大变化倍率 = {max_ratio_r:.2f}",
                        suggestion="相邻网格尺寸渐变不宜超过1.5倍",
                    ))

        # 单元总数检查
        total_cells = nrow * ncol
        if total_cells > 1000000:
            results.append(ConstraintResult(
                constraint_type="grid_quality",
                param_name="cell_count",
                is_valid=False,
                severity=ConstraintSeverity.WARNING,
                current_value=float(total_cells),
                valid_range=(1.0, 1000000.0),
                unit="cells",
                description=f"网格总数 {total_cells:,} > 1,000,000，可能计算量过大",
                suggestion="考虑减少行列数或使用非均匀网格局部加密",
            ))

        return results

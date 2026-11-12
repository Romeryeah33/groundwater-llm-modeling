"""
主控过程智能辨识算法

通过分析模拟输出的浓度时空场，自动判断：
- 对流主导 vs 弥散主导
- 降解贡献 vs 稀释贡献
- 吸附迟滞效应
"""

import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


class ProcessIdentifier:
    """主控过程智能辨识"""

    def identify_dominant_process(
        self,
        conc_history: list,
        velocity_field: list,
        parameters: dict,
        delr: np.ndarray = None,
        delc: np.ndarray = None,
    ) -> dict:
        """
        按时间/空间分解各过程对污染物质量变化的贡献

        各过程通量：
        1. 对流通量: J_adv = v · C
        2. 弥散通量: J_disp = -αL·v · ∂C/∂x
        3. 降解损失: R_decay = λ · C
        4. 吸附迟滞: R_ret = 1 + ρ·Kd/n (迟滞因子)

        Args:
            conc_history: 各时间步浓度场列表
            velocity_field: 各时间步速度场列表
            parameters: {
                al: 弥散度, decay_rate: 降解速率,
                kd: 分配系数, porosity: 孔隙度,
                rho_bulk: 容重 (默认1.7 g/cm³)
            }
            delr/delc: 网格尺寸

        Returns:
            {advection_pct, dispersion_pct, decay_pct, sorption_factor, dominant_process, ...}
        """
        if len(conc_history) < 2:
            return {"dominant_process": "insufficient_data"}

        n_times = len(conc_history)

        # 提取参数
        al = parameters.get("al", 10.0)
        decay_rate = parameters.get("decay_rate", 0.0)
        kd = parameters.get("kd", 0.0)
        porosity = parameters.get("porosity", 0.25)
        rho_bulk = parameters.get("rho_bulk", 1.7)

        # 计算平均通量
        adv_fluxes = []
        disp_fluxes = []
        decay_losses = []

        for t in range(min(n_times - 1, len(velocity_field))):
            conc = conc_history[t]
            vel = velocity_field[t]

            # 对流通量: v * C (单位时间内运移的质量)
            if isinstance(vel, (list, np.ndarray)):
                v_mag = np.linalg.norm(vel, axis=0) if vel.ndim > 2 else np.abs(vel)
            else:
                v_mag = np.abs(vel)

            adv_flux = np.mean(v_mag * conc)
            adv_fluxes.append(float(adv_flux))

            # 弥散通量: αL * v * |∂C/∂x|
            if delr is not None:
                dx = np.mean(delr)
            else:
                dx = 20.0  # 默认
            dc_dx = np.gradient(conc, dx, axis=1)
            disp_mag = np.mean(np.abs(al * v_mag * dc_dx))
            disp_fluxes.append(float(disp_mag))

            # 降解损失: λ * C
            if decay_rate > 0:
                decay_loss = np.mean(decay_rate * conc)
                decay_losses.append(float(decay_loss))

        # 计算各过程相对贡献
        avg_adv = np.mean(adv_fluxes) if adv_fluxes else 0.0
        avg_disp = np.mean(disp_fluxes) if disp_fluxes else 0.0
        avg_decay = np.mean(decay_losses) if decay_losses else 0.0

        total_flux = avg_adv + avg_disp + avg_decay

        if total_flux > 1e-20:
            adv_pct = avg_adv / total_flux * 100
            disp_pct = avg_disp / total_flux * 100
            decay_pct = avg_decay / total_flux * 100
        else:
            adv_pct = disp_pct = decay_pct = 0.0

        # 迟滞因子
        retardation = 1.0 + (rho_bulk * kd / max(porosity, 0.01))

        # 判断主控过程
        # Peclet 数 (对流 vs 弥散)
        if avg_disp > 1e-20:
            Pe = avg_adv / avg_disp
        else:
            Pe = 1e10

        # Damkohler 数 (降解 vs 对流)
        if avg_adv > 1e-20 and decay_rate > 0:
            Da = avg_decay / avg_adv
        else:
            Da = 0.0

        # 判定
        if Pe > 10:
            transport_regime = "advection_dominated"
        elif Pe < 0.1:
            transport_regime = "dispersion_dominated"
        else:
            transport_regime = "mixed_transport"

        if Da > 1:
            reaction_regime = "reaction_limited"
        elif Da < 0.01:
            reaction_regime = "transport_limited"
        else:
            reaction_regime = "mixed_kinetics"

        # 生成文字解释
        explanation = self._generate_explanation(
            transport_regime, reaction_regime, retardation,
            Pe, Da, adv_pct, disp_pct, decay_pct
        )

        return {
            "contributions": {
                "advection": round(adv_pct, 1),
                "dispersion": round(disp_pct, 1),
                "decay": round(decay_pct, 1),
            },
            "peclet_number": round(Pe, 1),
            "damkohler_number": round(Da, 5),
            "retardation_factor": round(retardation, 1),
            "transport_regime": transport_regime,
            "reaction_regime": reaction_regime,
            "dominant_process": self._dominant_label(
                transport_regime, reaction_regime, retardation
            ),
            "explanation": explanation,
        }

    def _dominant_label(
        self, transport_regime: str, reaction_regime: str, R: float
    ) -> str:
        """生成主控过程标签"""
        parts = []
        if transport_regime == "advection_dominated":
            parts.append("advection")
        elif transport_regime == "dispersion_dominated":
            parts.append("dispersion")

        if reaction_regime == "reaction_limited":
            parts.append("degradation")
        elif reaction_regime == "mixed_kinetics":
            parts.append("mixed")

        if R > 1.5:
            parts.append("sorption_retarded")

        return "+".join(parts) if parts else "mixed"

    def _generate_explanation(
        self, t_regime: str, r_regime: str, R: float,
        Pe: float, Da: float, adv: float, disp: float, decay: float,
    ) -> str:
        """生成中文解释文字"""
        lines = []

        # 运移机制
        if t_regime == "advection_dominated":
            lines.append(f"运移以对流为主导 (Pe={Pe:.1f}>{'>'}10)，"
                        f"贡献率 {adv:.0f}%。")
        elif t_regime == "dispersion_dominated":
            lines.append(f"运移以弥散为主导 (Pe={Pe:.2f}<0.1)，"
                        f"贡献率 {disp:.0f}%。")
        else:
            lines.append(f"对流({adv:.0f}%)与弥散({disp:.0f}%)混合作用 (Pe={Pe:.1f})。")

        # 降解
        if decay > 1:
            lines.append(f"降解作用显著 (Da={Da:.3f}>1)，贡献率 {decay:.0f}%。")
        elif decay > 0.1:
            lines.append(f"降解有一定影响 (Da={Da:.3f})，贡献率 {decay:.0f}%。")
        else:
            lines.append("降解作用可忽略。")

        # 吸附
        if R > 1.5:
            lines.append(f"吸附迟滞因子 R={R:.1f}，污染物迁移受到显著迟滞。")

        return " ".join(lines)


# 全局实例
process_identifier = ProcessIdentifier()

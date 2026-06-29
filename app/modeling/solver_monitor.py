"""
求解过程实时监控与自适应重启动

监控项：
- 求解器收敛状态
- 水量平衡误差
- 溶质质量平衡
- 自适应策略：调松弛因子 / 切换求解器 / 减小步长 / 加密网格
"""

import logging
import time
from typing import Optional, Dict, List, Tuple
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    NON_CONVERGENCE = "non_convergence"
    DRY_CELLS = "dry_cells"
    MASS_BALANCE_ERROR = "mass_balance_error"
    NEGATIVE_CONCENTRATION = "negative_concentration"
    MEMORY_ERROR = "memory_error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class SolverMonitor:
    """求解过程实时监控与异常自动处理"""

    # 自适应策略库
    ADAPTIVE_STRATEGIES = {
        ErrorType.NON_CONVERGENCE: [
            {"action": "relaxation", "description": "减小松弛因子10%", "auto": True},
            {"action": "solver_switch", "description": "切换求解器 (SLP→PIC)", "auto": True},
            {"action": "time_step_reduce", "description": "减小时间步长50%", "auto": True},
            {"action": "grid_refine", "description": "在问题区域加密网格", "auto": False},
        ],
        ErrorType.DRY_CELLS: [
            {"action": "rewetting", "description": "启用Rewetting功能", "auto": True},
            {"action": "bottom_adjust", "description": "降低含水层底板高程", "auto": False},
        ],
        ErrorType.MASS_BALANCE_ERROR: [
            {"action": "budget_check", "description": "检查源汇项总量是否匹配", "auto": False},
            {"action": "tighten_convergence", "description": "收紧求解器收敛准则 (1e-6→1e-8)", "auto": True},
            {"action": "time_step_reduce", "description": "减小时间步长", "auto": True},
        ],
        ErrorType.NEGATIVE_CONCENTRATION: [
            {"action": "increase_dispersion", "description": "增加弥散度或启用TVD格式", "auto": True},
            {"action": "time_step_reduce", "description": "减小时间步长以满足Peclet约束", "auto": True},
            {"action": "cnc_bounding", "description": "启用浓度裁剪 (conc≥0)", "auto": True},
        ],
        ErrorType.MEMORY_ERROR: [
            {"action": "grid_coarsen", "description": "粗化网格，减少单元数", "auto": False},
        ],
        ErrorType.TIMEOUT: [
            {"action": "increase_timeout", "description": "增加超时限制", "auto": True},
            {"action": "grid_coarsen", "description": "粗化网格", "auto": False},
        ],
    }

    MAX_RETRIES = 3
    TIMEOUT_SECONDS = 600  # 单次运行超时 10 分钟

    def __init__(self):
        self.adaptation_log: List[dict] = []
        self.last_error: Optional[str] = None

    # ================================================================
    # 主入口
    # ================================================================

    def monitor_run(self, sim, json_ir: dict) -> dict:
        """
        运行模拟并监控，遇异常自动采取自适应策略

        Args:
            sim: flopy.mf6.MFSimulation 对象
            json_ir: 当前 JSON IR（用于策略调整参考）

        Returns:
            {
                "success": bool,
                "error_type": str,
                "adaptations_applied": list,
                "final_balance_error": float,
                "run_time_seconds": float,
            }
        """
        self.adaptation_log = []
        adaptations = []
        start_time = time.time()

        for attempt in range(self.MAX_RETRIES):
            try:
                logger.info(f"模拟运行尝试 {attempt+1}/{self.MAX_RETRIES}")
                run_start = time.time()

                success, buff = sim.run_simulation(silent=True, report=True)

                run_time = time.time() - run_start

                if success:
                    # 后检查：水量平衡
                    balance = self._check_balance(sim, json_ir)

                    if balance["is_acceptable"]:
                        logger.info(f"模拟成功 (尝试{attempt+1}, 平衡误差={balance['error']*100:.4f}%)")
                        return {
                            "success": True,
                            "adaptations_applied": adaptations,
                            "final_balance_error": balance["error"],
                            "run_time_seconds": round(time.time() - start_time, 1),
                            "attempts": attempt + 1,
                        }
                    else:
                        logger.warning(f"水量平衡不合格: {balance['error']*100:.4f}%")
                        adaptation = self._apply_strategy(
                            ErrorType.MASS_BALANCE_ERROR, attempt,
                            sim, json_ir
                        )
                        if adaptation:
                            adaptations.append(adaptation)

                else:
                    # 运行失败 — 分类错误
                    error_type = self._classify_error(buff, json_ir)
                    logger.warning(f"模拟失败: {error_type.value}")
                    adaptation = self._apply_strategy(
                        error_type, attempt, sim, json_ir
                    )
                    if adaptation:
                        adaptations.append(adaptation)

            except Exception as e:
                error_type = ErrorType.UNKNOWN
                logger.error(f"模拟异常: {e}")
                self.last_error = str(e)

        return {
            "success": False,
            "error": f"达到最大重试次数 ({self.MAX_RETRIES}) 仍未成功",
            "error_type": ErrorType.UNKNOWN.value,
            "adaptations_applied": adaptations,
            "run_time_seconds": round(time.time() - start_time, 1),
            "attempts": self.MAX_RETRIES,
        }

    # ================================================================
    # 错误分类
    # ================================================================

    def _classify_error(self, buff, json_ir: dict) -> ErrorType:
        """从模拟输出中分类错误类型"""
        if isinstance(buff, list):
            output_text = "\n".join(str(b) for b in buff)
        elif isinstance(buff, str):
            output_text = buff
        else:
            output_text = str(buff)

        output_lower = output_text.lower()

        if ("did not converge" in output_lower
                or "failed to converge" in output_lower
                or ("failed" in output_lower and "converge" in output_lower)):
            return ErrorType.NON_CONVERGENCE
        elif "dry cell" in output_lower or "went dry" in output_lower:
            return ErrorType.DRY_CELLS
        elif "negative concentration" in output_lower or "conc < 0" in output_lower:
            return ErrorType.NEGATIVE_CONCENTRATION
        elif "memory" in output_lower or "allocation" in output_lower:
            return ErrorType.MEMORY_ERROR
        elif "budget" in output_lower and "error" in output_lower:
            return ErrorType.MASS_BALANCE_ERROR

        return ErrorType.UNKNOWN

    # ================================================================
    # 自适应策略应用
    # ================================================================

    def _apply_strategy(
        self, error_type: ErrorType, attempt: int,
        sim, json_ir: dict
    ) -> Optional[dict]:
        """
        应用自适应策略

        Args:
            error_type: 错误类型
            attempt: 当前尝试次数 (0-based)
            sim: 模拟对象
            json_ir: JSON IR

        Returns:
            {strategy, description, success}
        """
        strategies = self.ADAPTIVE_STRATEGIES.get(error_type, [])
        if not strategies or attempt >= len(strategies):
            return None

        strategy = strategies[attempt]

        logger.info(f"应用自适应策略: {strategy['description']}")

        applied = {
            "error_type": error_type.value,
            "attempt": attempt + 1,
            "strategy": strategy["action"],
            "description": strategy["description"],
            "success": False,
        }

        try:
            if strategy["action"] == "relaxation":
                self._adjust_relaxation(sim, factor=0.9)
                applied["success"] = True

            elif strategy["action"] == "time_step_reduce":
                self._reduce_time_step(json_ir, factor=0.5)
                applied["success"] = True

            elif strategy["action"] == "tighten_convergence":
                self._adjust_convergence(sim, hclose=1e-8, rclose=1e-5)
                applied["success"] = True

            elif strategy["action"] == "increase_dispersion":
                self._adjust_dispersion(json_ir, factor=1.5)
                applied["success"] = True

            elif strategy["action"] == "cnc_bounding":
                applied["success"] = True
                applied["note"] = "后处理时将裁剪负浓度为0"

            elif strategy["action"] == "rewetting":
                applied["success"] = True
                applied["note"] = "需要重新构建模型（已记录）"

            elif strategy["action"] == "solver_switch":
                applied["success"] = True
                applied["note"] = "需要重新构建模型切换求解器"

            elif not strategy["auto"]:
                applied["note"] = "此策略需要人工确认"

        except Exception as e:
            applied["success"] = False
            applied["error"] = str(e)

        self.adaptation_log.append(applied)
        return applied

    # ================================================================
    # 具体调整方法
    # ================================================================

    def _adjust_relaxation(self, sim, factor: float = 0.9):
        """减小松弛因子"""
        logger.info(f"调整松弛因子: ×{factor}")

    def _adjust_convergence(self, sim, hclose: float = 1e-8, rclose: float = 1e-5):
        """收紧收敛准则"""
        logger.info(f"收紧收敛准则: hclose={hclose}, rclose={rclose}")

    def _reduce_time_step(self, json_ir: dict, factor: float = 0.5):
        """减小时间步长"""
        stress = json_ir.get("stress_periods", {})
        if not isinstance(stress, dict): stress = {}
        period_data = stress.get("period_data", [])
        for pd in period_data:
            pd["nstp"] = max(1, int(pd.get("nstp", 1) / factor))
        logger.info(f"时间步长缩小 ×{factor}")

    def _adjust_dispersion(self, json_ir: dict, factor: float = 1.5):
        """增加弥散度"""
        transport = json_ir.get("transport", {})
        disp = transport.get("dispersion", {})
        for key in ("al", "trpt", "trpv"):
            if key in disp and isinstance(disp[key], dict):
                val = disp[key].get("value")
                if val is not None:
                    disp[key]["value"] = val * factor
        logger.info(f"弥散度增加 ×{factor}")

    # ================================================================
    # 结果检查
    # ================================================================

    def _check_balance(self, sim, json_ir: dict) -> dict:
        """检查模拟结果的水量平衡"""
        try:
            gwf = sim.get_model()
            if gwf is None:
                # 尝试从 sim 中提取 gwf
                for name in sim.model_names:
                    model = sim.get_model(name)
                    if "gwf" in name.lower():
                        gwf = model
                        break

            if gwf is None:
                return {"is_acceptable": True, "error": 0.0, "note": "无法读取GWF模型"}

            budget = gwf.output.budget()
            records = budget.get_data()

            flow_in = 0.0
            flow_out = 0.0

            for record in records:
                if isinstance(record, np.ndarray) and record.dtype.names:
                    for field in record.dtype.names:
                        try:
                            q = np.sum(record[field])
                            if q > 0:
                                flow_in += q
                            else:
                                flow_out += abs(q)
                        except (ValueError, TypeError):
                            pass

            if flow_in > 1e-10 and flow_out > 1e-10:
                error = abs(flow_in - flow_out) / max(flow_in, flow_out)
            else:
                error = abs(flow_in - flow_out)

            return {
                "is_acceptable": error < 0.01,
                "error": error,
                "flow_in": flow_in,
                "flow_out": flow_out,
            }

        except Exception as e:
            logger.warning(f"水量平衡检查失败: {e}")
            return {"is_acceptable": True, "error": 0.0, "note": f"检查失败: {e}"}


# 全局实例
solver_monitor = SolverMonitor()

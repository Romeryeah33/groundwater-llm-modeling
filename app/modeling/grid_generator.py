"""
自动网格生成器

支持策略：
- 均匀网格 (uniform)
- 非均匀加密 (refined): 井/污染源附近局部加密
- 基于 Peclet 数的自适应网格尺寸推荐
"""

import numpy as np
import logging
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger(__name__)


class AutoGridGenerator:
    """自动时空离散与网格剖分"""

    # 默认网格参数
    DEFAULT_NROW = 50
    DEFAULT_NCOL = 50
    DEFAULT_NLAY = 1
    DEFAULT_CELL_SIZE = 20.0  # m

    def generate(
        self,
        domain: dict,
        nrow: Optional[int] = None,
        ncol: Optional[int] = None,
        nlay: Optional[int] = None,
        refinement_zones: Optional[List[dict]] = None,
        target_cell_size: Optional[float] = None,
    ) -> dict:
        """
        根据模型范围与精度需求自动生成网格方案

        Args:
            domain: 模型范围 {"x_range": [xmin, xmax], "y_range": [ymin, ymax]}
            nrow/ncol/nlay: 用户指定的行列层数（优先）
            refinement_zones: 加密区域
            target_cell_size: 目标网格尺寸 (m)

        Returns:
            {nrow, ncol, nlay, delr, delc, top, botm, cell_area_stats, ...}
        """
        xr = domain.get("x_range", [0, 1000])
        yr = domain.get("y_range", [0, 1000])
        Lx = xr[1] - xr[0]
        Ly = yr[1] - yr[0]

        # 确定行列数
        if nrow and ncol:
            pass  # 用户指定优先
        elif target_cell_size:
            ncol = max(1, int(Lx / target_cell_size))
            nrow = max(1, int(Ly / target_cell_size))
        else:
            # 基于面积自动推荐
            area = Lx * Ly
            if area < 1e5:      # < 0.1 km²
                ncol, nrow = 100, 80
            elif area < 1e6:    # < 1 km²
                ncol, nrow = 80, 60
            elif area < 1e7:    # < 10 km²
                ncol, nrow = 60, 50
            else:
                ncol, nrow = 50, 40

        nlay = nlay or self.DEFAULT_NLAY

        # 生成 delr/delc（均匀或非均匀）
        if refinement_zones:
            delr = self._generate_refined_array(
                Lx, ncol, refinement_zones, axis="x"
            )
            delc = self._generate_refined_array(
                Ly, nrow, refinement_zones, axis="y"
            )
        else:
            delr = np.full(ncol, Lx / ncol)
            delc = np.full(nrow, Ly / nrow)

        # 计算顶底板（如果有 dem 或简单平面）
        top = domain.get("top", 100.0)
        botm = domain.get("bottom", top - 50.0)

        result = {
            "nrow": nrow,
            "ncol": ncol,
            "nlay": nlay,
            "delr": delr.tolist() if isinstance(delr, np.ndarray) else delr,
            "delc": delc.tolist() if isinstance(delc, np.ndarray) else delc,
            "top": top,
            "botm": botm,
            "cell_count": nrow * ncol * nlay,
            "avg_cell_size": round(Lx / ncol, 1),
            "refinement_applied": refinement_zones is not None,
        }

        # 单元面积统计
        if isinstance(delr, np.ndarray) and isinstance(delc, np.ndarray):
            cell_areas = np.outer(delc, delr)
            result["cell_area_stats"] = {
                "min": round(float(cell_areas.min()), 2),
                "max": round(float(cell_areas.max()), 2),
                "mean": round(float(cell_areas.mean()), 2),
                "std": round(float(cell_areas.std()), 2),
            }

        logger.info(
            f"网格生成: {nrow}×{ncol}×{nlay} "
            f"(共{nrow*ncol*nlay:,}单元, 平均尺寸{result['avg_cell_size']}m)"
        )
        return result

    def _generate_refined_array(
        self, total_length: float, n_cells: int,
        refinement_zones: List[dict], axis: str = "x",
    ) -> np.ndarray:
        """生成非均匀加密网格"""
        # 基础均匀网格
        base_d = total_length / n_cells

        # 为每个加密区域计算乘子
        multipliers = np.ones(n_cells)
        cell_centers = np.arange(n_cells) * base_d + base_d / 2

        for zone in refinement_zones:
            coords = zone.get("coordinates", [])
            factor = zone.get("refinement_factor", 2.0)
            zone_type = zone.get("type", "point")

            for coord in coords:
                if zone_type == "point":
                    # 计算各单元到加密点的距离（沿该轴）
                    if axis == "x":
                        distances = np.abs(cell_centers - coord[0])
                    else:
                        distances = np.abs(cell_centers - coord[1])

                    # 距离越近，乘子越大（单元越小）
                    sigma = total_length / n_cells * 5  # 影响半径 = 5个基础单元
                    weights = np.exp(-distances**2 / (2 * sigma**2))
                    multipliers = np.maximum(multipliers, 1 + (factor - 1) * weights)

        # 根据乘子重新分配单元尺寸
        # 乘子大的区域获得更多单元（单元更小）
        raw_weights = 1.0 / multipliers
        normalized = raw_weights / raw_weights.sum()
        cell_sizes = normalized * total_length

        # 确保为正
        cell_sizes = np.maximum(cell_sizes, 0.1)

        return cell_sizes

    def suggest_from_peclet(
        self, al: float, v: float, domain: dict,
    ) -> Dict[str, any]:
        """
        基于 Peclet 数推荐网格尺寸

        推荐准测: Δx ≤ αL * 4（避免数值弥散）
                    Δx ≥ αL * 0.5（避免不必要的精细网格）

        Args:
            al: 纵向弥散度 (m)
            v: 达西流速 (m/d) — 暂未使用
            domain: 模型范围
        """
        if al < 1e-10:
            return {"recommended_dx": 20.0, "rationale": "弥散度极小，按对流主导处理"}

        dx_ideal = al * 2.0  # 理想Δx
        dx_min = al * 0.25   # 最小Δx（浪费计算资源）
        dx_max = al * 5.0    # 最大Δx（数值弥散严重）

        xr = domain.get("x_range", [0, 1000])
        Lx = xr[1] - xr[0]
        ncol_suggested = int(Lx / dx_ideal)

        # 限制在合理范围
        ncol_suggested = max(20, min(500, ncol_suggested))

        return {
            "recommended_dx": round(dx_ideal, 1),
            "dx_min": round(dx_min, 1),
            "dx_max": round(dx_max, 1),
            "suggested_ncol": ncol_suggested,
            "rationale": f"基于 αL={al}m: Δx_ideal={dx_ideal}m, 网格Peclet≈2 (理想范围)",
        }

    def check_grid_quality(self, grid_params: dict, parameters: dict = None) -> dict:
        """网格质量综合评估"""
        nrow = grid_params.get("nrow", 50)
        ncol = grid_params.get("ncol", 50)
        delr = grid_params.get("delr", [])
        delc = grid_params.get("delc", [])

        issues = []
        warnings = []
        score = 100  # 满分100

        # 1. 纵横比检查
        if isinstance(delr, (list, np.ndarray)) and isinstance(delc, (list, np.ndarray)):
            delr = np.asarray(delr)
            delc = np.asarray(delc)
            if len(delr) > 0 and len(delc) > 0:
                avg_dr = np.mean(delr)
                avg_dc = np.mean(delc)
                ratio = max(avg_dr, avg_dc) / max(min(avg_dr, avg_dc), 1e-10)
                if ratio > 5:
                    issues.append(f"纵横比 {ratio:.1f} > 5")
                    score -= 15
                elif ratio > 3:
                    warnings.append(f"纵横比 {ratio:.1f} > 3")

        # 2. 单元总数
        total = nrow * ncol
        if total > 1_000_000:
            warnings.append(f"单元总数 {total:,} > 1M")
            score -= 20
        elif total > 500_000:
            warnings.append(f"单元总数 {total:,} > 500k")
            score -= 10

        # 3. 最小单元尺寸
        if len(delr) > 0:
            min_dr = np.min(np.asarray(delr))
            if min_dr < 0.5:
                warnings.append(f"最小行宽 {min_dr:.2f}m < 0.5m（过小）")
                score -= 10

        return {
            "score": max(score, 0),
            "rating": "good" if score >= 80 else ("fair" if score >= 60 else "poor"),
            "issues": issues,
            "warnings": warnings,
        }


# 全局实例
grid_generator = AutoGridGenerator()

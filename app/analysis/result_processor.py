"""
模拟结果自动提取与后处理

从 FloPy/MODFLOW 6 输出中提取：
- 水头场（按层/时间步）
- 浓度场（按物种/时间步）
- 水量平衡预算表
- 污染羽核心指标（面积、质心、迁移速率）
"""

import logging
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ResultProcessor:
    """模拟结果自动提取与后处理"""

    def __init__(self):
        pass

    # ================================================================
    # 水头提取
    # ================================================================

    def extract_head_field(self, gwf, layer: int = 0, totim: float = None) -> np.ndarray:
        """提取指定层/时刻的水头场"""
        head_obj = gwf.output.head()
        if totim is not None:
            return head_obj.get_data(totim=totim)
        return head_obj.get_data()[layer] if head_obj.get_data().ndim == 3 else head_obj.get_data()

    def extract_head_timeseries(self, gwf, obs_points: List[tuple]) -> pd.DataFrame:
        """
        提取观测点的水头时间序列

        Args:
            gwf: ModflowGwf 对象
            obs_points: [(layer, row, col), ...]

        Returns:
            DataFrame: 时间 × 观测点
        """
        head_obj = gwf.output.head()
        times = head_obj.get_times()
        data = {}

        for i, (lay, row, col) in enumerate(obs_points):
            col_name = f"obs_{i+1}_L{lay}_R{row}_C{col}"
            series = []
            for t in times:
                h = head_obj.get_data(totim=t)
                series.append(float(h[lay, row, col]))
            data[col_name] = series

        df = pd.DataFrame(data, index=times)
        df.index.name = "time_days"
        return df

    # ================================================================
    # 浓度提取
    # ================================================================

    def extract_concentration_field(
        self, gwt, layer: int = 0, totim: float = None
    ) -> np.ndarray:
        """提取指定层/时刻的浓度场"""
        conc_obj = gwt.output.concentration()
        if totim is not None:
            data = conc_obj.get_data(totim=totim)
        else:
            data = conc_obj.get_data()
        if data.ndim >= 3:
            return data[layer]
        return data

    def extract_concentration_timeseries(
        self, gwt, obs_points: List[tuple]
    ) -> pd.DataFrame:
        """
        提取观测点浓度穿透曲线

        Returns:
            DataFrame: 时间 × 观测点
        """
        conc_obj = gwt.output.concentration()
        times = conc_obj.get_times()
        data = {}

        for i, (lay, row, col) in enumerate(obs_points):
            col_name = f"obs_{i+1}_L{lay}_R{row}_C{col}"
            series = []
            for t in times:
                c = conc_obj.get_data(totim=t)
                series.append(float(c[lay, row, col]))
            data[col_name] = series

        df = pd.DataFrame(data, index=times)
        df.index.name = "time_days"
        return df

    # ================================================================
    # 水量/质量平衡提取
    # ================================================================

    def extract_water_budget(self, gwf) -> dict:
        """
        提取水量平衡预算数据

        Returns:
            {inflow: {component: value}, outflow: {component: value}, balance_error: float}
        """
        try:
            budget_obj = gwf.output.budget()

            # FloPy 3.8+: get_data() 需要 kstpkper/totim/idx/text 参数
            try:
                kstpkper_list = budget_obj.get_kstpkper()
            except AttributeError:
                kstpkper_list = [(0, 0)]

            inflow = {}
            outflow = {}
            total_in = 0.0
            total_out = 0.0

            for kpk in kstpkper_list:
                try:
                    records = budget_obj.get_data(kstpkper=kpk)
                except Exception:
                    # 回退: 尝试用 idx
                    try:
                        records = budget_obj.get_data(idx=0)
                    except Exception:
                        # 最后回退: 旧版 FloPy 无参数调用
                        records = budget_obj.get_data()

                if isinstance(records, np.ndarray):
                    self._process_budget_records(records, inflow, outflow)
                    total_in += sum(inflow.values())
                    total_out += sum(outflow.values())
                elif isinstance(records, list):
                    for rec in records:
                        if isinstance(rec, np.ndarray):
                            self._process_budget_records(rec, inflow, outflow)
                            total_in += sum(inflow.values())
                            total_out += sum(outflow.values())

            error = abs(total_in - total_out) / max(total_in, total_out, 1e-10) if total_in > 0 else 0.0

            return {
                "inflow": inflow,
                "outflow": outflow,
                "total_in": total_in,
                "total_out": total_out,
                "balance_error": error,
            }
        except Exception as e:
            logger.warning(f"水量平衡提取失败: {e}")
            return {"error": str(e)}

    @staticmethod
    def _process_budget_records(records: np.ndarray, inflow: dict, outflow: dict):
        """处理单条预算记录，将流入/流出累加到对应字典"""
        if not (isinstance(records, np.ndarray) and records.dtype.names):
            return
        for field in records.dtype.names:
            q = np.sum(records[field])
            component = field.replace("_", " ").title()
            if q > 0:
                inflow[component] = inflow.get(component, 0.0) + float(q)
            elif q < 0:
                outflow[component] = outflow.get(component, 0.0) + float(abs(q))

    # ================================================================
    # 污染羽指标计算
    # ================================================================

    def compute_plume_metrics(
        self,
        concentration_history: List[np.ndarray],
        times: List[float],
        threshold: float = 1.0,
        cell_area: float = 400.0,  # m²/单元
        delr: np.ndarray = None,
        delc: np.ndarray = None,
    ) -> Dict[str, any]:
        """
        计算污染羽核心指标

        Args:
            concentration_history: 各时间步的浓度场列表
            times: 对应的时刻列表
            threshold: 污染羽判定阈值（如饮用水标准）
            cell_area: 单元面积（均匀网格）或
            delr/delc: 网格尺寸数组（非均匀网格）

        Returns:
            {
                max_conc_vs_time,
                plume_area_vs_time,
                plume_centroid_vs_time,
                total_mass_vs_time,
                migration_velocity,
                ...
            }
        """
        n_times = len(times)
        max_conc = np.zeros(n_times)
        plume_area = np.zeros(n_times)
        centroids_x = np.zeros(n_times)
        centroids_y = np.zeros(n_times)
        total_mass = np.zeros(n_times)

        # 计算单元面积数组
        if delr is not None and delc is not None:
            cell_areas = np.outer(np.asarray(delc), np.asarray(delr))
        else:
            cell_areas = np.full_like(concentration_history[0], cell_area)

        for i, conc in enumerate(concentration_history):
            max_conc[i] = float(np.max(conc))

            # 污染羽面积（超过阈值的区域）
            plume_mask = conc > threshold
            plume_area[i] = float(np.sum(cell_areas[plume_mask]))

            # 质心位置（浓度加权）
            if np.sum(conc[plume_mask]) > 0:
                rows, cols = np.meshgrid(
                    np.arange(conc.shape[0]),
                    np.arange(conc.shape[1]),
                    indexing="ij",
                )
                y_coords = np.sum(rows * cell_areas, axis=1).cumsum() - np.sum(cell_areas, axis=1) / 2
                x_coords = np.sum(cell_areas, axis=0).cumsum() - np.sum(cell_areas, axis=0) / 2

                weighted_conc = conc * cell_areas
                if np.sum(weighted_conc) > 0:
                    centroids_x[i] = float(np.sum(weighted_conc * cols.astype(float)) / np.sum(weighted_conc))
                    centroids_y[i] = float(np.sum(weighted_conc * rows.astype(float)) / np.sum(weighted_conc))

            # 总质量（浓度 × 体积，假设单位厚度 × 孔隙度）
            total_mass[i] = float(np.sum(conc * cell_areas))

        # 迁移速率（质心位移 / 时间）
        if n_times > 1:
            dt_total = times[-1] - times[0]
            if dt_total > 0:
                dx_total = centroids_x[-1] - centroids_x[0]
                dy_total = centroids_y[-1] - centroids_y[0]
                migration_v = np.sqrt(dx_total**2 + dy_total**2) / dt_total
            else:
                migration_v = 0.0
        else:
            migration_v = 0.0

        # 质量去除率
        mass_initial = total_mass[0]
        mass_final = total_mass[-1]
        removal_rate = (mass_initial - mass_final) / max(mass_initial, 1e-10) if mass_initial > 0 else 0.0

        return {
            "times": times,
            "max_conc": max_conc.tolist(),
            "plume_area_m2": plume_area.tolist(),
            "centroid_x": centroids_x.tolist(),
            "centroid_y": centroids_y.tolist(),
            "total_mass": total_mass.tolist(),
            "migration_velocity_m_d": float(migration_v),
            "mass_removal_rate": float(removal_rate),
            "threshold": threshold,
        }

    # ================================================================
    # 汇总报告
    # ================================================================

    def generate_summary(self, gwf, gwt=None, json_ir: dict = None) -> dict:
        """生成模拟结果汇总"""
        summary = {}

        # 水头
        try:
            head = self.extract_head_field(gwf, layer=0)
            summary["head_min"] = float(np.min(head))
            summary["head_max"] = float(np.max(head))
            summary["head_mean"] = float(np.mean(head))
        except Exception:
            pass

        # 水量平衡
        budget = self.extract_water_budget(gwf)
        summary["water_balance_error"] = budget.get("balance_error")
        summary["total_flow_in"] = budget.get("total_in")
        summary["total_flow_out"] = budget.get("total_out")

        # 浓度（如有GWT）
        if gwt is not None:
            try:
                conc = self.extract_concentration_field(gwt, layer=0)
                summary["conc_max"] = float(np.max(conc))
                summary["conc_min"] = float(np.min(conc))
            except Exception:
                pass

        return summary


# 全局实例
result_processor = ResultProcessor()

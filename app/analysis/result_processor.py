"""
模拟结果自动提取与后处理

从 FloPy/MODFLOW 6 输出中提取：
- 水头场（按层/时间步）
- 浓度场（按物种/时间步）
- 流速场（供主控过程辨识）
- 水量平衡预算表
"""

import re
import logging
from pathlib import Path
from typing import Optional

import numpy as np

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

    # ================================================================
    # 浓度提取
    # ================================================================

    def extract_concentration_field(
        self, gwt, layer: int = 0, totim: float = None
    ) -> np.ndarray:
        """提取指定层/时刻的浓度场

        Args:
            gwt: ModflowGwt 模型对象
            layer: 目标层索引 (0-based)
            totim: 指定时刻。为 None 时取最后一个时间步的结果。

        Returns:
            2D 浓度场 (nrow, ncol)
        """
        conc_obj = gwt.output.concentration()
        if totim is not None:
            data = conc_obj.get_data(totim=totim)
        else:
            # 取最后一个时间步
            times = conc_obj.get_times()
            if times is not None and len(times) > 0:
                data = conc_obj.get_data(totim=times[-1])
            else:
                data = conc_obj.get_data()

        # 处理多维数据: (nlay, nrow, ncol) 或 (ntimes, nlay, nrow, ncol)
        if data.ndim == 4:
            data = data[-1]  # 取最后一个时间步
        if data.ndim >= 3:
            return data[layer]
        return data

    # ================================================================
    # 流速场提取（供主控过程辨识使用）
    # ================================================================

    def extract_velocity_field(self, gwf, layer: int = 0) -> Optional[np.ndarray]:
        """由水头场按达西定律反算指定层的比流量场

        Returns:
            (3, nrow, ncol) 数组：第 0/1/2 个分量分别为 qx/qy/qz，单位 m/d；
            无法算出时返回 None。

        **形状是硬约束**：``ProcessIdentifier.identify_dominant_process`` 内部做
        ``np.linalg.norm(vel, axis=0)`` 再与二维浓度场相乘，所以 ``vel`` 的
        ``ndim`` 必须为 3 —— 即"3 个分量 × 单层网格"。若按 (3, nlay, nrow, ncol)
        堆叠会得到三维的 ``v_mag``，与二维浓度广播失败。故这里只取单层，
        与 ``extract_concentration_field(layer=...)`` 保持同一层、口径一致。

        **为什么不用 flopy 的 get_specific_discharge**：该函数对 MODFLOW 6 要求
        水流模型在 NPF 包里开启 ``save_specific_discharge``，本项目的 NPF 没开，
        预算文件里没有 DATA-SPDIS 记录，调用会失败。由水头场反算不依赖预算文件，
        对**已有的**运行结果也直接可用。水流模型是稳态时水头场不随时间变，
        故单次计算即可代表所有输出时刻。

        不做"比流量 ÷ 孔隙度"的换算：adv/disp/decay 三项通量都是 |v| 的一次
        齐次函数，Pe、Da 与各过程百分比在换算前后完全不变。
        """
        try:
            head = np.asarray(gwf.output.head().get_data(), dtype=float)
            h = head[layer] if head.ndim == 3 else head

            mg = gwf.modelgrid
            delr = np.atleast_1d(np.asarray(mg.delr, dtype=float)).ravel()
            delc = np.atleast_1d(np.asarray(mg.delc, dtype=float)).ravel()

            # 渗透系数：取同层。非均匀 K 也支持（逐格相乘）
            try:
                k = np.asarray(gwf.npf.k.get_data(), dtype=float)
                k = k[layer] if k.ndim == 3 else k
            except Exception as e:
                logger.warning(f"无法读取渗透系数 K（{e}），按 K=1 计算："
                               f"只影响流速量级，不影响 Pe/各过程占比")
                k = 1.0

            dy = float(delc[0]) if len(delc) == h.shape[0] else float(np.mean(delc))
            dx = float(delr[0]) if len(delr) == h.shape[1] else float(np.mean(delr))

            # np.gradient(f, spacing) 的 spacing 是"间距"而非坐标，只支持标量；
            # 本项目的网格恒为均匀（np.full），故取首元素即可代表整体
            dhdy, dhdx = np.gradient(h, dy, dx)
            qx = -k * dhdx
            qy = -k * dhdy
            return np.stack([qx, qy, np.zeros_like(qx)], axis=0)
        except Exception as e:
            logger.warning(f"流速场提取失败: {e}")
            return None

    # ================================================================
    # 水量/质量平衡提取
    # ================================================================

    # MODFLOW 6 在 .lst 里写出的全文水量平衡块标题与数值行
    _BUDGET_HEADER = "VOLUME BUDGET FOR ENTIRE MODEL"
    _ITEM_RE = re.compile(r"^\s*([A-Z][A-Z0-9\-]*)\s*=\s*([-\d.]+(?:[Ee][+-]?\d+)?)")
    _NUM_RE = re.compile(r"[-\d.]+(?:[Ee][+-]?\d+)?")

    def extract_water_budget(self, gwf) -> dict:
        """
        提取水量平衡预算数据

        优先解析 MODFLOW 6 自己写在 .lst 里的全文水量平衡块。理由是那份文本就是
        MF6 计算的权威结果，连 PERCENT DISCREPANCY 都直接给出，不必自己再推一遍；
        而且它不受 FloPy 版本差异影响 —— 本环境 flopy 3.10 读 .bud 得到的值与 .lst
        对不上（例如 CHD 流量读成 -1.8e-13，.lst 里是 1.0e-2，见
        `docs/gwt_verification.md` 第 5 节记录的「MF6 版本倒挂」）。
        .lst 不可用时回退到读 .bud。

        Returns:
            {inflow: {component: value}, outflow: {component: value},
             total_in: float, total_out: float, balance_error: float}
        """
        try:
            from_lst = self._budget_from_lst(gwf)
            if from_lst is not None:
                return from_lst
        except Exception as e:
            logger.warning(f"从 .lst 解析水量平衡失败，回退到 .bud: {e}")
        return self._budget_from_bud(gwf)

    # ---- .lst 路径（首选） ----

    @staticmethod
    def _model_lst_path(gwf):
        ws = getattr(gwf, "model_ws", None)
        name = getattr(gwf, "name", None)
        if not ws or not name:
            return None
        p = Path(ws) / f"{name}.lst"
        return p if p.exists() else None

    def _budget_from_lst(self, gwf) -> Optional[dict]:
        """解析 .lst 末尾那个全文水量平衡块；拿不到就返回 None 交给回退路径。

        块内每行形如
            STO-SS =       13580.7451      STO-SS =    3.2740E-04   STORAGE
        左列是累积量、右列是本步速率，两者各取**第一个**数（即累积量）。
        """
        p = self._model_lst_path(gwf)
        if p is None:
            return None
        text = p.read_text(encoding="utf-8", errors="replace")

        idx = text.rfind(self._BUDGET_HEADER)
        if idx < 0:
            return None

        inflow, outflow = {}, {}
        total_in = total_out = None
        discrepancy = None
        section = None

        for line in text[idx:].splitlines()[:120]:
            s = line.strip()
            if not s:
                continue
            # 先认三种汇总行，再认 IN:/OUT: 段头，最后才认分量行
            if s.startswith("TOTAL IN"):
                nums = self._NUM_RE.findall(s)
                if nums:
                    total_in = float(nums[0])
                continue
            if s.startswith("TOTAL OUT"):
                nums = self._NUM_RE.findall(s)
                if nums:
                    total_out = float(nums[0])
                continue
            if s.startswith("PERCENT DISCREPANCY"):
                nums = self._NUM_RE.findall(s)
                if nums:
                    discrepancy = float(nums[0])
                break
            if s.startswith("IN:") and "=" not in s:
                section = "in"
                continue
            if s.startswith("OUT:") and "=" not in s:
                section = "out"
                continue
            if section is None or "=" not in s:
                continue
            m = self._ITEM_RE.match(line)
            if not m:
                continue
            name, value = m.group(1), float(m.group(2))
            bucket = inflow if section == "in" else outflow
            bucket[name] = bucket.get(name, 0.0) + value

        if total_in is None or total_out is None:
            return None

        if discrepancy is not None:
            # .lst 给的是百分数（-0.01 表示 -0.01%）
            error = abs(discrepancy) / 100.0
        else:
            error = (abs(total_in - total_out) / max(total_in, total_out, 1e-10)
                     if total_in > 0 else 0.0)

        return {
            "inflow": inflow,
            "outflow": outflow,
            "total_in": total_in,
            "total_out": total_out,
            "balance_error": error,
            "source": "lst",
        }

    # ---- .bud 路径（回退） ----

    def _budget_from_bud(self, gwf) -> dict:
        try:
            budget_obj = gwf.output.budget()
            kstpkper_list = budget_obj.get_kstpkper()
        except Exception as e:
            logger.warning(f"水量平衡提取失败: {e}")
            return {"error": str(e)}

        kpk = kstpkper_list[-1] if kstpkper_list else None
        inflow, outflow = {}, {}

        try:
            try:
                names = budget_obj.get_unique_record_names(decode=True)
            except Exception:
                names = None

            if names:
                for nm in names:
                    recs = budget_obj.get_data(text=nm, kstpkper=kpk)
                    self._accumulate(recs, str(nm).strip().upper(), inflow, outflow)
            else:
                # 拿不到记录名时只能整批取，退化成一个 UNKNOWN 项
                self._accumulate(budget_obj.get_data(kstpkper=kpk),
                                 "UNKNOWN", inflow, outflow)
        except Exception as e:
            logger.warning(f"水量平衡提取失败: {e}")
            return {"error": str(e)}

        total_in = sum(inflow.values())
        total_out = sum(outflow.values())
        error = (abs(total_in - total_out) / max(total_in, total_out, 1e-10)
                 if total_in > 0 else 0.0)

        return {
            "inflow": inflow,
            "outflow": outflow,
            "total_in": total_in,
            "total_out": total_out,
            "balance_error": error,
            "source": "bud",
        }

    @staticmethod
    def _flow_of(rec) -> Optional[float]:
        """从一条预算记录里取出流量值。

        MF6 的预算记录分两种，绝不能按 dtype.names 全量累加：
          - 带命名字段（如 node/node2/q/CONCENTRATION）：**只有 `q` 是流量**。
            node/node2 是格点索引，CONCENTRATION 是辅助浓度，qx/qy/qz 是分量。
            全量累加时光 node 一项就是几万个索引之和，会把不平衡率撑到 100%。
          - 不带命名字段（如 STO-SS、FLOW-JA-FACE）：整个数组就是流量。
        """
        if not isinstance(rec, np.ndarray):
            return None
        if rec.dtype.names:
            if "q" not in rec.dtype.names:
                return 0.0
            return float(np.sum(rec["q"]))
        try:
            return float(np.sum(rec))
        except (TypeError, ValueError):
            return None

    @classmethod
    def _accumulate(cls, records, name: str, inflow: dict, outflow: dict) -> None:
        if records is None:
            return
        seq = records if isinstance(records, (list, tuple)) else [records]
        for rec in seq:
            v = cls._flow_of(rec)
            if v is None:
                continue
            if v >= 0:
                inflow[name] = inflow.get(name, 0.0) + v
            else:
                outflow[name] = outflow.get(name, 0.0) + abs(v)

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

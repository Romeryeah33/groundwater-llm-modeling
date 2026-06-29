"""
JSON IR → FloPy 参数映射器

建立从标准化 JSON IR 到 FloPy MODFLOW 6 / MT3DMS / RT3D API 调用的精准映射。
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class FloPyModelMapper:
    """JSON IR → FloPy MODFLOW 6 参数映射器"""

    # ================================================================
    # 映射表
    # ================================================================

    # 模型类型映射
    MODEL_TYPE_MAP = {
        "groundwater_flow": "gwf",
        "solute_transport": "gwt",
        "reactive_transport": "gwt",   # GWT + RT3D reaction
    }

    # 边界类型 → MODFLOW 6 包名
    BOUNDARY_PACKAGE_MAP = {
        "chd": "chd",    # 定水头
        "ghb": "ghb",    # 通用水头
        "wel": "wel",    # 井
        "riv": "riv",    # 河流
        "drn": "drn",    # 排水沟
        "rch": "rch",    # 面状补给
        "evt": "evt",    # 蒸散发
    }

    # 求解器名称映射: 传统 MODFLOW → MODFLOW 6 IMS linear_acceleration
    SOLVER_MAP = {
        "pcg": "CG",          # Preconditioned Conjugate Gradient → Conjugate Gradient
        "nwt": "BICGSTAB",    # Newton → Bi-Conjugate Gradient Stabilized
        "sip": "CG",
        "sor": "CG",
        "cg": "CG",
        "bicgstab": "BICGSTAB",
    }

    # ================================================================
    # 网格参数映射
    # ================================================================

    @staticmethod
    def _unwrap_value(v, default=None):
        """安全解包值：处理 LLM 可能输出的 {'value': ..., 'unit': ...} 格式"""
        if isinstance(v, dict):
            return v.get("value", default)
        if v is not None:
            return v
        return default

    def map_grid(self, json_ir: dict) -> dict:
        """提取网格参数"""
        grid = json_ir.get("grid", {})

        delr = self._unwrap_value(grid.get("delr"), 20.0)
        delc = self._unwrap_value(grid.get("delc"), 20.0)

        # 如果为数组则直接使用，如果是标量则需扩展
        ncol = int(self._unwrap_value(grid.get("ncol"), 50))
        nrow = int(self._unwrap_value(grid.get("nrow"), 50))

        if isinstance(delr, (int, float)):
            delr = np.full(ncol, delr)
        elif isinstance(delr, list):
            delr = np.asarray(delr[:ncol]) if len(delr) >= ncol else np.pad(np.asarray(delr), (0, ncol - len(delr)), 'edge')
        else:
            delr = np.full(ncol, 20.0)

        if isinstance(delc, (int, float)):
            delc = np.full(nrow, delc)
        elif isinstance(delc, list):
            delc = np.asarray(delc[:nrow]) if len(delc) >= nrow else np.pad(np.asarray(delc), (0, nrow - len(delc)), 'edge')
        else:
            delc = np.full(nrow, 20.0)

        top_raw = grid.get("top", 100.0)
        top = self._unwrap_value(top_raw, 100.0)

        bot_raw = grid.get("bottom", 50.0)
        if isinstance(bot_raw, dict):
            bot = self._unwrap_value(bot_raw, 50.0)
        elif isinstance(bot_raw, (int, float)):
            bot = bot_raw
        elif isinstance(bot_raw, list):
            bot = bot_raw
        else:
            bot = 50.0

        return {
            "nrow": nrow,
            "ncol": ncol,
            "nlay": int(self._unwrap_value(grid.get("nlay"), 1)),
            "delr": delr,
            "delc": delc,
            "top": top,
            "botm": bot,
        }

    # ================================================================
    # 含水层参数映射
    # ================================================================

    def map_aquifer_params(self, json_ir: dict) -> dict:
        """提取含水层参数用于 NPF 和 STO 包"""
        aqua = json_ir.get("aquifer", {})
        params = aqua.get("parameters", {})

        result = {
            "aquifer_type": aqua.get("type", "confined"),
        }

        # NPF 包参数（使用 _unwrap_value 兼容 dict 包装和裸值）
        hk_v = self._unwrap_value(params.get("hk"))
        if hk_v is not None:
            result["hk"] = hk_v

        vka_v = self._unwrap_value(params.get("vka"))
        if vka_v is not None:
            result["vka"] = vka_v

        # 各向异性: hani = Kh/Kv, 因此 Kv = Kh / hani
        hani_v = self._unwrap_value(params.get("hani"))
        if hani_v is not None and vka_v is not None:
            pass
        elif hani_v is not None and hk_v is not None:
            result["vka"] = hk_v / hani_v

        # STO 包参数
        sy_v = self._unwrap_value(params.get("sy"))
        if sy_v is not None:
            result["sy"] = sy_v

        ss_v = self._unwrap_value(params.get("ss"))
        if ss_v is not None:
            result["ss"] = ss_v

        # 孔隙度（用于运移）
        porosity_v = self._unwrap_value(params.get("porosity"))
        if porosity_v is not None:
            result["porosity"] = porosity_v

        return result

    # ================================================================
    # 边界条件映射
    # ================================================================

    def _infer_edge_indices(self, bc_name: str, grid: dict, layer: int = 0) -> list:
        """
        从边界名称推断网格边界的 IJ 索引

        支持：west/西/left/左, east/东/right/右, north/北/top/上, south/南/bottom/下

        注意：不使用单字符关键词（如 "w"/"e"/"n"/"s"），
        因为它们会与井名（"well"→含"w"→误判为west）、
        复合方向（"northeast"→含"east"→误判为east）等产生误匹配。
        """
        name_lower = (bc_name or "").lower()
        nrow = grid.get("nrow", 50)
        ncol = grid.get("ncol", 50)
        # 防御性解包：LLM 可能返回 {"value": 40} 而非裸 int
        if isinstance(nrow, dict): nrow = nrow.get("value", 50)
        if isinstance(ncol, dict): ncol = ncol.get("value", 50)

        # 注意：只使用有区分力的关键词，不含单字母匹配
        # 使用单词边界匹配避免 "northeast" 匹配到 "east" 等问题
        west_keywords = ("west", "西", "左", "left", "xmin")
        east_keywords = ("east", "东", "右", "right", "xmax")
        north_keywords = ("north", "北", "上", "top", "ymax")
        south_keywords = ("south", "南", "下", "bottom", "ymin")

        # 辅助函数：检查关键词是否作为独立单词出现（而非其他词的子串）
        def _word_match(text: str, keywords: tuple) -> bool:
            for kw in keywords:
                idx = text.find(kw)
                if idx < 0:
                    continue
                # 对中文字符（非ASCII）关键词，使用简单子串匹配
                # 因为 "西" 应该匹配 "西侧边界"
                if any(ord(c) > 127 for c in kw):
                    return True
                # 对 ASCII 关键词，检查单词边界：
                # "east" 应匹配 "east_boundary"，但不应匹配 "northeast"
                before_ok = idx == 0 or not text[idx - 1].isalpha()
                after_ok = (idx + len(kw) >= len(text)) or not text[idx + len(kw)].isalpha()
                if before_ok and after_ok:
                    return True
            return False

        indices = []

        if _word_match(name_lower, west_keywords):
            for r in range(nrow):
                indices.append({"layer": layer, "row": r, "col": 0, "period": 0})
        elif _word_match(name_lower, east_keywords):
            for r in range(nrow):
                indices.append({"layer": layer, "row": r, "col": ncol - 1, "period": 0})
        elif _word_match(name_lower, north_keywords):
            for c in range(ncol):
                indices.append({"layer": layer, "row": 0, "col": c, "period": 0})
        elif _word_match(name_lower, south_keywords):
            for c in range(ncol):
                indices.append({"layer": layer, "row": nrow - 1, "col": c, "period": 0})

        return indices

    def map_boundary_conditions(self, json_ir: dict) -> dict:
        """
        映射边界条件到 MODFLOW 6 包的 stress_period_data

        当 LLM 未生成 grid_indices 时，自动从边界名称推断网格位置。

        Returns:
            {package_name: {period: [(cell_data), ...], ...}, ...}
        """
        bcs = json_ir.get("boundary_conditions", [])
        stress_periods = json_ir.get("stress_periods", {})
        if not isinstance(stress_periods, dict):
            stress_periods = {}
        nper = stress_periods.get("nper", 1)
        grid = json_ir.get("grid", {})

        packages = {}

        for bc in bcs:
            bc_type = bc.get("type", "")
            package_name = self.BOUNDARY_PACKAGE_MAP.get(bc_type, bc_type)

            if package_name not in packages:
                packages[package_name] = {}

            # 提取参数
            params = bc.get("parameters", {})

            # 获取网格IJK索引（如果有）
            indices = bc.get("geometry", {}).get("grid_indices", [])

            # 如果没有显式索引，尝试从名称推断
            if not indices:
                bc_name = bc.get("name", "")
                inferred = self._infer_edge_indices(bc_name, grid)
                if inferred:
                    indices = inferred
                    logger.info(f"边界 '{bc_name}' 自动推断为 {len(indices)} 个网格单元")

            # 获取应力期数据
            sp_data = bc.get("stress_periods", [])

            if bc_type == "chd":
                # head 可以是 float 或 {"value": float}
                raw_head = params.get("head", 0.0)
                if isinstance(raw_head, dict):
                    head = self._unwrap_value(raw_head, 0.0)
                else:
                    head = float(raw_head)
                if indices:
                    for idx in indices:
                        per = idx.get("period", 0)
                        if per not in packages[package_name]:
                            packages[package_name][per] = []
                        packages[package_name][per].append([
                            (idx.get("layer", 0), idx.get("row", 0), idx.get("col", 0)),
                            head,
                        ])
                elif sp_data:
                    for sp in sp_data:
                        per = sp.get("period", 0)
                        if per not in packages[package_name]:
                            packages[package_name][per] = []
                        logger.warning(f"CHD '{bc.get('name')}' without grid indices — skipped")

            elif bc_type == "wel":
                raw_rate = params.get("rate", 0.0)
                rate = raw_rate.get("value", 0.0) if isinstance(raw_rate, dict) else float(raw_rate)
                if indices:
                    for idx in indices:
                        per = idx.get("period", 0)
                        if per not in packages[package_name]:
                            packages[package_name][per] = []
                        packages[package_name][per].append([
                            (idx.get("layer", 0), idx.get("row", 0), idx.get("col", 0)),
                            rate,
                        ])
                else:
                    # 井没有位置信息 → 放在模型中心
                    center_row = int(self._unwrap_value(grid.get("nrow"), 50)) // 2
                    center_col = int(self._unwrap_value(grid.get("ncol"), 50)) // 2
                    logger.info(f"井 '{bc.get('name')}' 默认放在中心 ({center_row}, {center_col})")
                    packages[package_name][0] = [
                        [(0, center_row, center_col), rate]
                    ]

            elif bc_type == "ghb":
                raw_h = params.get("head", 0.0); head = raw_h.get("value", 0.0) if isinstance(raw_h, dict) else float(raw_h)
                raw_c = params.get("conductance", 100.0); cond = raw_c.get("value", 100.0) if isinstance(raw_c, dict) else float(raw_c)
                if indices:
                    for idx in indices:
                        per = idx.get("period", 0)
                        if per not in packages[package_name]:
                            packages[package_name][per] = []
                        packages[package_name][per].append([
                            (idx.get("layer", 0), idx.get("row", 0), idx.get("col", 0)),
                            head, cond,
                        ])

            elif bc_type == "rch":
                raw_r = params.get("rate", 0.0)
                rate = raw_r.get("value", 0.0) if isinstance(raw_r, dict) else float(raw_r)
                if rate != 0:
                    # 面状补给使用 Recharge 包（全区域）
                    packages[package_name][0] = rate

            elif bc_type == "riv":
                raw_s = params.get("stage", 0.0); stage = raw_s.get("value", 0.0) if isinstance(raw_s, dict) else float(raw_s)
                raw_c = params.get("conductance", 100.0); cond = raw_c.get("value", 100.0) if isinstance(raw_c, dict) else float(raw_c)
                raw_r = params.get("rbot", stage - 1.0); rbot = raw_r.get("value", stage - 1.0) if isinstance(raw_r, dict) else float(raw_r)
                if indices:
                    for idx in indices:
                        per = idx.get("period", 0)
                        if per not in packages[package_name]:
                            packages[package_name][per] = []
                        packages[package_name][per].append([
                            (idx.get("layer", 0), idx.get("row", 0), idx.get("col", 0)),
                            stage, cond, rbot,
                        ])

        return packages

    # ================================================================
    # 溶质运移参数映射
    # ================================================================

    def map_transport_params(self, json_ir: dict) -> dict:
        """提取溶质运移参数"""
        transport = json_ir.get("transport", {})
        if not transport.get("enabled"):
            return {"enabled": False}

        disp = transport.get("dispersion", {})
        result = {
            "enabled": True,
            "species": transport.get("species", ["contaminant"]),
        }

        al = disp_unwrap_value(disp.get("al"))
        if al is not None:
            result["al"] = al

        trpt = disp_unwrap_value(disp.get("trpt"))
        if trpt is not None:
            result["trpt"] = trpt
        else:
            result["trpt"] = 0.1  # 默认 αTH/αL = 0.1

        trpv = disp_unwrap_value(disp.get("trpv"))
        if trpv is not None:
            result["trpv"] = trpv
        else:
            result["trpv"] = 0.01  # 默认 αTV/αL = 0.01

        diffc = disp_unwrap_value(disp.get("diffc"))
        if diffc is not None:
            result["diffc"] = diffc
        else:
            result["diffc"] = 1e-9  # 默认分子扩散 m²/d

        # 反应参数
        reaction = transport.get("reaction", {})
        if reaction.get("sorption", "none") != "none":
            result["sorption"] = reaction["sorption"]
            kd = _unwrap_value(reaction.get("kd"))
            if kd is not None:
                result["kd"] = kd

        decay = reaction.get("decay", {})
        if decay.get("type", "none") != "none":
            result["decay_type"] = decay["type"]
            rate = _unwrap_value(decay.get("first_order_rate"))
            if rate is not None:
                result["decay_rate"] = rate

        # 对流格式
        adv = transport.get("advection", {})
        result["advection_scheme"] = adv.get("scheme", "upstream")

        return result

    # ================================================================
    # 求解器参数映射
    # ================================================================

    def map_solver_params(self, json_ir: dict) -> dict:
        """提取求解器参数，将传统求解器名映射为 MODFLOW 6 原生名"""
        solver = json_ir.get("solver", {})
        params = solver.get("parameters", {})

        raw_name = solver.get("name", "pcg").lower()
        mapped_name = self.SOLVER_MAP.get(raw_name, "CG")

        result = {
            "name": mapped_name,
            "hclose": params.get("hclose", 1e-4),
            "max_iter": int(params.get("max_iter", 100)),
            "relax": params.get("relax", 1.0),
        }
        # rclose 只在用户显式指定时才传递，让 _build_solver 根据 hclose 自动推算
        if "rclose" in params:
            result["rclose"] = params["rclose"]
        return result

    # ================================================================
    # 完整映射
    # ================================================================

    def map_all(self, json_ir: dict) -> dict:
        """
        一键完整映射：JSON IR → FloPy 构建参数

        Returns:
            {grid, aquifer, boundaries, transport, solver, time_params, ...}
        """
        return {
            "grid": self.map_grid(json_ir),
            "aquifer": self.map_aquifer_params(json_ir),
            "boundaries": self.map_boundary_conditions(json_ir),
            "transport": self.map_transport_params(json_ir),
            "solver": self.map_solver_params(json_ir),
            "time": self._map_time_params(json_ir),
        }

    def _map_time_params(self, json_ir: dict) -> dict:
        """提取时间离散参数"""
        stress = json_ir.get("stress_periods", {})
        if not isinstance(stress, dict):
            stress = {}
        return {
            "nper": stress.get("nper", 1),
            "perlen": stress.get("perlen", 1.0),
            "steady_state": stress.get("steady_state", True),
            "period_data": stress.get("period_data", [
                {"perlen": 1.0, "nstp": 1, "tsmult": 1.0},
            ]),
        }


# 全局实例
model_mapper = FloPyModelMapper()

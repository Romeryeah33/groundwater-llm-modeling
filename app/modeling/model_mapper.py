"""
JSON IR → FloPy 参数映射器

建立从标准化 JSON IR 到 FloPy MODFLOW 6 / MT3DMS / RT3D API 调用的精准映射。
"""

import logging
from typing import Optional

import numpy as np

from app.modeling.source_zone import (
    NO_EDGE_INFERENCE_TYPES,
    SourceZoneUnresolved,
    infer_edge_indices,
    unresolved_message,
)
from app.utils.ir_normalize import (
    TIME_KEYS,
    merge_time_sources,
    normalize_stress_periods,
)

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

    # 对流格式白名单
    # MF6 6.3.0 的 gwt1adv1.f90 只认 UPSTREAM/CENTRAL/TVD；
    # 传入其它值（例如 UTVD）会 store_error(terminate=.TRUE.) 硬终止模拟。
    # 注意：不要给 ADV 传 ats_percel —— 6.3.0 无此选项，同样会硬终止。
    ADV_SCHEME_MAP = {
        "upstream": "UPSTREAM",
        "upstream weighting": "UPSTREAM",
        "上游加权": "UPSTREAM",
        "一阶上游": "UPSTREAM",
        "central": "CENTRAL",
        "central difference": "CENTRAL",
        "中心差分": "CENTRAL",
        "tvd": "TVD",
        "utvd": "TVD",              # 三阶 TVD 在 6.3.0 无对应关键字，退到 TVD
        "ultimate": "TVD",
        "tvd ultimate": "TVD",
        "三阶tvd": "TVD",
        "tvd三阶": "TVD",
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

        # NPF 包参数（使用 _unwrap_value 兼容 bare value 和 dict 包装两种格式）
        hk_v = self._unwrap_value(params.get("hk"))
        if hk_v is not None:
            result["hk"] = hk_v

        vka_v = self._unwrap_value(params.get("vka"))
        if vka_v is not None:
            result["vka"] = vka_v

        # 各向异性: hani = Kh/Kv, 因此 Kv = Kh / hani
        hani_v = self._unwrap_value(params.get("hani"))
        if hani_v is not None and vka_v is not None:
            # 两个都给定时 vka 优先，hani 被忽略。此处刻意不抛不喊 ——
            # 映射器是纯 IR→dict 变换，没有反馈通道；"谁被忽略、差多少"由
            # ClosedLoopValidator._check_physics 的 anisotropy_conflict 警告
            # 报给用户（那条能进界面）。
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

    def _infer_edge_indices(self, bc_name: str, grid: dict, layer=None,
                            bc_type: str = None) -> list:
        """
        从边界名称推断网格边界的 IJ 索引

        支持：west/西/left/左, east/东/right/右, north/北/top/上, south/南/bottom/下

        不使用单字符关键词（如 "w"/"e"/"n"/"s"），因为它们会与井名
        （"well"→含"w"→误判为west）、复合方向（"northeast"→含"east"→误判为east）
        等产生误匹配。

        实现已下沉到 `app.modeling.source_zone.infer_edge_indices`，与源区解析器
        共用同一份关键词表 —— 否则"是不是方位型名称"会有两份判断，对
        `top_of_landfill` 这类名字给出不同答案。`rch`/`evt`/`wel` 一律返回空。
        """
        return infer_edge_indices(bc_name, grid, layer=layer, bc_type=bc_type)

    def map_boundary_conditions(self, json_ir: dict) -> dict:
        """
        映射边界条件到 MODFLOW 6 包的 stress_period_data

        当 LLM 未生成 grid_indices 时，自动从边界名称推断网格位置。

        Returns:
            {package_name: {period: [(cell_data), ...], ...}, ...}
        """
        bcs = json_ir.get("boundary_conditions", [])
        # 归一化实现已下沉到 app.utils.ir_normalize，与 validator 共用一份
        # （stress_periods 可能是裸整数 / 字符串 / 列表 / bool，见该模块说明）
        stress_periods = normalize_stress_periods(json_ir.get("stress_periods", {}))
        nper = stress_periods.get("nper", 1) if isinstance(stress_periods, dict) else 1
        grid = json_ir.get("grid", {})

        packages = {}

        for bc in bcs:
            bc_type = bc.get("type", "")
            package_name = self.BOUNDARY_PACKAGE_MAP.get(bc_type, bc_type)

            if package_name not in packages:
                packages[package_name] = {}

            # 提取参数
            params = bc.get("parameters", {})

            # 浓度（可选）：携带时该包会声明 AUXILIARY CONCENTRATION，
            # 并由 GWT 的 SSM 包引用为该边界的水流浓度
            raw_conc = params.get("concentration")
            if isinstance(raw_conc, dict):
                conc = raw_conc.get("value")
            elif raw_conc is not None:
                try:
                    conc = float(raw_conc)
                except (TypeError, ValueError):
                    conc = None
            else:
                conc = None

            # 获取网格IJK索引（如果有）
            indices = bc.get("geometry", {}).get("grid_indices", [])

            # 如果没有显式索引，尝试从名称推断
            if not indices:
                bc_name = bc.get("name", "")
                inferred = self._infer_edge_indices(bc_name, grid, bc_type=bc_type)
                if inferred:
                    indices = inferred
                    logger.info(f"边界 '{bc_name}' 自动推断为 {len(indices)} 个网格单元")

            # 获取应力期数据
            sp_data = bc.get("stress_periods", [])

            if bc_type == "chd":
                # head 可以是 float 或 {"value": float}
                raw_head = params.get("head", 0.0)
                if isinstance(raw_head, dict):
                    head = raw_head.get("value", 0.0)
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
                        ] + ([conc] if conc is not None else []))
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
                        ] + ([conc] if conc is not None else []))
                else:
                    # 井没有位置信息 → 放在模型中心
                    center_row = grid.get("nrow", 50) // 2
                    center_col = grid.get("ncol", 50) // 2
                    logger.info(f"井 '{bc.get('name')}' 默认放在中心 ({center_row}, {center_col})")
                    packages[package_name][0] = [
                        [(0, center_row, center_col), rate]
                        + ([conc] if conc is not None else [])
                    ]

            elif bc_type == "ghb":
                # GHB 水头可能叫 head 或 stage，兼容两种命名。
                # 注意 head=0.0 是合法值，不能用 `or` 短路（会被当成缺失）
                raw_h = params["head"] if "head" in params else params.get("stage", 0.0)
                head = raw_h.get("value", raw_h) if isinstance(raw_h, dict) else float(raw_h)
                raw_c = params.get("conductance", 100.0); cond = raw_c.get("value", raw_c) if isinstance(raw_c, dict) else float(raw_c)
                if indices:
                    for idx in indices:
                        per = idx.get("period", 0)
                        if per not in packages[package_name]:
                            packages[package_name][per] = []
                        packages[package_name][per].append([
                            (idx.get("layer", 0), idx.get("row", 0), idx.get("col", 0)),
                            head, cond,
                        ] + ([conc] if conc is not None else []))

            elif bc_type == "rch":
                raw_r = params.get("rate", 0.0)
                rate = raw_r.get("value", 0.0) if isinstance(raw_r, dict) else float(raw_r)

                # 补给率作用范围：
                #   "zone"（默认）—— 补给只发生在源区。BC 是**源项**边界，其速率
                #                    只应作用于源区；全场天然补给是**另一条** BC
                #                    （无浓度、标量 rate），走下面 elif rate != 0 分支。
                #   "domain"      —— 全域按 rate 补给，但浓度只在源区内为 conc，
                #                    其余格子取 background_concentration。这是原始
                #                    MT3DMS 台面的形态（全域浓度数组，只有填埋场为
                #                    20000，其余为 0）。
                scope = str(self._unwrap_value(params.get("recharge_scope"), "zone") or "zone").lower()
                bg = self._unwrap_value(params.get("background_concentration"), 0.0)

                if indices:
                    # cell-by-cell 补给（如填埋场多边形）
                    nrow = int(grid.get("nrow", 1) or 1)
                    ncol = int(grid.get("ncol", 1) or 1)
                    if scope in ("domain", "all", "全域", "全局") and conc is not None:
                        # 源区浓度 + 补集本底浓度。显式给出全部格子，避免依赖 MF6
                        # 对未列出格子的缺省行为。
                        in_zone = {(int(i.get("row", 0)), int(i.get("col", 0)))
                                   for i in indices}
                        bg = 0.0 if bg is None else float(bg)
                        per0 = int(indices[0].get("period", 0) or 0)
                        packages[package_name][per0] = [
                            (0, r, c, rate, conc if (r, c) in in_zone else bg)
                            for r in range(nrow) for c in range(ncol)
                        ]
                        logger.info(
                            f"RCH 补给 scope=domain：{nrow*ncol} 格按 {rate:g} 补给，"
                            f"其中 {len(in_zone)} 格浓度 {conc:g}、"
                            f"{nrow*ncol-len(in_zone)} 格浓度 {bg:g}"
                        )
                        continue
                    for idx in indices:
                        per = idx.get("period", 0)
                        if per not in packages[package_name]:
                            packages[package_name][per] = []
                        cellid = (idx.get("layer", 0), idx.get("row", 0), idx.get("col", 0))
                        entry = list(cellid) + [rate]
                        if conc is not None:
                            entry.append(conc)
                        packages[package_name][per].append(tuple(entry))
                elif conc is not None:
                    # 有浓度但没有可解析的注入位置。
                    #
                    # 这里**必须拒绝**，绝不能像以前那样展开成 layer0 全网格：
                    # East Texas 案例中填埋场只有 132 格（0.59%），静默展开成
                    # 22454 格把源放大了 170 倍，污染羽于是变成一张覆盖全域的
                    # 毯子、看起来像水流模型结果，而日志里只有一句到不了 UI 的
                    # info。宁可让用户看见错误，也不要给出静默错误的模型。
                    raise SourceZoneUnresolved(unresolved_message(bc, grid))
                elif rate != 0:
                    # 全场天然补给：无浓度、标量速率。这正是 `scope="zone"` 所
                    # 依赖的独立通道（见上面注释）。
                    #
                    # 但 `package_name` 只由 bc_type 决定，所有 rch BC **共用同一个
                    # "rch" 包**。若前一条 rch BC 已经往 period 0 写了逐格记录，
                    # 这里直接赋值会把它整条抹掉（标量与列表在同一包里无法共存）。
                    # 那种静默覆盖和本轮要消灭的 bug 是同一类，所以拒绝覆盖。
                    if 0 in packages[package_name]:
                        logger.warning(
                            f"RCH '{bc.get('name')}' 是全场标量补给，但 period 0 已有"
                            f"{len(packages[package_name][0])} 条逐格记录（来自先前的 rch BC 或"
                            f"源区解析），已保留逐格记录、忽略该标量。"
                            f"如需「源区 + 全场天然补给」并存，请改用单条 BC 的 "
                            f"parameters.recharge_scope=\"domain\" + background_concentration，"
                            f"或把天然补给拆到独立模型"
                        )
                    else:
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
                        ] + ([conc] if conc is not None else []))

        return packages

    # ================================================================
    # 溶质运移参数映射
    # ================================================================

    def map_transport_params(self, json_ir: dict) -> dict:
        """提取溶质运移参数

        自动检测：即使 transport.enabled 未显式设置，如果有弥散/对流/物种参数
        或 project.model_type 为 solute_transport/reactive_transport，也自动启用。
        """
        transport = json_ir.get("transport", {})
        project = json_ir.get("project", {})

        # ── 自动启用检测 ──
        explicit_enabled = transport.get("enabled", False)
        # enabled 可能是 bool 或 {"value": True, ...} dict 包装
        if isinstance(explicit_enabled, dict):
            explicit_enabled = self._unwrap_value(explicit_enabled)

        model_type_is_transport = project.get("model_type", "") in (
            "solute_transport", "reactive_transport"
        )
        has_dispersion = any(
            self._unwrap_value(transport.get("dispersion", {}).get(k)) is not None
            for k in ("al", "trpt", "trpv", "diffc")
        )
        has_species = bool(transport.get("species"))
        has_advection = bool(transport.get("advection", {}).get("scheme"))
        has_ssm_sources = bool(transport.get("ssm_sources"))
        # 检查是否有任何非 enabled 的 transport 内容
        has_other_content = any(
            k not in ("enabled",) and v
            for k, v in transport.items()
        )

        should_enable = (
            explicit_enabled or model_type_is_transport or has_dispersion
            or has_species or has_advection or has_ssm_sources or has_other_content
        )
        logger.info(
            f"map_transport: explicit={explicit_enabled}, mt={model_type_is_transport}, "
            f"disp={has_dispersion}, sp={has_species}, adv={has_advection}, "
            f"ssm={has_ssm_sources}, other={has_other_content} → enable={should_enable}"
        )

        if not should_enable:
            return {"enabled": False}

        disp = transport.get("dispersion", {})
        result = {
            "enabled": True,
            "species": transport.get("species", ["contaminant"]),
        }

        # 弥散参数可能写在 transport.dispersion.* 里，也可能直接写在
        # transport.* 顶层（两种 IR 写法都见过）。只认前者会让顶层写法
        # **静默**退化成网格尺度默认值——参数看起来给了，实际没用上。
        def _disp(key):
            v = self._unwrap_value(disp.get(key))
            if v is not None:
                return v
            v = self._unwrap_value(transport.get(key))
            if v is not None:
                logger.info(f"弥散参数 {key} 取自 transport 顶层（而非 transport.dispersion）")
            return v

        al = _disp("al")
        if al is not None:
            result["al"] = al
            # 记录来源。调用方（model_builder）据此决定要不要往界面推提示：
            # 走默认值时用户以为自己给了值，实际没用上，必须看得见。
            result["al_source"] = "ir"
        else:
            # 未指定时按网格尺度取默认值：αL ≈ 2Δx 使网格 Péclet 数 ≈ 0.5，
            # 落在数值稳定的安全侧。固定写 10 m 在粗网格（如 20 m 以上）
            # 会得到过大的 Péclet 数，产生振荡。
            grid = json_ir.get("grid", {})
            delr = np.atleast_1d(np.asarray(self._unwrap_value(grid.get("delr"), 20.0), dtype=float))
            delc = np.atleast_1d(np.asarray(self._unwrap_value(grid.get("delc"), 20.0), dtype=float))
            result["al"] = max(10.0, 2.0 * max(float(delr.mean()), float(delc.mean())))
            result["al_source"] = "grid_default"
            logger.warning(f"未指定纵向弥散度 al，按网格尺度取默认值 {result['al']:.2f} m")

        trpt = _disp("trpt")
        if trpt is not None:
            result["trpt"] = trpt
        else:
            result["trpt"] = 0.1  # 默认 αTH/αL = 0.1

        trpv = _disp("trpv")
        if trpv is not None:
            result["trpv"] = trpv
        else:
            result["trpv"] = 0.01  # 默认 αTV/αL = 0.01

        diffc = _disp("diffc")
        if diffc is not None:
            result["diffc"] = diffc
        else:
            result["diffc"] = 1e-9  # 默认分子扩散 m²/d

        # 反应参数
        reaction = transport.get("reaction", {})
        if reaction.get("sorption", "none") != "none":
            result["sorption"] = reaction["sorption"]
            kd = self._unwrap_value(reaction.get("kd"))
            if kd is not None:
                result["kd"] = kd

        decay = reaction.get("decay", {})
        if decay.get("type", "none") != "none":
            result["decay_type"] = decay["type"]
            rate = self._unwrap_value(decay.get("first_order_rate"))
            if rate is not None:
                result["decay_rate"] = rate

        # 对流格式：白名单化，避免把 6.3.0 不认的关键字写进 ADV 包导致硬终止
        adv = transport.get("advection", {})
        raw_scheme = str(self._unwrap_value(adv.get("scheme"), "upstream")).strip().lower()
        scheme = self.ADV_SCHEME_MAP.get(raw_scheme)
        if scheme is None:
            # 宽松兜底：LLM 常给出 "TVD三阶ULTIMATE" 这类复合描述
            for key, val in self.ADV_SCHEME_MAP.items():
                if key in raw_scheme:
                    scheme = val
                    break
        if scheme is None:
            logger.warning(
                f"未知的对流格式 '{adv.get('scheme')}'，回退为 UPSTREAM"
                f"（MF6 6.3.0 仅支持 UPSTREAM/CENTRAL/TVD）"
            )
            scheme = "UPSTREAM"
        result["advection_scheme"] = scheme

        # 孔隙度（GWT 的 MST 包必需）
        por = self._unwrap_value(transport.get("porosity"))
        if por is not None:
            result["porosity"] = por
            result["porosity_source"] = "transport"

        # 初始浓度
        ic = self._unwrap_value(transport.get("initial_concentration"))
        if ic is not None:
            result["initial_concentration"] = ic

        # 源汇项浓度（SSM）
        ssm = transport.get("ssm_sources", [])
        if ssm:
            result["ssm_sources"] = ssm

        # 目标 Courant 数：供 model_builder._ensure_transport_timesteps
        # 反推 Δt 上限（MF6 6.3.0 的 ATS 不做 Courant 控制）
        courant = self._unwrap_value(transport.get("timestep_target_courant"))
        if courant is not None:
            try:
                courant = float(courant)
                if courant > 0:
                    result["timestep_target_courant"] = courant
                else:
                    logger.warning(f"timestep_target_courant={courant:g} 非正，忽略并使用默认 1.0")
            except (TypeError, ValueError):
                logger.warning(f"timestep_target_courant={courant!r} 无法转为数值，忽略")

        logger.info(f"溶质运移已启用: al={result.get('al')}, species={result.get('species')}")

        return result

    # ================================================================
    # 求解器参数映射
    # ================================================================

    def map_solver_params(self, json_ir: dict) -> dict:
        """提取求解器参数，将传统求解器名映射为 MODFLOW 6 原生名"""
        solver = json_ir.get("solver", {})
        params = solver.get("parameters", {})

        # 兼容 solver.name 和 solver.type 两种写法
        raw_name = solver.get("name") or solver.get("type", "pcg")
        raw_name = str(raw_name).lower()
        # 如果 solver 里没有 name/type，从参数里找 solver_type
        if raw_name == "pcg" and isinstance(params, dict):
            st = self._unwrap_value(params.get("solver_type"))
            if st is not None:
                raw_name = str(st).lower()
        mapped_name = self.SOLVER_MAP.get(raw_name, "CG")

        # ── 关键修复：使用 _unwrap_value 安全解包 dict-wrapped 值 ──
        result = {
            "name": mapped_name,
            "hclose": self._unwrap_value(params.get("hclose"), 1e-4),
            "max_iter": int(self._unwrap_value(params.get("max_iter"), 100)),
            "relax": self._unwrap_value(params.get("relax"), 1.0),
        }
        # rclose 只在用户显式指定时才传递，让 _build_solver 根据 hclose 自动推算
        if "rclose" in params:
            result["rclose"] = self._unwrap_value(params.get("rclose"))
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
        """提取时间离散参数

        注意：LLM 有时会把时间信息写进 transport.stress_periods，而顶层
        stress_periods 只剩一个裸整数（如 nper=10）。这里做了回退，避免
        退化成"只算 1 天"。
        """
        # 防御：stress_periods 可能是字符串 / 列表 / **裸数字**。
        # LLM 常写 "stress_periods": 10 表示"分 10 个应力期"——按 nper 读，
        # 否则会退化成"只算 1 个 1 天的应力期"。
        # 归一化实现已下沉到 app.utils.ir_normalize，与 validator 共用一份。
        stress = normalize_stress_periods(json_ir.get("stress_periods", {}))

        # 回退到 transport.stress_periods。**两种情况都要回退**：
        #   1) 顶层连 nper 都没有（缺失 / 非法）
        #   2) 顶层只带 nper、没有 perlen/nstp/period_data/total_time
        # 第 2 种正是 East Texas 的形态：顶层是裸整数 10，真实的时间信息
        # {nper:10, perlen:15000} 被 LLM 塞进了 transport.stress_periods。
        # 只看 nper 判断的话这里不会回退，perlen 就退化成 1 天。
        # 合并逻辑已下沉到 `app.utils.ir_normalize.merge_time_sources`，
        # 与校验层共用一份（否则"时间信息在哪"会有两种判断）。
        merged = merge_time_sources(json_ir)
        if merged is not stress:
            logger.warning("顶层 stress_periods 不完整，已从 transport.stress_periods 补齐时间信息")
        stress = merged

        nper = int(self._unwrap_value(stress.get("nper"), 1) or 1)
        perlen = float(self._unwrap_value(stress.get("perlen"), 1.0) or 1.0)
        nstp = int(self._unwrap_value(stress.get("nstp"), 1) or 1)
        tsmult = float(self._unwrap_value(stress.get("tsmult"), 1.0) or 1.0)

        # total_time 是**一等公民**的"总模拟时长"，优先级高于平铺 perlen。
        #
        # 为什么需要它：用户提示词说的是"模拟时长15000天，均匀分为10个应力期"，
        # 但 `perlen` 按 MF6 原生语义是**每个**应力期的长度，10 × 15000 就是
        # 150000 d（差 10 倍）。而 LLM 无处安放"总时长"这个量 —— schema 里没有
        # 这个字段，`param_guide` 里唯一的示例还把 perlen 平铺在顶层。
        # 给了 total_time 就按 nper 等分，这是确定性的，不依赖 LLM 猜。
        total_time = self._unwrap_value(stress.get("total_time"), None)
        period_data = stress.get("period_data") or []
        if total_time is not None and not period_data:
            try:
                total_time = float(total_time)
            except (TypeError, ValueError):
                total_time = None
            if total_time and total_time > 0 and nper > 0:
                stale = stress.get("perlen")
                if stale:
                    logger.warning(
                        f"time: total_time={total_time:g} d 覆盖平铺 perlen={stale}"
                        f"（后者按 MF6 语义是单期长度，二者极易混淆）"
                    )
                perlen = total_time / nper
                logger.info(
                    f"时间离散: total_time={total_time:g} d 按 {nper} 期等分，"
                    f"每期 perlen={perlen:g} d"
                )

        if not period_data:
            # 关键：由 nper/perlen/nstp 合成，而不是退化成单个 1 天步长
            period_data = [
                {"perlen": perlen, "nstp": nstp, "tsmult": tsmult}
                for _ in range(nper)
            ]
        elif len(period_data) != nper:
            logger.warning(
                f"period_data 长度({len(period_data)}) != nper({nper})，以 period_data 为准"
            )
            nper = len(period_data)

        total = sum(float(pd.get("perlen", 1.0) or 1.0) for pd in period_data)
        logger.info(
            f"时间离散: nper={nper}, 总时长={total:g} d, "
            f"总时间步={sum(int(pd.get('nstp', 1) or 1) for pd in period_data)}"
        )

        return {
            "nper": nper,
            "perlen": perlen,
            "steady_state": bool(stress.get("steady_state", True)),
            "period_data": period_data,
        }


# 全局实例
model_mapper = FloPyModelMapper()

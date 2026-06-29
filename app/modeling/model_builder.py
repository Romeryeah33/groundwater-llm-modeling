"""
FloPy 全自动模型构建器

从 JSON IR → 完整的 MODFLOW 6 模拟对象
处理流程：网格生成 → Sim创建 → GWF → (可选)GWT → 求解器 → 写入
"""

import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import flopy
from flopy.mf6 import MFSimulation, ModflowTdis, ModflowGwf, ModflowIms

from app.modeling.grid_generator import AutoGridGenerator, grid_generator
from app.modeling.model_mapper import FloPyModelMapper, model_mapper
from app.config import settings

logger = logging.getLogger(__name__)


class AutoModelBuilder:
    """
    从 JSON IR 到完整 FloPy 模拟的全自动构建器

    Usage:
        builder = AutoModelBuilder("./workspace")
        sim = builder.build_simulation(json_ir)
        sim.run_simulation()
    """

    def __init__(self, workspace: Optional[str] = None):
        self.workspace = Path(workspace or settings.workspace_dir)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.mapper = model_mapper
        self.grid_gen = grid_generator
        self._sim = None
        self._gwf = None
        self._gwt = None

    # ================================================================
    # 主入口：一键构建完整模拟
    # ================================================================

    def build_simulation(self, json_ir: dict, sim_name: str = "auto_model") -> MFSimulation:
        """
        从 JSON IR 一键构建完整 MODFLOW 6 模拟对象

        Returns:
            flopy.mf6.MFSimulation — 已写入输入文件的完整模拟对象
        """
        logger.info(f"开始构建模型: {sim_name}")

        # Step 1: 自动生成网格（如果 JSON IR 中未指定网格参数）
        if self._needs_grid_generation(json_ir):
            grid_params = self.grid_gen.generate(json_ir.get("domain", {}))
            if "grid" not in json_ir:
                json_ir["grid"] = {}
            json_ir["grid"].setdefault("nrow", grid_params["nrow"])
            json_ir["grid"].setdefault("ncol", grid_params["ncol"])
            json_ir["grid"].setdefault("nlay", grid_params["nlay"])
            json_ir["grid"].setdefault("delr", grid_params["delr"])
            json_ir["grid"].setdefault("delc", grid_params["delc"])

        # Step 2: 完整映射参数
        mapped = self.mapper.map_all(json_ir)

        # Step 3: 创建 Simulation
        self._sim = MFSimulation(
            sim_name=sim_name,
            version="mf6",
            exe_name="mf6",
            sim_ws=str(self.workspace / sim_name),
        )

        # Step 4: 时间离散 (TDIS)
        time_params = mapped["time"]
        self._build_tdis(time_params)

        # Step 5: 水流模型 (GWF)
        self._gwf = self._build_gwf(sim_name, mapped)

        # Step 6: (条件) 溶质运移模型 (GWT)
        transport = mapped.get("transport", {})
        if transport.get("enabled"):
            self._gwt = self._build_gwt(sim_name, mapped)

        # Step 7: 求解器 (IMS)
        self._build_solver(mapped["solver"])

        # Step 8: GWF-GWT 交换
        if self._gwt is not None:
            self._build_exchange(sim_name)

        # Step 9: 写入输入文件
        os.makedirs(str(self.workspace / sim_name), exist_ok=True)
        self._sim.write_simulation(silent=True)

        logger.info(f"模型构建完成: {sim_name}")
        return self._sim

    def _needs_grid_generation(self, json_ir: dict) -> bool:
        """判断是否需要自动生成网格"""
        grid = json_ir.get("grid", {})
        return not grid.get("nrow") or not grid.get("ncol")

    # ================================================================
    # TDIS
    # ================================================================

    def _build_tdis(self, time_params: dict):
        """构建时间离散包"""
        per_data = time_params.get("period_data", [{"perlen": 1.0, "nstp": 1, "tsmult": 1.0}])

        perioddata = []
        for pd in per_data:
            perioddata.append((
                pd.get("perlen", 1.0),
                pd.get("nstp", 1),
                pd.get("tsmult", 1.0),
            ))

        ModflowTdis(
            self._sim,
            time_units="days",
            nper=len(perioddata),
            perioddata=perioddata,
        )

    # ================================================================
    # GWF 水流模型
    # ================================================================

    def _build_gwf(self, sim_name: str, mapped: dict):
        """构建地下水流模型"""
        gwf = ModflowGwf(
            self._sim,
            modelname=f"{sim_name}_gwf",
            save_flows=True,
        )

        grid = mapped["grid"]
        aquifer = mapped["aquifer"]

        # DIS: 离散包
        nlay = grid["nlay"]
        delr = np.asarray(grid["delr"])
        delc = np.asarray(grid["delc"])

        # 处理 botm: [底1, 底2, ..., 底n]
        top_val = grid["top"]
        botm_val = grid["botm"]
        if isinstance(botm_val, (int, float)):
            if nlay == 1:
                botm = [botm_val]
            else:
                # 等分厚度
                dz = (top_val - botm_val) / nlay
                botm = [top_val - dz * (i + 1) for i in range(nlay)]
        elif isinstance(botm_val, list):
            botm = botm_val
        else:
            botm = [top_val - 50.0]

        flopy.mf6.ModflowGwfdis(
            gwf,
            nrow=grid["nrow"],
            ncol=grid["ncol"],
            nlay=nlay,
            delr=delr,
            delc=delc,
            top=top_val,
            botm=botm,
        )

        # NPF: 节点属性流（渗透系数）
        hk = aquifer.get("hk", 1.0)
        vka = aquifer.get("vka", hk * 0.1)
        icelltype = 1 if aquifer["aquifer_type"] in ("unconfined", "convertible") else 0

        flopy.mf6.ModflowGwfnpf(
            gwf,
            k=hk,
            k33=vka,
            icelltype=icelltype,
        )

        # STO: 储水
        sy = aquifer.get("sy", 0.15)
        ss = aquifer.get("ss", 1e-5)
        transient = not mapped["time"].get("steady_state", False)

        flopy.mf6.ModflowGwfsto(
            gwf,
            ss=ss,
            sy=sy,
            transient={0: transient},
        )

        # IC: 初始条件
        flopy.mf6.ModflowGwfic(gwf, strt=top_val)

        # 边界条件包
        self._build_boundary_packages(gwf, mapped["boundaries"], grid)

        # OC: 输出控制
        flopy.mf6.ModflowGwfoc(
            gwf,
            budget_filerecord=f"{sim_name}_gwf.bud",
            head_filerecord=f"{sim_name}_gwf.hds",
            saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        )

        return gwf

    def _build_boundary_packages(self, gwf, boundary_map: dict, grid: dict):
        """构建边界条件包"""
        nrow, ncol, nlay = grid["nrow"], grid["ncol"], grid["nlay"]

        for pkg_name, sp_data in boundary_map.items():
            if pkg_name == "chd" and sp_data:
                # 定水头
                chd_spd = {}
                for per, records in sp_data.items():
                    chd_spd[int(per)] = [(rec[0], rec[1]) for rec in records]
                flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chd_spd)

            elif pkg_name == "wel" and sp_data:
                # 井
                wel_spd = {}
                for per, records in sp_data.items():
                    wel_spd[int(per)] = [(rec[0], rec[1]) for rec in records]
                flopy.mf6.ModflowGwfwel(gwf, stress_period_data=wel_spd)

            elif pkg_name == "ghb" and sp_data:
                # 通用水头
                ghb_spd = {}
                for per, records in sp_data.items():
                    ghb_spd[int(per)] = records
                flopy.mf6.ModflowGwfghb(gwf, stress_period_data=ghb_spd)

            elif pkg_name == "rch" and sp_data:
                # 面状补给：FloPy 3.10 API 格式待适配，暂时跳过
                logger.warning(
                    "RCH (面状补给) 包在 FloPy 3.10 中 API 格式待适配，跳过包创建"
                )

            elif pkg_name == "riv" and sp_data:
                # 河流
                riv_spd = {}
                for per, records in sp_data.items():
                    riv_spd[int(per)] = records
                flopy.mf6.ModflowGwfriv(gwf, stress_period_data=riv_spd)

            elif pkg_name == "drn" and sp_data:
                # 排水沟
                drn_spd = {}
                for per, records in sp_data.items():
                    drn_spd[int(per)] = records
                flopy.mf6.ModflowGwfdrn(gwf, stress_period_data=drn_spd)

    # ================================================================
    # GWT 溶质运移模型
    # ================================================================

    def _build_gwt(self, sim_name: str, mapped: dict):
        """构建溶质运移模型"""
        gwt = flopy.mf6.ModflowGwt(
            self._sim,
            modelname=f"{sim_name}_gwt",
            save_flows=True,
        )

        grid = mapped["grid"]
        transport = mapped["transport"]

        # GWT DIS
        flopy.mf6.ModflowGwtdis(
            gwt,
            nrow=grid["nrow"],
            ncol=grid["ncol"],
            nlay=grid["nlay"],
            delr=np.asarray(grid["delr"]) if isinstance(grid["delr"], list) else grid["delr"],
            delc=np.asarray(grid["delc"]) if isinstance(grid["delc"], list) else grid["delc"],
            top=grid["top"],
            botm=grid["botm"][-1] if isinstance(grid["botm"], list) else grid["botm"],
        )

        # IC: 初始浓度
        flopy.mf6.ModflowGwtic(gwt, strt=0.0)

        # ADV: 对流包
        scheme = transport.get("advection_scheme", "upstream")
        flopy.mf6.ModflowGwtadv(gwt, scheme=scheme)

        # DSP: 弥散包
        al = transport.get("al", 10.0)
        trpt = transport.get("trpt", 0.1)
        trpv = transport.get("trpv", 0.01)
        diffc = transport.get("diffc", 1e-9)

        flopy.mf6.ModflowGwtdsp(
            gwt,
            alh=al,
            ath1=al * trpt,
            atv=al * trpv,
            diffc=diffc,
        )

        # SSM: 源汇项（从 GWF 流场继承）
        # 无额外质量源时，Sources 留空
        flopy.mf6.ModflowGwtssm(gwt, sources=None)

        # OC
        flopy.mf6.ModflowGwtoc(
            gwt,
            budget_filerecord=f"{sim_name}_gwt.bud",
            concentration_filerecord=f"{sim_name}_gwt.ucn",
            saverecord=[("CONCENTRATION", "ALL"), ("BUDGET", "LAST")],
        )

        return gwt

    # ================================================================
    # 求解器 & 交换
    # ================================================================

    def _build_solver(self, solver_params: dict):
        """构建求解器

        IMS 收敛准则：inner_dvclose 必须 ≦ outer_dvclose，否则线性求解器
        会因过于宽松而提前退出，导致非线性外迭代无法收敛。
        MODFLOW 6 官方建议 inner = outer / 10 ~ outer / 100。
        """
        # 防御性解包（LLM 可能输出 {"value": X} 包装值）
        def _sv(key, default):
            v = solver_params.get(key, default)
            if isinstance(v, dict): return v.get("value", default)
            return v
        outer_dvclose = _sv("hclose", 1e-5)
        # 如果用户未指定 rclose，默认设为 outer 的 1/10
        rclose = _sv("rclose", None)
        if rclose is None:
            inner_dvclose = outer_dvclose * 0.1
        else:
            inner_dvclose = rclose
            # 安全检查：inner 不应比 outer 更松
            if inner_dvclose > outer_dvclose:
                logger.warning(
                    f"IMS inner_dvclose ({inner_dvclose}) > outer_dvclose ({outer_dvclose}), "
                    f"自动调整为 outer_dvclose/10 ({outer_dvclose*0.1})"
                )
                inner_dvclose = outer_dvclose * 0.1

        ModflowIms(
            self._sim,
            print_option="summary",
            outer_dvclose=outer_dvclose,
            inner_dvclose=inner_dvclose,
            linear_acceleration=solver_params.get("name", "CG"),
        )

    def _build_exchange(self, sim_name: str):
        """构建 GWF-GWT 交换"""
        flopy.mf6.ModflowGwfgwt(
            self._sim,
            exgtype="GWF6-GWT6",
            exgmnamea=f"{sim_name}_gwf",
            exgmnameb=f"{sim_name}_gwt",
        )

    # ================================================================
    # 便捷方法
    # ================================================================

    def build_and_run(self, json_ir: dict, sim_name: str = "auto_model") -> dict:
        """
        构建并运行模拟，返回结果摘要

        Returns:
            {success, head_range, budget_error, output_path, ...}
        """
        sim = self.build_simulation(json_ir, sim_name)
        success, _ = sim.run_simulation(silent=True)

        if not success:
            return {"success": False, "error": "模拟运行失败"}

        # 读取结果
        try:
            head = self._gwf.output.head().get_data()
            head_range = (float(head.min()), float(head.max()))
        except Exception:
            head_range = (None, None)

        return {
            "success": True,
            "head_range": head_range,
            "output_path": str(self.workspace / sim_name),
            "sim": sim,
            "gwf": self._gwf,
            "gwt": self._gwt,
        }


# 全局实例
auto_builder = AutoModelBuilder()

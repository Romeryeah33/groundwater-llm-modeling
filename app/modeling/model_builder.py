"""
FloPy 全自动模型构建器

从 JSON IR → 完整的 MODFLOW 6 模拟对象
处理流程：网格生成 → Sim创建 → GWF → (可选)GWT → 求解器 → 写入
"""

import os
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Optional, List

import numpy as np
import flopy
from flopy.mf6 import MFSimulation, ModflowTdis, ModflowGwf, ModflowIms

from app.modeling.grid_generator import AutoGridGenerator, grid_generator
from app.modeling.model_mapper import FloPyModelMapper, model_mapper
from app.config import settings

logger = logging.getLogger(__name__)

# SSM 辅助变量名。必须全大写，且水流包与 SSM 包的声明要完全一致
# —— MF6 的 gwt1ssm1.f90 用全等比较查找该辅助变量。
AUX_CONC = "CONCENTRATION"

# 边界类型 → MODFLOW 6 水流包名（SSM 的 pname 需与此一致）
BOUNDARY_PKG_NAME = {
    "chd": "CHDA",
    "ghb": "GHBA",
    "wel": "WELA",
    "riv": "RIVA",
    "drn": "DRNA",
    "rch": "RCHA",
    "evt": "EVTA",
}

# 补给/蒸散可能以低于单元浓度的浓度排出，需用 AUXMIXED 而非 AUX
AUXMIXED_PKGS = {"rch", "evt"}

# 各水流包的 FloPy 构造器
PKG_BUILDERS = {
    "chd": flopy.mf6.ModflowGwfchd,
    "wel": flopy.mf6.ModflowGwfwel,
    "ghb": flopy.mf6.ModflowGwfghb,
    "rch": flopy.mf6.ModflowGwfrch,
    "riv": flopy.mf6.ModflowGwfriv,
    "drn": flopy.mf6.ModflowGwfdrn,
    "evt": flopy.mf6.ModflowGwfevt,
}

# 各水流包记录中「除 cellid 外的值列数」（不含浓度辅助列）
PKG_BASE_COLS = {
    "chd": 1,   # head
    "wel": 1,   # q
    "ghb": 2,   # head, cond
    "rch": 1,   # recharge
    "riv": 3,   # stage, cond, rbot
    "drn": 2,   # elev, cond
    "evt": 3,   # surface, rate, depth
}


def _as_float(v, default=0.0):
    """安全转 float：支持 {'value': ...} 包装与 None"""
    if isinstance(v, dict):
        v = v.get("value")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _resolve_botm(grid: dict) -> list:
    """把各种形态的 bottom/botm 归一化为长度恰好为 nlay 的列表。

    GWF 与 GWT 必须使用完全相同的网格几何。MF6 的 GWF-GWT 交换只比较
    节点数、不校验 top/botm，几何不一致会**静默**算出错误的物理场。
    因此这里作为唯一真源，禁止各构建分支再各自解析 botm。
    """
    nlay = int(_as_float(grid.get("nlay"), 1) or 1)
    top = _as_float(grid.get("top"), 100.0)

    raw = grid.get("botm", grid.get("bottom"))
    if isinstance(raw, dict):
        raw = raw.get("value")

    if isinstance(raw, (list, tuple, np.ndarray)):
        botm = [float(v) for v in np.asarray(raw, dtype=float).ravel()]
        if len(botm) >= nlay:
            return botm[:nlay]
        if botm:
            # 不足 nlay 时用末值补齐，保持单调不增
            botm.extend([botm[-1]] * (nlay - len(botm)))
            return botm
        raw = None

    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        if nlay == 1:
            return [float(raw)]
        # 多层且只给了总底界：等分厚度
        dz = (top - float(raw)) / nlay
        return [top - dz * (i + 1) for i in range(nlay)]

    # 兜底：每层 50 m 递减
    return [top - 50.0 * (i + 1) for i in range(nlay)]


def _resolve_dis(grid: dict) -> dict:
    """归一化 DIS 参数，作为 GWF/GWT 共用的唯一网格定义"""
    nlay = int(_as_float(grid.get("nlay"), 1) or 1)
    nrow = int(_as_float(grid.get("nrow"), 1) or 1)
    ncol = int(_as_float(grid.get("ncol"), 1) or 1)

    def _axis(v, n, default):
        if isinstance(v, dict):
            v = v.get("value")
        if isinstance(v, (list, tuple, np.ndarray)):
            a = np.asarray(v, dtype=float).ravel()
            if len(a) >= n:
                return a[:n]
            if len(a):
                return np.concatenate([a, np.full(n - len(a), a[-1])])
        try:
            return np.full(n, float(v))
        except (TypeError, ValueError):
            return np.full(n, float(default))

    dis = {
        "nlay": nlay,
        "nrow": nrow,
        "ncol": ncol,
        "delr": _axis(grid.get("delr"), ncol, 20.0),
        "delc": _axis(grid.get("delc"), nrow, 20.0),
        "top": _as_float(grid.get("top"), 100.0),
        "botm": _resolve_botm(grid),
    }
    assert len(dis["botm"]) == nlay, (
        f"botm 长度 {len(dis['botm'])} != nlay {nlay}（网格归一化失败）"
    )
    return dis


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
        self._json_ir = None
        # 归一化后的网格定义，GWF/GWT 共用，避免两者几何分叉
        self._dis = None
        # 由水流边界包产出的 SSM 源描述符 [(pname, srctype, auxname)]
        self._ssm_sources = []
        # 已创建的水流包名与各自声明的辅助变量，用于校验 SSM 引用
        self._built_pkgs = {}
        # 溶质运移已启用但没有任何浓度源时置位，供前端提示
        self._transport_degraded = False
        # 建模拟过程中产生的、用户应当看见的提示。**不能只写 logger** ——
        # 早先几处 `logger.warning`（全域展开、无浓度源降级）都到不了界面，
        # 用户只看到结果不对、看不到原因。这里攒起来，由 `build_and_run`
        # 的返回值带出去，前端渲染到结果面板。
        self._build_warnings: List[str] = []

    # ================================================================
    # 主入口：一键构建完整模拟
    # ================================================================

    def _warn(self, msg: str) -> None:
        """记录一条**用户应当看见**的构建提示。

        与裸 `logger.warning` 的区别是这个会随 `build_and_run` 的返回值流到
        前端结果面板。凡是"用户不改就得不到预期模型"的提示都该走这里。
        """
        self._build_warnings.append(msg)
        logger.warning(msg)

    def build_simulation(self, json_ir: dict, sim_name: str = "auto_model") -> MFSimulation:
        """
        从 JSON IR 一键构建完整 MODFLOW 6 模拟对象

        Returns:
            flopy.mf6.MFSimulation — 已写入输入文件的完整模拟对象
        """
        logger.info(f"开始构建模型: {sim_name}")
        self._json_ir = json_ir

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

        # 归一化网格：GWF/GWT 共用同一份 DIS，杜绝几何分叉
        self._dis = _resolve_dis(mapped["grid"])
        self._ssm_sources = []
        self._built_pkgs = {}
        self._transport_degraded = False
        self._build_warnings = []
        logger.info(
            f"DIS 归一化: nlay={self._dis['nlay']}, nrow={self._dis['nrow']}, "
            f"ncol={self._dis['ncol']}, top={self._dis['top']:g}, botm={self._dis['botm']}"
        )

        # ── 诊断日志：检查 transport 启用状态 ──
        transport_raw = json_ir.get("transport", {})
        project_raw = json_ir.get("project", {})
        logger.info(
            f"TRANSPORT DIAG: enabled={transport_raw.get('enabled')}, "
            f"model_type={project_raw.get('model_type')}, "
            f"transport_keys={list(transport_raw.keys()) if transport_raw else 'EMPTY'}, "
            f"mapped_enabled={mapped.get('transport', {}).get('enabled')}"
        )

        # Step 3: 创建 Simulation
        self._sim = MFSimulation(
            sim_name=sim_name,
            version="mf6",
            # 允许通过环境变量指定 mf6 可执行文件；从 .bat 启动时
            # PATH 里可能没有 MODFLOW 的 bin 目录
            exe_name=os.environ.get("MF6_EXE", "mf6"),
            sim_ws=str(self.workspace / sim_name),
        )

        # Step 4: 时间离散 (TDIS)
        time_params = mapped["time"]
        # ── Courant 保护必须在 TDIS 落盘之前 ──
        # MF6 6.3.0 的 ATS 不做 Courant 控制（Timing/ats.f90 把它列为未完成项），
        # 所以 Δt 只能自己估；否则溶质前沿会被数值弥散抹平，甚至出现负浓度。
        if mapped.get("transport", {}).get("enabled"):
            self._ensure_transport_timesteps(mapped, time_params)
        self._build_tdis(time_params)

        # Step 5: 水流模型 (GWF)
        self._gwf = self._build_gwf(sim_name, mapped)

        # Step 6: (条件) 溶质运移模型 (GWT)
        # 多级检测：mapped 结果 > project.model_type > transport 是否有任何内容
        transport = mapped.get("transport", {})
        project_mt = json_ir.get("project", {}).get("model_type", "")
        transport_has_content = bool(
            transport_raw and any(
                k for k in transport_raw.keys()
                if k not in ("enabled", "ir_version") and transport_raw.get(k)
            )
        )
        should_build = (
            transport.get("enabled")
            or project_mt in ("solute_transport", "reactive_transport")
            or transport_has_content
        )
        logger.info(
            f"GWT 决策: enabled={transport.get('enabled')}, "
            f"model_type='{project_mt}', has_content={transport_has_content} "
            f"→ should_build={should_build}"
        )

        if should_build:
            # 确保 mapped["transport"] 至少有 enabled 标记
            if not transport.get("enabled"):
                transport["enabled"] = True
            self._gwt = self._build_gwt(sim_name, mapped)
            logger.info("✓ GWT 溶质运移模型已构建")

        # Step 7: 求解器 (IMS)
        self._build_solver(mapped["solver"])

        # Step 8: GWF-GWT 交换
        if self._gwt is not None:
            self._build_exchange(sim_name)

        # Step 9: 写入输入文件
        ws = self.workspace / sim_name
        os.makedirs(str(ws), exist_ok=True)
        self._sim.write_simulation(silent=True)

        # Step 10: 网格一致性断言
        # MF6 的 GWF-GWT 交换只比较节点数、不校验 top/botm，几何不一致会
        # **静默**算出错误的物理场。这里读回写好的输入文件逐行比对。
        if self._gwt is not None:
            self._assert_grid_consistency(ws, sim_name)

        logger.info(f"模型构建完成: {sim_name}")
        return self._sim

    # ================================================================
    # 运行：自己调 mf6，不走 flopy 的 runner
    # ================================================================

    def _resolve_exe(self) -> str:
        """解析 mf6 可执行文件路径"""
        exe = os.environ.get("MF6_EXE") or "mf6"
        found = shutil.which(exe)
        if found:
            return found
        # 环境变量给的是绝对路径、或 PATH 里没有时，原样返回让 subprocess 报错
        return exe

    def _run_mf6(self, ws: Path, namefile: str = "mfsim.nam"):
        """直接调用 mf6 可执行文件，返回 (success, stdout_text)

        为什么不直接用 flopy 的 `sim.run_simulation()`：
        flopy 的 `mbase.run_model` 里写死了
            line = proc.stdout.readline().decode("utf-8")
        而 Windows 上的 mf6 6.3.0 会把**包含中文的路径按 GBK 写进 stdout**
        （例如出错时回显 `F:\\...\\环科所项目\\2026所基金\\...`）。
        只要路径里有中文，这一行就抛 UnicodeDecodeError，把"模型其实跑完了"
        误判成"程序崩溃"。本项目的工作目录恰好带中文，所以这条路必须自己走。

        另外，mf6 发现输出目录里已有旧的 .lst 时会**交互式**询问是否覆盖，
        无人值守时会卡住或误判；这里先删掉所有旧的 .lst 再跑。
        """
        # 清掉旧的列表文件，避免 mf6 的交互式覆盖询问
        for old in ws.glob("*.lst"):
            try:
                old.unlink()
            except OSError:
                pass

        exe = self._resolve_exe()
        logger.info(f"运行 mf6: {exe} (cwd={ws})")
        try:
            proc = subprocess.run(
                [exe, namefile],
                cwd=str(ws),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=3600,
            )
        except FileNotFoundError:
            msg = (
                f"找不到 mf6 可执行文件 '{exe}'。请把它加入 PATH，"
                f"或设置环境变量 MF6_EXE 指向可执行文件的完整路径。"
            )
            logger.error(msg)
            return False, msg
        except subprocess.TimeoutExpired:
            logger.error("mf6 运行超时（>1 小时）")
            return False, "mf6 运行超时"

        text = proc.stdout.decode("utf-8", errors="replace")
        # mf6 的收尾信息同时出现在 stdout 和 .lst 里。优先信 .lst——
        # stdout 重定向到管道时可能因缓冲丢掉尾部，而 .lst 一定写全。
        lst = ws / (Path(namefile).stem + ".lst")
        if lst.exists():
            listing = lst.read_text(encoding="utf-8", errors="replace")
            success = "normal termination" in listing.lower()
        else:
            success = "normal termination" in text.lower()

        if not success:
            logger.error(f"mf6 未正常结束 (returncode={proc.returncode})")
        return success, text

    @staticmethod
    def _griddata_block(dis_path) -> list:
        """取出 .dis 文件中 griddata 段的非注释行（用于逐行比对）"""
        lines, in_block = [], False
        with open(dis_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if s.upper().startswith("BEGIN GRIDDATA"):
                    in_block = True
                    continue
                if s.upper().startswith("END GRIDDATA"):
                    break
                if in_block and s and not s.startswith("#"):
                    lines.append(" ".join(s.split()))
        return lines

    def _assert_grid_consistency(self, ws, sim_name: str):
        """断言 GWF 与 GWT 的网格几何逐行一致

        MF6 不校验这件事，所以必须自己拦。不一致时立刻失败，避免拿一个
        静默错误的浓度场去做后续分析。
        """
        gwf_dis = ws / f"{sim_name}_gwf.dis"
        gwt_dis = ws / f"{sim_name}_gwt.dis"
        if not (gwf_dis.exists() and gwt_dis.exists()):
            logger.warning("找不到 .dis 文件，跳过网格一致性检查")
            return

        a = self._griddata_block(gwf_dis)
        b = self._griddata_block(gwt_dis)

        detail = (
            f"\n  GWF griddata: {a}\n  GWT griddata: {b}"
        )
        if len(a) != len(b):
            raise ValueError(f"GWF/GWT 网格几何不一致（行数不同）{detail}")
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                raise ValueError(f"GWF/GWT 网格几何不一致（第 {i + 1} 行）{detail}")

        logger.info(f"✓ 网格一致性检查通过（griddata {len(a)} 行逐行一致）")

    def _needs_grid_generation(self, json_ir: dict) -> bool:
        """判断是否需要自动生成网格"""
        grid = json_ir.get("grid", {})
        return not grid.get("nrow") or not grid.get("ncol")

    # ================================================================
    # TDIS
    # ================================================================

    # ================================================================
    # Courant 时间步长保护
    # ================================================================

    #: 各水流包 stress_period_data 记录中"水头/水位"所在的列（0 起算）
    #:   chd  [cellid, head, ...]
    #:   ghb  [cellid, head, cond, ...]
    #:   riv  [cellid, stage, cond, rbot, ...]
    #: wel / rch / drn / evt 的记录里没有水头，不参与梯度估算。
    _HEAD_COLUMN = {"chd": 1, "ghb": 1, "riv": 1}

    @classmethod
    def _collect_boundary_heads(cls, mapped: dict) -> list:
        """把 mapped['boundaries'] 里的水头值收集起来

        `map_boundary_conditions` 的实际结构是
            {包名: {应力期: [[cellid, 水头, ...], ...]}}
        记录**按列位置**排列而非键值，所以只能按包类型取固定列
        （见 _HEAD_COLUMN）。同时兼容 dict 记录形态，便于将来换写法。
        """
        heads = []
        for pkg, periods in (mapped.get("boundaries") or {}).items():
            col = cls._HEAD_COLUMN.get(str(pkg).lower())
            if col is None:
                continue
            groups = periods.values() if isinstance(periods, dict) else [periods]
            for group in groups:
                if not isinstance(group, (list, tuple)):
                    continue
                for rec in group:
                    if isinstance(rec, dict):
                        for k in ("head", "stage"):
                            if isinstance(rec.get(k), (int, float)):
                                heads.append(float(rec[k]))
                                break
                    elif isinstance(rec, (list, tuple)) and len(rec) > col:
                        try:
                            heads.append(float(rec[col]))
                        except (TypeError, ValueError):
                            pass
        return heads

    def _reference_velocity(self, mapped: dict):
        """由水头梯度粗估参考流速 v = K·i/ne (m/d)

        返回 (v, 说明)；估不出来时返回 (None, 原因)。

        ── 为什么是"粗估"就够 ──
        这个 v 只用来算 Δt 的上限，进而决定 nstp。估偏小 → Δt 偏大 → 分辨率
        不足（可接受，多跑几轮总能发现）；估偏大 → nstp 白白翻几倍（只是慢）。
        真正的风险是**估小 500 倍**：若把梯度路径长度取成域的最短边，Case A
        那种 nrow=1 / ncol=500 的细长柱体会算出 i = 1/0.2 的荒谬梯度。所以
        这里刻意取**最长边**作为梯度路径——宁可多算时间步，也不放过局部高速。
        """
        aqua = mapped.get("aquifer", {}) or {}
        hk = aqua.get("hk")
        if isinstance(hk, (list, tuple, np.ndarray)):
            arr = np.asarray(hk, dtype=float)
            hk = float(arr.mean()) if arr.size else None
        if hk is None:
            return None, "含水层缺少 hk"
        hk = float(hk)
        if hk <= 0:
            return None, f"hk={hk:g} 非正"

        heads = self._collect_boundary_heads(mapped)
        if len(heads) < 2:
            return None, f"边界水头不足（找到 {len(heads)} 个，至少需要 2 个）"
        dh = max(heads) - min(heads)
        if dh <= 0:
            return None, "所有边界水头相同，无驱动梯度"

        d = self._dis
        lx = float(np.sum(np.asarray(d["delc"], dtype=float)))
        ly = float(np.sum(np.asarray(d["delr"], dtype=float)))
        length = max(lx, ly)
        if length <= 0:
            return None, "域长度为 0"

        ne = (mapped.get("transport", {}) or {}).get("porosity")
        if ne is None:
            ne = aqua.get("porosity")
        try:
            ne = float(ne) if ne is not None else 0.25
        except (TypeError, ValueError):
            ne = 0.25
        if ne <= 0:
            logger.warning(f"孔隙度 {ne:g} 非正，按 0.25 估算参考流速")
            ne = 0.25

        v = hk * (dh / length) / ne
        return v, (
            f"hk={hk:g}, Δh={dh:g}, L={length:g} (max(Lx,Ly)), "
            f"ne={ne:g} → v≈{v:.6g} m/d"
        )

    def _ensure_transport_timesteps(self, mapped: dict, time_params: dict) -> None:
        """按 Courant 数反推时间步数，就地提升 nstp

        Δt_max = Cr_target · min(Δx) / v，再按每个应力期的 perlen 换算所需步数。
        只升不降（`max(nstp, need)`），并且：
          · 下限 10 —— 低于 10 步根本画不出穿透曲线；
          · 上限 20000 —— 防止粗估出的巨大步数把单次运行拖死。
        """
        per_data = time_params.get("period_data")
        if not per_data:
            return

        v, why = self._reference_velocity(mapped)
        if not v or v <= 0:
            logger.warning(f"无法估算参考流速（{why}），跳过 Courant 时间步长保护")
            return

        try:
            cr_target = float(
                (mapped.get("transport", {}) or {}).get(
                    "timestep_target_courant", 1.0
                )
                or 1.0
            )
        except (TypeError, ValueError):
            cr_target = 1.0
        if cr_target <= 0:
            cr_target = 1.0

        d = self._dis
        dx = min(
            float(np.min(np.asarray(d["delr"], dtype=float))),
            float(np.min(np.asarray(d["delc"], dtype=float))),
        )
        dt_max = cr_target * dx / v
        if dt_max <= 0:
            logger.warning(f"Δt_max={dt_max:g} 非正，跳过 Courant 时间步长保护")
            return

        logger.info(
            f"Courant 保护: {why}; Cr_target={cr_target:g}, "
            f"min(Δx)={dx:g} → Δt_max={dt_max:g} d"
        )

        changes = []
        for i, pd in enumerate(per_data):
            perlen = float(pd.get("perlen", 1.0) or 1.0)
            nstp = int(pd.get("nstp", 1) or 1)
            need = int(np.ceil(perlen / dt_max)) if perlen > 0 else nstp
            need = max(10, min(20000, max(nstp, need)))
            if need != nstp:
                changes.append(f"期{i + 1}: {nstp}→{need} (Δt={perlen / need:g} d)")
                pd["nstp"] = need

        if changes:
            logger.warning(
                "Courant 保护提升了时间步数 —— " + "; ".join(changes)
            )
        else:
            logger.info("Courant 保护: 现有 nstp 已满足，未改动")

    def _build_tdis(self, time_params: dict):
        """构建时间离散包"""
        per_data = time_params.get("period_data", [{"perlen": 1.0, "nstp": 1, "tsmult": 1.0}])

        perioddata = []
        for pd in per_data:
            perioddata.append((
                float(pd.get("perlen", 1.0) or 1.0),
                int(pd.get("nstp", 1) or 1),
                float(pd.get("tsmult", 1.0) or 1.0),
            ))

        if not perioddata:
            raise ValueError("时间离散为空：period_data 至少需要一个应力期")

        # nper 与 period_data 必须一致，否则 MF6 读 TDIS 时会错位
        declared_nper = time_params.get("nper")
        if declared_nper is not None and int(declared_nper) != len(perioddata):
            logger.warning(
                f"TDIS nper({declared_nper}) 与 period_data 长度({len(perioddata)}) 不一致，"
                f"以 period_data 为准"
            )

        total = sum(p[0] for p in perioddata)
        logger.info(
            f"TDIS: {len(perioddata)} 个应力期, 总时长 {total:g} d, "
            f"总时间步 {sum(p[1] for p in perioddata)}"
        )

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

        # DIS: 离散包（消费归一化后的统一网格，与 GWT 共用）
        dis = self._dis
        nlay = dis["nlay"]
        delr = dis["delr"]
        delc = dis["delc"]
        top_val = dis["top"]
        botm = dis["botm"]

        flopy.mf6.ModflowGwfdis(
            gwf,
            nrow=dis["nrow"],
            ncol=dis["ncol"],
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
        nper = int(mapped["time"].get("nper", 1) or 1)
        is_transient = not bool(mapped["time"].get("steady_state", False))
        # transient 必须逐应力期给出：只写 {0: ...} 时 MF6 会把其余应力期
        # 按稳态处理（STO 未指定的 period 默认稳态）
        transient = {i: is_transient for i in range(nper)}

        flopy.mf6.ModflowGwfsto(
            gwf,
            ss=ss,
            sy=sy,
            # iconvert 需与 NPF 的 icelltype 一致，否则潜水/可转换类型的
            # 储水转换（SY↔SS）失效
            iconvert=icelltype,
            transient=transient,
        )

        # IC: 初始条件（逐层取层中点；统一写 top 会让上层初始水头高于下层顶板）
        strt = self._initial_head(top_val, botm)
        flopy.mf6.ModflowGwfic(gwf, strt=strt)

        # 边界条件包（返回 SSM 源描述符，供 GWT 的 SSM 包引用）
        self._ssm_sources = self._build_boundary_packages(
            gwf, mapped["boundaries"], self._dis
        )

        # OC: 输出控制
        flopy.mf6.ModflowGwfoc(
            gwf,
            budget_filerecord=f"{sim_name}_gwf.bud",
            head_filerecord=f"{sim_name}_gwf.hds",
            saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        )

        return gwf

    def _initial_head(self, top: float, botm: list):
        """初始水头

        默认沿用 `strt = top`（与既有行为一致，回归测试依赖此约定）。
        只有 IR 里显式给了 `initial_conditions.head` 才覆盖：
          · 标量 → 全场统一值
          · 列表 → 逐层值（长度需等于 nlay）
        """
        ic = (self._json_ir or {}).get("initial_conditions") or {}
        raw = ic.get("head") if isinstance(ic, dict) else None
        if isinstance(raw, dict):
            raw = raw.get("value")
        if raw is None:
            return top

        values = np.atleast_1d(np.asarray(raw, dtype=float))
        if values.size == 1:
            return float(values[0])
        nlay = len(botm)
        if values.size != nlay:
            logger.warning(
                f"initial_conditions.head 有 {values.size} 个值，与 nlay={nlay} 不符，"
                f"回退为 top"
            )
            return top
        return values.tolist()

    def _build_boundary_packages(self, gwf, boundary_map: dict, grid: dict):
        """构建水流边界条件包

        Returns:
            SSM 源描述符列表 ``[(pname, srctype, auxname)]``，供 GWT 的 SSM 包引用。

        只有携带了浓度的包才会声明 ``AUXILIARY CONCENTRATION`` 并出现在返回值中。
        MF6 要求辅助变量真实存在于对应水流包里，否则报
        ``AUXILIARY NAME CANNOT BE FOUND``。
        """
        nrow, ncol = grid["nrow"], grid["ncol"]
        ssm_sources = []

        for pkg_name, sp_data in boundary_map.items():
            builder = PKG_BUILDERS.get(pkg_name)
            if builder is None or not sp_data:
                continue

            base = PKG_BASE_COLS[pkg_name]
            spd, has_conc = self._normalize_boundary_spd(
                pkg_name, sp_data, base, nrow, ncol
            )
            if not spd:
                continue

            pname = BOUNDARY_PKG_NAME[pkg_name]
            kwargs = {"pname": pname, "stress_period_data": spd}
            if has_conc:
                kwargs["auxiliary"] = [AUX_CONC]
            builder(gwf, **kwargs)

            self._built_pkgs[pname] = [AUX_CONC] if has_conc else []
            if has_conc:
                srctype = "AUXMIXED" if pkg_name in AUXMIXED_PKGS else "AUX"
                ssm_sources.append((pname, srctype, AUX_CONC))
                logger.info(
                    f"{pkg_name.upper()} 包已声明 AUXILIARY {AUX_CONC}"
                    f"（srctype={srctype}），可作为 SSM 源"
                )
            else:
                logger.info(f"{pkg_name.upper()} 包已创建: {len(spd)} 个应力期")

        return ssm_sources

    def _normalize_boundary_spd(self, pkg_name, sp_data, base, nrow, ncol):
        """把边界记录统一成 ``(cellid, *values[, conc])`` 形式并补齐浓度列

        记录有两种写法：嵌套的 ``[(l,r,c), v...]`` 与扁平的 ``(l,r,c,v...)``
        （RCH 用的是后者），这里统一成嵌套形式。

        只要任一记录带浓度，就为**所有**记录补齐浓度列 —— MF6 按辅助变量
        个数读取列表，列数不齐会读错位。
        """
        if isinstance(sp_data, (int, float)):
            return ({0: float(sp_data)}, False) if pkg_name == "rch" else ({}, False)
        if not isinstance(sp_data, dict):
            return {}, False

        # 第一遍：解析记录，判断是否存在浓度列
        parsed, scalar_periods, has_conc = {}, {}, False
        for per, data in sp_data.items():
            per = int(per)
            if isinstance(data, (int, float)):
                scalar_periods[per] = float(data)   # 均匀形式（仅 RCH 合法）
                continue
            if not isinstance(data, (list, tuple)):
                continue
            recs = []
            for r in data:
                if not isinstance(r, (list, tuple)) or len(r) < 2:
                    continue
                if isinstance(r[0], (list, tuple)):
                    cellid, values = tuple(r[0]), list(r[1:])
                else:
                    cellid, values = tuple(r[:3]), list(r[3:])
                recs.append((cellid, values))
                if len(values) >= base + 1:
                    has_conc = True
            if recs:
                parsed[per] = recs

        if not parsed:
            return dict(scalar_periods), False

        if has_conc and scalar_periods:
            # 声明 AUXILIARY 后就不能再用均匀标量形式（MF6 按 naux 列数读列表），
            # 把标量应力期展开成 layer0 全网格。展开本身是**语义等价**的
            # ——标量补给按定义就是全域均匀——但格数会从 1 变成 nrow×ncol，
            # 且这些格子带的是浓度 0（标量形式本就没有浓度），用户必须知道，
            # 否则会以为"只在一个期加了浓度，怎么整个域都在补给"。
            self._warn(
                f"{pkg_name.upper()} 同时存在均匀补给（标量）与逐格浓度记录："
                f"MF6 声明 AUXILIARY 后不能再混用标量形式，已将 "
                f"{len(scalar_periods)} 个均匀应力期展开为 layer0 全网格"
                f"（{nrow}×{ncol} 格），展开后的浓度取 0。"
            )
            for per, rate in scalar_periods.items():
                parsed[per] = [
                    ((0, r, c), [rate, 0.0]) for r in range(nrow) for c in range(ncol)
                ]

        # 第二遍：按需补齐 / 截断到 base(+1) 列
        width = base + 1 if has_conc else base
        spd = {} if has_conc else dict(scalar_periods)
        for per, recs in parsed.items():
            norm = []
            for cellid, values in recs:
                v = (values + [0.0] * width)[:width]
                norm.append((cellid, *v))
            if norm:
                spd[per] = norm
        return spd, has_conc

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

        # GWT DIS：必须与 GWF 使用完全相同的网格几何。
        # MF6 的 GWF-GWT 交换只比较节点数、不校验 top/botm，
        # 几何不一致会静默算出错误的物理场，所以两者共用 self._dis。
        dis = self._dis
        flopy.mf6.ModflowGwtdis(
            gwt,
            nrow=dis["nrow"],
            ncol=dis["ncol"],
            nlay=dis["nlay"],
            delr=dis["delr"],
            delc=dis["delc"],
            top=dis["top"],
            botm=dis["botm"],
        )

        # IC: 初始浓度
        ic_conc = transport.get("initial_concentration", 0.0)
        flopy.mf6.ModflowGwtic(gwt, strt=ic_conc)

        # ADV: 对流包（scheme 已在 mapper 中白名单化）
        scheme = transport.get("advection_scheme", "UPSTREAM")
        flopy.mf6.ModflowGwtadv(gwt, scheme=scheme)

        # MST: 移动存储与转化包（MODFLOW 6 GWT 必选，缺了会直接报
        # "Mass storage and transfer (MST6) package not specified"）
        # 孔隙度优先从 transport 取，其次从 aquifer 取，否则默认 0.3
        porosity = transport.get("porosity") or mapped.get("aquifer", {}).get("porosity")
        if porosity is None:
            porosity = 0.3
            logger.warning("未指定孔隙度，GWT MST 使用默认值 0.3")
        porosity = float(porosity)
        if not 0.0 < porosity <= 1.0:
            raise ValueError(f"孔隙度必须在 (0, 1] 区间内，当前为 {porosity}")
        flopy.mf6.ModflowGwtmst(gwt, porosity=porosity)
        logger.info(f"GWT MST 孔隙度: {porosity}")

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

        # SSM: 源汇项浓度注入
        #
        # FloPy 的 ModflowGwtssm(sources=...) 接受的是**引用描述符**
        # [(pname, srctype, auxname)]，不是逐单元格浓度值：
        #   pname   — 水流包名（如 "RCHA"），需与实际创建的一致
        #   srctype — "AUX" 或 "AUXMIXED"
        #   auxname — 该水流包中辅助变量的名字，必须真实存在且大小写一致
        # 浓度值本身由水流包的 AUXILIARY 列携带（见 _build_boundary_packages）。
        sources = list(self._ssm_sources)
        sources.extend(self._resolve_explicit_sources(mapped.get("transport", {})))

        flopy.mf6.ModflowGwtssm(gwt, sources=sources or None)
        if sources:
            for pname, srctype, auxname in sources:
                logger.info(f"GWT SSM 源: {pname} / {srctype} / {auxname}")
        else:
            logger.warning("GWT SSM 无任何源：未在边界条件中指定浓度")

        # OC
        flopy.mf6.ModflowGwtoc(
            gwt,
            budget_filerecord=f"{sim_name}_gwt.bud",
            concentration_filerecord=f"{sim_name}_gwt.ucn",
            saverecord=[("CONCENTRATION", "ALL"), ("BUDGET", "LAST")],
        )

        # 降级提示：溶质运移已启用，但既无浓度源、初始浓度也为 0，
        # 浓度场会恒为 0。这里不抛异常（无源建 GWT 是合法的回归用例），
        # 只置标志供前端提示。
        init_conc = ic_conc if isinstance(ic_conc, (int, float)) else 0.0
        if not sources and not init_conc:
            self._transport_degraded = True
            self._warn(
                "溶质运移已启用，但既没有浓度源、初始浓度也为 0，浓度场将恒为 0。"
                "请在边界条件（RCH/WEL/CHD 等）中指定注入浓度。"
                + self._source_concentration_hint()
            )

        return gwt

    def _resolve_explicit_sources(self, transport: dict) -> list:
        """解析 IR 中显式声明的 SSM 源，并校验其引用的包与辅助变量真实存在

        形如 ``[{"package": "rch", "srctype": "AUXMIXED", "auxname": "CONCENTRATION"}]``。
        校验的目的是把 MF6 运行期的 "AUXILIARY NAME CANNOT BE FOUND"
        提前到 Python 层，给出可读的错误。
        """
        raw = transport.get("sources") or transport.get("ssm_sources") or []
        if not isinstance(raw, list):
            return []

        resolved = []
        for src in raw:
            if not isinstance(src, dict):
                continue
            pkg = str(src.get("package", "")).lower()
            if not pkg:
                continue
            pname = BOUNDARY_PKG_NAME.get(pkg, pkg.upper())
            srctype = str(src.get("srctype", "AUX")).upper()
            auxname = str(src.get("auxname", AUX_CONC)).upper()

            if pname not in self._built_pkgs:
                logger.warning(f"SSM 源引用了不存在的水流包 '{pname}'，已跳过")
                continue
            if auxname not in self._built_pkgs[pname]:
                logger.warning(
                    f"SSM 源引用的辅助变量 '{auxname}' 未在 {pname} 包中声明，已跳过"
                )
                continue
            if srctype not in ("AUX", "AUXMIXED"):
                logger.warning(f"未知的 SSM srctype '{srctype}'，回退为 AUX")
                srctype = "AUX"
            resolved.append((pname, srctype, auxname))

        return resolved

    def _source_concentration_hint(self) -> str:
        """IR 里提到过源浓度却没落到具体包上时，给出可操作的提示"""
        try:
            aq = (self._json_ir or {}).get("aquifer", {}).get("parameters", {})
            raw = aq.get("source_concentration")
            if isinstance(raw, dict):
                raw = raw.get("value")
            if raw is not None:
                return (
                    f"（检测到已指定源浓度 {float(raw):g}，但未指定注入位置，"
                    f"请在对话中补充注入的边界条件）"
                )
        except (TypeError, ValueError):
            pass
        return ""

    # ================================================================
    # 求解器 & 交换
    # ================================================================

    def _build_solver(self, solver_params: dict):
        """构建求解器

        IMS 收敛准则：inner_dvclose 必须 ≦ outer_dvclose，否则线性求解器
        会因过于宽松而提前退出，导致非线性外迭代无法收敛。
        MODFLOW 6 官方建议 inner = outer / 10 ~ outer / 100。

        对流格式非 UPSTREAM 时，GWT 的系数矩阵是**非对称**的，MF6 会直接
        拒绝 CG 并中止：
            "MODEL AUTO_MODEL_GWT PRODUCES AN ASYMMETRIC COEFFICIENT
             MATRIX, BUT THE CONJUGATE GRADIENT METHOD WAS SELECTED.
             USE BICGSTAB INSTEAD."
        由于 GWF 与 GWT 共用同一个 solution group（一份 .ims），这里只能整组
        改用 BICGSTAB —— 它对对称与非对称矩阵都收敛。纯粹的 GWF 模型保持
        用户指定的求解器不变。
        """
        outer_dvclose = solver_params.get("hclose", 1e-5)
        # 如果用户未指定 rclose，默认设为 outer 的 1/10
        rclose = solver_params.get("rclose")
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

        accel = str(solver_params.get("name", "CG")).upper()
        if self._gwt is not None and accel == "CG":
            logger.warning(
                "溶质运移启用时 GWT 矩阵非对称，CG 会被 MF6 拒绝，"
                "线性加速改为 BICGSTAB"
            )
            accel = "BICGSTAB"

        ModflowIms(
            self._sim,
            print_option="summary",
            outer_dvclose=outer_dvclose,
            inner_dvclose=inner_dvclose,
            linear_acceleration=accel,
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
        # 不用 sim.run_simulation()：flopy 对 mf6 的 stdout 硬编码 utf-8 解码，
        # 项目路径含中文时会把"能跑"误判成"崩溃"。详见 _run_mf6 的说明。
        success, _ = self._run_mf6(self.workspace / sim_name)

        if not success:
            # 只回一句"模拟运行失败"会让用户无从下手，
            # 把 mfsim.lst 里的 ERROR REPORT 原文带出来
            return {
                "success": False,
                "error": "模拟运行失败",
                "detail": self._extract_error_report(self.workspace / sim_name / "mfsim.lst"),
                # 失败前攒下的提示照样带出去：很可能正是其中一条解释了失败原因
                "build_warnings": list(self._build_warnings),
            }

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
            "transport_degraded": self._transport_degraded,
            # 构建期攒下的、用户应当看见的提示（原先只有 logger，到不了界面）
            "build_warnings": list(self._build_warnings),
            "sim": sim,
            "gwf": self._gwf,
            "gwt": self._gwt,
        }

    @staticmethod
    def _extract_error_report(lst_path) -> str:
        """从 mfsim.lst 中截取 ERROR REPORT / UNIT ERROR REPORT 段"""
        try:
            with open(lst_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except OSError:
            return ""

        out, capturing = [], False
        for line in lines:
            s = line.strip()
            if "ERROR REPORT" in s.upper():
                capturing = True
            if capturing:
                out.append(line.rstrip())
                if len(out) > 40:
                    break
        return "\n".join(out).strip()


# 全局实例
auto_builder = AutoModelBuilder()

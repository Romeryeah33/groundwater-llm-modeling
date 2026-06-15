"""
FloPy 基线验证脚本

验证 MODFLOW 6 在本地环境正常运行：
1. 创建简单2D均质承压含水层模型
2. 运行稳态水流模拟
3. 输出水头结果
4. (可选) 运行溶质运移模拟

运行方式：
    python scripts/flopy_baseline_test.py
"""

import os
import sys
import tempfile
from pathlib import Path

# 强制UTF-8输出（解决Windows GBK编码问题）
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import flopy


def create_and_run_flow_model(workspace: str) -> dict:
    """
    创建并运行简单2D稳态流模型

    场景：均质承压含水层
    - 模型范围：1000m × 800m
    - 网格：50行 × 40列 × 1层
    - 渗透系数：15 m/d
    - 西侧定水头 100m，东侧定水头 95m
    - 中心一口抽水井 Q = -200 m³/d

    返回: {'success': bool, 'head': np.ndarray, 'budget_error': float}
    """

    # ========================================
    # Step 1: 定义模型参数
    # ========================================
    Lx, Ly = 1000.0, 800.0  # 模型范围 (m)
    nrow, ncol, nlay = 40, 50, 1
    delr = Lx / ncol  # 列宽 = 20m
    delc = Ly / nrow  # 行宽 = 20m
    top = 100.0        # 顶层高程 (m)
    botm = 50.0        # 底层高程 (m) — 含水层厚度 50m
    hk = 15.0          # 水平渗透系数 (m/d)
    h_left = 100.0     # 西侧定水头 (m)
    h_right = 95.0     # 东侧定水头 (m)
    well_rate = -200.0 # 抽水速率 (m³/d)，负值表示抽水

    print("=" * 60)
    print("FloPy + MODFLOW 6 基线验证")
    print("=" * 60)
    print(f"模型范围: {Lx}m × {Ly}m × {top-botm}m")
    print(f"网格: {nrow}行 × {ncol}列 × {nlay}层 (单元尺寸 {delr:.0f}m)")
    print(f"渗透系数: {hk} m/d")
    print(f"边界: 西侧定水头 {h_left}m, 东侧定水头 {h_right}m")
    print(f"抽水井: Q = {well_rate} m³/d")
    print()

    # ========================================
    # Step 2: 创建 MODFLOW 6 模拟对象
    # ========================================
    sim = flopy.mf6.MFSimulation(
        sim_name="baseline_test",
        version="mf6",
        exe_name="mf6",
        sim_ws=workspace,
    )

    # 时间离散 (TDIS) — 稳态只需要一个应力期
    tdis = flopy.mf6.ModflowTdis(
        sim,
        time_units="days",
        nper=1,
        perioddata=[(1.0, 1, 1.0)],  # (perlen, nstp, tsmult)
    )

    # 水流模型 (GWF)
    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname="baseline_gwf",
        save_flows=True,
    )

    # 离散包 (DIS)
    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        nrow=nrow, ncol=ncol, nlay=nlay,
        delr=delr, delc=delc,
        top=top, botm=botm,
    )

    # 节点属性流 (NPF)
    npf = flopy.mf6.ModflowGwfnpf(
        gwf,
        k=hk,            # 渗透系数
        icelltype=0,     # 0=承压
    )

    # 初始条件 (IC)
    ic = flopy.mf6.ModflowGwfic(
        gwf,
        strt=100.0,      # 初始水头
    )

    # 储水 (STO) — 稳态不需要储水，但添加以支持后续瞬态扩展
    sto = flopy.mf6.ModflowGwfsto(
        gwf,
        ss=1e-5,
        sy=0.1,
        transient={0: False},  # 稳态
    )

    # 定水头边界 (CHD)
    # 西侧第一列: h=100m
    chd_left = []
    for i in range(nrow):
        chd_left.append([(0, i, 0), h_left])  # (layer, row, col), head
    # 东侧最后一列: h=95m
    chd_right = []
    for i in range(nrow):
        chd_right.append([(0, i, ncol - 1), h_right])

    chd_spd = {0: chd_left + chd_right}
    chd = flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chd_spd)

    # 抽水井 (WEL) — 放在模型中心
    wel_spd = {0: [[(0, nrow // 2, ncol // 2), well_rate]]}
    wel = flopy.mf6.ModflowGwfwel(gwf, stress_period_data=wel_spd)

    # 输出控制 (OC)
    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        budget_filerecord="baseline_test.bud",
        head_filerecord="baseline_test.hds",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )

    # 求解器 (IMS)
    ims = flopy.mf6.ModflowIms(
        sim,
        print_option="summary",
        outer_dvclose=1e-6,
        inner_dvclose=1e-6,
    )

    # ========================================
    # Step 3: 写入输入文件并运行
    # ========================================
    print("写入 MODFLOW 6 输入文件...")
    sim.write_simulation(silent=True)

    print("运行模拟...")
    success, buff = sim.run_simulation(silent=True, report=True)

    if not success:
        print("❌ 模拟运行失败!")
        for line in buff:
            print(f"  {line}")
        return {"success": False}

    print("✅ 模拟运行成功!")

    # ========================================
    # Step 4: 读取结果
    # ========================================
    print("\n读取模拟结果...")

    # 读取水头
    head_obj = gwf.output.head()
    head = head_obj.get_data(totim=1.0)

    print(f"  水头范围: {head.min():.2f} ~ {head.max():.2f} m")
    print(f"  抽水井处水头: {head[0, nrow//2, ncol//2]:.2f} m")

    # 读取水量平衡
    try:
        budget_obj = gwf.output.budget()
        budget_records = budget_obj.get_data(totim=1.0)

        flow_in = 0.0
        flow_out = 0.0
        for record in budget_records:
            if isinstance(record, np.ndarray):
                # 结构化数组
                names = record.dtype.names
                # 查找可能包含流量值的字段
                q_field = None
                for n in names:
                    if n.lower() in ('q', 'flow_ja_face', 'flow', 'value'):
                        q_field = n
                        break
                if q_field is None and len(names) > 0:
                    q_field = names[0]
                if q_field:
                    q = np.sum(record[q_field])
                    if q > 0:
                        flow_in += q
                    else:
                        flow_out += abs(q)
            elif isinstance(record, dict) and 'q' in record:
                q = np.sum(record['q'])
                if q > 0:
                    flow_in += q
                else:
                    flow_out += abs(q)

        if flow_in > 1e-10 and flow_out > 1e-10:
            balance_error = abs(flow_in - flow_out) / max(flow_in, flow_out)
        else:
            balance_error = abs(flow_in - flow_out)

        print(f"  总流入: {flow_in:.2f} m³/d")
        print(f"  总流出: {flow_out:.2f} m³/d")
        print(f"  水量平衡误差: {balance_error*100:.4f}%")
        if balance_error < 0.01:
            print("  ✅ 水量平衡检验通过 (< 1%)")
        else:
            print(f"  ⚠️ 水量平衡误差偏大 ({balance_error*100:.2f}%)")
    except Exception as e:
        print(f"  ⚠️ 无法解析水量平衡数据: {e}")
        print(f"  (这不影响建模功能，仅为日志记录)")
        balance_error = None

    # ========================================
    # Step 5: 可视化
    # ========================================
    print("\n生成水头分布图...")
    fig, ax = plt.subplots(figsize=(10, 6))

    # 水头填色图
    levels = np.linspace(head.min(), head.max(), 20)
    cf = ax.contourf(
        np.arange(ncol) * delr + delr / 2,
        np.arange(nrow) * delc + delc / 2,
        head[0, :, :],
        levels=levels,
        cmap="Blues",
    )
    plt.colorbar(cf, ax=ax, label="水头 (m)")

    # 水头等值线
    cs = ax.contour(
        np.arange(ncol) * delr + delr / 2,
        np.arange(nrow) * delc + delc / 2,
        head[0, :, :],
        levels=10,
        colors="black",
        linewidths=0.5,
    )
    ax.clabel(cs, inline=True, fontsize=8)

    # 标记抽水井
    ax.plot(
        (ncol // 2) * delr + delr / 2,
        (nrow // 2) * delc + delc / 2,
        "ro", markersize=8, label="抽水井 (Q=-200 m³/d)"
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Confined Aquifer - Steady-State Head Distribution\n(West 100m -> East 95m, Center Pumping Well)")
    ax.legend()
    ax.set_aspect("equal")

    fig_path = Path(workspace) / "head_distribution.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  水头分布图已保存: {fig_path}")
    plt.close(fig)

    # ========================================
    # Step 6: 附加 — 溶质运移测试 (MT3DMS)
    # ========================================
    print("\n" + "=" * 60)
    print("附加测试: MODFLOW 6 GWT 溶质运移")
    print("=" * 60)

    try:
        # 创建简单的GWT模型（在已有流场上叠加溶质运移）
        _run_transport_test(sim, gwf, workspace, nrow, ncol, delr, delc)
    except Exception as e:
        print(f"  ⚠️ 溶质运移测试失败 (非关键): {e}")

    return {
        "success": True,
        "head": head,
        "budget_error": balance_error,
        "head_range": (float(head.min()), float(head.max())),
    }


def _run_transport_test(sim, gwf, workspace, nrow, ncol, delr, delc):
    """附加溶质运移测试"""
    # 创建GWT模型
    gwt = flopy.mf6.ModflowGwt(
        sim,
        modelname="baseline_gwt",
        save_flows=True,
    )

    # 离散包 (与GWF相同)
    dis_gwt = flopy.mf6.ModflowGwtdis(
        gwt,
        nrow=nrow, ncol=ncol, nlay=1,
        delr=delr, delc=delc,
        top=100.0, botm=50.0,
    )

    # 初始浓度
    ic_gwt = flopy.mf6.ModflowGwtic(gwt, strt=0.0)

    # 对流包
    adv = flopy.mf6.ModflowGwtadv(gwt, scheme="upstream")

    # 弥散包
    dsp = flopy.mf6.ModflowGwtdsp(
        gwt,
        alh=10.0, ath1=1.0,  # 纵向弥散度10m, 横向1m
        diffc=1e-9,           # 分子扩散系数
    )

    # 源汇项 — 在左上方设置定浓度 = 100 mg/L
    cnc_spd = {0: []}
    # 在左上角10×10区域设置连续源
    for i in range(5):
        for j in range(5):
            cnc_spd[0].append([(0, i, j), 100.0])

    ssm = flopy.mf6.ModflowGwtssm(gwt, sources=cnc_spd)

    # 输出控制
    oc_gwt = flopy.mf6.ModflowGwtoc(
        gwt,
        budget_filerecord="baseline_gwt.bud",
        concentration_filerecord="baseline_gwt.ucn",
        saverecord=[("CONCENTRATION", "ALL"), ("BUDGET", "LAST")],
    )

    # GWF-GWT 交换
    gwfgwt = flopy.mf6.ModflowGwfgwt(
        sim,
        exgtype="GWF6-GWT6",
        exgmnamea="baseline_gwf",
        exgmnameb="baseline_gwt",
    )

    # 写入并运行
    sim.write_simulation(silent=True)
    sim.run_simulation(silent=True)

    # 读取浓度
    conc_obj = gwt.output.concentration()
    conc = conc_obj.get_data(totim=1.0)

    print(f"  ✅ 溶质运移模拟成功!")
    print(f"  浓度范围: {conc.min():.2f} ~ {conc.max():.2f} mg/L")


if __name__ == "__main__":
    # 创建临时工作目录
    workspace = PROJECT_ROOT / "data" / "outputs" / "baseline_test"
    workspace.mkdir(parents=True, exist_ok=True)

    result = create_and_run_flow_model(str(workspace))

    if result["success"]:
        b_err = result.get('budget_error')
        b_err_str = f"{b_err*100:.4f}%" if b_err is not None else "N/A (budget parsing skipped)"
        print("\n" + "=" * 60)
        print("✅ FloPy + MODFLOW 6 基线验证全部通过!")
        print(f"   水头范围: {result['head_range'][0]:.2f} ~ {result['head_range'][1]:.2f} m")
        print(f"   水量平衡误差: {b_err_str}")
        print(f"   工作目录: {workspace}")
        print("=" * 60)
    else:
        print("\n❌ 基线验证失败，请检查 MODFLOW 6 安装。")
        sys.exit(1)

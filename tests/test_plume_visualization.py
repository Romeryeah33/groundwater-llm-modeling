"""
污染羽可视化改造 + 主控过程分析的纯函数层测试

覆盖范围（全部不跑 mf6）：
- VisualizationEngine._grid_coords      网格坐标基准（水头图/污染羽图统一后）
- VisualizationEngine._draw_plume_panel 色标范围 / 分层数 / 掩膜下界
- plot_plume_evolution                  总览图：布局、有/无底图、返回值
- plot_plume_at_time                    单帧图：colorbar 与图例均不压主图
- plot_process_analysis                 过程分析图：真实返回值 + 异常分支
- ResultProcessor.extract_velocity_field 流速场形状（与 ProcessIdentifier 复合）
- gradio_app._sample_conc_at            时刻采样/插值的全部分支

运行：python tests/test_plume_visualization.py
（仓库既有测试均为脚本式，这里沿用同一形态：check() + 末尾汇总）
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")          # 无显示环境；必须在 import pyplot 之前
import matplotlib.pyplot as plt
import numpy as np

from app.analysis.visualizer import VisualizationEngine
from app.analysis.process_identifier import ProcessIdentifier
from app.analysis.result_processor import ResultProcessor
from app.frontend.gradio_app import _sample_conc_at, _pick_vmin_vmax

# ============================================================
# 测试框架（与 test_e2e_scenarios.py 一致）
# ============================================================
passed = 0
failed = 0
_failures = []


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        _failures.append(name)
        print(f"  [FAIL] {name}  {detail}")


def section(title):
    print(f"\n{'-' * 60}\n{title}\n{'-' * 60}")


viz = VisualizationEngine()


# ============================================================
# 1. _grid_coords —— 网格坐标基准
# ============================================================
section("1. _grid_coords 网格坐标基准")

# 1.1 无 bounds：与旧的 "cumsum 从 0 起" 逐元素相同
nrow, ncol = 3, 4
delr = np.full(ncol, 20.0)
delc = np.full(nrow, 20.0)
X, Y, ext = viz._grid_coords(nrow, ncol, delr, delc)

old_x = (np.concatenate([[0.0], np.cumsum(delr)])[:-1]
         + np.concatenate([[0.0], np.cumsum(delr)])[1:]) / 2
check("1.1 无 bounds 时 X 等于旧 cumsum 基准",
      np.allclose(X[0, :], old_x),
      f"X[0,:]={X[0, :]} 期望={old_x}")
check("1.2 无 bounds 时 model_extent 从 0 起",
      np.allclose(ext, [0.0, 80.0, 0.0, 60.0]),
      f"ext={ext}")

# 1.3 有 bounds：整体平移 (x0, y0)，且外框吻合时不告警
import logging
_logs = []


class _Capture(logging.Handler):
    def emit(self, record):
        _logs.append(record.getMessage())


_logger = logging.getLogger("app.analysis.visualizer")
_logger.addHandler(_Capture())
bounds = [1000.0, 2000.0, 1080.0, 2060.0]     # 跨度 80 x 60，与模型域一致
X2, Y2, ext2 = viz._grid_coords(nrow, ncol, delr, delc, shp_bounds=bounds)
check("1.3 有 bounds 时 X 整体平移 +1000",
      np.allclose(X2, X + 1000.0), f"X2[0,:]={X2[0, :]}")
check("1.4 有 bounds 时 Y 整体平移 +2000",
      np.allclose(Y2, Y + 2000.0), f"Y2[:,0]={Y2[:, 0]}")
check("1.5 外框吻合时不告警",
      not any("错位" in m for m in _logs), f"logs={_logs}")

# 1.6 非均匀 delr：格子宽度等于输入，不被拉伸
delr_nu = np.array([10.0, 20.0, 30.0, 40.0])
Xn, _, extn = viz._grid_coords(nrow, ncol, delr_nu, delc, shp_bounds=None)
centers = (np.concatenate([[0.0], np.cumsum(delr_nu)])[:-1]
           + np.concatenate([[0.0], np.cumsum(delr_nu)])[1:]) / 2
check("1.6 非均匀 delr 的格心不被拉伸",
      np.allclose(Xn[0, :], centers),
      f"Xn[0,:]={Xn[0, :]} 期望={centers}")
check("1.7 非均匀 delr 的总跨度 = sum(delr)",
      abs(extn[1] - float(np.sum(delr_nu))) < 1e-9, f"ext={extn}")

# 1.8 模型域与 SHP 外框严重不吻合 → 必须告警（不静默错位）
_logs.clear()
viz._grid_coords(nrow, ncol, delr, delc,
                 shp_bounds=[0.0, 0.0, 800.0, 600.0])   # 外框是模型域的 10 倍
check("1.8 模型域与 SHP 外框不吻合时告警",
      any("错位" in m for m in _logs), f"logs={_logs}")
_logger.removeHandler(_logger.handlers[-1])


# ============================================================
# 2. _draw_plume_panel —— 色标范围 / 分层 / 掩膜下界
# ============================================================
section("2. _draw_plume_panel 色标与掩膜")

Xp, Yp, _ = viz._grid_coords(10, 10, np.full(10, 20.0), np.full(10, 20.0))
conc = np.zeros((10, 10))
conc[3:7, 3:7] = 100.0        # 只有中间 16 格有浓度

fig, ax = plt.subplots()
cf = viz._draw_plume_panel(ax, conc, Xp, Yp, vmin=0.0, vmax=50.0,
                           levels=11, threshold=1.0, alpha=0.8)
check("2.1 vmin/vmax 透传到 norm",
      np.allclose([cf.norm.vmin, cf.norm.vmax], [0.0, 50.0]),
      f"clim={cf.norm.vmin},{cf.norm.vmax}")
check("2.2 levels 决定分层数",
      len(cf.levels) == 11, f"len(levels)={len(cf.levels)}")

# 2.3 vmax=50 < 数据最大 100 → 必须开 extend="max"，否则色标说不清溢出
check("2.3 数据超出 vmax 时 extend=max",
      cf.extend == "max", f"extend={cf.extend}")
plt.close(fig)

# 2.4 数据未超出 vmax → 不得开 extend（否则白画一个箭头）
fig, ax = plt.subplots()
cf = viz._draw_plume_panel(ax, conc, Xp, Yp, vmin=0.0, vmax=200.0,
                           levels=11, threshold=1.0, alpha=0.8)
check("2.4 数据未超出 vmax 时 extend=neither",
      cf.extend == "neither", f"extend={cf.extend}")
plt.close(fig)

# 2.5 掩膜下界：vmin=0 但 threshold=50 → floor=min(50,0)=0，低值仍可见，
#     这样"手动把下限调到 0"时能看到完整分布，而不是被阈值切掉
fig, ax = plt.subplots()
cf = viz._draw_plume_panel(ax, conc, Xp, Yp, vmin=0.0, vmax=200.0,
                           levels=11, threshold=50.0, alpha=0.8)
check("2.5 vmin < threshold 时填色下限取 vmin",
      np.allclose(cf.norm.vmin, 0.0), f"vmin={cf.norm.vmin}")
plt.close(fig)

# 2.6 threshold 线：单层红色虚线必须存在
from matplotlib.contour import ContourSet

fig, ax = plt.subplots()
viz._draw_plume_panel(ax, conc, Xp, Yp, vmin=0.0, vmax=200.0,
                      levels=11, threshold=50.0, alpha=0.8)
css = [c for c in ax.collections if isinstance(c, ContourSet)]
check("2.6a 填色 + 阈值线共两个 ContourSet",
      len(css) == 2, f"n={len(css)}")
_single = [cs for cs in css if len(cs.levels) == 1]
check("2.6b 阈值线为单层", len(_single) == 1,
      f"levels={[len(cs.levels) for cs in css]}")
if _single:
    _ec = np.asarray(_single[0].get_edgecolor(), dtype=float).ravel()[:3]
    check("2.6c 阈值线颜色为红", np.allclose(_ec, [1.0, 0.0, 0.0]),
          f"edgecolor={_ec}")
    check("2.6d 阈值线画在 threshold 处",
          float(_single[0].levels[0]) == 50.0,
          f"level={_single[0].levels}")
plt.close(fig)

# 2.7 vmax <= vmin 的退化输入不得抛异常（用户可能把上限填得比下限小）
fig, ax = plt.subplots()
try:
    viz._draw_plume_panel(ax, conc, Xp, Yp, vmin=100.0, vmax=1.0,
                          levels=11, threshold=1.0, alpha=0.8)
    ok = True
except Exception as e:                       # noqa: BLE001
    ok = False
    print(f"       异常={e}")
check("2.7 vmax<vmin 的退化输入不抛异常", ok)
plt.close(fig)


# ============================================================
# 3. plot_plume_evolution —— 演化总览
# ============================================================
section("3. plot_plume_evolution 演化总览")

frames = [np.full((10, 10), float(i * 10)) for i in range(1, 7)]
times = [1000.0 * i for i in range(1, 7)]
_dr = np.full(10, 20.0)

fig = viz.plot_plume_evolution(frames, times, _dr, _dr, threshold=1.0)
check("3.1 返回 Figure", isinstance(fig, plt.Figure))
check("3.2 使用 constrained 布局引擎",
      fig.get_layout_engine() is not None,
      f"engine={fig.get_layout_engine()}")
check("3.3 6 帧 → 2x3 子图 + 1 个 colorbar 轴",
      len(fig.axes) == 7, f"len(axes)={len(fig.axes)}")
# colorbar 与每个子图都不重叠。
# constrained 布局是**延迟**计算的（首次 draw 时才排布），不 draw 就读位置会
# 读到未排布的默认值，断言会失去意义。
fig.canvas.draw()
cax = fig.axes[-1]
overlaps = [a.get_position().overlaps(cax.get_position())
            for a in fig.axes[:-1] if a.get_visible()]
check("3.4 colorbar 不与任何子图重叠", not any(overlaps), f"overlaps={overlaps}")
plt.close(fig)

# 无底图（4 个路径参数全 None）不得抛异常
try:
    fig = viz.plot_plume_evolution(frames, times, _dr, _dr, threshold=1.0,
                                   basemap_path=None, shp_path=None,
                                   clip_shp_path=None, shp_bounds=None)
    ok = True
    plt.close(fig)
except Exception as e:                       # noqa: BLE001
    ok = False
    print(f"       异常={e}")
check("3.5 底图参数全为 None 时不抛异常", ok)

# 显式 vmin/vmax 生效
fig = viz.plot_plume_evolution(frames, times, _dr, _dr, threshold=1.0,
                               vmin=0.0, vmax=30.0, levels=7)
cs = [a for a in fig.axes[0].collections]
check("3.6 手工 vmax 透传到首帧",
      any(abs(getattr(c, "norm", None).vmax - 30.0) < 1e-9
          for c in cs if getattr(c, "norm", None) is not None),
      f"collections={len(cs)}")
plt.close(fig)

# 流速场叠加不抛异常
vel = np.ones((3, 10, 10)) * 0.01
try:
    fig = viz.plot_plume_evolution(frames, times, _dr, _dr, threshold=1.0,
                                   velocity=vel)
    ok = True
    plt.close(fig)
except Exception as e:                       # noqa: BLE001
    ok = False
    print(f"       异常={e}")
check("3.7 叠加流速场箭头不抛异常", ok)


# ============================================================
# 4. plot_plume_at_time —— 单帧交互图
# ============================================================
section("4. plot_plume_at_time 单帧图")

fig = viz.plot_plume_at_time(frames[-1], 6000.0, _dr, _dr, threshold=1.0,
                             title="污染羽 t = 6000 d",
                             subtitle="t = 6000 d（输出时刻）")
check("4.1 返回 Figure", isinstance(fig, plt.Figure))
check("4.2 使用 constrained 布局引擎",
      fig.get_layout_engine() is not None)
check("4.3 单主图 + 单 colorbar",
      len(fig.axes) == 2, f"len(axes)={len(fig.axes)}")

fig.canvas.draw()
ax, cax = fig.axes[0], fig.axes[1]
check("4.4 colorbar 不与主图重叠",
      not ax.get_position().overlaps(cax.get_position()),
      f"ax={ax.get_position().bounds} cax={cax.get_position().bounds}")

# 图例必须挪到坐标轴之外，否则某个时刻必被污染羽盖住
leg = ax.get_legend()
check("4.5 图例存在", leg is not None)
if leg is not None:
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    lb = leg.get_window_extent(r)
    ab = ax.get_window_extent(r)
    check("4.6 图例不与主图数据区重叠",
          not lb.overlaps(ab), f"legend={lb.bounds} axes={ab.bounds}")
plt.close(fig)

# 无底图不抛异常
try:
    fig = viz.plot_plume_at_time(frames[-1], 6000.0, _dr, _dr, threshold=1.0)
    ok = True
    plt.close(fig)
except Exception as e:                       # noqa: BLE001
    ok = False
    print(f"       异常={e}")
check("4.7 无底图时不抛异常", ok)


# ============================================================
# 5. plot_process_analysis —— 主控过程分析
# ============================================================
section("5. plot_process_analysis 过程分析图")

pi = ProcessIdentifier()
rng = np.random.default_rng(42)
conc_hist = [rng.random((10, 10)) * (i + 1) * 5 for i in range(4)]
vel_hist = [np.tile(np.array([0.05, 0.0, 0.0]).reshape(3, 1, 1), (1, 10, 10))
            for _ in range(4)]
proc = pi.identify_dominant_process(
    conc_hist, vel_hist, {"al": 20.0, "porosity": 0.3,
                          "decay_rate": 0.0, "kd": 0.0})

fig = viz.plot_process_analysis(proc)
check("5.1 吃真实 ProcessIdentifier 返回值不抛异常",
      isinstance(fig, plt.Figure))
check("5.2 有 contributions 时画出内容",
      len(proc.get("contributions", {})) >= 2,
      f"contributions={proc.get('contributions')}")
check("5.3 贡献率合计约 100%",
      abs(sum(float(v) for v in proc.get("contributions", {}).values()) - 100.0) < 1.0,
      f"sum={sum(float(v) for v in proc.get('contributions', {}).values())}")
check("5.4 使用 constrained 布局引擎",
      fig.get_layout_engine() is not None)
# 底部注记必须让出空间：axes 下沿不得低于留白带顶部（rect 的 y0=0.18）
fig.canvas.draw()
_bottoms = [a.get_position().y0 for a in fig.axes]
check("5.5 底部为参数注记留白（axes 未压到底）",
      min(_bottoms) >= 0.15, f"min(y0)={min(_bottoms)}")
plt.close(fig)

# insufficient_data 分支
fig = viz.plot_process_analysis({"dominant_process": "insufficient_data",
                                 "contributions": {}})
check("5.6 insufficient_data 分支不抛异常且给出文案",
      isinstance(fig, plt.Figure) and len(fig.texts) >= 1,
      f"texts={len(fig.texts)}")
plt.close(fig)

# 全零 contributions 分支（浓度场全零）
fig = viz.plot_process_analysis({"dominant_process": "advection",
                                 "contributions": {"advection": 0.0,
                                                   "dispersion": 0.0,
                                                   "decay": 0.0}})
check("5.7 全零贡献率不抛异常（不画空白饼图）",
      isinstance(fig, plt.Figure) and len(fig.texts) >= 1)
plt.close(fig)


# ============================================================
# 6. extract_velocity_field —— 流速场形状
# ============================================================
section("6. extract_velocity_field 流速场")


class _Pkg:
    """伪造 flopy 的包对象（只需 get_data()）"""

    def __init__(self, data):
        self._d = data

    def get_data(self):
        return self._d


class _Out:
    def __init__(self, head):
        self._h = head

    def head(self):
        return _Pkg(self._h)


class _Grid:
    def __init__(self, delr, delc):
        self.delr = delr
        self.delc = delc


class _Npf:
    """伪造 npf 包：extract_velocity_field 走的是 gwf.npf.k.get_data()"""

    def __init__(self, k):
        self.k = _Pkg(k)


class _Gwf:
    """伪造 ModflowGwf：只需 output.head() / modelgrid / npf.k"""

    K_VAL = 5.0

    def __init__(self, nrow, ncol, nlay):
        # 线性水头场，保证梯度非零
        h = np.tile(np.linspace(100.0, 90.0, ncol), (nrow, 1))
        self._head = np.stack([h] * nlay, axis=0)
        self.output = _Out(self._head)
        self.modelgrid = _Grid(np.full(ncol, 20.0), np.full(nrow, 20.0))
        self.npf = _Npf(np.full((nlay, nrow, ncol), self.K_VAL))


NROW, NCOL, NLAY = 6, 8, 2
rp = ResultProcessor()
vel = rp.extract_velocity_field(_Gwf(NROW, NCOL, NLAY), layer=0)

check("6.1 返回非 None", vel is not None)
if vel is not None:
    # ProcessIdentifier 内部先 norm(vel, axis=0)，再与 2D 浓度相乘
    check("6.2 形状为 (3, nrow, ncol)",
          vel.shape == (3, NROW, NCOL), f"shape={vel.shape}")
    v_mag = np.linalg.norm(vel, axis=0) if vel.ndim > 2 else np.abs(vel)
    check("6.3 norm(axis=0) 后可与 2D 浓度广播",
          v_mag.shape == (NROW, NCOL), f"v_mag.shape={v_mag.shape}")
    # 达西定律：水头自西向东递减 → qx 应为正（向东），qy 为 0
    check("6.4 qx 方向与达西定律一致（向东为正）",
          float(np.mean(vel[0])) > 0, f"mean(qx)={float(np.mean(vel[0]))}")
    check("6.5 均质场 qy≈0",
          abs(float(np.mean(vel[1]))) < 1e-9, f"mean(qy)={float(np.mean(vel[1]))}")
    # 量级校验：qx = K * dh/dx = 5.0 * 10 / (7 * 20)
    _expect = _Gwf.K_VAL * 10.0 / (7 * 20.0)
    check("6.5b qx 量级等于 K·dh/dx",
          abs(float(np.mean(vel[0])) - _expect) < 1e-6,
          f"qx={float(np.mean(vel[0]))} 期望={_expect}")

# 与 ProcessIdentifier 复合端到端（这是形状约束的真正验收点）
_conc = [np.random.default_rng(0).random((NROW, NCOL)) for _ in range(3)]
try:
    _r = pi.identify_dominant_process(
        _conc, [vel] * 3, {"al": 10.0, "porosity": 0.25,
                           "decay_rate": 0.0, "kd": 0.0})
    ok = _r.get("dominant_process") != "insufficient_data"
except Exception as e:                       # noqa: BLE001
    ok = False
    print(f"       异常={e}")
check("6.6 与 ProcessIdentifier 复合可跑通", ok)


# ============================================================
# 7. _sample_conc_at —— 时刻采样与插值
# ============================================================
section("7. _sample_conc_at 时刻采样/插值")

TIMES = np.array([100.0, 200.0, 500.0, 1000.0])


class _FakeConc:
    """帧值恒等于时刻本身，便于直接验算插值结果"""

    def get_data(self, totim=None):
        return np.full((1, 2, 3), float(totim))


class _FakeGwt:
    def __init__(self):
        self.output = type("O", (), {"concentration": lambda s: _FakeConc()})()


CACHE = {"gwt": _FakeGwt(), "times": TIMES, "layer": 0}


def _val(frame):
    """取帧内的代表值（extract_concentration_field 已把 (1,nrow,ncol) 压成 2D）"""
    return float(np.asarray(frame).ravel()[0]) if frame is not None else None


# 7.1 t 早于首帧 → 钳到首帧，不插值
c, t0, t1, interp, note = _sample_conc_at(CACHE, 50.0)
check("7.1a 早于首帧时钳到首帧",
      np.allclose(c, 100.0) and t0 == 100.0 and not interp,
      f"c={_val(c)} t0={t0} interp={interp}")
check("7.1b 早于首帧的提示文案明确",
      "早于首个输出时刻" in note, f"note={note}")

# 7.2 t 恰为首帧 → 输出时刻
c, t0, t1, interp, note = _sample_conc_at(CACHE, 100.0)
check("7.2 恰为首帧时直接返回原帧",
      np.allclose(c, 100.0) and not interp and "输出时刻" in note,
      f"c={_val(c)} interp={interp} note={note}")

# 7.3 落在两个输出时刻之间 → 线性插值
c, t0, t1, interp, note = _sample_conc_at(CACHE, 150.0)
check("7.3a 中间时刻做线性插值",
      np.allclose(c, 150.0) and interp, f"c={_val(c)} interp={interp}")
check("7.3b 插值端点取自相邻输出时刻",
      t0 == 100.0 and t1 == 200.0, f"t0={t0} t1={t1}")
check("7.3c 插值提示写明两端时刻",
      "线性插值" in note and "100" in note and "200" in note, f"note={note}")

# 7.3d 非等距区间：350 落在 200~500 之间，权重 = 150/300 = 0.5
c, t0, t1, interp, note = _sample_conc_at(CACHE, 350.0)
check("7.3d 非等距区间按实际间隔加权",
      np.allclose(c, 350.0) and t0 == 200.0 and t1 == 500.0,
      f"c={_val(c)} t0={t0} t1={t1}")

# 7.4 t 精确命中中间输出时刻 → 不插值
c, t0, t1, interp, note = _sample_conc_at(CACHE, 500.0)
check("7.4 精确命中输出时刻时不插值",
      np.allclose(c, 500.0) and not interp, f"c={_val(c)} interp={interp}")

# 7.5 t 恰为末帧
c, t0, t1, interp, note = _sample_conc_at(CACHE, 1000.0)
check("7.5 恰为末帧时直接返回原帧",
      np.allclose(c, 1000.0) and not interp, f"c={_val(c)} interp={interp}")

# 7.6 t 超过模拟时长 → 钳到末帧并明示（不做外推）
c, t0, t1, interp, note = _sample_conc_at(CACHE, 5000.0)
check("7.6a 超出模拟时长时钳到末帧",
      np.allclose(c, 1000.0) and not interp, f"c={_val(c)} interp={interp}")
check("7.6b 超界提示文案明确", "超出模拟时长" in note, f"note={note}")

# 7.7 缓存为空 → 明确提示"先运行建模"，不抛异常
c, t0, t1, interp, note = _sample_conc_at({}, 100.0)
check("7.7 无结果时返回提示而不抛异常",
      c is None and "请先运行建模" in note, f"note={note}")

# 7.8 帧缓存生效：重复取同一时刻不应重复调用 get_data
CACHE2 = {"gwt": _FakeGwt(), "times": TIMES, "layer": 0}
_sample_conc_at(CACHE2, 350.0)
n_after_first = len(CACHE2["_frame_cache"])
_sample_conc_at(CACHE2, 360.0)          # 仍是 200~500 区间
check("7.8a 端点帧取自同一缓存",
      len(CACHE2["_frame_cache"]) == n_after_first,
      f"{n_after_first} -> {len(CACHE2['_frame_cache'])}")
check("7.8b 缓存键为原样的输出时刻",
      set(CACHE2["_frame_cache"].keys()) == {200.0, 500.0},
      f"keys={set(CACHE2['_frame_cache'].keys())}")


# ============================================================
# 8. _pick_vmin_vmax —— 色标范围解析
# ============================================================
section("8. _pick_vmin_vmax 色标范围解析")

span = (0.0, 100.0)
check("8.1 自动模式忽略输入框",
      _pick_vmin_vmax(True, 30, 60, span) == (0.0, 100.0),
      f"{_pick_vmin_vmax(True, 30, 60, span)}")
check("8.2 手动模式采用输入框",
      _pick_vmin_vmax(False, 10, 50, span) == (10.0, 50.0),
      f"{_pick_vmin_vmax(False, 10, 50, span)}")
check("8.3 单个框留空则该项回退自动",
      _pick_vmin_vmax(False, None, 50, span) == (0.0, 50.0),
      f"{_pick_vmin_vmax(False, None, 50, span)}")
check("8.4 上限留空同样回退自动",
      _pick_vmin_vmax(False, 10, None, span) == (10.0, 100.0),
      f"{_pick_vmin_vmax(False, 10, None, span)}")
# 上限小于下限 → 必须纠正，否则 contourf 报错
_v, _V = _pick_vmin_vmax(False, 80, 20, span)
check("8.5 上限小于下限时自动纠正", _V > _v, f"vmin={_v} vmax={_V}")


# ============================================================
# 汇总
# ============================================================
print(f"\n{'=' * 60}")
total = passed + failed
pct = 100.0 * passed / total if total > 0 else 0
print(f"污染羽可视化测试结果: {passed}/{total} 通过 ({pct:.1f}%)")
if failed:
    print(f"[FAIL] {failed} 项失败:")
    for n in _failures:
        print(f"   - {n}")
    sys.exit(1)
print("[OK] 全部通过")
print(f"{'=' * 60}")

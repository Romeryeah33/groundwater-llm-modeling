# -*- coding: utf-8 -*-
"""为中期报告生成「结果状态」的软件界面截图。

流程：
  1. 从 data/sessions/default.json **只读**加载 East Texas 真实 IR（与
     scripts/verify_east_texas.py 的 Case D 同源）；补 domain.origin_xy，
     并把 RCH 边界的 geometry 换成**显式 132 格**——源区解析走优先级 1
     （IR 直给索引），于是不依赖那个会被清理的上传临时目录里的 SHP；
  2. 用**非 default** 的 sid 调一次真实的 run_model_handler —— 于是拿到
     的是真图件 / 真结果摘要 / 真 FloPy 代码 / 真 Word 报告 / 真管线日志，
     而不是我另画一套；
  3. 同一进程内调 _sync_plume_controls(sid) 取单时刻交互图与控件范围
     （它读模块级 _result_cache，所以必须在 handler 之后调）；
  4. create_ui() 建一套离屏 UI，用 demo.load 把上面所有值注入组件；
  5. 起服务，交给外部 Chrome headless 截图（见本文件末尾的说明）。

**绝不写 data/sessions/default.json**（那是用户的真实会话）。

用法：
    python scripts/_ui_shot_make.py            # 起在 7999，常驻
    SHOT_PORT=8010 python scripts/_ui_shot_make.py
"""
import os
import sys
import copy
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 不自己包 sys.stdout：导入 gradio_app 时它会在 win32 上重绑一次，
# 自己包的那层会随之被关掉，后续 print 抛 "I/O operation on closed file"。
# 编码靠外部 `export PYTHONIOENCODING=utf-8` 保证。

# 截图专用 sid：既不是 "default"（用户真实会话），也不与验证脚本的 sid 撞车
SID = "report_shot"
PORT = int(os.environ.get("SHOT_PORT", "7999"))
IR_PATH = ROOT / "data" / "sessions" / "default.json"

# 与 scripts/verify_east_texas.py 的 Case D 验收锁一致
NPER, PERLEN = 10, 1500.0
EXPECT_ROWS = (52, 62)
EXPECT_COLS = (137, 148)
ORIGIN_XY = [612284.0, 3427412.0]

# 要注入的组件 id（来自 create_ui() 的 blocks 枚举）
#
# ⚠ 分三组注入，每组镜像真实 UI 的一个 handler（各自的输出数目与顺序都要对上）：
#   RESULT_IDS —— run_btn.click(run_model_handler, ...)        → 7 个输出
#   PLUME_IDS  —— .then(_sync_plume_controls, ...)             → 6 个输出
#   INPUT_IDS  —— 对话记录/输入框/参数表/IR/阶段/缺失参数（由别的 handler 更新）
# 曾把 19 个混在一个 demo.load 里发，结果 gr.Plot 渲染不出来；拆开后正常。
# 这条差异也是"项目 UI 画不出图"疑似为截图工具自身路径问题的证据，见开发计划第 3 项。
RESULT_IDS = [36, 34, 39, 55, 65, 30, 67]
PLUME_IDS = [42, 43, 47, 48, 51, 52]
INPUT_IDS = [6, 7, 21, 24, 26, 27]


def build_ir():
    """Case D 的真实 IR，但把源区换成显式 grid_indices。"""
    raw = json.loads(IR_PATH.read_text(encoding="utf-8"))
    ir = copy.deepcopy(raw.get("json_ir", raw))

    ir.setdefault("domain", {})["origin_xy"] = list(ORIGIN_XY)
    ir["stress_periods"] = {
        "nper": NPER,
        "period_data": [{"perlen": PERLEN, "nstp": 1, "tsmult": 1.0}
                        for _ in range(NPER)],
    }

    idx = [{"layer": 0, "row": r, "col": c, "period": 0}
           for r in range(EXPECT_ROWS[0], EXPECT_ROWS[1] + 1)
           for c in range(EXPECT_COLS[0], EXPECT_COLS[1] + 1)]

    n = 0
    for b in ir.get("boundary_conditions", []):
        if b.get("type") == "rch":
            b["geometry"] = {"type": "grid_indices", "grid_indices": idx}
            n += 1
    print(f"源区已写死 {len(idx)} 格（{n} 条 rch 边界）", flush=True)
    return ir, raw


def main():
    from app.frontend import gradio_app as G

    ir, session_raw = build_ir()

    # ---- 1. 跑一次真实建模 ----
    print("调用 run_model_handler（真实建模，约 1-2 分钟）...", flush=True)
    t0 = time.time()
    res = G.run_model_handler(ir, SID)
    print(f"  完成，用时 {time.time() - t0:.0f}s", flush=True)
    head_fig, result_html, plume_fig, proc_fig, report_path, code_str, dbg = res

    def _brief(v):
        if v is None:
            return "None"
        if isinstance(v, str):
            return f"str[{len(v)}] {v[:50]!r}"
        return f"{type(v).__name__}"

    for name, v in zip(("水头图", "结果摘要", "污染羽总览", "过程图",
                        "报告文件", "FloPy代码", "管线日志"), res):
        print(f"  {name:8s}: {_brief(v)}", flush=True)
    if report_path:
        print(f"  报告存在: {Path(report_path).exists()} -> {report_path}", flush=True)

    # ---- 2. 末时刻交互图（依赖上一步填充的 _result_cache）----
    print("调用 _sync_plume_controls 取单时刻图...", flush=True)
    ctrl = G._sync_plume_controls(SID)          # (slider, num, vmin, vmax, fig, status)
    print(f"  单时刻图: {_brief(ctrl[4])}", flush=True)
    print(f"  状态文字: {_brief(ctrl[5])}", flush=True)

    # ---- 3. 参数表与对话记录 ----
    param_df = G.build_param_table(ir)
    print(f"  参数表: {param_df.shape[0]} 行 × {param_df.shape[1]} 列", flush=True)
    print(f"  已填值行数: {int((param_df['当前值'].astype(str).str.strip() != '').sum())}",
          flush=True)

    hist_all = session_raw.get("history") or []
    hist = [m for m in hist_all[-4:] if isinstance(m, dict)]
    user_prompt = ""
    for m in reversed(hist_all):
        if isinstance(m, dict) and m.get("role") == "user":
            user_prompt = str(m.get("content") or "")
            break
    print(f"  对话记录注入 {len(hist)} 条；输入框 {len(user_prompt)} 字", flush=True)

    # ---- 4. 离屏 UI + 注入 ----
    #
    # 三组注入，各自镜像真实 UI 的一个 handler。**混成一组发时图渲染不出来**，
    # 拆开后正常 —— 这是"项目 UI 画不出图"疑似为截图工具自身路径问题的证据。
    demo = G.create_ui()

    # (a) 结果类：镜像 run_btn.click(run_model_handler, ...) 的 7 个输出
    def _inject_result():
        return (head_fig, result_html, plume_fig, proc_fig,
                report_path, code_str, dbg)

    # (b) 单时刻交互类：镜像 .then(_sync_plume_controls, ...) 的 6 个输出
    def _inject_plume():
        return (ctrl[0], ctrl[1], ctrl[2], ctrl[3], ctrl[4],
                ctrl[5] if isinstance(ctrl[5], str) else "")

    # (c) 输入类：对话记录、输入框、参数表、IR、会话阶段、缺失参数
    def _inject_input():
        return (hist, user_prompt, param_df, ir, "execution", [])

    # create_ui() 返回时已经退出 Blocks 上下文，事件注册必须在上下文里做，
    # 否则报 "Cannot call load outside of a gradio.Blocks context"。
    with demo:
        demo.load(_inject_input, outputs=[demo.blocks[i] for i in INPUT_IDS])
        demo.load(_inject_result, outputs=[demo.blocks[i] for i in RESULT_IDS])
        demo.load(_inject_plume, outputs=[demo.blocks[i] for i in PLUME_IDS])

    demo.launch(server_name="127.0.0.1", server_port=PORT,
                prevent_thread_lock=True, quiet=True, show_error=True)
    print(f"\n>>> READY  http://127.0.0.1:{PORT}/", flush=True)
    print(">>> 现在用 Chrome headless 截图；Ctrl-C 结束", flush=True)

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("bye", flush=True)


if __name__ == "__main__":
    main()

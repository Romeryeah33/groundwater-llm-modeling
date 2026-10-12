# -*- coding: utf-8 -*-
"""建模前的三道闸 —— 界面与 REST 接口**共用同一份**判定逻辑。

为什么必须抽出来：这三道闸原先只长在 `gradio_app.run_model_handler` 里。
`scripts/run_case.py` 已经是"无界面"的第二个调用方，它把这个顺序抄了一遍。
接口是第三个调用方。**三处各写一遍必然漂移** —— 某天界面把 P3 也改成阻断
条件、或把源区闸挪到校验闸前面，接口不会跟着变，于是出现"界面拦下的用例
接口照样能跑"。这种不一致最难查，因为没有报错，只有对不上的结果。

所以**顺序与判定条件只写在这一个函数里**，三个调用方都调它。

三道闸（顺序不可颠倒）：

    1. 缺失参数闸   ``detect_missing_params``      → P1 或 P2 非空则拦
                                                 （P3 只是提示，不拦）
    2. 物理校验闸   ``quick_validate``             → ``error_count > 0`` 则拦
    3. 源区定位闸   ``materialize_source_zones``   → 有 issues 则拦
                   不拦时把 ``geometry.grid_indices`` **就地物化进 IR**

第 3 道闸为什么不能省：East Texas 案例的污染羽之所以看起来像水流模型的结果，
根因就在这里 —— 填埋场只占 132 格（0.59%），但带浓度的 RCH 边界找不到注入
位置时，旧代码会**静默**展开成 layer0 全网格（22454 格），源被放大 170 倍。
现在解析不出位置就硬拒绝。

返回值刻意是**纯数据**（dict），不含 HTML 也不含 matplotlib Figure ——
界面拿它渲染卡片，接口拿它编 JSON。渲染是两边**唯一允许分叉**的地方。
"""
import logging
from typing import Iterable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# 闸门名称 —— 调用方按这个值判断"被拦在哪一道"
STAGE_MISSING = "missing"
STAGE_VALIDATE = "validate"
STAGE_SOURCE_ZONE = "source_zone"


def _resolve_source_zone_rows(ir: dict, resolved: dict) -> list:
    """把源区解析结果编成人类可读的一行行（界面直接贴，接口原样返回）。

    "132 格 (占全域 0.59%)" 这一行是**用户唯一能看出点源有没有被放大的地方** ——
    所以格数与占比必须算出来，不能只报"定位成功"。
    """
    from app.modeling.source_zone import grid_dims

    grid = ir.get("grid", {}) or {}
    total = max(int(grid.get("nrow", 1) or 1) * int(grid.get("ncol", 1) or 1), 1)
    _nrow, _ncol, delr, delc, _ = grid_dims(grid)
    # 补给量 = 格数 × 单格面积 × 速率。单格面积取 delr/delc 的均值乘积，
    # 不写死 20×20 —— 换个网格就是另一个数。
    cell_area = float(np.mean(delr)) * float(np.mean(delc))

    rows = []
    for name, r in resolved.items():
        cells = len(r.get("grid_indices") or [])
        rate = None
        for bc in ir.get("boundary_conditions", []) or []:
            if isinstance(bc, dict) and bc.get("name") == name:
                raw = (bc.get("parameters") or {}).get("rate")
                rate = raw.get("value") if isinstance(raw, dict) else raw
                break
        try:
            flux_txt = f"，补给 {cells * float(rate) * cell_area:.2f} m³/d"
        except (TypeError, ValueError):
            flux_txt = ""
        rows.append(
            f"源区 '{name}': {cells} 格 (占全域 {cells / total * 100:.2f}%)，"
            f"面积 {r.get('area_m2', 0):.0f} m²{flux_txt}"
            f"，来源 {r.get('resolved_from', '?')}"
        )
    return rows


def apply_pregates(
    ir: dict,
    *,
    uploaded_paths: Iterable[str] = (),
    file_index: Optional[dict] = None,
    stop_early: bool = True,
) -> dict:
    """依次过三道闸。**不修改入参以外的任何东西**（源区结果就地写进 `ir`）。

    Args:
        ir: JSON IR。源区定位成功时会被**就地**补上 ``geometry.grid_indices``，
            这是有意的 —— 物化进 IR 后重跑不再依赖上传文件所在的临时目录
            （下一次走优先级 1 直接命中）。
        uploaded_paths: 用户上传文件的本地路径（用于源区闸按文件解析）。
        file_index: ``{"shapefiles": {...}}`` 形态的上传索引。
        stop_early: **默认 True，与界面行为逐字一致** —— 任何一道闸拦下就
            立即返回，不再跑后面的闸（界面原本就是"被拦就 return"）。
            置 False 会把三道闸都跑完，用于"一次把所有问题都报给用户"
            的场景；接口默认也走 True，免得与界面结果不可比。

    Returns:
        dict：
          ``ok``                 是否放行建模
          ``stage``              被拦在哪一道（``None`` 表示全部放行）
          ``missing``            ``detect_missing_params`` 原样结果
          ``validation``         ``quick_validate`` 原样结果
          ``source_zones``       解析成功的 ``{名称: 结果}``
          ``source_zone_issues`` 失败项 ``[[名称, 消息], ...]``
          ``source_zone_rows``   人类可读的源区摘要行
          ``errors``             人话错误信息（界面/接口都能直接展示）
          ``ir``                 处理后的 IR（与入参同一个对象）
    """
    from app.llm_agent.intent_parser import intent_parser
    from app.modeling.source_zone import (
        SourceZoneUnresolved,
        materialize_source_zones,
    )
    from app.physics_engine.validator import quick_validate

    report = {
        "ok": True,
        "stage": None,
        # 闸门**自己抛异常**时填这里的原话，与"检查跑通了、确实发现错误"区分开。
        # 少了这个区分，界面会把"校验器炸了"渲染成"发现 0 个错误" —— 一句
        # 自相矛盾并且把人引向错误方向的话。
        "gate_error": None,
        "missing": {},
        "validation": {},
        "source_zones": {},
        "source_zone_issues": [],
        "source_zone_rows": [],
        "errors": [],
        "ir": ir,
    }

    def _blocked(stage: str) -> dict:
        """被拦下时的统一收尾：算好源区摘要行（此时通常为空）再返回。"""
        report["ok"] = False
        report["stage"] = stage
        report["source_zone_rows"] = _resolve_source_zone_rows(
            report["ir"], report["source_zones"])
        return report

    if not isinstance(ir, dict) or not (ir.get("project") or {}).get("name"):
        report["errors"].append(
            "IR 为空或缺 project.name —— 请先完成对话建模需求，再开始建模。")
        return _blocked(STAGE_MISSING)

    # ---- 闸 1：缺失参数 ----------------------------------------------------
    try:
        missing = intent_parser.detect_missing_params(ir)
    except Exception as e:  # 检测本身抛异常 = 没法确认参数是否齐全，按拦处理
        logger.warning("缺失参数检测抛异常: %s: %s", type(e).__name__, e)
        report["gate_error"] = f"{type(e).__name__}: {e}"
        report["errors"].append(f"缺失参数检测失败：{report['gate_error']}")
        return _blocked(STAGE_MISSING)

    report["missing"] = missing or {}
    p1 = report["missing"].get("priority_1", []) or []
    p2 = report["missing"].get("priority_2", []) or []
    if p1 or p2:
        report["errors"].append(
            f"关键参数缺失：P1={p1 or '无'}，P2={p2 or '无'}。"
            "请先在对话区补齐（展开界面上的「缺失参数」清单）。")
        if stop_early:
            return _blocked(STAGE_MISSING)

    # ---- 闸 2：物理校验 ----------------------------------------------------
    try:
        val = quick_validate(ir)
    except Exception as e:
        logger.warning("物理校验抛异常: %s: %s", type(e).__name__, e)
        report["gate_error"] = f"{type(e).__name__}: {e}"
        report["errors"].append(f"参数校验失败：{report['gate_error']}")
        return _blocked(STAGE_VALIDATE)

    report["validation"] = val or {}
    if report["validation"].get("error_count", 0) > 0:
        issues = "\n".join(
            f"  {i.get('param_name')}: {i.get('description')}"
            for i in report["validation"].get("issues", [])
            if i.get("severity") == "error"
        )
        report["errors"].append(
            f"参数校验发现 {report['validation']['error_count']} 个错误：\n{issues}")
        if stop_early:
            return _blocked(STAGE_VALIDATE)

    # ---- 闸 3：源区定位 ----------------------------------------------------
    # 这道闸**必须跑**（不能因为前面拦下就跳过）：源区结果要就地物化进 IR，
    # 而 `stop_early=False` 时前面的闸还没拦下建模，物化是必须做的。
    try:
        ir, resolved, sz_issues = materialize_source_zones(
            ir, uploaded_paths=uploaded_paths, file_index=file_index)
        report["source_zones"] = resolved or {}
        report["source_zone_issues"] = [list(x) for x in (sz_issues or [])]
    except SourceZoneUnresolved as e:
        # 这是**预期内**的失败（位置解析不出来），不是崩溃 —— 不填 gate_error
        report["source_zone_issues"] = [["?", str(e)]]
    except Exception as e:
        logger.warning("源区定位抛异常: %s: %s", type(e).__name__, e)
        report["gate_error"] = f"{type(e).__name__}: {e}"
        report["source_zone_issues"] = [["?", f"源区定位时发生意外错误：{e}"]]

    if report["source_zone_issues"]:
        report["ok"] = False
        # stage 记**第一道**拦下的闸（stop_early=False 时前面可能已经拦过）——
        # 报"最先出问题的那一道"比报"最后一道"更有用
        report["stage"] = report["stage"] or STAGE_SOURCE_ZONE
        report["errors"].append(
            "源区定位失败，已阻止建模（避免把点源静默放大成全域面源 —— "
            "那正是污染羽看起来像水流模型结果的原因）。"
            + "\n" + "\n\n".join(
                f"【{n}】{msg}" for n, msg in report["source_zone_issues"]))

    report["source_zone_rows"] = _resolve_source_zone_rows(ir, report["source_zones"])
    report["ir"] = ir
    return report


def workspace_for_sid(sid: str):
    """本次运行的模型工作目录。界面与接口必须用**同一个**规则算，
    否则同一个 sid 的两个入口会写进两个目录，结果对不上。"""
    from app.config import settings

    name = f"run_{sid}" if sid else "run_default"
    ws = settings.workspace_dir / name
    ws.mkdir(parents=True, exist_ok=True)
    return ws

# -*- coding: utf-8 -*-
"""最小一组 REST 接口 —— 对话、跑模型、查状态、取结果。

设计上的几个"为什么"
====================

**为什么跑模型是后台任务而不是同步返回**
`build_and_run` 里 mf6 子进程的 timeout 是 3600 s（`model_builder.py`），
而 nginx 的 `proxy_read_timeout` 是 600 s —— 同步阻塞的话，一次长算会被
反代直接掐断，客户端拿不到任何结果。所以提交即返回 `task_id`，用
`/model/status` 轮询。

**为什么工作线程只有 1 个**
`plot_styler` 与 `_result_cache` 是**进程级单例**，同时跑两个模型会互相串图。
`max_workers=1` 让接口内部串行；界面与接口**同时**跑仍有风险，见文档。

**为什么会话与界面共用同一个注册表**
`app.llm_agent.session_registry` 是唯一一份"sid → LLM 三件套"的实现。
接口与界面走同一个进程、同一张表，于是"网页里聊出来的参数，接口里能直接跑"。

**认证**
按"每人自持密钥"的设计，本接口不需要账号。若部署方想加一道门，
设环境变量 `API_TOKEN` 即启用（请求头 `X-API-Token`）；不设则完全开放。
"""
import logging
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import anyio
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.llm_agent import session_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


# ================================================================
# 可选的口令闸（默认关闭）
# ================================================================

def _check_token(token: Optional[str]) -> None:
    expected = (os.getenv("API_TOKEN") or "").strip()
    if not expected:
        return                      # 没设 = 不开门禁，按设计如此
    if token != expected:
        raise HTTPException(status_code=401, detail="X-API-Token 无效")


def _checked_sid(raw: str) -> str:
    """接口侧的会话编号检查：**消毒前后必须一致，否则拒绝**。

    与界面刻意不同。界面把非法字符删掉就好（编号是系统自己分配的，删几个
    字符不影响谁是谁）；接口的编号**由调用方给**，静默改写会制造碰撞 ——
    `../etc` 和 `/etc` 都变成 `etc`，两个客户端会共用一份会话，而且它们自己
    完全不知道发生了什么。宁可报 400 让调用方换一个合法编号。
    """
    sid = session_registry.sanitize_sid(raw)
    if not sid:
        raise HTTPException(status_code=400,
                            detail="session_id 消毒后为空（只允许 A-Za-z0-9_-）")
    if sid != str(raw):
        raise HTTPException(
            status_code=400,
            detail=f"session_id {raw!r} 含非法字符，只允许 A-Za-z0-9_-（不会替你改写，"
                   "因为改写可能让两个不同的编号撞成同一个）")
    return sid


# ================================================================
# 任务表：进程内、不持久化
# ================================================================
# 刻意不引 Celery / Redis —— 需求是"最小一组"，而单进程单线程池本来就
# 只能跑一个任务，加外部队列解决不了任何现有问题，只增加部署负担。
_TASKS: Dict[str, dict] = {}
_TASK_LOCK = threading.RLock()

# 见模块头"为什么工作线程只有 1 个"
_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gw-model")


def _task_update(task_id: str, **kw) -> None:
    with _TASK_LOCK:
        t = _TASKS.get(task_id)
        if t is not None:
            t.update(kw)


def _run_model_task(task_id: str, ir: dict, sid: str) -> None:
    """线程池里真正跑模型的那段。**任何异常都要落进任务表**，
    否则客户端会永远看到 running（那是最难查的一种"卡住"）。"""
    _task_update(task_id, status="running")
    try:
        from app.modeling.model_builder import AutoModelBuilder
        from app.modeling.run_gates import apply_pregates, workspace_for_sid

        # 与界面走的是**同一个** apply_pregates，判定条件不会漂移
        gates = apply_pregates(ir)
        if not gates["ok"]:
            _task_update(task_id, status="failed", stage=gates["stage"],
                         error="；".join(gates["errors"]), gate_report=_public_gates(gates))
            return

        ws = workspace_for_sid(sid)
        builder = AutoModelBuilder(str(ws))
        result = builder.build_and_run(gates["ir"], "auto_model")

        if not result.get("success"):
            _task_update(task_id, status="failed",
                         error=str(result.get("error") or "模型运行失败"),
                         detail=result.get("detail"))
            return

        # 摘要要**自己算**：build_and_run 的返回值里没有 summary 这个键
        # （它只回 gwf/gwt 这些活对象），界面也是拿 gwf/gwt 另外调
        # ResultProcessor.generate_summary 算出来的。这里同样算，用同一个类，
        # 于是两边口径一致。
        summary = {}
        try:
            from app.analysis.result_processor import ResultProcessor
            summary = ResultProcessor().generate_summary(
                result.get("gwf"), result.get("gwt"), gates["ir"]) or {}
        except Exception as e:
            # 摘要算不出来不该让整个任务失败 —— 模型本身已经跑完了
            logger.warning("任务 %s 结果摘要生成失败（忽略）: %s", task_id, e)

        # 只回传**可序列化**的东西：gwf/gwt 是 flopy 对象，进不了 JSON，
        # 也不该跨进程传给客户端（见文档"接口不返回可交互模型对象"）。
        _task_update(
            task_id,
            status="completed",
            summary=summary,
            head_range=result.get("head_range"),
            reaction_effective=result.get("reaction_effective") or {},
            build_warnings=result.get("build_warnings") or [],
            source_zone_rows=gates["source_zone_rows"],
            workspace=str(ws),
            files=_list_outputs(ws),
        )
    except Exception as e:                       # noqa: BLE001 —— 兜底，必须落表
        logger.exception("建模任务 %s 失败", task_id)
        _task_update(task_id, status="failed", error=f"{type(e).__name__}: {e}")


def _public_gates(gates: dict) -> dict:
    """把闸门报告裁成能进 JSON 的部分（去掉 IR 本身，太大了）。"""
    return {
        "ok": gates.get("ok"),
        "stage": gates.get("stage"),
        "gate_error": gates.get("gate_error"),
        "missing": gates.get("missing") or {},
        "validation": gates.get("validation") or {},
        "source_zone_rows": gates.get("source_zone_rows") or [],
        "errors": gates.get("errors") or [],
    }


def _list_outputs(ws) -> list:
    """列出工作目录下可供下载的结果文件（图 / 报告）。

    只列白名单后缀：工作目录里还有 mf6 的一堆 .dis/.nam/.lst 中间文件，
    全列出去既没用又容易让客户端去下载解析不了的东西。
    """
    exts = {".png", ".docx", ".pdf", ".csv", ".ucn", ".json"}
    out = []
    try:
        for p in sorted(ws.rglob("*")):
            if p.is_file() and p.suffix.lower() in exts:
                out.append({
                    "name": p.name,
                    "path": str(p.relative_to(ws)).replace("\\", "/"),
                    "size": p.stat().st_size,
                })
    except OSError as e:
        logger.warning("列工作目录文件失败: %s", e)
    return out


# ================================================================
# 请求 / 响应模型
# ================================================================

class ChatRequest(BaseModel):
    message: str = Field(..., description="用户自然语言输入")
    session_id: str = Field(..., description="会话编号；同一编号共享上下文")


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    json_ir: dict = {}
    missing_params: dict = {}
    phase: str = ""
    ready_to_run: bool = False


class ModelRunRequest(BaseModel):
    json_ir: dict = Field(..., description="JSON IR；可取自 /api/chat 的返回")
    session_id: str = Field(..., description="会话编号；决定输出目录 workspace/run_{sid}")


class ModelRunResponse(BaseModel):
    task_id: str
    status: str


class TaskStatus(BaseModel):
    task_id: str
    status: str
    stage: Optional[str] = None
    error: Optional[str] = None


# ================================================================
# 路由
# ================================================================

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, x_llm_api_key: Optional[str] = Header(None),
               x_llm_model: Optional[str] = Header(None),
               x_llm_api_base: Optional[str] = Header(None),
               x_api_token: Optional[str] = Header(None)):
    """
    对话接口 —— 与网页对话框走**同一个** DialogueManager。

    密钥通过请求头传（**不放在 body 里**：body 常被日志、网关、错误上报
    顺手记下来，请求头默认不会）。带了就为该会话启用，不带则用服务器默认。

    这个接口是**同步阻塞**的（LLM 调用本身就慢，一秒到几十秒）。
    超时由 LLM_REQUEST_TIMEOUT 控制，默认 120 s。
    """
    _check_token(x_api_token)

    sid = _checked_sid(request.session_id)

    if x_llm_api_key:
        session_registry.configure(sid, api_key=x_llm_api_key,
                                   model=x_llm_model, api_base=x_llm_api_base)

    dm = session_registry.get(sid)
    # DialogueManager.process 是同步的，放进线程池避免堵住事件循环 ——
    # 否则一个慢请求会让所有人的请求都排队
    result = await anyio.to_thread.run_sync(dm.process, request.message, sid)

    return ChatResponse(
        reply=result.get("reply", ""),
        session_id=sid,
        json_ir=result.get("json_ir") or {},
        missing_params=result.get("missing_params") or {},
        phase=result.get("phase", ""),
        ready_to_run=bool(result.get("ready_to_run")),
    )


@router.post("/model/run", response_model=ModelRunResponse)
async def run_model(request: ModelRunRequest,
                    x_llm_api_key: Optional[str] = Header(None),
                    x_api_token: Optional[str] = Header(None)):
    """提交建模任务，立即返回 task_id（建模可能要跑几十分钟）。"""
    _check_token(x_api_token)

    sid = _checked_sid(request.session_id)
    if not isinstance(request.json_ir, dict) or not request.json_ir:
        raise HTTPException(status_code=400, detail="json_ir 不能为空")

    if x_llm_api_key:
        session_registry.configure(sid, api_key=x_llm_api_key)

    task_id = uuid.uuid4().hex[:12]
    with _TASK_LOCK:
        _TASKS[task_id] = {"task_id": task_id, "status": "queued",
                           "sid": sid, "error": None}
    _POOL.submit(_run_model_task, task_id, request.json_ir, sid)
    logger.info("建模任务 %s 已提交（会话 %s）", task_id, sid)
    return ModelRunResponse(task_id=task_id, status="queued")


@router.get("/model/status/{task_id}", response_model=TaskStatus)
async def model_status(task_id: str, x_api_token: Optional[str] = Header(None)):
    """查询建模任务状态：queued / running / completed / failed。"""
    _check_token(x_api_token)
    with _TASK_LOCK:
        t = _TASKS.get(task_id)
        if t is None:
            raise HTTPException(status_code=404, detail="未知 task_id")
        return TaskStatus(task_id=task_id, status=t["status"],
                          stage=t.get("stage"), error=t.get("error"))


@router.get("/model/result/{task_id}")
async def model_result(task_id: str, x_api_token: Optional[str] = Header(None)):
    """
    取回建模结果：结构化摘要 + 结果文件清单（**不含图片字节**）。

    未完成时返回 404 —— 与旧占位实现的语义一致，且比"返回空对象"更不容易
    被客户端当成"跑完了但没结果"。
    """
    _check_token(x_api_token)
    with _TASK_LOCK:
        t = _TASKS.get(task_id)
        if t is None:
            raise HTTPException(status_code=404, detail="未知 task_id")
        if t["status"] == "failed":
            raise HTTPException(status_code=409, detail={
                "task_id": task_id, "status": "failed",
                "stage": t.get("stage"), "error": t.get("error"),
                "detail": t.get("detail")})
        if t["status"] != "completed":
            raise HTTPException(status_code=404,
                                detail=f"任务尚未完成（当前 {t['status']}）")
        return {
            "task_id": task_id,
            "status": "completed",
            "summary": t.get("summary") or {},
            "head_range": t.get("head_range"),
            "reaction_effective": t.get("reaction_effective") or {},
            "build_warnings": t.get("build_warnings") or [],
            "source_zone_rows": t.get("source_zone_rows") or [],
            "files": t.get("files") or [],
        }


@router.get("/model/result/{task_id}/files/{name:path}")
async def model_result_file(task_id: str, name: str,
                            x_api_token: Optional[str] = Header(None)):
    """下载某个结果文件（图 / 报告）。`name` 取 `/model/result` 里 files[].path。

    `{name:path}` **必须带 `:path` 转换器** —— 结果是分目录放的
    （`auto_model/auto_model_gwt.ucn`），普通路径参数不匹配斜杠，会直接 404。
    放行斜杠不会带来穿越风险：下面不是"拼路径再判断"，而是**拿 name 去清单里
    比对**，清单里没有的一律 404。"""

    _check_token(x_api_token)
    from pathlib import Path

    with _TASK_LOCK:
        t = _TASKS.get(task_id)
        if t is None or t["status"] != "completed":
            raise HTTPException(status_code=404, detail="结果尚未生成")
        root = Path(t.get("workspace") or "")
        entries = t.get("files") or []

    # **只允许下载清单里列过的文件**，而不是把 name 拼进路径再判断
    # —— 后者一旦判断写漏就是一个任意文件读取（`../../.env` 那种）。
    # 白名单比对从源头上没有这个面。
    hit = next((f for f in entries if f["path"] == name), None)
    if hit is None:
        raise HTTPException(status_code=404, detail="该文件不在结果清单里")
    path = root / hit["path"]
    if not path.is_file():
        raise HTTPException(status_code=404, detail="文件已被移除")
    return FileResponse(str(path), filename=hit["name"])

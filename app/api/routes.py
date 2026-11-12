"""
FastAPI 路由定义

提供 RESTful API 接口：
- POST /api/chat          — 对话接口
- POST /api/model/run     — 运行建模
- GET  /api/model/status  — 查询建模状态
- GET  /api/model/result  — 获取建模结果
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


# --- 请求/响应模型 ---

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    history: list = []


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    json_ir: dict = {}
    missing_params: list = []


class ModelRunRequest(BaseModel):
    json_ir: dict
    session_id: str = "default"


class ModelRunResponse(BaseModel):
    task_id: str
    status: str  # queued | running | completed | failed


# --- 路由 ---

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    对话接口 — 处理用户自然语言输入

    流程：
    1. 意图解析 (IntentParser)
    2. 参数提取与JSON IR更新
    3. 缺失参数检测
    4. 生成响应

    TODO: 接入 LLM Agent 层
    """
    # 临时占位：回显用户输入
    echo_msg = f"[系统收到] {request.message}"
    return ChatResponse(
        reply=echo_msg,
        session_id=request.session_id,
        json_ir={},
        missing_params=[]
    )


@router.post("/model/run", response_model=ModelRunResponse)
async def run_model(request: ModelRunRequest):
    """
    运行建模接口

    接收校验通过的 JSON IR，触发 FloPy 自动建模

    TODO: 接入 FloPy 执行层
    """
    import uuid
    task_id = str(uuid.uuid4())[:8]
    return ModelRunResponse(
        task_id=task_id,
        status="queued"
    )


@router.get("/model/status/{task_id}")
async def model_status(task_id: str):
    """查询建模任务状态"""
    # TODO: 接入任务队列
    return {"task_id": task_id, "status": "unknown"}


@router.get("/model/result/{task_id}")
async def model_result(task_id: str):
    """获取建模结果"""
    # TODO: 接入结果存储
    raise HTTPException(status_code=404, detail="结果尚未生成")

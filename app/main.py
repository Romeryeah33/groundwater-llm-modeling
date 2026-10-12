"""
FastAPI 应用入口

**推荐启动方式**（网页与接口一起起来，一个进程）：
    python -m app.main
    → 网页 http://127.0.0.1:7860
    → 接口 http://127.0.0.1:8000/docs

只起接口（不起网页）：
    uvicorn app.main:app --host 127.0.0.1 --port 8000
"""

import logging
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes
from app.config import settings

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _start_gradio_in_background() -> None:
    """把 Gradio 网页在后台线程里拉起来，与接口共用**同一个进程**。

    为什么要在同一个进程：会话注册表（`session_registry`）与结果缓存都是
    进程内的。两个进程各跑一份，会出现"网页里聊出来的参数，接口里查不到"。
    """
    from app.frontend import gradio_app

    host = os.getenv("GW_WEB_HOST", "127.0.0.1")
    port = int(os.getenv("GW_WEB_PORT", "7860"))

    def _run():
        try:
            # prevent_thread_lock=True：launch() 立刻返回，不接管主线程
            gradio_app.launch_ui(prevent_thread_lock=True, server_name=host,
                                 server_port=port)
            logger.info("网页界面已启动: http://%s:%d", host, port)
        except Exception:
            logger.exception("网页界面启动失败（接口不受影响，继续提供）")

    threading.Thread(target=_run, name="gradio-ui", daemon=True).start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("=" * 60)
    logger.info("地下水智能建模系统启动中...")
    logger.info(f"LLM Provider: {settings.llm_provider}")
    logger.info(f"LLM Model: {settings.llm_model}")
    logger.info(f"输出目录: {settings.output_dir}")
    logger.info("=" * 60)

    # 只在直接跑 `python -m app.main` 时拉起网页；用 uvicorn 起纯接口时不拉
    # —— 部署时接口与网页可能由 nginx 分到不同的后端，那时不该多开一个网页。
    if os.getenv("GW_WITH_UI", "1").strip().lower() not in ("0", "false", "no", ""):
        _start_gradio_in_background()

    yield
    logger.info("系统关闭")


app = FastAPI(
    title="物理约束嵌入LLM的地下水智能建模系统",
    description="Physical-LLM Groundwater Intelligent Modeling System",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 中间件（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂上 /api/* —— 这一行是"接口真的存在"的最小必要条件。
# 此前 app/api/routes.py 写好了四个端点却**没有任何地方 include_router**，
# 于是 README 承诺的 API 从来没生效过（四个函数全是 TODO 占位）。
app.include_router(routes.router)


@app.get("/")
async def root():
    """健康检查"""
    return {
        "name": "地下水智能建模系统",
        "version": "1.0.0",
        "status": "running",
        "llm_model": settings.llm_model,
    }


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "ok"}


if __name__ == "__main__":
    # 一个进程同时提供网页与接口（见模块头的启动说明）
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("GW_API_HOST", "127.0.0.1"),
        port=int(os.getenv("GW_API_PORT", "8000")),
        log_level=settings.log_level.lower(),
    )

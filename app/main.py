"""
FastAPI 应用入口

启动方式：
    uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("=" * 60)
    logger.info("地下水智能建模系统启动中...")
    logger.info(f"LLM Provider: {settings.llm_provider}")
    logger.info(f"LLM Model: {settings.llm_model}")
    logger.info(f"输出目录: {settings.output_dir}")
    logger.info("=" * 60)
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

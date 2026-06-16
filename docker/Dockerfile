# Physical-LLM Groundwater Intelligent Modeling System
# Docker 镜像 — 独立运行版
FROM python:3.11-slim

LABEL maintainer="luomoye@nies.org"
LABEL description="物理约束嵌入LLM的地下水智能建模原型系统"

# 安装 MODFLOW 6 二进制文件
ARG MODFLOW_VERSION=6.4.4
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ca-certificates && \
    wget -q https://github.com/MODFLOW-USGS/modflow6/releases/download/${MODFLOW_VERSION}/mf6_${MODFLOW_VERSION}_linux64.zip && \
    unzip mf6_${MODFLOW_VERSION}_linux64.zip -d /usr/local/bin/ && \
    chmod +x /usr/local/bin/mf6 && \
    rm mf6_${MODFLOW_VERSION}_linux64.zip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖清单
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 初始化语义库
RUN python scripts/init_semantic_db.py

# 创建输出目录
RUN mkdir -p /app/data/outputs/workspaces /app/data/outputs/figures /app/data/outputs/reports

# 暴露端口
EXPOSE 7860 8000

# 环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_ROOT_PATH=""

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860')" || exit 1

# 默认启动 Gradio 前端
CMD ["python", "-m", "app.frontend.gradio_app"]

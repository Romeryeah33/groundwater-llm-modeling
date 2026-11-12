# ============================================================
# 地下水智能建模系统 — Docker 镜像
# 分发给其他用户使用，用户只需提供自己的 API Key
#
# 构建:   docker build -f docker/Dockerfile -t gw-modeling:v0.2.0 .
# 运行:   docker run -p 7860:7860 -e LLM_API_KEY=your-key gw-modeling:v0.2.0
# ============================================================
FROM python:3.13-slim

LABEL maintainer="luomoye@nies.org"
LABEL version="v0.2.0"
LABEL description="物理约束嵌入LLM的地下水智能建模系统"

# ── 系统依赖：MODFLOW 6 + GIS 库（geopandas/rasterio） ──
ARG MODFLOW_VERSION=6.4.4
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ca-certificates unzip \
    libgdal-dev gdal-bin \
    && \
    wget -q https://github.com/MODFLOW-USGS/modflow6/releases/download/${MODFLOW_VERSION}/mf6_${MODFLOW_VERSION}_linux64.zip && \
    unzip mf6_${MODFLOW_VERSION}_linux64.zip -d /usr/local/bin/ && \
    chmod +x /usr/local/bin/mf6 && \
    rm mf6_${MODFLOW_VERSION}_linux64.zip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python 依赖 ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── 应用代码 ──
COPY . .

# ── 初始化语义库 ──
RUN python scripts/init_semantic_db.py

# ── 数据与输出目录 ──
RUN mkdir -p /app/data/outputs/workspaces /app/data/outputs/figures /app/data/outputs/reports /app/data/uploads /app/data/sessions

EXPOSE 7860

ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV GRADIO_SERVER_NAME=0.0.0.0

# 默认启动 — 用户通过 -e 传入 LLM_API_KEY
CMD ["python", "-m", "app.frontend.gradio_app"]

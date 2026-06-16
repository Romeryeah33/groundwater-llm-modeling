#!/bin/bash
# ============================================================
# 地下水智能建模系统 — 一键部署脚本
# Usage: bash deploy/deploy.sh
# ============================================================
set -euo pipefail

echo "=== 地下水智能建模系统 - 云端部署 ==="
echo ""

# 切换到项目根目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# -----------------------------------------------------------
# 1. 检查 Docker
# -----------------------------------------------------------
echo "[1/5] 检查 Docker..."
if ! command -v docker &> /dev/null; then
    echo "错误: Docker 未安装。请先运行: bash deploy/setup-server.sh"
    exit 1
fi

if ! docker compose version &> /dev/null 2>&1; then
    echo "错误: Docker Compose 插件未安装。请先运行: bash deploy/setup-server.sh"
    exit 1
fi

# -----------------------------------------------------------
# 2. 检查 .env 配置
# -----------------------------------------------------------
echo "[2/5] 检查配置..."
if [ ! -f ".env" ]; then
    echo "未找到 .env 文件，正在从模板创建..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "已创建 .env 文件，请编辑填入 API Key："
        echo "  nano .env"
        echo ""
        read -rp "编辑完成后按回车继续..."
    else
        echo "错误: .env.example 也不存在，请手动创建 .env 文件"
        exit 1
    fi
fi

# 检查 API Key 是否已填写
if grep -q "your_api_key_here" .env 2>/dev/null; then
    echo "警告: .env 中 API Key 未填写！"
    echo "请编辑 .env 文件填入 DeepSeek API Key："
    echo "  nano .env"
    echo ""
    read -rp "编辑完成后按回车继续..."
fi

# -----------------------------------------------------------
# 3. 构建镜像
# -----------------------------------------------------------
echo "[3/5] 构建 Docker 镜像..."
docker compose -f docker/docker-compose.yml build --no-cache

# -----------------------------------------------------------
# 4. 启动服务
# -----------------------------------------------------------
echo "[4/5] 启动服务..."
# 先停止旧容器
docker compose -f docker/docker-compose.yml down 2>/dev/null || true
# 启动
docker compose -f docker/docker-compose.yml up -d

# -----------------------------------------------------------
# 5. 验证
# -----------------------------------------------------------
echo "[5/5] 健康检查..."
sleep 5

if curl -sf -o /dev/null http://127.0.0.1:7860 2>&1; then
    echo ""
    echo "=========================================="
    echo "  ✅ 部署成功！"
    echo "  Gradio 应用运行在 http://127.0.0.1:7860"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "  ⚠️ 健康检查未通过，查看日志："
    echo "  docker compose -f docker/docker-compose.yml logs --tail=50"
    echo "=========================================="
fi

# 显示容器状态
echo ""
echo "容器状态:"
docker compose -f docker/docker-compose.yml ps

echo ""
echo "实时日志: docker compose -f docker/docker-compose.yml logs -f"
echo "重启服务: docker compose -f docker/docker-compose.yml restart"
echo "停止服务: docker compose -f docker/docker-compose.yml down"

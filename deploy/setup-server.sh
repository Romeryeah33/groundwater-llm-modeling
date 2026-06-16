#!/bin/bash
# ============================================================
# 地下水智能建模系统 — 服务器初始化脚本
# 在全新 Ubuntu 22.04 服务器上运行一次即可
# Usage: sudo bash setup-server.sh
# ============================================================
set -euo pipefail

echo "=== 地下水智能建模系统 - 服务器初始化 ==="
echo ""

# -----------------------------------------------------------
# 1. 系统更新
# -----------------------------------------------------------
echo "[1/6] 更新系统..."
apt-get update && apt-get upgrade -y

# -----------------------------------------------------------
# 2. 安装必要软件
# -----------------------------------------------------------
echo "[2/6] 安装必要软件..."
apt-get install -y --no-install-recommends \
    nginx \
    certbot \
    python3-certbot-nginx \
    curl \
    git \
    ufw \
    apache2-utils

# -----------------------------------------------------------
# 3. 安装 Docker
# -----------------------------------------------------------
echo "[3/6] 安装 Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com | sh
fi

# 安装 Docker Compose 插件
if ! docker compose version &> /dev/null 2>&1; then
    apt-get install -y docker-compose-plugin
fi

# -----------------------------------------------------------
# 4. 配置防火墙
# -----------------------------------------------------------
echo "[4/6] 配置防火墙..."
ufw allow 22/tcp comment "SSH"
ufw allow 80/tcp comment "HTTP"
ufw allow 443/tcp comment "HTTPS"
ufw --force enable
ufw status verbose

# -----------------------------------------------------------
# 5. 启用自动启动
# -----------------------------------------------------------
echo "[5/6] 配置开机自启..."
systemctl enable docker
systemctl enable nginx

# -----------------------------------------------------------
# 6. 设置时区
# -----------------------------------------------------------
echo "[6/6] 设置时区..."
timedatectl set-timezone Asia/Shanghai

echo ""
echo "=== 初始化完成！ ==="
echo "接下来请："
echo "  1. 配置域名 DNS A 记录指向本服务器 IP"
echo "  2. 运行: sudo certbot --nginx -d your-domain.com"
echo "  3. 复制 nginx.conf 到 /etc/nginx/sites-available/"
echo "  4. 运行 deploy.sh 部署应用"

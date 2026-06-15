@echo off
REM ============================================================
REM 地下水智能建模系统 — Windows 一键启动脚本
REM ============================================================
chcp 65001 > nul
echo.
echo ============================================================
echo   Physical-LLM Groundwater Intelligent Modeling System
echo   物理约束嵌入LLM的地下水智能建模原型系统 (MVP)
echo ============================================================
echo.

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.11+.
    pause
    exit /b 1
)

REM 检查依赖
echo [1/3] Checking dependencies...
pip show gradio >nul 2>&1
if errorlevel 1 (
    echo        Installing dependencies...
    pip install -r requirements.txt -q
)

REM 初始化语义库（如果不存在）
echo [2/3] Checking semantic database...
if not exist "..\data\semantic_db.sqlite" (
    echo        Initializing semantic database...
    python ..\scripts\init_semantic_db.py
)

REM 启动
echo [3/3] Starting Gradio frontend...
echo.
echo   Open http://127.0.0.1:7860 in your browser
echo   Press Ctrl+C to stop
echo ============================================================
echo.

python -m app.frontend.gradio_app

pause

@echo off
REM ============================================================
REM  STALE / UNREFERENCED -- the launcher users actually double-click is
REM  the Chinese-named one in the repo root. This file is kept only for
REM  reference; it still starts the web UI alone (no REST API).
REM
REM  NOTE ON THE TWO LINES OF CHINESE BELOW: cmd.exe reads a batch file
REM  using the SYSTEM CODEPAGE (GBK on Chinese Windows), so UTF-8 Chinese
REM  outside a UTF-8 codepage gets decoded into garbage -- and some of
REM  those byte pairs derail the parser for the whole file. That is why
REM  `chcp 65001` MUST come before the first Chinese byte. It is placed
REM  here, ahead of everything else, for exactly that reason.
REM ============================================================
chcp 65001 > nul
echo.
echo ============================================================
echo   Physical-LLM Groundwater Intelligent Modeling System
echo   物理约束嵌入LLM的地下水智能建模原型系统 (MVP)
echo ============================================================
echo.

REM check python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.11+.
    pause
    exit /b 1
)

REM check dependencies
echo [1/3] Checking dependencies...
pip show gradio >nul 2>&1
if errorlevel 1 (
    echo        Installing dependencies...
    pip install -r requirements.txt -q
)

REM init semantic db if missing
echo [2/3] Checking semantic database...
if not exist "..\data\semantic_db.sqlite" (
    echo        Initializing semantic database...
    python ..\scripts\init_semantic_db.py
)

REM start
echo [3/3] Starting Gradio frontend...
echo.
echo   Open http://127.0.0.1:7860 in your browser
echo   Press Ctrl+C to stop
echo ============================================================
echo.

python -m app.frontend.gradio_app

pause

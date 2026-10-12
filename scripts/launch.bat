@echo off
REM ============================================================
REM  Real launcher logic -- lives here, under an ASCII filename.
REM
REM  The user-facing entry points are the two Chinese-named scripts in
REM  the repo root ("start.bat" / "start-debug.bat"). They contain nothing
REM  but a `call` to this file.
REM
REM  WHY the split: cmd.exe reads a batch file using the SYSTEM CODEPAGE
REM  (GBK on Chinese Windows), not UTF-8. So a batch file whose name -- or
REM  whose body -- is Chinese cannot reliably `call` another Chinese-named
REM  batch file: the bytes get decoded as garbage and you get
REM  "'...start.bat' is not recognized". Tested, it fails.
REM  Referencing an ASCII path always works, so every cross-script reference
REM  goes through this ASCII file and the Chinese names stay entry-point only.
REM
REM  KEEP THIS FILE PURE ASCII -- see the note above. The one thing that is
REM  NOT allowed here is a well-meaning Chinese comment.
REM ============================================================
cd /d "%~dp0.."
title GW Modeling System
setlocal enabledelayedexpansion

echo ==========================================
echo    GW Modeling System v0.4.4
echo ==========================================
echo.

REM -- Find a Python interpreter that ACTUALLY HAS the dependencies --
REM Do NOT just take the first interpreter that exists on a hard-coded path
REM list. A Python installed later (e.g. 3.13) may sort first yet have none
REM of the packages, which fails the launch and then hangs the port wait.
set "PYTHON="
set "CANDIDATES="
set CANDIDATES=%CANDIDATES% "C:\Program Files\Python311\python.exe"
set CANDIDATES=%CANDIDATES% "C:\Program Files\Python312\python.exe"
set CANDIDATES=%CANDIDATES% "C:\Program Files\Python313\python.exe"
set CANDIDATES=%CANDIDATES% "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
set CANDIDATES=%CANDIDATES% "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
set CANDIDATES=%CANDIDATES% "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
set CANDIDATES=%CANDIDATES% "C:\Python311\python.exe"
set CANDIDATES=%CANDIDATES% "C:\Python312\python.exe"
set CANDIDATES=%CANDIDATES% "C:\Python313\python.exe"

for %%p in (%CANDIDATES%) do (
    if exist %%p (
        "%%~p" -c "import gradio, flopy, geopandas" >nul 2>&1
        if !errorlevel! equ 0 (
            set "PYTHON=%%~p"
            goto :found_python
        ) else (
            echo   [skip] %%~p  -- dependencies missing
        )
    )
)

REM -- Last resort: whatever "python" resolves to on PATH --
python -c "import gradio, flopy, geopandas" >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON=python"
    goto :found_python
)

echo.
echo [ERROR] No Python interpreter with gradio / flopy / geopandas found.
echo.
echo Install the dependencies into the interpreter you want to use, e.g.:
echo     "C:\Program Files\Python311\python.exe" -m pip install -r requirements.txt
echo Then run this script again.
echo.
pause
exit /b 1

:found_python
echo Python: "%PYTHON%"
"%PYTHON%" --version

REM -- Check .env --
if not exist ".env" (
    echo.
    echo [WARNING] .env file not found.
    echo Please create .env with your API Key:
    echo   LLM_API_KEY=your-api-key-here
    echo   LLM_MODEL=deepseek-chat
    echo   LLM_API_BASE=https://api.deepseek.com/v1
    echo.
)

if "%GW_SHOW_DEBUG_PANELS%"=="1" (
    echo.
    echo [DEBUG MODE] debug panels and the styling tab will be visible.
)

echo.
echo ==========================================
echo   Starting server, please wait...
echo   (first load may take 90s for LiteLLM init)
echo ==========================================

REM -- Start server in background --
REM Unified entry point: ONE process serves both the web UI (7860) and the
REM REST API (8000). They share a single session table, so parameters you
REM built up in the chat window can be run straight through the API.
start "GW Server" /min "%PYTHON%" -m app.main

REM -- Wait for port 7860, but give up eventually so the real error is visible --
echo Waiting for server to be ready...
set /a TRIES=0
:wait_port
ping -n 3 127.0.0.1 >nul 2>&1
set /a TRIES+=1
powershell -Command "try { $c = New-Object System.Net.Sockets.TcpClient('127.0.0.1', 7860); $c.Close(); exit 0 } catch { exit 1 }" >nul 2>&1
if %errorlevel% equ 0 goto :ready
if %TRIES% geq 60 (
    echo.
    echo.
    echo [ERROR] Server did not come up within 120 seconds.
    echo Check the minimized "GW Server" window for the traceback.
    echo.
    pause
    exit /b 1
)
echo|set /p="."
goto :wait_port

:ready
echo.
echo ==========================================
echo   Server is ready! Opening browser...
echo   Web UI       : http://127.0.0.1:7860
echo   REST API docs: http://127.0.0.1:8000/docs
echo ==========================================

start "" http://127.0.0.1:7860

echo.
echo Press Ctrl+C or close this window to stop the server.
echo ==========================================
pause

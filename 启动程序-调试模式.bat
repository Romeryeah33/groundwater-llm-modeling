@echo off
REM ============================================================
REM  调试模式启动 —— 与「启动程序.bat」唯一差别是多显示两个面板：
REM     * 左侧「流水线追踪 (debug log)」折叠面板
REM     * 顶部「图件修改」Tab
REM
REM  为什么要分开：这两个都挂在**全局单例**上（monitor / plot_styler），
REM  多人同时用会看到彼此的内容。所以分发给同事时用「启动程序.bat」
REM  （默认隐藏），只有自己排查问题才用这个。
REM
REM  实现方式：只设一个环境变量，再调用原脚本。不复制原脚本内容 ——
REM  复制一份的话，以后改原脚本就得记得改两处。
REM ============================================================
cd /d "%~dp0"
title GW Modeling System (DEBUG)

set GW_SHOW_DEBUG_PANELS=1
echo [DEBUG] GW_SHOW_DEBUG_PANELS=1 -- 调试面板与「图件修改」Tab 将显示
echo.

call "%~dp0启动程序.bat"

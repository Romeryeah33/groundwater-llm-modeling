@echo off
REM Double-click entry point -- DEBUG mode. Shows two panels that are hidden
REM in the normal launcher:
REM   * the "pipeline trace (debug log)" accordion on the left
REM   * the "plot styling" tab in the results area
REM
REM Why they are hidden by default: both sit on top of PROCESS-GLOBAL
REM singletons (monitor / plot_styler), so when several people use the site
REM at once they end up seeing each other's logs and plot state. Hand out
REM "start.bat" to colleagues; use this one only to troubleshoot your own
REM machine.
REM
REM Implementation: set one environment variable, then call the shared
REM launcher (ASCII path -- see the note in scripts\launch.bat for why a
REM batch file must not reference a Chinese-named one).
set GW_SHOW_DEBUG_PANELS=1
echo [DEBUG] GW_SHOW_DEBUG_PANELS=1 -- debug panels will be visible
echo.
call "%~dp0scripts\launch.bat"

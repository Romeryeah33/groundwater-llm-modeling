@echo off
REM Double-click entry point -- ordinary mode (debug panels hidden).
REM All the real logic is in scripts\launch.bat (ASCII filename) so that the
REM two launchers can share it without one Chinese-named batch file having to
REM call another, which cmd.exe cannot do reliably.
call "%~dp0scripts\launch.bat"

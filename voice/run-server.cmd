@echo off
REM Launch voice/server.py in a real console.
REM
REM Earlier attempts to run via `Start-Process -WindowStyle Hidden` or via
REM SSH-detached background mode produced silent uvicorn exits. uvicorn /
REM asyncio behave differently when stdout is a pipe vs a console — opening
REM a proper console window (even minimized) is enough to let it bind.
REM
REM Usage:  scripts call this as:  start "" /MIN run-server.cmd
REM         or just double-click it.

cd /d "%~dp0"
set PYTHONUNBUFFERED=1
.venv\Scripts\python.exe -u -m voice.server
echo.
echo === voice.server exited.  Press any key to close. ===
pause >nul

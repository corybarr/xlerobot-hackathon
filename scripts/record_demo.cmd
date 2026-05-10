@echo off
REM Demo capture helper.
REM Launches the orchestrator with a clean terminal output suitable for
REM screen-recording, and reminds you what to capture in parallel.
REM
REM Run this in a fresh PowerShell window with the terminal made
REM full-screen first. OBS or Windows Game Bar (Win+G) records the screen.
REM Phone on tripod records the arm.

echo === xlerobot demo capture ===
echo.
echo Pre-flight (do NOW, before pressing any key):
echo.
echo   1. Phone on tripod, framing the arm + workspace
echo   2. Camera on, recording (note start timestamp)
echo   3. OBS or Win+G screen capture armed and recording
echo   4. Terminal full-screen (this window)
echo   5. Cup, cutlery, and bowl placed on table within reach
echo   6. Arm in a known starting pose
echo.
echo When everything is rolling, press any key to launch the orchestrator.
echo Voiceover starts as soon as the first 'gemma decision' line appears.
echo.
pause >nul

REM Pull token from local file (gitignored)
if not defined GEMMA_PROXY_TOKEN (
    if exist "%USERPROFILE%\.gemma_token" (
        for /f "usebackq delims=" %%t in ("%USERPROFILE%\.gemma_token") do set GEMMA_PROXY_TOKEN=%%t
    )
)

REM Tailscale URL is the demo path (35ms latency vs bore's flakiness)
set OLLAMA_HOST=http://100.127.125.42:18080
set GEMMA_MODEL=gemma3:27b
set GOAL=set the table
set FOLLOWER_PORT=COM10
set LEADER_PORT=COM7

REM Use the unbuffered python flag so log lines appear immediately on screen
echo.
echo === orchestrator starting ===
echo Goal: %GOAL%
echo Gemma: %OLLAMA_HOST%
echo.

cd /d "%~dp0\.."
"%USERPROFILE%\miniconda3\envs\lerobot\python.exe" -u orchestrator/orchestrator.py

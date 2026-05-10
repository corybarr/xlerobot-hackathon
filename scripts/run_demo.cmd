@echo off
REM One-command demo launcher.
REM Reads GEMMA_PROXY_TOKEN from %USERPROFILE%\.gemma_token (mode 600).
REM Bakes in the Tailscale URL for the Spark Gemma proxy (35ms vs bore's
REM seconds-or-timeout). Override any var by setting it before calling.

if not defined OLLAMA_HOST set OLLAMA_HOST=http://100.127.125.42:18080
if not defined GEMMA_MODEL set GEMMA_MODEL=gemma3:27b
if not defined FOLLOWER_PORT set FOLLOWER_PORT=COM10
if not defined LEADER_PORT set LEADER_PORT=COM7
if not defined GOAL set GOAL=set the table

if not defined GEMMA_PROXY_TOKEN (
    if exist "%USERPROFILE%\.gemma_token" (
        for /f "usebackq delims=" %%t in ("%USERPROFILE%\.gemma_token") do set GEMMA_PROXY_TOKEN=%%t
    )
)
if not defined GEMMA_PROXY_TOKEN (
    echo ERROR: GEMMA_PROXY_TOKEN not set and ~/.gemma_token missing.
    echo Run scripts/deploy_gemma_proxy.sh on Spark to mint one.
    exit /b 1
)

echo === xlerobot-hackathon demo ===
echo   goal:           %GOAL%
echo   gemma host:     %OLLAMA_HOST%
echo   follower port:  %FOLLOWER_PORT%
echo   leader port:    %LEADER_PORT%
echo.

cd /d "%~dp0\.."
"%USERPROFILE%\miniconda3\envs\lerobot\python.exe" orchestrator/orchestrator.py

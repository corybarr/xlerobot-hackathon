@echo off
REM Thin Windows wrapper so you can type `scripts\mm.cmd <subcommand>` instead
REM of the full conda+python path. Source: scripts\mm.py
"%USERPROFILE%\miniconda3\envs\lerobot\python.exe" "%~dp0mm.py" %*

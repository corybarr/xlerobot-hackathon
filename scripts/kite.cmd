@echo off
REM Thin Windows wrapper. Reads KITE_TOKEN from env or ~/.kite_token.
"%USERPROFILE%\miniconda3\envs\lerobot\python.exe" "%~dp0kite.py" %*

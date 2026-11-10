@echo off
title Mirror Crystal Live Studio
cd /d "%~dp0"

echo =======================================================
echo   Starting Mirror Crystal Live Studio (Port 8787)...
echo   Zero-GPU Weightless Cognitive Engine
echo =======================================================
echo.

if exist ".venv\Scripts\python.exe" (
    set "PY_BIN=.venv\Scripts\python.exe"
) else (
    set "PY_BIN=python"
)

echo Opening browser at http://127.0.0.1:8787/ ...
start "" "http://127.0.0.1:8787/"

"%PY_BIN%" -m mirror7_backend.http_server
pause

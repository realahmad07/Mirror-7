@echo off
title Mirror Crystal Live Studio
set "SUITE_DIR=%~dp0"
for %%I in ("%SUITE_DIR%..\..") do set "REPO_ROOT=%%~fI"

echo =======================================================
echo   Starting Mirror Crystal Live Studio (Port 8787)...
echo =======================================================
echo.

if exist "%REPO_ROOT%\.venv\Scripts\python.exe" (
    set "PY_BIN=%REPO_ROOT%\.venv\Scripts\python.exe"
) else (
    set "PY_BIN=python"
)

set "PYTHONPATH=%REPO_ROOT%;%PYTHONPATH%"
pushd "%REPO_ROOT%"
echo Opening browser at http://127.0.0.1:8787/ ...
start "" "http://127.0.0.1:8787/"
"%PY_BIN%" -m mirror7_backend.http_server
set "SERVER_EXIT=%ERRORLEVEL%"
popd
pause
exit /b %SERVER_EXIT%

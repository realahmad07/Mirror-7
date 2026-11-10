@echo off
title Mirror 7 // Interactive Terminal Console
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PY_BIN=.venv\Scripts\python.exe"
) else (
    set "PY_BIN=python"
)

if /I "%PY_BIN%"=="python" (
    where python >nul 2>nul
    if errorlevel 1 (
        echo Python 3.11 or newer was not found. Install Python or create .venv.
        pause
        exit /b 1
    )
)

"%PY_BIN%" -u "%~dp0mirror_cli.py" %*
set "CLI_EXIT=%ERRORLEVEL%"
if not "%CLI_EXIT%"=="0" (
    echo Mirror 7 CLI exited with code %CLI_EXIT%.
)
pause
exit /b %CLI_EXIT%

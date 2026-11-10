@echo off
title Mirror 7 // Interactive Terminal Console
set "SUITE_DIR=%~dp0"
set "REPO_ROOT=%SUITE_DIR%..\.."
if exist "%REPO_ROOT%\.venv\Scripts\python.exe" (
    set "PY_BIN=%REPO_ROOT%\.venv\Scripts\python.exe"
) else if exist "%SUITE_DIR%\.venv\Scripts\python.exe" (
    set "PY_BIN=%SUITE_DIR%\.venv\Scripts\python.exe"
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

"%PY_BIN%" -u "%SUITE_DIR%tools\mirror_cli.py" %*
set "CLI_EXIT=%ERRORLEVEL%"
if not "%CLI_EXIT%"=="0" (
    echo Mirror 7 CLI exited with code %CLI_EXIT%.
)
pause
exit /b %CLI_EXIT%

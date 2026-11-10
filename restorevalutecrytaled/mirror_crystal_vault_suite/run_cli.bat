@echo off
title Mirror 7 // Interactive Terminal Console
set "SUITE_DIR=%~dp0"

if exist "%SUITE_DIR%..\..\.venv\Scripts\python.exe" (
    set "PY_BIN=%SUITE_DIR%..\..\.venv\Scripts\python.exe"
) else if exist "%SUITE_DIR%.venv\Scripts\python.exe" (
    set "PY_BIN=%SUITE_DIR%.venv\Scripts\python.exe"
) else (
    set "PY_BIN=python"
)

"%PY_BIN%" "%SUITE_DIR%tools\mirror_cli.py"
pause

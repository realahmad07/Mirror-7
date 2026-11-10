@echo off
title Mirror 7 // Interactive Terminal Console
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PY_BIN=.venv\Scripts\python.exe"
) else (
    set "PY_BIN=python"
)

"%PY_BIN%" mirror_cli.py
pause

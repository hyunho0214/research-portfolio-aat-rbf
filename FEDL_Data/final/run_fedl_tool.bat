@echo off
cd /d "%~dp0"

if exist "..\\.venv\\Scripts\\python.exe" (
    set PYTHON=..\.venv\Scripts\python.exe
) else (
    set PYTHON=python
)

%PYTHON% app.py
if errorlevel 1 (
    echo.
    echo [오류 발생] 위 내용을 확인하세요.
    pause
)

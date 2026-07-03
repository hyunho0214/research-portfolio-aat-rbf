@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "VENV_PYTHON=%SCRIPT_DIR%..\.venv\Scripts\python.exe"
cd /d "%SCRIPT_DIR%"

if exist "%VENV_PYTHON%" (
    "%VENV_PYTHON%" "%SCRIPT_DIR%plotting_gui.py"
) else (
    py -3 "%SCRIPT_DIR%plotting_gui.py"
)

if errorlevel 1 (
    echo.
    echo [ERROR] GUI launch failed. Press any key to close.
    pause >nul
)

endlocal
@echo off
setlocal
cd /d "%~dp0"

set "VENV_PYTHON=%~dp0..\.venv\Scripts\python.exe"

if exist "%VENV_PYTHON%" (
    "%VENV_PYTHON%" raw_data_gui.py
) else (
    py -3 raw_data_gui.py
)

if errorlevel 1 (
    echo.
    echo [ERROR] GUI launch failed. Press any key to close.
    pause >nul
)

endlocal

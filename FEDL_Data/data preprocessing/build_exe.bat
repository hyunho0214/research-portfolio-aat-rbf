@echo off
setlocal
cd /d "%~dp0"

echo [1/3] Installing or updating PyInstaller...
py -3 -m pip install --upgrade pyinstaller
if errorlevel 1 goto :err

echo [2/3] Building RawDataExtractor.exe...
py -3 -m PyInstaller --noconfirm --clean --onefile --windowed --name RawDataExtractor raw_data_gui.py
if errorlevel 1 goto :err

echo [3/3] Build complete.
echo exe path: %CD%\dist\RawDataExtractor.exe
echo.
echo Distribution tip: share dist\RawDataExtractor.exe for a one-file Windows GUI.
goto :end

:err
echo [ERROR] Build failed.
exit /b 1

:end
endlocal

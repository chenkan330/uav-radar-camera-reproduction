@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Project Python environment is missing.
    pause
    exit /b 1
)
if "%~1"=="" (
    echo Usage: run_mmaud_comparison.cmd --clip data\mmaud\prepared_clip
    echo A verified real clip and independent calibration are required.
    pause
    exit /b 2
)
".venv\Scripts\python.exe" "run_mmaud_comparison.py" %*
set "mmaud_run_code=%ERRORLEVEL%"
echo.
if "%mmaud_run_code%"=="0" echo Completed. Check output\v16_comparison\report.md.
pause
exit /b %mmaud_run_code%

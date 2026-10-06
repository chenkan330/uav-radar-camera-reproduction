@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo The project Python environment is missing.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" "demo_step1.py" --output "output\local_step1" %*
set "uav_run_code=%ERRORLEVEL%"
echo.
if "%uav_run_code%"=="0" echo Completed. Results are saved in output\local_step1.
pause
exit /b %uav_run_code%

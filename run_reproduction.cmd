@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo The project Python environment is missing. See README.md.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" "run_reproduction.py" --synthetic --output "output\local_reproduction" %*
set "uav_run_code=%ERRORLEVEL%"
echo.
if "%uav_run_code%"=="0" echo Completed. Open output\local_reproduction\report.md or comparison.png.
pause
exit /b %uav_run_code%

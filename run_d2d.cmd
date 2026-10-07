@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo The v2 project environment is missing. See README.md.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" "demo_d2d.py" --output "output\local_d2d" %*
set "d2d_run_code=%ERRORLEVEL%"
echo.
if "%d2d_run_code%"=="0" echo Completed. Open output\local_d2d\report.md.
pause
exit /b %d2d_run_code%

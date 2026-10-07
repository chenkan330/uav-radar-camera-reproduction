@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set "PYTHONUTF8=1"
if not exist ".venv\Scripts\python.exe" (
    echo 缺少本项目的 Python 环境，请参阅 docs\v1_6_run.md。
    pause
    exit /b 1
)
if not exist "data\public\mmaud\prepared_v16\clip.json" (
    echo 缺少已核验的真实片段，请参阅 docs\v1_6_run.md。
    pause
    exit /b 2
)
".venv\Scripts\python.exe" "run_mmaud_comparison.py" --clip "data\public\mmaud\prepared_v16"
if errorlevel 1 goto failed
".venv\Scripts\python.exe" "tools\evaluate_mmaud_sensitivity.py"
if errorlevel 1 goto failed
echo 对照与标定敏感性分析完成。报告位于 output\v16_comparison。
pause
exit /b 0
:failed
echo 运行未完成，请查看上面的原因。未使用合成数据替代真实数据。
pause
exit /b 1

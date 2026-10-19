@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0.runtime\python\python.exe" (
    echo Bundled Python not found. Run setup.bat first.
    pause
    exit /b 1
)
call "%~dp0packaging\runtime_env.bat"
"%~dp0.runtime\python\python.exe" -X utf8 -B "%~dp0tools\check_deployment.py" --models
set "CHECK_EXIT=%ERRORLEVEL%"
pause
exit /b %CHECK_EXIT%

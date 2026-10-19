@echo off
setlocal
cd /d "%~dp0"

if exist "%~dp0.runtime\python\python.exe" (
    call "%~dp0packaging\runtime_env.bat"
    goto portable
)

where conda >nul 2>nul
if errorlevel 1 (
    echo Conda is not on PATH. Activate the consult_asr_fast environment, then run consult_asr.py.
    pause
    exit /b 1
)

call conda run -n consult_asr_fast python "%~dp0consult_asr.py"
goto finished
:portable
"%~dp0.runtime\python\python.exe" -X utf8 -B "%~dp0consult_asr.py"
:finished
set "APP_EXIT=%ERRORLEVEL%"
if not "%APP_EXIT%"=="0" (
    echo The transcription app exited with code %APP_EXIT%.
    pause
)
exit /b %APP_EXIT%

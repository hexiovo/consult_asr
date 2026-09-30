@echo off
setlocal
cd /d "%~dp0"

where conda >nul 2>nul
if errorlevel 1 (
    echo Conda is not on PATH. Create or activate the consult_asr_fast environment first.
    pause
    exit /b 1
)

echo Preparing only the gated pyannote speaker-diarization models.
echo The ASR model is already stored locally. Your token input will be hidden and will not be saved.
call conda run --no-capture-output -n consult_asr_fast python "%~dp0tools\prepare_models.py" --diarization --prompt-token
set "PREP_EXIT=%ERRORLEVEL%"
if "%PREP_EXIT%"=="0" (
    echo.
    echo Model preparation completed. You can now run run_consult_asr.bat.
) else (
    echo.
    echo Model preparation stopped with code %PREP_EXIT%. Review the access message above.
)
pause
exit /b %PREP_EXIT%

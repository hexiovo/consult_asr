@echo off
rem Called by portable entry points; paths are relative to the application root.
set "APP_ROOT=%~dp0.."
set "CONDA_PREFIX=%APP_ROOT%\.runtime\python"
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONNOUSERSITE=1"
set "CONSULT_ASR_FFMPEG=%APP_ROOT%\.runtime\ffmpeg\ffmpeg.exe"
set "PATH=%APP_ROOT%\.runtime\ffmpeg;%CONDA_PREFIX%;%CONDA_PREFIX%\Scripts;%CONDA_PREFIX%\Library\bin;%CONDA_PREFIX%\Lib\site-packages\torch\lib;%PATH%"

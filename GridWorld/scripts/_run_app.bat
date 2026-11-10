@echo off
setlocal
pushd "%~dp0.."
if errorlevel 1 exit /b 1
if defined GRAG_PYTHON (
    set "GRAG_APP_PYTHON=%GRAG_PYTHON%"
) else if exist "..\venv\Scripts\python.exe" (
    set "GRAG_APP_PYTHON=%CD%\..\venv\Scripts\python.exe"
) else (
    set "GRAG_APP_PYTHON=python"
)
echo Open %~2 in your browser. Press Ctrl+C to stop.
"%GRAG_APP_PYTHON%" -m %~1
set "GRAG_APP_EXIT=%errorlevel%"
if not "%GRAG_APP_EXIT%"=="0" pause
popd
exit /b %GRAG_APP_EXIT%


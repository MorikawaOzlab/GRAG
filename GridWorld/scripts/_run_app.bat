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
set "PYTHONPATH=%CD%;%CD%\..;%PYTHONPATH%"
if /i "%~3"=="chrome" start "" /b "%GRAG_APP_PYTHON%" "%~dp0_open_training_chrome.py" "%~2"
"%GRAG_APP_PYTHON%" -c "import sys,urllib.request; assert b'GridWorld Lab' in urllib.request.urlopen(sys.argv[1], timeout=2).read()" "%~2" >nul 2>&1
if not errorlevel 1 goto already_running
echo Open %~2 in your browser. Press Ctrl+C to stop.
"%GRAG_APP_PYTHON%" -u -m %~1
set "GRAG_APP_EXIT=%errorlevel%"
if not "%GRAG_APP_EXIT%"=="0" pause
popd
exit /b %GRAG_APP_EXIT%

:already_running
echo GridWorld is already running at %~2
echo Open that URL in your browser. No second server was started.
popd
exit /b 0


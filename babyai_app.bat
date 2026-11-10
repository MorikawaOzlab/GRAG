@echo off
setlocal
pushd "%~dp0"
if defined GRAG_PYTHON (
  set "BABYAI_PYTHON=%GRAG_PYTHON%"
) else if exist "venv\Scripts\python.exe" (
  set "BABYAI_PYTHON=%CD%\venv\Scripts\python.exe"
) else (
  set "BABYAI_PYTHON=python"
)
"%BABYAI_PYTHON%" -m babyai.app
popd
pause

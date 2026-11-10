@echo off
call "%~dp0_run_app.bat" gridworld.settings_app http://127.0.0.1:5002
exit /b %errorlevel%

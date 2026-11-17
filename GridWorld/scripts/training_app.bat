@echo off
call "%~dp0_run_app.bat" gridworld.training_ui http://127.0.0.1:5003 chrome
exit /b %errorlevel%

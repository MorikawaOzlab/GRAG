@echo off
call "%~dp0_run_app.bat" gridworld.training_app http://127.0.0.1:5003
exit /b %errorlevel%

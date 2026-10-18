@echo off
call "%~dp0_run_app.bat" gridworld.map_app http://127.0.0.1:5001
exit /b %errorlevel%

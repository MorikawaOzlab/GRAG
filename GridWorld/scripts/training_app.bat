@echo off
start "GridWorld Map" /min "%ComSpec%" /d /c ""%~dp0map_app.bat""
start "GridWorld Settings" /min "%ComSpec%" /d /c ""%~dp0settings_app.bat""
call "%~dp0_run_app.bat" gridworld.training_ui http://127.0.0.1:5003 chrome
exit /b %errorlevel%

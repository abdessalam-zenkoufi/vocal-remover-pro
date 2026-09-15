@echo off
title Launching VoxMux Pro

echo ====================================================
echo [*] Launching Application via Virtual Environment...
echo ====================================================

echo [*] Checking Virtual Environment (venv)...
if exist venv goto ACTIVATE_AND_RUN

echo [ERROR] Virtual environment (venv) not found!
echo Please run 'Setup_Environment.bat' first to initialize the system.
echo.
pause
exit /b

:ACTIVATE_AND_RUN
echo [*] Activating environment and starting VoxMux Pro...
call venv\Scripts\activate

:: تشغيل الكود الرئيسي للبرنامج
python main.py

if errorlevel 1 goto APP_CRASHED
exit /b

:APP_CRASHED
echo.
echo [ERROR] Application crashed or exited with an error code.
pause
exit /b
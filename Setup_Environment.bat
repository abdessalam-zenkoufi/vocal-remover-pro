@echo off
title VoxMux Pro - Environment Setup

echo ====================================================
echo [VoxMux Pro] - Environment Setup
echo ====================================================

echo [*] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 goto NO_PYTHON

echo [*] Checking Virtual Environment (venv)...
if exist venv goto VENV_EXISTS

echo [*] Creating Virtual Environment...
python -m venv venv
echo [OK] Virtual Environment created.
goto ACTIVATE_VENV

:VENV_EXISTS
echo [INFO] Virtual Environment already exists.

:ACTIVATE_VENV
echo [*] Activating Virtual Environment...
call venv\Scripts\activate

echo [*] Upgrading pip...
python -m pip install --upgrade pip

echo [*] Installing PyTorch with CUDA 12.1 (GPU) support...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo [*] Installing dependencies from requirements.txt...
if exist requirements.txt (
    pip install -r requirements.txt
    echo [OK] All dependencies installed successfully.
) else (
    echo [WARNING] requirements.txt not found! Cannot install dependencies.
)

echo ====================================================
echo [SUCCESS] Setup Complete! You can now close this window.
echo ====================================================
pause
exit /b

:NO_PYTHON
echo [ERROR] Python is not installed or not added to PATH!
pause
exit /b
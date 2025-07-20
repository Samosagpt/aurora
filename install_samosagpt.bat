@echo off
title SamosaGPT Installer
echo.
echo ========================================
echo    SamosaGPT Installer
echo    Advanced AI Assistant Installation
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo.
    echo Please install Python 3.8 or higher from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

REM Check if Git is available
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Git is not installed or not in PATH
    echo.
    echo Please install Git from:
    echo https://git-scm.com/download/win
    echo.
    pause
    exit /b 1
)

echo Requirements check passed...
echo.
echo Starting SamosaGPT Installer...
echo.

REM Install required packages for the installer
echo Installing installer dependencies...
pip install pywin32 >nul 2>&1

REM Run the Python installer
python installer.py

if %errorlevel% equ 0 (
    echo.
    echo Installation process completed.
) else (
    echo.
    echo Installation encountered an error.
    pause
)

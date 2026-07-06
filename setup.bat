@echo off
:: ============================================================
:: Camera Map – Setup Script
:: Run this once to install all required Python dependencies.
:: ============================================================

echo.
echo  Camera Map – Dependency Installer
echo  ===================================

:: Check that Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  [ERROR] Python was not found on your PATH.
    echo  Please install Python 3.10+ from https://www.python.org/downloads/
    echo  and make sure "Add Python to PATH" is checked during installation.
    pause
    exit /b 1
)

echo.
for /f "tokens=*" %%v in ('python --version') do echo  Found: %%v

:: Bootstrap pip if missing
echo.
echo  Checking pip...
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo  pip not found – bootstrapping...
    python -m ensurepip --upgrade
)

:: Upgrade pip silently
python -m pip install --upgrade pip --quiet

:: Install dependencies from requirements.txt
echo.
echo  Installing dependencies from requirements.txt...
echo.
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo  [ERROR] One or more packages failed to install.
    echo  Try running this script as Administrator or check your internet connection.
    pause
    exit /b 1
)

echo.
echo  ============================================
echo   All dependencies installed successfully!
echo   Run the app with:  python camera_viewer.py
echo  ============================================
echo.
pause

@echo off
REM TINY 2.1 Flash Chat - Windows Launcher
REM This script automatically sets up and starts the application

setlocal enabledelayedexpansion

echo.
echo ================================
echo TINY 2.1 Flash Chat - Setup
echo ================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [!] Python is not installed or not in PATH
    echo Please install Python 3.7+ from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo [+] Found: %PYTHON_VERSION%
echo.

REM Get the directory where the script is located
set DIR=%~dp0
cd /d "%DIR%"

REM Check for virtual environment
if exist "venv\" (
    echo [+] Virtual environment found
    call venv\Scripts\activate.bat
) else (
    echo [*] Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo [+] Virtual environment created
)

echo.
echo Installing dependencies...

REM Install required packages
python -m pip install --upgrade pip setuptools wheel >nul 2>&1
python -m pip install -r requirements.txt >nul 2>&1

echo [+] Dependencies installed from requirements.txt
echo.

REM Check for TINY.BIN
echo Checking for TINY.BIN...
set TINY_BIN=

if exist "%DIR%TINY.BIN" (
    set TINY_BIN=%DIR%TINY.BIN
    echo [+] Found TINY.BIN in project root
) else if exist "%DIR%dos\TINY.BIN" (
    set TINY_BIN=%DIR%dos\TINY.BIN
    echo [+] Found TINY.BIN in dos\ directory
) else if exist "%DIR%Tiny-2.1\dos\TINY.BIN" (
    set TINY_BIN=%DIR%Tiny-2.1\dos\TINY.BIN
    echo [+] Found TINY.BIN in Tiny-2.1\dos\ directory
) else (
    echo.
    echo [!] TINY.BIN not found!
    echo.
    echo Please download TINY.BIN from:
    echo   https://github.com/WollyMonjac/Tiny-2.1
    echo.
    echo And place it in one of these locations:
    echo   * %DIR%TINY.BIN
    echo   * %DIR%dos\TINY.BIN
    echo   * %DIR%Tiny-2.1\dos\TINY.BIN
    echo.
    pause
    exit /b 1
)

echo.
echo ================================
echo Starting TINY 2.1 Flash Chat
echo ================================
echo.

REM Parse command line arguments
set PORT=5014
set HOST=0.0.0.0

:parse_args
if "%~1"=="" goto start_app
if "%~1"=="--port" (
    set PORT=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--host" (
    set HOST=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--bin" (
    set TINY_BIN=%~2
    shift
    shift
    goto parse_args
)

:start_app
echo [+] Model: %TINY_BIN%
echo [+] Server: http://%HOST%:%PORT%
echo [!] Press Ctrl+C to stop
echo.

REM Start the Flask app
python app.py --bin "%TINY_BIN%" --host %HOST% --port %PORT%
pause

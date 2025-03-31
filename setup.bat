@echo off
REM ========================================
REM Setup script for EEG Project on Windows
REM ========================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found. Downloading Python 3.10 installer...
    REM Download the Python 3.10 installer (64-bit version)
    powershell -command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.10.10/python-3.10.10-amd64.exe' -OutFile 'python-3.10.10-amd64.exe'"
    echo Installing Python 3.10 silently...
    REM The /quiet flag performs a silent installation,
    REM InstallAllUsers=1 installs for all users,
    REM PrependPath=1 adds Python to the PATH.
    python-3.10.10-amd64.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
    REM Pause briefly to let installation finish
    timeout /t 10
) else (
    for /f "tokens=2 delims= " %%i in ('python --version') do set version=%%i
    echo Found Python version: %version%
)

echo.
echo Installing required Python packages...
pip install mne numpy matplotlib rich jinja2 antropy nolds pandas scipy

echo.
echo Setup complete! You can now run the project by executing:
echo     python main.py
pause

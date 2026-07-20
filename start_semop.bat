@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"

py -3 -c "import sys" >nul 2>nul
if not errorlevel 1 (
    py -3 -m semop.beginner_web %*
    set "SEMOP_EXIT=%ERRORLEVEL%"
    goto :done
)

python -c "import sys" >nul 2>nul
if not errorlevel 1 (
    python -m semop.beginner_web %*
    set "SEMOP_EXIT=%ERRORLEVEL%"
    goto :done
)

for /f "usebackq delims=" %%P in (`powershell -NoProfile -Command "(Get-Command python -CommandType Application -ErrorAction SilentlyContinue).Source"`) do set "SEMOP_PYTHON=%%P"
if defined SEMOP_PYTHON (
    "%SEMOP_PYTHON%" -m semop.beginner_web %*
    set "SEMOP_EXIT=%ERRORLEVEL%"
    goto :done
)

echo Python 3.11 or newer was not found.
echo Install Python and run this file again.
set "SEMOP_EXIT=1"

:done
if not "%SEMOP_EXIT%"=="0" pause
exit /b %SEMOP_EXIT%

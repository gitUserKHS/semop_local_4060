@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"

where py >nul 2>nul
if not errorlevel 1 (
    py -3 -m semop.beginner_web
    set "SEMOP_EXIT=%ERRORLEVEL%"
    goto :done
)

where python >nul 2>nul
if not errorlevel 1 (
    python -m semop.beginner_web
    set "SEMOP_EXIT=%ERRORLEVEL%"
    goto :done
)

echo Python을 찾지 못했어.
echo Python 3.11 이상을 설치한 뒤 이 파일을 다시 실행해 줘.
set "SEMOP_EXIT=1"

:done
if not "%SEMOP_EXIT%"=="0" pause
exit /b %SEMOP_EXIT%

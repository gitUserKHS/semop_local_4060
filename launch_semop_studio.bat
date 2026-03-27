@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE="
if exist ".venv312\Scripts\python.exe" set "PYTHON_EXE=.venv312\Scripts\python.exe"
if not defined PYTHON_EXE set "PYTHON_EXE=python"

"%PYTHON_EXE%" run_beginner_local_stack.py
if errorlevel 1 (
  echo.
  echo SemOp one-click launcher failed.
  echo If dependencies are missing, run: "%PYTHON_EXE%" run_beginner_local_stack.py doctor
  echo.
  pause
)
endlocal

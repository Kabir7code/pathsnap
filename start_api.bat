@echo off
REM Laptop 1: start the PathSnap API.   Usage: start_api.bat [eval_trips_folder]   (default: data\student_test)
cd /d "%~dp0"
set "PATHSNAP_EVAL_DIR=%~1"
if "%PATHSNAP_EVAL_DIR%"=="" set "PATHSNAP_EVAL_DIR=data\student_test"
python scripts\check_setup.py "%PATHSNAP_EVAL_DIR%"
if errorlevel 1 (
  echo Fix the failed checks first.
  pause
  exit /b 1
)
echo.
echo Starting API on port 8000 (Ctrl+C to stop). Keep this window visible: every request is logged here.
echo.
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
pause

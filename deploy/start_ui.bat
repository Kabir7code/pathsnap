@echo off
REM Laptop 2: serve the UI.   Usage: deploy\start_ui.bat <laptop1_ip>
cd /d "%~dp0..\app\ui"
set "IP=%~1"
if "%IP%"=="" set "IP=localhost"
echo Open this in Chrome:  http://localhost:8080/?api=http://%IP%:8000
echo (Ctrl+C to stop)
start "" "http://localhost:8080/?api=http://%IP%:8000"
python -m http.server 8080

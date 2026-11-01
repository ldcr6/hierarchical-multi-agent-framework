@echo off
chcp 65001 >nul
cd /d "%~dp0.."
rem open the page after the server has had a few seconds to start
start "" /b cmd /c "timeout /t 4 /nobreak >nul && start http://127.0.0.1:8000"
python -m webapp --port 8000
pause

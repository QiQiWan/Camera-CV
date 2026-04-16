@echo off
setlocal
cd /d "%~dp0"
python crack_detection_app.py
if errorlevel 1 pause

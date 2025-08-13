@echo off
REM Change to your project folder
cd /d D:\Logic\KPI

REM Path to your Python executable in your venv
set PYTHON_EXEC=D:\Logic\KPI\venv\Scripts\python.exe

REM Run the update script
"%PYTHON_EXEC%" update_realtime_kpis.py

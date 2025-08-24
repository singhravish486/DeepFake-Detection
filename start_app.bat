@echo off
echo 🚀 Starting Aerial DeepFake Detection Web Application...
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Start the application
echo ✅ Virtual environment activated
echo 🌐 Starting web server...
echo.
python app.py

pause


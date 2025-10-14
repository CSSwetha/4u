@echo off
echo Starting 4u - Dermatological Ingredient Analyzer...
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if activation was successful
if errorlevel 1 (
    echo Error: Could not activate virtual environment
    echo Please run setup first: python -m venv venv
    pause
    exit /b 1
)

echo Virtual environment activated
echo.

REM Run Streamlit app
echo Launching application...
streamlit run app.py

REM Deactivate when done
deactivate

pause
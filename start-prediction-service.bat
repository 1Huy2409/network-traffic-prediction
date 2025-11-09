@echo off
REM Start Prediction Service on Windows

echo ======================================
echo Starting Prediction Service
echo ======================================

cd  E:\adocument\PBL4\Code\PBL4-Network-Traffic-Prediction

echo.
echo Checking dependencies...
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo FastAPI not installed
    echo Installing dependencies...
    pip install -r requirements-prediction-service.txt
)

echo.
echo Starting service on http://localhost:5000
echo    Press Ctrl+C to stop
echo.

py prediction_service.py --port 5000

pause

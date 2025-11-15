@echo off
echo ========================================
echo  Starting ALL Services for Dashboard
echo ========================================
echo.

REM Get current directory
set SCRIPT_DIR=%~dp0
set SAGSINS_DIR=%SCRIPT_DIR%..\SAGSINs-System

echo [1/5] Starting Docker Containers...
cd /d "%SAGSINS_DIR%\docker"
docker-compose up -d
if errorlevel 1 (
    echo ERROR: Failed to start Docker containers
    pause
    exit /b 1
)
echo     Docker containers started
echo.

echo [2/5] Starting Prediction Service...
cd /d "%SCRIPT_DIR%"
start "Prediction Service" cmd /k "python prediction_service.py --port 5000"
timeout /t 3 /nobreak >nul
echo     Prediction service started on port 5000
echo.

echo [3/5] Starting Node.js Backend...
cd /d "%SAGSINS_DIR%\wep-app\backend"
start "Node.js Backend" cmd /k "npm start"
timeout /t 3 /nobreak >nul
echo     Node.js backend started on port 3001
echo.

echo [4/5] Starting React Frontend...
cd /d "%SAGSINS_DIR%\wep-app\frontend"
start "React Frontend" cmd /k "npm run dev"
timeout /t 3 /nobreak >nul
echo     React frontend started on port 5173
echo.

echo [5/5] Opening Dashboard...
timeout /t 2 /nobreak >nul
start http://localhost:5000
echo     Dashboard opened in browser
echo.

echo ========================================
echo  All Services Started Successfully!
echo ========================================
echo.
echo URLs:
echo   Dashboard:  http://localhost:5000
echo   Web App:    http://localhost:5173
echo   Backend:    http://localhost:3001
echo   Docker:     http://localhost:8080
echo.
echo Press any key to view logs (or close to run in background)
pause

REM Follow logs
docker-compose -f "%SAGSINS_DIR%\docker\docker-compose.yml" logs -f

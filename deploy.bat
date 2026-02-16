@echo off
echo ========================================
echo    CoAct.AI - One-Click Deployment
echo ========================================
echo.

:: Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running!
    echo Please start Docker Desktop first.
    pause
    exit /b 1
)

echo [1/3] Stopping existing containers...
docker-compose down --remove-orphans 2>nul

echo [2/3] Building and starting services...
docker-compose up --build -d

echo [3/3] Waiting for services to be ready...
timeout /t 10 /nobreak >nul

echo.
echo ========================================
echo    Deployment Complete!
echo ========================================
echo.
echo Access your application:
echo   Frontend: http://localhost
echo   Public:   http://coact-ai.centralindia.cloudapp.azure.com
echo   API:      http://coact-ai.centralindia.cloudapp.azure.com/api/
echo.
echo Commands:
echo   View logs:    docker-compose logs -f
echo   Stop:         docker-compose down
echo   Restart:      docker-compose restart
echo.
pause

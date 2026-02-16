@echo off
echo Starting CoAct.AI Development Environment...
echo.

:: Start backend in a new window
start "CoAct.AI Backend" cmd /k "cd /d d:\COACTAI\inter-ai-backend && python app.py"

:: Wait a moment for backend to initialize
timeout /t 3 /nobreak >nul

:: Start frontend in a new window
start "CoAct.AI Frontend" cmd /k "cd /d d:\COACTAI\inter-ai-frontend && npm run dev"

echo.
echo Both servers are starting...
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo.
echo Opening browser in 5 seconds...
timeout /t 5 /nobreak >nul

:: Open browser
start http://localhost:3000

echo Done! Close this window when finished.

@echo off
chcp 65001 >nul

set PROJECT=C:\Users\camelo.cruz\Documents\GitHub\TGT
set LOGFILE=%PROJECT%\logs\update_check.log

REM ==== Ensure logs folder exists ====
if not exist "%PROJECT%\logs" mkdir "%PROJECT%\logs"

echo [%date% %time%] Checking for updates... >> "%LOGFILE%"

cd /d %PROJECT%

REM ==== Save current commit hash ====
for /f %%i in ('git rev-parse HEAD') do set BEFORE=%%i

REM ==== Pull latest from remote ====
git pull >> "%LOGFILE%" 2>&1

REM ==== Get new commit hash ====
for /f %%i in ('git rev-parse HEAD') do set AFTER=%%i

REM ==== Exit early if nothing changed ====
if "%BEFORE%"=="%AFTER%" (
    echo [%date% %time%] No changes detected. >> "%LOGFILE%"
    exit /b 0
)

echo [%date% %time%] Changes detected. Restarting server... >> "%LOGFILE%"
git log --oneline %BEFORE%..%AFTER% >> "%LOGFILE%" 2>&1

REM ==== Kill uvicorn (the Python process listening on port 8000) ====
for /f "tokens=5" %%p in ('netstat -ano ^| findstr "LISTENING" ^| findstr ":8000"') do (
    echo [%date% %time%] Killing PID %%p >> "%LOGFILE%"
    taskkill /f /pid %%p >nul 2>&1
)

REM ==== Wait for port to free up ====
timeout /t 4 /nobreak >nul

REM ==== Restart the full server in a new window ====
start "TGT Server" cmd /c "%PROJECT%\run_tgt.bat"

echo [%date% %time%] Server restarted. >> "%LOGFILE%"
exit /b 0

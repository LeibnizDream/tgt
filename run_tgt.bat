@echo off
REM Set UTF-8 code page for proper character encoding
chcp 65001 >nul



REM ==== Start Nginx if not already running ====
tasklist /fi "imagename eq nginx.exe" 2>nul | find /i "nginx.exe" >nul
if errorlevel 1 (
    cd /d C:\nginx
    start "" nginx
) else (
    cd /d C:\nginx
    nginx -s reload
)

REM ==== Start Ollama if not already running ====
tasklist /fi "imagename eq ollama.exe" 2>nul | find /i "ollama.exe" >nul
if errorlevel 1 (
    start "" "C:\Users\camelo.cruz\AppData\Local\Programs\Ollama\ollama.exe" serve
)

REM ==== Wait until Ollama is actually ready on 11434 ====
:wait_ollama
curl -s http://127.0.0.1:11434 >nul 2>&1
if errorlevel 1 (
    timeout /t 2 /nobreak >nul
    goto wait_ollama
)

REM ==== Switch to project root ====
cd /d C:\Users\camelo.cruz\Documents\GitHub\TGT

REM ==== Ensure logs folder exists ====
cd /d C:\Users\camelo.cruz\Documents\GitHub\TGT
if not exist ".\logs" mkdir ".\logs"

REM ==== Sync Python environment ====
set PATH=%PATH%;C:\Users\camelo.cruz\.local\bin
uv sync

REM ==== Build frontend ====
cd /d C:\Users\camelo.cruz\Documents\GitHub\TGT\frontend
if exist "dist" rmdir /s /q dist
call npm run build

REM ==== Set Python UTF-8 environment variables ====
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

REM ==== Launch Uvicorn ====
cd /d C:\Users\camelo.cruz\Documents\GitHub\TGT\backend
uv run uvicorn app:app --host 127.0.0.1 --port 8000 >> "..\logs\uvicorn.log" 2>&1

exit
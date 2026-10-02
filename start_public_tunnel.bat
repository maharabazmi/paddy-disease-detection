@echo off
title Paddy Doctor AI - Cloudflare Public Tunnel
echo ================================================================
echo   Paddy Doctor AI: Public HTTPS Tunnel (Zero-Config Sharing)
echo ================================================================
echo.
echo Make sure your backend server is running in another window!
echo (Double-click start_backend.bat first)
echo.
echo Creating global HTTPS link to your local server...
echo.
"%~dp0\cloudflared.exe" tunnel --url http://127.0.0.1:8000
pause

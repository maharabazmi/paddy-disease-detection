@echo off
title Paddy Disease Detection - FastAPI Backend Server
echo ================================================================
echo   Paddy Doctor AI: Backend Microservice (FastAPI + PyTorch)
echo ================================================================
echo.
echo Starting FastAPI server on port 8000 (all network interfaces)...
echo.
cd /d "%~dp0\paddy_disease_detection"
python server.py
pause

@echo off
REM Statistical Arbitrage System - Windows Launcher

echo Statistical Arbitrage Analysis System
echo =====================================

if "%1"=="" (
    echo Usage: run.bat [command] [options]
    echo.
    echo Commands:
    echo   setup     - Initialize the system and create database tables
    echo   scheduler - Run the full scheduler ^(daily + hourly^)
    echo   daily     - Run daily cointegration analysis once
    echo   hourly    - Run hourly statistics analysis once
    echo   status    - Show system status
    echo.
    echo Examples:
    echo   run.bat setup
    echo   run.bat daily
    echo   run.bat hourly
    echo   run.bat scheduler
    echo   run.bat status
    goto :eof
)

if "%1"=="setup" (
    echo Setting up Statistical Arbitrage System...
    python setup.py
    goto :eof
)

if "%1"=="scheduler" (
    echo Starting Statistical Arbitrage Scheduler...
    echo Press Ctrl+C to stop
    python main.py scheduler %2 %3 %4 %5
    goto :eof
)

if "%1"=="daily" (
    echo Running daily cointegration analysis...
    python main.py daily %2 %3 %4 %5
    goto :eof
)

if "%1"=="hourly" (
    echo Running hourly statistics analysis...
    python main.py hourly %2 %3 %4 %5
    goto :eof
)

if "%1"=="status" (
    echo Checking system status...
    python main.py status %2 %3 %4 %5
    goto :eof
)

echo Unknown command: %1
echo Run 'run.bat' without arguments to see available commands.

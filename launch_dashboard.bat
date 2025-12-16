@echo off
echo Starting Interactive Dashboard...
echo.

REM Check if Orders.json exists in the current directory
if exist Orders.json (
    python launch_dashboard.py Orders.json
) else (
    REM Check if Orders.json exists in the Csv & Json conversions directory
    if exist "Csv & Json conversions\Orders.json" (
        python launch_dashboard.py "Csv & Json conversions\Orders.json"
    ) else (
        echo Error: Orders.json not found.
        echo Please convert an Excel file first or specify a JSON file path.
        pause
        exit /b 1
    )
)

echo.
echo Dashboard closed.
pause